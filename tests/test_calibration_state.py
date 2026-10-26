import torch
import pytest
from safetensors.torch import load_file, save_file

from ming3_nvfp4.calibration_state import save_quantizer_buffers, restore_quantizer_file
from test_mse_calibration import static_fixture


def test_standard_file_parity_aliases_and_restore(tmp_path):
    model = static_fixture()
    path = tmp_path / 'quantizers.safetensors'
    save_quantizer_buffers(model, path, chunk_bytes=17)
    expected = {name: value.clone() for name, value in
                model.named_buffers(remove_duplicate=False) if 'quantizer' in name}
    actual = load_file(path)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], value) for name, value in expected.items())
    stock = tmp_path / 'stock.safetensors'
    save_file(expected, stock)
    restore_quantizer_file(model, stock, 'cpu')
    gate = model.experts[0].gate_proj.weight_quantizer
    up = model.experts[0].up_proj.weight_quantizer
    assert gate.global_amax is up.global_amax
    assert all(torch.equal(value, expected[name]) for name, value in
               model.named_buffers(remove_duplicate=False) if 'quantizer' in name)
    # Our file remains readable after closing the lazy safetensors reader.
    restore_quantizer_file(model, path, 'cpu')
    assert gate.global_amax is up.global_amax
    assert torch.equal(gate.amax, expected['experts.0.gate_proj.weight_quantizer._amax'])


def test_save_supports_scalar_empty_and_bfloat16(tmp_path):
    model = torch.nn.Module()
    model.weight_quantizer = torch.nn.Module()
    for name, value in {'scalar': torch.tensor(2.), 'empty': torch.empty(0),
                        'bf16': torch.arange(8).to(torch.bfloat16)}.items():
        model.weight_quantizer.register_buffer(name, value)
    path = tmp_path / 'states.safetensors'
    save_quantizer_buffers(model, path, chunk_bytes=3)
    loaded = load_file(path)
    assert all(torch.equal(value, loaded[name]) for name, value in model.named_buffers())


def test_mse_search_state_release_preserves_scale_sync_order(monkeypatch):
    import modelopt.torch.quantization.model_calib as calibration
    from modelopt.torch.quantization.calib.mse import NVFP4MSECalibrator
    import ming3_nvfp4.adapter as adapter

    model = static_fixture()
    quantizer = model.experts[0].gate_proj.weight_quantizer
    quantizer._calibrator = NVFP4MSECalibrator(
        amax=quantizer.amax.clone(), global_amax=quantizer.global_amax.clone())
    before = quantizer.amax.clone()
    events = []

    def sweep(current, *args, **kwargs):
        assert quantizer._calibrator._initial_amax is not None
        events.append('sweep')

    def sync(current):
        assert quantizer._calibrator._initial_amax is None
        events.append('sync')

    monkeypatch.setattr(calibration, '_mse_calibrate_weights', sweep)
    monkeypatch.setattr(adapter, 'synchronize_export_scales', sync)
    with adapter.calibration_memory_context():
        calibration._mse_calibrate_weights(model)
    assert events == ['sweep']
    adapter.synchronize_export_scales(model)
    assert events == ['sweep', 'sync']
    assert quantizer._calibrator._initial_amax is None
    assert torch.equal(quantizer.amax, before)
    assert calibration._mse_calibrate_weights is sweep


def test_invalid_source_does_not_partially_mutate_model(tmp_path):
    model = static_fixture()
    previous = model.experts[0].gate_proj.weight_quantizer.amax.clone()
    states = {name: value.clone() for name, value in
              model.named_buffers(remove_duplicate=False) if 'quantizer' in name}
    states['experts.0.up_proj.weight_quantizer._global_amax'].add_(1)
    path = tmp_path / 'invalid.safetensors'
    save_file(states, path)
    with pytest.raises(ValueError, match='Conflicting shared'):
        restore_quantizer_file(model, path, 'cpu')
    assert torch.equal(model.experts[0].gate_proj.weight_quantizer.amax, previous)


def test_writer_bounds_each_cpu_transfer(monkeypatch, tmp_path):
    model = torch.nn.Module()
    model.weight_quantizer = torch.nn.Module()
    model.weight_quantizer.register_buffer('_amax', torch.arange(1000, dtype=torch.float32))
    original = torch.Tensor.to
    transfers = []

    def observed(tensor, *args, **kwargs):
        if args and args[0] == 'cpu':
            transfers.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', observed)
    save_quantizer_buffers(model, tmp_path / 'chunked.safetensors', chunk_bytes=64)
    assert sum(transfers) == 4000
    assert max(transfers) <= 64
    assert len(transfers) > 1


def test_mse_context_preserves_all_buffers_and_mock_qdq(monkeypatch):
    import modelopt.torch.quantization.model_calib as calibration
    from modelopt.torch.quantization.calib.mse import NVFP4MSECalibrator
    from ming3_nvfp4.adapter import calibration_memory_context, synchronize_export_scales

    def sweep(model):
        for module in model.modules():
            quantizer = getattr(module, 'weight_quantizer', None)
            if quantizer is not None and quantizer.is_enabled:
                quantizer._calibrator = NVFP4MSECalibrator(
                    amax=quantizer.amax.clone(), global_amax=quantizer.global_amax)
                quantizer.amax = quantizer.amax * 0.875

    monkeypatch.setattr(calibration, '_mse_calibrate_weights', sweep)
    torch.manual_seed(42)
    original = static_fixture()
    torch.manual_seed(42)
    bounded = static_fixture()
    sweep(original)
    synchronize_export_scales(original)
    with calibration_memory_context():
        calibration._mse_calibrate_weights(bounded)
    synchronize_export_scales(bounded)
    expected = dict(original.named_buffers(remove_duplicate=False))
    assert all(torch.equal(value, expected[name])
               for name, value in bounded.named_buffers(remove_duplicate=False))
    inputs = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    for projection in ('gate_proj', 'up_proj'):
        left = getattr(original.experts[0], projection)
        right = getattr(bounded.experts[0], projection)

        def mock_qdq(linear):
            scale = linear.weight_quantizer.amax / 6
            weight = (linear.weight / scale).round().clamp(-6, 6) * scale
            return inputs @ weight.T

        assert torch.equal(mock_qdq(left), mock_qdq(right))

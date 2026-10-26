import copy
from types import SimpleNamespace

from accelerate import disk_offload
from accelerate.hooks import attach_execution_device_hook
import pytest
import torch

from ming3_nvfp4.adapter import calibration_memory_context


def fixture(directory, calibrated=False):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.model_calib import max_calibrate
    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    torch.manual_seed(42)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for _ in range(3):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        expert = torch.nn.Module()
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(expert, projection, torch.nn.Linear(16, 16, bias=False))
        layer.mlp.experts = torch.nn.ModuleList([expert])
        model.layers.append(layer)
    model.to(dtype=torch.bfloat16)
    config = copy.deepcopy(mtq.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG)
    config['algorithm'] = None
    mtq.quantize(model, config)
    if calibrated:
        max_calibrate(model)
        for module in model.modules():
            quantizer = getattr(module, 'weight_quantizer', None)
            if quantizer is not None:
                quantizer.amax = quantizer.amax * 0.875
                # Model input-global calibration values without CPU NVFP4 kernels;
                # each activation observer has independent storage as in stock.
                module.input_quantizer.amax = torch.tensor(3.0)
    for layer in model.layers:
        layer._layerwise_calib = SimpleNamespace(
            output_meta=LayerActivationCollector._extract_output_meta(torch.zeros(1, 16)))
    disk_offload(model, directory, execution_device=torch.device('cpu'), offload_buffers=False)
    attach_execution_device_hook(model, torch.device('cpu'))
    return model


def prefix_values(model):
    return {f'{index}.{name}': value.clone() for index, layer in enumerate(model.layers[:2])
            for name, value in layer.named_buffers(remove_duplicate=False) if 'quantizer' in name}


def graph(model):
    groups = {}
    for index, layer in enumerate(model.layers[:2]):
        for name, value in layer.named_buffers(remove_duplicate=False):
            groups.setdefault(id(value), []).append(f'{index}.{name}')
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def promoted_bytes(model):
    from modelopt.torch.quantization.plugins.accelerate import _get_offload_hook

    loaders = {}
    for module in model.modules():
        hook = _get_offload_hook(getattr(module, '_hf_hook', None))
        if hook is not None:
            loader = getattr(hook.weights_map, 'dataset', hook.weights_map)
            loaders[id(loader)] = loader
    return sum(value.numel() * value.element_size() for loader in loaders.values()
               for name, value in (getattr(loader, 'state_dict', {}) or {}).items()
               if name in (getattr(loader, 'index', {}) or {}) and name.endswith('.weight'))


def checkpoint(directory, producer):
    from modelopt.torch.quantization.utils.layerwise_calib import _CheckpointState

    state = _CheckpointState(str(directory), 3, save_every=2, calib_mutates_weights=False)
    inputs = [((torch.arange(16).reshape(1, 16).float(),), {})]
    for index in range(2):
        state.save(index, producer, producer.layers, inputs)
    return _CheckpointState.from_folder(str(directory), 3, save_every=2,
                                         calib_mutates_weights=False)


def test_resumed_disk_prefix_preserves_values_aliases_and_cpu_backing(tmp_path):
    producer = fixture(tmp_path / 'producer', calibrated=True)
    state = checkpoint(tmp_path / 'layers', producer)
    expected, expected_graph = prefix_values(producer), graph(producer)
    resumed = fixture(tmp_path / 'resumed')
    assert state.start_layer == 2 and promoted_bytes(resumed) == 0
    assert not hasattr(resumed.layers[2].mlp.experts[0], '_shared_quant_states')
    with calibration_memory_context():
        state.full_restore(resumed.layers, resumed)
    actual = prefix_values(resumed)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(value, expected[name]) for name, value in actual.items())
    assert graph(resumed) == expected_graph
    assert promoted_bytes(resumed) == 0
    # No alias repair touches the uncommitted suffix.
    assert not hasattr(resumed.layers[2].mlp.experts[0], '_shared_quant_states')


@pytest.mark.parametrize('start,mutates', [(0, False), (0, True), (2, True)])
def test_start_zero_and_mutating_restore_keep_stock_path(monkeypatch, start, mutates):
    from modelopt.torch.quantization.utils import core_utils
    from modelopt.torch.quantization.utils.layerwise_calib import _CheckpointState
    from modelopt.torch.quantization.utils.shared_input import SharedWeightGlobalAmaxState

    access = core_utils.enable_weight_access_and_writeback
    calls = []

    def original(checkpoint, layers, model):
        assert core_utils.enable_weight_access_and_writeback is access
        calls.append((checkpoint.start_layer, checkpoint.calib_mutates_weights))
        return 'stock-result'

    def unexpected(*args, **kwargs):
        raise AssertionError('Alias repair touched a stock path')

    monkeypatch.setattr(_CheckpointState, 'full_restore', original)
    monkeypatch.setattr(SharedWeightGlobalAmaxState, 'restore', unexpected)
    state = SimpleNamespace(start_layer=start, calib_mutates_weights=mutates)
    with calibration_memory_context():
        assert _CheckpointState.full_restore(state, [], None) == 'stock-result'
    assert calls == [(start, mutates)]
    assert _CheckpointState.full_restore is original


def test_conflicting_restored_globals_fail_without_cpu_weight_promotion(tmp_path):
    producer = fixture(tmp_path / 'producer', calibrated=True)
    state = checkpoint(tmp_path / 'layers', producer)
    path = tmp_path / 'layers/layer_0000/quantizer_buffers.pt'
    buffers = torch.load(path, weights_only=False)
    name = 'mlp.experts.0.up_proj.weight_quantizer'
    buffers[name]['_global_amax'] = buffers[name]['_global_amax'].clone() + 1
    torch.save(buffers, path)
    resumed = fixture(tmp_path / 'resumed')
    with calibration_memory_context(), pytest.raises(ValueError, match='Conflicting restored'):
        state.full_restore(resumed.layers, resumed)
    assert promoted_bytes(resumed) == 0

"""CPU regressions for static NVFP4 scale semantics; no CUDA kernels required."""
import copy

import pytest
import torch

from ming3_nvfp4.adapter import restore_quantizer_buffers, synchronize_export_scales


def static_fixture():
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.model_calib import max_calibrate

    model = torch.nn.Module()
    model.experts = torch.nn.ModuleList([torch.nn.Module()])
    expert = model.experts[0]
    expert.gate_proj = torch.nn.Linear(16, 16, bias=False)
    expert.up_proj = torch.nn.Linear(16, 16, bias=False)
    config = copy.deepcopy(mtq.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG)
    config['algorithm'] = None
    mtq.quantize(model, config)
    max_calibrate(model)
    return model


def test_static_gate_up_sync_preserves_independent_block_thresholds():
    model = static_fixture()
    gate, up = (getattr(model.experts[0], p).weight_quantizer
                for p in ('gate_proj', 'up_proj'))
    gate.amax = torch.full_like(gate.amax, 0.25)
    up.amax = torch.full_like(up.amax, 0.75)
    synchronize_export_scales(model)
    assert torch.all(gate.amax == 0.25)
    assert torch.all(up.amax == 0.75)
    assert torch.equal(gate.global_amax, up.global_amax)


def test_restore_shared_scale_aliases_and_reject_conflicts():
    model = static_fixture()
    original_weight = model.experts[0].gate_proj.weight
    buffers = {name: value.clone() for name, value in
               model.named_buffers(remove_duplicate=False) if 'quantizer' in name}
    globals_ = [name for name in buffers if name.endswith('_global_amax')]
    assert len(globals_) == 2
    for name in globals_:
        buffers[name].fill_(2.5)
    restore_quantizer_buffers(model, buffers, 'cpu')
    gate = model.experts[0].gate_proj.weight_quantizer
    up = model.experts[0].up_proj.weight_quantizer
    assert gate.global_amax is up.global_amax
    assert gate.global_amax.item() == 2.5
    assert model.experts[0].gate_proj.weight is original_weight
    assert all(value is gate.global_amax for name, value in
               model.named_buffers(remove_duplicate=False) if name.endswith('_global_amax'))
    buffers[globals_[1]].fill_(3.0)
    with pytest.raises(ValueError, match='Conflicting shared'):
        restore_quantizer_buffers(model, buffers, 'cpu')


def test_export_guard_restores_modelopt_functions():
    from ming3_nvfp4.adapter import export_scale_sync_context
    import modelopt.torch.export.unified_export_hf as unified

    original = unified.sync_moe_gate_up_amax
    with export_scale_sync_context():
        assert unified.sync_moe_gate_up_amax is synchronize_export_scales
    assert unified.sync_moe_gate_up_amax is original


def test_modelopt_checkpoint_roundtrip_preserves_static_scales():
    import modelopt.torch.opt as mto

    model = static_fixture()
    state = mto.modelopt_state(model)
    buffers = {name: value.clone() for name, value in
               model.named_buffers(remove_duplicate=False) if 'quantizer' in name}
    restored = torch.nn.Module()
    restored.experts = torch.nn.ModuleList([torch.nn.Module()])
    for projection in ('gate_proj', 'up_proj'):
        setattr(restored.experts[0], projection, torch.nn.Linear(16, 16, bias=False))
    mto.restore_from_modelopt_state(restored, state)
    restore_quantizer_buffers(restored, buffers, 'cpu')
    for name, value in restored.named_buffers(remove_duplicate=False):
        if 'quantizer' in name:
            assert torch.equal(value, buffers[name])
    gate = restored.experts[0].gate_proj.weight_quantizer
    up = restored.experts[0].up_proj.weight_quantizer
    assert gate.global_amax is up.global_amax

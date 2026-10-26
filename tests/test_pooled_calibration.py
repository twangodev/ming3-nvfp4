"""Focused CPU policy tests; no Ming weights or CUDA required."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ming3_nvfp4.pooled_calibration import (POLICY, observe_default_pooling,
                                          reuse_pooled_coverage, valid_scale,
                                          validate_selected_scales)


def identity():
    return {'provenance': {'recipe': {'mode': 'w4a4', 'algorithm': 'mse', 'retain_layers': [],
                                     'seed': 42, 'calibration_checkpoint_every': 8}, 'source': 'pinned'},
            'manifest_sha256': 'same', 'layerwise': True, 'smoke': False,
            'batch_limits': {'size': 8, 'tokens': 4096}}


def test_policy_reuse_preserves_deficits_and_exact_other_identity(tmp_path):
    producer = identity()
    (tmp_path / 'run.json').write_text(json.dumps(producer))
    (tmp_path / 'coverage.json').write_text(json.dumps({'counts': {'a': 0, 'b': 127, 'c': 128}}))
    consumer = copy.deepcopy(producer)
    consumer['provenance']['recipe']['expert_coverage_policy'] = POLICY
    result = reuse_pooled_coverage(tmp_path, consumer, ['a', 'b', 'c'], 128)
    assert result['counts'] == {'a': 0, 'b': 127, 'c': 128}
    assert result['under_observed'] == ['a', 'b']
    assert consumer['provenance']['recipe']['expert_coverage_policy'] == POLICY
    for change in ['manifest_sha256', 'batch_limits']:
        bad = copy.deepcopy(consumer)
        bad[change] = 'changed'
        with pytest.raises(ValueError, match='differs'):
            reuse_pooled_coverage(tmp_path, bad, ['a'], 128)
    consumer['provenance']['recipe']['calibration_checkpoint_every'] = 1
    with pytest.raises(ValueError, match='differs'):
        reuse_pooled_coverage(tmp_path, consumer, ['a'], 128)


@pytest.mark.parametrize('values,positive,expected', [([0., 1.], False, True),
    ([0., 1.], True, False), ([-1.], False, False), ([float('nan')], False, False),
    ([float('inf')], True, False), ([0.5], True, True)])
def test_bounded_scale_validation(values, positive, expected):
    assert valid_scale(torch.tensor(values), positive=positive, chunk_elements=1) is expected


class Quantizer(nn.Module):
    def __init__(self, value, measured=True):
        super().__init__()
        self.is_enabled = True
        self.register_buffer('_amax', None if value is None else torch.tensor(value))
        self._calibrator = SimpleNamespace(_calib_amax=self._amax if measured else None)

    @property
    def amax(self):
        return self._amax


def experts(values):
    result = nn.ModuleList()
    for value in values:
        expert = nn.Module()
        expert.add_module('input_quantizer', Quantizer(value))
        result.append(expert)
    return result


def test_observation_calls_stock_once_and_restores_hook(monkeypatch):
    from modelopt.torch.quantization.plugins import huggingface as hf
    calls = []
    def stock(items, *args, **kwargs):
        calls.append((args, kwargs))
        maximum = max(q.input_quantizer.amax.item() for q in items if q.input_quantizer.amax is not None)
        for item in items:
            item.input_quantizer._amax = torch.tensor(maximum)
        return 'original-result'
    monkeypatch.setattr(hf, 'sync_moe_expert_amax', stock)
    items, events = experts([2., None, 3.]), []
    with observe_default_pooling(events):
        assert hf.sync_moe_expert_amax(items, sync_weight_amax=False) == 'original-result'
    assert hf.sync_moe_expert_amax is stock
    assert calls == [((), {'sync_weight_amax': False})]
    assert len(events) == 1 and events[0]['positive_donors'] == 2
    assert all(item.input_quantizer.amax.item() == 3 for item in items)
    with pytest.raises(ValueError, match='donors'):
        with observe_default_pooling([]):
            hf.sync_moe_expert_amax(experts([None, None]))
    assert len(calls) == 1 and hf.sync_moe_expert_amax is stock


def test_synthetic_defaults_are_not_measured_donors(monkeypatch):
    from modelopt.torch.quantization.plugins import huggingface as hf
    items = experts([0.5])
    items[0].input_quantizer._calibrator._calib_amax = None
    monkeypatch.setattr(hf, 'sync_moe_expert_amax', lambda *_: pytest.fail('must reject before stock'))
    with pytest.raises(ValueError, match='statistics'):
        with observe_default_pooling([]):
            hf.sync_moe_expert_amax(items)


def test_static_scales_allow_zero_blocks_but_require_positive_global_and_pool(monkeypatch):
    from ming3_nvfp4 import adapter
    selected = {}
    for expert in range(2):
        layer = nn.Linear(2, 2)
        layer.weight_quantizer = Quantizer([0., 2.])
        layer.weight_quantizer.register_buffer('_global_amax', torch.tensor(3.))
        layer.input_quantizer = Quantizer(4.)
        selected[f'model.layers.2.mlp.experts.{expert}.gate_proj'] = layer
    monkeypatch.setattr(adapter, 'selected_linears', lambda *_: selected)
    assert validate_selected_scales(None, {})['positive_pooled_input_groups'] == 1
    selected[next(iter(selected))].weight_quantizer._global_amax.zero_()
    with pytest.raises(ValueError, match='weight'):
        validate_selected_scales(None, {})


def test_checkpoint_validation_precedes_exact_original_save(monkeypatch):
    from ming3_nvfp4.pooled_calibration import validate_layer_checkpoints
    from modelopt.torch.quantization.utils.layerwise_calib import _CheckpointState
    linear = nn.Linear(2, 2)
    linear.weight_quantizer = Quantizer([0., 1.])
    linear.weight_quantizer.register_buffer('_global_amax', torch.tensor(2.))
    linear.input_quantizer = Quantizer(3.)
    layer = SimpleNamespace(named_modules=lambda: [('mlp.experts.0.gate_proj', linear)])
    layers = {2: layer}
    calls = []
    def stock(*args, **kwargs):
        calls.append((args, kwargs))
        return 'saved'
    monkeypatch.setattr(_CheckpointState, 'save', stock)
    records = []
    with validate_layer_checkpoints({}, records):
        assert _CheckpointState.save('checkpoint', 2, 'model', layers, 'inputs') == 'saved'
        linear.weight_quantizer._global_amax.zero_()
        with pytest.raises(ValueError, match='weight'):
            _CheckpointState.save('checkpoint', 2, 'model', layers, 'inputs')
    assert calls == [(('checkpoint', 2, 'model', layers, 'inputs'), {})]
    assert records[0]['layer'] == 2
    assert _CheckpointState.save is stock

"""CPU checks for routing-only runs and their immutable execution identity."""
import json
from types import SimpleNamespace

import pytest
import torch

from ming3_nvfp4 import pipeline


@pytest.fixture
def routing(monkeypatch, tmp_path):
    recipe = dict(variant='tiny', seed=42, min_expert_tokens=128,
                  disk_floor_gib=1, scratch_memory_gib=1, retain_layers=[])
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.json').write_text(json.dumps({'text_config': {
        'first_k_dense_replace': 0, 'num_hidden_layers': 1, 'num_experts': 1}}))
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2)))
    monkeypatch.setattr(pipeline, 'load_manifest', lambda *args: ([{}], 'sample-sha'))
    monkeypatch.setattr(pipeline, '_seed', lambda *args: None)
    monkeypatch.setattr(pipeline, 'disk_budget', lambda *args: None)
    monkeypatch.setattr(pipeline, 'load_model', lambda *args: (model, None))

    class Batches:
        metadata = {'batches': 1}
        lengths = [2]

        def __init__(self, *args):
            pass

    class Observer:
        def __init__(self, *args):
            pass

        def close(self):
            calls.append('close')

        def report(self, minimum):
            return dict(counts={'model.layers.0.mlp.experts.0': 10}, minimum_tokens=minimum,
                        under_observed=['model.layers.0.mlp.experts.0'])

    monkeypatch.setattr(pipeline, 'CalibrationBatches', Batches)
    monkeypatch.setattr(pipeline, 'ExpertCoverage', Observer)
    monkeypatch.setattr(pipeline, '_measure_routes', lambda *args: calls.append('bf16'))
    return recipe, tmp_path / 'source', tmp_path / 'manifest', tmp_path / 'output', calls


def test_routing_only_persists_measured_counts_and_resumes(routing):
    recipe, source, manifest, output, calls = routing
    report = pipeline.coverage(recipe, source, manifest, output, smoke=True)
    assert report['under_observed'] == ['model.layers.0.mlp.experts.0']
    assert calls == ['bf16', 'close']
    identity = json.loads((output / 'run.json').read_text())
    assert identity['smoke'] is True
    assert identity['batch_limits'] == {'size': 1, 'tokens': 4096}
    resumed = pipeline.coverage(recipe, source, manifest, output, smoke=True)
    assert resumed['counts'] == report['counts']
    assert calls == ['bf16', 'close']
    with pytest.raises(ValueError, match='Resume configuration differs'):
        pipeline.coverage({**recipe, 'seed': 17}, source, manifest, output, smoke=True)


def test_observer_removed_on_failure(routing, monkeypatch):
    recipe, source, manifest, output, calls = routing

    def fail(*args):
        raise RuntimeError('invalid routes')

    monkeypatch.setattr(pipeline, '_measure_routes', fail)
    with pytest.raises(RuntimeError, match='invalid routes'):
        pipeline.coverage(recipe, source, manifest, output, smoke=True)
    assert calls == ['close']
    assert not (output / 'coverage.json').exists()


def test_coverage_requires_full_production_manifest_and_flash_layerwise(routing):
    recipe, source, manifest, output, _ = routing
    with pytest.raises(ValueError, match='at least 1024'):
        pipeline.coverage(recipe, source, manifest, output)
    with pytest.raises(ValueError, match='requires --layerwise'):
        pipeline.coverage({**recipe, 'variant': 'flash'}, source, manifest, output, smoke=True)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_nonfinite_outputs_cannot_pass(invalid):
    with pytest.raises(RuntimeError, match='Non-finite'):
        pipeline._require_finite_output({'hidden': (torch.tensor([invalid]),)})
    pipeline._require_finite_output({'logits': torch.ones(2), 'cache': None})


@pytest.mark.parametrize('counts', [{}, {'model.layers.0.mlp.experts.0': 10, 'unexpected': 1}])
def test_cached_coverage_rejects_missing_or_extra_expert_scope(routing, counts):
    recipe, source, manifest, output, calls = routing
    pipeline.coverage(recipe, source, manifest, output, smoke=True)
    (output / 'coverage.json').write_text(json.dumps({'counts': counts}))
    with pytest.raises(ValueError, match='missing selected experts|expert scope differs'):
        pipeline.coverage(recipe, source, manifest, output, smoke=True)
    assert calls == ['bf16', 'close']


def test_expected_scope_excludes_dense_retained_and_mtp_layers(routing):
    recipe, source, *_ = routing
    (source / 'config.json').write_text(json.dumps({'text_config': {
        'first_k_dense_replace': 1, 'num_hidden_layers': 3, 'num_experts': 2}}))
    assert pipeline._expected_coverage_experts(source, {**recipe, 'retain_layers': [2]}) == {
        'model.layers.1.mlp.experts.0', 'model.layers.1.mlp.experts.1'}

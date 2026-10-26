"""CPU contract checks for the portable Flash proof bundle."""
import copy
import json
from pathlib import Path

import pytest
import tomllib

from ming3_nvfp4.common import digest, file_hash, provenance, write_json
from ming3_nvfp4.flash_validation import (
    _comparable_config, _metadata_valid, _select_rows, _tensor_signature,
    _validation_recipe,
    validation_bundle,
)


RECIPE = tomllib.loads((Path(__file__).parents[1] / 'recipes/flash.toml').read_text())


def test_flash_recipe_is_explicit_pooled_mse_and_has_no_quality_report_dependency():
    assert RECIPE['mode'] == 'w4a4'
    assert RECIPE['algorithm'] == 'mse'
    assert RECIPE['expert_coverage_policy'] == 'modelopt_pooled_inputs_v1'
    assert RECIPE['offload_source_shards_only'] is True
    assert RECIPE['calibration_checkpoint_every'] == 8
    assert 'quality' not in RECIPE


def test_selection_is_bounded_deterministic_and_covers_audio_text():
    rows = [{'id': f'row-{i}', 'task': 'text' if i % 3 else 'asr'} for i in range(20)]
    selected = _select_rows(rows)
    assert len(selected) == 8
    assert selected[0]['id'] == 'row-0'
    assert selected[1]['id'] == 'row-1'
    assert {'asr', 'text'} <= {row['task'] for row in selected}
    assert _select_rows(rows) == selected
    with pytest.raises(ValueError, match='audio and text'):
        _select_rows([{'id': str(i), 'task': 'text'} for i in range(4)])


def test_validation_recipe_uses_current_tiny_layout(tmp_path):
    source = tmp_path / 'tiny'
    source.mkdir()
    (source / 'config.json').write_text(json.dumps({'_name_or_path': 'Tiny fixture',
        'text_config': {'first_k_dense_replace': 1, 'num_hidden_layers': 24}}))
    result, scope = _validation_recipe(RECIPE, source)
    assert scope == []
    assert result['variant'] == 'tiny'
    assert result['gpu_memory_gib'] == 4
    assert result['offload_source_shards_only'] is True
    assert result['retain_layers'] == list(range(1, 24))
    assert RECIPE['retain_layers'] == []
    with pytest.raises(ValueError, match='Flash W4A4 MSE'):
        _validation_recipe({**RECIPE, 'algorithm': 'max'}, source)


def test_only_observed_disabled_dtype_initialization_is_accepted():
    before = {'q': {'_disabled': True, '_input_dtype': None}}
    after = {'q': {'_disabled': True, '_input_dtype': 'bf16'}}
    assert _metadata_valid(before, after, {'q': ['bf16']})
    assert not _metadata_valid(before, after, {'q': ['float32']})
    assert not _metadata_valid(before, after, {})
    assert not _metadata_valid(before, {'q': {**after['q'], 'extra': None}}, {'q': ['bf16']})
    assert not _metadata_valid(before, {'q': {'_disabled': False, '_input_dtype': 'bf16'}},
                               {'q': ['bf16']})


def test_scalar_quantizer_scale_has_stable_byte_signature():
    import torch
    value = torch.tensor(1.0)
    assert _tensor_signature(value) == _tensor_signature(value.reshape(1)) | {'shape': []}


def test_calibration_configs_differ_only_in_checkpoint_path():
    left = {'quant_cfg': [{'quantizer_name': '*'}], 'algorithm': {'method': 'mse',
            'layerwise': {'checkpoint_dir': '/a', 'save_every': 1}}}
    right = copy.deepcopy(left)
    right['algorithm']['layerwise']['checkpoint_dir'] = '/b'
    assert _comparable_config(left) == _comparable_config(right)
    right['algorithm']['layerwise']['save_every'] = 2
    assert _comparable_config(left) != _comparable_config(right)


def test_bundle_rejects_nonfinite_router_and_changed_proof(tmp_path):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('{}\n')
    output = tmp_path / 'proof'
    for arm in ('forced', 'resident'):
        write_json(output / arm / 'result.json', {'forced': arm == 'forced',
            'stock_pool_events': [{'stock_pool_exact': True}],
            'projected': {'a': 'same'}, 'first_inputs': {'a': 'same'},
            'bf16': ['same'], 'quantizer_inventory': ['q'],
            'quantizer_buffers': {'q': 'same'}, 'qdq': ['same'],
            'export': {'weight': 'same'},
            'finite_counts': [{'elements': 4, 'finite': 4}]})
    write_json(output / 'router.json', {'experts_executed': 0, 'zero_experts': [1, 2],
        'arms': {'forced': {'counts': [2, 0, 0], 'batch_counts': [[2, 0, 0]]},
                 'resident': {'counts': [2, 0, 0], 'batch_counts': [[2, 0, 0]]}},
        'finite_counts': [{'elements': 4, 'finite': 4}]})
    import ming3_nvfp4.flash_validation as module
    proof_files = ('forced/result.json', 'resident/result.json', 'router.json')
    write_json(output / 'result.json', {'schema': 1, 'status': 'passed',
        'recipe_sha256': digest(RECIPE), 'manifest_sha256': file_hash(manifest),
        'validator_sha256': file_hash(Path(module.__file__)),
        'input_sha256': {str(manifest): file_hash(manifest)},
        'source': str(tmp_path), 'effective_recipe': RECIPE,
        'validation_provenance': provenance(RECIPE, tmp_path),
        'checks': dict.fromkeys(('stock_pool_exact', 'frontend_exact',
            'quantizer_state_exact', 'qdq_exact', 'export_exact',
            'unseen_router_exact_finite'), True),
        'proof_sha256': {name: file_hash(output / name) for name in proof_files}})
    assert validation_bundle(RECIPE, manifest, output)['bundle'] == str(output.resolve())
    router = json.loads((output / 'router.json').read_text())
    router['finite_counts'][0]['finite'] = 3
    write_json(output / 'router.json', router)
    with pytest.raises(ValueError, match='proof changed'):
        validation_bundle(RECIPE, manifest, output)
    result = json.loads((output / 'result.json').read_text())
    result['proof_sha256']['router.json'] = file_hash(output / 'router.json')
    write_json(output / 'result.json', result)
    with pytest.raises(ValueError, match='nonfinite'):
        validation_bundle(RECIPE, manifest, output)


def test_validation_source_is_pinned_by_content_including_weight_shards(tmp_path, monkeypatch):
    import hashlib
    import ming3_nvfp4.flash_validation as module
    source = tmp_path / 'renamed-local-copy'
    source.mkdir()
    config = b'{}'
    weights = b'original weight bytes'
    (source / 'config.json').write_bytes(config)
    (source / 'model.safetensors').write_bytes(weights)
    pin = tmp_path / 'pin.json'
    write_json(pin, {'files': {
        'config.json': {'size_bytes': len(config),
                        'git_blob_sha1': hashlib.sha1(b'blob 2\0' + config).hexdigest()},
        'model.safetensors': {'size_bytes': len(weights),
                              'sha256': hashlib.sha256(weights).hexdigest()},
    }})
    monkeypatch.setattr(module, 'SOURCE_PIN', pin)
    assert len(module._verify_validation_source(source)) == 2
    (source / 'model.safetensors').write_bytes(b'changed! weight bytes')
    with pytest.raises(ValueError, match='content differs'):
        module._verify_validation_source(source)
    (source / 'model.safetensors').write_bytes(weights)
    (source / 'config.json').write_bytes(b'[]')
    with pytest.raises(ValueError, match='content differs'):
        module._verify_validation_source(source)

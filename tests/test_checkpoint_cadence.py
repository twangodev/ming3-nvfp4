import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ming3_nvfp4.adapter import quant_config
from ming3_nvfp4.common import checkpoint_save_every, provenance, read_recipe


def model_fixture():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    expert = torch.nn.Module()
    for projection in ('gate_proj', 'up_proj', 'down_proj'):
        setattr(expert, projection, torch.nn.Linear(16, 16, bias=False))
    layer.mlp.experts = torch.nn.ModuleList([expert])
    model.model.layers = torch.nn.ModuleList([layer])
    model.config = SimpleNamespace(text_config=SimpleNamespace(
        first_k_dense_replace=0, num_hidden_layers=1))
    return model


def test_omitted_cadence_preserves_recipe_and_provenance_identity():
    recipe = read_recipe(Path('recipes/tiny.toml'))
    recipe.pop('calibration_checkpoint_every', None)
    before = copy.deepcopy(recipe)
    assert checkpoint_save_every(recipe) == 1
    result = provenance(recipe, Path('/pinned/source'))
    assert recipe == before and result['recipe'] == before
    assert 'calibration_checkpoint_every' not in result['recipe']
    assert set(result) == {'adapter_version', 'recipe', 'source', 'python', 'packages',
                           'uv_lock_sha256'}


@pytest.mark.parametrize('value', [0, -1, True, False, 1.5, '8', None])
def test_invalid_cadence_is_rejected_without_inserting_defaults(value):
    recipe = {'calibration_checkpoint_every': value}
    with pytest.raises(ValueError, match='positive integer'):
        checkpoint_save_every(recipe)
    with pytest.raises(ValueError, match='positive integer'):
        provenance(recipe, Path('/pinned/source'))


@pytest.mark.parametrize('literal', ['0', '-2', 'true', '1.5', '"8"'])
def test_recipe_loader_validates_cadence(tmp_path, literal):
    path = tmp_path / 'recipe.toml'
    path.write_text(f'calibration_checkpoint_every = {literal}\n' +
                    '\n'.join(line for line in Path('recipes/tiny.toml').read_text().splitlines()
                              if not line.startswith('calibration_checkpoint_every')))
    with pytest.raises(ValueError, match='positive integer'):
        read_recipe(path)


def test_explicit_cadence_is_recorded_and_layerwise_default_stays_one(tmp_path):
    recipe = read_recipe(Path('recipes/tiny.toml'))
    recipe.pop('calibration_checkpoint_every', None)
    recipe.update(mode='w4a4', algorithm='mse')
    model = model_fixture()
    default = quant_config(model, recipe, tmp_path)
    assert default['algorithm']['layerwise'] == {
        'enable': True, 'checkpoint_dir': str(tmp_path),
        'calib_mutates_weights': False, 'save_every': 1}
    changed = {**recipe, 'calibration_checkpoint_every': 8}
    altered = quant_config(model, changed, tmp_path)
    assert altered['algorithm']['layerwise']['save_every'] == 8
    altered['algorithm']['layerwise']['save_every'] = 1
    assert altered == default
    old = provenance(recipe, Path('/pinned/source'))
    new = provenance(changed, Path('/pinned/source'))
    assert old != new and new['recipe']['calibration_checkpoint_every'] == 8
    assert provenance({**recipe, 'calibration_checkpoint_every': 1}, Path('/pinned/source')) != old


def test_existing_checkpoint_cadence_drift_is_rejected(tmp_path):
    from modelopt.torch.quantization.utils.layerwise_calib import _CheckpointState

    (tmp_path / 'manifest.json').write_text(json.dumps({
        'last_completed_layer': 0, 'num_layers': 3, 'save_every': 8,
        'calib_mutates_weights': False}))
    with pytest.raises(ValueError, match='save_every mismatch'):
        _CheckpointState.from_folder(str(tmp_path), 3, save_every=1, calib_mutates_weights=False)

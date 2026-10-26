"""The public CLI must keep routing measurements and pooled policy distinct."""
from contextlib import nullcontext
from pathlib import Path
import sys

import pytest

from ming3_nvfp4 import cli, common, pipeline, pooled_calibration


def setup_flash(monkeypatch, arguments):
    recipe = common.read_recipe(Path('recipes/flash.toml'))
    monkeypatch.setattr(common, 'read_recipe', lambda _: recipe)
    monkeypatch.setattr(common, 'snapshot', lambda *_: Path('/source'))
    monkeypatch.setattr(cli, 'gpu_job', nullcontext)
    monkeypatch.setattr(sys, 'argv', ['ming-quant', *arguments])
    return recipe


def test_flash_coverage_records_pre_policy_identity(monkeypatch):
    recipe = setup_flash(monkeypatch, ['coverage', '--manifest', '/data',
                                       '--output', '/coverage', '--layerwise'])
    def measure(actual, source, manifest, output, layerwise, smoke):
        assert actual == {k: v for k, v in recipe.items() if k != 'expert_coverage_policy'}
        assert layerwise and not smoke
        return {'coverage_passed': False}
    monkeypatch.setattr(pipeline, 'coverage', measure)
    cli.main()
    assert 'expert_coverage_policy' in recipe


@pytest.mark.parametrize('options', [[], ['--layerwise'],
    ['--layerwise', '--coverage-from', '/coverage'],
    ['--layerwise', '--coverage-from', '/coverage', '--validation', '/proof', '--smoke']])
def test_flash_missing_preconditions_fail_before_gpu(monkeypatch, options):
    setup_flash(monkeypatch, ['calibrate', '--manifest', '/data', '--output', '/cal', *options])
    def forbidden():
        pytest.fail('Invalid Flash command tried to acquire GPU')
    monkeypatch.setattr(cli, 'gpu_job', forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_flash_calibration_dispatches_with_required_validation(monkeypatch):
    recipe = setup_flash(monkeypatch, ['calibrate', '--manifest', '/data', '--output', '/cal',
        '--layerwise', '--coverage-from', '/coverage', '--validation', '/proof'])
    def calibrate(actual, source, manifest, output, coverage, *, validation):
        assert actual == recipe
        assert coverage == Path('/coverage') and validation == Path('/proof')
        return {'calibration_policy_passed': True, 'coverage_passed': False}
    monkeypatch.setattr(pooled_calibration, 'calibrate_pooled', calibrate)
    monkeypatch.setattr(pipeline, 'calibrate', lambda *_: pytest.fail('Used strict coverage path'))
    cli.main()


def test_flash_export_revalidates_pooled_policy(monkeypatch):
    setup_flash(monkeypatch, ['export', '--calibration', '/cal', '--output', '/out'])
    calls = []
    monkeypatch.setattr(pooled_calibration, 'export_pooled', lambda *args: calls.append(args) or {})
    monkeypatch.setattr(pipeline, 'export', lambda *_: pytest.fail('Bypassed pooled validation'))
    cli.main()
    assert len(calls) == 1

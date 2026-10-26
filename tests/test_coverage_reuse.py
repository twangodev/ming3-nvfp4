import copy
import json

import pytest

from ming3_nvfp4.common import file_hash
from ming3_nvfp4.pipeline import reuse_coverage


@pytest.fixture
def producer(tmp_path):
    identity = {'provenance': {'adapter_version': 2, 'source': '/pinned/source', 'packages': {'torch': '2.8'},
                              'recipe': {'mode': 'w4a16', 'algorithm': 'mse',
                                         'retain_layers': [], 'quality': {'margin': 1},
                                         'min_expert_tokens': 128, 'seed': 42}},
                'manifest_sha256': 'same-data', 'layerwise': True, 'smoke': False,
                'batch_limits': {'size': 8, 'tokens': 4096}}
    (tmp_path / 'run.json').write_text(json.dumps(identity))
    (tmp_path / 'coverage.json').write_text(json.dumps({
        'counts': {'layer1.expert0': 200, 'layer2.expert0': 16},
        'minimum_tokens': 1, 'under_observed': []}))
    return tmp_path, identity


def test_reuse_filters_retained_layers_modes_and_recomputes_gate(producer):
    path, identity = producer
    identity = copy.deepcopy(identity)
    identity['provenance']['recipe'].update(mode='w4a4', algorithm='max',
                                           retain_layers=[2], quality={'margin': 2})
    report = reuse_coverage(path, identity, ['layer1.expert0'], 128)
    assert report['counts'] == {'layer1.expert0': 200}
    assert report['under_observed'] == []
    assert report['origin_coverage_sha256'] == file_hash(path / 'coverage.json')
    report = reuse_coverage(path, identity, ['layer2.expert0'], 128)
    assert report['under_observed'] == ['layer2.expert0']
    assert report['minimum_tokens'] == 128


@pytest.mark.parametrize('field', ['source', 'manifest', 'batch', 'layerwise', 'seed', 'adapter_version'])
def test_rejects_changed_execution_or_data_provenance(producer, field):
    path, identity = producer
    identity = copy.deepcopy(identity)
    if field == 'source':
        identity['provenance']['source'] = '/other/source'
    elif field == 'manifest':
        identity['manifest_sha256'] = 'different-samples'
    elif field == 'batch':
        identity['batch_limits']['size'] = 1
    elif field == 'layerwise':
        identity['layerwise'] = False
    elif field == 'adapter_version':
        identity['provenance']['adapter_version'] = 3
    else:
        identity['provenance']['recipe']['seed'] = 7
    with pytest.raises(ValueError, match='provenance differs'):
        reuse_coverage(path, identity, ['layer1.expert0'], 128)


def test_rejects_missing_or_invalid_measured_counts(producer):
    path, identity = producer
    with pytest.raises(ValueError, match='missing selected experts'):
        reuse_coverage(path, identity, ['missing.expert'], 128)
    for count in [-1, True, 12.5]:
        (path / 'coverage.json').write_text(json.dumps({'counts': {'layer1.expert0': count}}))
        with pytest.raises(ValueError, match='nonnegative integers'):
            reuse_coverage(path, identity, ['layer1.expert0'], 128)
    (path / 'coverage.json').unlink()
    with pytest.raises(FileNotFoundError):
        reuse_coverage(path, identity, ['layer1.expert0'], 128)

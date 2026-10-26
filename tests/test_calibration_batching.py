import torch
import pytest

from ming3_nvfp4.adapter import CalibrationBatches, collate_calibration_inputs


def example(length, audio=None):
    value = {'input_ids': torch.ones(1, length, dtype=torch.long),
             'attention_mask': torch.ones(1, length, dtype=torch.long)}
    if audio is not None:
        value['audio_feats'] = torch.full((1, audio + 2, 3), float(audio))
        value['audio_feats_lengths'] = torch.tensor([[audio, 0]])
    return value


def test_audio_collation_keeps_row_order_and_trims_unused_frames():
    result = collate_calibration_inputs([example(4, 2), example(4), example(4, 3)])
    assert result['input_ids'].shape == (3, 4)
    assert result['audio_feats'].shape == (1, 5, 3)
    assert result['audio_feats_lengths'].tolist() == [[2, 3]]
    assert result['audio_feats'][0, :, 0].tolist() == [2, 2, 3, 3, 3]
    assert result['attention_mask'].sum().item() == 12
    with pytest.raises(ValueError, match='equal token lengths'):
        collate_calibration_inputs([example(3), example(4)])


def test_exact_length_plan_token_limits_and_bounded_cache(monkeypatch):
    import ming3_nvfp4.adapter as adapter
    calls = []

    def prepare(_processor, row, **_kwargs):
        calls.append(row['id'])
        return example(row['length']), 0

    monkeypatch.setattr(adapter, 'prepare_inputs', prepare)
    rows = [{'id': str(i), 'length': length} for i, length in enumerate([4, 3, 4, 4, 3])]
    recipe = {'max_tokens': 8, 'calibration_batch_size': 8, 'calibration_batch_tokens': 8}
    batches = CalibrationBatches(None, rows, recipe, cache_bytes=64)
    assert batches.plan == [[0, 2], [3], [1, 4]]
    assert batches.metadata['max_batch_tokens'] == 8
    assert batches.metadata['cached_bytes'] <= 64
    original_calls = len(calls)
    assert sum(value['input_ids'].numel() for value in batches) == 18
    assert len(calls) == original_calls + 4
    # All tokens routed into experts are real: no padding inflates coverage.
    assert all(value['attention_mask'].all() for value in batches)


def test_batch_limits_are_validated(monkeypatch):
    for size in [0, 9, True]:
        with pytest.raises(ValueError, match='calibration_batch_size'):
            CalibrationBatches(None, [], {'calibration_batch_size': size})
    with pytest.raises(ValueError, match='calibration_batch_tokens'):
        CalibrationBatches(None, [], {'calibration_batch_tokens': 0})


def test_batch_limits_change_calibration_provenance(tmp_path):
    from ming3_nvfp4.common import provenance

    recipe = {'calibration_batch_size': 1, 'calibration_batch_tokens': 4096}
    baseline = provenance(recipe.copy(), tmp_path)
    assert provenance({**recipe, 'calibration_batch_size': 8}, tmp_path) != baseline
    assert provenance({**recipe, 'calibration_batch_tokens': 8192}, tmp_path) != baseline

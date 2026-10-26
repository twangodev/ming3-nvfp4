"""CPU-only reproduction with the actual pinned Transformers loader lifecycle."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


def load_fix():
    from ming3_nvfp4.adapter import rebuild_ming_rotary_buffers
    return rebuild_ming_rotary_buffers


def rope_init(config, device):
    return 1 / (config.theta ** (torch.arange(0, 8, 2, device=device).float() / 8)), 1.25


class BailingMoeV3RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(theta=10000)
        self.rope_init_fn = rope_init
        value, self.attention_scaling = rope_init(self.config, None)
        self.register_buffer('inv_freq', value, persistent=False)
        self.original_inv_freq = self.inv_freq


class MinimalMing(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.rotary = BailingMoeV3RotaryEmbedding()
        self.learned = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

    def _init_weights(self, module):
        # Ming's override handles Linear/Embedding only; neither is present here.
        pass


def test_actual_hf_nonpersistent_materialization_and_repair(monkeypatch):
    with torch.device('meta'):
        model = MinimalMing(PretrainedConfig())
    model.learned = nn.Parameter(torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16))
    model.all_tied_weights_keys = {}
    original = model.learned
    snapshot = original.detach().clone()
    # Poison only the actual HF empty_like allocation, making its missing
    # initialization deterministic rather than relying on allocator contents.
    real_empty = torch.empty_like

    def poisoned_empty(*args, **kwargs):
        value = real_empty(*args, **kwargs)
        if value.shape == (4,) and value.dtype == torch.float32:
            value.fill_(float('nan'))
        return value

    monkeypatch.setattr(torch, 'empty_like', poisoned_empty)
    model._move_missing_keys_from_meta_to_device(set(), {'': 'cpu'}, None, None)
    model._initialize_missing_keys(False)
    assert torch.isnan(model.rotary.inv_freq).all()
    assert model.rotary.original_inv_freq.is_meta
    fix = load_fix()
    fix(model)
    expected, scaling = rope_init(model.rotary.config, 'cpu')
    assert torch.equal(model.rotary.inv_freq, expected)
    assert model.rotary.original_inv_freq is model.rotary.inv_freq
    assert model.rotary.attention_scaling == scaling
    assert torch.isfinite(torch.cos(torch.arange(8)[:, None] * model.rotary.inv_freq)).all()
    assert model.learned is original
    assert torch.equal(model.learned, snapshot)
    assert model.learned.dtype == torch.bfloat16
    assert model.rotary.inv_freq.dtype == torch.float32
    assert 'inv_freq' in model.rotary._non_persistent_buffers_set
    assert 'rotary.inv_freq' not in model.state_dict()
    first = model.rotary.inv_freq.clone()
    fix(model)
    assert torch.equal(first, model.rotary.inv_freq)


def test_unmaterialized_meta_rejected():
    with torch.device('meta'):
        model = MinimalMing(PretrainedConfig())
    with pytest.raises(RuntimeError, match='not materialized'):
        load_fix()(model)


def test_other_rotary_classes_untouched():
    class OtherRotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('inv_freq', torch.tensor([42.]))
    model = OtherRotary()
    original = model.inv_freq
    load_fix()(model)
    assert model.inv_freq is original

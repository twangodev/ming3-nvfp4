from __future__ import annotations

import copy
from pathlib import Path

from .inventory import EXPERT, require_complete
from .common import checkpoint_save_every, disk_budget


class BaseLayerView:
    """Mutable view: ModelOpt must replace actual layers while excluding MTP."""

    def __init__(self, model):
        self.layers = model.model.layers
        self.count = model.config.text_config.num_hidden_layers

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.count))]
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        return self.layers[index]

    def __setitem__(self, index, layer):
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.layers[index] = layer


def prepare_layerwise_offload(model):
    def replay_before_accelerate(layer, args, kwargs):
        state = getattr(layer, "_layerwise_calib", None)
        if state is not None and state.mode == "run" and state.cached_inputs:
            # Accelerate runs before ModelOpt's patched forward. Give its device
            # hook the same real inputs that ModelOpt will pop and replay, rather
            # than the skipped predecessor's storage-free meta placeholders.
            return state.cached_inputs[0]

    for layer in BaseLayerView(model):
        if not getattr(layer, "_ming_replay_hook", False):
            layer.register_forward_pre_hook(replay_before_accelerate, with_kwargs=True)
            layer._ming_replay_hook = True


def restore_quantizer_buffers(model, buffers: dict, execution_device="cuda:0"):
    from .calibration_state import restore_quantizer_buffers as restore

    return restore(model, buffers, execution_device)


def rebuild_ming_rotary_buffers(model):
    """Rebuild Ming derived RoPE state discarded by Transformers 5 meta loading."""
    import torch

    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ != "BailingMoeV3RotaryEmbedding":
                continue
            device = module.inv_freq.device
            if device.type == "meta":
                raise RuntimeError("Ming rotary buffer was not materialized by the loader")
            inv_freq, attention_scaling = module.rope_init_fn(module.config, device)
            if inv_freq.device != device or not torch.isfinite(inv_freq).all():
                raise RuntimeError("Ming rotary initializer produced invalid frequencies")
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.original_inv_freq = module.inv_freq
            module.attention_scaling = attention_scaling


def load_model(source: Path, recipe: dict, offload_dir: Path):
    import torch
    from contextlib import nullcontext
    from transformers import AutoConfig, AutoProcessor
    from .source_offload import original_shard_offload_budget
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    source_only = recipe.get("offload_source_shards_only", False)
    if type(source_only) is not bool:
        raise ValueError("offload_source_shards_only must be a boolean")
    loading_budget = nullcontext(None)
    inventory = require_complete(source, recipe["retain_layers"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run outside the GPU-restricted sandbox")
    if torch.cuda.get_device_capability()[0] < 12:
        raise RuntimeError("This recipe targets the inspected SM120 Blackwell workstation")
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    # Eager matches the official reference and avoids changing attention implementations
    # between baseline and quantization. KDA still uses the original FLA operators.
    kwargs = dict(trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
                  attn_implementation="eager", low_cpu_mem_usage=True)
    if inventory["root_bytes"] < recipe["gpu_memory_gib"] * 2**30 * 0.7:
        kwargs["device_map"] = {"": 0}
    else:
        # Reserve enough for spill and a largest-layer loading buffer before
        # Accelerate starts creating disk-offload files.
        spill = max(0, inventory["root_bytes"] -
                    (recipe["gpu_memory_gib"] + recipe["cpu_memory_gib"]) * 2**30)
        if source_only:
            loading_budget = original_shard_offload_budget(
                source, offload_dir, recipe["disk_floor_gib"], 16 * 2**30)
            kwargs["disable_mmap"] = False
        else:
            disk_budget(offload_dir, recipe["disk_floor_gib"], spill + 16 * 2**30)
        kwargs.update(device_map="auto",
                      max_memory={0: f"{recipe['gpu_memory_gib']}GiB",
                                  "cpu": f"{recipe['cpu_memory_gib']}GiB"},
                      offload_folder=str(offload_dir), offload_state_dict=True)
    config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    model_cls = get_class_from_dynamic_module(
        config.auto_map["AutoModelForCausalLM"], str(source), local_files_only=True)
    # These source tensors are FP32. Transformers otherwise downcasts them even
    # though the recurrent update and router bias use higher-precision arithmetic.
    model_cls._keep_in_fp32_modules_strict = {"A_log", "dt_bias", "expert_bias"}
    with loading_budget as load_audit:
        model = model_cls.from_pretrained(source, config=config, **kwargs).eval()
    rebuild_ming_rotary_buffers(model)
    if load_audit is not None:
        model._source_offload_audit = load_audit
    model.requires_grad_(False)
    # The audio wrapper always supplies inputs_embeds, so the original forward skips
    # MTP execution. Keep its parameters intact for export.
    model._mtp_layer_prefixes = [f"model.layers.{i}." for i in range(
        model.config.text_config.num_hidden_layers, len(model.model.layers))]
    prepare_layerwise_offload(model)
    return model, processor


def prepare_inputs(processor, row: dict, *, teacher_forcing=False, max_tokens=2048):
    import torch

    messages = copy.deepcopy(row["messages"])
    audio_count = sum(item.get("type") == "audio" for msg in messages
                      if isinstance(msg["content"], list) for item in msg["content"])
    if audio_count:
        if audio_count != 1:
            raise ValueError("Pinned Ming processor supports exactly one audio item per conversation")
        inputs = dict(processor(messages))
    else:
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = dict(processor.tokenizer(prompt, return_tensors="pt"))
        inputs.pop("token_type_ids", None)
    prompt_length = inputs["input_ids"].shape[1]
    if teacher_forcing:
        answer = processor.tokenizer(row["reference"], add_special_tokens=False,
                                     return_tensors="pt")["input_ids"]
        inputs["input_ids"] = torch.cat((inputs["input_ids"], answer), dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    if inputs["input_ids"].shape[1] > max_tokens:
        raise ValueError(f"Sample {row['id']} exceeds {max_tokens} tokens; use a longer-context "
                         "recipe instead of truncating audio placeholders or answers")
    return inputs, prompt_length



def collate_calibration_inputs(examples: list[dict]) -> dict:
    """Combine equal-length examples without padding or crossing audio boundaries."""
    import torch

    lengths = {item["input_ids"].shape[1] for item in examples}
    if len(lengths) != 1:
        raise ValueError("Calibration batches require exactly equal token lengths")
    result = {}
    text_keys = set().union(*(item.keys() for item in examples)) - {
        "audio_feats", "audio_feats_lengths"}
    for key in text_keys:
        if any(key not in item for item in examples):
            raise ValueError(f"Inconsistent calibration input: {key}")
        result[key] = torch.cat([item[key] for item in examples], dim=0)
    audio = [item for item in examples if "audio_feats" in item]
    if audio:
        segments, lengths = [], []
        for item in audio:
            valid = item["audio_feats_lengths"].reshape(-1)
            valid = valid[valid > 0]
            frames = int(valid.sum())
            if item["audio_feats"].shape[0] != 1 or frames > item["audio_feats"].shape[1]:
                raise ValueError("Invalid audio calibration features")
            segments.append(item["audio_feats"][:, :frames])
            lengths.append(valid)
        result["audio_feats"] = torch.cat(segments, dim=1)
        result["audio_feats_lengths"] = torch.cat(lengths).unsqueeze(0)
    return result


class CalibrationBatches:
    """Bounded CPU preprocessing cache and a deterministic exact-length batch plan."""

    def __init__(self, processor, rows, recipe, cache_bytes=2 * 2**30):
        from collections import defaultdict

        size = recipe.get("calibration_batch_size", 1)
        tokens = recipe.get("calibration_batch_tokens", 4096)
        if type(size) is not int or not 1 <= size <= 8:
            raise ValueError("calibration_batch_size must be an integer from 1 to 8")
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("calibration_batch_tokens must be a positive integer")
        self.processor, self.rows, self.recipe = processor, rows, recipe
        self.cache, self.lengths, self.plan = {}, [], []
        groups = defaultdict(list)
        used = 0
        for index, row in enumerate(rows):
            inputs = self._prepare(row)
            length = inputs["input_ids"].shape[1]
            if length > tokens:
                raise ValueError(f"Sample {row['id']} exceeds calibration_batch_tokens={tokens}")
            self.lengths.append(length)
            groups[length].append(index)
            amount = sum(value.numel() * value.element_size() for value in inputs.values())
            if used + amount <= cache_bytes:
                self.cache[index] = inputs
                used += amount
        for length, indices in groups.items():
            count = min(size, tokens // length)
            self.plan.extend(indices[start:start + count]
                             for start in range(0, len(indices), count))
        # Preserve original order for the default single-example path.
        if size == 1:
            self.plan.sort(key=lambda indices: indices[0])
        counts = [len(indices) for indices in self.plan]
        self.metadata = {"batch_size_limit": size, "batch_token_limit": tokens,
                         "batch_count": len(counts), "example_count": len(rows),
                         "max_batch_size": max(counts, default=0),
                         "average_batch_size": len(rows) / len(counts) if counts else 0,
                         "max_batch_tokens": max((len(ids) * self.lengths[ids[0]]
                                                  for ids in self.plan), default=0),
                         "cached_examples": len(self.cache), "cached_bytes": used,
                         "cache_limit_bytes": cache_bytes, "plan": self.plan}

    def _prepare(self, row):
        return prepare_inputs(self.processor, row, teacher_forcing=True,
                              max_tokens=self.recipe["max_tokens"])[0]

    def __iter__(self):
        for indices in self.plan:
            yield collate_calibration_inputs([
                self.cache[index] if index in self.cache else self._prepare(self.rows[index])
                for index in indices])

def to_device(inputs: dict, device="cuda:0") -> dict:
    import torch

    return {k: v.to(device=device, dtype=torch.bfloat16 if k == "audio_feats" else v.dtype)
            for k, v in inputs.items()}


def register_ming():
    from modelopt.torch.export.layer_utils import set_expert_quantizer_amax
    from modelopt.torch.export.registry import PrepareMoEInputsRegistry
    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    LayerActivationCollector.register_decoder_layer_support(
        lambda m: type(m).__name__ == "BailingMoeV3AudioForConditionalGeneration",
        BaseLayerView,
    )

    @PrepareMoEInputsRegistry.register(
        predicate=lambda experts: len(experts) > 0
        and type(experts[0]).__name__ == "BailingMoeV3MLP"
        if hasattr(experts, "__len__") and hasattr(experts, "__getitem__") else False,
        prepend=True,
    )
    def prepare_ming_experts(name, module, context):
        # ModelOpt's generic iterable handler assumes Mixtral's w1/w2/w3 names.
        for projection in ("gate_proj", "up_proj", "down_proj"):
            linears = [getattr(expert, projection) for expert in module.experts]
            if any(getattr(linear, "input_quantizer", None) is not None
                   and linear.input_quantizer.is_enabled for linear in linears):
                set_expert_quantizer_amax(linears, quantizer_attrs=["input_quantizer"])


def synchronize_export_scales(model):
    import torch

    synced = 0
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or not hasattr(experts, "__iter__"):
            continue
        for expert in experts:
            gate = getattr(getattr(expert, "gate_proj", None), "weight_quantizer", None)
            up = getattr(getattr(expert, "up_proj", None), "weight_quantizer", None)
            if gate is None or up is None or not gate.is_enabled or not up.is_enabled:
                continue
            # Static MSE has independent per-block amax and shared global amax.
            # ModelOpt 0.47's stock sync incorrectly overwrites per-block thresholds.
            attr = "global_amax" if getattr(gate, "is_nvfp4_static", False) else "amax"
            left, right = getattr(gate, attr, None), getattr(up, attr, None)
            if left is None or right is None or left.is_meta or right.is_meta:
                raise ValueError("Missing calibrated gate/up scale before export")
            if not torch.equal(left, right):
                shared = torch.maximum(left, right.to(left.device))
                setattr(gate, attr, shared)
                setattr(up, attr, shared.to(right.device))
                synced += 1
    return synced




def _restore_prefix_shared_scales(layers, start_layer):
    """Rebuild static weight-global aliases after loading the resumed prefix."""
    import torch
    from modelopt.torch.quantization.utils.shared_input import (
        SharedWeightGlobalAmaxState, find_shared_input_groups,
    )

    patterns = SharedWeightGlobalAmaxState.default_patterns
    repairs = []
    for index in range(start_layer):
        layer = layers[index]
        groups = find_shared_input_groups(layer, patterns=patterns)
        has_static_group = False
        for _, members in groups:
            quantizers = [member.weight_quantizer for member in members]
            if not all(getattr(quantizer, "is_nvfp4_static", False)
                       for quantizer in quantizers):
                continue
            globals_ = [quantizer.global_amax for quantizer in quantizers]
            if any(value is None or value.is_meta for value in globals_):
                raise ValueError("Missing restored static shared global scale")
            reference = globals_[0]
            if any(not torch.equal(reference, value.to(reference.device))
                   for value in globals_[1:]):
                raise ValueError("Conflicting restored static shared global scales")
            has_static_group = True
        if has_static_group:
            repairs.append(layer)
    # Validate all prefix groups before tying any buffer. Do not derive globals
    # from learned block _amax or alter activation calibration/sweep ordering.
    for layer in repairs:
        SharedWeightGlobalAmaxState.restore(layer, patterns=patterns)

def calibration_memory_context():
    """Bound MSE search state and non-mutating resumed-prefix restoration."""
    from contextlib import contextmanager
    from unittest.mock import patch
    import modelopt.torch.quantization.model_calib as calibration
    from modelopt.torch.quantization.calib.mse import NVFP4MSECalibrator
    from modelopt.torch.quantization.utils import core_utils
    from modelopt.torch.quantization.utils.layerwise_calib import _CheckpointState

    original = calibration._mse_calibrate_weights
    original_restore = _CheckpointState.full_restore
    original_access = core_utils.enable_weight_access_and_writeback

    def bounded_mse(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        for module in model.modules():
            calibrator = getattr(module, "_calibrator", None)
            if isinstance(calibrator, NVFP4MSECalibrator):
                # Neither field participates in QDQ/export or ModelOpt state.
                # A subsequent calibration constructs a fresh MSE calibrator.
                calibrator._initial_amax = None
                calibrator._hessian = None
        return result

    def bounded_restore(checkpoint, layers, model):
        if checkpoint.start_layer == 0 or checkpoint.calib_mutates_weights:
            return original_restore(checkpoint, layers, model)

        @contextmanager
        def readonly_access(module, root_model, name_to_module=None, writeback=True):
            with original_access(module, root_model, name_to_module, writeback=False):
                yield

        # ModelOpt 0.47 otherwise promotes disk-backed BF16 weights into its CPU
        # weights_map even though this algorithm only restores quantizer state.
        with patch.object(core_utils, "enable_weight_access_and_writeback", readonly_access):
            result = original_restore(checkpoint, layers, model)
        _restore_prefix_shared_scales(layers, checkpoint.start_layer)
        return result

    @contextmanager
    def guard():
        with (
            patch.object(calibration, "_mse_calibrate_weights", bounded_mse),
            patch.object(_CheckpointState, "full_restore", bounded_restore),
        ):
            yield

    return guard()

def export_scale_sync_context():
    """Guard ModelOpt 0.47's exporter safety net against static-scale corruption."""
    from contextlib import ExitStack
    from unittest.mock import patch
    import modelopt.torch.export.layer_utils as layers
    import modelopt.torch.export.unified_export_hf as unified

    stack = ExitStack()
    stack.enter_context(patch.object(layers, "sync_moe_gate_up_amax", synchronize_export_scales))
    stack.enter_context(patch.object(unified, "sync_moe_gate_up_amax", synchronize_export_scales))
    return stack


def selected_linears(model, recipe: dict) -> dict:
    import torch

    selected = {}
    for name, module in model.named_modules():
        match = EXPERT.fullmatch(name + ".weight")
        if (match and int(match[1]) < model.config.text_config.num_hidden_layers
                and int(match[1]) not in recipe["retain_layers"]):
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"Unexpected expert type: {name}: {type(module)}")
            selected[name] = module
    if not selected:
        raise ValueError("Recipe selected no routed expert projections")
    return selected


def quant_config(model, recipe: dict, checkpoint_dir: Path | None = None) -> dict:
    import modelopt.torch.quantization as mtq

    if recipe["algorithm"] == "mse":
        base = copy.deepcopy(mtq.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG)
    elif recipe["algorithm"] == "local_hessian":
        base = copy.deepcopy(mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG)
    else:
        base = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    entries = base["quant_cfg"]
    weight = next(copy.deepcopy(e["cfg"]) for e in entries
                  if e["quantizer_name"] == "*weight_quantizer" and "cfg" in e)
    inputs = next(copy.deepcopy(e["cfg"]) for e in entries
                  if e["quantizer_name"] == "*input_quantizer" and "cfg" in e)
    rules = [{"quantizer_name": "*", "enable": False}]
    selected_linears(model, recipe)  # Fail on unexpected expert structure before conversion.
    for layer in range(model.config.text_config.first_k_dense_replace,
                       model.config.text_config.num_hidden_layers):
        if layer in recipe["retain_layers"]:
            continue
        prefix = f"model.layers.{layer}.mlp.experts.*"
        rules.append({"quantizer_name": prefix + ".weight_quantizer", "cfg": weight})
        if recipe["mode"] == "w4a4":
            rules.append({"quantizer_name": prefix + ".input_quantizer", "cfg": inputs})
    algorithm = base["algorithm"]
    if isinstance(algorithm, str):
        algorithm = {"method": algorithm}
    if checkpoint_dir:
        for entry in algorithm if isinstance(algorithm, list) else [algorithm]:
            entry["layerwise"] = {"enable": True, "checkpoint_dir": str(checkpoint_dir),
                                  "calib_mutates_weights": False,
                                  "save_every": checkpoint_save_every(recipe)}
    return {"quant_cfg": rules, "algorithm": algorithm}


class ExpertCoverage:
    """Measure actual routed token coverage without changing routing decisions."""

    def __init__(self, model, recipe):
        self.counts, self.handles = {}, []
        for name, module in selected_linears(model, recipe).items():
            if not name.endswith("gate_proj"):
                continue
            expert = name.rsplit(".", 1)[0]
            self.counts[expert] = 0

            def hook(_module, args, key=expert):
                self.counts[key] += args[0].numel() // args[0].shape[-1]

            self.handles.append(module.register_forward_pre_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def report(self, minimum: int) -> dict:
        return {"minimum_tokens": minimum, "counts": self.counts,
                "under_observed": [k for k, v in self.counts.items() if v < minimum]}

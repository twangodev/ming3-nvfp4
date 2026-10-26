from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .adapter import (BaseLayerView, CalibrationBatches, ExpertCoverage, load_model, quant_config,
                      register_ming, selected_linears, export_scale_sync_context, calibration_memory_context,
                      synchronize_export_scales, to_device)
from .common import disk_budget, file_hash, provenance, write_json
from .inventory import audit_export, header, require_complete
from .manifest import load_manifest


def _seed(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(12)


def restore_calibration(model, calibration: Path, recipe: dict, source: Path):
    import modelopt.torch.opt as mto
    import torch
    from .calibration_state import restore_quantizer_file

    run = json.loads((calibration / "calibration.json").read_text())
    if run["provenance"] != provenance(recipe, source):
        raise ValueError("Calibration recipe, source, Python packages, or uv lock changed")
    # This is a locally generated ModelOpt state, never a downloaded pickle.
    mto.restore_from_modelopt_state(model, torch.load(
        calibration / "modelopt_state.pt", map_location="cpu", weights_only=False))
    restore_quantizer_file(model, calibration / "quantizers.safetensors")
    synchronize_export_scales(model)
    return run


def reuse_coverage(coverage_from: Path, identity: dict, selected_experts, minimum: int) -> dict:
    """Reuse only measured BF16 routes with identical data and execution provenance."""
    import copy

    coverage_path = coverage_from / "coverage.json" if coverage_from.is_dir() else coverage_from
    producer_run = coverage_path.parent / "run.json"
    producer = json.loads(producer_run.read_text())

    def comparable(run):
        result = copy.deepcopy(run)
        recipe = result["provenance"]["recipe"]
        for key in ("mode", "algorithm", "retain_layers", "quality"):
            recipe.pop(key, None)
        return result

    if comparable(producer) != comparable(identity):
        raise ValueError("Coverage producer source, manifest, execution, or recipe provenance differs")
    measured = json.loads(coverage_path.read_text())
    counts = measured.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Coverage producer has no measured expert counts")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Coverage counts must be nonnegative integers")
    if missing := set(selected_experts) - counts.keys():
        raise ValueError(f"Coverage producer is missing selected experts: {sorted(missing)}")
    counts = {name: counts[name] for name in selected_experts}
    return {"minimum_tokens": minimum, "counts": counts,
            "under_observed": [name for name, count in counts.items() if count < minimum],
            "origin_run": str(producer_run.resolve()),
            "origin_coverage": str(coverage_path.resolve()),
            "origin_coverage_sha256": file_hash(coverage_path)}

def _require_finite_output(value):
    """Reject corrupted BF16/calibration activations before persisting success."""
    import torch

    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite model output during coverage or calibration")
    elif isinstance(value, dict):
        for item in value.values():
            _require_finite_output(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_output(item)


def _measure_routes(model, forward_loop, layerwise):
    """Run each original-precision layer exactly once, including the final layer."""
    import torch

    if layerwise:
        from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

        collector = LayerActivationCollector(model)
        layers = BaseLayerView(model)
        collector._patch_all_layers(layers)
        try:
            with torch.inference_mode():
                for layer in layers:
                    captured = collector.get_input_activations(layer, forward_loop)
                # Each preceding layer ran once while capturing the next one.
                for args, kwargs in captured:
                    _require_finite_output(layers[-1](*args, **kwargs))
        finally:
            collector._unpatch_all_layers()
    else:
        forward_loop(model)


def _expected_coverage_experts(source: Path, recipe: dict) -> set[str]:
    """Derive routed base-expert scope without loading weights or a GPU model."""
    text = json.loads((source / "config.json").read_text())["text_config"]
    first, layers, experts = (text["first_k_dense_replace"],
                              text["num_hidden_layers"], text["num_experts"])
    if (any(type(value) is not int for value in (first, layers, experts))
            or not 0 <= first < layers or experts < 1):
        raise ValueError("Invalid source routed-expert dimensions")
    retained = set(recipe["retain_layers"])
    selected = {f"model.layers.{layer}.mlp.experts.{expert}"
                for layer in range(first, layers) if layer not in retained
                for expert in range(experts)}
    if not selected:
        raise ValueError("Recipe selected no routed expert projections")
    return selected


def coverage(recipe: dict, source: Path, manifest: Path, output: Path,
             layerwise=False, smoke=False):
    """Persist real BF16 routing counts without converting or calibrating quantizers.

    The run identity is deliberately identical to calibration, so calibration can
    reuse these counts with --coverage-from while preserving all execution guards.
    Under-observed experts are reported; calibration enforces the production gate.
    """
    import torch

    rows, checksum = load_manifest(manifest, "smoke" if smoke else "calibration")
    if not smoke and len(rows) < 1024:
        raise ValueError("Production coverage requires at least 1024 examples")
    if recipe["variant"] == "flash" and not layerwise:
        raise ValueError("Flash coverage requires --layerwise")
    _seed(recipe["seed"])
    identity = {"provenance": provenance(recipe, source), "manifest_sha256": checksum,
                "layerwise": layerwise, "smoke": smoke,
                "batch_limits": {"size": recipe.get("calibration_batch_size", 1),
                                 "tokens": recipe.get("calibration_batch_tokens", 4096)}}
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "run.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Resume configuration differs; use a new coverage directory")
    coverage_path = output / "coverage.json"
    if coverage_path.exists():
        if not marker.exists():
            raise ValueError("Coverage has no run provenance")
        expected = _expected_coverage_experts(source, recipe)
        report = reuse_coverage(output, identity, expected, recipe["min_expert_tokens"])
        measured = json.loads(coverage_path.read_text())["counts"]
        if set(measured) != expected:
            raise ValueError("Cached coverage expert scope differs from source and recipe")
        return report
    disk_budget(output, recipe["disk_floor_gib"])
    write_json(marker, identity)
    model, processor = load_model(source, recipe, output / "offload")
    batches = CalibrationBatches(processor, rows, recipe)
    write_json(output / "batch-plan.json", batches.metadata)
    hidden = model.config.text_config.hidden_size
    estimate = sum(2 * (tokens * hidden * 2 + tokens * 256 * 2 + tokens * tokens * 2)
                   for tokens in batches.lengths)
    if layerwise and estimate > recipe["scratch_memory_gib"] * 2**30:
        raise RuntimeError(f"Layerwise activation cache estimate {estimate / 2**30:.1f} GiB "
                           f"exceeds scratch_memory_gib={recipe['scratch_memory_gib']}")

    def forward_loop(current):
        with torch.inference_mode():
            for inputs in batches:
                _require_finite_output(current(**to_device(inputs), use_cache=False, logits_to_keep=1))

    observer = ExpertCoverage(model, recipe)
    try:
        _measure_routes(model, forward_loop, layerwise)
        report = observer.report(recipe["min_expert_tokens"])
        write_json(coverage_path, report)
    finally:
        observer.close()
    return report


def calibrate(recipe: dict, source: Path, manifest: Path, output: Path,
              layerwise=False, smoke=False, coverage_from: Path | None = None):
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    import torch
    from .calibration_state import save_quantizer_buffers

    rows, checksum = load_manifest(manifest, "smoke" if smoke else "calibration")
    if not smoke and len(rows) < 1024:
        raise ValueError("Production calibration requires at least 1024 examples")
    if recipe["variant"] == "flash" and not layerwise:
        raise ValueError("Flash calibration requires --layerwise")
    _seed(recipe["seed"])
    identity = {"provenance": provenance(recipe, source), "manifest_sha256": checksum,
                "layerwise": layerwise, "smoke": smoke,
                "batch_limits": {"size": recipe.get("calibration_batch_size", 1),
                                 "tokens": recipe.get("calibration_batch_tokens", 4096)}}
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "run.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Resume configuration differs; use a new calibration directory")
    if (output / "calibration.json").exists():
        return json.loads((output / "calibration.json").read_text())
    if (output / "coverage.json").exists() and not marker.exists():
        raise ValueError("Coverage has no run provenance")
    disk_budget(output, recipe["disk_floor_gib"])
    write_json(marker, identity)
    model, processor = load_model(source, recipe, output / "offload")
    register_ming()
    # Bound the collector's in-memory replay cache before starting a long run.
    # Includes two copies of hidden states and two rotary embeddings; attention
    # masks have quadratic size with the reference eager implementation.
    hidden = model.config.text_config.hidden_size
    batches = CalibrationBatches(processor, rows, recipe)
    write_json(output / "batch-plan.json", batches.metadata)
    estimate = sum(2 * (tokens * hidden * 2 + tokens * 256 * 2 + tokens * tokens * 2)
                   for tokens in batches.lengths)
    if layerwise and estimate > recipe["scratch_memory_gib"] * 2**30:
        raise RuntimeError(f"Layerwise activation cache estimate {estimate / 2**30:.1f} GiB "
                           f"exceeds scratch_memory_gib={recipe['scratch_memory_gib']}")
    coverage = ExpertCoverage(model, recipe)

    def forward_loop(current):
        with torch.inference_mode():
            for inputs in batches:
                _require_finite_output(current(**to_device(inputs), use_cache=False, logits_to_keep=1))

    # Coverage comes from one original-precision pass, not repeated quantizer
    # replay passes. Persist it so a resumed run does not repeat this work.
    coverage_path = output / "coverage.json"
    if coverage_from is not None:
        coverage_report = reuse_coverage(coverage_from, identity, coverage.counts,
                                         recipe["min_expert_tokens"])
        write_json(coverage_path, coverage_report)
    elif coverage_path.exists():
        coverage_report = reuse_coverage(output, identity, coverage.counts,
                                         recipe["min_expert_tokens"])
    else:
        _measure_routes(model, forward_loop, layerwise)
        coverage_report = coverage.report(recipe["min_expert_tokens"])
        write_json(coverage_path, coverage_report)
    coverage.close()
    if coverage_report["under_observed"] and not smoke:
        raise RuntimeError(f"{len(coverage_report['under_observed'])} experts are under-observed; "
                           "expand calibration data or retain their layers at BF16")
    config = quant_config(model, recipe, output / "layers" if layerwise else None)
    write_json(output / "quantization.json", config)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with calibration_memory_context():
        model = mtq.quantize(model, config, forward_loop=forward_loop)
    synchronize_export_scales(model)
    selected = selected_linears(model, recipe)
    for name, module in selected.items():
        if not module.weight_quantizer.is_enabled:
            raise RuntimeError(f"Quantization silently skipped {name}")
    torch.save(mto.modelopt_state(model), output / "modelopt_state.pt")
    save_quantizer_buffers(model, output / "quantizers.safetensors")
    report = {**identity, "quantized_projections": len(selected),
              "seconds": time.perf_counter() - started,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "coverage_passed": not coverage_report["under_observed"],
              "release_approved": False, "batching": batches.metadata}
    write_json(output / "calibration.json", report)
    return report


def export(recipe: dict, source: Path, calibration: Path, output: Path):
    from modelopt.torch.export import export_hf_checkpoint

    if output.exists() and any(output.iterdir()):
        raise ValueError("Export directory must be empty; interrupted exports are not checkpoints")
    inventory = require_complete(source, recipe["retain_layers"])
    disk_budget(output, recipe["disk_floor_gib"], inventory["estimated_nvfp4_root_bytes"] +
                inventory["auxiliary_weight_bytes"])
    model, processor = load_model(source, recipe, calibration / "offload")
    register_ming()
    run = restore_calibration(model, calibration, recipe, source)
    output.mkdir(parents=True, exist_ok=True)
    with export_scale_sync_context():
        export_hf_checkpoint(model, export_dir=output, max_shard_size="2GB")
    # Export owns config.json; copy original custom code and tokenizer/processor
    # assets without replacing quantization metadata.
    for path in source.iterdir():
        if path.is_file() and path.suffix in {".py", ".json", ".jinja", ".txt"}:
            if path.name not in {"config.json", "model.safetensors.index.json"}:
                shutil.copy2(path, output / path.name)
    for directory in ("talker", "voice_presets"):
        if (source / directory).is_dir():
            shutil.copytree(source / directory, output / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    packed = {}
    for path in output.glob("*.safetensors"):
        for name, tensor in header(path).items():
            if ".experts." in name and name.endswith(".weight"):
                packed[name] = tensor["dtype"]
    quantized = sum(dtype in {"U8", "I8"} for dtype in packed.values())
    if quantized != run["quantized_projections"]:
        raise RuntimeError(f"Expected {run['quantized_projections']} packed projections; got {quantized}")
    audit = audit_export(source, output, recipe["retain_layers"])
    report = {"calibration": str(calibration.resolve()), "provenance": provenance(recipe, source),
              "packed_projections": quantized, "mode": recipe["mode"],
              "smoke_only": run["smoke"], "runtime_validated": False, "release_approved": False,
              "audit": audit}
    write_json(output / "ming_quantization.json", report)
    return report

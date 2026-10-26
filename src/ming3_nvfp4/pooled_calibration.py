"""Opt-in calibration policy using observed, unchanged ModelOpt expert input pooling.

Default CLI is CPU plan only. Raw routing coverage is never upgraded to passing.
This isolated entrypoint does not change the historical pipeline or recipe defaults.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import copy
import json
from pathlib import Path
import time

from .common import disk_budget, file_hash, provenance, write_json

POLICY = 'modelopt_pooled_inputs_v1'


def reuse_pooled_coverage(coverage_from, identity, selected_experts, minimum):
    """Ignore only the explicitly declared post-routing policy in consumer identity."""
    from .pipeline import reuse_coverage

    consumer = copy.deepcopy(identity)
    recipe = consumer['provenance']['recipe']
    if recipe.pop('expert_coverage_policy', None) != POLICY:
        raise ValueError('Explicit modelopt_pooled_inputs_v1 policy required')
    return reuse_coverage(coverage_from, consumer, selected_experts, minimum)


def valid_scale(value, *, positive, chunk_elements=1_048_576):
    """Bound temporary validation memory; never clone complete block-scale tensors."""
    import torch

    if value is None or value.is_meta or value.numel() == 0:
        return False
    if not value.is_contiguous():
        raise ValueError('Scale validation requires contiguous buffers; refusing a hidden full copy')
    flat = value.view(-1)
    for chunk in flat.split(chunk_elements):
        if not torch.isfinite(chunk).all() or not (chunk > 0 if positive else chunk >= 0).all():
            return False
    return True


@contextmanager
def observe_default_pooling(events):
    """Observe the existing hook, preserving its original invocation and arguments."""
    import torch
    from modelopt.torch.quantization.plugins import huggingface as hf

    original = hf.sync_moe_expert_amax

    def observed(experts, *args, **kwargs):
        sync_weight_amax = kwargs.get('sync_weight_amax', args[0] if args else False)
        if sync_weight_amax is not False:
            raise ValueError('Pooled-input policy requires stock sync_weight_amax=False')
        groups = {}
        for expert in experts:
            for name, quantizer in expert.named_modules():
                if name.endswith('input_quantizer') and getattr(quantizer, 'is_enabled', False):
                    groups.setdefault(name, []).append(quantizer)
        expected = {}
        donors = {}
        for name, quantizers in groups.items():
            values = [q.amax for q in quantizers if q.amax is not None]
            if not values or any(not valid_scale(value, positive=False) for value in values):
                raise ValueError(f'No finite positive measured input donors: {name}')
            # Derive expected maxima ONLY from real measured statistics. Inherited
            # synthetic defaults must never become donors or dominate the stock pool.
            measured = [getattr(getattr(q, '_calibrator', None), '_calib_amax', None)
                        for q in quantizers]
            measured = [value for value in measured if value is not None]
            if not measured or any(not valid_scale(value, positive=False) for value in measured):
                raise ValueError(f'No valid observed activation statistics for pool: {name}')
            pooled = measured[0].detach().clone()
            for value in measured[1:]:
                pooled = torch.maximum(pooled, value.to(pooled.device))
            if not valid_scale(pooled, positive=True):
                raise ValueError(f'No strictly positive measured donor maximum: {name}')
            inherited = values[0].detach().clone()
            for value in values[1:]:
                inherited = torch.maximum(inherited, value.to(inherited.device))
            if not torch.equal(inherited, pooled.to(inherited.device)):
                raise ValueError(f'Unmeasured/inherited input default dominates pool: {name}')
            expected[name] = pooled
            donors[name] = sum(valid_scale(value, positive=True) for value in measured)
        result = original(experts, *args, **kwargs)  # Exactly the stock call, once.
        for name, quantizers in groups.items():
            if any(not valid_scale(q.amax, positive=True)
                   or not torch.equal(q.amax, expected[name].to(q.amax.device)) for q in quantizers):
                raise ValueError(f'Standard input pooling result differs: {name}')
            events.append({'projection': name, 'experts': len(quantizers),
                           'positive_donors': donors[name], 'stock_pool_exact': True,
                           'quantizer_ids': [id(q) for q in quantizers]})
        return result

    hf.sync_moe_expert_amax = observed
    try:
        yield
    finally:
        hf.sync_moe_expert_amax = original


def _validate_linears(selected):
    import torch
    pools = {}
    for name, linear in selected.items():
        weight = linear.weight_quantizer
        if (not weight.is_enabled or not valid_scale(weight.amax, positive=False)
                or not valid_scale(getattr(weight, '_global_amax', None), positive=True)):
            raise ValueError(f'Invalid weight block/global scales: {name}')
        activation = linear.input_quantizer
        if not activation.is_enabled or not valid_scale(activation.amax, positive=True):
            raise ValueError(f'Invalid static input scale: {name}')
        prefix, projection = name.rsplit('.experts.', 1)
        key = prefix, projection.split('.', 1)[1]
        value = activation.amax
        if key in pools and not torch.equal(pools[key], value.to(pools[key].device)):
            raise ValueError(f'Expert input scales are not pooled: {name}')
        pools.setdefault(key, value)
    if not selected:
        raise ValueError('No selected projections')
    return {'selected_projections': len(selected), 'positive_pooled_input_groups': len(pools),
            'weight_block_scales_finite_nonnegative': True, 'weight_global_scales_positive': True,
            'synthetic_fallback_used': False}


def validate_selected_scales(model, recipe):
    """Validate complete final/restored state without cloning weight block tensors."""
    from .adapter import selected_linears
    return _validate_linears(selected_linears(model, recipe))


@contextmanager
def validate_layer_checkpoints(recipe, records):
    """Validate selected layer scales before the unchanged ModelOpt checkpoint save."""
    from unittest.mock import patch
    import torch
    from .inventory import EXPERT
    from modelopt.torch.quantization.utils.layerwise_calib import _CheckpointState

    original = _CheckpointState.save

    def checked_save(checkpoint, layer_idx, model, layers, *args, **kwargs):
        prefix = f'model.layers.{layer_idx}.'
        selected = {}
        if layer_idx not in recipe.get('retain_layers', []):
            for relative, linear in layers[layer_idx].named_modules():
                name = prefix + relative
                if EXPERT.fullmatch(name + '.weight'):
                    if not isinstance(linear, torch.nn.Linear):
                        raise TypeError(f'Unexpected expert type: {name}')
                    selected[name] = linear
        if selected:
            records.append({'layer': layer_idx, **_validate_linears(selected)})
        return original(checkpoint, layer_idx, model, layers, *args, **kwargs)

    with patch.object(_CheckpointState, 'save', checked_save):
        yield


def calibrate_pooled(recipe, source, manifest, output, coverage_from, *, frontend_per_pass=True, validation=None):
    """Fresh-only layerwise MSE orchestration; policy is separate from raw route coverage."""
    import torch
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from .adapter import (CalibrationBatches, ExpertCoverage, calibration_memory_context,
                          load_model, quant_config, register_ming, synchronize_export_scales, to_device)
    from .calibration_state import save_quantizer_buffers
    from .manifest import load_manifest
    from .pipeline import _seed

    from .flash_validation import validation_bundle
    if validation is None:
        raise ValueError('Fresh Flash validation bundle required')
    if frontend_per_pass is not True:
        raise ValueError('Validated Flash calibration requires frontend per-pass materialization')
    proof = validation_bundle(recipe, manifest, validation)
    code_hash = file_hash(Path(__file__))
    if recipe.get('expert_coverage_policy') != POLICY or recipe['mode'] != 'w4a4' or recipe['algorithm'] != 'mse':
        raise ValueError('Only explicit static W4A4 MSE pooled-input policy supported')
    if output.exists():
        raise ValueError('Fresh calibration directory required; this entrypoint does not resume')
    rows, checksum = load_manifest(manifest, 'calibration')
    if len(rows) < 1024:
        raise ValueError('At least1024 training examples required')
    identity = {'provenance': provenance(recipe, source), 'manifest_sha256': checksum,
                'layerwise': True, 'smoke': False,
                'batch_limits': {'size': recipe.get('calibration_batch_size', 1),
                                 'tokens': recipe.get('calibration_batch_tokens', 4096)}}
    disk_budget(output, recipe['disk_floor_gib'])
    output.mkdir(parents=True)
    write_json(output / 'run.json', identity)
    _seed(recipe['seed'])
    model, processor = load_model(source, recipe, output / 'offload')
    register_ming()
    coverage = ExpertCoverage(model, recipe)
    try:
        raw = reuse_pooled_coverage(coverage_from, identity, coverage.counts, recipe['min_expert_tokens'])
    finally:
        coverage.close()
    raw['coverage_passed'] = not raw['under_observed']
    write_json(output / 'coverage.json', raw)
    batches = CalibrationBatches(processor, rows, recipe)
    write_json(output / 'batch-plan.json', batches.metadata)
    hidden = model.config.text_config.hidden_size
    estimate = sum(2 * (n * hidden * 2 + n * 256 * 2 + n * n * 2) for n in batches.lengths)
    if estimate > recipe['scratch_memory_gib'] * 2**30:
        raise ValueError('Replay cache exceeds scratch budget')

    def forward_loop(current):
        from modelopt.torch.quantization.utils.core_utils import persistent_materialization
        with torch.inference_mode(), ExitStack() as stack:
            if frontend_per_pass:
                stack.enter_context(persistent_materialization(current.model.audio, writeback=False))
                stack.enter_context(persistent_materialization(current.linear_proj_audio, writeback=False))
            for inputs in batches:
                current(**to_device(inputs), use_cache=False, logits_to_keep=1)

    config = quant_config(model, recipe, output / 'layers')
    write_json(output / 'quantization.json', config)
    events = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    checkpoints = []
    with observe_default_pooling(events), calibration_memory_context(), validate_layer_checkpoints(recipe, checkpoints):
        model = mtq.quantize(model, config, forward_loop=forward_loop)
    synchronize_export_scales(model)
    scales = validate_selected_scales(model, recipe)
    from .adapter import selected_linears
    observed = {key for event in events for key in event['quantizer_ids']}
    if any(id(linear.input_quantizer) not in observed for linear in selected_linears(model, recipe).values()):
        raise ValueError('Standard ModelOpt pooling was not observed for every selected input')
    if validation_bundle(recipe, manifest, validation) != proof:
        raise ValueError('Flash validation changed during calibration')
    if file_hash(Path(__file__)) != code_hash:
        raise ValueError('Policy implementation changed during calibration')
    policy = {'policy_code_sha256': code_hash, 'validation': proof,
              'frontend_per_pass': frontend_per_pass, 'policy': POLICY, 'calibration_policy_passed': True,
              'coverage_passed': raw['coverage_passed'], 'raw_under_observed': raw['under_observed'],
              'observed_default_pooling': events, 'validated_layer_checkpoints': checkpoints, **scales}
    # Every policy/scale check precedes serialization. Export must revalidate after restore.
    write_json(output / 'calibration-policy.json', policy)
    torch.save(mto.modelopt_state(model), output / 'modelopt_state.pt')
    save_quantizer_buffers(model, output / 'quantizers.safetensors')
    report = {**identity, 'quantized_projections': scales['selected_projections'],
              'seconds': time.perf_counter() - started, 'coverage_passed': raw['coverage_passed'],
              'calibration_policy_passed': True, 'release_approved': False, 'batching': batches.metadata,
              'policy_code_sha256': code_hash, 'validation': proof,
              'frontend_per_pass': frontend_per_pass,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    write_json(output / 'calibration.json', report)
    return report


def export_pooled(recipe, source, calibration, output):
    """Reuse the standard exporter, validating policy/scales immediately after standard restore."""
    from unittest.mock import patch
    from . import pipeline

    policy = json.loads((calibration / 'calibration-policy.json').read_text())
    if recipe.get('expert_coverage_policy') != POLICY or policy.get('calibration_policy_passed') is not True:
        raise ValueError('Completed explicit pooled calibration policy required')
    if policy.get('policy_code_sha256') != file_hash(Path(__file__)):
        raise ValueError('Policy implementation changed since calibration')
    from .flash_validation import validation_bundle
    proof = policy.get('validation')
    if not proof or validation_bundle(recipe, None, Path(proof['bundle'])) != proof:
        raise ValueError('Flash validation changed since calibration')
    original = pipeline.restore_calibration

    def checked_restore(model, *args, **kwargs):
        run = original(model, *args, **kwargs)
        validate_selected_scales(model, recipe)
        return run

    with patch.object(pipeline, 'restore_calibration', checked_restore):
        result = pipeline.export(recipe, source, calibration, output)
    write_json(output / 'calibration-policy.json', policy)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--coverage-from', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    from .common import read_recipe, snapshot
    recipe = read_recipe(args.recipe)
    if recipe.get('expert_coverage_policy') != POLICY:
        raise ValueError('Explicit policy required')
    if not args.execute:
        print(json.dumps({'state': 'prepared_unrun', 'policy': POLICY, 'raw_coverage_preserved': True,
                          'gpu_launched': False, 'validation_bundle_required': True}, indent=2))
        return
    if args.validation is None:
        parser.error('Execution requires a fresh Flash validation bundle')
    # Exclusive launch ownership; no nested GPU loader/driver is invoked here.
    import fcntl
    import subprocess
    root = Path(__file__).resolve().parents[2]
    with (root / '.cache/blackwell-gpu-queue.lock').open('a') as queue, (root / '.cache/blackwell-gpu.lock').open('a') as gpu:
        fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], check=True, capture_output=True, text=True)
        if idle.stdout.strip():
            raise RuntimeError('GPU must be idle')
        calibrate_pooled(recipe, snapshot(recipe), args.manifest, args.output, args.coverage_from,
                         frontend_per_pass=True, validation=args.validation)


if __name__ == '__main__':
    main()

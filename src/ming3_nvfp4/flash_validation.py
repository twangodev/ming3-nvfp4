"""Portable, fresh Tiny control for the Flash pooled-calibration policy.

This runs real ModelOpt conversion twice on a separate pinned Tiny source: one
original-shard-backed forced-offload arm and one resident CUDA arm. It never
uses previously generated artifact reports as proof. The result is a validation
bundle consumed by full Flash calibration, not a coverage or quality approval.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import tempfile

from .common import digest, disk_budget, file_hash, provenance, write_json
from .inventory import audit_export, header, require_complete, tensor_hash
from .pooled_calibration import POLICY, observe_default_pooling, validate_selected_scales

SCHEMA = 1
SOURCE_PIN = Path(__file__).resolve().parents[2] / 'recipes/tiny-validation-source.json'


def _verify_validation_source(source):
    """Match source contents to the immutable public Tiny revision, not its directory name."""
    pinned = json.loads(SOURCE_PIN.read_text())
    hashes = {}
    for name, expected in pinned['files'].items():
        path = source / name
        if not path.is_file() or path.stat().st_size != expected['size_bytes']:
            raise ValueError(f'Pinned Tiny validation source missing or wrong size: {name}')
        actual = file_hash(path)
        if 'sha256' in expected:
            matches = actual == expected['sha256']
        else:
            content = path.read_bytes()
            git_hash = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            matches = git_hash == expected['git_blob_sha1']
        if not matches:
            raise ValueError(f'Pinned Tiny validation source content differs: {name}')
        hashes[str(path.resolve())] = actual
    return hashes


def _input_hashes(source_hashes):
    root = Path(__file__).resolve().parents[2]
    paths = [root / 'uv.lock', SOURCE_PIN]
    paths += sorted((root / 'src/ming3_nvfp4').glob('*.py'))
    return {**source_hashes, **{str(path.resolve()): file_hash(path) for path in paths}}


def _finite(tensor, label, records):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        raise ValueError(f'Empty/non-tensor observation: {label}')
    row = {'name': label, 'elements': tensor.numel(),
           'finite': int(torch.isfinite(tensor).sum().item())}
    records.append(row)
    if row['finite'] != row['elements']:
        raise ValueError(f'Nonfinite observation: {label}')


def _tensor_signature(tensor):
    import torch
    if tensor.is_meta:
        raise ValueError('Meta tensor cannot be signed')
    raw = tensor.detach().contiguous().reshape(-1).cpu().view(torch.uint8).numpy()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(memoryview(raw)).hexdigest()}


def _buffer_signatures(model, *, frontend_only=False):
    return {name: _tensor_signature(value) for name, value in
            model.named_buffers(remove_duplicate=False) if 'quantizer' in name and
            (not frontend_only or name.startswith(('model.audio.', 'linear_proj_audio.')))}


def _metadata(model):
    from modelopt.torch.quantization.nn import TensorQuantizer
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and name.startswith(('model.audio.', 'linear_proj_audio.')):
            if module.is_enabled:
                raise ValueError('Retained frontend quantizer is enabled')
            result[name] = copy.deepcopy(module.get_modelopt_state())
    if not result:
        raise ValueError('Converted frontend quantizer inventory missing')
    return result


def _metadata_valid(before, after, observed):
    """Disabled quantizers may initialize only their actually observed dtype."""
    if before.keys() != after.keys():
        return False
    for name in before:
        left, right = before[name], after[name]
        for key in left.keys() | right.keys():
            if key not in left or key not in right:
                return False
            if left[key] != right[key]:
                if (key != '_input_dtype' or left[key] is not None or
                        not observed.get(name) or
                        any(dtype != right[key] for dtype in observed[name])):
                    return False
    return True


def _frontend_snapshot(model):
    from accelerate.utils.offload import OffloadedWeightsLoader
    from modelopt.torch.quantization.plugins.accelerate import _get_offload_hook
    roots = (model.model.audio, model.linear_proj_audio)
    parameters = {f'{i}.{name}': (parameter.is_meta, str(parameter.device), str(parameter.dtype))
                  for i, root in enumerate(roots) for name, parameter in root.named_parameters()}
    hooks, loaders = {}, {}
    for i, root in enumerate(roots):
        for name, module in root.named_modules():
            hook = _get_offload_hook(getattr(module, '_hf_hook', None))
            if hook is None:
                continue
            hooks[f'{i}.{name}'] = (id(hook), hook.offload,
                                   str(hook.execution_device), hook.io_same_device)
            loader = getattr(hook.weights_map, 'dataset', hook.weights_map)
            if isinstance(loader, OffloadedWeightsLoader):
                loaders[id(loader)] = loader
    backing = {key: {'cpu_keys': sorted(loader.state_dict), 'index': loader.index.copy()}
               for key, loader in loaders.items()}
    return {'parameters': parameters, 'hooks': hooks, 'backing': backing}, loaders


def _source_backing(source, loaders):
    index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    disk_keys = []
    for loader in loaders.values():
        for key, entry in loader.index.items():
            if key not in index or Path(entry.get('safetensors_file', '')).resolve() != (source / index[key]).resolve():
                raise ValueError(f'Offload backing differs from source shard: {key}')
            if key not in loader.state_dict:
                disk_keys.append(key)
    frontend = [key for key in disk_keys if key.startswith(('model.audio.', 'linear_proj_audio.'))]
    if not frontend:
        raise ValueError('Forced arm did not disk offload original frontend')
    return {'disk_keys': len(disk_keys), 'frontend_disk_keys': len(frontend)}


def _resident(model):
    parameters = list(model.named_parameters(remove_duplicate=False))
    if not parameters or any(value.device.type != 'cuda' or value.device.index != 0
                             for _, value in parameters):
        raise ValueError('Resident control has parameters outside CUDA0')
    return {'parameter_count': len(parameters), 'all_parameters_cuda0': True}


def _physical_bytes(path):
    return sum(item.stat().st_blocks * 512 for item in path.rglob('*') if item.is_file())


def _export_signatures(path):
    result = {}
    for shard in path.glob('*.safetensors'):
        for name, spec in header(shard).items():
            if name in result:
                raise ValueError(f'Duplicate export tensor: {name}')
            result[name] = {'shape': spec['shape'], 'dtype': spec['dtype'],
                            'sha256': tensor_hash(shard, spec)}
    if not result:
        raise ValueError('No exported tensors')
    return result


def _comparable_config(config):
    normalized = copy.deepcopy(config)
    algorithms = normalized['algorithm']
    for item in algorithms if isinstance(algorithms, list) else [algorithms]:
        item['layerwise']['checkpoint_dir'] = '<arm-local-checkpoints>'
    return normalized


def _validation_recipe(recipe, source):
    """Use one explicit small offload control; target identity stays separate."""
    if (recipe.get('variant') != 'flash' or recipe.get('mode') != 'w4a4' or
            recipe.get('algorithm') != 'mse' or
            recipe.get('expert_coverage_policy') != POLICY):
        raise ValueError('Validation requires explicit Flash W4A4 MSE pooled recipe')
    config = json.loads((source / 'config.json').read_text())
    text = config['text_config']
    first, count = text['first_k_dense_replace'], text['num_hidden_layers']
    if count <= first + 1:
        raise ValueError('Validation source needs at least two MoE layers')
    result = {**recipe, 'variant': 'tiny', 'model_id': str(config.get('_name_or_path', source.name)),
              'revision': str(source.resolve()), 'gpu_memory_gib': 4,
              'cpu_memory_gib': 4, 'scratch_memory_gib': 2,
              'calibration_batch_size': min(recipe.get('calibration_batch_size', 1), 4),
              'calibration_batch_tokens': min(recipe.get('calibration_batch_tokens', 4096), 1024),
              'calibration_checkpoint_every': 1,
              'offload_source_shards_only': True,
              'retain_layers': list(range(first, count))}
    return result, []


def _placement(device_map, name):
    matches = [key for key in device_map if not key or name == key or name.startswith(key + '.')]
    return device_map[max(matches, key=len)] if matches else None


def _choose_scope(model):
    config = model.config.text_config
    candidates = {'gpu': [], 'disk': []}
    for index in range(config.first_k_dense_replace, config.num_hidden_layers):
        placement = _placement(model.hf_device_map,
                               f'model.layers.{index}.mlp.experts.0.gate_proj')
        key = 'disk' if placement == 'disk' else 'gpu' if str(placement) in ('0', 'cuda:0') else None
        if key:
            candidates[key].append(index)
    if not candidates['gpu'] or not candidates['disk']:
        raise ValueError('Tiny validation requires GPU and disk-backed MoE layers')
    return [candidates['gpu'][0], candidates['disk'][0]]


def _prime_audio_registry():
    from whisper.model import MultiHeadAttention
    from modelopt.torch.quantization.nn import QuantModuleRegistry
    from modelopt.torch.quantization.plugins.attention import register_attention_for_kv_quant
    if QuantModuleRegistry.get(MultiHeadAttention) is None:
        if not register_attention_for_kv_quant(MultiHeadAttention):
            raise ValueError('Whisper attention registry setup failed')
    if QuantModuleRegistry.get(MultiHeadAttention) is None:
        raise ValueError('Whisper attention registry setup missing')


def _quantizer_inventory(model):
    from modelopt.torch.quantization.conversion import quantizer_state
    state = quantizer_state(model)
    audio = [name for name in state if name.startswith('model.audio.')]
    if not audio or any(module.is_enabled for name, module in model.named_modules() if name in audio):
        raise ValueError('Retained audio quantizers missing or enabled')
    return sorted(state)


def _shared_scales(model, layers):
    pairs = 0
    for index in layers:
        for expert in model.model.layers[index].mlp.experts:
            gate = expert.gate_proj.weight_quantizer
            up = expert.up_proj.weight_quantizer
            if gate.global_amax is not up.global_amax or gate.amax is up.amax:
                raise ValueError('Gate/up scale aliases changed')
            pairs += 1
    return pairs


def _alias_groups(model):
    groups = {}
    for name, value in model.named_buffers(remove_duplicate=False):
        if 'quantizer' in name:
            groups.setdefault(id(value), []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def _restore_equal_globals(model, scope):
    import torch
    from modelopt.torch.quantization.utils.shared_input import (
        SharedWeightGlobalAmaxState, find_shared_input_groups)
    before = _buffer_signatures(model)
    objects = {name: value for name, value in model.named_buffers(remove_duplicate=False)
               if 'quantizer' in name and not name.endswith('._global_amax')}
    patterns = SharedWeightGlobalAmaxState.default_patterns
    layers = []
    for index in scope:
        layer = model.model.layers[index]
        found = False
        for _, members in find_shared_input_groups(layer, patterns=patterns):
            quantizers = [member.weight_quantizer for member in members]
            if not all(getattr(q, 'is_nvfp4_static', False) for q in quantizers):
                continue
            values = [q.global_amax for q in quantizers]
            if any(value is None or value.is_meta for value in values):
                raise ValueError('Missing restored global scale')
            first = values[0]
            if any(value.shape != first.shape or value.dtype != first.dtype or
                   value.device != first.device or not torch.equal(value, first)
                   for value in values[1:]):
                raise ValueError('Conflicting restored shared global scales')
            found = True
        if found:
            layers.append(layer)
    for layer in layers:
        SharedWeightGlobalAmaxState.restore(layer, patterns=patterns)
    if _buffer_signatures(model) != before:
        raise ValueError('Global scale relink changed numerical buffers')
    current = dict(model.named_buffers(remove_duplicate=False))
    if any(current[name] is not value for name, value in objects.items()):
        raise ValueError('Global scale relink changed independent buffer identity')


def _run_arm(name, source, recipe, scope, batches, output, *, forced):
    import torch
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from modelopt.torch.quantization.utils.core_utils import persistent_materialization
    from contextlib import ExitStack
    from .adapter import (calibration_memory_context, export_scale_sync_context,
                          load_model, quant_config, selected_linears,
                          synchronize_export_scales, to_device)
    from .calibration_state import save_quantizer_buffers, restore_quantizer_file
    from .pooled_calibration import validate_layer_checkpoints

    directory = output / name
    directory.mkdir()
    arm_recipe = recipe if forced else {**recipe, 'gpu_memory_gib': 80, 'cpu_memory_gib': 80}
    model, _ = load_model(source, arm_recipe, directory / 'offload')
    records = []
    if forced:
        scope[:] = _choose_scope(model)
        first = model.config.text_config.first_k_dense_replace
        count = model.config.text_config.num_hidden_layers
        recipe['retain_layers'] = [index for index in range(first, count) if index not in scope]
        arm_recipe = recipe
        before, loaders = _frontend_snapshot(model)
        backing = _source_backing(source, loaders)
        if _physical_bytes(directory / 'offload'):
            raise ValueError('Original shard offload created duplicate payload')
    else:
        before, backing = None, _resident(model)
    bf16 = []
    with torch.inference_mode(), ExitStack() as stack:
        if forced:
            stack.enter_context(persistent_materialization(model.model.audio, writeback=False))
            stack.enter_context(persistent_materialization(model.linear_proj_audio, writeback=False))
        for index, inputs in enumerate(batches):
            value = model(**to_device(inputs), use_cache=False, logits_to_keep=1).logits
            _finite(value, f'{name}.bf16.{index}', records)
            bf16.append(_tensor_signature(value))
    if forced and (_frontend_snapshot(model)[0] != before or _physical_bytes(directory / 'offload')):
        raise ValueError('BF16 frontend backing changed after baseline')
    selected_count = len(selected_linears(model, arm_recipe))
    if selected_count == 0:
        raise ValueError('No selected expert projections')
    config = quant_config(model, arm_recipe, directory / 'layers')
    projected, first_inputs = {}, {}
    events, checkpoints = [], []
    passes = []
    def forward_loop(current):
        phase = len(passes)
        state, _ = _frontend_snapshot(current) if forced else (None, None)
        metadata_before = _metadata(current)
        buffer_before = _buffer_signatures(current, frontend_only=True)
        observed = {key: [] for key in metadata_before}
        handles = []
        for key, module in current.named_modules():
            if key in observed:
                handles.append(module.register_forward_pre_hook(
                    lambda _module, args, key=key: observed[key].append(args[0].dtype)))
        original_audio = current._get_audio_features
        batch_index = [-1]
        def observe_audio(*args, **kwargs):
            result = original_audio(*args, **kwargs)
            _finite(result, f'{name}.projected.{phase}.{batch_index[0]}', records)
            projected[f'{phase}.{batch_index[0]}'] = _tensor_signature(result)
            return result
        current._get_audio_features = observe_audio
        first = current.model.layers[0]
        def observe_first(_module, args, kwargs):
            value = args[0] if args else kwargs['hidden_states']
            _finite(value, f'{name}.decoder_input.{phase}.{batch_index[0]}', records)
            first_inputs[f'{phase}.{batch_index[0]}'] = _tensor_signature(value)
        handles.append(first.register_forward_pre_hook(observe_first, with_kwargs=True))
        try:
            with torch.inference_mode(), ExitStack() as stack:
                if forced:
                    stack.enter_context(persistent_materialization(current.model.audio, writeback=False))
                    stack.enter_context(persistent_materialization(current.linear_proj_audio, writeback=False))
                for index, inputs in enumerate(batches):
                    batch_index[0] = index
                    current(**to_device(inputs), use_cache=False, logits_to_keep=1)
        finally:
            current._get_audio_features = original_audio
            for handle in handles:
                handle.remove()
        metadata_after = _metadata(current)
        if not _metadata_valid(metadata_before, metadata_after, observed):
            raise ValueError('Converted frontend quantizer metadata changed during callback')
        if _buffer_signatures(current, frontend_only=True) != buffer_before:
            raise ValueError('Converted frontend quantizer buffers changed during callback')
        if forced and (_frontend_snapshot(current)[0] != state or _physical_bytes(directory / 'offload')):
            raise ValueError('Frontend materialization did not restore original backing')
        passes.append(phase)

    with observe_default_pooling(events), calibration_memory_context(), \
            validate_layer_checkpoints(arm_recipe, checkpoints):
        model = mtq.quantize(model, config, forward_loop=forward_loop)
    synchronize_export_scales(model)
    scales = validate_selected_scales(model, arm_recipe)
    observed_ids = {identifier for event in events for identifier in event['quantizer_ids']}
    if any(id(linear.input_quantizer) not in observed_ids for linear in
           selected_linears(model, arm_recipe).values()):
        raise ValueError('Stock pooling not observed for every selected input')
    if not passes or not projected or not first_inputs or not checkpoints:
        raise ValueError('Calibration frontend or checkpoint observations missing')
    if forced and (_frontend_snapshot(model)[0] != before or _physical_bytes(directory / 'offload')):
        raise ValueError('Frontend backing changed after quantization')
    aliases = _shared_scales(model, scope)
    alias_groups = _alias_groups(model)
    inventory = _quantizer_inventory(model)
    buffers = _buffer_signatures(model)
    qdq = []
    with torch.inference_mode():
        for index, inputs in enumerate(batches):
            value = model(**to_device(inputs), use_cache=False, logits_to_keep=1).logits
            _finite(value, f'{name}.qdq.{index}', records)
            qdq.append(_tensor_signature(value))
    if forced:
        torch.save(mto.modelopt_state(model), directory / 'modelopt_state.pt')
        save_quantizer_buffers(model, directory / 'quantizers.safetensors')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model, _ = load_model(source, arm_recipe, directory / 'offload')
        mto.restore_from_modelopt_state(model, torch.load(directory / 'modelopt_state.pt',
                                                         map_location='cpu', weights_only=False))
        restore_quantizer_file(model, directory / 'quantizers.safetensors')
        synchronize_export_scales(model)
        if _buffer_signatures(model) != buffers or _quantizer_inventory(model) != inventory:
            raise ValueError('Restored quantizer state differs')
        _restore_equal_globals(model, scope)
        if _alias_groups(model) != alias_groups:
            raise ValueError('Restored quantizer alias graph differs')
        if _shared_scales(model, scope) != aliases:
            raise ValueError('Restored shared scale aliases differ')
        with torch.inference_mode():
            for index, inputs in enumerate(batches):
                value = model(**to_device(inputs), use_cache=False, logits_to_keep=1).logits
                _finite(value, f'{name}.restored_qdq.{index}', records)
                if _tensor_signature(value) != qdq[index]:
                    raise ValueError('Restored QDQ output differs')
    with tempfile.TemporaryDirectory(prefix='export-', dir=directory) as scratch:
        destination = Path(scratch) / 'checkpoint'
        with export_scale_sync_context():
            export_hf_checkpoint(model, export_dir=destination, max_shard_size='2GB')
        audit = audit_export(source, destination, arm_recipe['retain_layers'])
        if audit['packed_projections'] != selected_count:
            raise ValueError('Packed projection count differs')
        exported = _export_signatures(destination)
    report = {'arm': name, 'forced': forced, 'backing': backing,
              'selected_projections': selected_count, 'stock_pool_events': events,
              'checkpoints': checkpoints, 'scales': scales, 'passes': passes,
              'projected': projected, 'first_inputs': first_inputs,
              'quantizer_inventory': inventory, 'quantizer_buffers': buffers,
              'shared_scale_pairs': aliases, 'alias_groups': alias_groups,
              'bf16': bf16, 'qdq': qdq, 'export': exported,
              'export_audit': audit, 'finite_counts': records,
              'config': config}
    write_json(directory / 'result.json', report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _router_replay(source, directory, layer_index, batch_count):
    """Replay both saved layer prefixes through source decoder, stopping at router."""
    import importlib
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    cls = get_class_from_dynamic_module(config.auto_map['AutoModelForCausalLM'],
                                        str(source), local_files_only=True)
    module = importlib.import_module(cls.__module__.rsplit('.', 1)[0] + '.modeling_bailing_moe_v3')
    config.text_config._attn_implementation = 'eager'
    with torch.device('meta'):
        layer = module.BailingMoeV3DecoderLayer(config.text_config, layer_idx=layer_index)
    gate = layer.mlp.gate
    records, observed = [], []
    class StopBeforeExperts(Exception):
        pass
    class RouterOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = gate
        def forward(self, hidden):
            _finite(hidden, 'router_input', records)
            indices, weights, logits = self.gate(hidden)
            _finite(weights, 'router_weights', records)
            _finite(logits, 'router_logits', records)
            observed.append(indices.detach().cpu())
            raise StopBeforeExperts
    layer.mlp = RouterOnly()
    index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    state = {}
    for name in layer.state_dict():
        key = f'model.layers.{layer_index}.{name}'
        if key not in index or '.experts.' in key:
            raise ValueError(f'Router replay key missing or expert key requested: {key}')
        with safe_open(source / index[key], framework='pt', device='cpu') as handle:
            value = handle.get_tensor(key)
        dtype = torch.float32 if name.endswith(('A_log', 'dt_bias', 'expert_bias')) else torch.bfloat16
        state[name] = value.to(device='cuda:0', dtype=dtype)
    layer.load_state_dict(state, assign=True, strict=True)
    layer.eval()
    if any(value.is_meta for value in list(layer.parameters()) + list(layer.buffers())):
        raise ValueError('Router replay has meta parameters')
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to('cuda:0')
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value
    arms = {}
    for arm in ('forced', 'resident'):
        observed.clear()
        prefix = directory / arm / 'layers' / f'layer_{layer_index-1:04d}' / 'next_inputs.pt'
        if not prefix.is_file():
            raise ValueError(f'Saved prefix absent: {prefix}')
        batches = torch.load(prefix, map_location='cpu', weights_only=False)
        if len(batches) != batch_count:
            raise ValueError('Saved router prefix batch count differs')
        counts = []
        for args, kwargs in batches:
            _finite(args[0], f'{arm}.decoder_input', records)
            before = len(observed)
            try:
                layer(*move(args), **move(kwargs))
            except StopBeforeExperts:
                pass
            else:
                raise ValueError('Router replay executed experts')
            if len(observed) != before + 1:
                raise ValueError('Router replay did not observe exactly one route')
            counts.append(torch.bincount(observed[-1].flatten(),
                                         minlength=config.text_config.num_experts))
        arms[arm] = {'batch_counts': [value.tolist() for value in counts],
                     'counts': torch.stack(counts).sum(0).tolist()}
    if arms['forced'] != arms['resident']:
        raise ValueError('Saved-prefix routing differs between arms')
    zero = [index for index, value in enumerate(arms['forced']['counts']) if value == 0]
    if not zero:
        raise ValueError('No naturally unseen expert in router replay')
    return {'layer': layer_index, 'arms': arms, 'zero_experts': zero,
            'experts_executed': 0, 'finite_counts': records}


def _select_rows(rows):
    """Stable, bounded proof subset with audio and text where both are available."""
    if len(rows) < 4:
        raise ValueError('Validation requires at least four manifest examples')
    audio = [row for row in rows if row['task'] != 'text']
    text = [row for row in rows if row['task'] == 'text']
    if not audio or not text:
        raise ValueError('Validation manifest needs audio and text examples')
    chosen = [audio[0], text[0]]
    chosen += [row for row in rows if row['id'] not in {audio[0]['id'], text[0]['id']}][:6]
    if len(chosen) < 4:
        raise ValueError('Validation subset requires at least four examples')
    return chosen


def validation_bundle(recipe, manifest, output):
    """Verify a freshly produced bundle before full Flash calibration or export."""
    path = Path(output) / 'result.json'
    report = json.loads(path.read_text())
    expected = {'schema': SCHEMA, 'status': 'passed', 'recipe_sha256': digest(recipe),
                'validator_sha256': file_hash(Path(__file__))}
    if manifest is not None:
        expected['manifest_sha256'] = file_hash(Path(manifest))
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f'Flash validation bundle mismatch: {key}')
    checks = report.get('checks', {})
    required = ('stock_pool_exact', 'frontend_exact', 'quantizer_state_exact',
                'qdq_exact', 'export_exact', 'unseen_router_exact_finite')
    if any(checks.get(key) is not True for key in required):
        raise ValueError('Flash validation bundle has missing or failed checks')
    hashes = report.get('proof_sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Flash validation bundle has no proof files')
    root = Path(output).resolve()
    for relative, expected_hash in hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected_hash:
            raise ValueError(f'Flash validation proof changed: {relative}')
    input_hashes = report.get('input_sha256')
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise ValueError('Flash validation has no input hashes')
    for name, expected_hash in input_hashes.items():
        if file_hash(Path(name)) != expected_hash:
            raise ValueError(f'Flash validation input changed: {name}')
    effective_recipe = report.get('effective_recipe')
    source = report.get('source')
    if (not isinstance(effective_recipe, dict) or not isinstance(source, str) or
            report.get('validation_provenance') != provenance(effective_recipe, Path(source))):
        raise ValueError('Flash validation runtime provenance differs')
    if set(hashes) != {'forced/result.json', 'resident/result.json', 'router.json'}:
        raise ValueError('Flash validation proof inventory differs')
    forced = json.loads((root / 'forced/result.json').read_text())
    resident = json.loads((root / 'resident/result.json').read_text())
    router = json.loads((root / 'router.json').read_text())
    if (forced.get('forced') is not True or resident.get('forced') is not False or
            not forced.get('stock_pool_events') or not resident.get('stock_pool_events') or
            forced.get('projected') != resident.get('projected') or
            not forced.get('projected') or
            forced.get('first_inputs') != resident.get('first_inputs') or
            not forced.get('first_inputs') or
            forced.get('bf16') != resident.get('bf16') or
            forced.get('quantizer_inventory') != resident.get('quantizer_inventory') or
            forced.get('quantizer_buffers') != resident.get('quantizer_buffers') or
            forced.get('qdq') != resident.get('qdq') or
            forced.get('export') != resident.get('export') or
            not forced.get('export') or
            router.get('experts_executed') != 0 or
            not router.get('zero_experts') or
            router.get('arms', {}).get('forced') != router.get('arms', {}).get('resident')):
        raise ValueError('Flash validation proof contents differ')
    routed = router['arms']['forced']
    counts = routed.get('counts', [])
    if (not counts or any(type(value) is not int or value < 0 for value in counts) or
            router['zero_experts'] != [index for index, value in enumerate(counts) if value == 0] or
            not routed.get('batch_counts') or
            any(len(batch) != len(counts) for batch in routed['batch_counts']) or
            [sum(batch[index] for batch in routed['batch_counts']) for index in range(len(counts))] != counts):
        raise ValueError('Flash validation unseen-router counts differ')
    for proof in (forced, resident, router):
        records = proof.get('finite_counts', [])
        if not records or any(type(row.get('elements')) is not int or row['elements'] <= 0 or
                              row.get('finite') != row['elements'] for row in records):
            raise ValueError('Flash validation has nonfinite or missing observations')
    return {'result_sha256': file_hash(root / 'result.json'),
            'validator_sha256': expected['validator_sha256'],
            'recipe_sha256': expected['recipe_sha256'], 'bundle': str(root)}


def validate_flash(recipe, source, manifest, output):
    """Run real offloaded/resident Tiny parity and write a hashed validation bundle.

    Caller owns GPU exclusivity. Output must be fresh; no prior artifact is read
    or removed. The production Flash manifest is sampled for proof only.
    """
    import torch
    from .adapter import CalibrationBatches
    from .adapter import register_ming
    from .manifest import load_manifest
    from .pipeline import _seed
    source, manifest, output = (Path(value).resolve() for value in (source, manifest, output))
    if output.exists():
        raise ValueError('Validation output must be a fresh directory')
    require_complete(source)
    source_hashes = _verify_validation_source(source)
    rows, _ = load_manifest(manifest, 'calibration')
    selected = _select_rows(rows)
    tiny_recipe, scope = _validation_recipe(recipe, source)
    disk_budget(output, recipe['disk_floor_gib'])
    output.mkdir(parents=True)
    report = {'schema': SCHEMA, 'status': 'running', 'recipe_sha256': digest(recipe),
              'validator_sha256': file_hash(Path(__file__)),
              'manifest_sha256': file_hash(manifest), 'source': str(source.resolve()),
              'input_sha256': _input_hashes(source_hashes),
              'selected_ids': [row['id'] for row in selected],
              'selected_sha256': digest(selected), 'scope': scope,
              'release_approved': False}
    write_json(output / 'status.json', report)
    try:
        _seed(recipe['seed'])
        _prime_audio_registry()
        register_ming()
        # The processor is loaded once for deterministic sample preparation.
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        batches = CalibrationBatches(processor, selected, tiny_recipe)
        report['batching'] = batches.metadata
        forced = _run_arm('forced', source, tiny_recipe, scope, batches, output, forced=True)
        report['effective_recipe'] = tiny_recipe
        report['validation_provenance'] = provenance(tiny_recipe, source)
        gc.collect()
        torch.cuda.empty_cache()
        resident = _run_arm('resident', source, tiny_recipe, scope, batches, output, forced=False)
        if _comparable_config(forced['config']) != _comparable_config(resident['config']):
            raise ValueError('Calibration configurations differ outside checkpoint directory')
        if (forced['projected'] != resident['projected'] or
                forced['first_inputs'] != resident['first_inputs']):
            raise ValueError('Original frontend calibration observations differ')
        if forced['bf16'] != resident['bf16']:
            raise ValueError('Resident/offloaded BF16 outputs differ')
        if (forced['quantizer_buffers'] != resident['quantizer_buffers'] or
                forced['quantizer_inventory'] != resident['quantizer_inventory']):
            raise ValueError('Resident/offloaded quantizer state differs')
        if forced['qdq'] != resident['qdq']:
            raise ValueError('Resident/offloaded QDQ outputs differ')
        if forced['export'] != resident['export']:
            raise ValueError('Resident/offloaded exports differ')
        replay = _router_replay(source, output, scope[0], batches.metadata['batch_count'])
        write_json(output / 'router.json', replay)
        checks = {'stock_pool_exact': bool(forced['stock_pool_events'] and resident['stock_pool_events']),
                  'frontend_exact': True, 'quantizer_state_exact': True,
                  'qdq_exact': True, 'export_exact': True,
                  'unseen_router_exact_finite': True}
        proof_files = ('forced/result.json', 'resident/result.json', 'router.json')
        report.update(status='passed', checks=checks,
                      proof_sha256={name: file_hash(output / name) for name in proof_files})
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', report)
    write_json(output / 'result.json', report)
    return validation_bundle(recipe, manifest, output)

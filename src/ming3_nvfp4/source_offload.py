"""Guarded original-shard offloading for the pinned Transformers loader."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import importlib.metadata
import inspect
import json
from pathlib import Path
from unittest.mock import patch

from .common import disk_budget


def _physical_bytes(path):
    return sum(item.stat().st_blocks * 512 for item in path.rglob('*') if item.is_file())


@contextmanager
def original_shard_offload_budget(source: Path, offload_dir: Path, floor_gib: float,
                                  transient_bytes: int = 16 * 2**30):
    """Validate the actual index before budgeting; prohibit converted-weight writes.

    Only supports pinned local sharded safetensors, synchronous mmap loading and
    pure renamings. Other loader modes retain the normal conservative guard.
    """
    import transformers.modeling_utils as modeling
    import transformers.core_model_loading as core
    import transformers.integrations.accelerate as integration
    from transformers.core_model_loading import WeightConverter

    if importlib.metadata.version('transformers') != '5.14.1':
        raise RuntimeError('Original-shard budget requires reviewed Transformers5.14.1')
    source, offload_dir = Path(source), Path(offload_dir)
    original_map = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    allowed = {name: (source / filename).resolve() for name, filename in original_map.items()}
    shards = set(allowed.values())
    if not shards or any(not path.is_file() or modeling._is_on_hf_mount(path) for path in shards):
        raise RuntimeError('Original-shard budget requires existing local mmap-compatible source shards')
    if transient_bytes < 16 * 2**30:
        raise ValueError('Keep at least the reviewed16GiB transient disk reserve')
    before = _physical_bytes(offload_dir)
    audit = {'original_source_index': str((source / 'model.safetensors.index.json').resolve()),
             'index_verified_before_budget': False, 'reserved_transient_bytes': transient_bytes,
             'disk_floor_gib': floor_gib, 'physical_offload_bytes_before': before}
    original_index = modeling.accelerate_disk_offload
    loader = modeling.PreTrainedModel._load_pretrained_model
    loader_code = getattr(loader, "__func__", loader).__code__

    def checked_index(model, folder, checkpoint_files, device_map, metadata, weight_mapping=None):
        frame = inspect.currentframe()
        configuration = None
        try:
            while frame is not None:
                if frame.f_code is loader_code:
                    configuration = frame.f_locals.get("load_config")
                    break
                frame = frame.f_back
        finally:
            del frame
        if (configuration is None or configuration.disable_mmap is not False
                or configuration.hf_quantizer is not None or configuration.device_mesh is not None):
            raise RuntimeError("Source-only budget requires the reviewed explicit mmap, unquantized, non-TP loader")
        if {Path(path).resolve() for path in checkpoint_files} != shards:
            raise RuntimeError('Loader checkpoint files differ from pinned source index')
        if any(isinstance(operation, WeightConverter) for operation in (weight_mapping or [])):
            raise RuntimeError('WeightConverter requires conservative copied-spill budgeting')
        actual = original_index(model, folder, checkpoint_files, device_map, metadata, weight_mapping)
        expected = {name for name, device in integration.expand_device_map(
            device_map, model.state_dict().keys()).items() if device == 'disk'}
        if not expected or set(actual) != expected:
            raise RuntimeError('Incomplete original-source index for disk-assigned model state')
        for name, entry in actual.items():
            key, path = entry.get('weight_name'), entry.get('safetensors_file')
            if key not in allowed or path is None or Path(path).resolve() != allowed[key]:
                raise RuntimeError(f'Invalid original-source backing: {name}')
        # This callback occurs before allocator warmup/tensor materialization.
        # No nominal-spill relaxation occurs before this complete mapping check.
        audit.update(index_verified_before_budget=True, index_entries=len(actual),
                     original_source_files=sorted(str(path) for path in shards))
        disk_budget(offload_dir, floor_gib, transient_bytes)
        return actual

    def reject_write(*args, **kwargs):
        raise RuntimeError('Converted/missing-index weight spill is forbidden in source-shards-only mode')

    with ExitStack() as stack:
        stack.enter_context(patch.object(modeling, 'accelerate_disk_offload', checked_index))
        stack.enter_context(patch.object(core, 'offload_weight', reject_write))
        stack.enter_context(patch.object(integration, 'offload_weight', reject_write))
        yield audit
    if not audit['index_verified_before_budget']:
        raise RuntimeError('Source-only disk load did not validate an actual offload index')
    after = _physical_bytes(offload_dir)
    if after != before:
        raise RuntimeError('Source-only load unexpectedly created physical offload files')
    audit['physical_offload_bytes_after'] = after

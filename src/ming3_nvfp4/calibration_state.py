"""Standard safetensors calibration state with bounded staging memory."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path


def save_quantizer_buffers(model, path: Path, chunk_bytes=32 * 2**20):
    import torch
    from safetensors.torch import save

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    tensors = {name: value for name, value in model.named_buffers(remove_duplicate=False)
               if "quantizer" in name}
    if not tensors:
        raise RuntimeError("Calibration produced no quantizer buffers")
    specs, offset, dtypes = {}, 0, {}
    for name, value in sorted(tensors.items()):
        if value.is_meta:
            raise ValueError(f"Cannot save meta quantizer buffer: {name}")
        if value.dtype not in dtypes:
            probe = save({"v": torch.empty(0, dtype=value.dtype)})
            length = int.from_bytes(probe[:8], "little")
            dtypes[value.dtype] = json.loads(probe[8:8 + length])["v"]["dtype"]
        size = value.numel() * value.element_size()
        specs[name] = {"dtype": dtypes[value.dtype], "shape": list(value.shape),
                       "data_offsets": [offset, offset + size]}
        offset += size
    header = json.dumps(specs, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(len(header).to_bytes(8, "little"))
            stream.write(header)
            for name in specs:
                tensor = tensors[name].detach()
                if not tensor.is_contiguous():
                    raise ValueError(f"Noncontiguous quantizer buffer: {name}")
                flat = tensor.reshape(-1)
                count = max(1, chunk_bytes // tensor.element_size())
                for start in range(0, flat.numel(), count):
                    chunk = flat[start:start + count].to("cpu").view(torch.uint8)
                    stream.write(memoryview(chunk.numpy()))
                    del chunk
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_quantizer_buffers(model, buffers: Mapping, execution_device="cuda:0"):
    import torch

    # Keep names/metadata only: retaining all old tensor references here would
    # prevent their release while replacements are installed one alias group at a time.
    groups, expected = defaultdict(list), set()
    metadata = {}
    for name, value in model.named_buffers(remove_duplicate=False):
        groups[id(value)].append(name)
        metadata[name] = (value.device, value.shape, value.dtype)
        if "quantizer" in name:
            expected.add(name)
    if extra := buffers.keys() - expected:
        raise ValueError(f"Unexpected quantizer buffers: {sorted(extra)}")
    if missing := expected - buffers.keys():
        raise ValueError(f"Missing quantizer buffers: {sorted(missing)}")
    # Validate every source and shared alias before mutating any destination.
    for names in groups.values():
        sources = [name for name in names if name in expected]
        if not sources:
            continue
        reference = buffers[sources[0]]
        for name in sources:
            value = buffers[name]
            _, shape, dtype = metadata[name]
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"Quantizer buffer shape or dtype differs: {name}")
            if not torch.equal(reference, value):
                raise ValueError(f"Conflicting shared quantizer buffer: {name}")
    if metadata:
        del value
    for names in groups.values():
        sources = [name for name in names if name in expected]
        if not sources:
            continue
        name = sources[0]
        device, _, _ = metadata[name]
        value = buffers[name].to(execution_device if device.type == "meta" else device)
        for alias in names:
            parent, _, attr = alias.rpartition(".")
            model.get_submodule(parent)._buffers[attr] = value


def restore_quantizer_file(model, path: Path, execution_device="cuda:0"):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as reader:
        class LazyBuffers(Mapping):
            def __getitem__(self, key):
                return reader.get_tensor(key)

            def __iter__(self):
                return iter(reader.keys())

            def __len__(self):
                return len(reader.keys())

        restore_quantizer_buffers(model, LazyBuffers(), execution_device)

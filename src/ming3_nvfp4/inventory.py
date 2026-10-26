from __future__ import annotations

import json
import math
import re
import struct
import hashlib
from collections import defaultdict
from pathlib import Path

EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def header(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        size = struct.unpack("<Q", raw)[0]
        if size > 100_000_000:
            raise ValueError(f"Invalid safetensors header size: {path}")
        tensors = json.loads(stream.read(size))
    payload = path.stat().st_size - size - 8
    end = 0
    for key, value in tensors.items():
        if key == "__metadata__":
            continue
        start, stop = value["data_offsets"]
        if not 0 <= start <= stop <= payload:
            raise ValueError(f"Truncated or invalid tensor {key}: {path}")
        end = max(end, stop)
    if end != payload:
        raise ValueError(f"Safetensors payload length mismatch: {path}")
    return {k: v for k, v in tensors.items() if k != "__metadata__"}


def inspect_model(path: Path, retain_layers: list[int] | None = None) -> dict:
    config = json.loads((path / "config.json").read_text())
    text = config["text_config"]
    n_layers = text["num_hidden_layers"]
    retain = set(retain_layers or [])
    errors, tensors, weight_map = [], {}, {}
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    if index:
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = sorted(p.name for p in path.glob("*.safetensors"))
        if shards != ["model.safetensors"]:
            errors.append("Missing model.safetensors.index.json for sharded checkpoint")
    for filename in shards:
        shard = path / filename
        if not shard.is_file():
            errors.append(f"Missing shard: {filename}")
            continue
        try:
            contents = header(shard)
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
            continue
        for name, tensor in contents.items():
            if name in tensors:
                errors.append(f"Duplicate tensor: {name}")
            tensors[name] = tensor
            weight_map[name] = filename
    if index and weight_map != index["weight_map"]:
        errors.append("Shard tensor names do not match the index")
    groups = defaultdict(lambda: {"parameters": 0, "bytes": 0, "tensors": 0})
    expected_bytes = 0
    for name, tensor in tensors.items():
        match = EXPERT.fullmatch(name)
        layer = int(match[1]) if match else None
        eligible = match and layer < n_layers and layer not in retain
        if eligible:
            group = "nvfp4_eligible_experts"
            if tensor["shape"][-1] % 16:
                errors.append(f"Expert dimension not divisible by 16: {name}")
        elif name.startswith("model.audio.") or name.startswith("linear_proj_audio."):
            group = "audio"
        elif name.startswith(f"model.layers.{n_layers}."):
            group = "mtp"
        else:
            group = "retained"
        count = math.prod(tensor["shape"])
        size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        groups[group]["parameters"] += count
        groups[group]["bytes"] += size
        groups[group]["tensors"] += 1
        expected_bytes += math.ceil(count / 2) + math.ceil(count / 16) + 4 if eligible else size
    total = sum(g["bytes"] for g in groups.values())
    if index and total != index.get("metadata", {}).get("total_size", total):
        errors.append("Tensor byte count does not match index metadata")
    expected_experts = (n_layers - text["first_k_dense_replace"]) * text["num_experts"] * 3
    actual_experts = sum(bool((m := EXPERT.fullmatch(k)) and int(m[1]) < n_layers)
                         for k in tensors)
    if actual_experts != expected_experts:
        errors.append(f"Expected {expected_experts} base expert tensors; found {actual_experts}")
    auxiliary = sum(f.stat().st_size for f in (path / "talker").rglob("*.safetensors"))
    return {"source": str(path), "complete": not errors, "errors": errors,
            "layers": n_layers, "experts_per_layer": text["num_experts"],
            "groups": dict(groups), "root_bytes": total, "auxiliary_weight_bytes": auxiliary,
            "estimated_nvfp4_root_bytes": expected_bytes,
            "shards_found": len(set(weight_map.values())), "shards_expected": len(shards),
            "weight_map": weight_map}


def require_complete(path: Path, retain_layers=None) -> dict:
    report = inspect_model(path, retain_layers)
    if not report["complete"]:
        raise ValueError("Incomplete model: " + "; ".join(report["errors"]))
    return report


def tensor_hash(path: Path, tensor: dict) -> str:
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        start, stop = tensor["data_offsets"]
        stream.seek(8 + header_size + start)
        remaining = stop - start
        checksum = hashlib.sha256()
        while remaining:
            block = stream.read(min(remaining, 8 * 1024 * 1024))
            if not block:
                raise ValueError(f"Truncated tensor payload: {path}")
            checksum.update(block)
            remaining -= len(block)
    return checksum.hexdigest()


def audit_export(source: Path, output: Path, retain_layers: list[int]) -> dict:
    """Check exact quantization scope and unchanged source bytes, without loading a model."""
    config = json.loads((source / "config.json").read_text())["text_config"]
    original, exported = {}, {}
    for root, mapping in ((source, original), (output, exported)):
        for path in root.glob("*.safetensors"):
            for name, tensor in header(path).items():
                if name in mapping:
                    raise ValueError(f"Duplicate exported tensor: {name}")
                mapping[name] = (path, tensor)
    packed, unchanged = 0, 0
    for name, (source_file, tensor) in original.items():
        if name not in exported:
            raise ValueError(f"Export lost source tensor: {name}")
        target_file, target = exported[name]
        match = EXPERT.fullmatch(name)
        eligible = (match and int(match[1]) < config["num_hidden_layers"]
                    and int(match[1]) not in retain_layers)
        if eligible:
            expected_shape = tensor["shape"][:-1] + [tensor["shape"][-1] // 2]
            if target["dtype"] not in {"U8", "I8"} or target["shape"] != expected_shape:
                raise ValueError(f"Wrong NVFP4 packing: {name}")
            for suffix in ("weight_scale", "weight_scale_2"):
                if name.removesuffix("weight") + suffix not in exported:
                    raise ValueError(f"Missing {suffix} for {name}")
            packed += 1
        else:
            if (tensor["dtype"] != target["dtype"] or tensor["shape"] != target["shape"]
                    or tensor_hash(source_file, tensor) != tensor_hash(target_file, target)):
                raise ValueError(f"Retained tensor changed: {name}")
            unchanged += 1
    return {"packed_projections": packed, "unchanged_source_tensors": unchanged,
            "retained_tensors_byte_identical": True}

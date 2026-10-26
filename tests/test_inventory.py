import json
import struct

import pytest

from ming3_nvfp4.inventory import audit_export, header, inspect_model


def write_shard(path, tensors):
    offset = 0
    values = {}
    for name, shape in tensors.items():
        size = 2
        for dim in shape:
            size *= dim
        values[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [offset, offset + size]}
        offset += size
    data = json.dumps(values).encode()
    path.write_bytes(struct.pack("<Q", len(data)) + data + bytes(offset))
    return offset


def checkpoint(path):
    config = {"text_config": {"num_hidden_layers": 2, "num_experts": 1,
                              "first_k_dense_replace": 1}}
    (path / "config.json").write_text(json.dumps(config))
    tensors = {f"model.layers.{layer}.mlp.experts.0.{proj}.weight": [16, 16]
               for layer in (1, 2) for proj in ("gate_proj", "up_proj", "down_proj")}
    size = write_shard(path / "model-00001-of-00001.safetensors", tensors)
    index = {"metadata": {"total_size": size}, "weight_map": {
        name: "model-00001-of-00001.safetensors" for name in tensors}}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    return index


def test_mtp_is_preserved_in_estimate(tmp_path):
    checkpoint(tmp_path)
    report = inspect_model(tmp_path)
    assert report["complete"]
    assert report["groups"]["nvfp4_eligible_experts"]["tensors"] == 3
    assert report["groups"]["mtp"]["tensors"] == 3
    assert inspect_model(tmp_path, [1])["estimated_nvfp4_root_bytes"] == report["root_bytes"]


def test_missing_shards_and_index_are_not_complete(tmp_path):
    index = checkpoint(tmp_path)
    index["weight_map"]["missing"] = "model-00002-of-00002.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert not inspect_model(tmp_path)["complete"]
    (tmp_path / "model.safetensors.index.json").unlink()
    assert not inspect_model(tmp_path)["complete"]


def test_truncated_tensor_payload_is_rejected(tmp_path):
    path = tmp_path / "model.safetensors"
    write_shard(path, {"weight": [16, 16]})
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        header(path)


def test_export_audit_detects_retained_weight_changes(tmp_path):
    import torch
    from safetensors.torch import save_file

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 2}}))
    expert = "model.layers.1.mlp.experts.0.gate_proj."
    retained = "model.layers.1.attention.weight"
    weights = {expert + "weight": torch.zeros(16, 16, dtype=torch.bfloat16),
               retained: torch.ones(16, 16, dtype=torch.bfloat16)}
    save_file(weights, source / "model.safetensors")
    packed = {expert + "weight": torch.zeros(16, 8, dtype=torch.uint8),
              expert + "weight_scale": torch.ones(16, 1, dtype=torch.float8_e4m3fn),
              expert + "weight_scale_2": torch.ones(1), retained: weights[retained]}
    save_file(packed, output / "model.safetensors")
    assert audit_export(source, output, [])["unchanged_source_tensors"] == 1
    packed[retained][0, 0] = 2
    save_file(packed, output / "model.safetensors")
    with pytest.raises(ValueError, match="Retained tensor changed"):
        audit_export(source, output, [])

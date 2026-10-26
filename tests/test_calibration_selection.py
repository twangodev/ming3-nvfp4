import copy
import json

import pytest

from ming3_nvfp4.common import digest
from ming3_nvfp4.calibration_selection import (
    DEFAULT_SELECTION, _check_row, _source_file, portable_row_hash, prepare_calibration,
)


def text_row():
    return dict(id="gsm8k-train-1", split="calibration", task="text", language="en",
                messages=[dict(role="user", content="Question")], reference="Answer",
                source=dict(dataset="openai/gsm8k", revision="pinned", split="train",
                            sample_id="main/train.parquet/row/1"))


def test_portable_hash_ignores_only_local_audio_path_and_runtime_fingerprint():
    row = text_row()
    row["messages"][0]["content"] = [dict(type="audio", audio="/old/file.wav", sha256="a" * 64),
                                       dict(type="text", text="Question")]
    moved = copy.deepcopy(row)
    moved["messages"][0]["content"][0]["audio"] = "/new/file.wav"
    moved["sample_fingerprint"] = "runtime"
    assert portable_row_hash(row) == portable_row_hash(moved)
    moved["reference"] = "wrong"
    assert portable_row_hash(row) != portable_row_hash(moved)


def test_training_identity_and_prompt_leakage_are_rejected():
    row = text_row()
    entry = dict(sha256=portable_row_hash(row))
    _check_row(row, entry, set())
    for identity in [digest([row["source"]["dataset"], row["source"]["sample_id"]]),
                     digest(row["messages"])]:
        with pytest.raises(ValueError, match="leakage"):
            _check_row(row, entry, {identity})
    row["source"]["split"] = "test"
    with pytest.raises(ValueError, match="Non-training"):
        _check_row(row, entry, set())


def test_offline_source_rejects_corruption_before_download(tmp_path):
    spec = dict(dataset="org/data", file="train.jsonl", revision="pinned", sha256="0" * 64)
    path = tmp_path / "sources/org/data/train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="source checksum"):
        _source_file(spec, tmp_path)


def test_selection_has_exact_order_and_no_embedded_prompts_or_local_paths():
    spec = json.loads(DEFAULT_SELECTION.read_text())
    assert len(spec["rows"]) == 2053
    assert spec["original_manifest_sha256"] == "dcdd5004e411eb9ff24bbeadb69821a337f0c1dfbc7a67b8500cc738c40f5cd6"
    assert digest([e["sha256"] for e in spec["rows"]]) == spec["portable_manifest_sha256"]
    assert len({e["template"]["id"] for e in spec["rows"]}) == 2053
    assert "/home/" not in DEFAULT_SELECTION.read_text()
    for entry in spec["rows"]:
        assert not {"messages", "reference", "choices", "answer_text"} & entry["template"].keys()


def test_output_is_preserved_and_selection_tampering_fails_before_sources(tmp_path):
    output = tmp_path / "calibration.jsonl"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        prepare_calibration(output, tmp_path / "assets")
    assert output.read_text() == "existing"
    output.unlink()
    spec = json.loads(DEFAULT_SELECTION.read_text())
    spec["rows"].reverse()
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="order checksum"):
        prepare_calibration(output, tmp_path / "assets", selection)
    assert not output.exists()


def test_audio_identity_and_content_changes_are_rejected():
    row = text_row()
    row["messages"][0]["content"] = [dict(type="audio", audio="/local.wav", sha256="a" * 64)]
    entry = dict(sha256=portable_row_hash(row))
    with pytest.raises(ValueError, match="leakage"):
        _check_row(row, entry, {"a" * 64})
    row["reference"] = "changed teacher-forced answer"
    with pytest.raises(ValueError, match="row checksum"):
        _check_row(row, entry, set())

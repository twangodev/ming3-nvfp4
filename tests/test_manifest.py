import json

import pytest

from ming3_nvfp4.manifest import load_manifest, validate_disjoint


def sample(split="calibration", identifier="one"):
    return {"id": identifier, "split": split, "task": "text", "language": "en",
            "messages": [{"role": "user", "content": "What is two plus two?"}],
            "reference": "4", "source": {"dataset": "fixture", "revision": "1", "sample_id": "1"}}


def test_duplicates_and_wrong_split_rejected(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(sample()) + "\n" + json.dumps(sample()))
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(path)
    path.write_text(json.dumps(sample("acceptance")))
    with pytest.raises(ValueError, match="Expected split"):
        load_manifest(path, "calibration")


def test_same_example_cannot_cross_splits(tmp_path):
    paths = [tmp_path / "calibration.jsonl", tmp_path / "acceptance.jsonl"]
    for path, split in zip(paths, ("calibration", "acceptance")):
        path.write_text(json.dumps(sample(split, split)))
    with pytest.raises(ValueError, match="leakage"):
        validate_disjoint(paths)


def test_audio_content_is_part_of_manifest_identity(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"original")
    row = sample()
    row["messages"][0]["content"] = [{"type": "audio", "audio": "sample.wav"}]
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps(row))
    first = load_manifest(manifest)[1]
    audio.write_bytes(b"changed")
    assert load_manifest(manifest)[1] != first


def test_unicode_line_separator_inside_question_is_not_a_jsonl_boundary(tmp_path):
    row = sample("benchmark")
    row["messages"][0]["content"] = "Question\u2028continued\u2029here"
    path = tmp_path / "unicode.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    rows, _ = load_manifest(path)
    assert len(rows) == 1
    assert rows[0]["messages"] == row["messages"]

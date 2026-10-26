from __future__ import annotations

import json
from pathlib import Path

from .common import digest, file_hash

SPLITS = {"calibration", "development", "acceptance", "smoke", "benchmark"}
TASKS = {"asr", "audio_qa", "text", "speech"}


def load_manifest(path: Path, expected_split: str | None = None) -> tuple[list[dict], str]:
    rows, ids = [], set()
    # JSON strings may legally contain U+2028/U+2029 (present in MMLU-Pro).
    # Only a physical newline separates JSONL records.
    for line_no, line in enumerate(path.read_text().split("\n"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"id", "split", "task", "language", "messages", "reference", "source"}
        if missing := required - row.keys():
            raise ValueError(f"{path}:{line_no}: missing {sorted(missing)}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in SPLITS or row["task"] not in TASKS:
            raise ValueError(f"Invalid split or task in {row['id']}")
        if expected_split and row["split"] != expected_split:
            raise ValueError(f"Expected split {expected_split}, got {row['split']}")
        if not isinstance(row["source"], dict) or not {"dataset", "revision", "sample_id"} <= row["source"].keys():
            raise ValueError(f"Sample {row['id']} requires dataset/revision/sample_id provenance")
        if not isinstance(row["reference"], str) or not row["reference"].strip():
            raise ValueError(f"Sample {row['id']} needs a nonempty reference")
        if not row["messages"] or row["messages"][-1]["role"] != "user":
            raise ValueError("Messages must end in a user prompt; reference is supplied separately")
        for msg in row["messages"]:
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "audio":
                        audio = Path(item["audio"]).expanduser()
                        if not audio.is_absolute():
                            audio = path.parent / audio
                        audio = audio.resolve()
                        if not audio.is_file():
                            raise FileNotFoundError(audio)
                        actual = file_hash(audio)
                        if item.get("sha256", actual) != actual:
                            raise ValueError(f"Audio checksum mismatch: {audio}")
                        item.update(audio=str(audio), sha256=actual)
        row["sample_fingerprint"] = digest(row)
        rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows, digest(rows)


def validate_disjoint(paths: list[Path]) -> dict:
    seen, counts = {}, {}
    for path in paths:
        rows, checksum = load_manifest(path)
        counts[str(path)] = {"count": len(rows), "sha256": checksum}
        for row in rows:
            keys = [digest(row["source"]), digest(row["messages"])]
            for msg in row["messages"]:
                if isinstance(msg["content"], list):
                    keys += [item["sha256"] for item in msg["content"] if item["type"] == "audio"]
            for key in set(keys):
                if key in seen and seen[key] != row["split"]:
                    raise ValueError(f"Split leakage at {row['id']}: {seen[key]} / {row['split']}")
                seen[key] = row["split"]
    return counts

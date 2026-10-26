"""Replay the released, ordered training selection from immutable upstream sources.

No historical manifests, pool builders, model weights, or evaluation harness are needed.
Only selected Clotho audio members are fetched; FLEURS requires its two train shards.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

from .calibration_data import (ZENODO, choose_audio_questions, fetch_clotho_audio,
                               verify_official)
from .common import digest, file_hash, write_json
from .data import ASR_PROMPT
from .manifest import load_manifest

DEFAULT_SELECTION = Path(__file__).resolve().parents[2] / "recipes/calibration-selection.json"


def portable_row_hash(row: dict) -> str:
    """Hash exact source/prompt/reference content independently of local audio paths."""
    row = copy.deepcopy(row)
    row.pop("sample_fingerprint", None)
    for message in row["messages"]:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "audio":
                    item["audio"] = item["sha256"] + ".wav"
    return digest(row)


def _source_file(spec: dict, assets: Path) -> Path:
    from huggingface_hub import hf_hub_download

    # A prepopulated asset directory supports entirely offline replay.
    target = assets / "sources" / spec["dataset"] / spec["file"]
    path = target if target.is_file() else Path(hf_hub_download(
        spec["dataset"], spec["file"], repo_type="dataset", revision=spec["revision"]
    ))
    if file_hash(path) != spec["sha256"]:
        raise ValueError(f"Pinned source checksum mismatch: {spec['dataset']}/{spec['file']}")
    return path


def _clotho_metadata(assets: Path) -> tuple[list[dict], dict]:
    import requests

    for name in ("clotho_aqa_train.csv", "clotho_aqa_metadata.csv"):
        path = assets / name
        if not path.exists():
            response = requests.get(ZENODO + name + "/content", timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        verify_official(path, name)
    with (assets / "clotho_aqa_train.csv").open() as stream:
        questions = choose_audio_questions(list(csv.DictReader(stream)), 256, 42)
    with (assets / "clotho_aqa_metadata.csv").open() as stream:
        metadata = {r["file_name"]: r for r in csv.DictReader(stream)}
    return questions, metadata


def _check_row(row: dict, entry: dict, excluded: set[str]) -> None:
    if row["split"] != "calibration" or "/train/" not in row["source"]["sample_id"] and row["source"].get("split") != "train":
        raise ValueError(f"Non-training calibration row: {row['id']}")
    identities = {digest([row["source"]["dataset"], row["source"]["sample_id"]]),
                  digest(row["messages"])}
    identities.update(item["sha256"] for message in row["messages"]
                      if isinstance(message["content"], list)
                      for item in message["content"] if item["type"] == "audio")
    if identities & excluded:
        raise ValueError(f"Evaluation leakage: {row['id']}")
    if portable_row_hash(row) != entry["sha256"]:
        raise ValueError(f"Selected row checksum mismatch: {row['id']}")


def prepare_calibration(output: Path, assets: Path, selection: Path | None = None) -> dict:
    """Materialize exact final Flash/Tiny calibration rows; refuse to overwrite output.

    Assets may contain sources/<dataset>/<upstream filename> for offline operation.
    Otherwise only pinned source files and selected Clotho ZIP members are downloaded.
    The historical byte hash is provenance: output audio paths necessarily differ.
    """
    import pyarrow.parquet as pq

    output, assets = Path(output), Path(assets)
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads((Path(selection) if selection is not None else DEFAULT_SELECTION).read_text())
    if spec.get("version") != 1:
        raise ValueError("Unsupported calibration selection version")
    entries = spec["rows"]
    if len({e["template"]["id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate selected sample ID")
    if digest([e["sha256"] for e in entries]) != spec["portable_manifest_sha256"]:
        raise ValueError("Selection order checksum mismatch")
    assets.mkdir(parents=True, exist_ok=True)
    audio_dir = assets / "audio"
    audio_dir.mkdir(exist_ok=True)
    sources = {(s["dataset"], s["file"]): _source_file(s, assets) for s in spec["sources"]}
    examples = {}
    for (dataset, filename), path in sources.items():
        if dataset == "google/fleurs":
            wanted = {e["audio_sha256"] for e in entries if e["template"]["source"]["dataset"] == dataset}
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                    for row in batch.to_pylist():
                        import hashlib
                        raw = row["audio"]["bytes"]
                        checksum = hashlib.sha256(raw).hexdigest()
                        if checksum in wanted:
                            target = audio_dir / (checksum + ".wav")
                            if target.exists() and file_hash(target) != checksum:
                                raise ValueError(f"Cached audio checksum mismatch: {target}")
                            if not target.exists():
                                target.write_bytes(raw)
                            examples[checksum] = row["transcription"]
        elif filename.endswith(".parquet"):
            with pq.ParquetFile(path) as parquet:
                examples[dataset] = [r for b in parquet.iter_batches(batch_size=128, use_threads=False) for r in b.to_pylist()]
        else:
            with path.open() as stream:
                examples[dataset] = [json.loads(line) for line in stream if line.strip()]
    questions, metadata = _clotho_metadata(assets)
    names = [questions[e["template"]["question_index"]]["file_name"]
             if e["template"]["source"]["dataset"] == "Clotho-AQA"
             else e["template"]["source"]["underlying_audio"]["filename"]
             for e in entries if e["template"]["source"]["dataset"] in {"Clotho-AQA", "inclusionAI/AudioMCQ"}]
    train_names = set()
    with (assets / "clotho_aqa_train.csv").open() as stream:
        train_names.update(r["file_name"] for r in csv.DictReader(stream))
    if not set(names) <= train_names:
        raise ValueError("Selected Clotho audio is outside official training split")
    fetch_clotho_audio(assets, [dict(file_name=n) for n in dict.fromkeys(names)])
    receipts = json.loads((assets / "clotho-audio-range-receipts.json").read_text())
    rows, excluded = [], set(spec["excluded_evaluation_identities"])
    for entry in entries:
        row = copy.deepcopy(entry["template"])
        dataset = row["source"]["dataset"]
        checksum = entry["audio_sha256"]
        if dataset == "google/fleurs":
            prompt, answer = ASR_PROMPT, examples[checksum]
            audio = audio_dir / (checksum + ".wav")
        elif dataset == "Clotho-AQA":
            q = questions[row.pop("question_index")]
            prompt, answer = q["question"], q["answer"]
            row["source"]["sample_id"] = q["file_name"] + "/" + prompt
            if receipts[q["file_name"]]["sha256"] != checksum:
                raise ValueError("Selected Clotho audio checksum mismatch")
            audio = assets / "clotho-train" / (checksum + ".wav")
        else:
            index = int(row["source"]["sample_id"].rsplit("/", 1)[1])
            ex = examples[dataset][index]
            if dataset == "inclusionAI/AudioMCQ":
                name = row["source"]["underlying_audio"]["filename"]
                if receipts[name]["sha256"] != checksum:
                    raise ValueError("Selected AudioMCQ audio checksum mismatch")
                prompt = ("Choose the most suitable answer from options A, B, C, and D. "
                          "You must respond with only A, B, C, or D.\n\nQuestion: " + ex["question"] + "\n\n" +
                          "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ex["choices"])))
                answer = chr(65 + ex["choices"].index(ex["answer"]))
                row.update(choices=ex["choices"], answer_text=ex["answer"])
                audio = assets / "clotho-train" / (checksum + ".wav")
            elif dataset == "openai/gsm8k":
                prompt, answer = ex["question"], ex["answer"]
            else:
                prompt = ex["instruction"] + ("\n" + ex["input"] if ex["input"] else "")
                answer = ex["output"]
        content = prompt if checksum is None else [dict(type="audio", audio=str(audio.resolve()), sha256=checksum), dict(type="text", text=prompt)]
        row.update(messages=[dict(role="user", content=content)], reference=answer)
        _check_row(row, entry, excluded)
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        load_manifest(temporary, "calibration")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = dict(samples=len(rows), sha256=file_hash(output),
                  portable_manifest_sha256=spec["portable_manifest_sha256"],
                  original_manifest_sha256=spec["original_manifest_sha256"],
                  selection_sha256=file_hash(Path(selection) if selection is not None else DEFAULT_SELECTION))
    write_json(output.with_suffix(".metadata.json"), report)
    return report

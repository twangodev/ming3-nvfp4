"""Prepare full, pinned benchmark splits, reusing checksum-verified local audio."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from upstream import ROOT, mmlu, mmlu_prompt

REGISTRY = {
    "fleurs_en": ("google/fleurs", "168de341b3db6859a9bac1c50a2ef5e3b47647e0", ["en_us/test/0000.parquet"], 647),
    "fleurs_zh": ("google/fleurs", "168de341b3db6859a9bac1c50a2ef5e3b47647e0", ["cmn_hans_cn/test/0000.parquet"], 945),
    "librispeech_clean": ("openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1", ["all/test.clean/0000.parquet"], 2620),
    "librispeech_other": ("openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1", ["all/test.other/0000.parquet"], 2939),
    "mmau": ("gamma-lab-umd/MMAU-test-mini", "ccd9696c0111ea7060827598f310558df0b71b0a", ["test_mini.parquet"], 1000),
    "mmar": ("BoJack/MMAR", "3bd051123480e80d273ae9e8e9f1653f49010ac7", ["MMAR-meta.json", "mmar-audio.tar.gz"], 1000),
    "mmsu": ("ddwang2000/MMSU", "548e2283105825bf908a7db5c09c00dbcf42bd4c", [f"data/train-{i:05d}-of-00003.parquet" for i in range(3)], 5000),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", ["data/test-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"], 12032),
}
ASR_PROMPT = "Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe."


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def read_parquet(paths):
    for path in paths:
        with pq.ParquetFile(path) as source:
            for batch in source.iter_batches(batch_size=16, use_threads=False):
                yield from batch.to_pylist()


def prepare(output, reuse_manifest=None, download=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "full.jsonl").exists():
        raise ValueError("Output already prepared; use a new directory")
    audio_cache = {}
    if reuse_manifest:
        for line in Path(reuse_manifest).open():
            for message in json.loads(line)["messages"]:
                if isinstance(message["content"], list):
                    for item in message["content"]:
                        if item["type"] == "audio":
                            audio_cache[item["sha256"]] = Path(item["audio"])
    verified = set()

    def store_audio(raw, suffix):
        sha = hashlib.sha256(raw).hexdigest()
        path = audio_cache.get(sha, output / "audio" / (sha + suffix))
        if sha not in verified:
            if path.exists():
                if digest(path) != sha:
                    raise ValueError(f"Audio checksum mismatch: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            verified.add(sha)
        return {"path": str(path), "sha256": sha}

    official = mmlu()
    mmau_metadata = {item["id"]: item for item in
                     json.loads((ROOT / "upstream/MMAU/mmau-test-mini.json").read_text())}
    rows = []
    for name, (repo, revision, files, count) in REGISTRY.items():
        paths = [Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                     local_files_only=not download)) for filename in files]
        demos = None
        archive_audio = None
        if name == "mmlu_pro":
            demos = official.preprocess(list(read_parquet(paths[1:])))
            if any(len(examples) != 5 for examples in demos.values()):
                raise ValueError("MMLU-Pro must have five validation examples per subject")
            examples = read_parquet(paths[:1])
        elif name == "mmar":
            raw = paths[0].read_text()
            examples = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(x) for x in raw.splitlines()]
            archive_audio = {}
            needed = {e["audio_path"].removeprefix("./") for e in examples}
            # Stream gzip once: seeking to shuffled examples repeatedly would
            # decompress this multi-GB archive hundreds of times.
            with tarfile.open(paths[1], mode="r|gz") as archive:
                for member in archive:
                    key = member.name.removeprefix("./")
                    if member.isfile() and key in needed:
                        with archive.extractfile(member) as stream:
                            archive_audio[key] = store_audio(stream.read(), Path(key).suffix)
            if set(archive_audio) != needed:
                raise ValueError("MMAR archive is missing required audio")
        else:
            examples = read_parquet(paths)
        start = len(rows)
        try:
            for index, example in enumerate(examples):
                audio = None
                if name != "mmlu_pro":
                    if archive_audio is not None:
                        audio = archive_audio[example["audio_path"].removeprefix("./")]
                    else:
                        raw_audio = example.get("audio", example.get("context"))
                        audio = store_audio(raw_audio["bytes"], Path(raw_audio.get("path") or ".wav").suffix or ".wav")
                example = {k: v for k, v in example.items() if k not in {"audio", "context"}}
                if name == "mmau":
                    example.update(json.loads(example.get("other_attributes", "{}")))
                    example["question"] = example.get("question", example.get("instruction"))
                    example["choices"] = [re.sub(r"^\([A-Z]\)\s*", "", x) for x in example["choices"]]
                    example["answer"] = re.sub(r"^\([A-Z]\)\s*", "", example["answer"])
                    original = mmau_metadata[example["id"]]
                    for field in ("question", "choices", "answer", "task", "difficulty"):
                        cached, canonical = example[field], original[field]
                        cached = [x.strip() for x in cached] if isinstance(cached, list) else cached.strip()
                        canonical = [x.strip() for x in canonical] if isinstance(canonical, list) else canonical.strip()
                        if cached != canonical:
                            raise ValueError(f"MMAU cached data differs from pinned official metadata: {example['id']} {field}")
                    example = original
                source_id = str(example.get("question_id", example.get("id", index)))
                if name.startswith(("fleurs_", "librispeech_")):
                    prompt = ASR_PROMPT
                    reference = example.get("transcription", example.get("text"))
                    max_tokens, ngram = 512, 10
                elif name == "mmlu_pro":
                    # Same N/A preprocessing as the upstream evaluator.
                    example = official.preprocess([example])[example["category"]][0]
                    prompt = mmlu_prompt(official, example, demos)
                    reference = example["answer"]
                    max_tokens, ngram = 4000, 0
                else:
                    choices = ([example.get("choice_" + letter, "") or "" for letter in "abcd"]
                               if name == "mmsu" else example["choices"])
                    lead = ("Choose the most suitable answer from options A, B, C, and D. "
                            "You must respond with only A, B, C, or D." if name == "mmsu" else
                            "Choose the correct answer. Respond with the exact text of the correct option only.")
                    prompt = lead + "\n\nQuestion: " + example["question"] + "\n\n" + "\n".join(
                        f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
                    reference = example["answer_gt"] if name == "mmsu" else example["answer"]
                    max_tokens, ngram = 1024, 0
                rows.append({"id": f"{name}-{index}-{source_id}", "benchmark": name,
                             "dataset": repo, "revision": revision, "source_id": source_id,
                             "example": example, "audio": audio, "prompt": prompt,
                             "reference": reference, "max_tokens": max_tokens, "ngram": ngram})
        finally:
            archive_audio = None
        if len(rows) - start != count:
            raise ValueError(f"Wrong full split count for {name}: {len(rows)-start} != {count}")
        print(f"Prepared {name}: {count}", flush=True)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")
    path = output / "full.jsonl"
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)
    write_json(output / "preparation.json", {"registry": REGISTRY, "count": len(rows),
               "manifest_sha256": digest(path), "reused_manifest": str(reuse_manifest),
               "reused_manifest_sha256": digest(reuse_manifest) if reuse_manifest else None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-audio-manifest", type=Path)
    parser.add_argument("--download", action="store_true", help="Allow missing pinned dataset downloads")
    args = parser.parse_args()
    prepare(args.output, args.reuse_audio_manifest, args.download)

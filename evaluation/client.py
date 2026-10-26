"""Resumable inference against the local Ming vLLM-Omni endpoint."""
import argparse
import base64
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import time

import requests

from prepare import REGISTRY, digest, write_json
from upstream import ROOT


def select_rows(manifest, limit=0):
    counts = Counter()
    rows = []
    with Path(manifest).open() as source:
        for line in source:
            row = json.loads(line)
            name = row["benchmark"]
            if not limit or counts[name] < limit:
                rows.append(row)
                counts[name] += 1
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty manifest or duplicate IDs")
    return rows


def payload(row, model):
    content = [{"type": "text", "text": row["prompt"]}]
    if row["audio"]:
        path = Path(row["audio"]["path"])
        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.insert(0, {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{encoded}"}})
    sampling = {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                "seed": 42, "max_tokens": row["max_tokens"], "stream_interval": 8,
                "extra_args": {"no_repeat_ngram_size": row["ngram"], "no_repeat_ngram_window": 16384}}
    return {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0.0, "max_tokens": row["max_tokens"], "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True, "stream_interval": 8, "stream_options": {"include_usage": True},
            "sampling_params_list": [sampling]}


def request(row, model, endpoint):
    start = time.monotonic()
    body = payload(row, model)
    pieces, finish, usage = [], None, None
    with requests.post(endpoint.rstrip("/") + "/chat/completions", json=body,
                       stream=True, timeout=(15, 1200)) as response:
        if response.status_code != 200:
            raise RuntimeError(f"{row['id']}: HTTP {response.status_code}: {response.text[:2000]}")
        for raw in response.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            raw = raw[6:]
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(f"{row['id']}: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if not isinstance(content, str):
                        raise ValueError(f"Unexpected non-text response: {content}")
                    pieces.append(content)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if finish not in {"stop", "length"}:
        raise RuntimeError(f"{row['id']}: Missing or unexpected finish reason: {finish}")
    return {"id": row["id"], "benchmark": row["benchmark"], "text": "".join(pieces),
            "finish_reason": finish, "usage": usage, "elapsed_seconds": time.monotonic() - start}


def read_journal(path, repair=False):
    records = {}
    if not path.exists():
        return records
    with path.open("rb+" if repair else "rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if not repair:
                    raise ValueError(f"Incomplete final journal record: {path}")
                stream.truncate(offset)
                print("Recovered incomplete final journal record", flush=True)
                break
            row = json.loads(line)
            if row["id"] in records:
                raise ValueError(f"Duplicate journal ID: {row['id']}")
            records[row["id"]] = row
    return records


def provenance():
    upstream = {}
    for path in sorted((ROOT / "upstream").iterdir()):
        if not path.is_dir():
            continue
        status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
        if status.strip():
            raise ValueError(f"Modified upstream evaluator: {path}")
        upstream[path.name] = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    paths = list(ROOT.glob("*.py")) + [ROOT / "uv.lock"]
    return {"upstream": upstream, "implementation": {p.name: digest(p) for p in sorted(paths)}}


def checkpoint_identity(model):
    path = Path(model)
    if not path.is_dir():
        raise ValueError(f"Expected a local checkpoint directory: {model}")
    return {"path": str(path.resolve()),
            "config_sha256": digest(path / "config.json"),
            "index_sha256": digest(path / "model.safetensors.index.json"),
            "weights": {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                        for p in sorted(path.glob("*.safetensors"))}}


def run(manifest, output, model, endpoint, concurrency=64, limit=0):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(manifest, output, model, endpoint, concurrency, limit)


def _run(manifest, output, model, endpoint, concurrency, limit):
    if concurrency < 1 or limit < 0:
        raise ValueError("Concurrency must be positive and limit nonnegative")
    manifest = Path(manifest).resolve()
    actual_counts = Counter(row["benchmark"] for row in select_rows(manifest))
    expected_counts = {name: entry[3] for name, entry in REGISTRY.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Manifest does not contain the full pinned suite: {dict(actual_counts)}")
    rows = select_rows(manifest, limit)
    runtime_path = output / "server.json"
    if not runtime_path.exists():
        runtime_path = output.parent / "server.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    runtime = {key: runtime[key] for key in ("profile_sha256", "serving_lock_sha256", "plugin_files") if key in runtime}
    identity = {"protocol": "ming-standard-v1", "manifest": str(manifest),
                "manifest_sha256": digest(manifest), "model": model,
                "checkpoint": checkpoint_identity(model), "endpoint": endpoint,
                "concurrency": concurrency, "limit_per_benchmark": limit,
                "counts": dict(Counter(r["benchmark"] for r in rows)), "runtime": runtime, **provenance()}
    identity_path = output / "run.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Run identity changed; use a fresh output directory")
    else:
        if (output / "predictions.jsonl").exists():
            raise ValueError("Refusing journal without run identity")
        write_json(identity_path, identity)
    response = requests.get(endpoint.rstrip("/") + "/models", timeout=10)
    response.raise_for_status()
    served = [item["id"] for item in response.json()["data"]]
    if model not in served:
        raise ValueError(f"Wrong served model: {served}; expected {model}")
    verified = set()
    for row in rows:
        audio = row["audio"]
        if audio and audio["path"] not in verified:
            if digest(audio["path"]) != audio["sha256"]:
                raise ValueError(f"Audio changed: {audio['path']}")
            verified.add(audio["path"])
    journal_path = output / "predictions.jsonl"
    records = read_journal(journal_path, repair=True)
    if set(records) - {r["id"] for r in rows}:
        raise ValueError("Journal contains unexpected examples")
    remaining = iter(row for row in rows if row["id"] not in records)
    initial, start, last_report = len(records), time.monotonic(), 0.0
    errors = []

    def report(state):
        elapsed = time.monotonic() - start
        completed_now = len(records) - initial
        rate = completed_now / elapsed if elapsed else 0
        value = {"state": state, "completed": len(records), "total": len(rows),
                 "counts": dict(Counter(r["benchmark"] for r in records.values())),
                 "elapsed_seconds": elapsed, "samples_per_second": rate,
                 "eta_seconds": (len(rows) - len(records)) / rate if rate else None,
                 "generation_limit_count": sum(r["finish_reason"] == "length" for r in records.values()),
                 "errors": errors, "updated_unix": time.time()}
        write_json(output / "progress.json", value)
        print(json.dumps(value), flush=True)

    with journal_path.open("a") as journal, ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}
        exhausted = False
        while pending or not exhausted:
            while not errors and not exhausted and len(pending) < concurrency:
                row = next(remaining, None)
                if row is None:
                    exhausted = True
                    break
                pending[pool.submit(request, row, model, endpoint)] = row["id"]
            if errors:
                exhausted = True
            completed, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in completed:
                sample_id = pending.pop(future)
                try:
                    result = future.result()
                    journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    records[sample_id] = result
                except Exception as error:
                    errors.append({"id": sample_id, "error": repr(error)})
            if time.monotonic() - last_report >= 30:
                report("running" if not errors else "draining_after_error")
                last_report = time.monotonic()
        report("failed" if errors else "complete")
    if errors:
        raise RuntimeError(f"{len(errors)} requests failed; successful predictions saved for resume")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8091/v1")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit-per-benchmark", type=int, default=0, help="Nonzero marks a smoke run")
    args = parser.parse_args()
    run(args.manifest, args.output, args.model, args.endpoint, args.concurrency, args.limit_per_benchmark)

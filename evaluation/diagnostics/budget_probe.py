"""Isolated Flash output-budget experiment on capped audio MC and fixed controls."""
import argparse
from concurrent.futures import as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client import checkpoint_identity, provenance, read_journal, request, request_pool
from prepare import digest, write_json
from runtime import launch, lock_held, stop, wait_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-reference", type=Path)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("Token budget must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = [r for r in map(json.loads, args.manifest.open())]
    baseline = read_journal(args.baseline / "predictions.jsonl")
    entry = next(e for e in json.loads(args.config.read_text()) if e["name"] == "flash-nvfp4")
    selected, groups = [], {}
    for name in ["mmau", "mmar", "mmsu"]:
        subset = [r for r in rows if r["benchmark"] == name]
        capped = [r for r in subset if baseline[r["id"]]["finish_reason"] == "length"]
        controls = [r for r in subset if baseline[r["id"]]["finish_reason"] == "stop"]
        controls.sort(key=lambda r: hashlib.sha256(("budget-control-v1:" + r["id"]).encode()).hexdigest())
        for group, examples in [("previously_capped", capped), ("uncapped_control", controls[:32])]:
            for row in examples:
                selected.append({**row, "max_tokens": args.max_tokens})
                groups[row["id"]] = group
    identity = {"purpose": "Output budget sensitivity only; selected capped cases are not a full benchmark",
                "script_sha256": digest(__file__), "manifest_sha256": digest(args.manifest),
                "baseline_journal_sha256": digest(args.baseline / "predictions.jsonl"),
                "checkpoint": checkpoint_identity(entry["checkpoint"]), "groups": groups,
                "max_tokens": args.max_tokens, "concurrency": 64, **provenance()}
    path = args.output / "experiment.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Experiment identity changed")
    write_json(path, identity)
    (args.output / "runner-source.py").write_bytes(Path(__file__).read_bytes())
    journal = args.output / "predictions.jsonl"
    saved = read_journal(journal, repair=True)
    if not set(saved).issubset(groups):
        raise ValueError("Unexpected saved IDs")
    if len(saved) == len(selected):
        return
    started = time.monotonic()
    while args.wait_for_reference:
        state = json.loads((args.wait_for_reference / "progress.json").read_text())
        active = any(lock_held(args.wait_for_reference / name)
                     for name in (".lock", ".experiment.lock", ".suite.lock"))
        if state["state"] == "complete" and not active:
            break
        if state["state"] == "failed" or time.monotonic() - started > 21600:
            raise RuntimeError("Reference failed or exceeded six hours")
        write_json(args.output / "progress.json", {"state": "waiting_for_reference", "total": len(selected),
                                                   "reference_completed": state.get("completed", 0)})
        print("Waiting for HF reference", flush=True)
        time.sleep(30)
    root = Path(__file__).resolve().parents[2]
    endpoint = "http://127.0.0.1:8091/v1"
    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                       "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    if free < 90000:
        raise RuntimeError(f"GPU occupied: {free} MiB free")

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    process = None
    try:
        process = launch(entry, root / "serving", args.output, 8091)
        wait_ready(process, entry["checkpoint"], endpoint)
        remaining = [r for r in selected if r["id"] not in saved]
        for row in remaining:
            if digest(row["audio"]["path"]) != row["audio"]["sha256"]:
                raise ValueError("Audio checksum mismatch")
        errors = []
        tick = time.monotonic()
        with journal.open("a") as stream, request_pool(64) as pool:
            futures = {pool.submit(request, row, entry["checkpoint"], endpoint): row["id"] for row in remaining}
            for future in as_completed(futures):
                sample_id = futures[future]
                try:
                    value = future.result()
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    saved[sample_id] = value
                except Exception as error:
                    errors.append({"id": sample_id, "error": repr(error)})
                progress = {"state": "running", "completed": len(saved), "total": len(selected),
                            "elapsed_seconds": time.monotonic() - tick, "errors": errors}
                write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        progress["state"] = "failed" if errors else "complete"
        write_json(args.output / "progress.json", progress)
        if errors:
            raise RuntimeError(f"{len(errors)} diagnostic requests failed")
    finally:
        stop(process)


if __name__ == "__main__":
    main()

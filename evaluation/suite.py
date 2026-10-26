"""Run models sequentially inside one GPU-enabled Docker job."""
import argparse
import fcntl
from functools import partial
import json
from pathlib import Path
import shutil
import signal

import requests

from client import run
from prepare import write_json
from runtime import launch, stop, wait_ready
from score import score
from upstream import ROOT


def suite(config, output, manifest, serving, port, limit=0, min_free_gib=100):
    if limit < 0 or min_free_gib < 0:
        raise ValueError("Limit and free-disk floor must be nonnegative")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, serving = manifest.resolve(), serving.resolve()
    entries = json.loads(config.read_text())
    if not entries or len({e["name"] for e in entries}) != len(entries):
        raise ValueError("Model configuration must contain unique entries")
    endpoint = f"http://127.0.0.1:{port}/v1"
    scope = "smoke" if limit else "full"
    with (output / ".suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            requests.get(endpoint + "/models", timeout=3)
        except requests.ConnectionError:
            pass
        else:
            raise RuntimeError("Benchmark port is already occupied")
        for index, entry in enumerate(entries):
            if not entry["name"] or Path(entry["name"]).name != entry["name"] or entry["name"] in {".", ".."}:
                raise ValueError("Model name must be a directory basename")
            if shutil.disk_usage(output).free < min_free_gib * 2**30:
                raise RuntimeError(f"Less than {min_free_gib} GiB free on the output filesystem")
            target = output / entry["name"]
            target.mkdir(parents=True, exist_ok=True)
            progress = {"state": "starting", "scope": scope, "model": entry["name"],
                        "model_number": index + 1, "model_count": len(entries), "endpoint": endpoint,
                        "min_free_gib": min_free_gib}
            write_json(output / "suite-progress.json", progress)
            process = None

            def terminate(signum, frame):
                raise SystemExit(128 + signum)

            previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                process = launch(entry, serving, target, port)
                wait_ready(process, entry["checkpoint"], endpoint)
                invoke = partial(run, manifest, model=entry["checkpoint"], endpoint=endpoint)
                if not limit:
                    invoke(target / "smoke", concurrency=4, limit=2)
                    smoke = score(target / "smoke")
                    journal = target / "smoke/predictions.jsonl"
                    if any(not json.loads(line)["text"].strip() for line in journal.open()):
                        raise RuntimeError("Smoke test produced an empty response")
                    if any(s["generation_limit_count"] for s in smoke["benchmarks"].values()):
                        raise RuntimeError("Smoke test hit a generation cap; inspect before the full run")
                write_json(output / "suite-progress.json", {**progress, "state": "running"})
                invoke(target, concurrency=entry["concurrency"], limit=limit)
                score(target)
            except BaseException as error:
                write_json(output / "suite-progress.json", {**progress, "state": "failed", "error": repr(error)})
                raise
            finally:
                try:
                    stop(process)
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)
        write_json(output / "suite-progress.json", {"state": "complete", "scope": scope,
                   "models": [e["name"] for e in entries], "min_free_gib": min_free_gib})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--serving", type=Path, default=ROOT.parent / "serving")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--limit-per-benchmark", type=int, default=0)
    parser.add_argument("--min-free-gib", type=float, default=100)
    args = parser.parse_args()
    suite(args.config, args.output, args.manifest, args.serving, args.port,
          args.limit_per_benchmark, args.min_free_gib)

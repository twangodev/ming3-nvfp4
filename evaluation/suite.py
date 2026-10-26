"""Run models sequentially on one GPU, with an owned systemd server per model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import requests

from client import run
from prepare import digest, write_json
from score import score
from upstream import ROOT


def stop(unit):
    subprocess.run(["systemctl", "--user", "stop", unit], check=False)


def wait_ready(unit, model, endpoint):
    start = time.monotonic()
    while time.monotonic() - start < 1800:
        state = subprocess.check_output(
            ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"], text=True).strip()
        if state not in {"active", "activating"}:
            raise RuntimeError(f"Server {unit} exited: {state}; see server log")
        try:
            response = requests.get(endpoint + "/models", timeout=5)
            response.raise_for_status()
            served = [x["id"] for x in response.json()["data"]]
            if model not in served:
                raise RuntimeError(f"Unexpected server on benchmark port: {served}")
            return
        except requests.RequestException:
            time.sleep(5)
    raise TimeoutError(f"Server {unit} did not become ready within 30 minutes")


def launch(unit, entry, serving, output, port):
    profile = (ROOT / entry["profile"]).resolve()
    # Activate the serving venv for ninja and other compiler helpers. Match the
    # selected Tiny runtime's deliberate removal of system CUDA from PATH.
    path = [str(serving / ".venv/bin"), *os.environ.get("PATH", "").split(":")]
    if entry["name"].startswith("tiny"):
        path = [part for part in path if part != "/usr/local/cuda/bin"]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MING_KDA_", "MING_BLACKWELL_", "MING_NVFP4_"))}
    env.update(PATH=":".join(dict.fromkeys(path)), VLLM_PLUGINS="ming_nvfp4,vllm_omni_register_models",
               MING_NVFP4_FUSED_INPUT="0", MING_NVFP4_FUSED_OUTPUT="0")
    argv = [str(serving / ".venv/bin/python"), "-m", "vllm.entrypoints.cli.main", "serve",
            entry["checkpoint"], "--omni", "--deploy-config", str(profile),
            "--host", "127.0.0.1", "--port", str(port), "--init-timeout", "1800", "--stage-init-timeout", "1800"]
    # systemd-run receives explicit required variables; never serialize credentials.
    forwarded = [f"--setenv={key}={env[key]}" for key in (
        "PATH", "VLLM_PLUGINS", "MING_NVFP4_FUSED_INPUT", "MING_NVFP4_FUSED_OUTPUT")]
    subprocess.run(["systemd-run", "--user", "--collect", "--unit=" + unit,
                    "--property=WorkingDirectory=" + str(ROOT.parent),
                    "--property=KillMode=control-group", "--property=TimeoutStopSec=60",
                    "--property=StandardOutput=append:" + str(output / "server.log"),
                    "--property=StandardError=inherit", *forwarded, *argv], check=True, env=env)
    write_json(output / "server.json", {"argv": argv, "profile_sha256": digest(profile),
                "unit": unit, "serving_lock_sha256": digest(serving / "uv.lock"),
                "plugin_files": {str(p.relative_to(serving)): digest(p) for p in sorted((serving / "src").rglob("*.py"))},
                "environment": {key: env[key] for key in ("PATH", "VLLM_PLUGINS", "MING_NVFP4_FUSED_INPUT", "MING_NVFP4_FUSED_OUTPUT")}})


def suite(config, output, manifest, serving, port, reuse_server_unit=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, serving = manifest.resolve(), serving.resolve()
    entries = json.loads(config.read_text())
    endpoint = f"http://127.0.0.1:{port}/v1"
    with (output / ".suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not reuse_server_unit:
            try:
                requests.get(endpoint + "/models", timeout=3)
            except requests.ConnectionError:
                pass
            else:
                raise RuntimeError("Benchmark port is already occupied")
        for index, entry in enumerate(entries):
            if shutil.disk_usage(output).free < 100 * 2**30:
                raise RuntimeError("Less than 100 GiB free on the output filesystem")
            target = output / entry["name"]
            target.mkdir(parents=True, exist_ok=True)
            unit = reuse_server_unit if index == 0 and reuse_server_unit else f"ming-eval-{output.name}-{entry['name']}"
            write_json(output / "suite-progress.json", {"state": "starting", "model": entry["name"],
                       "model_number": index + 1, "model_count": len(entries), "unit": unit})

            def terminate(signum, frame):
                subprocess.run(["systemctl", "--user", "stop", "--no-block", unit], check=False)
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, terminate)
            signal.signal(signal.SIGINT, terminate)
            try:
                if not (index == 0 and reuse_server_unit):
                    launch(unit, entry, serving, target, port)
                else:
                    profile = ROOT / entry["profile"]
                    write_json(target / "server.json", {"reused_unit": unit, "profile_sha256": digest(profile),
                                "serving_lock_sha256": digest(serving / "uv.lock"),
                                "plugin_files": {str(p.relative_to(serving)): digest(p) for p in sorted((serving / "src").rglob("*.py"))}})
                wait_ready(unit, entry["checkpoint"], endpoint)
                # Exercise all eight datasets before each full model run.
                run(manifest, target / "smoke", entry["checkpoint"], endpoint, 4, 2)
                smoke = score(target / "smoke")
                smoke_journal = target / "smoke/predictions.jsonl"
                if any(not json.loads(line)["text"].strip() for line in smoke_journal.open()):
                    raise RuntimeError("Smoke test produced an empty response")
                if any(s["generation_limit_count"] for s in smoke["benchmarks"].values()):
                    raise RuntimeError("Smoke test hit a generation cap; inspect before the full run")
                write_json(output / "suite-progress.json", {"state": "running", "model": entry["name"],
                           "model_number": index + 1, "model_count": len(entries), "unit": unit})
                run(manifest, target, entry["checkpoint"], endpoint, entry["concurrency"])
                score(target)
            except BaseException as error:
                write_json(output / "suite-progress.json", {"state": "failed", "model": entry["name"],
                           "error": repr(error), "unit": unit})
                raise
            finally:
                stop(unit)
        write_json(output / "suite-progress.json", {"state": "complete", "models": [e["name"] for e in entries]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--serving", type=Path, default=ROOT.parent / "serving")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--reuse-server-unit")
    args = parser.parse_args()
    suite(args.config, args.output, args.manifest, args.serving, args.port, args.reuse_server_unit)

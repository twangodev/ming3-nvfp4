"""Own inference process groups inside a Docker job, without a host supervisor."""
import fcntl
import json
import os
import signal
import subprocess
import time

import requests

from prepare import digest, write_json
from upstream import ROOT


def lock_held(path):
    """A shared-volume advisory lock works across container PID namespaces."""
    if not path.exists():
        return False
    with path.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def stop(process, timeout=60):
    """Terminate the group we created, including workers after parent exit."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()  # Reap the parent promptly.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def wait_ready(process, model, endpoint, timeout=1800):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"Inference process exited with {process.returncode}; see server.log")
        try:
            response = requests.get(endpoint + "/models", timeout=5)
            response.raise_for_status()
            served = [x["id"] for x in response.json()["data"]]
            if model not in served:
                raise RuntimeError(f"Unexpected model on benchmark port: {served}")
            return
        except requests.RequestException:
            time.sleep(1)
    raise TimeoutError("Inference server did not become ready")


def launch(entry, serving, output, port):
    profile = (ROOT / entry["profile"]).resolve()
    python = serving / ".venv/bin/python"
    path = [str(python.parent), *os.environ.get("PATH", "").split(":")]
    if entry["name"].startswith("tiny"):
        path = [part for part in path if part != "/usr/local/cuda/bin"]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MING_KDA_", "MING_BLACKWELL_", "MING_NVFP4_"))}
    env.update(PATH=":".join(dict.fromkeys(path)), VLLM_PLUGINS="ming_nvfp4,vllm_omni_register_models",
               MING_NVFP4_FUSED_INPUT="0", MING_NVFP4_FUSED_OUTPUT="0")
    argv = [str(python), "-m", "vllm.entrypoints.cli.main", "serve", entry["checkpoint"],
            "--omni", "--deploy-config", str(profile), "--host", "127.0.0.1", "--port", str(port),
            "--init-timeout", "1800", "--stage-init-timeout", "1800"]
    packages = json.loads(subprocess.check_output([str(python), "-c",
        "import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in "
        "['torch','vllm','vllm-omni','triton','nvidia-cuda-runtime','nvidia-cuda-nvrtc']}))"], text=True))
    identity = {"argv": argv, "profile_sha256": digest(profile),
                "serving_lock_sha256": digest(serving / "uv.lock"), "packages": packages,
                "measured_runtime_sha256": digest(serving / "measured-runtime.txt"),
                "plugin_files": {str(p.relative_to(serving)): digest(p)
                                 for p in sorted((serving / "src").rglob("*.py"))},
                "environment": {key: env[key] for key in ("VLLM_PLUGINS", "MING_NVFP4_FUSED_INPUT", "MING_NVFP4_FUSED_OUTPUT")}}
    image_recipe = ROOT.parent / "docker/Dockerfile"
    if image_recipe.exists():
        identity["dockerfile_sha256"] = digest(image_recipe)
    # Write provenance before starting a child so failures cannot leak a worker.
    write_json(output / "server.json", identity)
    with (output / "server.log").open("a") as log:
        return subprocess.Popen(argv, cwd=ROOT.parent, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)

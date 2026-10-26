from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tomllib
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)



def checkpoint_save_every(recipe: dict) -> int:
    """Validate cadence without adding a field to historical recipe identities."""
    cadence = recipe.get("calibration_checkpoint_every", 1)
    if type(cadence) is not int or cadence < 1:
        raise ValueError("calibration_checkpoint_every must be a positive integer")
    return cadence

def read_recipe(path: Path) -> dict:
    recipe = tomllib.loads(path.read_text())
    checkpoint_save_every(recipe)
    batch_size = recipe.get("calibration_batch_size", 1)
    batch_tokens = recipe.get("calibration_batch_tokens", 4096)
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError("calibration_batch_size must be an integer from 1 to 8")
    if type(batch_tokens) is not int or batch_tokens < 1:
        raise ValueError("calibration_batch_tokens must be a positive integer")
    if recipe["mode"] not in {"w4a16", "w4a4"}:
        raise ValueError("mode must be w4a16 or w4a4")
    if recipe["algorithm"] not in {"max", "mse", "local_hessian"}:
        raise ValueError("algorithm must be max, mse, or local_hessian")
    for name in ("max_tokens", "min_expert_tokens", "gpu_memory_gib", "cpu_memory_gib"):
        if recipe[name] <= 0:
            raise ValueError(f"{name} must be positive")
    return recipe


def snapshot(recipe: dict, override: Path | None = None) -> Path:
    if override:
        path = override.expanduser().resolve()
    else:
        root = Path(os.environ.get("HF_HUB_CACHE", os.environ.get(
            "HUGGINGFACE_HUB_CACHE", str(Path(os.environ.get(
                "HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"))))
        path = root / ("models--" + recipe["model_id"].replace("/", "--"))
        path = path / "snapshots" / recipe["revision"]
    if not path.is_dir():
        raise FileNotFoundError(f"Pinned snapshot is not cached: {path}")
    return path


def disk_budget(path: Path, floor_gib: float, additional_bytes: int = 0) -> None:
    parent = path.resolve()
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free - additional_bytes < floor_gib * 2**30:
        raise RuntimeError(f"Insufficient disk space: {free / 2**30:.1f} GiB free; "
                           f"need {additional_bytes / 2**30:.1f} GiB plus {floor_gib} GiB floor")


def provenance(recipe: dict, source: Path) -> dict:
    checkpoint_save_every(recipe)
    packages = {}
    for name in ("torch", "transformers", "nvidia-modelopt", "accelerate", "fla-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    # Version 3 rebuilds Ming's derived RoPE buffers after HF meta loading.
    # Earlier coverage/calibration may have used uninitialized frequencies.
    return {"adapter_version": 3, "recipe": recipe, "source": str(source), "python": platform.python_version(),
            "packages": packages, "uv_lock_sha256": file_hash(lock) if lock.exists() else None}


def host_info() -> dict:
    import psutil

    cpu = platform.processor()
    if not cpu and Path("/proc/cpuinfo").exists():
        cpu = next((line.partition(":")[2].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), "unknown")
    info = {"platform": platform.platform(), "cpu": cpu,
            "os": platform.freedesktop_os_release(),
            "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(False),
            "memory": dict(psutil.virtual_memory()._asdict()),
            "disk": dict(shutil.disk_usage(Path.cwd())._asdict())}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,"
                                 "driver_version,compute_cap", "--format=csv"],
                                capture_output=True, text=True, check=True)
        info["gpu"] = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        info["gpu_error"] = str(exc)
    return info

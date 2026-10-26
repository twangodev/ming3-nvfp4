"""Materialize pinned evaluators with self-contained Git metadata in an image."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
for name, (url, revision) in json.loads(Path(__file__).with_name("upstream.json").read_text()).items():
    path = root / "evaluation/upstream" / name
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--depth=1", url, revision], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError(f"Unexpected evaluator revision: {name}")
    subprocess.run(["git", "config", "--system", "--add", "safe.directory", str(path)], check=True)

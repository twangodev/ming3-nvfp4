"""Read-only benchmark watchdog; persists health, response counters, and alerts."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


class Journal:
    def __init__(self):
        self.offset = 0
        self.counts = defaultdict(Counter)
        self.ids = set()

    def update(self, path):
        if path.stat().st_size < self.offset:
            raise ValueError(f"Prediction journal shrank: {path}")
        with path.open("rb") as stream:
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break  # The writer may still be completing this record.
                row = json.loads(line)
                if row["id"] in self.ids:
                    raise ValueError(f"Duplicate prediction: {row['id']}")
                self.ids.add(row["id"])
                counts = self.counts[row["benchmark"]]
                text = row["text"].strip()
                counts["completed"] += 1
                counts["empty"] += not text
                counts["generation_caps"] += row["finish_reason"] == "length"
                counts["letter_only"] += bool(re.fullmatch(r"[A-Z][.)]?", text))
                self.offset = stream.tell()
        return {name: dict(counts) for name, counts in self.counts.items()}


def watch(root, unit, interval, once=False):
    target = root / "monitoring"
    target.mkdir(parents=True, exist_ok=True)
    journals, last_progress = {}, {}
    previous_alerts = None
    while True:
        now = time.time()
        alerts, models = [], {}
        try:
            suite = read_json(root / "suite-progress.json")
            service = command(["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"])
            if suite.get("state") == "failed":
                alerts.append(f"Suite failed: {suite.get('error')}")
            elif suite.get("state") != "complete" and service not in {"active", "activating"}:
                alerts.append(f"Suite service is {service}")
            free_gib = shutil.disk_usage(root).free / 2**30
            if free_gib < 100:
                alerts.append(f"Disk below launch floor: {free_gib:.1f} GiB free")
            for path in sorted(root.glob("*/predictions.jsonl")):
                name = path.parent.name
                journal = journals.setdefault(name, Journal())
                counts = journal.update(path)
                total = sum(c["completed"] for c in counts.values())
                previous_count, changed = last_progress.get(name, (total, path.stat().st_mtime))
                if total != previous_count:
                    changed = now
                last_progress[name] = (total, changed)
                progress = read_json(path.parent / "progress.json")
                summary = read_json(path.parent / "summary.json")
                models[name] = {"state": progress.get("state"), "completed": total,
                                "total": progress.get("total"), "benchmarks": counts,
                                "seconds_since_progress": now - changed,
                                "scores": summary.get("benchmarks", {})}
                if progress.get("errors"):
                    alerts.append(f"{name}: {len(progress['errors'])} request errors")
                if suite.get("model") == name and suite.get("state") == "running" and now - changed > 600:
                    alerts.append(f"{name}: no new predictions for {int(now-changed)} seconds")
                for benchmark, count in counts.items():
                    completed = count["completed"]
                    if count.get("empty"):
                        alerts.append(f"{name}/{benchmark}: empty responses")
                    if completed >= 100 and count.get("generation_caps", 0) / completed > .05:
                        alerts.append(f"{name}/{benchmark}: more than 5% of responses hit generation limits")
                    if benchmark in {"mmau", "mmar"} and completed >= 100 and count.get("letter_only", 0) / completed > .1:
                        alerts.append(f"{name}/{benchmark}: letter answers require separate interpretation from raw string scores")
                for benchmark, result in summary.get("benchmarks", {}).items():
                    if result.get("excluded_by_upstream_parser", 0):
                        alerts.append(f"{name}/{benchmark}: upstream parser excluded {result['excluded_by_upstream_parser']} answers")
            if suite.get("state") == "running":
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8091/v1/models", timeout=5) as response:
                        served = [item["id"] for item in json.load(response)["data"]]
                    current = root / suite["model"]
                    identity = read_json(current / "run.json") or read_json(current / "smoke/run.json")
                    if identity and identity["model"] not in served:
                        alerts.append("Server model identity does not match the active run")
                except Exception as error:
                    alerts.append(f"Server health check failed: {error}")
            gpu = command(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
            health = {"checked_at": datetime.now(timezone.utc).isoformat(), "interval_seconds": interval,
                      "suite": suite, "service": service, "gpu_utilization_and_memory_mib": gpu,
                      "free_disk_gib": round(free_gib, 2), "models": models, "alerts": alerts}
        except Exception as error:
            alerts = [f"Monitor check failed: {error!r}"]
            health = {"checked_at": datetime.now(timezone.utc).isoformat(), "alerts": alerts}
        temporary = target / "health.json.tmp"
        temporary.write_text(json.dumps(health, indent=2) + "\n")
        temporary.replace(target / "health.json")
        concise = {"checked_at": health["checked_at"], "suite": health.get("suite"),
                   "completed": {name: value["completed"] for name, value in health.get("models", {}).items()},
                   "alerts": alerts}
        print(json.dumps(concise), flush=True)
        if alerts != previous_alerts:
            with (target / "alert-history.jsonl").open("a") as stream:
                stream.write(json.dumps(concise) + "\n")
            previous_alerts = alerts
        if once or health.get("suite", {}).get("state") == "complete":
            return
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suite-unit", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("Interval must be at least 10 seconds")
    watch(args.run_dir.resolve(), args.suite_unit, args.interval, args.once)

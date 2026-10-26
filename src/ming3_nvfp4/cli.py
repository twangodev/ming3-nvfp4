"""Command-line tools for reproducible Ming NVFP4 checkpoint production."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path


@contextmanager
def gpu_job():
    """Serialize this checkout's GPU jobs and reject an already occupied device."""
    import fcntl
    import subprocess

    cache = Path.cwd() / ".cache"
    cache.mkdir(exist_ok=True)
    with (cache / "blackwell-gpu-queue.lock").open("a") as queue, \
            (cache / "blackwell-gpu.lock").open("a") as gpu:
        fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        if active.stdout.strip():
            raise RuntimeError("GPU is occupied; stop competing compute jobs before quantization")
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("recipes/tiny.toml"))
    parser.add_argument("--source", type=Path, help="Complete local source snapshot override")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("host", help="Report local CPU, GPU, memory and disk resources")
    prepare = sub.add_parser("prepare-calibration", help="Reconstruct the released training selection")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--assets", type=Path, required=True)
    prepare.add_argument("--selection", type=Path)
    manifests = sub.add_parser("check-manifests", help="Validate manifests and split disjointness")
    manifests.add_argument("paths", type=Path, nargs="+")
    inspect = sub.add_parser("inspect", help="Inspect source layout and estimate export storage")
    inspect.add_argument("--output", type=Path)
    for name in ("coverage", "calibrate"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--layerwise", action="store_true")
        command.add_argument("--smoke", action="store_true", help="Non-release integration only")
        if name == "calibrate":
            command.add_argument("--coverage-from", type=Path)
    export = sub.add_parser("export", help="Export saved calibration in a fresh process")
    export.add_argument("--calibration", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit-export", help="Verify packing and retained source tensors")
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "host":
        from .common import host_info
        result = host_info()
    elif args.command == "prepare-calibration":
        from .calibration_selection import prepare_calibration
        result = prepare_calibration(args.output, args.assets, args.selection)
    elif args.command == "check-manifests":
        from .manifest import validate_disjoint
        result = validate_disjoint(args.paths)
    else:
        from .common import read_recipe, snapshot, write_json
        recipe = read_recipe(args.recipe)
        source = snapshot(recipe, args.source)
        if args.command == "inspect":
            from .inventory import inspect_model
            result = inspect_model(source, recipe["retain_layers"])
            if args.output:
                write_json(args.output, result)
            result = {k: v for k, v in result.items() if k != "weight_map"}
        elif args.command == "audit-export":
            from .inventory import audit_export
            result = audit_export(source, args.checkpoint, recipe["retain_layers"])
            write_json(args.output, result)
        else:
            from . import pipeline
            with gpu_job():
                if args.command == "coverage":
                    result = pipeline.coverage(recipe, source, args.manifest, args.output,
                                               args.layerwise, args.smoke)
                elif args.command == "calibrate":
                    result = pipeline.calibrate(recipe, source, args.manifest, args.output,
                                                args.layerwise, args.smoke, args.coverage_from)
                else:
                    result = pipeline.export(recipe, source, args.calibration, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command == "inspect" and not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

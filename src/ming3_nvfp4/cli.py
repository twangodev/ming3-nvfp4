"""Command-line tools for reproducible Ming NVFP4 checkpoint production."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("host", help="Report local CPU, GPU, memory and disk resources")
    prepare = sub.add_parser("prepare-calibration", help="Reconstruct the released training selection")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--assets", type=Path, required=True)
    prepare.add_argument("--selection", type=Path)
    manifests = sub.add_parser("check-manifests", help="Validate manifests and split disjointness")
    manifests.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "host":
        from .common import host_info
        result = host_info()
    elif args.command == "prepare-calibration":
        from .calibration_selection import prepare_calibration
        result = prepare_calibration(args.output, args.assets, args.selection)
    else:
        from .manifest import validate_disjoint
        result = validate_disjoint(args.paths)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

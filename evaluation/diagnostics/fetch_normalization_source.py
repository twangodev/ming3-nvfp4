"""Fetch the pinned, small UltraEval normalization sources used by diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import requests

REVISION = "bead726925d43f526bed48a4a6a827595169429d"
REPOSITORY = "OpenBMB/UltraEval-Audio"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}",
                            params={"recursive": 1}, timeout=60)
    response.raise_for_status()
    tree = response.json()
    if tree.get("truncated") or tree["sha"] != REVISION:
        raise ValueError("Unexpected upstream tree")
    hashes = {}
    for item in tree["tree"]:
        name = item["path"]
        wanted = (name.startswith("audio_evals/lib/text_normalization/")
                  or name in {"audio_evals/lib/wer.py", "audio_evals/lib/evaluate_tokenizer.py",
                              "registry/dataset/fleurs.yaml", "LICENSE"})
        if not wanted or item["type"] != "blob" or item.get("size", 0) > 1500000:
            continue
        response = requests.get(f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{name}",
                                timeout=60)
        response.raise_for_status()
        raw = response.content
        git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if git_blob != item["sha"]:
            raise ValueError(f"Git blob mismatch: {name}")
        path = args.output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    (args.output / "tree.json").write_text(json.dumps(tree) + "\n")
    (args.output / "source-provenance.json").write_text(json.dumps({"repository": REPOSITORY,
              "revision": REVISION, "sha256": hashes}, indent=2) + "\n")
    print(f"Verified {len(hashes)} source files")


if __name__ == "__main__":
    main()

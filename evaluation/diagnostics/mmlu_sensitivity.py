"""Bound truncation effects and compare first versus last explicit MMLU answers."""
import argparse
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = {r["id"]: r for r in map(json.loads, args.manifest.open()) if r["benchmark"] == "mmlu_pro"}
    results = {}
    for run in args.run:
        exports = {str(x["question_id"]): x for x in
                   json.loads((run / "scores/mmlu_pro/official-input.json").read_text())}
        counts = dict.fromkeys(["total", "strict_correct", "capped", "capped_correct",
                               "last_explicit_correct", "last_explicit_changed"], 0)
        for prediction in map(json.loads, (run / "predictions.jsonl").open()):
            if prediction["benchmark"] != "mmlu_pro":
                continue
            row = rows[prediction["id"]]
            original = exports[row["source_id"]]["pred"]
            matches = re.findall(r"answer is\s*\(?([A-J])\)?", prediction["text"].replace("**", ""), flags=re.I)
            last = matches[-1] if matches else original
            correct = original == row["reference"]
            capped = prediction["finish_reason"] == "length"
            counts["total"] += 1
            counts["strict_correct"] += correct
            counts["capped"] += capped
            counts["capped_correct"] += capped and correct
            counts["last_explicit_correct"] += last == row["reference"]
            counts["last_explicit_changed"] += last != original
        if counts["total"] != len(rows):
            raise ValueError("MMLU run is incomplete")
        counts["maximum_score_if_all_capped_correct"] = (
            100 * (counts["strict_correct"] + counts["capped"] - counts["capped_correct"]) / counts["total"])
        results[run.name] = counts
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

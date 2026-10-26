"""Summarize the budget experiment without replacing any full-suite answers."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument("--probe-details", type=Path, required=True)
    args = parser.parse_args()
    experiment = read(args.probe / "experiment.json")
    baseline = {r["id"]: r for r in map(json.loads, (args.baseline / "predictions.jsonl").open())}
    probe = {r["id"]: r for r in map(json.loads, (args.probe / "predictions.jsonl").open())}
    before = {r["id"]: r for r in read(args.baseline_details)}
    after = {r["id"]: r for r in read(args.probe_details)}
    if set(probe) != set(experiment["groups"]):
        raise ValueError("The selected budget experiment is incomplete")
    result = {"scope": "Selected capped cases plus fixed uncapped controls; never substituted into full benchmark scores",
              "total": len(probe), "max_tokens": experiment["max_tokens"], "benchmarks": {}}
    for benchmark in ["mmau", "mmar", "mmsu"]:
        result["benchmarks"][benchmark] = {}
        for group in ["previously_capped", "uncapped_control"]:
            ids = [i for i, p in probe.items() if p["benchmark"] == benchmark and experiment["groups"][i] == group]
            result["benchmarks"][benchmark][group] = {
                "count": len(ids), "before_correct": sum(before[i]["correct"] for i in ids),
                "after_correct": sum(after[i]["correct"] for i in ids),
                "after_capped": sum(probe[i]["finish_reason"] == "length" for i in ids),
                "stopped_within_original_budget": sum(probe[i]["finish_reason"] == "stop"
                                                       and probe[i]["usage"]["completion_tokens"] <= 1024 for i in ids),
                "selection_value_matches": sum(
                    (before[i]["choices"][before[i]["selected"]] if before[i]["selected"] is not None else None) ==
                    (after[i]["choices"][after[i]["selected"]] if after[i]["selected"] is not None else None)
                    for i in ids),
                "exact_text_matches": sum(probe[i]["text"] == baseline[i]["text"] for i in ids),
                "old_text_is_prefix": sum(probe[i]["text"].startswith(baseline[i]["text"]) for i in ids),
                "prompt_count_matches": sum(probe[i]["usage"]["prompt_tokens"] == baseline[i]["usage"]["prompt_tokens"] for i in ids),
                "before_tokens": sum(baseline[i]["usage"]["completion_tokens"] for i in ids),
                "after_tokens": sum(probe[i]["usage"]["completion_tokens"] for i in ids),
            }
    (args.probe / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

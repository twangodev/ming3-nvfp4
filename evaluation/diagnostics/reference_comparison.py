"""Compare an HF reference journal with the same original-model serving examples."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score import chinese_normalize, score_asr
from upstream import english_normalizer

from choice_scoring import parse_choice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--serving", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in map(json.loads, args.manifest.open())]
    hf = {r["id"]: r for r in map(json.loads, (args.reference / "predictions.jsonl").open())}
    serving = {r["id"]: r for r in map(json.loads, (args.serving / "predictions.jsonl").open())}
    identity = json.loads((args.reference / "run.json").read_text())
    if set(hf) - set(identity["ids"]):
        raise ValueError("Unexpected reference examples")
    rows = [r for r in rows if r["id"] in hf]
    en = english_normalizer()
    report = {"state": "complete" if len(hf) == len(identity["ids"]) else "partial",
              "completed": len(hf), "planned": len(identity["ids"]),
              "scope": "Same original Tiny checkpoint, prompts and examples; HF serial versus vLLM concurrent",
              "benchmarks": {}}
    for name in sorted({r["benchmark"] for r in rows}):
        subset = [r for r in rows if r["benchmark"] == name]
        entry = {"count": len(subset),
                 "exact_text_matches": sum(hf[r["id"]]["text"] == serving[r["id"]]["text"] for r in subset),
                 "prompt_token_count_matches": sum(hf[r["id"]]["prompt_tokens"] ==
                                                   serving[r["id"]]["usage"]["prompt_tokens"] for r in subset),
                 "hf_caps": sum(hf[r["id"]]["finish_reason"] == "length" for r in subset),
                 "serving_caps": sum(serving[r["id"]]["finish_reason"] == "length" for r in subset)}
        if name.startswith("fleurs_"):
            zh = name == "fleurs_zh"
            norm = chinese_normalize if zh else en
            entry["hf"], _ = score_asr(subset, hf, norm, zh)
            entry["serving"], _ = score_asr(subset, serving, norm, zh)
            entry["normalized_text_matches"] = sum(
                norm(hf[r["id"]]["text"].partition("\t")[-1]) ==
                norm(serving[r["id"]]["text"].partition("\t")[-1]) for r in subset)
        else:
            for label, predictions in [("hf", hf), ("serving", serving)]:
                correct = 0
                counts = Counter()
                for row in subset:
                    e = row["example"]
                    choices = [e.get("choice_" + a, "") or "" for a in "abcd"] if name == "mmsu" else e["choices"]
                    pred = predictions[row["id"]]
                    index, kind = parse_choice(pred["text"], choices)
                    counts[kind] += 1
                    correct += (index is not None and choices[index] == row["reference"]
                                and pred["finish_reason"] != "length")
                entry[label] = {"accuracy_percent": 100 * correct / len(subset), "counts": dict(counts)}
        report["benchmarks"][name] = entry
    (args.reference / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

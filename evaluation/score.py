"""Score a complete run with pinned upstream evaluators and explicit ASR rules."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

import jiwer

from client import provenance, read_journal, select_rows
from prepare import digest, write_json
from upstream import ROOT, english_normalizer, mmlu, module


def chinese_normalize(text):
    # Explicit local strict-CER recipe, preserved from the previous comparison.
    # Keep Latin text and digits. No deletion of parenthetical reference text.
    return "".join(c for c in text.lower()
                   if not c.isspace() and unicodedata.category(c)[0] not in "PS")


def score_asr(rows, predictions, normalizer, chinese=False):
    details = []
    slices = defaultdict(lambda: {"examples": 0, "edits": 0, "reference_units": 0})
    for row in rows:
        hypothesis = predictions[row["id"]]["text"]
        # Ming's documented language<TAB>transcript response convention.
        hypothesis = hypothesis.partition("\t")[2] if "\t" in hypothesis else hypothesis
        reference, hypothesis = normalizer(row["reference"]), normalizer(hypothesis)
        alignment = (jiwer.process_characters if chinese else jiwer.process_words)(reference, hypothesis)
        edits = alignment.substitutions + alignment.deletions + alignment.insertions
        units = alignment.hits + alignment.substitutions + alignment.deletions
        details.append({"id": row["id"], "reference": reference, "hypothesis": hypothesis,
                        "edits": edits, "reference_units": units,
                        "substitutions": alignment.substitutions, "deletions": alignment.deletions,
                        "insertions": alignment.insertions})
        key = "reference_has_latin_or_digits" if re.search(r"[A-Za-z0-9]", row["reference"]) else "other"
        for name in ("all", key):
            slices[name]["examples"] += 1
            slices[name]["edits"] += edits
            slices[name]["reference_units"] += units
    for counts in slices.values():
        counts["error_percent"] = 100 * counts["edits"] / counts["reference_units"] if counts["reference_units"] else None
    return {"metric": "CER" if chinese else "WER", "score_percent": slices["all"]["error_percent"],
            "normalization": "local_strict_unicode_chinese" if chinese else "official_whisper_english",
            "scored_count": len(rows), "slices": dict(slices)}, details


def score(run_dir):
    run_dir = Path(run_dir).resolve()
    identity = json.loads((run_dir / "run.json").read_text())
    manifest = Path(identity["manifest"])
    if digest(manifest) != identity["manifest_sha256"]:
        raise ValueError("Manifest changed since inference")
    rows = select_rows(manifest, identity["limit_per_benchmark"])
    predictions = read_journal(run_dir / "predictions.jsonl")
    if set(predictions) != {row["id"] for row in rows}:
        raise ValueError("Scoring requires exactly one successful prediction for every selected example")
    current = provenance()
    if any(current[key] != identity[key] for key in ("upstream", "implementation")):
        raise ValueError("Evaluation source changed since inference; restore the recorded version before scoring")
    if any(p["finish_reason"] not in {"stop", "length"} for p in predictions.values()):
        raise ValueError("Invalid inference completion")
    results = {"scope": "smoke" if identity["limit_per_benchmark"] else "full",
               "total": len(rows), "benchmarks": {}}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)
    english = english_normalizer()
    for name, subset in grouped.items():
        target = run_dir / "scores" / name
        target.mkdir(parents=True, exist_ok=True)
        exports = [{**row["example"]} for row in subset]
        if name in {"mmau", "mmar"}:
            relative = "MMAU/evaluation.py" if name == "mmau" else "MMAR/code/evaluation.py"
            official = module("official_" + name, relative)
            key = "model_output" if name == "mmau" else "answer_prediction"
            for row, export in zip(subset, exports):
                export[key] = predictions[row["id"]]["text"]
            correct = sum(official.string_match(e["answer"], e[key], e["choices"]) for e in exports)
            result = {"metric": "accuracy", "score_percent": 100 * correct / len(subset),
                      "correct": correct, "scored_count": len(subset)}
            input_path = target / "official-input.json"
            write_json(input_path, exports)
            with (target / "official.log").open("w") as log:
                subprocess.run([sys.executable, str(ROOT / "upstream" / relative), "--input", str(input_path)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        elif name == "mmsu":
            official = module("official_mmsu", "MMSU/mmsu_evaluation.py")
            for row, export in zip(subset, exports):
                export["response"] = predictions[row["id"]]["text"]
            input_path = target / "official-input.jsonl"
            with input_path.open("w") as stream:
                for export in exports:
                    stream.write(json.dumps(export, ensure_ascii=False) + "\n")
            with (target / "parser.log").open("w") as log, redirect_stdout(log):
                _, categories, accuracy, counted = official.calculate_accuracy_per_task_and_category(exports)
            correct = sum(c["total_correct"] for c in categories.values())
            result = {"metric": "accuracy", "score_percent": 100 * accuracy,
                      "correct": correct, "scored_count": counted, "excluded_by_upstream_parser": len(subset) - counted,
                      "all_examples_accuracy_percent": 100 * correct / len(subset), "categories": dict(categories)}
            with (target / "official.log").open("w") as log:
                subprocess.run([sys.executable, str(ROOT / "upstream/MMSU/mmsu_evaluation.py"), str(input_path)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        elif name == "mmlu_pro":
            official = mmlu()
            with (target / "parser.log").open("w") as log, redirect_stdout(log):
                for row, export in zip(subset, exports):
                    # This is the exact postprocessing in upstream single_request.
                    export["model_outputs"] = predictions[row["id"]]["text"].replace("**", "")
                    export["pred"] = official.extract_answer(export["model_outputs"])
            input_path = target / "official-input.json"
            write_json(input_path, exports)
            # Upstream update_result guesses on unparseable answers. Evaluate once
            # in manifest order with its published seed; also show strict accuracy.
            _, categories = official.update_result(str(input_path))
            correct = sum(c["corr"] for c in categories.values())
            strict = sum(e["pred"] == e["answer"] for e in exports)
            result = {"metric": "accuracy", "score_percent": 100 * correct / len(subset),
                      "correct": correct, "scored_count": len(subset), "categories": categories,
                      "unparsed_count": sum(e["pred"] is None for e in exports),
                      "unparsed_as_incorrect_accuracy_percent": 100 * strict / len(subset),
                      "upstream_random_fallback_seed": 12345}
        else:
            chinese = name == "fleurs_zh"
            result, details = score_asr(subset, predictions, chinese_normalize if chinese else english, chinese)
            write_json(target / "alignments.json", details)
        result["count"] = len(subset)
        result["generation_limit_count"] = sum(predictions[r["id"]]["finish_reason"] == "length" for r in subset)
        results["benchmarks"][name] = result
    write_json(run_dir / "summary.json", results)
    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    score(parser.parse_args().run_dir)

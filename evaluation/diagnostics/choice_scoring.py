"""Answer-format diagnostics with fixed denominators and no gold-aware parsing."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upstream import module


def clean(text):
    return " ".join("".join(c.lower() if unicodedata.category(c)[0] not in "PS" else " "
                            for c in text).split())


def parse_choice(text, choices, matcher=None):
    """Only predictions and offered choices are visible here, never the gold label."""
    text = text.strip().replace("**", "").replace("`", "")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)]
    # Delimiters are explicit; never interpret the B in 'Based on...' as B.
    letter = rf"[\[(]?([{letters}])[\])]?[.!:]?"
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for value, kind in [(text, "letter_only"), (lines[-1] if lines else "", "final_letter_line")]:
        match = re.fullmatch(letter, value)
        if match and choices[ord(match[1]) - 65].strip():
            return ord(match[1]) - 65, kind
    # An explicitly introduced answer at the end, including Markdown-wrapped letters.
    match = re.search(rf"(?:answer|option|choice)\s*(?:is\s*)?[:=]?\s*[\[(]?([{letters}])[\])]?[.!]?\s*$",
                      text, flags=re.IGNORECASE)
    if match and match[1].isupper() and choices[ord(match[1]) - 65].strip():
        return ord(match[1]) - 65, "explicit_final_answer"
    explicit = re.findall(rf"(?:the (?:correct )?answer|the (?:correct )?option)\s+is\s+[\[(]?([{letters}])\b",
                          text, flags=re.IGNORECASE)
    explicit = {x for x in explicit if x.isupper() and choices[ord(x) - 65].strip()}
    if len(explicit) == 1:
        return ord(next(iter(explicit))) - 65, "unique_explicit_answer"
    if len(explicit) > 1:
        return None, "conflicting_explicit_answers"
    match = re.match(rf"^([{letters}])[.)\]:]\s+", text)
    if match and choices[ord(match[1]) - 65].strip():
        return ord(match[1]) - 65, "labeled_option"
    matches = [i for i, choice in enumerate(choices) if choice.strip() and clean(text) == clean(choice)]
    if matches and len({clean(choices[i]) for i in matches}) == 1:
        return matches[0], "exact_option_text"
    if matcher:
        matches = [i for i, choice in enumerate(choices) if choice.strip()
                   and matcher(choice, text, choices)]
        if matches and len({clean(choices[i]) for i in matches}) == 1:
            return matches[0], "unique_upstream_option_match"
    return None, "unparsed"


def mmsu_official_label(text):
    """Literal label extraction in the pinned MMSU scorer, for comparison only."""
    text = text.strip().replace("\n", "")
    if text and text != "None":
        if text[0] in "ABCD":
            return text[0]
        if len(text) > 1 and text[-2] in "ABCD":
            return text[-2]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.open()]
    rows = [r for r in rows if r["benchmark"] in {"mmau", "mmar", "mmsu"}]
    matchers = {"mmau": module("choice_mmau", "MMAU/evaluation.py").string_match,
                "mmar": module("choice_mmar", "MMAR/code/evaluation.py").string_match}
    result = {"purpose": "Post-hoc answer-format diagnosis; not unchanged official scoring or a Ming reproduction",
              "policy": "Gold-independent parsing; all examples retained; capped and unparsed answers count wrong",
              "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "models": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for run in args.run:
        predictions = {r["id"]: r for r in map(json.loads, (run / "predictions.jsonl").open())}
        reports = {}
        details = []
        for benchmark in ["mmau", "mmar", "mmsu"]:
            subset = [r for r in rows if r["benchmark"] == benchmark and r["id"] in predictions]
            counts = Counter()
            correct = parsed_correct = mapped_upstream_correct = 0
            for row in subset:
                prediction = predictions[row["id"]]
                example = row["example"]
                choices = ([example.get("choice_" + a, "") or "" for a in "abcd"]
                           if benchmark == "mmsu" else example["choices"])
                selected, method = parse_choice(prediction["text"], choices, matchers.get(benchmark))
                # Apply the gold only after parsing is finished; duplicate option text is allowed.
                good = selected is not None and choices[selected] == row["reference"]
                capped = prediction["finish_reason"] == "length"
                correct += good and not capped
                parsed_correct += good
                counts[method] += 1
                counts["generation_cap"] += capped
                if benchmark in matchers:
                    adapted = choices[selected] if selected is not None and not capped else ""
                    mapped_upstream_correct += matchers[benchmark](row["reference"], adapted, choices)
                detail = {"id": row["id"], "benchmark": benchmark, "text": prediction["text"],
                          "choices": choices, "gold": row["reference"], "selected": selected,
                          "method": method, "capped": capped, "correct": good and not capped}
                if benchmark == "mmsu":
                    original = mmsu_official_label(prediction["text"])
                    chosen = chr(65 + selected) if selected is not None else None
                    detail["official_selected"] = original
                    counts["official_label_disagrees"] += original is not None and chosen is not None and original != chosen
                    counts["official_unparsed"] += original is None
                    counts["based_prefix_selected_as_b"] += prediction["text"].startswith("Based")
                details.append(detail)
            reports[benchmark] = {"count": len(subset), "correct": correct,
                                  "accuracy_percent": 100 * correct / len(subset) if subset else None,
                                  "including_capped_accuracy_percent": 100 * parsed_correct / len(subset) if subset else None,
                                  "counts": dict(counts)}
            if benchmark in matchers:
                reports[benchmark]["mapped_upstream_accuracy_percent"] = 100 * mapped_upstream_correct / len(subset) if subset else None
        (args.output / f"{run.name}-details.json").write_text(json.dumps(details, ensure_ascii=False) + "\n")
        result["models"][run.name] = reports
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

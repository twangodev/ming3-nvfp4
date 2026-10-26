"""Generate a reviewable full-suite report, keeping diagnostic policies explicit."""
import argparse
import csv
import json
from pathlib import Path


MODELS = ["tiny-bf16", "tiny-nvfp4", "flash-nvfp4"]
NAMES = {"fleurs_en": "FLEURS English WER ↓", "fleurs_zh": "FLEURS Chinese CER ↓",
         "librispeech_clean": "LibriSpeech clean WER ↓", "librispeech_other": "LibriSpeech other WER ↓",
         "mmau": "MMAU test-mini accuracy ↑", "mmar": "MMAR accuracy ↑", "mmsu": "MMSU accuracy ↑",
         "mmlu_pro": "MMLU-Pro accuracy ↑"}


def read(path):
    return json.loads(path.read_text())


def fmt(value):
    return "—" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--budget-probe", type=Path)
    parser.add_argument("--budget-replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if read(args.run / "suite-progress.json")["state"] != "complete":
        raise ValueError("The full suite must finish before generating its report")
    summaries = {m: read(args.run / m / "summary.json")["benchmarks"] for m in MODELS}
    for model in MODELS:
        progress = read(args.run / model / "progress.json")
        if progress["completed"] != 26183 or progress.get("errors"):
            raise ValueError(f"Incomplete or failed run: {model}")
    diagnostics = args.run / "diagnostics"
    numbers = read(diagnostics / "normalization-v2/summary.json")["models"]
    choices = read(diagnostics / "choice-format-v2/summary.json")["models"]
    published = read(Path(__file__).with_name("published-values.json"))
    mmlu = read(diagnostics / "mmlu-extraction-and-cap-bounds.json")
    original_zh = read(diagnostics / "normalization-v2/tiny-bf16-strict_raw.json")
    normalized_zh = {r["id"]: r for r in read(diagnostics / "normalization-v2/tiny-bf16-ultraeval_text_character_units.json")}
    improved = sum(r["edits"] > normalized_zh[r["id"]]["edits"] for r in original_zh)
    same = sum(r["edits"] == normalized_zh[r["id"]]["edits"] for r in original_zh)
    worsened = sum(r["edits"] < normalized_zh[r["id"]]["edits"] for r in original_zh)
    args.output.mkdir(parents=True, exist_ok=True)
    table = []
    for benchmark, name in NAMES.items():
        values = {}
        for model in MODELS:
            source = summaries[model][benchmark]
            if benchmark == "fleurs_zh":
                value = numbers[model]["ultraeval_text_character_units"]["error_percent"]
                method = "UltraEval text normalization; character units"
            elif benchmark in {"mmau", "mmar", "mmsu"}:
                value = choices[model][benchmark]["accuracy_percent"]
                method = "Gold-independent answer parsing; capped/unparsed wrong; full denominator"
            elif benchmark == "mmlu_pro":
                value = source["unparsed_as_incorrect_accuracy_percent"]
                method = "Official answer extraction; unparsed wrong; no random guessing"
            else:
                value = source["score_percent"]
                method = "Official Whisper English normalization; corpus WER"
            values[model] = value
            table.append({"model": model, "benchmark": benchmark, "count": source["count"],
                          "score_percent": value, "method": method,
                          "unchanged_scorer_percent": source["score_percent"],
                          "generation_caps": source["generation_limit_count"]})
    with (args.output / "full-suite.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    by = {(r["model"], r["benchmark"]): r for r in table}
    text = ["# Full evaluation and parity investigation\n",
            "All three model runs completed all **26,183 examples across eight benchmarks**, "
            "for **78,549 predictions**, with no request errors. The original predictions and "
            "unchanged upstream scorer outputs are preserved.\n",
            "The large apparent Chinese FLEURS and audio multiple-choice gaps are strongly affected "
            "by normalization and answer parsing. These diagnostics explain much of the discrepancy; "
            "they do not establish Ming's unpublished evaluation recipe.\n",
            "The table below uses the explicitly described diagnostic policies: Chinese text "
            "normalization with character-level scoring; unambiguous multiple-choice answer parsing "
            "with all examples retained and capped/unparsed outputs wrong; and MMLU-Pro's official "
            "answer extraction with unparsed outputs wrong. English ASR is unchanged. All values are percentages. "
            "Published values are context, not a matched-protocol baseline.\n",
            "| Benchmark | Examples | Tiny published | Tiny BF16 | Tiny NVFP4 | Flash published | Flash NVFP4 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for b, name in NAMES.items():
        tp, fp = published["tiny"][b], published["flash"][b]
        text.append(f"| {name} | {by['tiny-bf16', b]['count']:,} | {fmt(tp)} | "
                    f"{fmt(by['tiny-bf16', b]['score_percent'])} | {fmt(by['tiny-nvfp4', b]['score_percent'])} | "
                    f"{fmt(fp)} | {fmt(by['flash-nvfp4', b]['score_percent'])} |")
    text += ["\n[Machine-readable table](full-suite.csv). Tiny has a local paired BF16/NVFP4 comparison. "
             "Flash has no local BF16 run; published Flash values cannot isolate quantization effects. "
             "These datasets overlap prior local development, so this is not an untouched holdout.\n",
             "<details>\n<summary>Why Chinese FLEURS differed</summary>\n",
             "English already used Whisper's number normalization. The strict Chinese recipe removed "
             "punctuation but retained digit/spelled-number differences and parenthetical Latin text. "
             "For example, `50` and `五十` counted as different strings. Some references also contain "
             "English expansions that the predictions omit.\n",
             "The independent [UltraEval-Audio normalization pipeline](https://github.com/OpenBMB/UltraEval-Audio/blob/main/audio_evals/lib/wer.py) "
             "uses raw FLEURS transcripts, its modified Whisper normalizer, and SpeechIO Chinese number "
             "normalization. Applying its text normalization to the same saved predictions gives:\n",
             "| Scoring policy | Tiny BF16 | Tiny NVFP4 | Flash NVFP4 |",
             "| --- | ---: | ---: | ---: |"]
    for key, name in [("strict_raw", "Original strict character score"),
                      ("ultraeval_text_character_units", "UltraEval text normalization, character score"),
                      ("ultraeval_raw", "Unmodified UltraEval mixed-token score")]:
        text.append("| " + name + " | " + " | ".join(f"{numbers[m][key]['error_percent']:.2f}" for m in MODELS) + " |")
    text += ["\nUltraEval's default tokenizer counts Chinese characters but groups Latin words. Its last "
             "row is **not the same metric as pure CER** and is not substituted into the main table. "
             f"Normalization improved {improved} Tiny BF16 recordings, left {same} unchanged, and worsened {worsened} "
             "by edit count. The [OWSM v4 paper](https://www.isca-archive.org/interspeech_2025/peng25c_interspeech.pdf) "
             "also documents inconsistent reading of FLEURS parenthetical text, so blanket deletion is "
             "not universally correct.\n",
             "The [sibling Ming ASR evaluator](https://github.com/inclusionAI/Ming-3.0-asr/blob/main/tools/evaluate_asr.py) "
             "gave 8.54% for Tiny BF16 and 8.45% for Tiny NVFP4 on these predictions. Its generic "
             "normalization does not supply the missing Audio Tiny/Flash FLEURS recipe.\n",
             "</details>\n",
             "<details>\n<summary>Multiple-choice parsing and original scorer outputs</summary>\n",
             "MMAU/MMAR's official string matchers expect option text; many responses contain only "
             "a letter or put their final letter after an explanation. The diagnostic adapter resolves "
             "clear selections using only the response and offered options, then checks the gold. "
             "Capped or unparseable responses remain wrong.\n",
             "MMSU's official parser accepts the first character when it is A–D. Consequently it can "
             "interpret `Based on...` as answer B, regardless of the explicit final answer, and excludes "
             "other nonempty responses entirely. The diagnostic score retains all 5,000 questions.\n",
             "These are the **unchanged scorer outputs**, retained for reproducibility rather than treated "
             "as reliable final-answer accuracy when the response format is incompatible:\n",
             "| Benchmark | Tiny BF16 | Tiny NVFP4 | Flash NVFP4 |",
             "| --- | ---: | ---: | ---: |"]
    for b, name in NAMES.items():
        text.append("| " + name + " | " + " | ".join(f"{summaries[m][b]['score_percent']:.2f}" for m in MODELS) + " |")
    text.append("\nMMSU original scored denominators: " + "; ".join(
        f"{m}: {summaries[m]['mmsu']['scored_count']:,}/5,000" for m in MODELS) + ". "
        "MMLU-Pro's original aggregation randomly guesses on unparsed answers; the main table reports "
        "them as wrong instead.\n")
    text += ["Feeding the resolved option text into the original MMAU/MMAR matchers gives the "
             "following alternative scores. Those matchers ignore word order and can accept a wrong "
             "option such as `From near to far` for `From far to near`. The main table therefore uses "
             "the selected option's exact value.\n",
             "| Adapted answer + original matcher | Tiny BF16 | Tiny NVFP4 | Flash NVFP4 |",
             "| --- | ---: | ---: | ---: |"]
    for b in ["mmau", "mmar"]:
        text.append("| " + b.upper() + " | " + " | ".join(
            f"{choices[m][b]['mapped_upstream_accuracy_percent']:.2f}" for m in MODELS) + " |")
    text.append("\n</details>\n")
    text += ["<details>\n<summary>Generation limits and remaining uncertainty</summary>\n",
             "The full suite used greedy decoding, disabled internal thinking, 512-token ASR outputs, "
             "1,024-token audio multiple-choice outputs, and the upstream MMLU-Pro 4,000-token budget. "
             "ASR used 10-gram blocking. No ASR recording was empty or truncated.\n",
             "| Benchmark | Tiny BF16 caps | Tiny NVFP4 caps | Flash NVFP4 caps |",
             "| --- | ---: | ---: | ---: |"]
    for b in ["mmau", "mmar", "mmsu", "mmlu_pro"]:
        text.append("| " + NAMES[b] + " | " + " | ".join(str(summaries[m][b]["generation_limit_count"]) for m in MODELS) + " |")
    text += [f"\nEven making every currently wrong capped Flash MMLU-Pro response correct would only "
             f"raise accuracy to **{mmlu['flash-nvfp4']['maximum_score_if_all_capped_correct']:.2f}%**, "
             "below the published 80%. Selecting the last explicit answer instead of the original "
             f"first-match rule gives {100 * mmlu['flash-nvfp4']['last_explicit_correct'] / mmlu['flash-nvfp4']['total']:.2f}%, "
             "also essentially unchanged. Neither alone explains this remaining gap.\n",
             "Long reasoning and generation limits can affect the remaining accuracy gaps. "
             "The exact published dataset/reference processing, prompts, answer extraction, and generation "
             "settings were not found in the inspected public Ming repositories. A lower diagnostic score "
             "or proximity to the published number is not proof of an exact reproduction.\n\n</details>\n"]
    if args.reference and (args.reference / "comparison.json").exists():
        ref = read(args.reference / "comparison.json")
        text += ["<details>\n<summary>Independent Hugging Face runtime control</summary>\n",
                 f"Reference status: **{ref['state']}**, {ref['completed']:,}/{ref['planned']:,} planned examples. "
                 "The same original Tiny checkpoint was run serially through Hugging Face with matching "
                 "chat prompts and generation budgets, versus concurrent vLLM serving. Original FP32 "
                 "recurrent/router parameters were retained, and derived rotary buffers were rebuilt "
                 "from their source initializer to avoid the known meta-loading issue.\n",
                 "| Benchmark | Matched examples | Exact output matches | HF score | Serving score |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for b, v in ref["benchmarks"].items():
            key = "score_percent" if b.startswith("fleurs_") else "accuracy_percent"
            text.append(f"| {NAMES[b]} | {v['count']} | {v['exact_text_matches']} | {v['hf'][key]:.2f} | {v['serving'][key]:.2f} |")
        text += ["\nChinese in this control table uses the original strict character scorer. "
                 "All 1,784 prompt-token counts matched. The identical 8.54% aggregate on all 945 "
                 "Chinese recordings shows that the large strict-scoring gap also occurs with the "
                 "original, unquantized checkpoint in its Hugging Face implementation. "
                 "The audio multiple-choice controls are subsets and are not substitutes for full-split results. "
                 "This comparison changes backend and batch schedule together.\n\n</details>\n"]
    else:
        text.append("Independent Hugging Face reference control is pending; no runtime-parity conclusion is claimed.\n")
    if args.budget_probe and (args.budget_probe / "comparison.json").exists():
        probe = read(args.budget_probe / "comparison.json")
        text += ["<details>\n<summary>Flash output-budget experiment</summary>\n",
                 "A separate experiment raised the audio multiple-choice output budget from 1,024 to 4,096 "
                 "tokens on every previously capped Flash answer, plus 32 fixed uncapped controls per task. "
                 "All other request settings and the server profile were preserved. The batch schedule "
                 "changed, so the controls also measure variation between runs. These outputs are **not** "
                 "substituted into the full-suite scores above.\n",
                 "| Task | Previously capped | Now correct | Still capped | Control accuracy before/after | Identical control outputs |",
                 "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, groups in probe["benchmarks"].items():
            cap, control = groups["previously_capped"], groups["uncapped_control"]
            text.append(f"| {name.upper()} | {cap['count']} | {cap['after_correct']} | {cap['after_capped']} | "
                        f"{control['before_correct']}/{control['count']} → {control['after_correct']}/{control['count']} | "
                        f"{control['exact_text_matches']}/{control['count']} |")
        within = sum(g["previously_capped"]["stopped_within_original_budget"] for g in probe["benchmarks"].values())
        text.append(f"\nOf the 215 previously capped cases, {within} stopped within the original 1,024-token "
                    "budget in this run. Extra tokens alone cannot explain those changes.\n")
        if args.budget_replay:
            replay = read(args.budget_replay / "comparison.json")
            probe_identity = read(args.budget_probe / "experiment.json")
            replay_identity = read(args.budget_replay / "experiment.json")
            for key in ["manifest_sha256", "baseline_journal_sha256", "checkpoint", "groups", "concurrency", "upstream", "implementation"]:
                if probe_identity[key] != replay_identity[key]:
                    raise ValueError(f"Budget experiment and replay differ in {key}")
            for path in [args.budget_probe, args.budget_replay]:
                progress = read(path / "progress.json")
                if progress["state"] != "complete" or progress.get("errors"):
                    raise ValueError(f"Incomplete or failed budget run: {path}")
            if replay["max_tokens"] != 1024 or probe["max_tokens"] != 4096:
                raise ValueError("Expected 1,024-token replay and 4,096-token budget experiment")
            text += ["A further replay used the **original 1,024-token budget** on exactly the same "
                     "311 selected recordings. Both separate runs completed without request errors.\n",
                     "| Task | Previously capped | Correct at 1,024 / 4,096 | Still capped at 1,024 / 4,096 | Control correct: original / replay / larger budget |",
                     "| --- | ---: | ---: | ---: | ---: |"]
            for name, groups in probe["benchmarks"].items():
                cap, control = groups["previously_capped"], groups["uncapped_control"]
                replay_cap = replay["benchmarks"][name]["previously_capped"]
                replay_control = replay["benchmarks"][name]["uncapped_control"]
                text.append(f"| {name.upper()} | {cap['count']} | {replay_cap['after_correct']} / {cap['after_correct']} | "
                            f"{replay_cap['after_capped']} / {cap['after_capped']} | "
                            f"{control['before_correct']} / {replay_control['after_correct']} / {control['after_correct']} of {control['count']} |")
            replay_stopped = sum(g["previously_capped"]["stopped_within_original_budget"] for g in replay["benchmarks"].values())
            replay_correct = sum(g["previously_capped"]["after_correct"] for g in replay["benchmarks"].values())
            probe_correct = sum(g["previously_capped"]["after_correct"] for g in probe["benchmarks"].values())
            text.append(f"\nAt the unchanged budget, {replay_stopped}/215 previously capped responses now stopped normally "
                        f"and {replay_correct} scored correctly, versus {probe_correct} correct with the larger budget. "
                        "This demonstrates repeat-run variation under the changed workload schedule even with "
                        "greedy decoding and the same seed. It does not isolate the numerical mechanism or prove "
                        "that every recovered answer was caused by the larger budget. A future comparison should "
                        "predeclare a sufficient budget and rerun each complete split; selected retries must not "
                        "replace only failed answers.\n")
        text.append("\n</details>\n")
    text += ["Reproduction: [suite protocol](../../evaluation/README.md), "
             "[diagnostic scripts](../../evaluation/diagnostics/README.md). Local full artifacts are under "
             "`artifacts/evaluation/standard-v1-run/`; reference artifacts are under "
             "`artifacts/evaluation/parity-hf-tiny-chat/`.\n",
             "Published sources: [ASR table](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/blob/main/assets/benchmark-asr.png), "
             "[audio understanding table](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/blob/main/assets/benchmark-audio-intelligence.png), "
             "[text table](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/blob/main/assets/benchmark-text.png).\n"]
    (args.output / "full-suite-parity.md").write_text("\n".join(text))
    print(args.output / "full-suite-parity.md")


if __name__ == "__main__":
    main()

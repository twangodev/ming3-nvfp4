"""Post-hoc Chinese normalization sensitivity; does not replace suite scores."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import unicodedata


def strict(text):
    return "".join(c for c in text.lower() if not c.isspace()
                   and unicodedata.category(c)[0] not in "PS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.source / "source"), str(args.source / "dependencies")]
    from audio_evals.lib import wer
    from audio_evals.lib.text_normalization.basic import BasicTextNormalizer
    import editdistance

    basic = BasicTextNormalizer()
    tokenizer = wer.EvaluationTokenizer(tokenizer_type="zh", lowercase=True,
                                       punctuation_removal=False, character_tokenization=False)
    protocols = {
        "strict_raw": lambda s: list(strict(s)),
        "speechio_numbers_raw": lambda s: list(strict(wer.chinese_normalizer(s))),
        "whisper_basic_raw": lambda s: list(strict(basic(s))),
        "whisper_basic_then_speechio_numbers": lambda s: list(strict(wer.chinese_normalizer(basic(s)))),
        "speechio_numbers_then_whisper_basic": lambda s: list(strict(basic(wer.chinese_normalizer(s)))),
        "ultraeval_text_character_units": lambda s: list("".join(wer.chinese_normalizer(wer.english_normalizer(s)).split())),
        "ultraeval_raw": lambda s: tokenizer.tokenize(wer.chinese_normalizer(wer.english_normalizer(s))).split(),
    }
    rows = [json.loads(line) for line in args.manifest.open()]
    rows = [r for r in rows if r["benchmark"] == "fleurs_zh"]
    results = {"purpose": "Independent normalization sensitivity, not an established Ming published protocol",
               "upstream": "https://github.com/OpenBMB/UltraEval-Audio",
               "upstream_commit": json.loads((args.source / "tree.json").read_text())["sha"],
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "models": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for run in args.run:
        predictions = {r["id"]: r for r in (json.loads(line) for line in (run / "predictions.jsonl").open())}
        subset = [r for r in rows if r["id"] in predictions]
        report = {}
        for name, normalize in protocols.items():
            details = []
            for row in subset:
                ref = row["example"]["raw_transcription"]
                hyp = predictions[row["id"]]["text"]
                hyp = hyp.partition("\t")[2] if "\t" in hyp else hyp
                r, h = normalize(ref), normalize(hyp)
                details.append({"id": row["id"], "reference": ref, "hypothesis": hyp,
                                "normalized_reference": r, "normalized_hypothesis": h,
                                "edits": editdistance.eval(r, h), "reference_units": len(r)})
            edits = sum(d["edits"] for d in details)
            units = sum(d["reference_units"] for d in details)
            report[name] = {"examples": len(subset), "edits": edits, "reference_units": units,
                            "unit": "Chinese characters plus Latin word tokens" if name == "ultraeval_raw" else "characters",
                            "error_percent": 100 * edits / units if units else None,
                            "mean_utterance_error_percent": sum(100 * d["edits"] / d["reference_units"]
                                                                for d in details) / len(details)}
            if name == "ultraeval_raw":
                # Verify our exported counts against the untouched upstream function.
                with redirect_stdout(io.StringIO()):
                    exact = wer.compute_wer([d["reference"] for d in details],
                                            [d["hypothesis"] for d in details], language="zh")
                assert abs(exact * 100 - report[name]["error_percent"]) < 1e-10
            (args.output / f"{run.name}-{name}.json").write_text(json.dumps(details, ensure_ascii=False) + "\n")
        results["models"][run.name] = report
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

# Evaluation

Run Tiny BF16, Tiny NVFP4, and Flash NVFP4 on identical full dataset splits.
The benchmark repositories are Git submodules; the client environment is separate
from quantization and GPU serving. Predictions, datasets, and logs stay under
ignored `artifacts/` directories. No weights are modified.

This protocol standardizes our local comparison. It does **not** establish that
we have reproduced Ming's published numbers: the inspected public Ming resources
do not specify the complete evaluation configuration or Chinese normalization recipe. These test sets also
overlap the data used during previous local development; this is a reproduction
and comparison exercise, not an untouched holdout.

| Benchmark | Complete split | Examples | Prompt and scoring |
| --- | --- | ---: | --- |
| FLEURS English | `en_us/test` | 647 | Ming transcription prompt; official Whisper English normalizer; corpus WER with jiwer |
| FLEURS Chinese | `cmn_hans_cn/test` | 945 | Same prompt; explicitly local strict Chinese normalization; corpus CER with jiwer |
| LibriSpeech clean | `test.clean` | 2,620 | Same English ASR recipe |
| LibriSpeech other | `test.other` | 2,939 | Same English ASR recipe |
| MMAU | `test-mini` | 1,000 | Official question/option metadata and unmodified `evaluation.py`; model adapter requests exact option text |
| MMAR | full labeled set | 1,000 | Unmodified `code/evaluation.py`; model adapter requests exact option text |
| MMSU | `train` (the released evaluation set) | 5,000 | Upstream `mmsu_inference.py` prompt and `mmsu_evaluation.py` |
| MMLU-Pro | `test`, five validation demonstrations per subject | 12,032 | Upstream `evaluate_from_api.py` prompt construction, answer extraction, and final aggregation |
| **Total per model** | | **26,183** | |

All dataset repositories, revisions, files, and expected counts are fixed in
[`prepare.py`](prepare.py). MMAU's cached HF representation is checked against
the pinned repository's original metadata after removing transport option labels
and edge whitespace; the original metadata supplies the actual prompt and scoring
input. Audio is verified against its SHA-256 digest before reuse. No development
partition is removed from these runs.

## Set up and prepare

From the repository root:

```bash
git submodule update --init --recursive
uv sync --project evaluation --locked

uv run --project evaluation --no-sync python evaluation/prepare.py \
  --output artifacts/evaluation/standard-v1/data --download
```

`--download` permits missing pinned dataset downloads on a fresh machine. Omit
it for a cache-only run. If an earlier local audio manifest is available, add
`--reuse-audio-manifest /absolute/path/to/earlier/full.jsonl` to avoid writing
duplicate audio. Preparation reads the original pinned HF datasets, not old
prompts or labels. Use a new output directory for a new preparation.

Install the separate [serving environment](../serving/README.md) before launching
GPU evaluations. It contains vLLM 0.30.0, torch 2.13.0, the pinned inclusionAI
vLLM-Omni fork, and the `ming3_serving` model-mapping/ngram plugin. Its setup also
records three measured dependency overrides that differ from its lockfile.
The CPU-only evaluation project does not install that runtime. Lockfile and
plugin fingerprints are recorded per run. These launch profiles target the
96 GB SM120 GPU; they are not portable hardware defaults.

Copy [`models.example.json`](models.example.json) to `evaluation/models.local.json`
and set the three absolute checkpoint paths. `*.local.json` stays outside Git.
Use the same Tiny profile and concurrency for BF16 and NVFP4. Flash uses its
native NVFP4 runtime and a 16,384-token context to accommodate the upstream
MMLU-Pro 4,000-token answer budget.

## Start and monitor

The suite requires Linux user systemd. It starts one owned server at a time,
checks the served model identity, runs two smoke examples from each benchmark,
then runs and scores the full split before moving to the next model. It stops
the owned server on completion or failure. It never stops an unrelated server.
The suite stops before launching a model if the output filesystem has less than
100 GiB free.

```bash
mkdir -p artifacts/evaluation/standard-v1
systemd-run --user --unit=ming-benchmarks-standard-v1-run \
  --property="WorkingDirectory=$PWD" \
  --property="StandardOutput=append:$PWD/artifacts/evaluation/standard-v1/suite.log" \
  --property=StandardError=inherit \
  "$PWD/evaluation/.venv/bin/python" evaluation/suite.py \
  --config evaluation/models.local.json \
  --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
  --output artifacts/evaluation/standard-v1-run

cat artifacts/evaluation/standard-v1-run/suite-progress.json
cat artifacts/evaluation/standard-v1-run/tiny-bf16/progress.json
tail -f artifacts/evaluation/standard-v1/suite.log
```

`suite-progress.json` names the current model and owned server unit. Each model's
`progress.json` reports completed examples, per-benchmark counts, throughput,
generation caps, and an estimate based on throughput so far. That estimate can
change substantially when the workload moves from short ASR outputs to long
MMLU-Pro reasoning.

The optional [health monitor](monitoring/README.md) checks progress, server health,
resources, and response formatting every minute and keeps a local alert history.

[Parity diagnostics](diagnostics/README.md) compare alternative Chinese text
normalization, model answer formats, and an independent Hugging Face reference.
They preserve the original suite scores and prediction journals.

To stop, use `systemctl --user stop ming-benchmarks-standard-v1-run`. Re-run the same
command to resume (reset a failed unit with `systemctl --user reset-failed
ming-benchmarks-standard-v1-run` first). Successful predictions are journaled and
skipped on resume. Changing the manifest, checkpoint identity, client code,
evaluator revision, serving profile, plugin files, or concurrency requires a
fresh run directory. The runner refuses concurrent writers and duplicate IDs.

To connect the client to an already running compatible server:

```bash
uv run --project evaluation --no-sync python evaluation/client.py \
  --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
  --model /absolute/path/to/checkpoint \
  --output artifacts/evaluation/manual-run \
  --concurrency 64 --limit-per-benchmark 2
```

Omit `--limit-per-benchmark` for a full run. A nonzero limit is permanently
recorded as a smoke run. The standalone client checks model identity but relies
on the operator to record the external server configuration.

## Protocol details

All requests use greedy decoding, seed 42, disabled thinking, and the Ming
thinker-only pipeline. The client sets both OpenAI request fields and Omni stage
sampling parameters. Generation budgets are 512 tokens for ASR, 1,024 for audio
multiple choice, and the official API evaluator's 4,000 for MMLU-Pro. ASR retains
Ming's 10-gram repetition blocker; multiple-choice tasks have no local repetition
blocker. Raw responses and finish reasons are retained. Generation caps are
reported separately and do not silently override upstream scoring.

Whisper's official normalizer is loaded from its submodule without importing
Whisper's GPU model. English normalization handles written-out numbers. Chinese
strict CER lowercases text and removes Unicode whitespace, punctuation, and
symbols. It **retains Latin text and digits**, and does not equate `50` with
`五十` or convert Traditional to Simplified Chinese. This deliberately preserves
the prior local Chinese recipe instead of choosing rules to approach a published
score. ASR hypotheses drop the language prefix before the first tab, as required
by Ming's transcription prompt. Both metrics sum edits and reference units over
the corpus; WER and CER are different units and are not directly comparable.
Chinese reports include slices with and without Latin text/digits in references
to help investigate the discrepancy. Every normalized alignment is saved.

MMAU and MMAR score the raw answer with the official string matcher. The adapter
does not turn a letter answer into option text. MMSU's official parser excludes
some malformed nonempty responses: the report includes its scored denominator,
exclusion count, and an additional accuracy over all examples.

MMLU-Pro's API driver imports several unrelated commercial SDKs. The adapter
compiles an explicit set of **unmodified function bodies** from the pinned file
and replaces only its API transport while constructing prompts. Its final
`update_result` guesses randomly on unparseable answers. We run that aggregation
once, in manifest order, with seed 12345 and additionally report accuracy with
unparseable answers counted as incorrect. No answer-dependent retries occur.

## Score and validate

```bash
uv run --project evaluation --no-sync python evaluation/score.py \
  artifacts/evaluation/standard-v1-run/tiny-bf16
uv run --project evaluation --no-sync pytest -q -c evaluation/pytest.ini evaluation/tests
```

Scoring requires exactly one successful prediction per selected example. Each
model directory contains `run.json`, `server.json`, `predictions.jsonl`,
`summary.json`, and benchmark-specific scoring inputs/logs under `scores/`.
For MMAU, MMAR, and MMSU, the unmodified upstream command-line scorer is also
executed against the exported inputs. MMLU-Pro uses the original API scorer's
functions directly. These artifacts allow rescoring without another GPU run.

Flash BF16 is outside this suite: its weights alone exceed this GPU's memory.
A separate offload feasibility run is needed before promising a paired full
Flash BF16 comparison. Existing release scores and model cards are not replaced
automatically by these runs.

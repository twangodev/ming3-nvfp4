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

Follow the [Docker setup](../docker/README.md) first. From the repository root:

```bash
docker compose build evaluation suite
docker compose run --rm evaluation evaluation/prepare.py \
  --output /work/data --download
cp evaluation/models.example.json "$MING_WORK_DIR/models.json"
```

Edit the copied model config to use checkpoint paths **inside the container**:
`/models/...`, `/work/...`, or a complete snapshot under `/cache/huggingface/hub`.
Use the same Tiny profile and concurrency for BF16 and NVFP4. These profiles
target the 96 GB SM120 GPU; they are not portable hardware defaults. Flash uses
a 16,384-token context for MMLU-Pro's 4,000-token answer budget.

`--download` permits missing pinned dataset downloads. Omit it for a cache-only
run. `--reuse-audio-manifest /work/earlier/full.jsonl` can reuse audio from a
previous preparation when its container paths still exist. Preparation reads
the original pinned datasets, not old prompts or labels. Use a new output
directory for a new preparation.

The images include self-contained copies of the five pinned upstream evaluator
repositories. Their revisions match this checkout's submodules and are recorded
in each run. Host submodule initialization is only needed for local development.
The [serving image](../serving/README.md) includes the measured runtime pins.

## Start and monitor

The suite starts one inference process group at a time inside its GPU container,
checks the model identity, scores two smoke examples per benchmark, then runs
and scores the full split. It terminates its server on completion, interruption,
or failure. Docker's init process reaps orphan workers. A shared run lock prevents
concurrent writers. Each model launch requires 100 GiB free on the output filesystem.

```bash
docker compose run -d --name ming-eval suite
docker logs -f ming-eval
cat "$MING_WORK_DIR/full-suite/suite-progress.json"
cat "$MING_WORK_DIR/full-suite/tiny-bf16/progress.json"
```

The detached job survives terminal disconnection. Check its final exit code with
`docker inspect ming-eval --format '{{.State.ExitCode}}'`; successful completion
also writes `state: complete` to `suite-progress.json`. Progress includes the
current model, completed examples, throughput, generation caps, and a changing
ETA. The optional [monitor](monitoring/README.md) records resource and response alerts.

```bash
docker stop --time 90 ming-eval
# Remove the stopped container, retaining the bind-mounted run artifacts.
docker rm ming-eval
# Resume with the same image, config, mounts, and output directory.
docker compose run -d --name ming-eval suite
```

Successful predictions are journaled and skipped on resume. Changes to the
manifest, checkpoint identity, client, evaluators, runtime packages, Dockerfile,
profile, plugins, or concurrency require a fresh output directory. **Use a new
run directory when migrating a historical run into Docker.** The committed full
results predate this migration and are preserved; this change does not claim a
fresh full-suite reproduction. Historical scoring requires its matching source
revision (the pre-migration checkout is `19f2b90`).

For a bounded check, choose a separate output directory:

```bash
docker compose run --rm suite --config /work/models.json \
  --manifest /work/data/full.jsonl --output /work/smoke \
  --limit-per-benchmark 2
```

The nonzero limit is recorded as a smoke run. `--min-free-gib` can adjust the
launch floor for such bounded checks; it does not estimate full-run storage.

To use the standalone client against `docker compose up -d serve`:

```bash
docker compose run --rm evaluation evaluation/client.py \
  --endpoint http://serve:8091/v1 --manifest /work/data/full.jsonl \
  --model /models/tiny-nvfp4 --output /work/manual-smoke \
  --concurrency 64 --limit-per-benchmark 2
```

The client checks model identity; the operator must record the external server's
configuration. [Parity diagnostics](diagnostics/README.md) retain separate journals
for normalization, answer formats, and independent Hugging Face comparisons.

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
docker compose run --rm evaluation evaluation/score.py /work/full-suite/tiny-bf16
docker compose run --rm evaluation -m pytest -p no:cacheprovider -q -c evaluation/pytest.ini evaluation/tests
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

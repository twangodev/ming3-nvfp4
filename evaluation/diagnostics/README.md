# Parity diagnostics

These tools investigate the completed suite without changing its manifests,
predictions, source fingerprints, or official scoring outputs. Their results are
separate diagnostic measurements, not a claim to reproduce Ming's private
evaluation configuration. Keep all examples in each reported denominator and
identify partial subsets explicitly.

Follow the [Docker setup](../../docker/README.md) and run these commands from the
repository root against a new Docker suite under `/work/full-suite`. Historical
runs require their matching source revision; keep their artifacts intact.

## Answer format

`choice_scoring.py` resolves unambiguous option letters, a final letter on its
own line, explicitly stated answers, and option text. Parsing receives the model
response and offered choices, never the gold answer. Unparsed and capped outputs
count as incorrect. Duplicate option texts are scored by their value. It also
records disagreements with MMSU's original first-character parser, which can
interpret the start of `Based on...` as answer B.

```bash
docker compose run --rm evaluation evaluation/diagnostics/choice_scoring.py \
  --manifest /work/data/full.jsonl \
  --run /work/full-suite/tiny-bf16 \
  --run /work/full-suite/tiny-nvfp4 \
  --run /work/full-suite/flash-nvfp4 \
  --output /work/full-suite/diagnostics/choice-format-v2

docker compose run --rm evaluation -m pytest -p no:cacheprovider -q -c evaluation/pytest.ini \
  evaluation/diagnostics/test_choice_scoring.py
```

## Chinese normalization

`normalization.py` applies several fixed normalization pipelines to the same
saved predictions. The independent source is
[UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio), including its
SpeechIO Chinese number normalizer and modified Whisper English normalizer.
Its source snapshot and isolated dependencies are under
`/work/research/ultraeval-normalization/`; the snapshot revision is recorded
in each report. No benchmark or serving environment is modified.

UltraEval's Chinese tokenizer uses Chinese characters and Latin word tokens.
Its result is therefore distinguished from character error rate on the same
normalized text. Normalization can remove parenthetical content that was spoken
in some recordings, so a lower error rate alone does not establish correctness
of the normalization policy. Original raw and strict scores remain available.

```bash
docker compose run --rm evaluation evaluation/diagnostics/fetch_normalization_source.py \
  --output /work/research/ultraeval-normalization
docker compose run --rm --entrypoint uv evaluation pip install --python /opt/ming/evaluation/.venv/bin/python \
  --target /work/research/ultraeval-normalization/dependencies \
  -r evaluation/diagnostics/normalization-requirements.txt

docker compose run --rm evaluation evaluation/diagnostics/normalization.py \
  --source /work/research/ultraeval-normalization \
  --manifest /work/data/full.jsonl \
  --run /work/full-suite/tiny-bf16 \
  --run /work/full-suite/tiny-nvfp4 \
  --run /work/full-suite/flash-nvfp4 \
  --output /work/full-suite/diagnostics/normalization-v2
```

The script exports normalized references, hypotheses, edit counts, and denominators.
Its UltraEval aggregate is checked against the unmodified upstream `compute_wer`.

## Hugging Face reference

`hf_reference.py` runs the cached original Tiny checkpoint through its Hugging
Face implementation. It uses the quantization image, greedy generation,
eager attention, original FP32 recurrent/router parameters, and the suite's
prompts and token budgets. Derived rotary buffers are checked and rebuilt from
the model's own initializer to avoid the known Transformers meta-loading issue;
the check is saved in `loader-audit.json`. No checkpoint tensors are overwritten.

The default diagnostic covers all 945 Chinese and 647 English FLEURS recordings,
plus 64 examples from each audio multiple-choice benchmark. Subset selection is
deterministic from example IDs and does not inspect model answers. Inference is
one example at a time. This isolates a different backend and schedule together;
it is not a test of batch invariance alone.

```bash
docker compose run --rm --entrypoint /opt/ming/.venv/bin/python quantize evaluation/diagnostics/hf_reference.py \
  --model /models/tiny-bf16 \
  --manifest /work/data/full.jsonl \
  --output /work/parity-hf-tiny-chat \
  --wait-for-suite /work/full-suite/suite-progress.json
```

The optional wait prevents the reference run from competing with the standard
suite for GPU memory. A source/configuration change requires a fresh output
directory. Completed predictions are flushed individually and skipped on resume.
`--mode bare_asr` is a separate prompt-format experiment, not the matched chat
reference; it must use its own output directory and ASR-only counts.

## Output budget

`budget_probe.py` is a separately recorded experiment on Flash's truncated
audio multiple-choice answers, plus 32 deterministically selected untruncated
controls per benchmark. It changes the output budget from 1,024 to 4,096 tokens,
retains the other request settings and the original server profile, and waits
for the Hugging Face run to release its shared lock before loading Flash. The new batch schedule
can also affect responses; the controls help assess that variation.

Selection uses completion status and IDs, not correctness. These selected cases
are not a benchmark-wide sample, and their new outputs must not be spliced into
the original full-suite scores. Its separate journal and `experiment.json`
preserve the selection and configuration.

The follow-up uses `--max-tokens 1024` on exactly the same selected IDs to assess
repeat-run variation at the original budget. It uses a separate output directory
and container. Each experiment saves its runner source alongside its identity.

```bash
docker compose run --rm --entrypoint /opt/ming/evaluation/.venv/bin/python suite evaluation/diagnostics/budget_probe.py \
  --manifest /work/data/full.jsonl \
  --baseline /work/full-suite/flash-nvfp4 \
  --config /work/models.json \
  --output /work/parity-flash-budget \
  --wait-for-reference /work/parity-hf-tiny-chat

docker compose run --rm --entrypoint /opt/ming/evaluation/.venv/bin/python suite evaluation/diagnostics/budget_probe.py \
  --manifest /work/data/full.jsonl \
  --baseline /work/full-suite/flash-nvfp4 \
  --config /work/models.json \
  --output /work/parity-flash-replay1024 \
  --wait-for-reference /work/parity-flash-budget \
  --max-tokens 1024
```

For detached jobs, replace `--rm` with `-d --name UNIQUE_NAME` and follow
progress with `docker logs -f UNIQUE_NAME`, as in the [suite guide](../README.md).
The optional wait path identifies the preceding run and its shared-volume lock. Use new output directories for a fresh run; source changes
are intentionally rejected when resuming an existing experiment.

Score both experiments with `choice_scoring.py`, then summarize them with
`budget_comparison.py`:

```bash
for PROBE in parity-flash-budget parity-flash-replay1024; do
  docker compose run --rm evaluation evaluation/diagnostics/choice_scoring.py \
    --manifest /work/data/full.jsonl \
    --run "/work/$PROBE" \
    --output "/work/$PROBE/scores"
  docker compose run --rm evaluation evaluation/diagnostics/budget_comparison.py \
    --probe "/work/$PROBE" \
    --baseline /work/full-suite/flash-nvfp4 \
    --baseline-details /work/full-suite/diagnostics/choice-format-v2/flash-nvfp4-details.json \
    --probe-details "/work/$PROBE/scores/$PROBE-details.json"
done
```

## Completed report

`reference_comparison.py` compares the independent HF journal with the Tiny BF16
serving journal. Run it after completing the optional reference run:

```bash
docker compose run --rm evaluation evaluation/diagnostics/reference_comparison.py \
  --manifest /work/data/full.jsonl \
  --reference /work/parity-hf-tiny-chat \
  --serving /work/full-suite/tiny-bf16
```

`mmlu_sensitivity.py` checks answer extraction and computes an upper bound if
every capped MMLU-Pro response were correct. Generate this input after the
answer-format and normalization diagnostics above:

```bash
docker compose run --rm evaluation evaluation/diagnostics/mmlu_sensitivity.py \
  --manifest /work/data/full.jsonl \
  --run /work/full-suite/tiny-bf16 \
  --run /work/full-suite/tiny-nvfp4 \
  --run /work/full-suite/flash-nvfp4 \
  --output /work/full-suite/diagnostics/mmlu-extraction-and-cap-bounds.json
```

The report generator reads those comparisons and the scoring diagnostics. Use
the fixed model-directory names shown here. The reference and two budget
arguments are optional; omit them if you did not run those experiments:

```bash
docker compose run --rm evaluation evaluation/diagnostics/build_report.py \
  --run /work/full-suite \
  --reference /work/parity-hf-tiny-chat \
  --budget-probe /work/parity-flash-budget \
  --budget-replay /work/parity-flash-replay1024 \
  --output /work/report
```

The generated table labels its scoring policies and keeps published values as
context, not as an exact protocol match. Raw results, alternative policies, and
the controlled experiments remain separate in expandable notes.

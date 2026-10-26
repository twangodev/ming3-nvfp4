# Parity diagnostics

These tools investigate the completed suite without changing its manifests,
predictions, source fingerprints, or official scoring outputs. Their results are
separate diagnostic measurements, not a claim to reproduce Ming's private
evaluation configuration. Keep all examples in each reported denominator and
identify partial subsets explicitly.

## Answer format

`choice_scoring.py` resolves unambiguous option letters, a final letter on its
own line, explicitly stated answers, and option text. Parsing receives the model
response and offered choices, never the gold answer. Unparsed and capped outputs
count as incorrect. Duplicate option texts are scored by their value. It also
records disagreements with MMSU's original first-character parser, which can
interpret the start of `Based on...` as answer B.

```bash
evaluation/.venv/bin/python evaluation/diagnostics/choice_scoring.py \
  --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
  --run artifacts/evaluation/standard-v1-run/tiny-bf16 \
  --run artifacts/evaluation/standard-v1-run/tiny-nvfp4 \
  --run artifacts/evaluation/standard-v1-run/flash-nvfp4 \
  --output artifacts/evaluation/standard-v1-run/diagnostics/choice-format-v2

evaluation/.venv/bin/python -m pytest -q -c evaluation/pytest.ini \
  evaluation/diagnostics/test_choice_scoring.py
```

## Chinese normalization

`normalization.py` applies several fixed normalization pipelines to the same
saved predictions. The independent source is
[UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio), including its
SpeechIO Chinese number normalizer and modified Whisper English normalizer.
Its source snapshot and isolated dependencies are under
`artifacts/research/ultraeval-normalization/`; the snapshot revision is recorded
in each report. No benchmark or serving environment is modified.

UltraEval's Chinese tokenizer uses Chinese characters and Latin word tokens.
Its result is therefore distinguished from character error rate on the same
normalized text. Normalization can remove parenthetical content that was spoken
in some recordings, so a lower error rate alone does not establish correctness
of the normalization policy. Original raw and strict scores remain available.

```bash
evaluation/.venv/bin/python evaluation/diagnostics/fetch_normalization_source.py \
  --output artifacts/research/ultraeval-normalization
uv pip install --python evaluation/.venv/bin/python \
  --target artifacts/research/ultraeval-normalization/dependencies \
  -r evaluation/diagnostics/normalization-requirements.txt

evaluation/.venv/bin/python evaluation/diagnostics/normalization.py \
  --source artifacts/research/ultraeval-normalization \
  --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
  --run artifacts/evaluation/standard-v1-run/tiny-bf16 \
  --run artifacts/evaluation/standard-v1-run/tiny-nvfp4 \
  --run artifacts/evaluation/standard-v1-run/flash-nvfp4 \
  --output artifacts/evaluation/standard-v1-run/diagnostics/normalization-v2
```

The script exports normalized references, hypotheses, edit counts, and denominators.
Its UltraEval aggregate is checked against the unmodified upstream `compute_wer`.

## Hugging Face reference

`hf_reference.py` runs the cached original Tiny checkpoint through its Hugging
Face implementation. It uses the repository root environment, greedy generation,
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
.venv/bin/python evaluation/diagnostics/hf_reference.py \
  --model /absolute/path/to/cached/original-tiny-snapshot \
  --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
  --output artifacts/evaluation/parity-hf-tiny-chat \
  --wait-for-suite artifacts/evaluation/standard-v1-run/suite-progress.json
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
for the Hugging Face service to exit before loading Flash. The new batch schedule
can also affect responses; the controls help assess that variation.

Selection uses completion status and IDs, not correctness. These selected cases
are not a benchmark-wide sample, and their new outputs must not be spliced into
the original full-suite scores. Its separate journal and `experiment.json`
preserve the selection and configuration.

The follow-up uses `--max-tokens 1024` on exactly the same selected IDs to assess
repeat-run variation at the original budget. It uses a separate output directory
and server unit. Each experiment saves its runner source alongside its identity.

```bash
evaluation/.venv/bin/python evaluation/diagnostics/budget_probe.py \
  --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
  --baseline artifacts/evaluation/standard-v1-run/flash-nvfp4 \
  --config evaluation/models.local.json \
  --output artifacts/evaluation/parity-flash-budget \
  --wait-for-reference artifacts/evaluation/parity-hf-tiny-chat \
  --reference-unit ming-parity-hf-tiny-chat

evaluation/.venv/bin/python evaluation/diagnostics/budget_probe.py \
  --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
  --baseline artifacts/evaluation/standard-v1-run/flash-nvfp4 \
  --config evaluation/models.local.json \
  --output artifacts/evaluation/parity-flash-replay1024 \
  --wait-for-reference artifacts/evaluation/parity-flash-budget \
  --reference-unit ming-parity-flash-budget \
  --max-tokens 1024 \
  --server-unit ming-parity-flash-replay1024-server
```

Launch each long-running command under a persistent user service, as described
in the [suite protocol](../README.md). The wait-unit argument must name the
preceding service. Use new output directories for a fresh run; source changes
are intentionally rejected when resuming an existing experiment.

Score both experiments with `choice_scoring.py`, then summarize them with
`budget_comparison.py`:

```bash
for PROBE in parity-flash-budget parity-flash-replay1024; do
  evaluation/.venv/bin/python evaluation/diagnostics/choice_scoring.py \
    --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
    --run "artifacts/evaluation/$PROBE" \
    --output "artifacts/evaluation/$PROBE/scores"
  evaluation/.venv/bin/python evaluation/diagnostics/budget_comparison.py \
    --probe "artifacts/evaluation/$PROBE" \
    --baseline artifacts/evaluation/standard-v1-run/flash-nvfp4 \
    --baseline-details artifacts/evaluation/standard-v1-run/diagnostics/choice-format-v2/flash-nvfp4-details.json \
    --probe-details "artifacts/evaluation/$PROBE/scores/$PROBE-details.json"
done
```

## Completed report

`reference_comparison.py` compares the independent HF journal with the Tiny BF16
serving journal. Run it after completing the optional reference run:

```bash
evaluation/.venv/bin/python evaluation/diagnostics/reference_comparison.py \
  --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
  --reference artifacts/evaluation/parity-hf-tiny-chat \
  --serving artifacts/evaluation/standard-v1-run/tiny-bf16
```

`mmlu_sensitivity.py` checks answer extraction and computes an upper bound if
every capped MMLU-Pro response were correct. Generate this input after the
answer-format and normalization diagnostics above:

```bash
evaluation/.venv/bin/python evaluation/diagnostics/mmlu_sensitivity.py \
  --manifest artifacts/evaluation/standard-v1/data/full.jsonl \
  --run artifacts/evaluation/standard-v1-run/tiny-bf16 \
  --run artifacts/evaluation/standard-v1-run/tiny-nvfp4 \
  --run artifacts/evaluation/standard-v1-run/flash-nvfp4 \
  --output artifacts/evaluation/standard-v1-run/diagnostics/mmlu-extraction-and-cap-bounds.json
```

The report generator reads those comparisons and the scoring diagnostics. Use
the fixed model-directory names shown here. The reference and two budget
arguments are optional; omit them if you did not run those experiments:

```bash
evaluation/.venv/bin/python evaluation/diagnostics/build_report.py \
  --run artifacts/evaluation/standard-v1-run \
  --reference artifacts/evaluation/parity-hf-tiny-chat \
  --budget-probe artifacts/evaluation/parity-flash-budget \
  --budget-replay artifacts/evaluation/parity-flash-replay1024 \
  --output docs/results
```

The generated table labels its scoring policies and keeps published values as
context, not as an exact protocol match. Raw results, alternative policies, and
the controlled experiments remain separate in expandable notes.

# Reproducing quantization

Use Linux, Python 3.12 and an NVIDIA CUDA-capable GPU with enough memory for the
chosen placement. Historical calibration/export ran on Debian 13.7 with an RTX
PRO 6000 Blackwell (97,887 MiB, SM120), about 125 GiB RAM and a Threadripper 9960X.
[Provenance](results/provenance.json) records the host receipt hash. These are
observed host specifications, not tested minimum requirements. No fresh clean
GPU reproduction is claimed by this documentation update.

From a fresh checkout, install the locked quantization/data environment:

```bash
uv sync --locked --extra data
uv run --no-sync ming-quant host
```

`uv run --no-sync` requires that prior sync; it does not install missing packages.
For development checks use `uv sync --locked --extra data --group dev`, then
`uv run --no-sync pytest -q` and `uv run --no-sync ruff check src tests`.
The lock pins PyTorch 2.8.0 (CUDA 12.8), Transformers 5.14.1, Accelerate 1.10.1,
and Model Optimizer 0.47.0. Run GPU commands where NVIDIA devices are accessible.

Obtain complete source snapshots separately at the revisions in the recipes,
including remote-code and audio resources, then set explicit absolute paths.
The commands below never fetch source weights. Tiny and Flash are separate runs;
do not launch competing calibration jobs on one GPU.

```bash
MODEL_SOURCE=/absolute/path/to/source/snapshot
RECIPE=recipes/tiny.toml
RUN_ROOT=/absolute/path/to/local/tiny-run
```

For Flash use `RECIPE=recipes/flash.toml`, its pinned source, and a distinct run
root. Create the run root on a filesystem with space for source weights,
calibration state, offload spill and exported shards. The recipes require a
100 GiB free-disk floor; this is additional to already occupied files and is
not a complete storage estimate. Flash source root weights alone are about
239 GiB. Export requires an empty destination and never overwrites source weights.

## Recipe behavior

Both recipes use seed 42, MSE W4A4 calibration, maximum 2,048 input tokens,
batches of up to eight examples / 4,096 tokens, and a 128-token per-expert
observation heuristic. Only equal-length examples batch together because Ming's
recurrent layers do not mask padding. Inputs exceeding the token limit fail
rather than silently truncate audio placeholders.

Tiny retains routed layers 5, 10, 15, 18, 20, 21 and 22 in BF16; the other 16 routed
layers export 6,144 packed projections. Its placement budgets are 80 GiB GPU /
80 GiB CPU, with 16 GiB scratch allowance. These budgets are not allocation peaks.

Flash exports all 61,440 projections across 20,480 routed experts. Its original
raw coverage has 341 experts below 128 observations, including four unobserved
experts. The explicit `modelopt_pooled_inputs_v1` policy shares activation scale
inputs per projection using Model Optimizer's standard expert behavior. Raw
coverage and this scale-policy result remain separate; successful export does
not turn the original coverage deficits into observed data. Flash budgets are
40 GiB GPU / 56 GiB CPU / 8 GiB scratch, with layer checkpoint cadence eight and
source-shard-only offload. No synthetic scale fallback is used.

The loader reconstructs Ming's derived nonpersistent rotary-frequency buffer
from the upstream initializer before calibration; learned weights remain
unchanged. Auditing verifies packed projection counts/scales and compares
retained root tensors byte-for-byte against source. This does not certify task
quality; see the separate [aggregate evidence](results/README.md).

## Data and commands

Reconstruct the ordered 2,053-example training selection from pinned upstream
sources. This command downloads calibration sources/assets, verifies recorded
identities and hashes, and refuses to overwrite an existing output:

```bash
uv run --no-sync ming-quant prepare-calibration \
  --output "$RUN_ROOT/calibration.jsonl" --assets "$RUN_ROOT/assets"
uv run --no-sync ming-quant --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  inspect --output "$RUN_ROOT/inventory.json"
uv run --no-sync ming-quant --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  coverage --manifest "$RUN_ROOT/calibration.jsonl" \
  --output "$RUN_ROOT/coverage" --layerwise
```

[calibration-selection.json](../recipes/calibration-selection.json) records the
ordered sample choices, pinned source revisions, content hashes and evaluation
exclusions. Absolute audio paths differ across hosts, so the historical raw JSONL
byte hash is not the portability identity. Keep generated metadata and use
`check-manifests` with your own development/acceptance manifests when adding data:

```bash
uv run --no-sync ming-quant check-manifests \
  "$RUN_ROOT/calibration.jsonl" /absolute/path/to/evaluation.jsonl
```

Tiny calibration and export:

```bash
uv run --no-sync ming-quant --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  calibrate --manifest "$RUN_ROOT/calibration.jsonl" \
  --coverage-from "$RUN_ROOT/coverage" --output "$RUN_ROOT/calibration" --layerwise
uv run --no-sync ming-quant --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  export --calibration "$RUN_ROOT/calibration" --output "$RUN_ROOT/checkpoint"
uv run --no-sync ming-quant --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  audit-export --checkpoint "$RUN_ROOT/checkpoint" --output "$RUN_ROOT/export-audit.json"
```

Flash additionally requires a bounded calibration preflight using the pinned Tiny
source snapshot, even when the calibration target is Flash. Set the validation
source to Tiny revision `3c9578f641b6745abf2d1239dc8419726f37c0e2`:

```bash
TINY_VALIDATION_SOURCE=/absolute/path/to/pinned/tiny/snapshot
uv run --no-sync ming-quant --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  validate-flash --manifest "$RUN_ROOT/calibration.jsonl" \
  --validation-source "$TINY_VALIDATION_SOURCE" --output "$RUN_ROOT/validation"
uv run --no-sync ming-quant --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  calibrate --manifest "$RUN_ROOT/calibration.jsonl" \
  --coverage-from "$RUN_ROOT/coverage" --validation "$RUN_ROOT/validation" \
  --output "$RUN_ROOT/calibration" --layerwise
```

Then use the same export/audit commands above. This preflight checks bounded
calibration behavior; it is neither a quality evaluation nor a complete Flash
calibration. Raw coverage deficits remain visible under the pooled policy.
The preflight verifies Tiny's source files, including weight shards, against the
immutable revision hashes in `recipes/tiny-validation-source.json`. Its bundle
is also tied to the current package versions and implementation. These checks
read source weights but do not copy or modify them.

Calibration includes reference answer tokens. Prepared manifests contain absolute
audio paths; keep those asset locations unchanged, or prepare a new manifest at
the new location. Preserve pinned source
metadata and audio checksums, and check evaluation disjointness before using new
calibration data. The prepared mixture is a local reproducibility input, not a
claim of coverage for every Ming capability. Dataset terms still apply; the BELLE training subset has research/noncommercial
restrictions, and Clotho recordings retain individual licenses/attribution. Neither
audio nor examples are checked into this repository.

The selected historical checkpoints used this expanded 2,053-example selection.
Reconstructing the data and settings does not guarantee identical GPU outputs or
checkpoint bytes across hosts and dependency environments. A different manifest
does not transfer the reported quality. Preserve exact source revisions, data
choices and normalized manifest identity when comparing a new candidate.

Tiny calibration can resume with the identical command and paths. Recipe, manifest,
source revision, package versions and lockfile must match the recorded state.
Flash pooled calibration requires a fresh output directory. Its pooling evidence
is not persisted at interruption boundaries, so calibration resume is unsupported;
export from completed Flash calibration is supported in a separate process.
Treat an interrupted export as incomplete; use a new empty destination before
retrying. Keep the generated manifest/provenance, coverage, calibration state,
and audit outputs locally to trace a newly produced checkpoint.

## Validation of this repository

The curated implementation passed 102 CPU tests, Ruff, and locked dependency
resolution. All 2,053 selected calibration examples were reconstructed offline
from cached pinned sources and checked against their recorded content hashes.
Existing released checkpoint headers confirm 6,144 Tiny and 61,440 Flash packed
projections. The eight Flash preflight inputs also passed CPU preprocessing.

The extracted Flash preflight has not been rerun end-to-end on a GPU. Historical
GPU evidence supports the underlying checks; CPU tests of this extraction do not
replace a fresh calibration/export run. No full recalibration or benchmark was
performed while preparing these commits.

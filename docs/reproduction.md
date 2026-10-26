# Reproducing quantization

Use the [Docker setup](../docker/README.md) with NVIDIA GPU access. The three
images isolate quantization, serving, and CPU evaluation dependencies. Historical
calibration/export ran on an RTX PRO 6000 Blackwell (97,887 MiB, SM120), about
125 GiB RAM and a Threadripper 9960X. These are observed specifications, not tested
minimum requirements. [Provenance](results/provenance.json) records that host.

From the repository root, after configuring the persistent Docker mounts:

```bash
docker compose build quantize
docker compose run --rm quantize host
```

The image pins Python 3.12.13, CUDA 12.8, PyTorch 2.8.0, Transformers 5.14.1,
Accelerate 1.10.1, and Model Optimizer 0.47.0. Host Python and CUDA toolkits are
unnecessary; the NVIDIA driver and container runtime remain host prerequisites.

Obtain complete source snapshots separately at the revisions in the recipes,
including remote-code and audio resources, then set paths inside the container.
The commands below never fetch source weights. Tiny and Flash are separate runs;
do not launch competing calibration jobs on one GPU.

```bash
MODEL_SOURCE=/models/tiny-bf16
RECIPE=recipes/tiny.toml
RUN_ROOT=/work/tiny-run
```

For Flash use `RECIPE=recipes/flash.toml`, its pinned source, and a distinct run
root. Place the host directory mounted at `/work` on a filesystem with space for
source weights, calibration state, offload spill and exported shards. The recipes require a
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
docker compose run --rm quantize prepare-calibration \
  --output "$RUN_ROOT/calibration.jsonl" --assets "$RUN_ROOT/assets"
docker compose run --rm quantize --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  inspect --output "$RUN_ROOT/inventory.json"
docker compose run --rm quantize --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  coverage --manifest "$RUN_ROOT/calibration.jsonl" \
  --output "$RUN_ROOT/coverage" --layerwise
```

[calibration-selection.json](../recipes/calibration-selection.json) records the
ordered sample choices, pinned source revisions, content hashes and evaluation
exclusions. Absolute audio paths differ across hosts, so the historical raw JSONL
byte hash is not the portability identity. Keep generated metadata and use
`check-manifests` with your own development/acceptance manifests when adding data:

```bash
docker compose run --rm quantize check-manifests \
  "$RUN_ROOT/calibration.jsonl" /work/evaluation.jsonl
```

Tiny calibration and export:

```bash
docker compose run --rm quantize --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  calibrate --manifest "$RUN_ROOT/calibration.jsonl" \
  --coverage-from "$RUN_ROOT/coverage" --output "$RUN_ROOT/calibration" --layerwise
docker compose run --rm quantize --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  export --calibration "$RUN_ROOT/calibration" --output "$RUN_ROOT/checkpoint"
docker compose run --rm quantize --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  audit-export --checkpoint "$RUN_ROOT/checkpoint" --output "$RUN_ROOT/export-audit.json"
```

Flash additionally requires a bounded calibration preflight using the pinned Tiny
source snapshot, even when the calibration target is Flash. Set the validation
source to Tiny revision `3c9578f641b6745abf2d1239dc8419726f37c0e2`:

```bash
TINY_VALIDATION_SOURCE=/models/tiny-bf16
docker compose run --rm quantize --recipe "$RECIPE" --source "$MODEL_SOURCE" \
  validate-flash --manifest "$RUN_ROOT/calibration.jsonl" \
  --validation-source "$TINY_VALIDATION_SOURCE" --output "$RUN_ROOT/validation"
docker compose run --rm quantize --recipe "$RECIPE" --source "$MODEL_SOURCE" \
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

## Evaluate an exported checkpoint

Quantization, serving, and benchmark scoring have separate environments. After
export and the retained-tensor audit, build the [serving runtime](../serving/README.md)
and follow the [full-suite evaluation guide](../evaluation/README.md). Set
`$MING_WORK_DIR/models.json` to the container paths for your exported checkpoints
and pinned original Tiny source. Generated checkpoints are under `/work`.

The evaluation image includes the pinned benchmark repositories. The evaluator
prepares pinned full dataset splits, runs original Tiny BF16 and both NVFP4
checkpoints, and preserves responses for separate scoring. Its data
preparation downloads are distinct from the training-only calibration selection
above. Do not use benchmark answers to recalibrate a candidate and then call the
same benchmark an untouched holdout.

The [completed full-suite report](results/full-suite-parity.md) records the
published local scores and their interpretation. The
[diagnostic workflow](../evaluation/diagnostics/README.md) regenerates its
normalization and answer-parsing tables from a completed run. The repository
contains aggregate results; per-example predictions, benchmark audio, exported
weights, calibration state, and release-upload workspaces remain local.

## Validation of the quantization extraction

The curated implementation passed 102 CPU tests, Ruff, and locked dependency
resolution. All 2,053 selected calibration examples were reconstructed offline
from cached pinned sources and checked against their recorded content hashes.
Existing released checkpoint headers confirm 6,144 Tiny and 61,440 Flash packed
projections. The eight Flash preflight inputs also passed CPU preprocessing.

The extracted Flash preflight has not been rerun end-to-end on a GPU. Historical
GPU evidence supports the underlying checks; CPU tests of this extraction do not
replace a fresh calibration/export run. No full recalibration or benchmark was
performed while preparing the original quantization commits. The later full
evaluation ran existing checkpoints; it did not rerun calibration or export.

## Development checks

For local CPU development, install the locked development environment with
`uv sync --locked --extra data --group dev`, then run `uv run --no-sync pytest -q`
and `uv run --no-sync ruff check src tests`. This optional development environment
is separate from the Docker workflow.

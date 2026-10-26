# Docker setup

Run all commands from the repository root. Install Docker Engine with Compose
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Verify GPU access with `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi`.
The provided profiles target a 96 GB Blackwell SM120 GPU on Linux x86-64.

## Persistent paths

```bash
cp .env.example .env
# Edit .env with absolute paths for your machine, then load it:
set -a
. ./.env
set +a
export MING_UID="$(id -u)" MING_GID="$(id -g)"
mkdir -p "$MING_MODEL_DIR" "$MING_WORK_DIR" "$MING_HF_HUB_DIR"
docker compose config --quiet
docker compose build quantize evaluation suite
```

Use shell-compatible absolute paths in `.env` (quote paths containing spaces).
The UID/GID settings make generated files belong to you. Create these directories
before launching containers; Docker otherwise creates missing bind paths as root.

| Host setting | Container path | Purpose |
| --- | --- | --- |
| `MING_MODEL_DIR` | `/models` (read-only) | Complete original and exported checkpoints |
| `MING_WORK_DIR` | `/work` | Calibration, datasets, outputs, journals and compiler caches |
| `MING_HF_HUB_DIR` | `/cache/huggingface/hub` | Persistent Hugging Face downloads |

Mount the **whole Hub cache**, including `blobs/` and `snapshots/`, when using
cached snapshots with relative symlinks. A snapshot can then be referenced as
`/cache/huggingface/hub/models--ORG--MODEL/snapshots/REVISION`. Alternatively use
self-contained checkpoint directories under `/models`. Absolute paths recorded
in manifests refer to container paths; keep those paths stable across runs.

Place `/work` on a large filesystem. Calibration and full evaluation enforce a
100 GiB free-space floor, in addition to existing weights and datasets. The first
build of both CUDA images used roughly 90 GiB for images and build cache on the
validation host; allow for that separately in Docker's storage. GPU selection is
controlled by `MING_GPU` (default `0`). Run one GPU job at a time.

## Images and jobs

| Service | Image target | Command |
| --- | --- | --- |
| `quantize` | CUDA 12.8 + root lock | `docker compose run --rm quantize --help` |
| `evaluation` | CPU + evaluation lock | `docker compose run --rm evaluation evaluation/prepare.py --help` |
| `serve` | CUDA 13 + serving lock and measured runtime pins | `docker compose up -d serve` |
| `suite` | Same serving image, with CPU evaluator | `docker compose run -d --name ming-eval suite` |

See [quantization](../docs/reproduction.md), [serving](../serving/README.md), and
[evaluation](../evaluation/README.md) for complete commands. Use `docker logs -f`
to follow detached jobs, `docker stop --time 90` to interrupt them, and `docker rm`
to remove stopped containers. Bind-mounted outputs survive container removal.
Jobs do not restart automatically; inspect the failure before resuming. Tiny
calibration and evaluation support resume; interrupted Flash pooled calibration
requires a fresh output directory.

Base images and uv are pinned by digest, Python and package dependencies are
pinned, and evaluator revisions match the Git submodules. The build downloads
public dependencies and evaluators; weights and datasets are mounted separately.
The build context excludes local weights, environments, run artifacts, and secrets.
No Docker socket or host process-manager interface is mounted into jobs.

For CPU development checks, the evaluation image includes its locked test tools.
Serving repetition tests also run in the serving image:

```bash
docker compose run --rm evaluation -m pytest -p no:cacheprovider -q -c evaluation/pytest.ini evaluation/tests evaluation/diagnostics/test_choice_scoring.py
docker compose run --rm --entrypoint /opt/ming/evaluation/.venv/bin/python suite -m pytest -p no:cacheprovider -q -c serving/pytest.ini serving/tests
```

## Validation

All three images built on the 96 GB SM120 host. The quantization image passed a
GPU compute check and inspected the complete pinned Tiny source. Tiny NVFP4
completed two examples from each of the eight benchmarks, with no empty answers,
request errors, or generation caps; its container exited successfully and released
GPU memory. Thirteen of the sixteen raw responses matched the historical smoke
run exactly; the others differed in capitalization or answer wording.

This validates container integration. Full calibration, the full evaluation
suite, and Flash GPU inference were not rerun for this migration. Published
scores remain the measurements in the existing full-suite report.

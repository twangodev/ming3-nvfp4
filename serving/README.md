# Serving the exported checkpoints

This separate environment supplies the native vLLM-Omni runtime used by the
full evaluation suite. Quantization and export use the root environment; do not
install these serving dependencies into that environment.

The measured target is Linux x86-64, Python 3.12, and an NVIDIA RTX PRO 6000
Blackwell with approximately 96 GiB VRAM (SM120). Flash's exported NVFP4 weights
fit on this GPU; its original BF16 weights do not. The evaluation profiles are
hardware-specific, not tested minimum-memory defaults.

## Installation

From the repository root, with NVIDIA drivers and a working CUDA toolchain:

```bash
uv sync --project serving --locked
uv pip install --python serving/.venv/bin/python --no-deps \
  -r serving/measured-runtime.txt
```

The lock pins vLLM 0.30.0, torch 2.13.0, and the exact inclusionAI vLLM-Omni
revision. The **measured environment also has three dependency overrides**:
Triton 3.7.1, nvidia-cuda-nvrtc 13.0.88, and nvidia-cuda-runtime 13.0.96.
The requirements file records those observed versions. Its `--no-deps` install
is intentional: these overrides differ from the resolved dependency metadata.
The original lock and runtime source are preserved because their identities
were recorded with the evaluation results.

Use the environment's Python directly, or `uv run --project serving --no-sync`.
Running `uv sync` again restores locked versions; reapply the overrides afterward.
Locked resolution and installed versions were checked locally. A fresh full GPU
evaluation after reinstalling this environment has not been performed.

## Evaluation runtime

The plugin maps Ming's quantization exclusions through the audio wrapper so
retained BF16 attention tensors load correctly. It also supplies incremental
prompt-plus-output n-gram blocking for ASR. The evaluation launcher registers
both `ming_nvfp4` and `vllm_omni_register_models` and explicitly disables the
optional Tiny FP4 input/output fusion kernels, matching the quality runs.

The [evaluation guide](../evaluation/README.md) launches the three models in
sequence with owned systemd services, fixed profiles, and resumable journals.
For a single Tiny server, replace the example checkpoint path:

```bash
VLLM_PLUGINS=ming_nvfp4,vllm_omni_register_models \
MING_NVFP4_FUSED_INPUT=0 MING_NVFP4_FUSED_OUTPUT=0 \
serving/.venv/bin/python -m vllm.entrypoints.cli.main serve \
  /absolute/path/to/exported/tiny/checkpoint \
  --omni --deploy-config evaluation/profiles/tiny.yaml \
  --host 127.0.0.1 --port 8091 --init-timeout 1800 --stage-init-timeout 1800
```

Flash uses `evaluation/profiles/flash.yaml` and its own checkpoint. Run one server
at a time. The full-suite launcher also sets the serving environment's `bin`
directory on `PATH` for compiler helpers.

The optional fusion modules are retained with the measured source snapshot, but
are outside the evaluation path above. Their default `auto` mode may activate
on supported SM120 Tiny shapes. Set both flags explicitly to `0` when comparing
against the published local quality results. GPU scheduling and numerical
variation can still change greedy responses; a fixed seed does not guarantee
identical answers across runs.

## CPU checks and attribution

The repetition blocker is checked against an independent full-scan oracle,
including window expiration, repeated calls, and output rollbacks. Run it with
the evaluation or root development environment; GPU libraries are unnecessary:

```bash
evaluation/.venv/bin/python -m pytest -q -c serving/pytest.ini serving/tests
```

`ngram.py` adapts request-lifecycle handling from
[vLLM-Omni](https://github.com/inclusionAI/vllm-omni). Its original copyright and
Apache-2.0 notice are retained; see [the license](LICENSE-APACHE-2.0).

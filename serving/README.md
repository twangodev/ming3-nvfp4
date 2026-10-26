# Serving the exported checkpoints

Follow the [Docker setup](../docker/README.md) first. Serving uses a dedicated
CUDA 13 image; quantization uses CUDA 12.8 in a separate image. The measured
target is an NVIDIA RTX PRO 6000 Blackwell with approximately 96 GiB VRAM (SM120).
Flash NVFP4 fits on this GPU; its original BF16 weights do not. The profiles are
hardware-specific, not tested minimum-memory defaults.

## Run a server

From the repository root, with the checkpoint mounted at `/models/tiny-nvfp4`:

```bash
docker compose build serve
docker compose up -d serve
docker compose logs -f serve
curl http://127.0.0.1:8091/v1/models
```

The endpoint is published on host loopback only. Set `MING_CHECKPOINT` and
`MING_PROFILE` in `.env` for Flash or another checkpoint; the Flash profile is
`/opt/ming/evaluation/profiles/flash.yaml`. Reload exported settings as in the setup
guide, then run `docker compose up -d serve` to apply changes. Stop it with
`docker compose down` before launching a suite.
The [evaluation runner](../evaluation/README.md) owns its own inference processes
inside a detached container and needs no separately running `serve` container.

## Runtime identity

The lock pins vLLM 0.30.0, torch 2.13.0, and the inclusionAI vLLM-Omni revision.
The image enforces the three [measured runtime pins](measured-runtime.txt) after
the locked install: Triton 3.7.1, nvidia-cuda-nvrtc 13.0.88, and nvidia-cuda-runtime
13.0.96. These agree with the Linux resolution of the lockfile. The explicit
`--no-deps` check keeps the measured versions visible without re-resolving dependencies. Suite provenance records actual versions,
lockfile, Dockerfile, profile, and plugin fingerprints.

The plugin maps Ming's quantization exclusions through the audio wrapper so
retained BF16 attention tensors load correctly. It also supplies incremental
prompt-plus-output n-gram blocking for ASR. The image registers both
`ming_nvfp4` and `vllm_omni_register_models` and disables Tiny FP4 input/output
fusion kernels, matching the quality runs. Optional fusion modules remain in
source but are outside this evaluation path.

Existing full-suite results were produced before the Docker migration. A new
image does not establish full numerical parity: scheduling and GPU variation
can change greedy responses even with a fixed seed. Use fresh run directories
and the pinned protocol when comparing new measurements.

## CPU checks and attribution

The repetition blocker is checked against an independent full-scan oracle,
including window expiration, repeated calls, and output rollbacks. Run the
[container test commands](../docker/README.md#images-and-jobs) to check it.

`ngram.py` adapts request-lifecycle handling from
[vLLM-Omni](https://github.com/inclusionAI/vllm-omni). Its original copyright and
Apache-2.0 notice are retained; see [the license](LICENSE-APACHE-2.0).

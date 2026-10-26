<div align="center">
  <img src="https://huggingface.co/twangodev/Ming-3.0-audio-tiny-NVFP4/resolve/main/assets/ming-3.0-audio-logo.png" width="110" alt="Ming logo">
  <h1>Ming 3.0 Audio NVFP4</h1>
  <p>Audio understanding and reasoning, quantized for NVIDIA Blackwell.</p>
  <p>
    <a href="https://huggingface.co/twangodev/Ming-3.0-audio-tiny-NVFP4"><img src="https://img.shields.io/badge/Tiny-NVFP4-FFD21E?style=flat-square&amp;logo=huggingface" alt="Tiny NVFP4 on Hugging Face"></a>
    <a href="https://huggingface.co/twangodev/Ming-3.0-audio-flash-NVFP4"><img src="https://img.shields.io/badge/Flash-NVFP4-FFD21E?style=flat-square&amp;logo=huggingface" alt="Flash NVFP4 on Hugging Face"></a>
    <a href="LICENSE"><img src="https://img.shields.io/github/license/twangodev/ming3-nvfp4?style=flat-square" alt="License"></a>
  </p>
  <p><a href="docs/reproduction.md">Reproduce</a> · <a href="serving/README.md">Serve</a> · <a href="evaluation/README.md">Evaluate</a> · <a href="docs/results/full-suite-parity.md">Results</a></p>
</div>

Community quantizations of inclusionAI's Ming 3.0 Audio Tiny and Flash using
**4-bit weights and activations (W4A4)** with
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).
This repository contains the calibration, export, audit, and evaluation workflows.

## Models

| Checkpoint | Routed-expert precision | Recipe |
| --- | --- | --- |
| [Tiny NVFP4](https://huggingface.co/twangodev/Ming-3.0-audio-tiny-NVFP4) | 16 layers NVFP4, 7 layers BF16 | [Tiny](recipes/tiny.toml) |
| [Flash NVFP4](https://huggingface.co/twangodev/Ming-3.0-audio-flash-NVFP4) | All 40 layers NVFP4 | [Flash](recipes/flash.toml) |

<details>
<summary>Quantization details</summary>

Both recipes use MSE calibration and quantize routed-expert projections only.
Attention, shared experts, routers, and audio components retain source precision;
KV and recurrent caches are unchanged. Tiny retains selected layers in BF16;
Flash uses Model Optimizer's pooled input-scale policy. Exact source revisions
and settings are pinned in the recipes.

</details>

## Reproduce

Follow the [reproduction guide](docs/reproduction.md) to prepare calibration data,
quantize, export, and audit a checkpoint. The [serving setup](serving/README.md)
and [evaluation guide](evaluation/README.md) cover validation with pinned dependencies
and upstream benchmarks. Weights, datasets, and run artifacts stay outside Git.

## Results

[Full results and parity investigation](docs/results/full-suite-parity.md) cover
**eight benchmarks and 26,183 examples per model**. Tiny includes a local BF16
comparison; Flash has no local BF16 baseline. Normalization, scoring, and comparison
limits are documented in expandable notes. These local measurements are not exact
reproductions of Ming's published evaluation.

Base models by inclusionAI: [Tiny](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny)
and [Flash](https://huggingface.co/inclusionAI/Ming-3.0-audio-flash).

Original project code is [MIT licensed](LICENSE). Third-party components retain
their existing licenses, including the [Apache-2.0 vLLM-Omni adaptation](serving/LICENSE-APACHE-2.0).

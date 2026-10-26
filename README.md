# Ming 3.0 NVFP4

Reproducible quantization of Ming 3.0 Audio Tiny and Flash with NVIDIA Model
Optimizer. This repository prepares calibration data, measures expert coverage,
calibrates routed experts, exports packed checkpoints, and audits retained tensors.
Weights, datasets, calibration state, and exported checkpoints stay outside Git.

The final recipes use **W4A4 NVFP4 with MSE calibration**. They quantize routed
expert `gate_proj`, `up_proj`, and `down_proj` only. Attention, shared experts,
routers, embeddings, output heads, audio components, Talker/VAE, and MTP retain
source precision. FP32 `A_log`, `dt_bias`, and `expert_bias` are preserved.
No recipe quantizes KV or recurrent caches.

| Recipe | Routed experts | Source revision |
| --- | --- | --- |
| [Tiny](recipes/tiny.toml) | 16 layers NVFP4; layers 5, 10, 15, 18, 20, 21, 22 BF16 | `3c9578f641b6745abf2d1239dc8419726f37c0e2` |
| [Flash pooled inputs](recipes/flash.toml) | All 40 routed layers NVFP4; Model Optimizer pooled input-scale policy | `77e851a4675a93181c30f50426507d076cc55a1b` |

See [reproduction](docs/reproduction.md) for setup, data preparation, explicit local
paths, calibration/export commands, and resource assumptions. Source weights are
never downloaded automatically or overwritten.

[Aggregate quality evidence](docs/results/README.md) contains eight tasks per
model. Tiny has paired BF16/NVFP4 post-selection results; Flash reports absolute
native NVFP4 quality without a matched BF16 comparison. These adapted local evaluations do not establish an untouched
holdout, published benchmark reproduction, or family-wide quality.

Model cards and downloadable checkpoints:
[Tiny NVFP4](https://huggingface.co/twangodev/Ming-3.0-audio-tiny-NVFP4),
[Flash NVFP4](https://huggingface.co/twangodev/Ming-3.0-audio-flash-NVFP4).
Upstream sources:
[Ming Tiny](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny),
[Ming Flash](https://huggingface.co/inclusionAI/Ming-3.0-audio-flash),
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

# Full evaluation and parity investigation

All three model runs completed all **26,183 examples across eight benchmarks**, for **78,549 predictions**, with no request errors. The original predictions and unchanged upstream scorer outputs are preserved.

The large apparent Chinese FLEURS and audio multiple-choice gaps are strongly affected by normalization and answer parsing. These diagnostics explain much of the discrepancy; they do not establish Ming's unpublished evaluation recipe.

The table below uses the explicitly described diagnostic policies: Chinese text normalization with character-level scoring; unambiguous multiple-choice answer parsing with all examples retained and capped/unparsed outputs wrong; and MMLU-Pro's official answer extraction with unparsed outputs wrong. English ASR is unchanged. All values are percentages. Published values are context, not a matched-protocol baseline.

| Benchmark | Examples | Tiny published | Tiny BF16 | Tiny NVFP4 | Flash published | Flash NVFP4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLEURS English WER ↓ | 647 | 3.30 | 3.29 | 3.44 | 2.86 | 2.92 |
| FLEURS Chinese CER ↓ | 945 | 2.99 | 3.52 | 3.43 | 2.92 | 3.45 |
| LibriSpeech clean WER ↓ | 2,620 | 1.42 | 1.42 | 1.44 | 1.20 | 1.20 |
| LibriSpeech other WER ↓ | 2,939 | 3.06 | 3.04 | 3.07 | 2.40 | 2.45 |
| MMAU test-mini accuracy ↑ | 1,000 | 75.90 | 76.70 | 77.30 | 82.00 | 78.20 |
| MMAR accuracy ↑ | 1,000 | 67.70 | 63.90 | 65.20 | 74.40 | 69.50 |
| MMSU accuracy ↑ | 5,000 | 73.06 | 68.04 | 67.14 | 78.82 | 77.80 |
| MMLU-Pro accuracy ↑ | 12,032 | — | 61.43 | 60.16 | 80.00 | 75.24 |

[Machine-readable table](full-suite.csv). Tiny has a local paired BF16/NVFP4 comparison. Flash has no local BF16 run; published Flash values cannot isolate quantization effects. These datasets overlap prior local development, so this is not an untouched holdout.

<details>
<summary>Why Chinese FLEURS differed</summary>

English already used Whisper's number normalization. The strict Chinese recipe removed punctuation but retained digit/spelled-number differences and parenthetical Latin text. For example, `50` and `五十` counted as different strings. Some references also contain English expansions that the predictions omit.

The independent [UltraEval-Audio normalization pipeline](https://github.com/OpenBMB/UltraEval-Audio/blob/main/audio_evals/lib/wer.py) uses raw FLEURS transcripts, its modified Whisper normalizer, and SpeechIO Chinese number normalization. Applying its text normalization to the same saved predictions gives:

| Scoring policy | Tiny BF16 | Tiny NVFP4 | Flash NVFP4 |
| --- | ---: | ---: | ---: |
| Original strict character score | 8.54 | 8.45 | 8.10 |
| UltraEval text normalization, character score | 3.52 | 3.43 | 3.45 |
| Unmodified UltraEval mixed-token score | 2.83 | 2.79 | 2.72 |

UltraEval's default tokenizer counts Chinese characters but groups Latin words. Its last row is **not the same metric as pure CER** and is not substituted into the main table. Normalization improved 245 Tiny BF16 recordings, left 683 unchanged, and worsened 17 by edit count. The [OWSM v4 paper](https://www.isca-archive.org/interspeech_2025/peng25c_interspeech.pdf) also documents inconsistent reading of FLEURS parenthetical text, so blanket deletion is not universally correct.

The [sibling Ming ASR evaluator](https://github.com/inclusionAI/Ming-3.0-asr/blob/main/tools/evaluate_asr.py) gave 8.54% for Tiny BF16 and 8.45% for Tiny NVFP4 on these predictions. Its generic normalization does not supply the missing Audio Tiny/Flash FLEURS recipe.

</details>

<details>
<summary>Multiple-choice parsing and original scorer outputs</summary>

MMAU/MMAR's official string matchers expect option text; many responses contain only a letter or put their final letter after an explanation. The diagnostic adapter resolves clear selections using only the response and offered options, then checks the gold. Capped or unparseable responses remain wrong.

MMSU's official parser accepts the first character when it is A–D. Consequently it can interpret `Based on...` as answer B, regardless of the explicit final answer, and excludes other nonempty responses entirely. The diagnostic score retains all 5,000 questions.

These are the **unchanged scorer outputs**, retained for reproducibility rather than treated as reliable final-answer accuracy when the response format is incompatible:

| Benchmark | Tiny BF16 | Tiny NVFP4 | Flash NVFP4 |
| --- | ---: | ---: | ---: |
| FLEURS English WER ↓ | 3.29 | 3.44 | 2.92 |
| FLEURS Chinese CER ↓ | 8.54 | 8.45 | 8.10 |
| LibriSpeech clean WER ↓ | 1.42 | 1.44 | 1.20 |
| LibriSpeech other WER ↓ | 3.04 | 3.07 | 2.45 |
| MMAU test-mini accuracy ↑ | 7.30 | 8.70 | 17.40 |
| MMAR accuracy ↑ | 9.00 | 10.50 | 18.40 |
| MMSU accuracy ↑ | 61.08 | 60.11 | 80.87 |
| MMLU-Pro accuracy ↑ | 61.50 | 60.26 | 75.24 |

MMSU original scored denominators: tiny-bf16: 3,844/5,000; tiny-nvfp4: 3,808/5,000; flash-nvfp4: 2,640/5,000. MMLU-Pro's original aggregation randomly guesses on unparsed answers; the main table reports them as wrong instead.

Feeding the resolved option text into the original MMAU/MMAR matchers gives the following alternative scores. Those matchers ignore word order and can accept a wrong option such as `From near to far` for `From far to near`. The main table therefore uses the selected option's exact value.

| Adapted answer + original matcher | Tiny BF16 | Tiny NVFP4 | Flash NVFP4 |
| --- | ---: | ---: | ---: |
| MMAU | 76.80 | 77.50 | 78.30 |
| MMAR | 64.30 | 65.50 | 70.20 |

</details>

<details>
<summary>Generation limits and remaining uncertainty</summary>

The full suite used greedy decoding, disabled internal thinking, 512-token ASR outputs, 1,024-token audio multiple-choice outputs, and the upstream MMLU-Pro 4,000-token budget. ASR used 10-gram blocking. No ASR recording was empty or truncated.

| Benchmark | Tiny BF16 caps | Tiny NVFP4 caps | Flash NVFP4 caps |
| --- | ---: | ---: | ---: |
| MMAU test-mini accuracy ↑ | 19 | 16 | 50 |
| MMAR accuracy ↑ | 22 | 17 | 34 |
| MMSU accuracy ↑ | 97 | 74 | 131 |
| MMLU-Pro accuracy ↑ | 1217 | 1077 | 328 |

Even making every currently wrong capped Flash MMLU-Pro response correct would only raise accuracy to **77.23%**, below the published 80%. Selecting the last explicit answer instead of the original first-match rule gives 75.27%, also essentially unchanged. Neither alone explains this remaining gap.

Long reasoning and generation limits can affect the remaining accuracy gaps. The exact published dataset/reference processing, prompts, answer extraction, and generation settings were not found in the inspected public Ming repositories. A lower diagnostic score or proximity to the published number is not proof of an exact reproduction.

</details>

<details>
<summary>Independent Hugging Face runtime control</summary>

Reference status: **complete**, 1,784/1,784 planned examples. The same original Tiny checkpoint was run serially through Hugging Face with matching chat prompts and generation budgets, versus concurrent vLLM serving. Original FP32 recurrent/router parameters were retained, and derived rotary buffers were rebuilt from their source initializer to avoid the known meta-loading issue.

| Benchmark | Matched examples | Exact output matches | HF score | Serving score |
| --- | ---: | ---: | ---: | ---: |
| FLEURS English WER ↓ | 647 | 612 | 3.38 | 3.29 |
| FLEURS Chinese CER ↓ | 945 | 918 | 8.54 | 8.54 |
| MMAR accuracy ↑ | 64 | 47 | 67.19 | 70.31 |
| MMAU test-mini accuracy ↑ | 64 | 52 | 68.75 | 68.75 |
| MMSU accuracy ↑ | 64 | 49 | 76.56 | 76.56 |

Chinese in this control table uses the original strict character scorer. All 1,784 prompt-token counts matched. The identical 8.54% aggregate on all 945 Chinese recordings shows that the large strict-scoring gap also occurs with the original, unquantized checkpoint in its Hugging Face implementation. The audio multiple-choice controls are subsets and are not substitutes for full-split results. This comparison changes backend and batch schedule together.

</details>

<details>
<summary>Flash output-budget experiment</summary>

A separate experiment raised the audio multiple-choice output budget from 1,024 to 4,096 tokens on every previously capped Flash answer, plus 32 fixed uncapped controls per task. All other request settings and the server profile were preserved. The batch schedule changed, so the controls also measure variation between runs. These outputs are **not** substituted into the full-suite scores above.

| Task | Previously capped | Now correct | Still capped | Control accuracy before/after | Identical control outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMAU | 50 | 14 | 15 | 25/32 → 25/32 | 9/32 |
| MMAR | 34 | 16 | 6 | 23/32 → 24/32 | 10/32 |
| MMSU | 131 | 66 | 14 | 25/32 → 25/32 | 16/32 |

Of the 215 previously capped cases, 82 stopped within the original 1,024-token budget in this run. Extra tokens alone cannot explain those changes.

A further replay used the **original 1,024-token budget** on exactly the same 311 selected recordings. Both separate runs completed without request errors.

| Task | Previously capped | Correct at 1,024 / 4,096 | Still capped at 1,024 / 4,096 | Control correct: original / replay / larger budget |
| --- | ---: | ---: | ---: | ---: |
| MMAU | 50 | 8 / 14 | 38 / 15 | 25 / 25 / 25 of 32 |
| MMAR | 34 | 14 / 16 | 13 / 6 | 23 / 22 / 24 of 32 |
| MMSU | 131 | 28 / 66 | 72 / 14 | 25 / 27 / 25 of 32 |

At the unchanged budget, 92/215 previously capped responses now stopped normally and 50 scored correctly, versus 96 correct with the larger budget. This demonstrates repeat-run variation under the changed workload schedule even with greedy decoding and the same seed. It does not isolate the numerical mechanism or prove that every recovered answer was caused by the larger budget. A future comparison should predeclare a sufficient budget and rerun each complete split; selected retries must not replace only failed answers.


</details>

Reproduction: [suite protocol](../../evaluation/README.md), [diagnostic scripts](../../evaluation/diagnostics/README.md). Local full artifacts are under `artifacts/evaluation/standard-v1-run/`; reference artifacts are under `artifacts/evaluation/parity-hf-tiny-chat/`.

Published sources: [ASR table](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/blob/main/assets/benchmark-asr.png), [audio understanding table](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/blob/main/assets/benchmark-audio-intelligence.png), [text table](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/blob/main/assets/benchmark-text.png).

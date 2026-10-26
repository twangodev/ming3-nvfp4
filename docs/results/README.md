# Aggregate quality evidence

[Full-suite results and parity investigation](full-suite-parity.md) contains the
new complete eight-benchmark runs and their separate scoring/runtime diagnostics.
The tables below remain the historical post-selection results.

These compact historical tables contain the final selected checkpoints' primary
quality results, with 25,159 examples per model across eight tasks. They were
copied byte-for-byte from the staged model-local evaluation bundles and checked
against their recorded SHA-256 hashes. [provenance.json](provenance.json) records
source revisions, report hashes, table hashes, and public methods links. No new
inference was performed for this repository cleanup.

- [Tiny paired quality](tiny-paired-quality.csv): BF16 and hybrid NVFP4 percentages,
  paired changes, conservative MC intervals, ASR bootstrap intervals, caps, and
  invalid formats. Positive changes improve accuracy but worsen WER/CER.
- [Flash native quality](flash-native-quality.csv): absolute NVFP4 percentages,
  caps, and invalid formats. It has no BF16 retention or speedup estimate.

## Local evaluation protocol and limitations

The protocol adapts FLEURS English/Chinese test, LibriSpeech test-clean/test-other,
MMAR, public MMAU test-mini, MMSU, and MMLU-Pro test (five-shot demonstrations).
Greedy output caps are 512 ASR, 1,024 audio MC, and 2,048 MMLU-Pro tokens. It uses
10-gram blocking; MMLU-Pro disables Ming internal thinking. Primary MC scoring
retains every example and counts capped/malformed answers wrong. ASR reports
corpus WER/CER. Scores are percentages; deltas and intervals are percentage points.

Selection used inspected outcomes, including this partition. These descriptive
post-selection results are neither an untouched holdout nor equivalence proof.
Tiny reuses a historical native BF16 reference on a separate serving schedule;
schedule-dependent responses and reference reuse prevent isolating quantization
as the sole cause. Tiny's MC intervals use exact discordance-marginal
Clopper-Pearson differences; ASR uses 20,000 paired bootstrap replicates (seed
20261025), with speaker clusters for LibriSpeech and utterances for FLEURS because
speaker IDs were unavailable. Multiplicity corrections do not repair selection
bias. Flash's primary MMSU score includes 2,317 invalid-format answers; no post-hoc
alternative scorer replaces it here.

Full protocol descriptions:
[Tiny methods](https://huggingface.co/twangodev/Ming-3.0-audio-tiny-NVFP4/blob/main/evals/methods.md),
[Flash methods](https://huggingface.co/twangodev/Ming-3.0-audio-flash-NVFP4/blob/main/evals/methods.md).
These reports cover local adapted tasks, not Ming's complete capabilities. No
weights, datasets, predictions, traces, or internal reviews are included.

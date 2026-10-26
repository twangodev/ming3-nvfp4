# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental n-gram blocking; request lifecycle adapted from vLLM-Omni."""
from vllm.v1.sample.logits_processor.interface import LogitsProcessor, MoveDirectionality

from ._ngram import NGramState


class IncrementalNoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, prompt_ids, output_ids in batch_update.added:
            extras = params.extra_args or {}
            self.requests[index] = (NGramState(
                prompt_ids or [], int(extras.get("no_repeat_ngram_size", 0)),
                int(extras.get("no_repeat_ngram_window", 8192)),
                extras.get("no_repeat_ngram_whitelist", [])), output_ids)
        for source, target, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                self.requests[source], self.requests[target] = (
                    self.requests[target], self.requests[source])
            else:
                self.requests[target] = self.requests.pop(source)

    def apply(self, logits):
        for index, (state, output) in self.requests.items():
            banned = state.banned(output)
            if banned:
                logits[index, list(banned)] = -float("inf")
        return logits

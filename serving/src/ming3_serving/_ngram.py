"""Incremental equivalent of vLLM-Omni's prompt-plus-output n-gram scan."""
from collections import Counter, deque


class NGramState:
    """Track append-only generated tokens; rebuild after an observed rollback.

    vLLM keeps the prompt fixed and updates the same output list in place. Its
    async placeholders are resolved before logits processors run. Shrinking the
    output or replacing its previously observed final token invalidates state.
    """

    def __init__(self, prompt, n, window, whitelist=()):
        self.prompt = tuple(prompt)
        self.n = n
        self.window = window
        self.whitelist = frozenset(whitelist)
        self.output_length = 0
        self.last_output_token = None
        self.length = 0
        self.entries = deque()
        self.counts = {}

    def _token(self, output, index):
        return (self.prompt[index] if index < len(self.prompt)
                else output[index - len(self.prompt)])

    def _prefix(self, output, start, size):
        return tuple(self._token(output, i) for i in range(start, start + size))

    def banned(self, output):
        if self.n <= 0 or self.window <= 0:
            return set()
        if (len(output) < self.output_length
                or (self.output_length
                    and output[self.output_length - 1] != self.last_output_token)):
            self.length = 0
            self.entries.clear()
            self.counts.clear()
        target_length = len(self.prompt) + len(output)
        # On initialization only n-grams still inside the current window matter.
        first_end = max(self.length + 1, self.n, target_length - self.window + self.n)
        for end in range(first_end, target_length + 1):
            start = end - self.n
            prefix = self._prefix(output, start, self.n - 1)
            token = self._token(output, end - 1)
            self.entries.append((start, prefix, token))
            self.counts.setdefault(prefix, Counter())[token] += 1
        minimum = max(0, target_length - self.window)
        while self.entries and self.entries[0][0] < minimum:
            _, prefix, token = self.entries.popleft()
            counts = self.counts[prefix]
            counts[token] -= 1
            if not counts[token]:
                del counts[token]
            if not counts:
                del self.counts[prefix]
        self.length = target_length
        self.output_length = len(output)
        self.last_output_token = output[-1] if output else None
        if target_length < self.n:
            return set()
        prefix = self._prefix(output, target_length - self.n + 1, self.n - 1)
        return set(self.counts.get(prefix, ())).difference(self.whitelist)

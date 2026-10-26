"""CPU oracle checks for the ASR prompt-plus-output repetition blocker."""
import random

import pytest

from ming3_serving._ngram import NGramState


def full_scan(prompt, output, n, window, whitelist):
    tokens = (prompt + output)[-window:]
    if len(tokens) < n:
        return set()
    prefix = tokens[-(n - 1):] if n > 1 else []
    return {tokens[i + n - 1] for i in range(len(tokens) - n + 1)
            if tokens[i:i + n - 1] == prefix} - set(whitelist)


@pytest.mark.parametrize("n,window", [(1, 4), (2, 16), (3, 16), (10, 64), (10, 4)])
@pytest.mark.parametrize("prompt", [[], [1, 2, 1, 2, 1], list(range(100))])
def test_incremental_matches_full_scan_with_rollbacks(n, window, prompt):
    randomizer = random.Random(42)
    whitelist = [2]
    state = NGramState(prompt, n, window, whitelist)
    output = []
    for step in range(250):
        if step % 17 == 0 and output:
            del output[-3:]
        elif step % 29 == 0 and output:
            output[-1] = (output[-1] + 1) % 4
        else:
            output.extend(randomizer.randrange(4) for _ in range(1 + step % 3))
        expected = full_scan(prompt, output, n, window, whitelist)
        assert state.banned(output) == expected
        assert state.banned(output) == expected


def test_disabled_runtime_does_not_import_gpu_dependencies(monkeypatch):
    import builtins
    from ming3_serving.fp4_runtime import install

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "triton", "vllm", "vllm_omni"}
        return original(name, *args, **kwargs)

    monkeypatch.setenv("MING_NVFP4_FUSED_INPUT", "0")
    monkeypatch.setenv("MING_NVFP4_FUSED_OUTPUT", "0")
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert install() is False

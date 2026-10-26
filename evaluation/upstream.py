"""Load pinned evaluator functions without importing unrelated model runtimes."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import types

ROOT = Path(__file__).resolve().parent


def module(name, relative):
    path = ROOT / "upstream" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(result)
    finally:
        sys.dont_write_bytecode = previous
    return result


def mmlu():
    # The upstream API driver imports five commercial SDKs at module scope.
    # Compile these exact, unmodified function ASTs; only transport is replaced.
    path = ROOT / "upstream/MMLU-Pro/evaluate_from_api.py"
    names = {"preprocess", "format_example", "extract_answer", "extract_again",
             "extract_final", "single_request", "update_result"}
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in tree.body} != names:
        raise RuntimeError("Pinned MMLU-Pro API functions changed")
    result = types.ModuleType("official_mmlu")
    result.__dict__.update(re=re, json=json, os=os, random=random.Random(12345), time=time)
    exec(compile(tree, str(path), "exec"), result.__dict__)
    return result


def mmlu_prompt(official, example, demonstrations):
    # Execute upstream's complete prompt construction, including five-shot CoT.
    captured = []

    def capture(client, instruction, inputs):
        captured.append(instruction + inputs)
        return "The answer is (A)"  # Used only to finish the prompt builder.

    official.call_api = capture
    official.single_request(None, example, demonstrations, [])
    if len(captured) != 1:
        raise RuntimeError("Could not capture the upstream MMLU-Pro prompt")
    return captured[0]


def english_normalizer():
    # Load the official package directly, avoiding Whisper's torch/model imports.
    normalizers = module("official_whisper_normalizers", "whisper/whisper/normalizers/__init__.py")
    return normalizers.EnglishTextNormalizer()

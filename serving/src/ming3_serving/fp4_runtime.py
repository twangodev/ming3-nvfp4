"""Compose optional Tiny input/output fusions in one guarded runtime patch.

Plugin registration invokes this installer. Both fusions remain opt-in;
disabled imports and installation require no GPU packages.
Previously installed flags are preserved; changing flags requires a new process
when disabling a fusion. No installed package files or global ops are modified.
"""
import ast
import inspect
from importlib import metadata
import os
import textwrap
import types

_INPUT = """
a = ops.shuffle_rows(a, a_map)
rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
    a, a1_gscale, expert_offsets, blockscale_offsets, num_topk,
)
"""
_OUTPUT = """
c3 = ops.shuffle_rows(c3, c_map)
assert output.dtype == out_dtype
if not apply_router_weight_on_input:
    output.copy_(
        (c3.view(m, num_topk, k)
         * topk_weights.view(m, num_topk, 1).to(out_dtype)).sum(dim=1),
        non_blocking=True,
    )
else:
    output.copy_(c3.view(m, num_topk, k).sum(dim=1), non_blocking=True)
return
"""


def _dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]))


def _code_shape(code):
    """Ignore source locations, while rejecting source/bytecode mismatches."""
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars,
            code.co_cellvars, code.co_argcount, code.co_kwonlyargcount,
            code.co_posonlyargcount,
            tuple(_code_shape(c) if isinstance(c, types.CodeType) else c
                  for c in code.co_consts))


def _transform(original, input_enabled, output_enabled):
    try:
        source = textwrap.dedent(inspect.getsource(original))
        tree = ast.parse(source)
        function = tree.body[0]
        if (len(tree.body) != 1 or not isinstance(function, ast.FunctionDef)
                or function.name != "run_cutlass_moe_fp4" or function.decorator_list):
            raise ValueError("unexpected function or decorators")
        # CPython 3.12 knows which globals are imported modules from the
        # surrounding module symbol table. That changes LOAD_GLOBAL/LOAD_ATTR
        # call flags even though an isolated function has identical semantics.
        # Compile the whole module (never execute it) to retain that context.
        module = inspect.getmodule(original)
        comparison_tree = tree
        if module is not None and getattr(module, "__file__", None):
            comparison_tree = ast.parse(inspect.getsource(module))
            definitions = [node for node in comparison_tree.body
                           if isinstance(node, ast.FunctionDef)
                           and node.name == function.name]
            if len(definitions) != 1 or _dump(definitions) != _dump([function]):
                raise ValueError("function source differs from module source")
        compiled = compile(comparison_tree, "<ming_fp4_source_check>", "exec")
        source_code = next(c for c in compiled.co_consts
                           if isinstance(c, types.CodeType)
                           and c.co_name == function.name)
        if _code_shape(source_code) != _code_shape(original.__code__):
            raise ValueError("function bytecode differs from inspected source")
        expected_input = _dump(ast.parse(textwrap.dedent(_INPUT)).body)
        input_positions = [i for i in range(len(function.body) - 1)
                           if _dump(function.body[i:i + 2]) == expected_input]
        if len(input_positions) != 1:
            raise ValueError("native input shuffle/quantization changed")
        if _dump(function.body[-4:]) != _dump(ast.parse(textwrap.dedent(_OUTPUT)).body):
            raise ValueError("native output combine changed")
    except (OSError, TypeError, SyntaxError, ValueError, StopIteration) as exc:
        raise RuntimeError(f"Ming NVFP4 fusion: pinned vLLM source guard failed: {exc}") from exc

    if output_enabled:
        function.body[-4:-4] = ast.parse("""
if _ming_try_fused_output(c3, c_map, topk_weights, output, apply_router_weight_on_input):
    return
""").body
    if input_enabled:
        replacement = ast.parse("""
_ming_input_result = _ming_try_gather_quant(
    a, a_map, a1_gscale, expert_offsets, blockscale_offsets, num_topk)
if _ming_input_result is None:
    a = ops.shuffle_rows(a, a_map)
    rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
        a, a1_gscale, expert_offsets, blockscale_offsets, num_topk)
else:
    rep_a_fp4, rep_a_blockscale = _ming_input_result
del _ming_input_result
""").body
        position = input_positions[0]
        function.body[position:position + 2] = replacement
    ast.fix_missing_locations(tree)
    return tree


def _mode(name):
    value = os.environ.get(name, "auto").strip().lower()
    if value not in ("auto", "0", "1"):
        raise ValueError(f"{name} must be auto, 0, or 1 (got {value!r})")
    return value


def _auto_supported():
    # Plugin registration also runs in the parent process: never initialize
    # CUDA here. Explicitly use NVML, never the non-NVML Torch fallback.
    try:
        if (metadata.version("torch").split("+", 1)[0] != "2.13.0"
                or metadata.version("vllm").split("+", 1)[0] != "0.30.0"):
            return False
        from vllm.platforms.cuda import NvmlCudaPlatform, pynvml
        try:
            return NvmlCudaPlatform.get_device_capability(0) == (12, 0)
        except (pynvml.NVMLError, RuntimeError, ValueError, IndexError):
            return False
    except (ImportError, RuntimeError, AssertionError):
        return False


def install():
    """Install supported defaults or explicit flags, preserving live flags.

    Both original sections must match even if only one fusion is requested.
    Auto failures retain native behavior; an explicit 1 makes source failures
    fatal. Explicitly setting both flags to 0 imports no GPU/runtime packages.
    Helpers retain the native path for tensors outside validated Tiny shapes.
    """
    modes = (_mode("MING_NVFP4_FUSED_INPUT"), _mode("MING_NVFP4_FUSED_OUTPUT"))
    if modes == ("0", "0"):
        return False
    auto = "auto" in modes and _auto_supported()
    requested = tuple(mode == "1" or (mode == "auto" and auto) for mode in modes)
    if not any(requested):
        return False
    strict = "1" in modes
    try:
        from vllm.model_executor.layers.fused_moe.experts import cutlass_moe
        current = cutlass_moe.run_cutlass_moe_fp4
    except (ImportError, AttributeError):
        if strict:
            raise
        return False
    if getattr(current, "_ming_fp4_runtime_installed", False):
        original = current._ming_fp4_runtime_original
        previous = current._ming_fp4_runtime_flags
    elif getattr(current, "_ming_fused_output_installed", False):
        # Explicitly recognized legacy installer; never inspect its generated
        # function, whose filename points at the unpatched source on disk.
        original = current._ming_fused_output_original
        previous = (False, True)
    else:
        original = current
        previous = (False, False)
    enabled = tuple(old or new for old, new in zip(previous, requested))
    if (getattr(current, "_ming_fp4_runtime_installed", False)
            and enabled == previous):
        return True
    try:
        tree = _transform(original, *enabled)
    except RuntimeError:
        if strict:
            raise
        return False
    namespace = dict(original.__globals__)
    if enabled[0]:
        from .fp4_input import try_gather_scaled_fp4_experts_quant
        namespace["_ming_try_gather_quant"] = try_gather_scaled_fp4_experts_quant
    if enabled[1]:
        from .fp4_output import try_fused_output
        namespace["_ming_try_fused_output"] = try_fused_output
    exec(compile(tree, "<ming_fp4_runtime>", "exec"), namespace)
    patched = namespace[original.__name__]
    patched.__defaults__ = original.__defaults__
    patched.__kwdefaults__ = original.__kwdefaults__
    patched.__module__ = original.__module__
    patched.__doc__ = original.__doc__
    patched._ming_fp4_runtime_installed = True
    patched._ming_fp4_runtime_original = original
    patched._ming_fp4_runtime_flags = enabled
    # Preserve compatibility with the earlier output installer, so invoking it
    # later cannot replace this composed function or inspect generated source.
    if enabled[1]:
        patched._ming_fused_output_installed = True
        patched._ming_fused_output_original = original
    cutlass_moe.run_cutlass_moe_fp4 = patched
    return True

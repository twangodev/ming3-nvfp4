"""Opt-in, in-memory fusion of the pinned Tiny NVFP4 output combine.

The original vLLM output path remains the fallback. Importing this module does
not import torch, Triton, or vLLM; disabled installation is safe on CPU hosts.
"""

import ast
import inspect
import os
import textwrap

_KERNEL = None


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def combine(C, Map, Weights, Out, H: tl.constexpr, BLOCK: tl.constexpr):
        token = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        # PyTorch Reduce.cuh uses four independent accumulators here, then
        # combines them left to right: ((p0+p4)+(p1+p5))+(p2+p6)+(p3+p7).
        result = tl.full((BLOCK,), 0, tl.float32)
        for route in tl.static_range(4):
            partial = tl.full((BLOCK,), 0, tl.float32)
            for offset in tl.static_range(2):
                r = route + offset * 4
                row = tl.load(Map + token * 8 + r)
                values = tl.load(C + row * H + col, mask=col < H,
                                 other=0).to(tl.float32)
                weight = tl.load(Weights + token * 8 + r).to(tl.bfloat16).to(tl.float32)
                # Both casts are observable in the original BF16 expression.
                product = (values * weight).to(tl.bfloat16).to(tl.float32)
                partial = partial + product
            result = result + partial
        tl.store(Out + token * H + col, result, mask=col < H)

    _KERNEL = combine
    return combine


def try_fused_output(c3, c_map, topk_weights, output,
                     apply_router_weight_on_input=False):
    """Launch for supported Tiny tensors, otherwise return False untouched.

    c_map is the gather map consumed by vLLM's shuffle_rows, mapping flattened
    token/route positions to expert-ordered c3 rows. All decisions use tensor
    metadata, so the call can run during CUDA graph capture.
    """
    import torch

    if apply_router_weight_on_input:
        return False
    if (c3.ndim != 2 or output.ndim != 2 or topk_weights.ndim != 2
            or c_map.ndim != 1):
        return False
    m, h = output.shape
    if (m == 0 or h != 1536 or c3.shape != (m * 8, h)
            or topk_weights.shape != (m, 8) or c_map.shape != (m * 8,)):
        return False
    if (c3.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
            or topk_weights.dtype != torch.float32 or c_map.dtype != torch.int32):
        return False
    tensors = (c3, c_map, topk_weights, output)
    if (not all(t.is_cuda and t.is_contiguous() for t in tensors)
            or not all(t.device == output.device for t in tensors)):
        return False
    # Validation covers the Blackwell SM120 implementation only.
    if torch.cuda.get_device_capability(output.device) != (12, 0):
        return False
    # The source workspace is live across the gather: overlapping writes race.
    if output.untyped_storage().data_ptr() == c3.untyped_storage().data_ptr():
        return False
    kernel = _get_kernel()
    kernel[(m, 6)](c3, c_map, topk_weights, output, H=h, BLOCK=256,
                   num_warps=4, enable_fp_fusion=False)
    return True


_EXPECTED_TAIL = """
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


def install():
    """Install once when MING_NVFP4_FUSED_OUTPUT=1; reject changed source."""
    if os.environ.get("MING_NVFP4_FUSED_OUTPUT") != "1":
        return False
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe

    original = cutlass_moe.run_cutlass_moe_fp4
    if getattr(original, "_ming_fused_output_installed", False):
        return True
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    expected = ast.parse(textwrap.dedent(_EXPECTED_TAIL)).body
    if (not isinstance(function, ast.FunctionDef) or function.decorator_list
            or ast.dump(ast.Module(body=function.body[-4:], type_ignores=[]))
            != ast.dump(ast.Module(body=expected, type_ignores=[]))):
        raise RuntimeError("MING_NVFP4_FUSED_OUTPUT: pinned vLLM output source changed")
    guard = ast.parse("""
if _ming_try_fused_output(c3, c_map, topk_weights, output, apply_router_weight_on_input):
    return
""").body
    function.body[-4:-4] = guard
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["_ming_try_fused_output"] = try_fused_output
    exec(compile(tree, inspect.getsourcefile(original) or "<ming_fp4_output>", "exec"),
         namespace)
    patched = namespace[original.__name__]
    patched.__defaults__ = original.__defaults__
    patched.__kwdefaults__ = original.__kwdefaults__
    patched.__module__ = original.__module__
    patched.__doc__ = original.__doc__
    patched._ming_fused_output_installed = True
    patched._ming_fused_output_original = original
    cutlass_moe.run_cutlass_moe_fp4 = patched
    return True

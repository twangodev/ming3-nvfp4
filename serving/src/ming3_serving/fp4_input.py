"""Experimental gather + NVFP4 expert quantization; never installed implicitly.

The caller must validate packed bytes and all initialized scale bytes against
vLLM's shuffle_rows + scaled_fp4_experts_quant before enabling this candidate.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[x],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _gather_quant(A, Map, GlobalScale, ExpertOffsets, ScaleOffsets, Q, S,
                  K: tl.constexpr, E: tl.constexpr, GROUPS: tl.constexpr,
                  EXPERT_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    source = tl.load(Map + row)
    experts = tl.arange(0, EXPERT_BLOCK)
    starts = tl.load(ExpertOffsets + experts, experts < E, other=2147483647)
    expert = tl.sum((starts <= row).to(tl.int32), 0) - 1
    local_row = row - tl.load(ExpertOffsets + expert)
    scale_base = tl.load(ScaleOffsets + expert) * (K // 64) * 4
    global_scale = tl.load(GlobalScale + expert)
    groups = tl.arange(0, GROUPS)
    cols = tl.arange(0, 16)
    offsets = groups[:, None] * 16 + cols[None, :]
    values = tl.load(A + source * K + offsets, offsets < K, other=0).to(tl.float32)
    maxima = tl.max(tl.abs(values), axis=1)
    sf = global_scale * (maxima * _rcp(6.0))
    # PTX operand 1 supplies the high byte; operand 2 the low byte.
    sf_pair = tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, 0f00000000, $1;",
        constraints="=h,f", args=[sf], dtype=tl.uint16, is_pure=True, pack=1,
    )
    sf_bits = sf_pair.to(tl.uint8)
    sf_rounded = sf_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    output_scale = tl.where(sf_rounded != 0., _rcp(sf_rounded * _rcp(global_scale)), 0.)
    scaled = values * output_scale[:, None]
    pair_cols = tl.broadcast_to(tl.arange(0, 8)[None, :], (GROUPS, 8))
    even = tl.gather(scaled, pair_cols * 2, axis=1)
    odd = tl.gather(scaled, pair_cols * 2 + 1, axis=1)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $2, $1; cvt.u32.u8 $0, b; }",
        constraints="=r,f,f", args=[even, odd], dtype=tl.uint32, is_pure=True, pack=1,
    ).to(tl.uint8)
    q_offsets = groups[:, None] * 8 + pair_cols
    tl.store(Q + row * (K // 2) + q_offsets, packed, q_offsets < K // 2)
    sf_offsets = (((local_row // 128) * (K // 64) + groups // 4) * 512
                  + (local_row % 32) * 16 + ((local_row // 32) % 4) * 4 + groups % 4)
    tl.store(S + scale_base + sf_offsets, sf_bits, groups < K // 16)


def gather_scaled_fp4_experts_quant(a, a_map, input_global_scale,
                                    expert_offsets, blockscale_offsets, topk):
    """Return native-shaped buffers for Tiny H=1536, E=128, topk=8.

    a_map is the actual get_cutlass_moe_mm_data destination-to-source map;
    replication is encoded directly by its repeated source row indices.
    Padding in the scale allocation is uninitialized, matching the native op.
    """
    from vllm import envs

    if (a.ndim != 2 or a.shape[1] != 1536 or topk != 8
            or expert_offsets.numel() != 129 or blockscale_offsets.numel() != 129):
        raise ValueError("candidate supports Tiny H=1536, E=128, topk=8 only")
    if a.dtype not in (torch.float16, torch.bfloat16) or not a.is_contiguous():
        raise ValueError("input must be contiguous FP16/BF16")
    if input_global_scale.dtype != torch.float32 or input_global_scale.numel() != 128:
        raise ValueError("preserve all 128 native per-expert FP32 calibration scales")
    tensors = (a, a_map, input_global_scale, expert_offsets, blockscale_offsets)
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        raise ValueError("all inputs must be contiguous and on the same device")
    if a_map.dtype != torch.int32 or expert_offsets.dtype != torch.int32 or blockscale_offsets.dtype != torch.int32:
        raise ValueError("native routing maps and offsets must be int32")
    if a_map.numel() != a.shape[0] * topk:
        raise ValueError("map must include all topk replicated rows")
    rows, k = a_map.numel(), a.shape[1]
    maximum = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk
    if rows > maximum:
        raise ValueError("native FP4 MoE row allocation limit exceeded")
    q = torch.empty((rows, k // 2), dtype=torch.uint8, device=a.device)
    scales = torch.empty((maximum, k // 16), dtype=torch.uint8, device=a.device)
    if rows:
        _gather_quant[(rows,)](a, a_map, input_global_scale, expert_offsets,
                               blockscale_offsets, q, scales, k, 128,
                               triton.next_power_of_2(k // 16), 128,
                               num_warps=4, enable_fp_fusion=False)
    return q, scales.view(torch.float8_e4m3fn)


def try_gather_scaled_fp4_experts_quant(a, a_map, input_global_scale,
                                        expert_offsets, blockscale_offsets, topk):
    """Return None for unsupported static metadata; preserve native fallback.

    Routing contents must come from the native get_cutlass_moe_mm_data op.
    No GPU errors are swallowed. The caller must retain the original native
    shuffle + quantization path when this function returns None.
    This helper does not register a serving patch.
    """
    if (not a.is_cuda or a.ndim != 2 or a.shape[1] != 1536 or topk != 8
            or a.dtype not in (torch.float16, torch.bfloat16)
            or a_map.ndim != 1 or a_map.numel() != a.shape[0] * topk
            or a_map.dtype != torch.int32
            or input_global_scale.dtype != torch.float32
            or input_global_scale.numel() != 128
            or expert_offsets.shape != (129,)
            or blockscale_offsets.shape != (129,)
            or expert_offsets.dtype != torch.int32
            or blockscale_offsets.dtype != torch.int32):
        return None
    tensors = (a, a_map, input_global_scale, expert_offsets, blockscale_offsets)
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        return None
    from vllm import envs
    if a_map.numel() > envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk:
        return None
    # This implementation is validated for the Blackwell SM120 path only.
    # Torch caches device properties; this metadata check launches no kernels.
    if torch.cuda.get_device_capability(a.device) != (12, 0):
        return None
    return gather_scaled_fp4_experts_quant(
        a, a_map, input_global_scale, expert_offsets, blockscale_offsets, topk)

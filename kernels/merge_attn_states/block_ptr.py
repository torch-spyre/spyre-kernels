# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Block-pointer conversion of merge_attn_states_kernel.
# Original: vllm/v1/attention/ops/triton_merge_attn_states.py
#
# Changes from original:
#   - The 1D loads/stores of head-dimension data (prefix_output, suffix_output,
#     output) use block pointers with boundary_check.
#   - Scalar LSE loads/stores remain as raw pointers.
#   - Simplified: no FP8 path, all tokens have prefix context.

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_attn_states_kernel_block_ptr(
    output,          # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,      # unused (signature compat with original)
    prefix_output,   # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,      # [NUM_HEADS, NUM_TOKENS]
    suffix_output,   # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,      # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    output_scale,    # unused
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    prefill_tokens_with_context: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """
    Merge prefix and suffix attention outputs — block-pointer version.

    Grid: (num_tokens, num_heads)
    """
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    # Scalar LSE loads — remain raw
    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    p_scale = p_se / out_se
    s_scale = s_se / out_se

    # Block pointers for head-dimension loads
    p_block_ptr = tl.make_block_ptr(
        base=prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride,
        shape=(HEAD_SIZE,),
        strides=(1,),
        offsets=(0,),
        block_shape=(PADDED_HEAD_SIZE,),
        order=(0,),
    )

    s_block_ptr = tl.make_block_ptr(
        base=suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride,
        shape=(HEAD_SIZE,),
        strides=(1,),
        offsets=(0,),
        block_shape=(PADDED_HEAD_SIZE,),
        order=(0,),
    )

    o_block_ptr = tl.make_block_ptr(
        base=output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride,
        shape=(HEAD_SIZE,),
        strides=(1,),
        offsets=(0,),
        block_shape=(PADDED_HEAD_SIZE,),
        order=(0,),
    )

    p_out = tl.load(p_block_ptr, boundary_check=(0,), padding_option="zero")
    s_out = tl.load(s_block_ptr, boundary_check=(0,), padding_option="zero")

    out = p_out * p_scale + s_out * s_scale
    out = out.to(output.dtype.element_ty)

    tl.store(o_block_ptr, out, boundary_check=(0,))

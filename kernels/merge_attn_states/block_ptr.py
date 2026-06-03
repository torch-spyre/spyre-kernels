# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of merge_attn_states_kernel.
# Original: vllm/v1/attention/ops/triton_merge_attn_states.py
#
# Changes from original:
#   - Head-dimension loads/stores use 3D tensor descriptors over the
#     (num_tokens, num_heads, head_size) tensors. Tensor descriptors
#     require >= 2 dims, and a 3D descriptor is a clean fit here.
#   - Scalar LSE loads stay raw.

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
    num_tokens,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    prefill_tokens_with_context: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """
    Merge prefix and suffix attention outputs — tensor-descriptor version.

    Grid: (num_tokens, num_heads)
    """
    token_idx = tl.program_id(0)
    num_heads = tl.num_programs(1)
    head_idx = tl.program_id(1)

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

    p_desc = tl.make_tensor_descriptor(
        prefix_output,
        shape=[num_tokens, num_heads, HEAD_SIZE],
        strides=[num_heads * prefix_head_stride, prefix_head_stride, 1],
        block_shape=[1, 1, PADDED_HEAD_SIZE],
    )
    s_desc = tl.make_tensor_descriptor(
        suffix_output,
        shape=[num_tokens, num_heads, HEAD_SIZE],
        strides=[num_heads * prefix_head_stride, prefix_head_stride, 1],
        block_shape=[1, 1, PADDED_HEAD_SIZE],
    )
    o_desc = tl.make_tensor_descriptor(
        output,
        shape=[num_tokens, num_heads, HEAD_SIZE],
        strides=[num_heads * output_head_stride, output_head_stride, 1],
        block_shape=[1, 1, PADDED_HEAD_SIZE],
    )

    p_out = p_desc.load([token_idx, head_idx, 0]).reshape([PADDED_HEAD_SIZE])
    s_out = s_desc.load([token_idx, head_idx, 0]).reshape([PADDED_HEAD_SIZE])

    out = p_out * p_scale + s_out * s_scale
    out = out.to(output.dtype.element_ty)

    o_desc.store([token_idx, head_idx, 0], out.reshape([1, 1, PADDED_HEAD_SIZE]))

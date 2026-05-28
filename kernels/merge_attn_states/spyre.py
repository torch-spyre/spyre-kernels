# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Spyre-aware conversion of merge_attn_states_kernel.
# Original: kernels/merge_attn_states/original.py
#
# Conversion from original:
#   - All pointer arithmetic replaced with tl.make_tensor_descriptor
#   - Grid capped at 32 cores with explicit distribution loop
#   - 2D grid (num_tokens, num_heads) flattened to 1D distribution
#   - LSE scalars stay as raw pointer loads (one element per token/head)
#   - Head-dimension vectors loaded via 2D descriptor over
#     [NUM_TOKENS * NUM_HEADS, HEAD_SIZE]
#   - Simplified: no FP8 path, all tokens assumed to have prefix context

import triton
import triton.language as tl


@triton.jit
def merge_attn_states_kernel_spyre(
    output_ptr,         # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_output_ptr,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse_ptr,     # [NUM_HEADS, NUM_TOKENS]
    suffix_output_ptr,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse_ptr,     # [NUM_HEADS, NUM_TOKENS]
    num_tokens,
    num_heads,
    head_size,
    BLOCK_HEAD: tl.constexpr,
):
    """Merge prefix and suffix attention outputs using log-sum-exp rescaling."""
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    # Treat output tensors as 2D: [NUM_TOKENS * NUM_HEADS, HEAD_SIZE]
    total_rows = num_tokens * num_heads

    prefix_out_desc = tl.make_tensor_descriptor(
        prefix_output_ptr,
        shape=[total_rows, head_size],
        strides=[head_size, 1],
        block_shape=[1, BLOCK_HEAD],
    )
    suffix_out_desc = tl.make_tensor_descriptor(
        suffix_output_ptr,
        shape=[total_rows, head_size],
        strides=[head_size, 1],
        block_shape=[1, BLOCK_HEAD],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[total_rows, head_size],
        strides=[head_size, 1],
        block_shape=[1, BLOCK_HEAD],
    )

    # Distribute (token, head) pairs across cores
    rows_per_core = tl.cdiv(total_rows, num_cores)
    row_start = pid * rows_per_core
    row_end = tl.minimum(row_start + rows_per_core, total_rows)

    head_tiles = tl.cdiv(head_size, BLOCK_HEAD)

    for row in range(row_start, row_end):
        token_idx = row // num_heads
        head_idx = row % num_heads

        # Scalar LSE loads — raw pointers (one element per token/head pair)
        p_lse = tl.load(prefix_lse_ptr + head_idx * num_tokens + token_idx).to(tl.float32)
        s_lse = tl.load(suffix_lse_ptr + head_idx * num_tokens + token_idx).to(tl.float32)

        # Handle FA2 inf → -inf
        p_lse = tl.where(p_lse == float("inf"), float("-inf"), p_lse)
        s_lse = tl.where(s_lse == float("inf"), float("-inf"), s_lse)

        # Log-sum-exp rescaling
        max_lse = tl.maximum(p_lse, s_lse)
        p_lse = p_lse - max_lse
        s_lse = s_lse - max_lse
        p_se = tl.exp(p_lse)
        s_se = tl.exp(s_lse)
        out_se = p_se + s_se
        p_scale = p_se / out_se
        s_scale = s_se / out_se

        # Merge head-dimension vectors tile by tile
        for c in range(head_tiles):
            p_out = prefix_out_desc.load([row, c * BLOCK_HEAD])
            s_out = suffix_out_desc.load([row, c * BLOCK_HEAD])
            p_out_f32 = p_out.to(tl.float32)
            s_out_f32 = s_out.to(tl.float32)
            out = p_out_f32 * p_scale + s_out_f32 * s_scale
            output_desc.store([row, c * BLOCK_HEAD], out.to(p_out.dtype))

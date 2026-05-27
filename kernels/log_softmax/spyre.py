# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Spyre-aware conversion of _topk_log_softmax_kernel.
# Original: kernels/log_softmax/original.py
#
# Conversion from original:
#   - Reduction loops over vocab use tl.make_tensor_descriptor
#   - Grid capped at 32 cores with explicit distribution loop over requests
#   - Top-k gather (data-dependent indirect load) remains raw pointer
#   - Top-k id load and output store remain raw pointer (indirect offsets)
#   - @triton.autotune not present in original (no change needed)

import triton
import triton.language as tl


@triton.jit
def _topk_log_softmax_kernel_spyre(
    output_ptr,
    logits_ptr,
    topk_ids_ptr,
    num_requests,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    """Log-softmax at top-k positions for each request row."""
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    requests_per_core = tl.cdiv(num_requests, num_cores)
    req_start = pid * requests_per_core
    req_end = tl.minimum(req_start + requests_per_core, num_requests)

    for req_idx in range(req_start, req_end):
        row_desc = tl.make_tensor_descriptor(
            logits_ptr + req_idx * vocab_size,
            shape=[vocab_size],
            strides=[1],
            block_shape=[BLOCK_SIZE],
        )

        # Pass 1: find row max
        num_blocks = tl.cdiv(vocab_size, BLOCK_SIZE)
        max_val = float("-inf")
        for i in range(num_blocks):
            logits = row_desc.load([i * BLOCK_SIZE])
            valid = (i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < vocab_size
            logits = tl.where(valid, logits, float("-inf"))
            max_val = tl.max(tl.maximum(logits, max_val))
        max_val = max_val.to(tl.float32)

        # Pass 2: compute sum(exp(logit - max))
        se = 0.0
        for i in range(num_blocks):
            logits = row_desc.load([i * BLOCK_SIZE])
            logits = logits.to(tl.float32)
            e = tl.exp(logits - max_val)
            valid = (i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < vocab_size
            e = tl.where(valid, e, 0.0)
            se += tl.sum(e)
        lse = tl.log(se)

        # Gather top-k logits — data-dependent indirect load, stays raw pointer
        k_offset = tl.arange(0, PADDED_TOPK)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0
        )
        row_ptr = logits_ptr + req_idx * vocab_size
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)

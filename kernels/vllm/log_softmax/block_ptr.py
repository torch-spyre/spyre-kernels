# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of _topk_log_softmax_kernel.
# Original: vllm/v1/worker/gpu/sample/logprob.py
#
# Changes from original:
#   - The two reduction loops over vocab use a 2D tensor descriptor over
#     the (num_requests, vocab_size) logits tensor. The descriptor is reused
#     across both passes — no need to "reset" like the block-pointer version.
#   - tl.advance is replaced by recomputing the column offset each iter.
#   - The top-k gather, top-k id load, and output store remain raw pointers
#     because they use data-dependent indirect offsets from topk_ids.
#   - tl.where masking is still needed in the reduction passes — descriptor
#     padding only zero-fills, but pass 1 needs -inf for OOB and pass 2 needs
#     0 (for the post-exp value), so we keep an explicit mask.

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_log_softmax_kernel_block_ptr(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    num_requests,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    """
    Log-softmax at top-k positions — tensor-descriptor version.

    Grid: (num_requests,)
    """
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    logits_desc = tl.make_tensor_descriptor(
        logits_ptr,
        shape=[num_requests, vocab_size],
        strides=[logits_stride, 1],
        block_shape=[1, BLOCK_SIZE],
    )

    # Pass 1: find max
    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        logits = logits_desc.load([req_idx, i]).reshape([BLOCK_SIZE])
        valid = (i + tl.arange(0, BLOCK_SIZE)) < vocab_size
        logits = tl.where(valid, logits, float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)

    # Pass 2: compute sum(exp(logit - max))
    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        logits = logits_desc.load([req_idx, i]).reshape([BLOCK_SIZE])
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        valid = (i + tl.arange(0, BLOCK_SIZE)) < vocab_size
        e = tl.where(valid, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    # Gather top-k logits — data-dependent indirect load, remains raw pointer
    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)
    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - max_val - lse
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)

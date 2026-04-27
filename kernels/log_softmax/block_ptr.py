# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Block-pointer conversion of _topk_log_softmax_kernel.
# Original: vllm/v1/worker/gpu/sample/logprob.py
#
# Changes from original:
#   - The two reduction loops over vocab use block pointers with tl.advance.
#   - The top-k gather (tl.load(row_ptr + topk_ids)) remains raw pointer
#     since it's a data-dependent indirect load (scatter/gather pattern).
#   - The top-k id load and output store also remain raw pointer since they
#     use data-dependent offsets from topk_ids.
#   - Pass 1 (max): uses padding_option="zero" then masks via tl.where
#     to get -inf semantics for OOB.
#   - Pass 2 (sum-exp): uses padding_option="zero", then masks exp results.

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
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    """
    Log-softmax at top-k positions — block-pointer version.

    Grid: (num_requests,)
    """
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    logits_block_ptr = tl.make_block_ptr(
        base=row_ptr,
        shape=(vocab_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Pass 1: find max
    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        logits = tl.load(logits_block_ptr, boundary_check=(0,), padding_option="zero")
        valid = (i + tl.arange(0, BLOCK_SIZE)) < vocab_size
        logits = tl.where(valid, logits, float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
        logits_block_ptr = tl.advance(logits_block_ptr, (BLOCK_SIZE,))
    max_val = max_val.to(tl.float32)

    # Reset block pointer for pass 2
    logits_block_ptr = tl.make_block_ptr(
        base=row_ptr,
        shape=(vocab_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Pass 2: compute sum(exp(logit - max))
    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        logits = tl.load(logits_block_ptr, boundary_check=(0,), padding_option="zero")
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        valid = (i + tl.arange(0, BLOCK_SIZE)) < vocab_size
        e = tl.where(valid, e, 0.0)
        se += tl.sum(e)
        logits_block_ptr = tl.advance(logits_block_ptr, (BLOCK_SIZE,))
    lse = tl.log(se)

    # Gather top-k logits — data-dependent indirect load, remains raw pointer
    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)
    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - max_val - lse
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)

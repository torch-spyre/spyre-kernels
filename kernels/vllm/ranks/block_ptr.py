# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of _ranks_kernel.
# Original: vllm/v1/worker/gpu/sample/logprob.py
#
# Changes from original:
#   - The main reduction loop uses a 2D tensor descriptor over the
#     (num_requests, vocab_size) logits tensor.
#   - Scalar loads (token_id, ref logit) and the scalar output store remain
#     as raw pointer ops since they are single-element data-dependent accesses.

import torch
import triton
import triton.language as tl


@triton.jit
def _ranks_kernel_block_ptr(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    num_requests,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Ranks kernel — tensor-descriptor version.

    Grid: (num_requests,)
    """
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    # Scalar loads — data-dependent, remain as raw pointers
    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    logits_desc = tl.make_tensor_descriptor(
        logits_ptr,
        shape=[num_requests, vocab_size],
        strides=[logits_stride, 1],
        block_shape=[1, BLOCK_SIZE],
    )

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        logits = logits_desc.load([req_idx, i]).reshape([BLOCK_SIZE])
        # Descriptor padding fills OOB with zero; mask to avoid spurious
        # counts when x <= 0.
        valid = (i + tl.arange(0, BLOCK_SIZE)) < vocab_size
        n += tl.sum((valid & (logits >= x)).to(tl.int32))

    tl.store(output_ptr + req_idx, n)

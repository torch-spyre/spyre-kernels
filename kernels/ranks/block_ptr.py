# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Block-pointer conversion of _ranks_kernel.
# Original: vllm/v1/worker/gpu/sample/logprob.py
#
# Changes from original:
#   - The main loop's tl.load(row_ptr + block, mask=...) is converted to
#     tl.load(block_ptr, boundary_check=...) with tl.advance per iteration.
#   - Scalar loads (token_id, ref logit) remain as raw pointer loads since
#     they are single-element data-dependent accesses that cannot be
#     expressed as block pointers.
#   - The scalar tl.store for output also remains raw since it's a single element.

import torch
import triton
import triton.language as tl


@triton.jit
def _ranks_kernel_block_ptr(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Ranks kernel with block pointers for the main loop.

    Grid: (num_requests,)
    """
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    # Scalar loads — data-dependent, remain as raw pointers
    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    logits_block_ptr = tl.make_block_ptr(
        base=row_ptr,
        shape=(vocab_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        logits = tl.load(logits_block_ptr, boundary_check=(0,), padding_option="zero")
        # Block-pointer padding_option only supports "zero" or "nan", not "-inf".
        # OOB zeros would falsely count when x <= 0, so we mask them out.
        valid = (i + tl.arange(0, BLOCK_SIZE)) < vocab_size
        n += tl.sum((valid & (logits >= x)).to(tl.int32))
        logits_block_ptr = tl.advance(logits_block_ptr, (BLOCK_SIZE,))

    tl.store(output_ptr + req_idx, n)

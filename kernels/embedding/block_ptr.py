# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Copyright 2024 LinkedIn Corporation (Liger-Kernel contributors)
#
# Block-pointer conversion of embedding_forward_kernel.
# Original: src/liger_kernel/ops/experimental/embedding.py
#
# Changes from original:
#   - The 1D load of `indices` becomes a block-pointer load over the M axis.
#   - The 2D contiguous store to `output_ptr` becomes a block-pointer store.
#   - The 2D gather of `embeddings` is data-dependent on `indices` (each row
#     of the output picks a different row from the embedding table), so it
#     stays as a raw-pointer masked load. This is the same pattern as the
#     scalar lookup in the ranks kernel, generalized to a vector per row.

import torch
import triton
import triton.language as tl


@triton.jit
def embedding_forward_kernel_block_ptr(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Embedding forward with block pointers for contiguous ops.

    Grid: (cdiv(n_elements, BLOCK_SIZE_M), cdiv(embedding_dim, BLOCK_SIZE_N))
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    indices_block_ptr = tl.make_block_ptr(
        base=indices_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(start_m,),
        block_shape=(BLOCK_SIZE_M,),
        order=(0,),
    )
    indices = tl.load(indices_block_ptr, boundary_check=(0,), padding_option="zero")

    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    # Data-dependent gather: each row of `embeddings` is chosen by `indices`.
    # Block pointers don't express indirect row selection, so this stays raw.
    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    embeddings = tl.load(
        embeddings_ptr + embedding_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(n_elements, embedding_dim),
        strides=(embedding_dim, 1),
        offsets=(start_m, start_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(output_block_ptr, embeddings, boundary_check=(0, 1))

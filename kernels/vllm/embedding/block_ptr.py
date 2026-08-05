# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Copyright 2024 LinkedIn Corporation (Liger-Kernel contributors)
#
# Tensor-descriptor conversion of embedding_forward_kernel.
# Original: src/liger_kernel/ops/experimental/embedding.py
#
# Changes from original:
#   - The 2D contiguous store to `output_ptr` becomes a tensor-descriptor store.
#   - The 1D load of `indices` stays as a raw masked load — tensor descriptors
#     require >= 2 dimensions, so a 1D index vector is not expressible.
#   - The 2D gather of `embeddings` is data-dependent on `indices` (each row
#     of the output picks a different row from the embedding table), so it
#     stays as a raw-pointer masked load.

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
    Embedding forward using a tensor descriptor for the contiguous output store.

    Grid: (cdiv(n_elements, BLOCK_SIZE_M), cdiv(embedding_dim, BLOCK_SIZE_N))
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

    # Data-dependent gather: each row of `embeddings` is chosen by `indices`.
    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    embeddings = tl.load(
        embeddings_ptr + embedding_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[n_elements, embedding_dim],
        strides=[embedding_dim, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    output_desc.store([start_m, start_n], embeddings)

# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Copyright 2024 LinkedIn Corporation (Liger-Kernel contributors)
#
# Spyre-aware conversion of embedding_forward_kernel.
# Original: kernels/embedding/original.py
#
# Conversion from original:
#   - Regular memory access replaced with tl.make_tensor_descriptor
#   - Grid capped at 32 cores with explicit distribution loop
#   - Embedding table access uses tl.descriptor_gather for indirect indexing
#   - Indices loaded via raw pointer (1D indirect access — descriptor requires
#     ≥16 bytes in last dimension, incompatible with scalar index loads)
#   - Removed masks for descriptor-managed accesses

import triton
import triton.language as tl


@triton.jit
def embedding_forward_kernel_spyre(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    vocab_size,
    embedding_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Embedding table lookup: output[i, :] = embeddings[indices[i], :]."""
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    embeddings_desc = tl.make_tensor_descriptor(
        embeddings_ptr,
        shape=[vocab_size, embedding_dim],
        strides=[embedding_dim, 1],
        block_shape=[1, BLOCK_SIZE_N],
    )

    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[n_elements, embedding_dim],
        strides=[embedding_dim, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    m_blocks = tl.cdiv(n_elements, BLOCK_SIZE_M)
    n_blocks = tl.cdiv(embedding_dim, BLOCK_SIZE_N)
    total_blocks = m_blocks * n_blocks
    blocks_per_core = tl.cdiv(total_blocks, num_cores)
    start = pid * blocks_per_core
    end = tl.minimum(start + blocks_per_core, total_blocks)

    for block_idx in range(start, end):
        m = block_idx // n_blocks
        n = block_idx % n_blocks

        m_offset = m * BLOCK_SIZE_M
        n_offset = n * BLOCK_SIZE_N

        offsets_m = m_offset + tl.arange(0, BLOCK_SIZE_M)
        indices_tile = tl.load(indices_ptr + offsets_m, mask=offsets_m < n_elements, other=0)
        indices_tile = indices_tile.to(tl.int32)

        emb_tile = embeddings_desc.gather(indices_tile, n_offset)

        output_desc.store([m_offset, n_offset], emb_tile)

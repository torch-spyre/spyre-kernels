# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2018-2020 Philippe Tillet, 2020-2022 OpenAI
#
# Spyre-aware conversion of matmul_kernel.
# Original: kernels/matmul/original.py
#
# Conversion from original:
#   - All pointer arithmetic replaced with tl.make_tensor_descriptor
#   - Grid capped at 32 cores with explicit distribution loop
#   - @triton.autotune removed; tile sizes are explicit constexpr params
#   - tl.assume calls removed
#   - GROUP_SIZE_M L2 tiling removed (not applicable to Spyre)
#   - Activation support preserved (ACTIVATION constexpr)

import triton
import triton.language as tl


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def matmul_kernel_spyre(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Compute C = A x B. A is [M, K], B is [K, N], C is [M, N]."""
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)

    total_blocks = m_blocks * n_blocks
    blocks_per_core = tl.cdiv(total_blocks, num_cores)
    block_start = pid * blocks_per_core
    block_end = tl.minimum(block_start + blocks_per_core, total_blocks)

    for block_idx in range(block_start, block_end):
        m = block_idx // n_blocks
        n = block_idx % n_blocks

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(k_tiles):
            a_tile = a_desc.load([m * BLOCK_M, k * BLOCK_K])
            b_tile = b_desc.load([k * BLOCK_K, n * BLOCK_N])
            acc = tl.dot(a_tile, b_tile, acc)

        if ACTIVATION == "leaky_relu":
            acc = leaky_relu(acc)

        c_desc.store([m * BLOCK_M, n * BLOCK_N], acc.to(tl.float16))

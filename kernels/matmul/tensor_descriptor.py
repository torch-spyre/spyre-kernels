# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2018-2020 Philippe Tillet, 2020-2022 OpenAI
#
# Tensor-descriptor conversion of matmul_kernel.
# Original: kernels/matmul/original.py
# Changes summarized in kernels/matmul/conversion-notes.md.

import triton
import triton.language as tl

from kernels.matmul.original import get_autotune_config


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel_td(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides: how much to increase a ptr by when moving 1 element in a dim.
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B using tensor descriptors.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N).
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # Grouped ordering promotes L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # One descriptor per tensor; shape carries the boundary, strides come
    # straight from the signature (no row-major assumption needed).
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    off_m = pid_m * BLOCK_SIZE_M
    off_n = pid_n * BLOCK_SIZE_N

    # -----------------------------------------------------------
    # Accumulate a [BLOCK_SIZE_M, BLOCK_SIZE_N] block in fp32 over the K dim.
    # The descriptor zero-fills the partial K tail tile, which is the additive
    # identity for the dot accumulation, so no K-dimension mask is needed.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        off_k = k * BLOCK_SIZE_K
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_k, off_n])
        accumulator = tl.dot(a, b, accumulator)

    # Fuse activation while the accumulator is still fp32.
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of C. The store clamps at shape=[M, N], so the tail
    # tile never writes past the matrix bounds — no output mask needed.
    c_desc.store([off_m, off_n], c)

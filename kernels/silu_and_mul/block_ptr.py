# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Block-pointer conversion of _swiglustep_and_mul_kernel.
# Original: vllm/model_executor/layers/activation.py
#
# Changes from original:
#   - All tl.load(ptr + offset, mask=...) replaced with
#     tl.load(block_ptr, boundary_check=...)
#   - All tl.store(ptr + offset, val, mask=...) replaced with
#     tl.store(block_ptr, val, boundary_check=...)
#   - Block pointers created via tl.make_block_ptr
#   - gate and up are accessed via two 1D block pointers into the same row,
#     offset by 0 and d respectively.

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglustep_and_mul_kernel_block_ptr(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    limit: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SwiGLU with clamping — block-pointer version.

    Input x is [n_rows, 2*d]. gate = x[:, :d], up = x[:, d:].
    Output o is [n_rows, d].
    Each program handles one row and one block of columns.
    """
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    col_offset = j * BLOCK_SIZE

    gate_block_ptr = tl.make_block_ptr(
        base=x_ptr + i * x_stride,
        shape=(d,),
        strides=(1,),
        offsets=(col_offset,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    up_block_ptr = tl.make_block_ptr(
        base=x_ptr + i * x_stride + d,
        shape=(d,),
        strides=(1,),
        offsets=(col_offset,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    out_block_ptr = tl.make_block_ptr(
        base=o_ptr + i * o_stride,
        shape=(d,),
        strides=(1,),
        offsets=(col_offset,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    gate = tl.load(gate_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    up = tl.load(up_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    gate_silu = tl.sigmoid(gate) * gate
    gate_clamped = tl.minimum(gate_silu, limit)
    up_clamped = tl.minimum(tl.maximum(up, -limit), limit)

    result = gate_clamped * up_clamped
    result = result.to(x_ptr.dtype.element_ty)
    tl.store(out_block_ptr, result, boundary_check=(0,))

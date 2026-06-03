# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of _swiglustep_and_mul_kernel.
# Original: vllm/model_executor/layers/activation.py
#
# Changes from original:
#   - tl.load/tl.store with pointer arithmetic + masks replaced with
#     tensor descriptors (tl.make_tensor_descriptor).
#   - Tensor descriptors require >= 2 dimensions, so the row index lives
#     inside the descriptor offsets rather than the base pointer. We pass
#     n_rows from the wrapper for this purpose.
#   - For the input we use a single descriptor over the (n_rows, 2*d) tensor
#     and select the gate/up halves via the column offset (0 vs d).

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglustep_and_mul_kernel_block_ptr(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    n_rows,
    limit: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SwiGLU with clamping — tensor-descriptor version.

    Input x is [n_rows, 2*d]. gate = x[:, :d], up = x[:, d:].
    Output o is [n_rows, d].
    Each program handles one row and one block of columns.
    """
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    col_offset = j * BLOCK_SIZE
    row = i.to(tl.int32)

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[n_rows, 2 * d],
        strides=[x_stride, 1],
        block_shape=[1, BLOCK_SIZE],
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr,
        shape=[n_rows, d],
        strides=[o_stride, 1],
        block_shape=[1, BLOCK_SIZE],
    )

    gate = x_desc.load([row, col_offset]).to(tl.float32)
    up = x_desc.load([row, d + col_offset]).to(tl.float32)

    gate_silu = tl.sigmoid(gate) * gate
    gate_clamped = tl.minimum(gate_silu, limit)
    up_clamped = tl.minimum(tl.maximum(up, -limit), limit)

    result = gate_clamped * up_clamped
    result = result.to(x_ptr.dtype.element_ty)
    o_desc.store([row, col_offset], result)

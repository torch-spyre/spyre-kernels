# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of _swiglustep_and_mul_kernel.
# Original: kernels/silu_and_mul/original.py
# Changes summarized in kernels/silu_and_mul/conversion-notes.md.

import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel_td(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    n_rows,
    limit: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """SwiGLU step-and-mul: out = clamp(silu(gate)) * clamp(up).

    The input row holds gate in columns [0, d) and up in columns [d, 2*d).
    Each program handles one (row, column-tile) pair, matching the original
    2D grid (n_rows, cdiv(d, BLOCK_SIZE)).
    """
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    col = j * BLOCK_SIZE

    # Input view spans both halves [n_rows, 2*d]; gate and up are read from
    # the same descriptor at column offsets `col` and `col + d`. The shape's
    # 2*d boundary zero-fills the partial column tail, replacing the mask.
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_rows, 2 * d], strides=[x_stride, 1],
        block_shape=[1, BLOCK_SIZE],
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[n_rows, d], strides=[o_stride, 1],
        block_shape=[1, BLOCK_SIZE],
    )

    gate = x_desc.load([i, col]).to(tl.float32)
    up = x_desc.load([i, col + d]).to(tl.float32)

    gate_silu = tl.sigmoid(gate) * gate
    gate_clamped = tl.minimum(gate_silu, limit)
    up_clamped = tl.minimum(tl.maximum(up, -limit), limit)

    result = gate_clamped * up_clamped
    result = result.to(x_ptr.dtype.element_ty)
    o_desc.store([i, col], result)

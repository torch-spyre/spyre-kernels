# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Spyre-aware conversion of _swiglustep_and_mul_kernel.
# Original: kernels/silu_and_mul/original.py
#
# Conversion from original:
#   - All pointer arithmetic replaced with tl.make_tensor_descriptor
#   - Grid capped at 32 cores with explicit distribution loop
#   - Single descriptor over full [n_rows, 2*d] input; gate/up accessed via column offset
#   - d changed from constexpr to runtime arg (problem size)
#   - limit changed from constexpr to runtime arg (float parameter)

import triton
import triton.language as tl


@triton.jit
def _swiglustep_and_mul_kernel_spyre(
    x_ptr,
    o_ptr,
    n_rows,
    d,
    limit,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute SiLU(gate) * up with clamping. Input x is [n_rows, 2*d], output is [n_rows, d]."""
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_rows, 2 * d], strides=[2 * d, 1],
        block_shape=[1, BLOCK_SIZE],
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[n_rows, d], strides=[d, 1],
        block_shape=[1, BLOCK_SIZE],
    )

    rows_per_core = tl.cdiv(n_rows, num_cores)
    row_start = pid * rows_per_core
    row_end = tl.minimum(row_start + rows_per_core, n_rows)

    col_tiles = tl.cdiv(d, BLOCK_SIZE)

    for row in range(row_start, row_end):
        for c in range(col_tiles):
            gate_raw = x_desc.load([row, c * BLOCK_SIZE])
            up_raw = x_desc.load([row, d + c * BLOCK_SIZE])
            gate = gate_raw.to(tl.float32)
            up = up_raw.to(tl.float32)

            gate_silu = tl.sigmoid(gate) * gate
            gate_clamped = tl.minimum(gate_silu, limit)
            up_clamped = tl.minimum(tl.maximum(up, -limit), limit)

            result = gate_clamped * up_clamped
            o_desc.store([row, c * BLOCK_SIZE], result.to(gate_raw.dtype))

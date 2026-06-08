# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of _rms_norm_kernel.
# Original: kernels/rms_norm/original.py
#
# Changes from original:
#   - All pointer arithmetic + masked loads/stores replaced with
#     tl.make_tensor_descriptor. The descriptor's block_shape handles
#     out-of-bounds columns (zero padding on load), removing the need
#     for explicit masks.
#   - Row batching: the grid size is decoupled from n_rows. Each program
#     processes a contiguous block of rows, evenly dividing n_rows across
#     the programs in the grid. One-program-per-row is the special case
#     where the grid has n_rows programs.

import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel_td(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute RMS normalization: y = x / sqrt(mean(x^2) + eps) * weight.

    Rows are batched: each program processes a contiguous block of rows,
    so the grid size is independent of n_rows. A grid of n_rows programs
    recovers one row per program.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    input_desc = tl.make_tensor_descriptor(
        input_ptr, shape=[n_rows, n_cols], strides=[n_cols, 1],
        block_shape=[1, BLOCK_SIZE],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_rows, n_cols], strides=[n_cols, 1],
        block_shape=[1, BLOCK_SIZE],
    )
    weight_desc = tl.make_tensor_descriptor(
        weight_ptr, shape=[1, n_cols], strides=[n_cols, 1],
        block_shape=[1, BLOCK_SIZE],
    )

    rows_per_program = tl.cdiv(n_rows, num_programs)
    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)

    col_tiles = tl.cdiv(n_cols, BLOCK_SIZE)

    for row in range(row_start, row_end):
        # Step 1: Compute sum of squares
        sum_sq = tl.zeros([1, BLOCK_SIZE], dtype=tl.float32)
        for c in range(col_tiles):
            vals = input_desc.load([row, c * BLOCK_SIZE])
            vals_f32 = vals.to(tl.float32)
            sum_sq += vals_f32 * vals_f32

        total_sq = tl.sum(sum_sq)

        # Step 2: Compute inverse RMS
        mean_sq = total_sq / n_cols
        inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

        # Step 3: Normalize and apply weight
        for c in range(col_tiles):
            vals = input_desc.load([row, c * BLOCK_SIZE])
            w = weight_desc.load([0, c * BLOCK_SIZE])
            vals_f32 = vals.to(tl.float32)
            w_f32 = w.to(tl.float32)
            out_f32 = vals_f32 * inv_rms * w_f32
            output_desc.store([row, c * BLOCK_SIZE], out_f32.to(vals.dtype))

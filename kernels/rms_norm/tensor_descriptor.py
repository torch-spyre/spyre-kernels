# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of _rms_norm_kernel.
# Original: kernels/rms_norm/original.py
#
# Changes from original:
#   - All pointer arithmetic + masked loads/stores replaced with
#     tl.make_tensor_descriptor. The descriptor's block_shape handles
#     out-of-bounds rows and columns (zero padding on load), removing the
#     need for explicit masks.
#   - input_row_stride / output_row_stride are kept as runtime args and
#     passed straight into the descriptor strides, preserving the original's
#     support for strided (non-contiguous) rows. Column stride is 1, matching
#     the original's contiguous row_start_ptr + col_idx indexing.
#   - Row batching: each program processes ROWS_PER_PROGRAM rows at once.
#     The rows are loaded together as a [ROWS_PER_PROGRAM, BLOCK_SIZE] tile
#     and reduced/normalized with vectorized ops over the row axis.
#     ROWS_PER_PROGRAM=1 recovers one program per row.
#
# NOTE: ROWS_PER_PROGRAM is used as the descriptor block_shape's row
# dimension, which tl.make_tensor_descriptor requires to be a power of 2.
# Pass only power-of-2 values.

import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel_td(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """Compute RMS normalization: y = x / sqrt(mean(x^2) + eps) * weight.

    Each program handles a contiguous block of ROWS_PER_PROGRAM rows,
    starting at pid * ROWS_PER_PROGRAM, so the grid has
    cdiv(n_rows, ROWS_PER_PROGRAM) programs. ROWS_PER_PROGRAM=1 recovers
    one row per program.

    ROWS_PER_PROGRAM must be a power of 2: it is the row dimension of the
    descriptor block_shape, which tl.make_tensor_descriptor requires to be
    a power of 2.
    """
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM

    input_desc = tl.make_tensor_descriptor(
        input_ptr, shape=[n_rows, n_cols], strides=[input_row_stride, 1],
        block_shape=[ROWS_PER_PROGRAM, BLOCK_SIZE],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_rows, n_cols], strides=[output_row_stride, 1],
        block_shape=[ROWS_PER_PROGRAM, BLOCK_SIZE],
    )
    weight_desc = tl.make_tensor_descriptor(
        weight_ptr, shape=[1, n_cols], strides=[n_cols, 1],
        block_shape=[1, BLOCK_SIZE],
    )

    col_tiles = tl.cdiv(n_cols, BLOCK_SIZE)

    # Step 1: Compute per-row sum of squares. Out-of-bounds rows (when
    # n_rows is not a multiple of ROWS_PER_PROGRAM) load as zero and so
    # contribute nothing to the reduction.
    sum_sq = tl.zeros([ROWS_PER_PROGRAM, BLOCK_SIZE], dtype=tl.float32)
    for c in range(col_tiles):
        vals = input_desc.load([row_start, c * BLOCK_SIZE])
        vals_f32 = vals.to(tl.float32)
        sum_sq += vals_f32 * vals_f32

    row_sum_sq = tl.sum(sum_sq, axis=1)  # [ROWS_PER_PROGRAM]

    # Step 2: Compute inverse RMS per row.
    mean_sq = row_sum_sq / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)  # [ROWS_PER_PROGRAM]
    inv_rms = inv_rms[:, None]  # [ROWS_PER_PROGRAM, 1] for broadcasting

    # Step 3: Normalize and apply weight.
    for c in range(col_tiles):
        vals = input_desc.load([row_start, c * BLOCK_SIZE])
        w = weight_desc.load([0, c * BLOCK_SIZE])  # [1, BLOCK_SIZE]
        vals_f32 = vals.to(tl.float32)
        w_f32 = w.to(tl.float32)
        out_f32 = vals_f32 * inv_rms * w_f32
        output_desc.store([row_start, c * BLOCK_SIZE], out_f32.to(vals.dtype))

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
#   - Grid is unchanged from the original: one program per row,
#     grid = (n_rows,).

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

    One program per row (grid = (n_rows,)), matching the original kernel.
    """
    row = tl.program_id(0)

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

    col_tiles = tl.cdiv(n_cols, BLOCK_SIZE)

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

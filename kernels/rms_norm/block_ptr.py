# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of _rms_norm_kernel.
# Original: vllm/model_executor/layers/batch_invariant.py
#
# Changes from original:
#   - tl.load/tl.store with pointer arithmetic + masks replaced with
#     tensor descriptors (tl.make_tensor_descriptor).
#   - Tensor descriptors require >= 2 dimensions. Input/output are
#     described as the full (n_rows, n_cols) tensor; weight is described
#     as a (1, n_cols) tensor.
#   - tl.advance is replaced by recomputing the column offset each iter.
#   - Out-of-bounds is handled by descriptor padding_option="zero" (default).

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel_block_ptr(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS Norm with tensor descriptors.
    y = x / sqrt(mean(x^2) + eps) * weight

    Each program handles one row.
    """
    row_idx = tl.program_id(0).to(tl.int32)

    in_desc = tl.make_tensor_descriptor(
        input_ptr,
        shape=[n_rows, n_cols],
        strides=[input_row_stride, 1],
        block_shape=[1, BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[n_rows, n_cols],
        strides=[output_row_stride, 1],
        block_shape=[1, BLOCK_SIZE],
    )
    w_desc = tl.make_tensor_descriptor(
        weight_ptr,
        shape=[1, n_cols],
        strides=[n_cols, 1],
        block_shape=[1, BLOCK_SIZE],
    )

    # Step 1: Compute sum of squares in float32
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        vals = in_desc.load([row_idx, col_offset])
        vals_f32 = vals.to(tl.float32)
        sum_sq += tl.sum(vals_f32 * vals_f32)

    # Step 2: Compute RMS
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        vals = in_desc.load([row_idx, col_offset])
        weight = w_desc.load([0, col_offset])

        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)

        out_desc.store([row_idx, col_offset], output)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Block-pointer conversion of _rms_norm_kernel.
# Original: vllm/model_executor/layers/batch_invariant.py
#
# Changes from original:
#   - All tl.load(ptr + offset, mask=...) replaced with
#     tl.load(block_ptr, boundary_check=...)
#   - All tl.store(ptr + offset, val, mask=...) replaced with
#     tl.store(block_ptr, val, boundary_check=...)
#   - Block pointers created via tl.make_block_ptr and advanced via tl.advance
#   - The "other=1.0" default for weight loads is handled by padding_option="one"

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
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS Norm with block pointers.
    y = x / sqrt(mean(x^2) + eps) * weight

    Each program handles one row. We create 2D block pointers into the
    [n_rows, n_cols] tensor and fix the row dimension to a block of size 1,
    sliding the column dimension in steps of BLOCK_SIZE.

    Note: We use 1D block pointers into each row. The row offset is computed
    from program_id and baked into the base pointer, then we have a 1D
    block pointer over columns.
    """
    row_idx = tl.program_id(0).to(tl.int64)

    # --- Block pointers for input row, output row, and weight vector ---
    # Input: 1D view into row `row_idx`, shape=(n_cols,), stride=(1,)
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr + row_idx * input_row_stride,
        shape=(n_cols,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Step 1: Compute sum of squares in float32
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(input_block_ptr, boundary_check=(0,), padding_option="zero")
        vals_f32 = vals.to(tl.float32)
        sum_sq += tl.sum(vals_f32 * vals_f32)
        input_block_ptr = tl.advance(input_block_ptr, (BLOCK_SIZE,))

    # Step 2: Compute RMS
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Reset input block pointer to start of row for the second pass
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr + row_idx * input_row_stride,
        shape=(n_cols,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Weight: 1D, shape=(n_cols,), stride=(1,)
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(n_cols,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Output: 1D view into row `row_idx`
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + row_idx * output_row_stride,
        shape=(n_cols,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(input_block_ptr, boundary_check=(0,), padding_option="zero")
        # Original used other=1.0 for OOB weights, but OOB input vals are 0.0
        # and the store has boundary_check, so OOB positions are never written.
        # Zero-padding the weight is safe here.
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)

        tl.store(output_block_ptr, output, boundary_check=(0,))

        input_block_ptr = tl.advance(input_block_ptr, (BLOCK_SIZE,))
        weight_block_ptr = tl.advance(weight_block_ptr, (BLOCK_SIZE,))
        output_block_ptr = tl.advance(output_block_ptr, (BLOCK_SIZE,))

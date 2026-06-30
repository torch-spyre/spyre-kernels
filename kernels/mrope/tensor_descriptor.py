# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of _triton_mrope_forward.
# Original: kernels/mrope/original.py
# Changes summarized in kernels/mrope/conversion-notes.md.

import triton
import triton.language as tl


@triton.jit
def _mrope_kernel_td(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    """mRoPE forward, tensor-descriptor form. One program per token."""
    pid = tl.program_id(0)

    half_rd = rd // 2
    BLOCK_HALF: tl.constexpr = max(pad_hd // 2, 8)

    # ----------------------------------------------------------------
    # cos / sin: shape (3, num_tokens, half_rd), the 3 axis is t/h/w.
    # Load this token's row from each section, then select the section
    # range with tl.where and sum (sections are disjoint -> additive).
    # ----------------------------------------------------------------
    cos_desc = tl.make_tensor_descriptor(
        cos,
        shape=[3, num_tokens, half_rd],
        strides=[num_tokens * half_rd, half_rd, 1],
        block_shape=[1, 1, BLOCK_HALF],
    )
    sin_desc = tl.make_tensor_descriptor(
        sin,
        shape=[3, num_tokens, half_rd],
        strides=[num_tokens * half_rd, half_rd, 1],
        block_shape=[1, 1, BLOCK_HALF],
    )

    t_cos_row = cos_desc.load([0, pid, 0]).reshape(BLOCK_HALF)
    h_cos_row = cos_desc.load([1, pid, 0]).reshape(BLOCK_HALF)
    w_cos_row = cos_desc.load([2, pid, 0]).reshape(BLOCK_HALF)
    t_sin_row = sin_desc.load([0, pid, 0]).reshape(BLOCK_HALF)
    h_sin_row = sin_desc.load([1, pid, 0]).reshape(BLOCK_HALF)
    w_sin_row = sin_desc.load([2, pid, 0]).reshape(BLOCK_HALF)

    cos_offsets = tl.arange(0, BLOCK_HALF)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    cos_row = (
        tl.where(t_mask, t_cos_row, 0.0)
        + tl.where(h_mask, h_cos_row, 0.0)
        + tl.where(w_mask, w_cos_row, 0.0)
    )
    sin_row = (
        tl.where(t_mask, t_sin_row, 0.0)
        + tl.where(h_mask, h_sin_row, 0.0)
        + tl.where(w_mask, w_sin_row, 0.0)
    )

    # ----------------------------------------------------------------
    # q / k: rows of n_qh*hd / n_kh*hd, viewed as
    # (num_tokens, heads, 2 rotary halves, half_rd lanes). This keeps descriptor
    # offsets block-aligned even when rotary_dim is smaller than head_size.
    # ----------------------------------------------------------------
    q_desc = tl.make_tensor_descriptor(
        q_ptr,
        shape=[num_tokens, n_qh, 2, half_rd],
        strides=[n_qh * hd, hd, half_rd, 1],
        block_shape=[1, pad_n_qh, 1, BLOCK_HALF],
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr,
        shape=[num_tokens, n_kh, 2, half_rd],
        strides=[n_kh * hd, hd, half_rd, 1],
        block_shape=[1, pad_n_kh, 1, BLOCK_HALF],
    )

    q_tile_1 = q_desc.load([pid, 0, 0, 0]).reshape(pad_n_qh, BLOCK_HALF).to(sin_row.dtype)
    q_tile_2 = q_desc.load([pid, 0, 1, 0]).reshape(pad_n_qh, BLOCK_HALF).to(sin_row.dtype)
    k_tile_1 = k_desc.load([pid, 0, 0, 0]).reshape(pad_n_kh, BLOCK_HALF).to(sin_row.dtype)
    k_tile_2 = k_desc.load([pid, 0, 1, 0]).reshape(pad_n_kh, BLOCK_HALF).to(sin_row.dtype)

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    cos_b = cos_row[None, :]
    sin_b = sin_row[None, :]

    new_q_tile_1 = q_tile_1 * cos_b - q_tile_2 * sin_b
    new_q_tile_2 = q_tile_2 * cos_b + q_tile_1 * sin_b
    q_desc.store(
        [pid, 0, 0, 0],
        new_q_tile_1.reshape(1, pad_n_qh, 1, BLOCK_HALF).to(q_ptr.dtype.element_ty),
    )
    q_desc.store(
        [pid, 0, 1, 0],
        new_q_tile_2.reshape(1, pad_n_qh, 1, BLOCK_HALF).to(q_ptr.dtype.element_ty),
    )

    new_k_tile_1 = k_tile_1 * cos_b - k_tile_2 * sin_b
    new_k_tile_2 = k_tile_2 * cos_b + k_tile_1 * sin_b
    k_desc.store(
        [pid, 0, 0, 0],
        new_k_tile_1.reshape(1, pad_n_kh, 1, BLOCK_HALF).to(k_ptr.dtype.element_ty),
    )
    k_desc.store(
        [pid, 0, 1, 0],
        new_k_tile_2.reshape(1, pad_n_kh, 1, BLOCK_HALF).to(k_ptr.dtype.element_ty),
    )

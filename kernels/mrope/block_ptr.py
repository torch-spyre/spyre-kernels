# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Block-pointer conversion of _triton_mrope_forward.
# Original: vllm/model_executor/layers/rotary_embedding/mrope.py
#
# Changes from original:
#   - q/k 2D loads/stores (heads × half_rd) use 2D block pointers.
#   - cos/sin loads remain as raw masked pointers since they involve
#     masked gathering from 3 different base addresses (T/H/W dims)
#     which cannot be expressed as block pointers.
#   - The kernel is in-place on q and k, same as original.
#
# Optimizations over naive block-ptr conversion:
#   - order=(0, 1) to match row-major layout (stride-1 along dim 1)
#   - tl.advance to reuse block pointers for second-half loads
#   - boundary_check only on dimensions that actually need it

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_mrope_forward_block_ptr(
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
    pid = tl.program_id(0)
    q_base = q_ptr + pid * (n_qh * hd)
    k_base = k_ptr + pid * (n_kh * hd)

    half_rd = rd // 2

    # ─── cos/sin loads: raw pointers (masked gather from 3 sources) ───
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    cos_offsets = tl.arange(0, pad_hd // 2)
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

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # ─── q loads/stores: 2D block pointers [n_heads, half_rd] ───
    # order=(0, 1): dim 1 (columns) is contiguous in memory (stride=1)
    q_block_ptr = tl.make_block_ptr(
        base=q_base,
        shape=(n_qh, rd),
        strides=(hd, 1),
        offsets=(0, 0),
        block_shape=(pad_n_qh, pad_hd // 2),
        order=(0, 1),
    )

    q_tile_1 = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(sin_row.dtype)
    q_block_ptr = tl.advance(q_block_ptr, (0, half_rd))
    q_tile_2 = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(sin_row.dtype)

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row

    tl.store(q_block_ptr, new_q_tile_2, boundary_check=(0, 1))
    q_block_ptr = tl.advance(q_block_ptr, (0, -half_rd))
    tl.store(q_block_ptr, new_q_tile_1, boundary_check=(0, 1))

    # ─── k loads/stores: 2D block pointers [n_kh, half_rd] ───
    k_block_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(n_kh, rd),
        strides=(hd, 1),
        offsets=(0, 0),
        block_shape=(pad_n_kh, pad_hd // 2),
        order=(0, 1),
    )

    k_tile_1 = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero").to(sin_row.dtype)
    k_block_ptr = tl.advance(k_block_ptr, (0, half_rd))
    k_tile_2 = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero").to(sin_row.dtype)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row

    tl.store(k_block_ptr, new_k_tile_2, boundary_check=(0, 1))
    k_block_ptr = tl.advance(k_block_ptr, (0, -half_rd))
    tl.store(k_block_ptr, new_k_tile_1, boundary_check=(0, 1))

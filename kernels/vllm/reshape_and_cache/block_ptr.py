# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of reshape_and_cache_kernel_flash.
# Original: vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
#
# Changes from original:
#   - Source key/value loads use 2D tensor descriptors over the
#     (num_tokens, num_heads * head_size) flattened tensor.
#   - Cache stores remain as raw masked stores because the destination
#     address depends on slot_mapping (data-dependent scatter).
#   - slot_mapping scalar load also remains raw.

import torch
import triton
import triton.language as tl


@triton.jit
def _reshape_and_cache_kernel_block_ptr(
    key_ptr,          # [num_tokens, num_heads, head_size]
    value_ptr,        # [num_tokens, num_heads, head_size]
    key_cache_ptr,    # [num_blocks, block_size, num_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping_ptr, # [num_tokens]
    k_scale,
    v_scale,
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    head_stride: tl.int64,
    dim_stride_k: tl.int64,
    dim_stride_v: tl.int64,
    page_stride: tl.int64,
    num_tokens,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    x: tl.constexpr,
    USE_HEAD_MAJOR_LAYOUT: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """
    Copy key/value into paged KV cache — tensor-descriptor version.
    Source loads use descriptors; cache stores remain raw (scatter).

    Grid: (num_tokens, cdiv(num_heads * head_size, TILE_SIZE))
    """
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    tile_i = tl.program_id(axis=1)
    n = num_heads * head_size

    key_desc = tl.make_tensor_descriptor(
        key_ptr,
        shape=[num_tokens, n],
        strides=[key_stride, 1],
        block_shape=[1, TILE_SIZE],
    )
    value_desc = tl.make_tensor_descriptor(
        value_ptr,
        shape=[num_tokens, n],
        strides=[value_stride, 1],
        block_shape=[1, TILE_SIZE],
    )

    key_load = key_desc.load([token_idx, tile_i * TILE_SIZE]).reshape([TILE_SIZE])
    value_load = value_desc.load([token_idx, tile_i * TILE_SIZE]).reshape([TILE_SIZE])

    # Cache stores: data-dependent scatter, remains raw pointer
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    mask = tile_pos < n

    cur_head = tile_pos // head_size
    cur_dim = tile_pos % head_size
    tgt_idx = (
        block_idx * block_stride
        + block_offset * page_stride
        + cur_head * head_stride
        + cur_dim
    )

    tl.store(key_cache_ptr + tgt_idx, key_load, mask=mask)
    tl.store(value_cache_ptr + tgt_idx, value_load, mask=mask)

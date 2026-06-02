# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Spyre-aware conversion of _fwd_kernel_stage2.
# Original: kernels/decode_softmax_reducev/original.py
#
# Conversion from original:
#   - Grid capped at 32 cores; (batch, head_tile) pairs distributed via loop
#   - num_batches and num_heads added as runtime args (original used grid dims)
#   - All memory access uses tl.make_tensor_descriptor (no raw pointer loads)
#   - BLOCK_DV replaced with fixed BLOCK_SIZE; value dim tiled with inner loop
#   - Heads vectorized with BLOCK_H to improve scratchpad utilization
#   - Split reduction recomputed per d-tile (weights are scalar, independent of d)
#
# Known gaps (backend fixes required before this compiles on Spyre):
#
#   GAP 1 — scalar descriptors (≥16 bytes in last dim):
#     seqlen_desc block_shape=[1] → 4 bytes < 16 bytes minimum.
#     mid_lse_desc block_shape=[1,BLOCK_H,1,1] → 4 bytes < 16 bytes minimum.
#     These are rejected at trace time by the Triton frontend.
#
#   GAP 2 — rank-reduced loads/stores (msrivats/triton#99):
#     Descriptor loads with leading singleton dims produce ND results that
#     must be reshaped for use with lower-rank accumulators/stores.
#     LowerDescriptorMemory does not yet handle rank-reduced descriptor ops.

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_stage2_spyre(
    Mid_O,
    o,
    lse,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    num_batches,
    num_heads,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lv: tl.constexpr,
):
    """Online softmax merge of partial attention outputs across KV splits."""
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    # [gap] scalar descriptor — requires ≥16 bytes in last dim
    seqlen_desc = tl.make_tensor_descriptor(
        B_Seqlen,
        shape=[num_batches],
        strides=[1],
        block_shape=[1],
    )

    # [gap] rank-reduced load — msrivats/triton#99
    mid_v_desc = tl.make_tensor_descriptor(
        Mid_O,
        shape=[num_batches, num_heads, NUM_KV_SPLITS, Lv],
        strides=[stride_mid_ob, stride_mid_oh, stride_mid_os, 1],
        block_shape=[1, BLOCK_H, 1, BLOCK_SIZE],
    )
    # [gap] scalar descriptor — requires ≥16 bytes in last dim
    # [gap] rank-reduced load — msrivats/triton#99
    mid_lse_desc = tl.make_tensor_descriptor(
        Mid_O,
        shape=[num_batches, num_heads, NUM_KV_SPLITS, Lv + 1],
        strides=[stride_mid_ob, stride_mid_oh, stride_mid_os, 1],
        block_shape=[1, BLOCK_H, 1, 1],
    )

    # [gap] rank-reduced store — msrivats/triton#99
    o_desc = tl.make_tensor_descriptor(
        o,
        shape=[num_batches, num_heads, Lv],
        strides=[stride_obs, stride_oh, 1],
        block_shape=[1, BLOCK_H, BLOCK_SIZE],
    )
    lse_desc = tl.make_tensor_descriptor(
        lse,
        shape=[num_batches, num_heads],
        strides=[stride_lse_bs, 1],
        block_shape=[1, BLOCK_H],
    )

    d_tiles = tl.cdiv(Lv, BLOCK_SIZE)
    head_tiles = tl.cdiv(num_heads, BLOCK_H)

    total_work = num_batches * head_tiles
    work_per_core = tl.cdiv(total_work, num_cores)
    work_start = pid * work_per_core
    work_end = tl.minimum(work_start + work_per_core, total_work)

    for work_idx in range(work_start, work_end):
        cur_batch = work_idx // head_tiles
        h_tile = work_idx % head_tiles
        h_offset = h_tile * BLOCK_H

        # [gap] scalar descriptor — requires ≥16 bytes in last dim
        cur_batch_seq_len = seqlen_desc.load([cur_batch])

        for d in range(d_tiles):
            e_sum = tl.zeros([1, BLOCK_H, 1, 1], dtype=tl.float32)
            e_max = tl.full([1, BLOCK_H, 1, 1], -float("inf"), dtype=tl.float32)
            acc = tl.zeros([1, BLOCK_H, 1, BLOCK_SIZE], dtype=tl.float32)

            for split_kv_id in range(0, NUM_KV_SPLITS):
                kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
                split_kv_start = kv_len_per_split * split_kv_id
                split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

                if split_kv_end > split_kv_start:
                    # [gap] rank-reduced load — msrivats/triton#99
                    tv = mid_v_desc.load([cur_batch, h_offset, split_kv_id, d * BLOCK_SIZE])
                    # [gap] scalar descriptor — requires ≥16 bytes in last dim
                    # [gap] rank-reduced load — msrivats/triton#99
                    tlogic = mid_lse_desc.load([cur_batch, h_offset, split_kv_id, Lv])
                    n_e_max = tl.maximum(tlogic, e_max)

                    old_scale = tl.exp(e_max - n_e_max)
                    acc *= old_scale
                    exp_logic = tl.exp(tlogic - n_e_max)
                    acc += exp_logic * tv

                    e_sum = e_sum * old_scale + exp_logic
                    e_max = n_e_max

            # [gap] rank-reduced store — msrivats/triton#99
            result = (acc / e_sum).reshape([1, BLOCK_H, BLOCK_SIZE])
            o_desc.store([cur_batch, h_offset, d * BLOCK_SIZE], result)

        lse_val = (e_max + tl.log(e_sum)).reshape([1, BLOCK_H])
        lse_desc.store([cur_batch, h_offset], lse_val)

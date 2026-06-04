# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Spyre-aware conversion of _fwd_kernel_stage2.
# Original: kernels/decode_softmax_reducev/original.py
#
# Conversion from original:
#   - Grid capped at 32 cores; (batch*head) tiles distributed via loop
#   - B_Seqlen expanded to [batch*heads] in wrapper (repeat_interleave)
#   - All memory access uses tl.make_tensor_descriptor (no raw pointer loads)
#   - BLOCK_DV replaced with fixed BLOCK_SIZE; value dim tiled with inner loop
#   - BLOCK_BH (batch, head) pairs processed per tile for scratchpad utilization
#   - Tiles may cross batch boundaries — divergent seq_lens handled via masking
#   - Split reduction recomputed per d-tile (weights are scalar, independent of d)
#
# Known gaps (backend fixes required before this compiles on Spyre):
#
#   GAP 1 — scalar descriptor (≥16 bytes in last dim):
#     mid_lse_desc block_shape=[BLOCK_BH, 1, 1] → 4 bytes < 16 bytes minimum.
#     Rejected at trace time by the Triton frontend.
#     Fix requires layout change: separate LSE tensor from stage 1.
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
    BLOCK_BH: tl.constexpr,
    Lv: tl.constexpr,
):
    """Online softmax merge of partial attention outputs across KV splits."""
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    total_bh = num_batches * num_heads

    # B_Seqlen: [batch * heads] — wrapper expands via repeat_interleave
    # BLOCK_BH × 4 bytes ≥ 16 for BLOCK_BH ≥ 4
    seqlen_desc = tl.make_tensor_descriptor(
        B_Seqlen,
        shape=[total_bh],
        strides=[1],
        block_shape=[BLOCK_BH],
    )

    # Mid_O: [batch, heads, splits, Lv+1] — original layout preserved
    # Descriptor uses stride_mid_ob and stride_mid_oh to handle non-contiguous
    # access across batch boundaries. Within a BLOCK_BH tile, consecutive
    # (batch, head) pairs step by stride_mid_oh; crossing a batch boundary
    # steps by stride_mid_ob - (num_heads-1)*stride_mid_oh, which equals
    # stride_mid_oh for standard contiguous layout.
    #
    # Key insight: for contiguous [batch, heads, splits, Lv+1] layout,
    # stride_mid_ob = heads * splits * (Lv+1) and stride_mid_oh = splits * (Lv+1),
    # so the (batch*heads) flattening has uniform stride = stride_mid_oh between
    # all consecutive (batch, head) pairs regardless of batch boundaries.
    mid_v_desc = tl.make_tensor_descriptor(
        Mid_O,
        shape=[total_bh, NUM_KV_SPLITS, Lv],
        strides=[stride_mid_oh, stride_mid_os, 1],
        block_shape=[BLOCK_BH, 1, BLOCK_SIZE],
    )
    # [gap] scalar descriptor — requires ≥16 bytes in last dim
    mid_lse_desc = tl.make_tensor_descriptor(
        Mid_O,
        shape=[total_bh, NUM_KV_SPLITS, Lv + 1],
        strides=[stride_mid_oh, stride_mid_os, 1],
        block_shape=[BLOCK_BH, 1, 1],
    )

    # o: [batch, heads, Lv] — same contiguous flattening applies
    o_desc = tl.make_tensor_descriptor(
        o,
        shape=[total_bh, Lv],
        strides=[stride_oh, 1],
        block_shape=[BLOCK_BH, BLOCK_SIZE],
    )

    # lse: [batch, heads] — stride_lse_bs between batches, 1 between heads
    # For contiguous layout, stride between consecutive (batch,head) pairs = 1
    # within a batch, but stride_lse_bs - (num_heads-1) at batch boundary.
    # For standard contiguous [batch, heads]: stride_lse_bs = heads, so
    # consecutive elements have stride 1 throughout the flattened view.
    lse_desc = tl.make_tensor_descriptor(
        lse,
        shape=[total_bh],
        strides=[1],
        block_shape=[BLOCK_BH],
    )

    d_tiles = tl.cdiv(Lv, BLOCK_SIZE)
    bh_tiles = tl.cdiv(total_bh, BLOCK_BH)

    work_per_core = tl.cdiv(bh_tiles, num_cores)
    work_start = pid * work_per_core
    work_end = tl.minimum(work_start + work_per_core, bh_tiles)

    for tile_idx in range(work_start, work_end):
        bh_offset = tile_idx * BLOCK_BH

        # Load BLOCK_BH seq_lens (one per (batch, head) pair in this tile)
        # Pairs in the same batch share the same value; cross-batch pairs differ
        cur_seq_lens = seqlen_desc.load([bh_offset])

        for d in range(d_tiles):
            e_sum = tl.zeros([BLOCK_BH, 1, 1], dtype=tl.float32)
            e_max = tl.full([BLOCK_BH, 1, 1], -float("inf"), dtype=tl.float32)
            acc = tl.zeros([BLOCK_BH, 1, BLOCK_SIZE], dtype=tl.float32)

            for split_kv_id in range(0, NUM_KV_SPLITS):
                kv_len_per_split = tl.cdiv(cur_seq_lens, NUM_KV_SPLITS)
                split_kv_start = kv_len_per_split * split_kv_id
                split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_seq_lens)
                active = (split_kv_end > split_kv_start).reshape([BLOCK_BH, 1, 1])

                # [gap] rank-reduced load — msrivats/triton#99
                tv = mid_v_desc.load([bh_offset, split_kv_id, d * BLOCK_SIZE])
                # [gap] scalar descriptor — requires ≥16 bytes in last dim
                # [gap] rank-reduced load — msrivats/triton#99
                tlogic = mid_lse_desc.load([bh_offset, split_kv_id, Lv])

                n_e_max = tl.maximum(tlogic, e_max)
                old_scale = tl.exp(e_max - n_e_max)
                exp_logic = tl.exp(tlogic - n_e_max)

                # Mask inactive splits — zero contribution for lanes with empty splits
                ones = tl.full([BLOCK_BH, 1, 1], 1.0, dtype=tl.float32)
                zeros = tl.zeros([BLOCK_BH, 1, 1], dtype=tl.float32)
                masked_old_scale = tl.where(active, old_scale, ones)
                masked_exp = tl.where(active, exp_logic, zeros)
                masked_e_max = tl.where(active, n_e_max, e_max)

                acc *= masked_old_scale
                acc += masked_exp * tv

                e_sum = e_sum * masked_old_scale + masked_exp
                e_max = masked_e_max

            # [gap] rank-reduced store — msrivats/triton#99
            result = (acc / e_sum).reshape([BLOCK_BH, BLOCK_SIZE])
            o_desc.store([bh_offset, d * BLOCK_SIZE], result)

        lse_val = (e_max + tl.log(e_sum)).reshape([BLOCK_BH])
        lse_desc.store([bh_offset], lse_val)

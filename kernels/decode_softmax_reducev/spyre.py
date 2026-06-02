# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Spyre-aware conversion of _fwd_kernel_stage2.
# Original: kernels/decode_softmax_reducev/original.py
#
# Conversion from original:
#   - Grid capped at 32 cores; (batch, heads) work distributed via loop
#   - num_batches and num_heads added as runtime args (original used grid dims)
#   - All memory access uses tl.make_tensor_descriptor (no raw pointer loads)
#   - V tiles loaded via 4D descriptor over Mid_O with block_shape [1,1,1,BLOCK_DV]
#   - LSE scalars loaded via 4D descriptor with block_shape [1,1,1,1]
#   - Output V stored via 3D descriptor; output LSE via 2D descriptor
#   - B_Seqlen loaded via 1D descriptor
#
# Known gaps (backend fixes required before this compiles on Spyre):
#
#   GAP 1 — scalar descriptors (≥16 bytes in last dim):
#     seqlen_desc block_shape=[1] → 4 bytes < 16 bytes minimum.
#     mid_lse_desc block_shape=[1,1,1,1] → 4 bytes < 16 bytes minimum.
#     lse_desc block_shape=[1,1] → 4 bytes < 16 bytes minimum.
#     These are rejected at trace time by the Triton frontend.
#
#   GAP 2 — rank-reduced loads/stores (msrivats/triton#99):
#     mid_v_desc.load produces tensor<1x1x1xBLOCK_DV>; acc is [1,1,1,BLOCK_DV]
#     to avoid rank mismatch. Stores reshape back to match descriptor block_shape.
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
    BLOCK_DV: tl.constexpr,
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
        block_shape=[1, 1, 1, BLOCK_DV],
    )
    # [gap] scalar descriptor — requires ≥16 bytes in last dim
    # [gap] rank-reduced load — msrivats/triton#99
    mid_lse_desc = tl.make_tensor_descriptor(
        Mid_O,
        shape=[num_batches, num_heads, NUM_KV_SPLITS, Lv + 1],
        strides=[stride_mid_ob, stride_mid_oh, stride_mid_os, 1],
        block_shape=[1, 1, 1, 1],
    )

    # [gap] rank-reduced store — msrivats/triton#99
    o_desc = tl.make_tensor_descriptor(
        o,
        shape=[num_batches, num_heads, Lv],
        strides=[stride_obs, stride_oh, 1],
        block_shape=[1, 1, BLOCK_DV],
    )
    # [gap] scalar descriptor — requires ≥16 bytes in last dim
    # [gap] rank-reduced store — msrivats/triton#99
    lse_desc = tl.make_tensor_descriptor(
        lse,
        shape=[num_batches, num_heads],
        strides=[stride_lse_bs, 1],
        block_shape=[1, 1],
    )

    total_work = num_batches * num_heads
    work_per_core = tl.cdiv(total_work, num_cores)
    work_start = pid * work_per_core
    work_end = tl.minimum(work_start + work_per_core, total_work)

    for work_idx in range(work_start, work_end):
        cur_batch = work_idx // num_heads
        cur_head = work_idx % num_heads

        # [gap] scalar descriptor — requires ≥16 bytes in last dim
        cur_batch_seq_len = seqlen_desc.load([cur_batch])

        e_sum = 0.0
        e_max = -float("inf")
        acc = tl.zeros([1, 1, 1, BLOCK_DV], dtype=tl.float32)

        for split_kv_id in range(0, NUM_KV_SPLITS):
            kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
            split_kv_start = kv_len_per_split * split_kv_id
            split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

            if split_kv_end > split_kv_start:
                # [gap] rank-reduced load — msrivats/triton#99
                tv = mid_v_desc.load([cur_batch, cur_head, split_kv_id, 0])
                # [gap] scalar descriptor — requires ≥16 bytes in last dim
                # [gap] rank-reduced load — msrivats/triton#99
                tlogic = mid_lse_desc.load([cur_batch, cur_head, split_kv_id, Lv])
                n_e_max = tl.maximum(tlogic, e_max)

                old_scale = tl.exp(e_max - n_e_max)
                acc *= old_scale
                exp_logic = tl.exp(tlogic - n_e_max)
                acc += exp_logic * tv

                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max

        # [gap] rank-reduced store — msrivats/triton#99
        result = (acc / e_sum).reshape([1, 1, BLOCK_DV])
        o_desc.store([cur_batch, cur_head, 0], result)
        lse_val = e_max + tl.log(e_sum)
        # [gap] scalar descriptor — requires ≥16 bytes in last dim
        # [gap] rank-reduced store — msrivats/triton#99
        lse_desc.store([cur_batch, cur_head], lse_val)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of _fwd_kernel_stage2.
# Original: vllm/v1/attention/ops/triton_decode_attention.py
#
# Changes from original:
#   - The per-split V load uses a 4D tensor descriptor over the
#     (batch, heads, splits, Lv) view of Mid_O.
#   - The final output store uses a 3D descriptor over (batch, heads, Lv).
#   - Per-split LSE scalar loads stay raw (single-element).

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_stage2_block_ptr(
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
    batch,
    heads,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """
    Online softmax merge — tensor-descriptor version.

    Grid: (batch, heads)
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    # Mid_O view: (batch, heads, splits, Lv) — last dim contiguous.
    # Lv may not be a power of two; the descriptor's last-dim shape is Lv,
    # block_shape is BLOCK_DV (next pow2). OOB lanes get zero.
    v_desc = tl.make_tensor_descriptor(
        Mid_O,
        shape=[batch, heads, NUM_KV_SPLITS, Lv],
        strides=[stride_mid_ob, stride_mid_oh, stride_mid_os, 1],
        block_shape=[1, 1, 1, BLOCK_DV],
    )

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = v_desc.load([cur_batch, cur_head, split_kv_id, 0]).reshape([BLOCK_DV])

            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    o_desc = tl.make_tensor_descriptor(
        o,
        shape=[batch, heads, Lv],
        strides=[stride_obs, stride_oh, 1],
        block_shape=[1, 1, BLOCK_DV],
    )
    o_desc.store([cur_batch, cur_head, 0], (acc / e_sum).reshape([1, 1, BLOCK_DV]))

    lse_val = e_max + tl.log(e_sum)
    tl.store(lse + cur_batch * stride_lse_bs + cur_head, lse_val)

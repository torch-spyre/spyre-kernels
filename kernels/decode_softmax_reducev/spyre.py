# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Spyre-aware conversion of _fwd_kernel_stage2.
# Original: kernels/decode_softmax_reducev/original.py
#
# Conversion from original:
#   - Grid capped at 32 cores; (batch, heads) work distributed via loop
#   - num_batches and num_heads added as runtime args (original used grid dims)
#   - V tile loads and output store use raw pointer + tl.arange (strides are
#     runtime values and offsets depend on loop variables, so descriptors
#     would require addptr as base — rejected by Spyre compiler)
#   - Scalar loads (seq_len, per-split LSE) and scalar store (final LSE)
#     remain as raw pointers — single-element accesses

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

    total_work = num_batches * num_heads
    work_per_core = tl.cdiv(total_work, num_cores)
    work_start = pid * work_per_core
    work_end = tl.minimum(work_start + work_per_core, total_work)

    for work_idx in range(work_start, work_end):
        cur_batch = work_idx // num_heads
        cur_head = work_idx % num_heads

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

        e_sum = 0.0
        e_max = -float("inf")
        acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

        offs_d = tl.arange(0, BLOCK_DV)
        mask_d = offs_d < Lv
        offs_v = Mid_O + cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
        offs_logic = Mid_O + cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

        for split_kv_id in range(0, NUM_KV_SPLITS):
            kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
            split_kv_start = kv_len_per_split * split_kv_id
            split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

            if split_kv_end > split_kv_start:
                tv = tl.load(
                    offs_v + split_kv_id * stride_mid_os,
                    mask=mask_d, other=0.0,
                )
                tlogic = tl.load(offs_logic + split_kv_id * stride_mid_os)
                n_e_max = tl.maximum(tlogic, e_max)

                old_scale = tl.exp(e_max - n_e_max)
                acc *= old_scale
                exp_logic = tl.exp(tlogic - n_e_max)
                acc += exp_logic * tv

                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max

        tl.store(
            o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
            acc / e_sum,
            mask=mask_d,
        )
        lse_val = e_max + tl.log(e_sum)
        tl.store(lse + cur_batch * stride_lse_bs + cur_head, lse_val)

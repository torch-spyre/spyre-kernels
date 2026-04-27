# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Block-pointer conversion of prefill attention _fwd_kernel.
# Original: vllm/v1/attention/ops/triton_prefill_attention.py
#
# Changes from original:
#   - Q load uses a 2D block pointer [BLOCK_M, BLOCK_DMODEL].
#   - K loads in the inner loop use a 2D block pointer [BLOCK_DMODEL, BLOCK_N]
#     advanced by BLOCK_N each iteration.
#   - V loads use a 2D block pointer [BLOCK_N, BLOCK_DMODEL] advanced similarly.
#   - O store uses a 2D block pointer [BLOCK_M, BLOCK_DMODEL].
#   - Scalar loads (B_Start_Loc, B_Seqlen) remain raw.
#   - Causal and sequence-length masking are applied post-load using
#     tl.where, since block-pointer boundary_check only handles OOB
#     (not causal masking).
#   - Simplified: no sliding window.

import math
import torch
import triton
import triton.language as tl

RCP_LN2 = 1.0 / math.log(2.0)


@triton.jit
def _fwd_kernel_block_ptr(
    Q, K, V,
    sm_scale,
    B_Start_Loc, B_Seqlen,
    Out,
    stride_qbs, stride_qh,
    stride_kbs, stride_kh,
    stride_vbs, stride_vh,
    stride_obs, stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW_Q: tl.constexpr,
    SLIDING_WINDOW_K: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk

    # ─── Q: 2D block pointer [BLOCK_M, BLOCK_DMODEL] ───
    # Q is [total_tokens, heads, head_dim], stride_qbs = stride(0), stride_qh = stride(1)
    # We need Q[cur_batch_start + start_m*BLOCK_M : ..., cur_head, :]
    # With raw offsets: (start_idx + offs_m) * stride_qbs + cur_head * stride_qh + offs_d
    # As block ptr: base at Q[start_idx, cur_head, 0], shape over the tokens dim
    q_base = Q + cur_batch_in_all_start_index * stride_qbs + cur_head * stride_qh
    q_block_ptr = tl.make_block_ptr(
        base=q_base,
        shape=(cur_batch_seq_len, Lk),
        strides=(stride_qbs, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # ─── K: 2D block pointer [BLOCK_DMODEL, BLOCK_N] (transposed for Q@K) ───
    k_base = K + cur_batch_in_all_start_index * stride_kbs + cur_kv_head * stride_kh
    k_block_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(Lk, cur_batch_seq_len),
        strides=(1, stride_kbs),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )

    # ─── V: 2D block pointer [BLOCK_N, BLOCK_DMODEL] ───
    v_base = V + cur_batch_in_all_start_index * stride_vbs + cur_kv_head * stride_vh
    v_block_ptr = tl.make_block_ptr(
        base=v_base,
        shape=(cur_batch_seq_len, Lk),
        strides=(stride_vbs, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    end_n = cur_batch_seq_len
    end_n = tl.minimum(end_n, (start_m + 1) * BLOCK_M) if IS_CAUSAL else end_n

    start_n_limit = 0
    end_n_limit = block_mask * end_n

    for start_n in range(start_n_limit, end_n_limit, BLOCK_N):
        pos_q = offs_m[:, None]
        pos_k = start_n + offs_n[None, :]
        mask = pos_k < cur_batch_seq_len
        if IS_CAUSAL:
            mask &= pos_q >= pos_k

        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    acc = acc.to(Out.dtype.element_ty)

    # ─── O: 2D block pointer [BLOCK_M, BLOCK_DMODEL] ───
    o_base = Out + cur_batch_in_all_start_index * stride_obs + cur_head * stride_oh
    o_block_ptr = tl.make_block_ptr(
        base=o_base,
        shape=(cur_batch_seq_len, Lk),
        strides=(stride_obs, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(o_block_ptr, acc, boundary_check=(0, 1))

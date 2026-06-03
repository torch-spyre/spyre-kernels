# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of prefill attention _fwd_kernel.
# Original: vllm/v1/attention/ops/triton_prefill_attention.py
#
# Changes from original:
#   - Q/K/V/O accesses use 3D tensor descriptors over the full
#     (total_tokens, num_*_heads, head_dim) tensors so the descriptor
#     base is the input pointer (16-byte aligned trivially).
#   - K is loaded as (BLOCK_N, BLOCK_DMODEL) and transposed via tl.trans
#     before tl.dot(q, k_t). Tensor descriptors require the last dim
#     contiguous, so K cannot be loaded pre-transposed (as block pointers
#     allowed via strides=(1, stride_kbs)).
#   - tl.advance is replaced with offset arithmetic per loop iteration.
#   - Causal and sequence-length masking are still applied post-load via
#     tl.where, identical to the block-pointer version.

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
    num_q_heads,
    num_kv_heads,
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

    q_base = Q + cur_batch_in_all_start_index * stride_qbs
    k_base = K + cur_batch_in_all_start_index * stride_kbs
    v_base = V + cur_batch_in_all_start_index * stride_vbs
    o_base = Out + cur_batch_in_all_start_index * stride_obs

    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[cur_batch_seq_len, num_q_heads, Lk],
        strides=[stride_qbs, stride_qh, 1],
        block_shape=[BLOCK_M, 1, BLOCK_DMODEL],
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[cur_batch_seq_len, num_kv_heads, Lk],
        strides=[stride_kbs, stride_kh, 1],
        block_shape=[BLOCK_N, 1, BLOCK_DMODEL],
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[cur_batch_seq_len, num_kv_heads, Lk],
        strides=[stride_vbs, stride_vh, 1],
        block_shape=[BLOCK_N, 1, BLOCK_DMODEL],
    )
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[cur_batch_seq_len, num_q_heads, Lk],
        strides=[stride_obs, stride_oh, 1],
        block_shape=[BLOCK_M, 1, BLOCK_DMODEL],
    )

    q_row0 = start_m * BLOCK_M
    q = q_desc.load([q_row0, cur_head, 0]).reshape([BLOCK_M, BLOCK_DMODEL])

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

        k_row0 = start_n
        k_tile = k_desc.load([k_row0, cur_kv_head, 0]).reshape([BLOCK_N, BLOCK_DMODEL])
        k_t = tl.trans(k_tile)

        qk = tl.dot(q, k_t)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = v_desc.load([k_row0, cur_kv_head, 0]).reshape([BLOCK_N, BLOCK_DMODEL])
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    acc = acc.to(Out.dtype.element_ty)

    o_desc.store([q_row0, cur_head, 0], acc.reshape([BLOCK_M, 1, BLOCK_DMODEL]))

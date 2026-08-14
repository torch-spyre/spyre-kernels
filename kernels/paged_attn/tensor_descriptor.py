# SPDX-License-Identifier: Apache-2.0
#
# Tensor-descriptor paged-attention kernel.
# Original: kernels/paged_attn/original.py
# Changes summarized in kernels/paged_attn/conversion-notes.md.
#
# Batched gather over B + 4-D batched tl.dot, grid = (1,):
#   The B loop steps by BLK_B, and the BLK_B batches in each block are
#   vectorized -- a single descriptor_gather pulls both batches' rows at once
#   (rows = the batches' slot indices flattened batch-major), so there is no
#   per-batch loop. tl.dot then batches over (BLK_B, BLK_H) -> 4-D.
#
# K/V (CACHE, H, D) are viewed 2-D as (CACHE, H*D); the gather pulls
# BLK_B*KV_BLOCK rows and the BLK_H*D columns starting at h_start*D, giving the
# BLK_H heads for the gathered slots in one shot. The result is reshaped to
# (BLK_B, KV_BLOCK, BLK_H, D). This variant uses the base descriptor_gather
# (1-D row index, 2-D src); the any-rank form lives in spyre_aware.py.
#
# Descriptors only -- no raw pointer arithmetic anywhere.

import triton
import triton.language as tl


@triton.jit
def _paged_attn_kernel_NHD_td(
    Q,      # (B, Lq, H, D)
    K,      # (CACHE, H, D)
    V,      # (CACHE, H, D)
    SLOTS,  # (B, Lk) absolute physical slot index per (request, token)
    Out,    # (B, H, Lq, D)
    scale,
    B: tl.constexpr, H: tl.constexpr, Lq: tl.constexpr, Lk: tl.constexpr,
    CACHE: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLK_B: tl.constexpr,     # B blocking factor
    BLK_H: tl.constexpr,     # H blocking factor
):
    HD: tl.constexpr = H * BLOCK_D                # flattened (H, D) row width
    BHD: tl.constexpr = BLK_H * BLOCK_D           # flattened (BLK_H, D) gather width
    GATHER_ROWS: tl.constexpr = BLK_B * KV_BLOCK  # both batches' rows per gather

    # --- tensor descriptors (no pointer arithmetic) ---
    # Q tile spans BLK_B batches x BLOCK_Q queries x BLK_H heads x D
    q_desc = tl.make_tensor_descriptor(
        Q, shape=[B, Lq, H, BLOCK_D], strides=[Lq * H * BLOCK_D, H * BLOCK_D, BLOCK_D, 1],
        block_shape=[BLK_B, BLOCK_Q, BLK_H, BLOCK_D],
    )
    # K/V as 2-D (CACHE, H*D); gather BLK_B*KV_BLOCK rows, BHD columns at h_start*D
    k_desc = tl.make_tensor_descriptor(
        K, shape=[CACHE, HD], strides=[HD, 1], block_shape=[1, BHD],
    )
    v_desc = tl.make_tensor_descriptor(
        V, shape=[CACHE, HD], strides=[HD, 1], block_shape=[1, BHD],
    )
    # SLOTS for both batches at once: (BLK_B, KV_BLOCK)
    s_desc = tl.make_tensor_descriptor(
        SLOTS, shape=[B, Lk], strides=[Lk, 1], block_shape=[BLK_B, KV_BLOCK],
    )
    # Out tile spans BLK_B batches x BLK_H heads x BLOCK_Q queries x D
    o_desc = tl.make_tensor_descriptor(
        Out, shape=[B, H, Lq, BLOCK_D], strides=[H * Lq * BLOCK_D, Lq * BLOCK_D, BLOCK_D, 1],
        block_shape=[BLK_B, BLK_H, BLOCK_Q, BLOCK_D],
    )

    n_pages = tl.cdiv(Lk, KV_BLOCK)

    # ===== batch loop, blocked by BLK_B  (for b_start in range(0, B, 2)) =====
    for b_start in range(0, B, BLK_B):
        # ===== query-block loop  (for lq_start in range(0, Lq, q_block_size)) =====
        for lq_start in range(0, Lq, BLOCK_Q):
            # ===== head loop, blocked by BLK_H  (for h_start in range(0, H, 4)) =====
            for h_start in range(0, H, BLK_H):

                # load Q (BLK_B, BLOCK_Q, BLK_H, D) -> (BLK_B, BLK_H, BLOCK_Q, D); scale
                q = q_desc.load([b_start, lq_start, h_start, 0])
                q = tl.permute(q, (0, 2, 1, 3))                    # (BLK_B, BLK_H, BLOCK_Q, D)
                q = (q.to(tl.float32) * scale).to(tl.float16)

                # online-softmax state, batched over (BLK_B, BLK_H)
                m_i = tl.full([BLK_B, BLK_H, BLOCK_Q], float("-inf"), tl.float32)
                l_i = tl.zeros([BLK_B, BLK_H, BLOCK_Q], tl.float32)
                acc = tl.zeros([BLK_B, BLK_H, BLOCK_Q, BLOCK_D], tl.float32)

                # ===== KV-page loop =====
                for j in range(0, n_pages):
                    # both batches' absolute slot indices: (BLK_B, KV_BLOCK)
                    slots = s_desc.load([b_start, j * KV_BLOCK])           # (BLK_B, KV_BLOCK)
                    rows = slots.reshape(GATHER_ROWS).to(tl.int32)         # batch-major rows (BLK_B * KV_BLOCK)

                    # ONE batched gather over both batches -> (BLK_B*KV_BLOCK, BLK_H*D)
                    #   -> (BLK_B, KV_BLOCK, BLK_H, D)
                    k_g = k_desc.gather(rows, h_start * BLOCK_D).reshape(BLK_B, KV_BLOCK, BLK_H, BLOCK_D)
                    v_g = v_desc.gather(rows, h_start * BLOCK_D).reshape(BLK_B, KV_BLOCK, BLK_H, BLOCK_D)
                    k_g = (k_g.to(tl.float32) * scale).to(tl.float16)

                    kT = tl.permute(k_g, (0, 2, 3, 1))             # (BLK_B, BLK_H, D, KV_BLOCK)
                    vv = tl.permute(v_g, (0, 2, 1, 3))             # (BLK_B, BLK_H, KV_BLOCK, D)

                    # 4-D batched matmul over (BLK_B, BLK_H)
                    scores = tl.dot(q, kT)                         # (BLK_B, BLK_H, BLOCK_Q, KV_BLOCK)

                    block_max = tl.max(scores, axis=3)             # (BLK_B, BLK_H, BLOCK_Q)
                    m_new = tl.maximum(m_i, block_max)
                    correction = tl.exp(m_i - m_new)
                    p = tl.exp(scores - m_new[:, :, :, None])

                    l_i = l_i * correction + tl.sum(p, axis=3)
                    acc = acc * correction[:, :, :, None] + tl.dot(p.to(tl.float16), vv)
                    m_i = m_new

                acc = acc / l_i[:, :, :, None]                     # (BLK_B, BLK_H, BLOCK_Q, D)
                o_desc.store([b_start, h_start, lq_start, 0], acc.to(tl.float16))

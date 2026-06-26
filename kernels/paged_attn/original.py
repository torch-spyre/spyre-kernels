# SPDX-License-Identifier: Apache-2.0
#
# Reference paged-attention kernel (hand-written, NOT auto-generated).
#
# Unlike the other kernels' original.py, this is not extracted verbatim from an
# upstream vLLM commit -- it is a clean, self-contained Triton paged-attention
# forward that the descriptor variants (tensor_descriptor.py, spyre_aware.py)
# are validated against. It is deliberately the simplest faithful form: one
# program per (request, head, query block), a single (BLK_B=1, BLK_H=1) gather
# per KV page, and the same effective 1/sqrt(D) score scale as the descriptor
# variants (which fold sqrt of it onto q and k separately; s*s = 1/sqrt(D)).
#
# Layout:
#   Q     : (B, Lq, H, D)        contiguous queries
#   K, V  : (CACHE, H, D)        paged KV cache, one row per physical slot
#   SLOTS : (B, Lk)  int32       absolute physical slot of each (request, token)
#   Out   : (B, H, Lq, D)
#
# Descriptors only -- no raw pointer arithmetic. The data-dependent K/V gather
# uses descriptor_gather over the cache viewed 2-D as (CACHE, H*D).

import triton
import triton.language as tl


@triton.jit
def _paged_attn_kernel_NHD(
    Q,      # (B, Lq, H, D)
    K,      # (CACHE, H, D)
    V,      # (CACHE, H, D)
    SLOTS,  # (B, Lk) absolute physical slot index per (request, token)
    Out,    # (B, H, Lq, D)
    scale,  # = 1/sqrt(sqrt(D)); applied to q and k, so scores carry 1/sqrt(D)
    B: tl.constexpr, H: tl.constexpr, Lq: tl.constexpr, Lk: tl.constexpr,
    CACHE: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    HD: tl.constexpr = H * BLOCK_D

    cur_b = tl.program_id(0)
    cur_h = tl.program_id(1)
    lq_start = tl.program_id(2) * BLOCK_Q

    q_desc = tl.make_tensor_descriptor(
        Q, shape=[B, Lq, H, BLOCK_D], strides=[Lq * H * BLOCK_D, H * BLOCK_D, BLOCK_D, 1],
        block_shape=[1, BLOCK_Q, 1, BLOCK_D],
    )
    k_desc = tl.make_tensor_descriptor(
        K, shape=[CACHE, HD], strides=[HD, 1], block_shape=[KV_BLOCK, BLOCK_D],
    )
    v_desc = tl.make_tensor_descriptor(
        V, shape=[CACHE, HD], strides=[HD, 1], block_shape=[KV_BLOCK, BLOCK_D],
    )
    s_desc = tl.make_tensor_descriptor(
        SLOTS, shape=[B, Lk], strides=[Lk, 1], block_shape=[1, KV_BLOCK],
    )
    o_desc = tl.make_tensor_descriptor(
        Out, shape=[B, H, Lq, BLOCK_D], strides=[H * Lq * BLOCK_D, Lq * BLOCK_D, BLOCK_D, 1],
        block_shape=[1, 1, BLOCK_Q, BLOCK_D],
    )

    # load Q (1, BLOCK_Q, 1, D) -> (BLOCK_Q, D); scale
    q = q_desc.load([cur_b, lq_start, cur_h, 0]).reshape([BLOCK_Q, BLOCK_D])
    q = (q.to(tl.float32) * scale).to(tl.float16)

    m_i = tl.full([BLOCK_Q], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)

    n_pages = tl.cdiv(Lk, KV_BLOCK)
    for j in range(0, n_pages):
        slots = s_desc.load([cur_b, j * KV_BLOCK])            # (1, KV_BLOCK)
        rows = slots.reshape(KV_BLOCK).to(tl.int32)           # (KV_BLOCK,)

        # gather this head's columns for the KV_BLOCK slots -> (KV_BLOCK, D)
        k_g = tl.descriptor_gather(k_desc, rows, cur_h * BLOCK_D)
        v_g = tl.descriptor_gather(v_desc, rows, cur_h * BLOCK_D)
        k_g = (k_g.to(tl.float32) * scale).to(tl.float16)

        kT = tl.trans(k_g)                                    # (D, KV_BLOCK)
        scores = tl.dot(q, kT)                                # (BLOCK_Q, KV_BLOCK)

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        correction = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = l_i * correction + tl.sum(p, axis=1)
        acc = acc * correction[:, None] + tl.dot(p.to(tl.float16), v_g)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_desc.store([cur_b, cur_h, lq_start, 0],
                 acc.to(tl.float16).reshape([1, 1, BLOCK_Q, BLOCK_D]))

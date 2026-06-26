import math

import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.paged_attn.original import _paged_attn_kernel_NHD
from kernels.paged_attn.tensor_descriptor import _paged_attn_kernel_NHD_td
from kernels.paged_attn.spyre_aware import _paged_attn_kernel_NHD_sa

# Defaults for the batched descriptor variants. BLK_B / BLK_H are the B / H
# blocking factors of the 4-D batched gather + tl.dot.
KV_BLOCK = 16
BLOCK_Q = 16
BLK_B = 2
BLK_H = 4

_BATCHED = (_paged_attn_kernel_NHD_td, _paged_attn_kernel_NHD_sa)


def paged_attention_NHD(
    q: torch.Tensor,       # (B, Lq, H, D)
    k: torch.Tensor,       # (CACHE, H, D)
    v: torch.Tensor,       # (CACHE, H, D)
    slots: torch.Tensor,   # (B, Lk) integer physical slot indices
    *,
    softmax_scale: float | None = None,
    kernel_fn=_paged_attn_kernel_NHD,
    kv_block: int = KV_BLOCK,
    block_q: int = BLOCK_Q,
    blk_b: int = BLK_B,
    blk_h: int = BLK_H,
) -> torch.Tensor:
    """Launch the paged-attention kernel selected by ``kernel_fn``.

    The reference (``_paged_attn_kernel_NHD``) and the two descriptor variants
    (``_paged_attn_kernel_NHD_td`` base gather, ``_paged_attn_kernel_NHD_sa`` extended
    any-rank gather) all go through this one launch path so tests can dispatch
    with ``kernel_fn=`` and sweep the tile sizes.

    Layout: ``q`` is ``(B, Lq, H, D)``; ``k``/``v`` are the paged KV cache
    ``(CACHE, H, D)``; ``slots`` is ``(B, Lk)`` and names, per request, the
    absolute physical cache slot of each key/value token. Output is
    ``(B, H, Lq, D)``.

    The descriptor variants scale q and k each by ``sqrt(softmax_scale)``
    (so scores carry ``softmax_scale``); the default ``softmax_scale`` is
    ``1/sqrt(D)``. The kernels take ``scale = sqrt(softmax_scale)``.
    """
    assert q.dim() == 4, "q must be (B, Lq, H, D)"
    assert k.dim() == 3 and v.dim() == 3, "k/v must be (CACHE, H, D)"
    B, Lq, H, D = q.shape
    CACHE = k.shape[0]
    Lk = slots.shape[1]
    softmax_scale = 1.0 / math.sqrt(D) if softmax_scale is None else softmax_scale
    # kernels apply `scale` to q and to k, so scores carry scale**2 == softmax_scale
    scale = math.sqrt(softmax_scale)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    slots = slots.to(torch.int32).contiguous()
    out = torch.empty((B, H, Lq, D), device=q.device, dtype=q.dtype)

    if kernel_fn in _BATCHED:
        # Batched descriptor variants: descriptor shapes are built from the
        # problem dims (constexprs) and a single program walks all (B, Lq, H)
        # work via the explicit BLK_B / BLOCK_Q / BLK_H loops.
        ensure_triton_allocator()
        grid = (1,)
        kernel_fn[grid](
            q, k, v, slots, out,
            scale,
            B=B, H=H, Lq=Lq, Lk=Lk, CACHE=CACHE,
            KV_BLOCK=kv_block, BLOCK_Q=block_q, BLOCK_D=D,
            BLK_B=blk_b, BLK_H=blk_h,
        )
    else:
        # Reference kernel: one program per (request, head, query block).
        ensure_triton_allocator()
        grid = (B, H, triton.cdiv(Lq, block_q))
        kernel_fn[grid](
            q, k, v, slots, out,
            scale,
            B=B, H=H, Lq=Lq, Lk=Lk, CACHE=CACHE,
            KV_BLOCK=kv_block, BLOCK_Q=block_q, BLOCK_D=D,
        )
    return out

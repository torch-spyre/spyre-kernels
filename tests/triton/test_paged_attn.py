# SPDX-License-Identifier: Apache-2.0
"""GPU equivalence tests for the paged-attention kernels.

Compares the two descriptor variants (_paged_attn_kernel_NHD_td base gather,
_paged_attn_kernel_NHD_sa extended any-rank gather) against the reference kernel and
against a pure PyTorch SDPA-with-gather, all launched through
kernels/paged_attn/wrapper.py via its kernel_fn= dispatch — no forked path.

Run: pytest tests/triton/test_paged_attn.py -v
Requires: GPU with triton tensor-descriptor + descriptor_gather support (the
descriptor variants use the gather primitive, which stock PyPI Triton lacks).
"""

import math

import pytest
import torch

from kernels.paged_attn.original import _paged_attn_kernel
from kernels.paged_attn.tensor_descriptor import _paged_attn_kernel_NHD_td
from kernels.paged_attn.spyre_aware import _paged_attn_kernel_NHD_sa
from kernels.paged_attn.wrapper import paged_attention


# ─── Reference ─────────────────────────────────────────────────────

def torch_reference(q, k, v, slots):
    """Pure PyTorch paged SDPA — the ground truth.

    q (B,Lq,H,D); k,v (CACHE,H,D); slots (B,Lk) -> out (B,H,Lq,D).
    """
    B, Lq, H, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)
    out = torch.empty((B, H, Lq, D), device=q.device, dtype=q.dtype)
    for b in range(B):
        idx = slots[b].long()
        kb = k[idx]  # (Lk, H, D)
        vb = v[idx]
        for h in range(H):
            qbh = q[b, :, h, :].float()  # (Lq, D)
            kbh = kb[:, h, :].float()    # (Lk, D)
            vbh = vb[:, h, :].float()
            scores = (qbh @ kbh.T) * sm_scale
            weights = torch.softmax(scores, dim=-1)
            out[b, h] = (weights @ vbh).to(q.dtype)
    return out


# ─── Test parameters ───────────────────────────────────────────────
#
# (B, H, Lq, Lk, D, CACHE). The batched variants block by BLK_B=2, BLK_H=4 and
# step the query loop by BLOCK_Q, so B is a multiple of 2, H of 4, and Lq/Lk of
# the block sizes (the explicit loops assume even tiling, like the source draft).
PARAMS = [
    (2, 4, 16, 16, 64, 256),
    (4, 8, 32, 32, 64, 512),
]

VARIANTS = {
    "td": _paged_attn_kernel_NHD_td,
    "sa": _paged_attn_kernel_NHD_sa,
}


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_inputs(B, H, Lq, Lk, D, CACHE, device, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, Lq, H, D, device=device, dtype=dtype)
    k = torch.randn(CACHE, H, D, device=device, dtype=dtype)
    v = torch.randn(CACHE, H, D, device=device, dtype=dtype)
    # distinct random physical slots per request
    slots = torch.stack(
        [torch.randperm(CACHE, device=device)[:Lk] for _ in range(B)]
    ).to(torch.int32)
    return q, k, v, slots


class TestPagedAttnEquivalence:

    @pytest.mark.parametrize("B,H,Lq,Lk,D,CACHE", PARAMS)
    @pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
    @pytest.mark.parametrize(
        "dtype", [torch.float16], ids=lambda d: str(d).split(".")[-1],
    )
    def test_variant_vs_original(self, device, variant, B, H, Lq, Lk, D, CACHE, dtype):
        q, k, v, slots = _make_inputs(B, H, Lq, Lk, D, CACHE, device, dtype)
        o_orig = paged_attention(q, k, v, slots)
        o_var = paged_attention(q, k, v, slots, kernel_fn=VARIANTS[variant])
        torch.testing.assert_close(o_var, o_orig, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("B,H,Lq,Lk,D,CACHE", PARAMS)
    def test_original_vs_torch(self, device, B, H, Lq, Lk, D, CACHE):
        q, k, v, slots = _make_inputs(B, H, Lq, Lk, D, CACHE, device, torch.float16)
        o_orig = paged_attention(q, k, v, slots, kernel_fn=_paged_attn_kernel)
        ref = torch_reference(q, k, v, slots)
        torch.testing.assert_close(o_orig, ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("B,H,Lq,Lk,D,CACHE", PARAMS)
    @pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
    def test_variant_vs_torch(self, device, variant, B, H, Lq, Lk, D, CACHE):
        q, k, v, slots = _make_inputs(B, H, Lq, Lk, D, CACHE, device, torch.float16)
        o_var = paged_attention(q, k, v, slots, kernel_fn=VARIANTS[variant])
        ref = torch_reference(q, k, v, slots)
        torch.testing.assert_close(o_var, ref, atol=2e-2, rtol=2e-2)

    def test_td_matches_sa(self, device):
        """The two gather expressions have identical semantics."""
        q, k, v, slots = _make_inputs(2, 4, 16, 16, 64, 256, device, torch.float16)
        o_td = paged_attention(q, k, v, slots, kernel_fn=_paged_attn_kernel_NHD_td)
        o_sa = paged_attention(q, k, v, slots, kernel_fn=_paged_attn_kernel_NHD_sa)
        torch.testing.assert_close(o_td, o_sa, atol=1e-3, rtol=1e-3)

# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for prefill attention kernel.

Run: pytest kernels/prefill_attention/test_equivalence.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.prefill_attention.wrapper import context_attention_fwd
from kernels.prefill_attention.original import _fwd_kernel
from kernels.prefill_attention.block_ptr import _fwd_kernel_block_ptr


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    b_start_loc: torch.Tensor, b_seq_len: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """Pure PyTorch SDPA — the ground truth."""
    Lk = q.shape[-1]
    sm_scale = 1.0 / (Lk ** 0.5)
    batch = b_seq_len.shape[0]
    kv_group_num = q.shape[1] // k.shape[1]
    o = torch.zeros_like(q)

    for b in range(batch):
        start = b_start_loc[b].item()
        seq_len = b_seq_len[b].item()
        for h in range(q.shape[1]):
            kv_h = h // kv_group_num
            q_b = q[start:start + seq_len, h, :].float()  # [S, D]
            k_b = k[start:start + seq_len, kv_h, :].float()
            v_b = v[start:start + seq_len, kv_h, :].float()
            scores = q_b @ k_b.T * sm_scale  # [S, S]
            if is_causal:
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
                scores = scores.masked_fill(causal_mask == 0, -1e9)
            weights = torch.softmax(scores, dim=-1)
            o[start:start + seq_len, h, :] = (weights @ v_b).to(q.dtype)
    return o


# ─── Test Parameters ───────────────────────────────────────────────

SEQ_LENS_CONFIGS = [
    [16],
    [32],
    [64],
    [16, 32],
    [16, 24, 32],
]

HEAD_DIMS = [64]
NUM_Q_HEADS = [4]
NUM_KV_HEADS = [4]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_inputs(seq_lens, num_q_heads, num_kv_heads, head_dim, device, dtype):
    torch.manual_seed(42)
    total_tokens = sum(seq_lens)
    batch = len(seq_lens)
    q = torch.randn(total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    b_seq_len = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    b_start_loc = torch.zeros(batch, device=device, dtype=torch.int32)
    for i in range(1, batch):
        b_start_loc[i] = b_start_loc[i - 1] + seq_lens[i - 1]
    max_input_len = max(seq_lens)
    return q, k, v, b_start_loc, b_seq_len, max_input_len


class TestPrefillAttentionEquivalence:

    @pytest.mark.parametrize("seq_lens", SEQ_LENS_CONFIGS)
    @pytest.mark.parametrize("head_dim", HEAD_DIMS)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence_causal(self, device, seq_lens, head_dim, dtype):
        num_q_heads, num_kv_heads = 4, 4
        q, k, v, b_start_loc, b_seq_len, max_len = _make_inputs(
            seq_lens, num_q_heads, num_kv_heads, head_dim, device, dtype
        )

        o_raw = torch.zeros_like(q)
        o_blk = torch.zeros_like(q)

        context_attention_fwd(q, k, v, o_raw, b_start_loc, b_seq_len, max_len, is_causal=True)
        context_attention_fwd(q, k, v, o_blk, b_start_loc, b_seq_len, max_len, is_causal=True, kernel_fn=_fwd_kernel_block_ptr)

        torch.testing.assert_close(o_raw, o_blk, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("seq_lens", [[16], [16, 32]])
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence_non_causal(self, device, seq_lens, dtype):
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        q, k, v, b_start_loc, b_seq_len, max_len = _make_inputs(
            seq_lens, num_q_heads, num_kv_heads, head_dim, device, dtype
        )

        o_raw = torch.zeros_like(q)
        o_blk = torch.zeros_like(q)

        context_attention_fwd(q, k, v, o_raw, b_start_loc, b_seq_len, max_len, is_causal=False)
        context_attention_fwd(q, k, v, o_blk, b_start_loc, b_seq_len, max_len, is_causal=False, kernel_fn=_fwd_kernel_block_ptr)

        torch.testing.assert_close(o_raw, o_blk, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("seq_lens", [[16], [16, 24]])
    def test_correctness_vs_pytorch(self, device, seq_lens):
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        dtype = torch.float16
        q, k, v, b_start_loc, b_seq_len, max_len = _make_inputs(
            seq_lens, num_q_heads, num_kv_heads, head_dim, device, dtype
        )

        ref = attention_reference(q, k, v, b_start_loc, b_seq_len, is_causal=True)
        o_raw = torch.zeros_like(q)
        o_blk = torch.zeros_like(q)

        context_attention_fwd(q, k, v, o_raw, b_start_loc, b_seq_len, max_len, is_causal=True)
        context_attention_fwd(q, k, v, o_blk, b_start_loc, b_seq_len, max_len, is_causal=True, kernel_fn=_fwd_kernel_block_ptr)

        torch.testing.assert_close(o_raw, ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(o_blk, ref, atol=1e-2, rtol=1e-2)

    def test_gqa(self, device):
        """GQA: num_q_heads > num_kv_heads."""
        seq_lens = [16]
        num_q_heads, num_kv_heads, head_dim = 8, 2, 64
        dtype = torch.bfloat16
        q, k, v, b_start_loc, b_seq_len, max_len = _make_inputs(
            seq_lens, num_q_heads, num_kv_heads, head_dim, device, dtype
        )

        o_raw = torch.zeros_like(q)
        o_blk = torch.zeros_like(q)

        context_attention_fwd(q, k, v, o_raw, b_start_loc, b_seq_len, max_len, is_causal=True)
        context_attention_fwd(q, k, v, o_blk, b_start_loc, b_seq_len, max_len, is_causal=True, kernel_fn=_fwd_kernel_block_ptr)

        torch.testing.assert_close(o_raw, o_blk, atol=1e-2, rtol=1e-2)

    def test_single_token_sequence(self, device):
        """Edge case: seq_len=16 (minimum for tl.dot)."""
        seq_lens = [16]
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        dtype = torch.bfloat16
        q, k, v, b_start_loc, b_seq_len, max_len = _make_inputs(
            seq_lens, num_q_heads, num_kv_heads, head_dim, device, dtype
        )

        o_raw = torch.zeros_like(q)
        o_blk = torch.zeros_like(q)

        context_attention_fwd(q, k, v, o_raw, b_start_loc, b_seq_len, max_len, is_causal=True)
        context_attention_fwd(q, k, v, o_blk, b_start_loc, b_seq_len, max_len, is_causal=True, kernel_fn=_fwd_kernel_block_ptr)

        torch.testing.assert_close(o_raw, o_blk, atol=1e-2, rtol=1e-2)

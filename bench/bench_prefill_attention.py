# SPDX-License-Identifier: Apache-2.0
"""Benchmark Prefill Attention (SDPA) Triton kernels."""

import torch
import triton.testing

from kernels.prefill_attention.wrapper import context_attention_fwd
from kernels.prefill_attention.block_ptr import _fwd_kernel_block_ptr


def bench_prefill_attention():
    """Benchmark raw-pointer vs block-pointer Prefill Attention kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    seq_lens_configs = [[16], [32], [64]]
    head_dim = 64
    num_q_heads = 4
    num_kv_heads = 4

    print("Prefill Attention Benchmark")
    print("=" * 60)
    print(f"{'Seq Len':<12} {'Raw (ms)':<12} {'BlockPtr (ms)':<14} {'Slowdown':<10}")
    print("-" * 60)

    for seq_lens in seq_lens_configs:
        torch.manual_seed(42)
        total_tokens = sum(seq_lens)
        batch = len(seq_lens)
        q = torch.randn(total_tokens, num_q_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
        b_seq_len = torch.tensor(seq_lens, device=device, dtype=torch.int32)
        b_start_loc = torch.zeros(batch, device=device, dtype=torch.int32)
        for i in range(1, batch):
            b_start_loc[i] = b_start_loc[i - 1] + seq_lens[i - 1]
        max_input_len = max(seq_lens)

        ms_raw = triton.testing.do_bench(lambda: context_attention_fwd(q.clone(), k.clone(), v.clone(), torch.zeros_like(q), b_start_loc, b_seq_len, max_input_len, is_causal=True))
        ms_blk = triton.testing.do_bench(lambda: context_attention_fwd(q.clone(), k.clone(), v.clone(), torch.zeros_like(q), b_start_loc, b_seq_len, max_input_len, is_causal=True, kernel_fn=_fwd_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"{str(seq_lens):<12} {ms_raw:<12.3f} {ms_blk:<14.3f} {slowdown:<10.2f}x")


if __name__ == "__main__":
    bench_prefill_attention()

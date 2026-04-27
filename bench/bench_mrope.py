# SPDX-License-Identifier: Apache-2.0
"""Benchmark MRoPE Triton kernels."""

import torch
import triton.testing

from kernels.mrope.wrapper import triton_mrope
from kernels.mrope.block_ptr import _triton_mrope_forward_block_ptr


def bench_mrope():
    """Benchmark raw-pointer vs block-pointer MRoPE kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    head_sizes = [64, 128]
    num_tokens = 16
    num_q_heads = 4
    num_kv_heads = 4

    print("MRoPE Benchmark")
    print("=" * 60)
    print(f"{'Head Size':<12} {'Raw (ms)':<12} {'BlockPtr (ms)':<14} {'Slowdown':<10}")
    print("-" * 60)

    for head_size in head_sizes:
        torch.manual_seed(42)
        q = torch.randn(num_tokens, num_q_heads * head_size, device=device, dtype=torch.bfloat16)
        k = torch.randn(num_tokens, num_kv_heads * head_size, device=device, dtype=torch.bfloat16)
        half_rd = head_size // 2
        cos = torch.randn(3, num_tokens, half_rd, device=device, dtype=torch.bfloat16)
        sin = torch.randn(3, num_tokens, half_rd, device=device, dtype=torch.bfloat16)
        t = half_rd // 3
        h = half_rd // 3
        w = half_rd - t - h
        mrope_section = [t, h, w]
        rotary_dim = head_size

        ms_raw = triton.testing.do_bench(lambda: triton_mrope(q.clone(), k.clone(), cos, sin, mrope_section, head_size, rotary_dim, False))
        ms_blk = triton.testing.do_bench(lambda: triton_mrope(q.clone(), k.clone(), cos, sin, mrope_section, head_size, rotary_dim, False, kernel_fn=_triton_mrope_forward_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"{head_size:<12} {ms_raw:<12.3f} {ms_blk:<14.3f} {slowdown:<10.2f}x")


if __name__ == "__main__":
    bench_mrope()

# SPDX-License-Identifier: Apache-2.0
"""Benchmark Merge Attention States Triton kernels."""

import torch
import triton.testing

from kernels.merge_attn_states.wrapper import merge_attn_states
from kernels.merge_attn_states.block_ptr import _merge_attn_states_kernel_block_ptr


def bench_merge_attn_states():
    """Benchmark raw-pointer vs block-pointer Merge Attention States kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    head_sizes = [64, 128]
    num_tokens = 8
    num_heads = 4

    print("Merge Attention States Benchmark")
    print("=" * 60)
    print(f"{'Head Size':<12} {'Raw (ms)':<12} {'BlockPtr (ms)':<14} {'Slowdown':<10}")
    print("-" * 60)

    for head_size in head_sizes:
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        ms_raw = triton.testing.do_bench(lambda: merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse))
        ms_blk = triton.testing.do_bench(lambda: merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse, kernel_fn=_merge_attn_states_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"{head_size:<12} {ms_raw:<12.3f} {ms_blk:<14.3f} {slowdown:<10.2f}x")


if __name__ == "__main__":
    bench_merge_attn_states()

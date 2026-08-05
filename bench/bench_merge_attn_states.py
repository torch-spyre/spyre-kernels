# SPDX-License-Identifier: Apache-2.0
"""Benchmark Merge Attention States Triton kernels."""

import torch

from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.vllm.merge_attn_states.block_ptr import _merge_attn_states_kernel_block_ptr
from kernels.vllm.merge_attn_states.wrapper import merge_attn_states


def bench_merge_attn_states():
    """Benchmark raw-pointer vs block-pointer Merge Attention States kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    device = torch.device("cuda")
    head_sizes = [64, 128]
    num_tokens = 8
    num_heads = 4

    print("Merge Attention States Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    for head_size in head_sizes:
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        ms_raw = stable_bench(lambda: merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse))
        ms_blk = stable_bench(lambda: merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse, kernel_fn=_merge_attn_states_kernel_block_ptr))

        print(format_result(head_size, ms_raw, ms_blk))


if __name__ == "__main__":
    bench_merge_attn_states()

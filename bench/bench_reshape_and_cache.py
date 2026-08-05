# SPDX-License-Identifier: Apache-2.0
"""Benchmark Reshape and Cache Triton kernels."""

import torch

from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.vllm.reshape_and_cache.block_ptr import _reshape_and_cache_kernel_block_ptr
from kernels.vllm.reshape_and_cache.wrapper import reshape_and_cache


def bench_reshape_and_cache():
    """Benchmark raw-pointer vs block-pointer Reshape and Cache kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    device = torch.device("cuda")
    num_tokens = 16
    num_heads = 4
    head_size = 64
    block_size = 16
    num_blocks = 8

    print("Reshape and Cache Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    torch.manual_seed(42)
    key = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, device=device, dtype=torch.bfloat16)
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, device=device, dtype=torch.bfloat16)
    max_slots = num_blocks * block_size
    slot_mapping = torch.randint(0, max_slots, (num_tokens,), device=device, dtype=torch.int64)

    ms_raw = stable_bench(lambda: reshape_and_cache(key.clone(), value.clone(), key_cache.clone(), value_cache.clone(), slot_mapping))
    ms_blk = stable_bench(lambda: reshape_and_cache(key.clone(), value.clone(), key_cache.clone(), value_cache.clone(), slot_mapping, kernel_fn=_reshape_and_cache_kernel_block_ptr))

    print(format_result("16x4x64", ms_raw, ms_blk))


if __name__ == "__main__":
    bench_reshape_and_cache()

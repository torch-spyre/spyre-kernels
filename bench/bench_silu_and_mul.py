# SPDX-License-Identifier: Apache-2.0
"""Benchmark SwiGLU (silu_and_mul) Triton kernels."""

import torch
import triton.testing

from kernels.silu_and_mul.wrapper import silu_and_mul
from kernels.silu_and_mul.block_ptr import _swiglustep_and_mul_kernel_block_ptr


def bench_silu_and_mul():
    """Benchmark raw-pointer vs block-pointer SwiGLU kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    half_hidden_sizes = [4096, 5120]
    batch_size = 128

    print("SwiGLU (silu_and_mul) Benchmark")
    print("=" * 60)
    print(f"{'Half Hidden':<12} {'Raw (ms)':<12} {'BlockPtr (ms)':<14} {'Slowdown':<10}")
    print("-" * 60)

    for d in half_hidden_sizes:
        x = torch.randn(batch_size, 2 * d, device=device, dtype=torch.bfloat16)

        ms_raw = triton.testing.do_bench(lambda: silu_and_mul(x))
        ms_blk = triton.testing.do_bench(lambda: silu_and_mul(x, kernel_fn=_swiglustep_and_mul_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"{d:<12} {ms_raw:<12.3f} {ms_blk:<14.3f} {slowdown:<10.2f}x")


if __name__ == "__main__":
    bench_silu_and_mul()

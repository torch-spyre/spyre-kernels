# SPDX-License-Identifier: Apache-2.0
"""Benchmark RMSNorm Triton kernels."""

import torch
import triton.testing

from kernels.rms_norm.wrapper import rms_norm
from kernels.rms_norm.block_ptr import _rms_norm_kernel_block_ptr


def bench_rms_norm():
    """Benchmark raw-pointer vs block-pointer RMSNorm kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    hidden_sizes = [4096, 5120]
    batch_size = 128

    print("RMSNorm Benchmark")
    print("=" * 60)
    print(f"{'Hidden Size':<12} {'Raw (ms)':<12} {'BlockPtr (ms)':<14} {'Slowdown':<10}")
    print("-" * 60)

    for hidden_size in hidden_sizes:
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)

        ms_raw = triton.testing.do_bench(lambda: rms_norm(x, w))
        ms_blk = triton.testing.do_bench(lambda: rms_norm(x, w, kernel_fn=_rms_norm_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"{hidden_size:<12} {ms_raw:<12.3f} {ms_blk:<14.3f} {slowdown:<10.2f}x")


if __name__ == "__main__":
    bench_rms_norm()

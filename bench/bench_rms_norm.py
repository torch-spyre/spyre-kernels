# SPDX-License-Identifier: Apache-2.0
"""Benchmark RMSNorm Triton kernels."""

import torch

from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.rms_norm.block_ptr import _rms_norm_kernel_block_ptr
from kernels.rms_norm.wrapper import rms_norm


def bench_rms_norm():
    """Benchmark raw-pointer vs block-pointer RMSNorm kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    device = torch.device("cuda")
    hidden_sizes = [4096, 5120]
    batch_size = 128

    print("RMSNorm Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    for hidden_size in hidden_sizes:
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)

        ms_raw = stable_bench(lambda: rms_norm(x, w))
        ms_blk = stable_bench(lambda: rms_norm(x, w, kernel_fn=_rms_norm_kernel_block_ptr))

        print(format_result(hidden_size, ms_raw, ms_blk))


if __name__ == "__main__":
    bench_rms_norm()

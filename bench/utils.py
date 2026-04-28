# SPDX-License-Identifier: Apache-2.0
"""Shared benchmarking utilities for consistent, reproducible results."""

import os
import subprocess
import statistics

import torch
import triton.testing


BENCH_HEADER = (
    f"{'Config':<12} {'Raw median [p20-p80]':<28} "
    f"{'BlkPtr median [p20-p80]':<28} {'Slowdown':<10}"
)
BENCH_SEP = "-" * 80

N_TRIALS = int(os.environ.get("BENCH_TRIALS", 5))


def gpu_warmup():
    """Run dummy work to stabilize GPU clocks before benchmarking."""
    if not torch.cuda.is_available():
        return
    x = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    for _ in range(20):
        x = x @ x
    torch.cuda.synchronize()


def lock_gpu_clocks():
    """Try to lock GPU clocks to max stable frequency for consistent benchmarks.

    Requires root/sudo. Silently skips if unavailable.
    Returns True if clocks were locked (caller should unlock on exit).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return False
        max_clock = result.stdout.strip().split("\n")[0].strip()
        result = subprocess.run(
            ["sudo", "nvidia-smi", "-lgc", max_clock],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            print(f"  GPU clocks locked at {max_clock} MHz")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def unlock_gpu_clocks():
    """Reset GPU clocks to default. Silently skips if unavailable."""
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "-rgc"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _single_bench(fn, warmup=100, rep=500, quantiles=(0.5, 0.2, 0.8)):
    """Single do_bench call. Returns (median, p20, p80)."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=list(quantiles))


def stable_bench(fn, warmup=100, rep=500, quantiles=(0.5, 0.2, 0.8), n_trials=N_TRIALS):
    """Run multiple independent do_bench trials and return trimmed-mean median.

    Runs n_trials independent measurements, drops the highest and lowest
    median, and averages the rest. This filters out outlier runs caused by
    transient system activity.

    Returns (median_ms, p20_ms, p80_ms) aggregated across trials.
    """
    if n_trials < 3:
        return _single_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)

    medians = []
    p20s = []
    p80s = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        med, p20, p80 = _single_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
        medians.append(med)
        p20s.append(p20)
        p80s.append(p80)

    medians.sort()
    p20s.sort()
    p80s.sort()
    trimmed_med = statistics.mean(medians[1:-1])
    trimmed_p20 = statistics.mean(p20s[1:-1])
    trimmed_p80 = statistics.mean(p80s[1:-1])
    return trimmed_med, trimmed_p20, trimmed_p80


def format_result(name, ms_raw, ms_blk):
    """Format a raw-vs-blockptr comparison line with percentile bands.

    ms_raw and ms_blk are (median, p20, p80) tuples from stable_bench.
    """
    med_raw, p20_raw, p80_raw = ms_raw
    med_blk, p20_blk, p80_blk = ms_blk
    slowdown = med_blk / med_raw
    raw_str = f"{med_raw:>7.3f} [{p20_raw:.3f}-{p80_raw:.3f}]"
    blk_str = f"{med_blk:>7.3f} [{p20_blk:.3f}-{p80_blk:.3f}]"
    return f"{str(name):<12} {raw_str:<28} {blk_str:<28} {slowdown:.2f}x"

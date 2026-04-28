#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run all Triton kernel benchmarks."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Run all benchmark scripts and collect results."""
    parser = argparse.ArgumentParser(description="Run Triton kernel benchmark suite")
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of times to run the full suite (default: 1)",
    )
    parser.add_argument(
        "--lock-clocks", action="store_true",
        help="Lock GPU clocks to max frequency for stability (requires sudo)",
    )
    args = parser.parse_args()

    bench_dir = Path(__file__).parent
    bench_files = sorted(bench_dir.glob("bench_*.py"))

    if not bench_files:
        print("No benchmark files found!")
        return

    clocks_locked = False
    if args.lock_clocks:
        from bench.utils import lock_gpu_clocks, unlock_gpu_clocks
        clocks_locked = lock_gpu_clocks()
        if not clocks_locked:
            print("WARNING: Could not lock GPU clocks (needs sudo). Continuing anyway.")

    try:
        for trial in range(1, args.trials + 1):
            if args.trials > 1:
                print()
                print("#" * 80)
                print(f"  TRIAL {trial}/{args.trials}")
                print("#" * 80)

            print()
            print("=" * 80)
            print("TRITON KERNEL BENCHMARK SUITE")
            print("=" * 80)
            print()

            results = {}

            for bench_file in bench_files:
                kernel_name = bench_file.stem.replace("bench_", "")
                print(f"Running {kernel_name} benchmark...")
                print("-" * 80)

                result = subprocess.run(
                    [sys.executable, str(bench_file)],
                    capture_output=True,
                    text=True,
                )

                if result.stdout:
                    print(result.stdout)
                    results[kernel_name] = result.stdout.strip()

                if result.returncode != 0:
                    print(f"ERROR: {kernel_name} benchmark failed")
                    if result.stderr:
                        print(f"stderr: {result.stderr}")
                print()

            print("=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Ran {len(results)} benchmarks")

            for name, output in results.items():
                if "CUDA not available" in output:
                    print(f"  - {name}: SKIPPED (no CUDA)")
                else:
                    print(f"  - {name}: OK")
    finally:
        if clocks_locked:
            unlock_gpu_clocks()
            print("\n  GPU clocks unlocked.")


if __name__ == "__main__":
    main()

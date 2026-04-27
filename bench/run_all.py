#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run all Triton kernel benchmarks."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run all benchmark scripts and collect results."""
    bench_dir = Path(__file__).parent
    bench_files = sorted(bench_dir.glob("bench_*.py"))

    if not bench_files:
        print("No benchmark files found!")
        return

    print("=" * 70)
    print("TRITON KERNEL BENCHMARK SUITE")
    print("=" * 70)
    print()

    results = {}

    for bench_file in bench_files:
        kernel_name = bench_file.stem.replace("bench_", "")
        print(f"Running {kernel_name} benchmark...")
        print("-" * 70)

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

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ran {len(results)} benchmarks")

    # Check for CUDA availability warning
    for name, output in results.items():
        if "CUDA not available" in output:
            print(f"  - {name}: SKIPPED (no CUDA)")
        else:
            print(f"  - {name}: OK")


if __name__ == "__main__":
    main()

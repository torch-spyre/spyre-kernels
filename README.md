# spyre-kernels

Spyre-aware Triton kernels for IBM Spyre/AIU accelerators.

## Overview

This repo is the canonical home for authoring, validating, and tracking Triton kernels targeting IBM Spyre hardware. Kernels here reflect Spyre's execution model: fixed 32 cores, explicit inner-core tiling that fits scratchpad, and descriptor-based IO via `tl.make_tensor_descriptor`.

## Validation Tiers

Every kernel should be validated across four tiers:

| Tier | Name | Description |
|------|------|-------------|
| **T0** | Numerical equivalence | Matches a PyTorch/reference implementation within tolerance, validated on GPU |
| **T1** | Spyre-shape compliance | Tiles fit scratchpad, grid fits 32 cores, runtime-arg agnostic |
| **T2** | KTIR/Spyre validation | Output matches reference on `ktir_cpu` and/or real Spyre hardware |
| **T3** | Human-reviewed | Reviewed by a domain expert and signed off |

## Project Structure

```
tritokti/
├── kernels/                   # Kernel implementations
│   ├── <name>/
│   │   ├── original.py        # Original kernel (e.g., from vLLM)
│   │   ├── block_ptr.py       # Block-pointer version (deprecated)
│   │   ├── spyre_aware.py     # Spyre-aware Triton kernel
│   │   ├── wrapper.py         # Python launcher
│   │   └── kernel.ktir        # KTIR kernel (optional)
│
├── tests/
│   ├── triton/                # GPU equivalence tests (T0)
│   │   └── test_<name>.py
│   └── ktir/                  # KTIR/CPU validation tests (T2)
│       └── test_<name>.py
│
├── bench/
│   ├── utils.py               # Shared benchmarking utilities
│   ├── bench_<name>.py        # Per-kernel benchmarks
│   └── run_all.py             # Run all benchmarks
│
├── scripts/
│   └── fetch_originals.py     # Extract kernels from upstream sources
│
└── kernels.json               # Kernel registry
```

## Quick Start

```bash
# Install with test dependencies
uv sync --extra test

# GPU equivalence tests (T0)
pytest tests/triton/ -v

# KTIR validation tests (T2)
pytest tests/ktir/ -v

# All tests
pytest tests/ -v
```

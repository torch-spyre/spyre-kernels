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
spyre-kernels/
├── kernels/                       # Kernel implementations
│   ├── <name>/
│   │   ├── original.py            # Original kernel (e.g., from vLLM)
│   │   ├── block_ptr.py           # Block-pointer version (deprecated)
│   │   ├── tensor_descriptor.py   # Tensor-descriptor version
│   │   ├── spyre_aware.py         # Spyre-aware version
│   │   ├── lower.py               # KTIR lowering driver
│   │   ├── wrapper.py             # Python launcher
│   │   └── <variant>.ktir         # Generated KTIR, one per lowered variant (e.g. tensor_descriptor.ktir)
│
├── tests/
│   ├── triton/                    # GPU equivalence tests (T0)
│   │   └── test_<name>.py
│   └── ktir/                      # KTIR/CPU validation tests (T2)
│       └── test_<name>.py
│
├── bench/
│   ├── utils.py                   # Shared benchmarking utilities
│   ├── bench_<name>.py            # Per-kernel benchmarks
│   └── run_all.py                 # Run all benchmarks
│
├── scripts/
│   └── fetch_originals.py         # Extract kernels from upstream sources
│
└── kernels.json                   # Kernel registry
```

## Quick Start

For running the kernels and validating committed KTIR on the `ktir_cpu` simulator:

```bash
uv sync --extra test
```

Then run the tests:

```bash
# GPU equivalence tests (T0)
.venv/bin/python -m pytest tests/triton/ -v

# KTIR validation tests (T2) — committed KTIR + ktir_cpu simulator
.venv/bin/python -m pytest tests/ktir/ -v

# All tests
.venv/bin/python -m pytest tests/ -v
```

## Authoring kernels (regenerate KTIR)

Lowering a Triton kernel to KTIR (`scripts/gen_ktir.py`) needs the
**spyre-enabled** Triton build from
[`torch-spyre/triton`](https://github.com/torch-spyre/triton) — only kernel
authors need this. It builds from source (a few minutes) and requires a GitHub
token for the LLVM fetch. Install it **into your existing project venv**:

```bash
GIT_PAT=$(gh auth token) scripts/install-ktir-gen.sh
```

Then regenerate KTIR:

```bash
# Regenerate every kernel that has a lower.py driver:
.venv/bin/python scripts/gen_ktir.py

# CI drift guard — fail if any committed kernel.ktir is stale:
.venv/bin/python scripts/gen_ktir.py --check
```

A plain `uv sync` switches the venv back to stock PyPI Triton (the base tier).
Run `gen_ktir.py` with `.venv/bin/python` directly — never `uv run`, which
re-syncs to the base tier and drops the spyre backend.
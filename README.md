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
│   ├── vllm/                       # Hand-authored kernels (source: vLLM)
│   │   └── <name>/
│   │       ├── original.py            # Original kernel (e.g., from vLLM)
│   │       ├── block_ptr.py           # Block-pointer version (deprecated)
│   │       ├── tensor_descriptor.py   # Tensor-descriptor version
│   │       ├── spyre_aware.py         # Spyre-aware version
│   │       ├── lower.py               # KTIR lowering driver
│   │       ├── wrapper.py             # Python launcher
│   │       └── <variant>.ktir         # Generated KTIR, one per lowered variant (e.g. tensor_descriptor.ktir)
│   │
│   └── models/                     # Kernels traced out of real models
│       └── Meta-Llama-3.1-8B-Instruct/
│           └── <op_name>_spyre/
│               ├── wrapper.py              # SIGNATURE + make_inputs()/run() oracle + launcher
│               ├── triton_kernel.py         # Traced Triton kernel source
│               ├── test_ktir_<op_name>.py   # KTIR/CPU validation test (T2), see Quick Start
│               └── <stage>.ktir             # Generated KTIR — one file for a single-stage
│                                            #   kernel, or one per stage for bundled/
│                                            #   multi-stage kernels (e.g. linear, silu)
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

# KTIR validation tests (T2) for kernels traced out of real models
# (not part of `tests/` — run explicitly, for all ops or a single op)
.venv/bin/python -m pytest kernels/models/ -v
.venv/bin/python -m pytest kernels/models/Meta-Llama-3.1-8B-Instruct/torch.add.1_spyre/ -v
```

## Authoring kernels (regenerate KTIR)

Lowering a Triton kernel to KTIR (`scripts/gen_ktir.py`) needs the
**spyre-enabled** Triton build from
[`torch-spyre/triton`](https://github.com/torch-spyre/triton) — only kernel
authors need it, and only for this one task. Rather than install it into your
`.venv` (which would mutate the shared base-tier env), layer it in for the
single run with `uv run --with`. It builds from source on first use (a few
minutes; needs a GitHub token for the LLVM fetch) and runs in a separate
ephemeral env, leaving `.venv` on stock PyPI Triton:

```bash
SPYRE_TRITON="triton @ git+https://github.com/torch-spyre/triton@5b467467c883c53ec7a8a89f9e89cfd55241034b"

# Regenerate every variant of every kernel with a lower.py driver:
GIT_PAT=$(gh auth token) uv run --with "$SPYRE_TRITON" python scripts/gen_ktir.py

# One kernel / one variant:
GIT_PAT=$(gh auth token) uv run --with "$SPYRE_TRITON" python scripts/gen_ktir.py rms_norm
GIT_PAT=$(gh auth token) uv run --with "$SPYRE_TRITON" python scripts/gen_ktir.py rms_norm:tensor_descriptor

# CI drift guard — fail if any committed <variant>.ktir is stale:
GIT_PAT=$(gh auth token) uv run --with "$SPYRE_TRITON" python scripts/gen_ktir.py --check
```

# Project Information

## For Claude Code / Context Management

### What This Project Does

Port Triton kernels from vLLM to run on IBM Spyre/AIU accelerators via a three-phase pipeline:

```
vLLM Triton (raw pointers) → Block-pointer Triton → KTIR MLIR → Spyre hardware
```

**Why:** Spyre uses a tiled "stick" memory layout (128-byte aligned). Raw pointer arithmetic from GPU Triton kernels doesn't map to this. Block pointers abstract memory access into structured operations that can be lowered to Spyre's model.

### Current State

- **9 kernels fully converted** through Phase 3 (KTIR)
- **773 GPU tests** validate block-pointer equivalence
- **18 CPU tests** validate KTIR correctness
- **0 performance regressions** — block-pointer kernels are 0.97x–1.34x of original (all within 2x threshold)

### Key Files

| File | Purpose |
|------|---------|
| `kernels.json` | Kernel registry — vLLM source mapping |
| `scripts/fetch_originals.py` | Extract kernels from vLLM (auto-gen `original.py` files) |
| `kernels/<name>/original.py` | Verbatim vLLM kernel (DO NOT EDIT) |
| `kernels/<name>/block_ptr.py` | Block-pointer conversion |
| `kernels/<name>/kernel.ktir.mlir` | KTIR output |
| `tests/triton/test_*.py` | GPU equivalence tests |
| `tests/ktir/test_*.py` | CPU validation tests |
| `bench/bench_*.py` | Performance benchmarks |

### How to Add a Kernel

1. Add entry to `kernels.json`
2. Run `python scripts/fetch_originals.py`
3. Create `block_ptr.py` conversion
4. Add wrapper, tests, benchmark
5. Run `pytest tests/triton/` and `python bench/run_all.py`

### Memory Types

This project uses file-based memory (`/home/ohad/.claude/projects/-home-ohad-Projects-tritokti/memory/`) for:

- **User context:** Ohad's role, expertise, preferences
- **Feedback:** How to approach work (e.g., "prefer integrated tests over mocks")
- **Project state:** Active initiatives, decisions, constraints
- **References:** External resources (Linear projects, dashboards)

Memory files are indexed in `MEMORY.md`. Update when learning:
- User preferences ("don't summarize diffs")
- Project decisions ("merge freeze after 2026-03-05")
- External resources ("Grafana dashboard at grafana.internal/d/api-latency")

### Commands & Permissions

**Allowed in project settings:**
- `pytest` — run tests
- `python scripts/*` — run maintenance scripts
- `git status`, `git diff`, `git log` — repo state

**Require confirmation:**
- `git push` — pushes to remote
- `rm -rf` — destructive operations
- Any write to `external/` — third-party code

### Running Context

- **Venv:** `.venv-office/` (primary), `.venv/` (dev)
- **Test runner:** `pytest tests/` (auto-discovers `test_*.py`)
- **Benchmark runner:** `python bench/run_all.py`

### vLLM Provenance

All kernels extracted from vLLM commit [`cde8d2471026`](https://github.com/vllm-project/vllm/commit/cde8d2471026).

To verify sync:
```bash
python scripts/fetch_originals.py --diff
# Should report 0 differences
```

### KTIR Limitations to Remember

When reviewing KTIR kernels:

1. No `arith.cmpf` — uses `arith.cmpi` on floats (interpreter quirk)
2. No `arith.minimumf`/`arith.maximumf` — uses `-max(-a, -b)` pattern
3. No `tensor.generate` — tensors pre-computed on host
4. No multi-result `scf.for` — algorithms restructured to single-result
5. f32 memref output may not dispatch — truncates to f16

These are known workarounds, not bugs.

### Test/Validation Layers

| Layer | What it validates | Where |
|-------|------------------|-------|
| GPU (Triton) | Block-ptr == raw-ptr numerically | `tests/triton/` |
| CPU (KTIR) | KTIR == NumPy reference | `tests/ktir/` |
| Benchmark | Block-ptr not >2x slower | `bench/` |
| Integration | vLLM generation unchanged | (future CI) |

### Decision Log

- **Block-pointer API:** Using `tl.make_block_ptr` (stable) vs `tl.make_tensor_descriptor` (newer). Current choice: `make_block_ptr` for compatibility.
- **Excluded kernels:** Top-k/Top-p, decode stage 1, unified attention — too complex for KTIR (iterative/indirect access). May require CUDA fallback.
- **Test structure:** Separate `triton/` (GPU) and `ktir/` (CPU) — different backends, different dependencies.

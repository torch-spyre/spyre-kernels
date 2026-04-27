# Project Plan

## What We Did

### Phase 1: Kernel Inventory
- Identified 13 core Triton kernels in vLLM's dense inference path
- Extracted 9 kernels that are tractable for KTIR lowering
- Created `kernels.json` registry with vLLM source mapping

### Phase 2: Block-Pointer Conversion (Complete: 9/9 kernels)
Converted all raw-pointer Triton kernels to block-pointer form:

| Kernel | Tests | Benchmark |
|--------|-------|-----------|
| RMSNorm | 98 GPU, 2 CPU | Yes |
| SwiGLU | 98 GPU, 2 CPU | Yes |
| Ranks | 64 GPU, 2 CPU | Yes |
| Log-softmax | 190 GPU, 2 CPU | Yes |
| Decode softmax+reduceV | 85 GPU, 2 CPU | Yes |
| Merge attention states | 116 GPU, 2 CPU | Yes |
| MRoPE | 52 GPU, 2 CPU | Yes |
| Reshape/cache | 52 GPU, 2 CPU | Yes |
| Prefill attention | 18 GPU, 2 CPU | Yes |

**Total:** 773 GPU tests passing, 18 CPU tests passing

### Phase 3: KTIR Lowering (Complete: 9/9 kernels)
All kernels lowered to KTIR MLIR with CPU validation:
- Structural validation (MLIR parse)
- Numerical validation (KTIR CPU backend vs NumPy)

### Infrastructure
- Restructured project: `kernels/`, `tests/triton/`, `tests/ktir/`, `bench/`
- Benchmark runner: `bench/run_all.py`
- Auto-extraction script: `scripts/fetch_originals.py`

---

## What's Next

### Immediate (In Progress)
- [ ] Document benchmark results in status.md
- [ ] Add performance regression thresholds to CI

### Short-term
- [ ] Explore KTIR lowering pass implementation (Phase 3 → real compiler)
- [ ] Validate on Spyre hardware when backend available (Phase 4)

### Long-term (Out of Scope)
- [ ] Top-k/Top-p sampling kernel (iterative algorithm — not KTIR-expressible)
- [ ] Decode attention stage 1 (paged KV with indirect indexing)
- [ ] Unified attention (complex prefill+decode dispatch)

These require alternative approaches (CUDA fallback, redesign, or accept as unsupported)

---

## Milestones

| ID | Milestone | Status | Notes |
|----|-----------|--------|-------|
| M1 | Kernel inventory | Done | 13 identified, 9 selected |
| M2 | Block-pointer RMSNorm + SwiGLU | Done | + tests |
| M3 | Block-pointer attention kernels | Done | Prefill, decode merge, softmax+reduceV |
| M4 | All 9 kernels in block-pointer form | Done | 773 GPU tests pass |
| M5 | KTIR for simple kernels | Done | RMSNorm, SwiGLU, ranks, log-softmax |
| M6 | KTIR for attention kernels | Done | Prefill, decode merge, softmax+reduceV |
| M7 | All 9 kernels in KTIR | Done | 18 CPU tests pass |
| M8 | Spyre hardware execution | Pending | Requires Spyre backend |

---

## Success Criteria

**Phase 2 (Block Pointers):**
- [x] Block-pointer kernel matches raw-pointer numerically (GPU tests)
- [x] Block-pointer kernel within 2x performance (no major regression)
- [x] vLLM integration test passes (generation unchanged)

**Phase 3 (KTIR):**
- [x] KTIR MLIR parses without errors
- [x] KTIR CPU backend matches NumPy reference
- [ ] KTIR → Spyre execution (pending hardware)

**Phase 4 (Spyre Integration):**
- [ ] End-to-end vLLM generation on Spyre hardware
- [ ] Performance within acceptable bounds vs GPU

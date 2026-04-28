# Project Plan

## Completed

### Phase 1: Kernel Inventory
- Identified 13 core Triton kernels in vLLM's dense inference path
- Selected 9 kernels tractable for KTIR lowering
- Created `kernels.json` registry with vLLM source mapping

### Phase 2: Block-Pointer Conversion (9/13 kernels)
Converted all raw-pointer Triton kernels to block-pointer form:

| Kernel | GPU Tests | CPU Tests |
|--------|-----------|-----------|
| RMSNorm | 98 | 2 |
| SwiGLU | 98 | 2 |
| Ranks | 64 | 2 |
| Log-softmax | 190 | 2 |
| Decode softmax+reduceV | 85 | 2 |
| Merge attention states | 116 | 2 |
| MRoPE | 52 | 2 |
| Reshape/cache | 52 | 2 |
| Prefill attention | 18 | 2 |

**Total:** 773 GPU tests, 18 CPU tests — all passing

### Phase 3: KTIR Lowering (9/13 kernels)
All kernels lowered to KTIR MLIR with CPU validation:
- Structural validation (MLIR parse)
- Numerical validation (KTIR CPU interpreter vs NumPy reference)

---

## What's Next

1. **Improve benchmarks and tests** — align with real inference workloads, improve cross-run consistency
2. **Add missing kernels** — top-k/top-p sampling, decode attention stage 1, unified attention
3. **Full inference pass** — run end-to-end generation with converted kernels on the simulator
4. **Spyre hardware validation** *(future)* — test on real Spyre/AIU when backend is available
5. **Expand kernel coverage** *(future)* — port additional vLLM kernels as needed

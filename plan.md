# Project Plan

## Completed

### Phase 1: Kernel Inventory
- Identified 13 core Triton kernels in vLLM's dense inference path
- Selected 9 kernels tractable for KTIR lowering
- Created `kernels.json` registry with vLLM source mapping

### Phase 2: Block-Pointer Conversion (10/13 kernels)
Converted all raw-pointer Triton kernels to block-pointer form:

| Kernel | Source | GPU Tests | CPU Tests |
|--------|--------|-----------|-----------|
| RMSNorm | vLLM | 98 | 2 |
| SwiGLU | vLLM | 98 | 2 |
| Ranks | vLLM | 64 | 2 |
| Log-softmax | vLLM | 190 | 2 |
| Decode softmax+reduceV | vLLM | 85 | 2 |
| Merge attention states | vLLM | 116 | 2 |
| MRoPE | vLLM | 52 | 2 |
| Reshape/cache | vLLM | 52 | 2 |
| Prefill attention | vLLM | 18 | 2 |
| Matmul (GEMM) | Triton | 20 | 3 |

**Total:** 793 GPU tests, 21 CPU tests — all passing

### Phase 3: KTIR Lowering (10/13 kernels)
All kernels lowered to KTIR MLIR with CPU validation:
- Structural validation (MLIR parse)
- Numerical validation (KTIR CPU interpreter vs NumPy reference)

### Infrastructure: Multi-Source Fetch
- Generalized `kernels.json` to support multiple source repos (vLLM, Triton)
- Extended `fetch_originals.py` with per-kernel `source` field and `helpers` extraction
- Enables adding kernels from any GitHub-hosted Triton codebase

---

## What's Next

1. **Embedding lookup kernel** — table gather for token embeddings (simple but needed for end-to-end)
2. **Top-k/top-p sampling** — final token selection step; may need simplified variant for KTIR
3. **Full inference pass** — run end-to-end generation with converted kernels on the simulator
4. **Spyre hardware validation** *(future)* — test on real Spyre/AIU when backend is available
5. **Expand kernel coverage** *(future)* — decode attention stage 1, unified attention

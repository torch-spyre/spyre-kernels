# Kernel Conversion Status

## Pipeline Status

All 9 tractable kernels have completed the full conversion pipeline. 4 kernels remain unconverted.

| Kernel | Block-Ptr | KTIR | GPU Tests | CPU Tests |
|--------|-----------|------|-----------|-----------|
| `rms_norm` | Done | Done | 98 | 2 |
| `silu_and_mul` | Done | Done | 98 | 2 |
| `ranks` | Done | Done | 64 | 2 |
| `log_softmax` | Done | Done | 190 | 2 |
| `decode_softmax_reducev` | Done | Done | 85 | 2 |
| `merge_attn_states` | Done | Done | 116 | 2 |
| `mrope` | Done | Done | 52 | 2 |
| `reshape_and_cache` | Done | Done | 52 | 2 |
| `prefill_attention` | Done | Done | 18 | 2 |
| **Total** | **9/9** | **9/9** | **773** | **18** |

---

## Per-Kernel Notes

### RMSNorm
- 1D block pointers for input, weight, output — fully converted, no raw-pointer ops remain
- KTIR: added f32 accumulation (`arith.extf`/`arith.truncf`) for numerical stability; original was all f16

### SwiGLU
- 1D block pointers for gate, up, output — fully converted
- KTIR: f32 upcast for accumulation

### Ranks
- 1D block pointer for logits loop; scalar loads remain for `token_id` and `ref_logit` (data-dependent)
- KTIR: `%token_ids` (memref<32xi64>) passed directly; on-chip gather via `construct_indirect_access_tile` + `ind()`

### Log-Softmax
- 1D block pointers for vocab reduction loops; top-k gather remains raw (indirect)
- KTIR: f32 accumulation for both reduction passes; `%topk_ids` (memref<32x8xi64>) passed directly; on-chip gather via `construct_indirect_access_tile` + `ind()`

### Decode Softmax+ReduceV
- 1D block pointer per split for V data; scalar LSE and seq_len loads remain
- KTIR: two-pass algorithm (find max+sum, then weighted sum) to avoid multi-result `scf.for`

### Merge Attention States
- 1D block pointers for head-dim loads; scalar LSE loads remain
- KTIR: FA2 inf→-inf handling via `arith.cmpf oeq` on scalar LSE values

### MRoPE
- 2D block pointers `[n_heads, half_rd]` for q/k halves; cos/sin masked gathers remain (3 T/H/W sources)
- Uses `tl.advance` to reuse block pointers for second-half loads; `order=(0, 1)` for row-major layout
- Initial 1.13x regression on H100 from incorrect `order=(1, 0)` and redundant `make_block_ptr` — fixed
- KTIR: pre-merged cos/sin to avoid 3-way masked gather; added K processing loop (original only did Q)

### KV Cache Reshape
- 1D block pointers for source key/value loads; cache stores remain raw (scatter via `slot_mapping`)
- KTIR: f16 `slot_mapping` replaced with i64 memref; direct i64→index_cast eliminates f16 precision limit and removes dependency on unimplemented `arith.fptosi`; negative-slot guard omitted

### Prefill Attention
- 2D block pointers for Q `[M,D]`, K `[D,N]`, V `[N,D]`, O `[M,D]`; K/V advance by `BLOCK_N` per iteration
- KTIR: causal mask generated on-chip via `tensor.generate`

---

## Block-Pointer Takeaways

**What converts to block pointers:** contiguous and strided accesses (row loads, matrix tiles).

**What stays raw:** data-dependent accesses — scatter/gather, indirect indexing, scalar lookups.

| Lesson | Detail |
|--------|--------|
| `order` must match memory layout | `(0, 1)` for row-major with stride-1 columns; wrong order causes silent performance regression |
| Reuse with `tl.advance` | Avoids redundant `make_block_ptr` calls; measurable perf difference on small kernels |
| `boundary_check` always needed | Compiler doesn't optimize it away even for power-of-2 sizes |
| Sub-15us kernels are noisy | Use larger problem sizes for reliable block-ptr vs raw-ptr comparison |

**Numerical precision:**
- bf16/fp16: bitwise identical (`atol=0, rtol=0`)
- fp32: minor differences (≤1e-5) from zero-padding vs explicit masks

---

## KTIR Takeaways

**Simplifications required by KTIR's programming model:**

| Pattern | Kernels affected | Approach |
|---------|-----------------|----------|
| No masked gather | mrope | Pre-merge cos/sin tables |

**Interpreter workarounds:**

| Limitation | Workaround |
|-----------|------------|

---

## Kernels Not Yet Converted

| Kernel | Lines | Reason |
|--------|-------|--------|
| Top-k/Top-p sampling | 1057 | Iterative pivot selection, sorting — not expressible in KTIR |
| Decode attention stage 1 | 778 | Paged KV with indirect block table indexing |
| Unified attention | 1268 | Combined prefill+decode, complex control flow |
| Rotary embedding (non-multi) | — | Standard RoPE; lower priority since target models use MRoPE |

---

## Design Decisions

- **Block-pointer API:** Using `tl.make_block_ptr` (stable) over `tl.make_tensor_descriptor` (newer) for compatibility.

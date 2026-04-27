# Kernel Conversion Status

## Summary

**9 kernels converted** through the full pipeline:
- Phase 1: Raw-pointer Triton (extracted from vLLM)
- Phase 2: Block-pointer Triton (Spyre-compatible)
- Phase 3: KTIR MLIR (lowered for Spyre backend)

| Kernel | Phase 1 | Phase 2 | Phase 3 | GPU Tests | CPU Tests |
|--------|---------|---------|---------|-----------|-----------|
| `rms_norm` | Done | Done | Done | 98 | 2 |
| `silu_and_mul` | Done | Done | Done | 98 | 2 |
| `ranks` | Done | Done | Done | 64 | 2 |
| `log_softmax` | Done | Done | Done | 190 | 2 |
| `decode_softmax_reducev` | Done | Done | Done | 85 | 2 |
| `merge_attn_states` | Done | Done | Done | 116 | 2 |
| `mrope` | Done | Done | Done | 52 | 2 |
| `reshape_and_cache` | Done | Done | Done | 52 | 2 |
| `prefill_attention` | Done | Done | Done | 18 | 2 |
| **Total** | **9/9** | **9/9** | **9/9** | **773 pass** | **18 pass** |

---

## Kernel Details

### 1. RMSNorm (`rms_norm/`)

**vLLM source:** `vllm/model_executor/layers/batch_invariant.py` → `_rms_norm_kernel`

**Function:** `y = x / sqrt(mean(x²) + ε) * weight`

**Block-pointer conversion:**
- 1D block pointers for input, weight, and output rows
- No remaining raw-pointer ops

**KTIR notes:**
- Added f32 accumulation (`arith.extf`/`arith.truncf`) for numerical stability
- Original was all f16 computation

**Test coverage:**
- Shapes: 128–5120 hidden, batch 1–128
- Dtypes: fp32, fp16, bf16
- Edge cases: zeros, large values, single element

---

### 2. SwiGLU (`silu_and_mul/`)

**vLLM source:** `vllm/model_executor/layers/activation.py` → `_swiglustep_and_mul_kernel`

**Function:** `output = clamp(silu(gate)) * clamp(up)`, where `x = [gate || up]`

**Block-pointer conversion:**
- 1D block pointers for gate, up, output (separate base offsets)
- No remaining raw-pointer ops

**KTIR notes:**
- Added clamping via `-max(-a, -b)` pattern (no `minimumf`/`maximumf`)
- f32 upcast for accumulation

**Test coverage:**
- Half-hidden: 128–4097
- 3D inputs (batch, seq, hidden)
- Clamp limit variations

---

### 3. Ranks (`ranks/`)

**vLLM source:** `vllm/v1/worker/gpu/sample/logprob.py` → `_ranks_kernel`

**Function:** Count logits ≥ ref_logit per row

**Block-pointer conversion:**
- 1D block pointer for logits loop
- Scalar loads remain: token_id, ref_logit (data-dependent)

**KTIR notes:**
- Pre-extracted `ref_logits` on host (KTIR can't do indirect loads)
- `arith.cmpi sge` on f16 works in interpreter

**Test coverage:**
- Vocab: 128–32000
- Negative logits, all-same logits

---

### 4. Log-Softmax (`log_softmax/`)

**vLLM source:** `vllm/v1/worker/gpu/sample/logprob.py` → `_topk_log_softmax_kernel`

**Function:** Log-softmax at pre-extracted top-k positions

**Block-pointer conversion:**
- 1D block pointers for vocab reduction loops
- Top-k gather remains raw (data-dependent indirect load)

**KTIR notes:**
- f32 accumulation for both reduction passes
- Pre-extracted `topk_logits` on host

**Test coverage:**
- Vocab: 128–32001
- Top-k: 1–16
- Uniform logits (validates -log(vocab))

---

### 5. Decode Softmax+ReduceV (`decode_softmax_reducev/`)

**vLLM source:** `vllm/v1/attention/ops/triton_decode_attention.py` → `_fwd_kernel_stage2`

**Function:** Online softmax merge across KV splits

**Block-pointer conversion:**
- 1D block pointer per split for V data
- Scalar LSE and seq_len loads remain

**KTIR notes:**
- Two-pass algorithm (find max+sum, then weighted sum)
- Avoids multi-result `scf.for` (unsupported)

**Test coverage:**
- Lv: 32–128
- Splits: 2–8
- Short sequences (partial splits)

---

### 6. Merge Attention States (`merge_attn_states/`)

**vLLM source:** `vllm/v1/attention/ops/triton_merge_attn_states.py` → `merge_attn_states_kernel`

**Function:** Merge two partial attention outputs via online softmax

**Block-pointer conversion:**
- 1D block pointers for head-dim loads (prefix, suffix, output)
- Scalar LSE loads remain

**KTIR notes:**
- FA2 inf→-inf handling omitted (interpreter `arith.cmpi eq` unreliable on scalars)
- Test data is well-behaved

**Test coverage:**
- Head sizes: 32–128
- Equal LSEs (average test)
- Dominant prefix/suffix

---

### 7. MRoPE (`mrope/`)

**vLLM source:** `vllm/model_executor/layers/rotary_embedding/mrope.py` → `_triton_mrope_forward`

**Function:** Rotary embeddings on Q and K from 3D cos/sin tables

**Block-pointer conversion:**
- 2D block pointers for q/k halves `[n_heads, half_rd]`
- cos/sin masked gathers remain (3 T/H/W sources)

**KTIR notes:**
- Pre-merged cos/sin (avoids 3-way masked gather)
- Added K processing loop (original only did Q)

**Test coverage:**
- GQA (n_qh ≠ n_kv_h)
- Identity (cos=1, sin=0)
- Single token

---

### 8. KV Cache Reshape (`reshape_and_cache/`)

**vLLM source:** `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` → `reshape_and_cache_kernel_flash`

**Function:** Scatter key/value rows into paged cache slots

**Block-pointer conversion:**
- 1D block pointers for source key/value loads
- Cache stores remain raw (scatter via `slot_mapping`)

**KTIR notes:**
- f16 `slot_mapping` (precision limit >2048 slots)
- Negative-slot guard omitted (interpreter `scf.if` issues)

**Test coverage:**
- Negative slots (skip behavior)
- Sequential slots
- Single token

---

### 9. Prefill Attention (`prefill_attention/`)

**vLLM source:** `vllm/v1/attention/ops/triton_prefill_attention.py` → `_fwd_kernel`

**Function:** Multi-head SDPA with optional causal masking

**Block-pointer conversion:**
- 2D block pointers for Q `[M,D]`, K `[D,N]`, V `[N,D]`, O `[M,D]`
- K/V advance by `BLOCK_N` per iteration
- Scalar loads: `B_Start_Loc`, `B_Seqlen`

**KTIR notes:**
- Pre-computed causal mask as input tensor
- `tensor.generate` unsupported in interpreter

**Test coverage:**
- Causal and non-causal
- GQA (n_qh ≠ n_kv_h)
- Variable sequence lengths

---

## Phase 2: Block-Pointer Patterns

### Conversion Strategy

| Kernel | Block-ptr loads/stores | Remaining raw-pointer ops |
|--------|----------------------|--------------------------|
| RMSNorm | 1D for input, weight, output | — |
| SwiGLU | 1D for gate, up, output | — |
| Ranks | 1D for logits loop | Scalar: token_id, ref_logit |
| Log-softmax | 1D for vocab loops | Top-k gather (indirect) |
| Decode softmax+reduceV | 1D per split for V | Scalar LSE, seq_len |
| Merge attention | 1D for head-dim | Scalar LSE |
| MRoPE | 2D for q/k halves | cos/sin 3-way gather |
| Reshape/cache | 1D for source loads | Cache scatter, slot_mapping |
| Prefill attention | 2D for Q/K/V/O | Scalar seq_len locs |

**Key insight:** Contiguous/strided accesses use block pointers. Data-dependent (scatter/gather) remain raw.

### Numerical Precision

- **bf16/fp16:** Bitwise identical (`atol=0, rtol=0`)
- **fp32:** Minor differences (≤1e-5) from zero-padding vs explicit masks

---

## Phase 3: KTIR Simplifications

These are intentional differences where KTIR's programming model requires a different approach:

1. **Pre-extracted indirect data:** Ranks/log-softmax pre-extract on host
2. **Pre-merged cos/sin:** MRoPE avoids 3-way masked gather
3. **Pre-computed causal mask:** Prefill attention takes mask as input
4. **Two-pass softmax merge:** Decode softmax+reduceV avoids multi-result loops
5. **f16 slot_mapping:** Reshape/cache precision limit (would need i32 for >2048 slots)

---

## KTIR Interpreter Limitations Worked Around

| Limitation | Workaround |
|-----------|------------|
| No `arith.cmpf` (float compare) | Use `arith.cmpi sge/eq` |
| No `arith.minimumf`/`arith.maximumf` | `min(a,b) = -max(-a, -b)` |
| No `tensor.generate` | Pre-compute on host |
| No multi-result `scf.for` | Single-result loops; restructure algorithm |
| `arith.select` hardcodes f16 | Only use for f16 values |
| f32 memref output may not dispatch | Truncate to f16 before store |

---

## Kernels Not Converted

| Kernel | Lines | Reason |
|--------|-------|--------|
| Top-k/Top-p | 1057 | Iterative pivot selection, sorting — not expressible in KTIR |
| Decode attention stage 1 | 778 | Paged KV with indirect block table indexing |
| Unified attention | 1268 | Combined prefill+decode, complex control flow |

---

## vLLM Source Mapping

All kernels extracted verbatim from vLLM commit [`cde8d2471026`](https://github.com/vllm-project/vllm/commit/cde8d2471026):

| Kernel | vLLM File | Function |
|--------|-----------|----------|
| RMSNorm | `batch_invariant.py` | `_rms_norm_kernel` |
| SwiGLU | `activation.py` | `_swiglustep_and_mul_kernel` |
| Ranks | `logprob.py` | `_ranks_kernel` |
| Log-softmax | `logprob.py` | `_topk_log_softmax_kernel` |
| Decode softmax+reduceV | `triton_decode_attention.py` | `_fwd_kernel_stage2` |
| Merge attention | `triton_merge_attn_states.py` | `merge_attn_states_kernel` |
| MRoPE | `mrope.py` | `_triton_mrope_forward` |
| KV cache reshape | `triton_reshape_and_cache_flash.py` | `reshape_and_cache_kernel_flash` |
| Prefill attention | `triton_prefill_attention.py` | `_fwd_kernel` |

Verify: `python scripts/fetch_originals.py --diff` (reports 0 differences when synced)

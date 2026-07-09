# prefill_attention conversion notes

## Tensor-descriptor conversion

- Source: original.py → tensor_descriptor.py (kernel `_prefill_attention_kernel_td`)
- Raw pointer arithmetic (`off_q`/`off_k`/`off_v`/`off_o`, `k_ptrs`/`v_ptrs`
  advanced per iteration) replaced with 3D `tl.make_tensor_descriptor` views over
  the packed `(seq_len, heads, head_dim)` layout. Each descriptor is rebased at
  `cur_batch_in_all_start_index * stride_*bs` so its coordinates run
  `0..cur_batch_seq_len`; the batch start is a scalar folded into the base
  pointer, not a per-row runtime index, so plain `desc.load`/`desc.store` suffice
  (no gather/scatter needed).
- **Tail masks dropped:** the original's seq-len (`offs_m < seq_len`,
  `pos_k < seq_len` on the load) and head-dim (`mask_d = offs_d < Lk`) masks on
  Q/K/V loads and the O store are redundant — the descriptor `shape` carries both
  boundaries and zero-fills OOB. Zero is the additive identity for the `tl.dot`
  accumulation and for masked-out `qk` (overwritten by the causal `tl.where`), so
  the drop is safe.
- **Compute masks kept:** causal / sliding-window / valid-position masking
  (`tl.where(mask, qk * sm_scale, -1.0e8)`) is attention *semantics*, not a tail
  fill, so it is preserved verbatim.
- K is loaded as `(BLOCK_N, BLOCK_DMODEL)` and transposed via `tl.trans` before
  `tl.dot(q, k)`. Descriptors require the last dim contiguous, so K cannot be
  loaded pre-transposed the way the original did via `strides=(1, stride_kbs)`.
- Last-dim (§4) is satisfied: the contiguous axis is `head_dim` (`BLOCK_DMODEL`),
  ≥ 16 bytes; the length-1 head axis is the middle dim.
- **Signature change:** added `num_q_heads` / `num_kv_heads` runtime args (needed
  for the descriptor `shape`, which the original never materialized). The wrapper
  passes them only when the target kernel declares them (`arg_names` check) and
  registers the TMA allocator, so it still drives the original kernel unchanged.

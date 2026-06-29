# matmul conversion notes

## Tensor-descriptor conversion

- Source: original.py → tensor_descriptor.py (kernel `_matmul_kernel_td`)
- Pointer arithmetic (`a_ptrs`/`b_ptrs`/`c_ptrs` + `tl.arange` offsets, in-loop
  `+= BLOCK_SIZE_K * stride`) replaced with one `tl.make_tensor_descriptor` per
  tensor; the K-loop advances by passing `off_k` to each `.load()`.
- Tail masks dropped (descriptor `shape` carries the boundary):
  - K-dimension load mask (`offs_k < K - k*BLOCK_SIZE_K`, `other=0.0`) — the
    descriptor zero-fills the partial K tail, which is the additive identity for
    the `tl.dot` accumulation, so it is correct to drop.
  - Output store mask (`(offs_cm < M) & (offs_cn < N)`) — `c_desc.store` clamps
    at `shape=[M, N]`.
- The original's `offs_am % M` / `offs_bn % N` wraparound is also dropped: the
  M/N boundaries now live in the A/B descriptor shapes, which zero-fill OOB rows
  and columns. Those zero lanes only feed `tl.dot` over K and the C store clamps
  to `[M, N]`, so the valid output region is unaffected.
- Signature, autotune config, grid, and grouped-pid scheduling preserved
  unchanged; the wrapper can call this form via its `kernel_fn` parameter.

## Helion round-trip

- `original.py` → `helion_kernel.py` (`matmul_helion`) → autotune-emit
  (`indexing=tensor_descriptor` pinned) → arg-ify → `triton_helion_roundtrip.py`.
- Stage 1 mapping: `hl.tile([m, n])` for the parallel M/N output axes, inner
  `hl.tile(k)` reduction, `torch.addmm` accumulating in f32, cast to f16 on store.
- Stage 2 ran on an H100. Winning config:
  `block_sizes=[64, 128, 64]`, `indexing='tensor_descriptor'`,
  `l2_groupings=[8]`, `num_stages=6`, `num_warps=4`, `pid_type='flat'`.
  `make_tensor_descriptor occurrences: 2` — descriptors fired (one per input;
  output uses a masked `tl.store`). ~187/921 autotune configs were skipped on
  ptxas register-pressure failures (`maxnreg=64` too low) — autotuner discards
  them, winner is unaffected.
- Stage 4 constexpr naming: `_BLOCK_SIZE_0`→`BLOCK_M` (cdiv(m,·), a_desc rows),
  `_BLOCK_SIZE_1`→`BLOCK_N` (cdiv(n,·), b_desc cols), `_BLOCK_SIZE_2`→`BLOCK_K`
  (`tl.range(0, k, ·)` reduction step) — matching the original's tile names.

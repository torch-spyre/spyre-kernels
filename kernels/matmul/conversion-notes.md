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

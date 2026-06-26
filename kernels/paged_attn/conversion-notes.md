# paged_attn conversion notes

Paged attention over a KV cache: queries `Q (B, Lq, H, D)` attend to the cache
rows `K/V (CACHE, H, D)` named, per request, by `SLOTS (B, Lk)`. Output is
`(B, H, Lq, D)`. Scores use the standard `1/sqrt(D)` softmax scale, split as
`scale = 1/sqrt(sqrt(D))` applied to both q and k (`scale*scale = 1/sqrt(D)`).

Two descriptor variants are generated, with identical semantics and signature —
they differ **only** in how the data-dependent K/V gather is expressed. Both are
**descriptors only**: there is no raw pointer arithmetic anywhere, including the
gather. The gather is the data-dependent step (the cache row is chosen at
runtime by `SLOTS`), and it is expressed with `descriptor_gather`, which lowers
to the Spyre indirect-access tile (`ktdp.construct_indirect_access_tile`).

## tensor_descriptor.py — base `descriptor_gather`

- Source: `paged_attn.py` (the base draft).
- K/V are viewed **2-D** as `(CACHE, H*D)`; the descriptor `block_shape` is
  `[BLK_B*KV_BLOCK, BLK_H*D]`.
- The `(BLK_B, KV_BLOCK)` slot tile is flattened to a **1-D** batch-major row
  vector, and `descriptor_gather(k_desc, rows, h_start*D)` pulls those rows and
  the `BLK_H*D` columns at head offset `h_start*D`. The `(rows, columns)` result
  is `.reshape`d to `(BLK_B, KV_BLOCK, BLK_H, D)`.
- This uses the base `descriptor_gather` (1-D index, 2-D src).

## spyre_aware.py — extended any-rank `descriptor_gather`

- Source: `paged_attn_ext.py`.
- Same algorithm, but `descriptor_gather` is **extended to accept an index and a
  src descriptor of any rank**. So K/V stay **3-D** `(CACHE, H, D)` with
  `block_shape=[1, BLK_H, D]`, the 2-D slot tile `(BLK_B, KV_BLOCK)` is passed
  straight in as the index, and `h_start` is the second index — yielding
  `(BLK_B, KV_BLOCK, BLK_H, D)` directly, with **no reshape** of the index or of
  the gathered result. This maps the gather onto the Spyre indirect-access tile
  more directly than the flatten/reshape of the base form.

## Shared structure (both variants)

- 4-D batched online softmax: the B loop steps by `BLK_B` and the H loop by
  `BLK_H`; the `BLK_B` batches and `BLK_H` heads in each block are vectorized so
  a single gather serves them all and `tl.dot` batches over `(BLK_B, BLK_H)`.
- Q, SLOTS, Out use plain `make_tensor_descriptor` loads/stores (static offsets
  from the loop counters), so no tail masks are needed there.
- `grid=(1,)`: one program walks all `(B, Lq, H)` work via the explicit loops,
  so the output is partition-independent and the grid fits 32 cores.
- KV-page loop bound is `tl.cdiv(Lk, KV_BLOCK)` (runtime-arg agnostic; the base
  draft's bare `Tk = Lk // kv_block_size` constexpr is replaced).
- 16-byte last-dim rule: every descriptor's last dim is `BLOCK_D` (`D*2` bytes),
  `BLK_H*D` (`*2`), or `KV_BLOCK` (`*4`) — all >= 16 bytes for the chosen sizes.

## original.py — reference

A simpler hand-written reference (one program per `(request, head, query
block)`, single-head gather per KV page) that the two batched variants are
checked against. Also descriptors only; uses the base `descriptor_gather`.

# Invariant — Tiles fit scratchpad

Each Spyre core has **2 MB of scratchpad**. The sum of concurrently-live tile
bytes — all loaded tiles + accumulators + partial outputs alive at the same
program point — must be ≤ 2 MB. Tile sizes are passed as `tl.constexpr`; the
kernel assumes they are valid.

```
tile_bytes = product(block_shape) * dtype_bytes
peak = sum(tile_bytes for every tile alive simultaneously)
```

Common dtype sizes: f16 = 2, bf16 = 2, f32 = 4, i32 = 4 bytes.

**Critical: tile constexprs must be independent of problem dimensions.** If a
tile size is derived from a problem size (e.g. `BLOCK = next_power_of_2(N)` to
cover a whole dimension in one load), scratchpad usage grows with the problem
and the invariant does not hold — even when the byte count is small for typical
values. The fix is a fixed tile size plus a loop over the dimension in chunks.

**Red flags**
- A `block_shape` entry computed from `M`/`N`/`K`/`seq_len` rather than a
  constexpr tile constant.
- Peak live bytes that exceed 2 MB for plausible constexprs (e.g. matmul with
  `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64`).

See the other invariants in this directory, and
`../spyre/scratchpad-batching.md` (using the *rest* of the 2 MB once the hard
constraint holds).

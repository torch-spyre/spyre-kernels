# Scratchpad batching — use the rest of the 2 MB

Once the hard constraint holds (tiles fit — see
`invariants/tile-fits-scratchpad.md`), check whether the kernel *under-uses*
the scratchpad by processing less data per iteration than it could. Low
utilization leaves performance on the table: the scratchpad exists to hold
working data, and using 1 KB of 2 MB underexploits the hardware.

The lever is not just *filling* the scratchpad but *reusing* it — keeping
working data resident and reusing the same region as much as possible so the
kernel re-fetches from HBM as little as possible. Batching (below) is the common
way to get that reuse: process more independent work per iteration so the data
already loaded does more work before it is evicted.

This is a **performance** concern (WARN, not a correctness FAIL).

## Detecting under-use

1. Compute peak live bytes. If utilization is below ~10% (< ~200 KB), flag it.
2. Look for a loop axis that processes **one item at a time** when it could
   batch:
   ```python
   for work_idx in range(start, end):
       acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)   # one item per iteration
   ```
   The tell-tale shape is a distribution loop whose body holds a single
   per-item accumulator (e.g. `[BLOCK_SIZE]`) — one independent work item per
   iteration, with most of the scratchpad left idle.

## When batching is possible

Batch `BLOCK_ITEMS` work items per iteration when:
- the items are **independent** — each has its own accumulator/reduction state,
  no cross-item dependency;
- the scratchpad fits `BLOCK_ITEMS × per-item` live bytes within 2 MB.

How:
1. Introduce a `BLOCK_ITEMS: tl.constexpr` for the work axis.
2. Distribute over `cdiv(total_work, BLOCK_ITEMS)` tiles, not individual items.
3. Widen the accumulator: `[BLOCK_SIZE]` → `[BLOCK_ITEMS, BLOCK_SIZE]`.
4. Load `BLOCK_ITEMS` elements per descriptor access on the work axis.

Choosing `BLOCK_ITEMS`:
```
per_item_bytes = (acc_tile + loaded_tiles) * dtype_bytes
BLOCK_ITEMS    = floor((1-2 MB budget) / per_item_bytes)   # clamp to pow2, ~16-64 max
```
Keep `cdiv(total_work, BLOCK_ITEMS) ≥ 32` for typical sizes, or some of the 32
cores sit idle (the distribution-granularity trade-off). Power-of-2 preferred.

## Side benefit — escapes the 16-byte minimum

When a scalar load becomes a `[BLOCK_ITEMS]` vector load,
`BLOCK_ITEMS × dtype_bytes ≥ 16` satisfies the descriptor's last-dim minimum
(see `../descriptor-rules.md` §4) **without** an over-fetch workaround. So if a
kernel was flagged for a scalar descriptor, batching the work axis is not just a
perf win — it is a **gap-resolution path** that removes the scalar descriptor.

## Trade-off — divergent control flow vs uniform data load

Batching across an axis whose items have **divergent control flow** (e.g. a
per-item length or count that decides which iterations are active) trades
branching for over-fetch: instead of branching per item, load uniformly across
all `BLOCK_ITEMS` lanes and **mask the math**.

The general move: compute a per-lane `active` mask, load every lane
unconditionally, then neutralize inactive lanes with the **identity for the
reduction** so they contribute nothing — `where(active, x, 0.0)` for a sum,
`where(active, x, -inf)` before a max/softmax, `where(active, x, 1.0)` for a
product. The batched op then runs as a single vector reduction over all lanes.

- **Cost:** over-fetched memory bandwidth (inactive lanes are loaded but
  contribute nothing).
- **Benefit:** uniform control flow + a vector reduction on the inner axis.

Worth calling out so a reviewer does not flag per-item divergence as a blocker —
it is feasible-with-masking, not infeasible.

## When batching is NOT possible

- Cross-item dependency on the axis (a reduction that accumulates across the
  batched dimension).
- Multi-item loads that are non-contiguous in memory (verify the stride between
  consecutive items is uniform; if items cross a higher-level boundary, restrict
  batching within one such dimension or provide a uniformly-strided view from
  the wrapper).

In those cases note the under-use as a known performance gap and move on.

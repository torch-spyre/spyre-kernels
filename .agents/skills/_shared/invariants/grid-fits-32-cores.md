# Invariant — Grid fits 32 cores

Total cooperating program count must be **≤ 32**. If there is more work than 32
programs, express it as an **explicit outer loop inside the kernel** (a
distribution loop), not by launching more programs. Query the grid with
`tl.num_programs(axis)` and compute per-core bounds:

```python
pid = tl.program_id(0)
num_cores = tl.num_programs(0)
blocks_per_core = tl.cdiv(total_blocks, num_cores)
start = pid * blocks_per_core
end = tl.minimum(start + blocks_per_core, total_blocks)
for i in range(start, end):
    ...
```

The result must be **independent of how work is partitioned** — running with 1,
4, 16, or 32 cores produces the same output (see `../../spyre-test/SKILL.md`
distribution-invariance tests).

**Red flags**
- `pid` used directly as a tile index without a distribution loop.
- Grid size computed from problem dimensions (`grid = cdiv(M, BLOCK_M)`).
- Missing `tl.num_programs()`.
- A distribution loop missing `tl.minimum` on the end bound (tail cores
  overrun).

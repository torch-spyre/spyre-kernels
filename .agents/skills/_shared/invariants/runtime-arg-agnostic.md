# Invariant — Runtime-arg agnostic

The kernel must produce correct output for **any valid combination** of problem
sizes (`M`, `N`, `n_elements`, sequence length, batch) given appropriately-sized
tile constexprs — **including non-divisible shapes**. Don't write code that only
works when the tile divides the problem size evenly.

- Use `tl.cdiv` for block counts and `tl.minimum` for loop bounds.
- Never assume divisibility; never use `//` or `%` on a problem size without a
  guard.

**Correct**
```python
for i in range(tl.cdiv(M, BLOCK_M)):          # covers the ragged tail
    tile = in_desc.load([i * BLOCK_M, 0])     # descriptor zero-fills out-of-range lanes
    ...
```

**Red flags**
- `range(0, M // BLOCK_M)` instead of `range(tl.cdiv(M, BLOCK_M))` — drops the
  ragged tail when `BLOCK_M` does not divide `M`.
- Missing `tl.minimum` on a distribution-loop end bound.
- `//` or `%` on a problem size with no accompanying guard.

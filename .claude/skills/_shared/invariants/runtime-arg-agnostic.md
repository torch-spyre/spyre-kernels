# Invariant — Runtime-arg agnostic

Problem-size args (`M`, `N`, `n_elements`, sequence length, batch) are runtime
`i32`. Tile-size args (`BLOCK_M`, `BLOCK_SIZE`, …) are `tl.constexpr`. The
kernel must produce correct output for **any valid combination** of runtime
args given appropriately-sized constexprs — **including non-divisible shapes**.

- Use `tl.cdiv` for block counts and `tl.minimum` for loop bounds.
- Never assume divisibility; never use `//` or `%` on a problem size without a
  guard.

**Red flags**
- `M: tl.constexpr` — a problem size annotated as constexpr.
- `range(0, M // BLOCK_M)` instead of `range(tl.cdiv(M, BLOCK_M))`.
- Missing `tl.minimum` on a distribution-loop end bound.

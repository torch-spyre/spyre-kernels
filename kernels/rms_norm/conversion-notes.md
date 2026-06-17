# rms_norm conversion notes

## Tensor-descriptor conversion

- Source: original.py → tensor_descriptor.py
- Pointer arithmetic / masked loads/stores replaced with
  `tl.make_tensor_descriptor` (one descriptor each for input, weight, output).
- Tail masks dropped — the descriptor `shape` carries the column boundary, so
  the original's `mask` / `tl.where` / `other=` machinery is redundant. The
  sum-of-squares reduction is additive, so descriptor zero-fill is the correct
  identity. The weight's `other=1.0` is not needed: out-of-range weight lanes
  zero-fill, but the matching input lanes are also zero and the output store
  clamps at `n_cols`, so those lanes are never written.
- Signature change: `input_row_stride` / `output_row_stride` are kept as runtime
  args and passed straight into the descriptor strides (`strides=[stride, 1]`),
  preserving the original's support for strided (non-contiguous) rows. Column
  stride is 1, matching the original's contiguous `row_start_ptr + col_idx`
  indexing. The wrapper passes `input_2d.stride(0)` / `output.stride(0)`.
- Row batching: each program processes `ROWS_PER_PROGRAM` rows, loaded together
  as a `[ROWS_PER_PROGRAM, BLOCK_SIZE]` tile and reduced over the column axis
  with `tl.sum(..., axis=1)`. `ROWS_PER_PROGRAM=1` recovers one program per row.
  It is the descriptor `block_shape` row dimension, so it must be a power of 2.
- No scalar descriptors: every descriptor's last block dim is `BLOCK_SIZE`
  (≥ 16 bytes), so the 16-byte last-dim minimum holds without a workaround.

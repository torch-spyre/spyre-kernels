# silu_and_mul conversion notes

## Tensor-descriptor conversion

- Source: original.py → tensor_descriptor.py
- Pointer arithmetic (`o_row_ptr`/`x_row_ptr` + offsets) replaced with
  `tl.make_tensor_descriptor`.
- A single input descriptor spans both halves `[n_rows, 2*d]`; gate and up are
  read from it at column offsets `col` and `col + d`, mirroring the original's
  `x_row_ptr + offsets` and `x_row_ptr + offsets + d`. Column stride is 1; row
  strides come straight from the `x_stride` / `o_stride` runtime args.
- Tail mask dropped: the descriptor `shape` carries the column boundary (`2*d`
  for input, `d` for output), so the partial column tile zero-fills on load and
  clamps on store. Zero-fill is safe here — the result is purely multiplicative
  elementwise with no reduction, and clamped stores never write past `d`.
- Signature change: added an `n_rows` parameter. The original inferred the row
  count implicitly from the grid; the descriptor `shape` needs it explicitly.
  The wrapper already branches on `"n_rows" in kernel_fn.arg_names` and passes
  it, so no wrapper edit was required.
- Kept the 2D grid `(n_rows, cdiv(d, BLOCK_SIZE))` — one program per
  (row, column-tile). 16-byte rule is satisfied: the last-dim block is
  `BLOCK_SIZE` (1024), well over 16 bytes.

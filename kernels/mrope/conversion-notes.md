# mrope conversion notes

## Tensor-descriptor conversion

- Source: original.py → tensor_descriptor.py (kernel `_mrope_kernel_td`)
- Pointer arithmetic replaced with `tl.make_tensor_descriptor`:
  - cos/sin viewed as 3D `[3, num_tokens, half_rd]` (the lead axis is the
    t/h/w section); each section row loaded with `desc.load([sec, pid, 0])`.
  - q/k viewed as 4D `[num_tokens, heads, 2, half_rd]`; the third axis selects
    left/right rotary halves, so descriptor offsets stay block-aligned even
    when `rotary_dim < head_size`.
- **Section masks kept (not tail masks).** The t/h/w masks select *which*
  section contributes to each lane — value selection, not a boundary. Kept as
  `tl.where(mask, row, 0.0)` over the three section rows; sections are disjoint
  so the sum reproduces the original merge. Descriptor zero-fill does NOT
  replace these.
- Tail masks on q/k dropped: descriptor `shape`'s last dim is `half_rd`, so the
  padded tile tail is zero-filled on load and clamped on store. Non-rotary
  lanes beyond `rotary_dim` are outside the descriptor shape and remain
  untouched, matching the original `arange < rd//2` load/store masks.
- Signature preserved; wrapper dispatches via existing `kernel_fn=` arg.
- 16-byte last dim: descriptor block width is `max(pad_hd // 2, 8)`, so the
  last dimension is at least 8 elements (16 bytes for fp16/bf16; larger for
  fp32). Descriptor shape may be smaller; out-of-range lanes zero-fill/clamp.
- Scope: supports full and partial rotary (`rotary_dim <= head_size`) for the
  same non-interleaved/interleaved section semantics as the original.

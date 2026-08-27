# Spyre compiler descriptor gaps — mark as a gap, don't bake a workaround

These are **Spyre-compiler / KTIR-lowering** limitations (the
`LowerDescriptorMemory` pass and the Spyre backend). They are **spyre-family
only** — the TD family targets the Triton descriptor API generically and does
not hit them. (The one descriptor gap that *is* backend-independent — the
16-byte last-dim minimum — lives in `../descriptor-rules.md`.)

## The contract

The skill's job is two things, **not** a hierarchy of fallbacks:

1. **Write the kernel in the cleanest descriptor-first form.**
2. **If that form is unsupported, mark the site with a `# [gap]` annotation**
   naming the limitation — do **not** bake a raw-pointer workaround or an
   invented layout into the kernel.

The kernel stays in its clean, descriptor-first form; the annotation identifies
the unsupported operation. Do not replace the clean form with a workaround.

## Where the gaps are tracked

This file defines the annotation policy rather than duplicating the supported
surface. Consult these authoritative sources:

- **KB** — query `spyre-kb` for the current state of descriptor lowering
  (`search("descriptor lowering gap")`; the `triton` repo sync-status page
  tracks `LowerDescriptorMemory` work with dated commits).
- **Spyre docs** — https://github.com/torch-spyre/triton/tree/main/third_party/spyre/docs
  (memory patterns under `docs/patterns/`).
- **Issues** — `torch-spyre/triton` (e.g. the rank-reduced `descriptor_load`
  gap is tracked there).

When you hit a gap, name it in the `# [gap]` annotation and link the tracking
issue if you know it; do not encode the workaround into the kernel.

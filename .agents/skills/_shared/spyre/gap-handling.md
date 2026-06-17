# Spyre compiler descriptor gaps — mark as a gap, don't bake a workaround

These are **Spyre-compiler / KTIR-lowering** limitations (the
`LowerDescriptorMemory` pass and the Spyre backend). They are **spyre-family
only** — the TD family targets the Triton descriptor API generically and does
not hit them. (The one descriptor gap that *is* backend-independent — the
16-byte last-dim minimum — lives in `../descriptor-rules.md`.)

## The contract

The skill's job is two things, **not** a hierarchy of fallbacks:

1. **Write the kernel in the cleanest descriptor-first form**, even if it does
   not compile yet. That form is the target for when the gap is closed.
2. **If the form hits a gap, mark the site with a `# [gap]` annotation**
   naming the limitation — do **not** bake a raw-pointer workaround or an
   invented layout into the kernel.

The kernel stays in its clean, descriptor-first form; the annotation records
where it does not yet lower so whoever closes the gap knows what to target. Do
not replace the clean form with a workaround to make it compile today.

## Where the gaps are tracked

The specific gaps move quickly — the lowering passes are under active refactor,
and what fails to lower today may lower next week. **This file does not restate
the current gap list**; that would drift out of date. Consult the live sources
instead:

- **KB** — query `spyre-kb` for the current state of descriptor lowering
  (`search("descriptor lowering gap")`; the `triton` repo sync-status page
  tracks `LowerDescriptorMemory` work with dated commits).
- **Spyre docs** — https://github.com/torch-spyre/triton/tree/main/third_party/spyre/docs
  (memory patterns under `docs/patterns/`).
- **Issues** — `torch-spyre/triton` (e.g. the rank-reduced `descriptor_load`
  gap is tracked there).

When you hit a gap, name it in the `# [gap]` annotation and link the tracking
issue if you know it; do not encode the workaround into the kernel.

---
name: helion-convert
description: "Round-trip a Triton kernel through Helion: hand-port original.py to a Helion kernel, autotune-compile it back to Triton with the tensor_descriptor indexing lever pinned, then rewrite the emitted Triton so baked tl.constexpr block sizes become function arguments. Emit-only — no GPU validation. Use when asked to convert a kernel to Helion, round-trip Triton through Helion, autotune-emit a tensor_descriptor Triton kernel, or arg-ify the block-size constexprs of a Helion-emitted kernel."
---

# Helion Conversion Skill

Round-trip a kernel through Helion. Four stages: stages 1 and 4 need model
judgment; stages 2 and 3 are scripts.

1. **Triton → Helion** (model) — port `original.py` to `helion_kernel.py`.
2. **Helion → Triton** (script, TMA-capable GPU) — autotune-emit Triton with
   `indexing=tensor_descriptor` pinned.
3. **arg-ify constexpr** (script, local) — rewrite the emitted Triton so the
   baked `tl.constexpr` values become function arguments.
4. **name the arg-ified constexprs** (model) — rename the lifted `tl.constexpr`
   params from Helion's positional names to meaningful ones from the kernel.

This is **emit-only**: it does not run the kernel or check numerics.

## Trigger

Use when the user asks to convert a kernel to Helion, round-trip Triton through
Helion, autotune-emit a tensor_descriptor Triton kernel, or arg-ify the
block-size constexprs of a Helion-emitted kernel.

## Pre-flight

Run the KB consult in [`../_shared/preflight.md`](../_shared/preflight.md)
(search "helion", "tensor descriptor").

## Inputs

- **kernel_name**: directory under `kernels/` (e.g. `matmul`, `rms_norm`)
- Source: `kernels/<name>/original.py`
- Stage 1 out: `kernels/<name>/helion_kernel.py` (intermediate)
- Stage 2 out: `kernels/<name>/triton_emitted.py` (intermediate)
- Stage 3 out: `kernels/<name>/triton_helion_roundtrip.py` (arg-ified)
- Stage 4 out: `kernels/<name>/triton_helion_roundtrip.py` (**final deliverable**
  — same file, block-size params renamed in place; stages 1–2 outputs are
  deleted in cleanup)

## Procedure

### Stage 1 — Triton → Helion (MODEL judgment)

Port `original.py` to `kernels/<name>/helion_kernel.py`, forward pass only.
Pattern:
- `for tile_m, tile_n in hl.tile([m, n])` for the **parallel** output axes.
- inner `for tile_k in hl.tile(k)` for a **reduction** axis.
- tensor-level PyTorch in the body: `torch.addmm` (matmul accumulate),
  `torch.mean` / `torch.rsqrt` (norms), etc.
- accumulate in **float32**, cast back to input dtype on the output store.

Keep the decorator minimal (`@helion.kernel` or `@helion.kernel(static_shapes=…)`)
— the autotune knobs are applied by the stage-2 script, not baked here.

Also define `example_args(dev) -> tuple` in the same file, returning
representative, descriptor-eligible args for the kernel (reasonable shapes — they
drive autotuning, not correctness). Stage 2 calls it.

### Stage 2 — Helion → Triton (SCRIPT, needs a TMA-capable GPU)

`tensor_descriptor` lowering requires TMA hardware (NVIDIA Hopper / H100), so
this stage runs on a GPU host, not a laptop. Run:

```bash
HELION_SKIP_CACHE=1 \
HELION_AUTOTUNE_LOG_LEVEL=WARNING \
HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0 \
python -m scripts.helion_emit <name>
```

Env vars: `HELION_SKIP_CACHE=1` forces a fresh autotune; in-process
benchmarking (`..._SUBPROCESS=0`) avoids the autotuner's worker subprocesses
failing to open an exclusively-held GPU.

Autotune takes a few minutes. The script prints the chosen config and a
`make_tensor_descriptor occurrences` count — **confirm it is > 0** (descriptors
fired, not pointer fallback). Output lands in `kernels/<name>/triton_emitted.py`.

If your GPU is on a remote host/cluster, sync `scripts/` and
`kernels/<name>/helion_kernel.py` there, run the command above, and copy
`triton_emitted.py` back. (The job scheduler / SSH specifics are environment
dependent and out of scope for this skill.)

### Stage 3 — arg-ify constexpr (SCRIPT, local, no GPU)

```bash
python -m scripts.argify_constexpr <name>
```

Reads `triton_emitted.py`, writes `triton_helion_roundtrip.py`. Pure `ast`
transform (see the script): every module-level `X = tl.constexpr(V)` is removed,
added as a `X: tl.constexpr` parameter on the `@triton.jit` kernel, and threaded
into the `_launcher(...)` call as a kwarg. The wrapper keeps its local
`_BLOCK_SIZE_N = V` defs (the launch grid is computed from them; the same locals
feed the new kwargs, so kernel and grid never disagree) — and for a block size
the grid never uses (a **reduction axis**, e.g. K in matmul), Helion emits no
wrapper local, so the script re-injects one from the captured literal. Errors
loudly if the expected structure is missing. It also **deletes
`triton_emitted.py`** — redundant once argified exists.

### Stage 4 — name the arg-ified constexprs (MODEL judgment)

Stage 3 leaves every lifted constexpr param with Helion's positional name
(`_BLOCK_SIZE_0`, `_BLOCK_SIZE_1`, …, and any other `tl.constexpr` it baked) —
meaningless out of context. Rename each to what it actually is, using the
**original kernel** and how the param is used in `triton_helion_roundtrip.py`:
- read each param's uses to infer its role — a tile size matched to an axis by
  `tl.cdiv(<dim>, X)` in the grid (parallel), the `tl.range(0, <dim>, X)` step
  (reduction), a descriptor `block_shape`, or `tl.arange(0, X)` lanes; any other
  constexpr by where it feeds the computation.
- name it as the source kernel names that concept: matmul tile sizes → `BLOCK_M`
  / `BLOCK_N` / `BLOCK_K`; a row-reduction norm → `BLOCK_ROW` / `BLOCK_D`; a
  non-tile flag or count → whatever the original calls it.
- this is judgment, not a regex — the mapping differs per kernel, and only the
  original kernel says what each constexpr *means*. Do not script it.

Rename **every** occurrence of each constexpr consistently (kernel signature
param, body uses, wrapper local, `_launcher` kwarg) so kernel and grid still
agree, then `python -m py_compile` to confirm it still parses.

### Cleanup

`triton_helion_roundtrip.py` is the only artifact worth keeping. Stage 3 already
removes `triton_emitted.py`; also delete the stage-1 intermediate:

```bash
rm kernels/<name>/helion_kernel.py
```

## Output file structure

Stage 3 turns this:
```python
_BLOCK_SIZE_0 = tl.constexpr(256)          # module-level literal

@triton.jit
def _helion_<name>(x, y, out, ..., m, n, k):
    ... uses _BLOCK_SIZE_0 ...
```
into this:
```python
@triton.jit
def _helion_<name>(x, y, out, ..., m, n, k,
                   _BLOCK_SIZE_0: tl.constexpr, ...):     # now a param
    ... uses _BLOCK_SIZE_0 ...

def <name>_helion(...):
    _BLOCK_SIZE_0 = 256                                   # wrapper local (grid)
    _launcher(_helion_<name>, (grid...), ..., m, n, k,
              _BLOCK_SIZE_0=_BLOCK_SIZE_0, ...)            # threaded kwarg
```
Stage 4 then renames each positional param to a meaningful name (matmul shown):
```python
@triton.jit
def _helion_<name>(x, y, out, ..., m, n, k,
                   BLOCK_M: tl.constexpr, ...):           # named after its axis
    ... uses BLOCK_M ...

def <name>_helion(...):
    BLOCK_M = 256
    _launcher(_helion_<name>, (grid...), ..., m, n, k, BLOCK_M=BLOCK_M, ...)
```

## Document the conversion

Append a `## Helion round-trip` section to `kernels/<name>/conversion-notes.md`
(shared file; leave other sections untouched). Record: the winning autotune
config, whether descriptors fired, and any non-obvious stage-1 mapping choice.

## Checklist

1. [ ] `kernels/<name>/helion_kernel.py` — fn named `<name>_helion`, f32 accumulate, forward-only
2. [ ] `example_args(dev)` defined in `helion_kernel.py`
3. [ ] Stage 2 ran on a TMA-capable GPU, `make_tensor_descriptor occurrences > 0`
4. [ ] `python -m scripts.argify_constexpr <name>` → `triton_helion_roundtrip.py`
5. [ ] Stage 4: every arg-ified constexpr renamed from its positional name to a meaningful one (model), all occurrences updated consistently
6. [ ] `python -m py_compile kernels/<name>/triton_helion_roundtrip.py` passes
7. [ ] `grep -n "_BLOCK_SIZE\|tl.constexpr" triton_helion_roundtrip.py` — Helion's positional names gone (module level + params); meaningful constexpr params present as `: tl.constexpr` and as `_launcher` kwargs
8. [ ] `conversion-notes.md` updated with the Helion round-trip section
9. [ ] Cleanup: `helion_kernel.py` removed (stage 3 already removed `triton_emitted.py`); only `triton_helion_roundtrip.py` remains

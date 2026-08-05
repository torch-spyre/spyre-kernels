"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.mul.16_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py`` docstring
for the exact path):

- ``assert_size_stride(arg0_1, (1, 1, 14336), (14336, 14336, 1))`` and
  ``assert_size_stride(arg1_1, (1, 1, 14336), (14336, 14336, 1))`` —
  logical inputs, identical shape/dtype (``torch.float16``), no broadcast
  dim.
- ``buf0 = spyre_empty_with_layout((1, 1, 14336), (14336, 14336, 1),
  torch.float16, SpyreTensorLayout(device_size=[1, 224, 1, 64], ...))`` —
  logical output, same device layout as both inputs.
- ``triton_unk_fused_mul_0.run(arg0_1, arg1_1, buf0, 14336, stream=stream0)``
  — the kernel is launched with ``xnumel=14336``.
- ``config={'XBLOCK': 448}`` — not a power of 2, so substituted with the
  literal ``64`` (the value actually usable by ``tl.arange`` on this dead
  code path, per Spyre kernel convention); ``triton_meta={..., 'spyre_grid':
  (32,)}`` — 32-program grid, each covering a 7-row slab of the padded
  ``[224, 64]`` device layout (``32 * 7 == 224``).

Structurally identical to ``torch.mul.8_spyre`` (plain elementwise multiply
of two identical-shape tensors, no broadcast, `in_ptr0`/`out_ptr0` share
device layout), just with a "row" count of 1 (logical shape `[1, 1, 14336]`)
instead of `torch.mul.8_spyre`'s 12 (logical shape `[1, 12, 14336]`) — the
physical row axis is nonetheless 224 in both cases, since it derives from
`14336 / 64` regardless of the logical batch/row split.
"""

import numpy as np
import torch

from . import triton_kernel


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "in_ptr1":  "*fp16",
    "out_ptr0": "*fp16",
    "xnumel":   "i32",
    "XBLOCK":   "i32",
}


# ---------------------------------------------------------------------------
# Launch parameters, tags/doc, from the traced ``.run(...)`` call and
# ``spyre_grid`` (see original meta.py's VARIANTS["default"]).
# ---------------------------------------------------------------------------

TAGS = ["descriptor-load-static", "descriptor-store-static", "program-id-1d"]

SUMMARY = (
    "Plain elementwise multiply of two identically-shaped "
    "`f16[1, 1, 14336]` operands on Meta-Llama-3.1-8B-Instruct's "
    "traced `torch.mul.16` op -- no broadcast dim, same pattern as "
    "`torch.mul.8_spyre` but with a row count of 1 instead of 12."
)

DOC = (
    "Multiplies two logical `[1, 1, 14336]` f16 tensors elementwise, "
    "no broadcast. On the Spyre device layout the 14336-wide "
    "innermost dim splits into 224 64-element f16 sticks, giving "
    "physical shape `[1, 224, 1, 64]` shared identically by both "
    "inputs and the output (no axis reordering). The grid is sized "
    "to 32 programs, each covering a fixed 7-row slab "
    "(`32 * 7 == 224`). `xnumel`/`XBLOCK`/`xindex` are boilerplate "
    "inherited from Inductor's pointwise codegen: `xnumel` is "
    "immediately overwritten with the literal `14336`, and `xmask` "
    "is hardcoded to all-`True` (`tl.full([XBLOCK], True, "
    "tl.int1)`) rather than derived from `xindex`, confirmed dead "
    "by the traced `.ttir` (no `tt.load`/`tt.store` depends on "
    "either). The source `config={'XBLOCK': 448}` is not a power "
    "of 2 (required by `tl.arange` even on this dead code path), "
    "so it is substituted here with `64`, matching the convention "
    "used across the rest of the `torch.mul.N_spyre` family (e.g. "
    "`torch.mul.2/4/5/8_spyre`)."
)

CONSTEXPR = ["XBLOCK"]
XNUMEL = 14336
XBLOCK = 64
GRID = (32,)

# grid (32) * 7 (rows per program) == 224 (the padded axis's row
# count): every program gets a fixed 7-row chunk with nothing left
# over, same reasoning as torch.mul.1_spyre's
# distribution_loop: False.
DISTRIBUTION_LOOP = False

OUTPUT_KEY = "out_ptr0"


# ---------------------------------------------------------------------------
# Wrapper — launches the traced kernel on real tensors.
# ---------------------------------------------------------------------------

def mul(
    in_ptr0: torch.Tensor,
    in_ptr1: torch.Tensor,
    kernel_fn=triton_kernel.triton_unk_fused_mul_0,
) -> torch.Tensor:
    """Multiplies two identically-shaped tensors elementwise (device-layout
    shape `[1, 224, 1, 64]`), no broadcast dim, output sharing `in_ptr0`'s
    own device layout. See `DOC` above for the full derivation."""
    out_ptr0 = torch.empty_like(in_ptr0)
    kernel_fn[GRID](in_ptr0, in_ptr1, out_ptr0, XNUMEL, XBLOCK=XBLOCK)
    return out_ptr0


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker — preserved from the original
# meta.py for standalone verification independent of the wrapper above.
# ---------------------------------------------------------------------------

def make_inputs(xnumel: int = XNUMEL, XBLOCK: int = XBLOCK, **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    ``xnumel``/``XBLOCK`` are accepted so the signature matches the full
    param set, but neither shapes the data: the kernel body reassigns
    ``xnumel = 14336`` and never reads ``xindex``/``xmask`` to gate the
    descriptor load/store, so the buffers are built directly at the shared
    device-layout shape `[1, 224, 1, 64]` (see ``triton_kernel.py``).
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((1, 224, 1, 64)).astype(np.float16)
    in_ptr1 = rng.standard_normal((1, 224, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 224, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: plain elementwise `aten.mul.Tensor`, in the kernel's
    own compute precision (f16 multiply of two f16 tensors, no broadcast,
    no intermediate reshape/permute since in_ptr0/in_ptr1/out_ptr0 all
    share the identical device-layout shape)."""
    x = inputs["in_ptr0"]
    y = inputs["in_ptr1"]
    return (x.astype(np.float32) * y.astype(np.float32)).astype(np.float16)

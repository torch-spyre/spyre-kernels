"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.rsqrt.1_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py`` docstring
for the exact path):

- ``assert_size_stride(arg0_1, (1, 12, 1), (12, 1, 1))`` — logical input,
  dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 12, 1), (12, 1, 1), torch.float16,
  SpyreTensorLayout(device_size=[12, 1, 1, 64], ...))`` — logical output,
  same shape/dtype as the input.
- ``triton_unk_fused_rsqrt_0.run(arg0_1, buf0, 12, stream=stream0)`` — the
  kernel is launched with ``xnumel=12``.
- ``triton_meta={..., 'spyre_grid': (12,)}`` — 12-program grid, one core
  per row of the device-layout shape ``[12, 1, 1, 64]``, exactly like
  ``torch.add.1_spyre``.

The kernel itself hardcodes shape ``[12, 1, 1, 64]`` into both tensor
descriptors, so the pointer args here are built directly at that shape.
Since ``rsqrt`` is undefined for non-positive inputs, the input generator
draws strictly positive values (unlike the signed-normal draws used by the
elementwise-add examples).
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
    "Elementwise `out = 1/sqrt(in)` on Meta-Llama-3.1-8B-Instruct's "
    "traced `torch.rsqrt.1` op, one program per row."
)

DOC = (
    "Computes `rsqrt` of every element of a logical `[1, 12, 1]` "
    "f16 tensor. On the Spyre device layout the innermost dim pads "
    "to a 64-element f16 stick, giving a physical shape "
    "`[12, 1, 1, 64]`; the grid is sized to exactly 12 programs "
    "(one per row), and each program reads `tl.program_id(0)` "
    "directly as its row index — no `num_programs`/`cdiv` "
    "distribution loop in the source. `xnumel`/`XBLOCK`/`xindex`/"
    "`xmask` are boilerplate inherited from Inductor's pointwise "
    "codegen: `xnumel` is immediately overwritten with the literal "
    "`12` and `xindex`/`xmask` are never read, confirmed dead by "
    "the traced `.ttir` (no `tt.load`/`tt.store` depends on them)."
)

CONSTEXPR = ["XBLOCK"]
XNUMEL = 12
XBLOCK = 1
GRID = (12,)

# grid (12) == xnumel (12): one program per row, so DistributeWork
# emits ktdp.get_compute_tile_id but no residual scf.for — there is
# no remaining work per program to loop over. Same shape as
# torch.add.1_spyre; verified empirically via
# test_work_distribution rather than assumed.
DISTRIBUTION_LOOP = False

OUTPUT_KEY = "out_ptr0"


# ---------------------------------------------------------------------------
# Wrapper — launches the traced kernel on real tensors.
# ---------------------------------------------------------------------------

def rsqrt(in_ptr0: torch.Tensor, kernel_fn=triton_kernel.triton_unk_fused_rsqrt_0) -> torch.Tensor:
    """Computes `rsqrt` of every element of `in_ptr0`: `out = 1/sqrt(in_ptr0)`."""
    out_ptr0 = torch.empty_like(in_ptr0)
    kernel_fn[GRID](in_ptr0, out_ptr0, XNUMEL, XBLOCK=XBLOCK)
    return out_ptr0


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker — preserved from the original
# meta.py for standalone verification independent of the wrapper above.
# ---------------------------------------------------------------------------

def make_inputs(xnumel: int = XNUMEL, XBLOCK: int = XBLOCK, **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    ``xnumel``/``XBLOCK`` are accepted so the signature matches the full
    param set, but neither shapes the data: the kernel body reassigns
    ``xnumel = 12`` and never reads ``xindex``/``xmask``, so the buffers are
    built directly at the descriptor's hardcoded shape ``[12, 1, 1, 64]``
    (see ``triton_kernel.py``). Values are drawn strictly positive (uniform
    in ``[0.1, 10.1)``) since ``rsqrt`` of a non-positive input is undefined.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = (rng.random((12, 1, 1, 64)) * 10.0 + 0.1).astype(np.float16)
    out_ptr0 = np.zeros((12, 1, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.rsqrt.default(x)`` in the kernel's own compute
    precision (the kernel itself casts to f32 for `tl.rsqrt`, then back to
    f16: ``tl.rsqrt(tmp0.to(tl.float32)).to(tl.float16)``)."""
    x = inputs["in_ptr0"].astype(np.float32)
    return (1.0 / np.sqrt(x)).astype(np.float16)

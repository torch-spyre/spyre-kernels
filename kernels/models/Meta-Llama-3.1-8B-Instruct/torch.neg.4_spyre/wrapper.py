"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.neg.4_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py`` docstring
for the exact path):

- ``assert_size_stride(arg0_1, (1, 8, 1, 64), (512, 64, 64, 1))`` — logical
  input, dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 8, 1, 64), (512, 64, 64, 1),
  torch.float16, SpyreTensorLayout(device_size=[8, 1, 1, 1, 64], ...))`` —
  logical output, same shape/dtype as the input.
- ``triton_unk_fused_neg_0.run(arg0_1, buf0, 512, stream=stream0)`` — the
  kernel is launched with ``xnumel=512``.
- ``triton_meta={..., 'spyre_grid': (8,)}`` — 8-program grid (grid !=
  xnumel): each program handles one `dim0` row, a `[1, 1, 1, 1, 64]` block
  (64 elements, matching `XBLOCK=64`).

The kernel itself hardcodes shape ``[8, 1, 1, 1, 64]`` into both tensor
descriptors, so the pointer args here are built directly at that shape.
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
    "Elementwise `out = -in` on Meta-Llama-3.1-8B-Instruct's traced "
    "`torch.neg.4` op, one program per `dim0` row."
)

DOC = (
    "Negates every element of a logical `[1, 8, 1, 64]` f16 "
    "tensor. On the Spyre device layout the innermost dim pads to a "
    "64-element f16 stick, giving a physical shape "
    "`[8, 1, 1, 1, 64]`; the grid is sized to exactly 8 programs "
    "(one per `dim0` row of 64 elements, matching `XBLOCK=64`), and "
    "each program reads `tl.program_id(0)` directly as its row "
    "index — no `num_programs`/`cdiv` distribution loop in the "
    "source. `xnumel` (512) is a dead Inductor-side placeholder "
    "unrelated to the grid; the real per-program work unit is "
    "desc_0's shape[0] (8), tiled by block_shape[0]==1 with grid=8 "
    "-- nothing left over, so `distribution_loop: False` applies "
    "here the same as `torch.add.1_spyre`/`torch.rsqrt.1_spyre`, "
    "just expressed via the real descriptor tiling instead of the "
    "dead xnumel/XBLOCK placeholders. `xindex`/`xmask` "
    "are boilerplate inherited from Inductor's pointwise codegen: "
    "`xnumel` is immediately overwritten with the literal `512`; "
    "`xmask = xindex < xnumel` is a real bounds check but the "
    "descriptor indexing never reads `xindex`/`xmask` either way — "
    "confirmed dead by the traced `.ttir`."
)

CONSTEXPR = ["XBLOCK"]
XNUMEL = 512
XBLOCK = 64
GRID = (8,)

# grid (8) exactly tiles desc_0's shape[0] (8) with block_shape[0]
# == 1: nothing left over, so DistributeWork emits no residual
# scf.for -- same generalized rule as add.1/matmul.1.
DISTRIBUTION_LOOP = False

OUTPUT_KEY = "out_ptr0"


# ---------------------------------------------------------------------------
# Wrapper — launches the traced kernel on real tensors.
# ---------------------------------------------------------------------------

def neg(in_ptr0: torch.Tensor, kernel_fn=triton_kernel.triton_unk_fused_neg_0) -> torch.Tensor:
    """Negates every element of `in_ptr0`: `out = -in_ptr0`."""
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
    ``xnumel = 512`` and never reads ``xindex``/``xmask`` for the actual
    descriptor indexing (`dim0`..`dim4` are derived straight from
    `tl.program_id(0)`), so the buffers are built directly at the
    descriptors' hardcoded shape ``[8, 1, 1, 1, 64]`` (see
    ``triton_kernel.py``).
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((8, 1, 1, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((8, 1, 1, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.neg.default(x)`` in the kernel's own compute
    precision (extend to f32, negate, truncate back to f16)."""
    x = inputs["in_ptr0"].astype(np.float32)
    return (-x).astype(np.float16)

"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.matmul.2_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py``'s
docstring for the exact path):

- ``assert_size_stride(arg0_1, (1, 64, 1), (64, 1, 1))`` and
  ``assert_size_stride(arg1_1, (1, 1, 1), (1, 1, 1))`` — logical inputs,
  dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 64, 1), (64, 1, 1), torch.float16,
  SpyreTensorLayout(device_size=[64, 1, 1, 64], ...))`` — logical output
  ``f16[1, 64, 1]``.
- ``triton_unk_fused_bmm_0.run(arg0_1, arg1_1, buf0, 64, stream=stream0)``
  — launched with ``xnumel=64``.
- ``config={'XBLOCK': 2}``, ``triton_meta={..., 'spyre_grid': (32,)}`` — a
  32-program grid, each covering 2 rows (``32 * 2 == 64 == xnumel``).

Unlike ``torch.matmul.1_spyre`` (where the second operand carries 12 real
output columns in its padded stick, broadcast across rows), both the batch
and contracted dimensions are size 1 *and* the second operand is itself a
single scalar here, so ``aten.bmm`` degenerates into a broadcast scalar
multiply: the second operand's stick (``in_ptr1``, shape ``[1, 1, 64]``,
only element 0 logically real) is loaded once per program and broadcast
across that program's 2-row tile of the first operand. Every load/store
shape lines up exactly (no shape-mismatch bug, unlike ``torch.float.1_spyre``
/``torch.float.3_spyre``), so the NumPy oracle below just replicates the
same stick-wise broadcast multiply on the physical (padded) buffers.
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
    "Broadcast scalar multiply implementing a doubly-degenerate "
    "`torch.matmul.2` (`aten.bmm`, batch dim and contracted dim "
    "both size 1) on Meta-Llama-3.1-8B-Instruct's traced op."
)

DOC = (
    "`aten.bmm.default` of logical `f16[1, 64, 1]` and "
    "`f16[1, 1, 1]` tensors has both its batch and contracted "
    "dimensions equal to 1, so it degenerates into a broadcast "
    "*scalar* multiply rather than a real matmul. The first "
    "operand's device layout pads its size-1 feature dim out to a "
    "full 64-wide f16 stick (`[1, 64, 64]`, only element 0 of each "
    "row logically real); the second operand is a single scalar, "
    "likewise padded to a 64-wide stick (`[1, 1, 64]`). "
    "`tl.program_id(0)` selects a 2-row tile (`XBLOCK=2`) of the "
    "64-row first operand; the scalar's stick is loaded once per "
    "program and broadcast across those 2 rows before the "
    "multiply. Every load/store shape lines up exactly here — no "
    "shape-mismatch bug, unlike `torch.float.1_spyre`."
)

CONSTEXPR = ["XBLOCK"]
XNUMEL = 64
XBLOCK = 2
GRID = (32,)

# grid (32) * XBLOCK (2) == xnumel (64): every program gets a fixed
# 2-row chunk with nothing left over, same reasoning as
# torch.matmul.1_spyre's distribution_loop: False.
DISTRIBUTION_LOOP = False

OUTPUT_KEY = "out_ptr0"


# ---------------------------------------------------------------------------
# Wrapper — launches the traced kernel on real tensors.
# ---------------------------------------------------------------------------

def matmul(
    in_ptr0: torch.Tensor,
    in_ptr1: torch.Tensor,
    kernel_fn=triton_kernel.triton_unk_fused_bmm_0,
) -> torch.Tensor:
    """Broadcast scalar multiply implementing the doubly-degenerate bmm:
    each of `in_ptr0`'s padded rows is multiplied by `in_ptr1`'s single
    padded scalar stick. Output shape matches `in_ptr0`, so `empty_like`
    is used.
    """
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
    param set, but neither shapes the data: the buffers are built directly
    at each descriptor's full (non-tiled) ``shape`` (see
    ``triton_kernel.py``): ``in_ptr0``/``out_ptr0`` at ``[1, 64, 64]`` and
    ``in_ptr1`` at ``[1, 1, 64]``.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((1, 64, 64)).astype(np.float16)
    in_ptr1 = rng.standard_normal((1, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 64, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: broadcast scalar multiply of the padded sticks, in
    the kernel's own compute precision (extend to f32, multiply, truncate
    back to f16)."""
    x = inputs["in_ptr0"].astype(np.float32).reshape(64, 64)   # [row, stick]
    w = inputs["in_ptr1"].astype(np.float32).reshape(64)        # [stick]
    out = x * w[None, :]
    return out.reshape(1, 64, 64).astype(np.float16)

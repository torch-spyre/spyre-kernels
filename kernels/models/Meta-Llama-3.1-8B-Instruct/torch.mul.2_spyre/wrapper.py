"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.mul.2_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py`` docstring
for the exact path):

- ``assert_size_stride(arg0_1, (1, 12, 4096), (49152, 4096, 1))`` — logical
  ``x``, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (1, 12, 1), (12, 1, 1))`` — logical ``y``, a
  true per-row scalar broadcast over the last dim of ``x``.
- ``buf0 = spyre_empty_with_layout((1, 12, 4096), (49152, 4096, 1),
  torch.float16, SpyreTensorLayout(device_size=[1, 64, 12, 64], ...))`` —
  logical output, same logical shape as ``x`` but a *different* physical
  device layout (row/stick axes swapped relative to ``in_ptr0``).
- ``triton_unk_fused_mul_0.run(arg0_1, arg1_1, buf0, 49152, stream=stream0)``
  — the kernel is launched with ``xnumel=49152``.
- ``triton_meta={..., 'spyre_grid': (32,)}`` — 32-program grid, each program
  handling a 2-stick-wide (128-wide logical) slab of the 64-stick-wide
  ``in_ptr0``/``out_ptr0`` dim.

The three descriptors are NOT all the same shape (unlike ``torch.mul.8/16``):
``desc_0`` (``in_ptr0``) has device shape ``[12, 64, 1, 64]``, ``desc_1``
(``in_ptr1``) has device shape ``[12, 1, 1, 64]``, and ``desc_2`` (out) has
device shape ``[1, 64, 12, 64]``. The kernel body reshapes/permutes/
broadcasts ``in_ptr0``'s per-tile block from ``[12, 2, 64]`` into
``[1, 2, 12, 64]`` (swapping the row axis and the stick axis) before
multiplying by ``in_ptr1``'s block reshaped to ``[1, 1, 12, 64]`` and
broadcast across the (2-wide, tile-local) stick axis. The NumPy oracle
below replicates this at full scale (all 64 sticks / all 12 rows at once,
matching how ``ktir_cpu`` executes the whole grid and produces a result in
``out_ptr0``'s own device-layout order) rather than per-tile.
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
    "Broadcasting `out = x * y` on Meta-Llama-3.1-8B-Instruct's "
    "traced `torch.mul.2` op: a per-row scalar `y` broadcast over "
    "`x`'s last logical dim, with a row/stick axis swap between "
    "`x`'s and `out`'s device layouts."
)

DOC = (
    "Multiplies a logical `[1, 12, 4096]` f16 tensor `x` by a "
    "logical `[1, 12, 1]` f16 tensor `y` broadcast over the last "
    "dim. On the Spyre device layout `x`'s 4096-wide innermost "
    "dim splits into 64 64-element f16 sticks (`[12, 64, 1, 64]`), "
    "while the output stores those same 64 sticks with the row "
    "axis and stick axis swapped (`[1, 64, 12, 64]`) — the kernel "
    "reshapes/permutes/reshapes `x`'s tile to match before "
    "multiplying. `y`'s device layout is `[12, 1, 1, 64]`, "
    "replicated across all 64 lanes of its single stick (a true "
    "per-row scalar, not a real vector); the kernel broadcasts it "
    "over the 2-stick-wide tile before multiplying. The grid is "
    "sized to exactly 32 programs, each covering a 128-wide "
    "(2-stick) slab of the innermost dim. `xnumel`/`XBLOCK`/"
    "`xindex` are boilerplate inherited from Inductor's pointwise "
    "codegen: `xnumel` is immediately overwritten with the "
    "literal `49152`, and `xmask` is hardcoded to all-`True` "
    "(`tl.full([XBLOCK], True, tl.int1)`) rather than derived from "
    "`xindex`, confirmed dead by the traced `.ttir` (no "
    "`tt.load`/`tt.store` depends on either). NOTE: the source's "
    "own `config={'XBLOCK': 1536}` is not a power of 2, which "
    "trips `tl.arange`'s power-of-2 requirement on the (dead) "
    "`xindex` line above even though the value is never read; "
    "since XBLOCK has no effect on the actual computation, "
    "`params` below substitutes `64` (reviewer: double-check this "
    "substitution, since it deviates from the literal source "
    "value — same rationale applies to `torch.mul.4/5/8/16`)."
)

CONSTEXPR = ["XBLOCK"]
XNUMEL = 49152
XBLOCK = 64
GRID = (32,)

# grid (32) exactly partitions in_ptr0's 64-wide stick axis into
# 32 2-stick tiles (block_shape[1]=2), and in_ptr1/out_ptr0 have no
# residual axis either — DistributeWork emits
# ktdp.get_compute_tile_id but no residual scf.for, mirroring
# add.1's "grid == xnumel" reasoning even though this kernel's own
# (dead) xnumel/XBLOCK never actually drive the tiling (verified
# empirically: test_work_distribution's assert_present("scf.for")
# fails without this flag).
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
    """Multiplies `in_ptr0` (device-layout shape `[12, 64, 1, 64]`) by
    `in_ptr1` (a per-row scalar, device-layout shape `[12, 1, 1, 64]`),
    storing into a differently-laid-out output `[1, 64, 12, 64]` (row/stick
    axes swapped). See `DOC` above for the full derivation."""
    out_ptr0 = torch.empty(
        (1, 64, 12, 64), dtype=in_ptr0.dtype, device=in_ptr0.device
    )
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
    ``xnumel = 49152`` and never reads ``xindex``/``xmask`` to gate the
    descriptor load/store, so the buffers are built directly at each
    descriptor's hardcoded shape (see ``triton_kernel.py``).

    ``in_ptr1``'s logical last dim is 1 (a true per-row scalar), so every
    one of the 64 physical lane values within a row's stick must be
    identical — the real value only varies along the row axis (dim0, size
    12), not along the lane axis. Built via ``np.broadcast_to`` + a copy so
    the array is writable/contiguous.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((12, 64, 1, 64)).astype(np.float16)
    row_scalars = rng.standard_normal((12, 1, 1, 1)).astype(np.float32)
    in_ptr1 = np.broadcast_to(row_scalars, (12, 1, 1, 64)).astype(np.float16).copy()
    out_ptr0 = np.zeros((1, 64, 12, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.mul.Tensor(x, y)`` (``y`` broadcast over ``x``'s
    last logical dim), replicated at full scale in each pointer's own
    device-layout order, mirroring the kernel body's
    reshape/permute/reshape (on ``in_ptr0``) and reshape/broadcast_to (on
    ``in_ptr1``) sequence:

    - ``in_ptr0``: ``[12, 64, 1, 64]`` -> squeeze dim2 -> ``[12, 64, 64]``
      -> transpose(1, 0, 2) -> ``[64, 12, 64]`` -> add leading 1 ->
      ``[1, 64, 12, 64]`` (matches ``desc_2``'s / the output's shape).
    - ``in_ptr1``: ``[12, 1, 1, 64]`` -> reshape (order-preserving, since
      the reshaped-away dims are both size 1) -> ``[1, 1, 12, 64]`` ->
      broadcast over the 64-wide stick axis -> ``[1, 64, 12, 64]``.
    """
    x = inputs["in_ptr0"].astype(np.float32)
    y = inputs["in_ptr1"].astype(np.float32)

    x = x.reshape(12, 64, 64)
    x = np.transpose(x, (1, 0, 2))
    x = x.reshape(1, 64, 12, 64)

    y = y.reshape(1, 1, 12, 64)
    y = np.broadcast_to(y, (1, 64, 12, 64))

    return (x * y).astype(np.float16)

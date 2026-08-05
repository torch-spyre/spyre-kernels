"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.mul.1_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py`` docstring
for the exact path):

- ``assert_size_stride(arg0_1, (1, 12, 128), (1536, 128, 1))`` — logical
  input, dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 12, 128), (1536, 128, 1), torch.float16,
  SpyreTensorLayout(device_size=[12, 2, 1, 64], ...))`` — logical output,
  same shape/dtype as the input.
- ``triton_unk_fused_mul_0.run(arg0_1, buf0, 1536, stream=stream0)`` — the
  kernel is launched with ``xnumel=1536``.
- ``triton_meta={..., 'spyre_grid': (24,)}`` — 24-program grid: the
  device-layout shape ``[12, 2, 1, 64]`` splits into 24 one-element-wide
  ``[1, 1, 1, 64]`` tiles (12 rows x 2 sub-chunks of the padded 128-wide
  last logical dim), one tile per program.

The kernel itself hardcodes shape ``[12, 2, 1, 64]`` into both tensor
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
    "Elementwise `out = in * 1.0` on Meta-Llama-3.1-8B-Instruct's "
    "traced `torch.mul.1` op, one program per (row, half-stick) pair."
)

DOC = (
    "Multiplies every element of a logical `[1, 12, 128]` f16 "
    "tensor by the compile-time constant `1.0`. On the Spyre "
    "device layout the innermost dim splits into two 64-element "
    "f16 sticks, giving a physical shape `[12, 2, 1, 64]`; the "
    "grid is sized to exactly 24 programs (`12 * 2`), and each "
    "program reads `tl.program_id(0) // 2` / `% 2` directly as "
    "its row/stick index — no `num_programs`/`cdiv` distribution "
    "loop in the source. `xnumel`/`XBLOCK`/`xindex`/`xmask` are "
    "boilerplate inherited from Inductor's pointwise codegen: "
    "`xnumel` is immediately overwritten with the literal `1536` "
    "and `xindex`/`xmask` are never read, confirmed dead by the "
    "traced `.ttir` (no `tt.load`/`tt.store` depends on them)."
)

CONSTEXPR = ["XBLOCK"]
XNUMEL = 1536
XBLOCK = 64
GRID = (24,)

# grid (24) exactly partitions the (12 rows x 2 sticks) space into
# one program per (row, stick) pair — DistributeWork emits
# ktdp.get_compute_tile_id but no residual scf.for. Empirically
# verified via test_work_distribution's assert_present("scf.for")
# failing without this flag; note this kernel's own xnumel (1536)
# != grid (24), unlike add.1's literal "grid == xnumel" case — the
# real per-program work unit here is defined by the (dead)
# xnumel/XBLOCK-independent program_id-derived tile offsets, which
# partition exactly with no remainder.
DISTRIBUTION_LOOP = False

OUTPUT_KEY = "out_ptr0"


# ---------------------------------------------------------------------------
# Wrapper — launches the traced kernel on real tensors.
# ---------------------------------------------------------------------------

def mul(in_ptr0: torch.Tensor, kernel_fn=triton_kernel.triton_unk_fused_mul_0) -> torch.Tensor:
    """Multiplies every element of `in_ptr0` (device-layout shape
    `[12, 2, 1, 64]`) by the compile-time constant `1.0`, one program per
    (row, stick) pair. See `DOC` above for the full derivation."""
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
    ``xnumel = 1536`` and never reads the resulting ``xmask`` values to
    gate the descriptor load/store (only ``dim0..dim3``, derived from
    ``tl.program_id(0)``, are used), so the buffers are built directly at
    the descriptors' hardcoded shape ``[12, 2, 1, 64]`` (see
    ``triton_kernel.py``).
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((12, 2, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((12, 2, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.mul.Tensor(x, 1.0)`` in the kernel's own
    compute precision (extend to f32, multiply, truncate back to f16).

    ``in_ptr0`` and ``out_ptr0`` share the same device-layout shape (no
    reshape/permute in the kernel body), so this is a plain elementwise
    multiply.
    """
    x = inputs["in_ptr0"].astype(np.float32)
    return (x * 1.0).astype(np.float16)

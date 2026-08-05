"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.add.8_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py`` docstring
for the exact path):

- ``assert_size_stride(arg0_1, (1, 1, 4096), (4096, 4096, 1))`` / same for
  ``arg1_1`` — two logical inputs, both dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 1, 4096), (4096, 4096, 1),
  torch.float16, SpyreTensorLayout(device_size=[1, 64, 1, 64], ...))`` —
  logical output, same shape/dtype as the inputs.
- ``triton_unk_fused_add_0.run(arg0_1, arg1_1, buf0, 4096, stream=stream0)``
  — the kernel is launched with ``xnumel=4096``.
- ``triton_meta={..., 'spyre_grid': (32,)}`` — 32-program grid (grid !=
  xnumel): each program covers a `[1, 2, 1, 64]` block (128 elements,
  matching `XBLOCK=128`) sliced out of the single `4096`-wide row via
  `tl.program_id(0) * 128`.

The kernel itself hardcodes shape ``[1, 64, 1, 64]`` into all three tensor
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
    "Elementwise `out = in0 + in1` on Meta-Llama-3.1-8B-Instruct's "
    "traced `torch.add.8` op, one program per `[1, 2, 1, 64]` "
    "block."
)

DOC = (
    "Adds two logical `[1, 1, 4096]` f16 tensors elementwise. On "
    "the Spyre device layout the flattened `4096`-wide row splits "
    "into a `[64, 1, 64]` stick shape, giving a physical shape "
    "`[1, 64, 1, 64]`; the grid is sized to 32 programs, each "
    "covering a `[1, 2, 1, 64]` block (128 elements, matching "
    "`XBLOCK=128`) via `tl.program_id(0) * 128` — no "
    "`num_programs`/`cdiv` distribution loop in the source. "
    "`xindex`/`xmask` are boilerplate inherited from Inductor's "
    "pointwise codegen: `xnumel` is immediately overwritten with the "
    "literal `4096` and `xmask` is a trivial all-`True` mask "
    "(`tl.full([XBLOCK], True, tl.int1)`), never read for masking."
)

CONSTEXPR = ["XBLOCK"]
XNUMEL = 4096
XBLOCK = 128
GRID = (32,)

# grid (32) exactly tiles desc_0's shape[1] (64) via block_shape[1]
# == 2: 64/2 == 32, nothing left over, so DistributeWork emits no
# residual scf.for -- same generalized rule as add.1/matmul.1.
DISTRIBUTION_LOOP = False

OUTPUT_KEY = "out_ptr0"


# ---------------------------------------------------------------------------
# Wrapper — launches the traced kernel on real tensors.
# ---------------------------------------------------------------------------

def add(
    in_ptr0: torch.Tensor,
    in_ptr1: torch.Tensor,
    kernel_fn=triton_kernel.triton_unk_fused_add_0,
) -> torch.Tensor:
    """Adds two f16 tensors elementwise: `out = in_ptr0 + in_ptr1`."""
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
    ``xnumel = 4096`` and never reads ``xindex``/``xmask`` (``xmask`` is a
    trivial all-``True`` mask), so the buffers are built directly at the
    descriptors' hardcoded shape ``[1, 64, 1, 64]`` (see
    ``triton_kernel.py``).
    """
    del xnumel, XBLOCK
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    in_ptr0 = rng0.standard_normal((1, 64, 1, 64)).astype(np.float16)
    in_ptr1 = rng1.standard_normal((1, 64, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 64, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.add.Tensor(x, y)`` in the kernel's own compute
    precision (extend both operands to f32, add, truncate back to f16)."""
    x = inputs["in_ptr0"].astype(np.float32)
    y = inputs["in_ptr1"].astype(np.float32)
    return (x + y).astype(np.float16)

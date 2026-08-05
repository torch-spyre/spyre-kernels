"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.mul.11_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py``'s
docstring for the exact path):

- ``assert_size_stride(arg0_1, (4096,), (1,))`` and
  ``assert_size_stride(arg1_1, (1, 1, 4096), (4096, 4096, 1))`` — logical
  inputs, dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 1, 4096), (4096, 4096, 1),
  torch.float16, SpyreTensorLayout(device_size=[1, 64, 1, 64], ...))`` —
  logical output, same shape/dtype as ``arg1_1``.
- ``triton_unk_fused_mul_0.run(arg0_1, arg1_1, buf0, 4096, stream=stream0)``
  — launched with ``xnumel=4096``.
- ``config={'XBLOCK': 128}``, ``triton_meta={..., 'spyre_grid': (32,)}`` —
  32-program grid, each covering 2 rows of the padded ``[64, 64]`` device
  layout (``32 * 2 == 64``).

Both operands' flat 4096-element shape collapses to the same physical
``[64, 64]`` device layout (64 rows of 64-element f16 sticks), and neither
carries an extra broadcast dim here (unlike ``torch.mul.3_spyre``), so this
is a plain elementwise multiply with all three descriptors sharing
identical ``shape``/``strides``/``block_shape``.
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
    "Plain elementwise multiply of two flat `f16[4096]`-shaped "
    "operands, on Meta-Llama-3.1-8B-Instruct's traced "
    "`torch.mul.11` op -- no broadcast dim, unlike "
    "`torch.mul.3_spyre`."
)

DOC = (
    "`aten.mul.Tensor` of a logical flat `f16[4096]` tensor and a "
    "logical `f16[1, 1, 4096]` tensor with the same element count "
    "and no extra batch dim, so both operands share the identical "
    "physical `[64, 64]` device layout (64 rows of 64-element f16 "
    "sticks) and the multiply is a plain elementwise op with no "
    "reshape/broadcast in the kernel body. `tl.program_id(0)` "
    "selects a 2-row tile (`c0 = program_id(0) * 128`, "
    "`dim0 = c0 // 64`) per program; `dim1` is always the literal "
    "`0` since `c0` is always a multiple of 64. No "
    "`xnumel`/`xoffset`/`xindex`/`xmask` boilerplate is present in "
    "this trace."
)

CONSTEXPR = ["XBLOCK"]
XNUMEL = 4096
XBLOCK = 128
GRID = (32,)

# grid (32) * 2 (dim0 step size) == 64 (the padded axis's row
# count): every program gets a fixed 2-row chunk with nothing left
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
    shape `[64, 64]`), no broadcast dim. See `DOC` above for the full
    derivation."""
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
    param set, but neither shapes the data (every descriptor index derives
    directly from ``tl.program_id(0)``, see ``triton_kernel.py``): all
    three buffers are built at the shared full ``shape`` ``[64, 64]``.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((64, 64)).astype(np.float16)
    in_ptr1 = rng.standard_normal((64, 64)).astype(np.float16)
    out_ptr0 = np.zeros((64, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: plain elementwise `aten.mul.Tensor`, in the kernel's
    own compute precision (f16 multiply, no intermediate f32 widening in
    the traced source -- unlike the scalar-constant `torch.mul.1_spyre`
    case, this kernel multiplies two f16 tensors directly)."""
    x = inputs["in_ptr0"]
    y = inputs["in_ptr1"]
    return (x.astype(np.float32) * y.astype(np.float32)).astype(np.float16)

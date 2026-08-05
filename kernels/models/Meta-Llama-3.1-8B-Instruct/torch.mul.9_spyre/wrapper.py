"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.mul.9_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py`` docstring
for the exact path):

- ``assert_size_stride(arg0_1, (1, 1, 128), (128, 128, 1))`` — logical
  input, dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 1, 128), (128, 128, 1), torch.float16,
  SpyreTensorLayout(device_size=[1, 2, 1, 64], ...))`` — logical output,
  same shape/dtype as the input.
- ``triton_unk_fused_mul_0.run(arg0_1, buf0, 128, stream=stream0)`` — the
  kernel is launched with ``xnumel=128``.
- ``triton_meta={..., 'spyre_grid': (2,)}`` — 2-program grid: the
  device-layout shape ``[1, 2, 1, 64]`` splits into 2 one-element-wide
  ``[1, 1, 1, 64]`` tiles (the padded 128-wide last logical dim splits
  into 2 sticks), one tile per program.

The kernel itself hardcodes shape ``[1, 2, 1, 64]`` into both tensor
descriptors, so the pointer args here are built directly at that shape.
This is the same shape of kernel as ``torch.mul.1_spyre`` (scalar `* 1.0`),
just with a smaller logical row count (1 instead of 12).
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
    "traced `torch.mul.9` op, one program per 64-element stick."
)

DOC = (
    "Multiplies every element of a logical `[1, 1, 128]` f16 "
    "tensor by the compile-time constant `1.0`. On the Spyre "
    "device layout the innermost dim splits into two 64-element "
    "f16 sticks, giving a physical shape `[1, 2, 1, 64]`; the "
    "grid is sized to exactly 2 programs, and each program reads "
    "`tl.program_id(0)` directly (scaled by 64) as its stick "
    "offset — no `num_programs`/`cdiv` distribution loop in the "
    "source. `xnumel`/`XBLOCK`/`xindex`/`xmask` are boilerplate "
    "inherited from Inductor's pointwise codegen: `xnumel` is "
    "immediately overwritten with the literal `128` and "
    "`xindex`/`xmask` are never read, confirmed dead by the "
    "traced `.ttir` (no `tt.load`/`tt.store` depends on them)."
)

CONSTEXPR = ["XBLOCK"]
XNUMEL = 128
XBLOCK = 64
GRID = (2,)

# grid (2) exactly partitions the 2-stick axis into one stick per
# program — no residual scf.for (see torch.mul.1_spyre/wrapper.py for
# the full rationale; empirically verified via
# test_work_distribution).
DISTRIBUTION_LOOP = False

OUTPUT_KEY = "out_ptr0"


# ---------------------------------------------------------------------------
# Wrapper — launches the traced kernel on real tensors.
# ---------------------------------------------------------------------------

def mul(in_ptr0: torch.Tensor, kernel_fn=triton_kernel.triton_unk_fused_mul_0) -> torch.Tensor:
    """Multiplies every element of `in_ptr0` (device-layout shape
    `[1, 2, 1, 64]`) by the compile-time constant `1.0`, one program per
    stick. See `DOC` above for the full derivation."""
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
    ``xnumel = 128`` and never reads ``xindex``/``xmask`` to gate the
    descriptor load/store (only ``dim0..dim3``, derived from
    ``tl.program_id(0)``, are used), so the buffers are built directly at
    the descriptors' hardcoded shape ``[1, 2, 1, 64]`` (see
    ``triton_kernel.py``).
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((1, 2, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 2, 1, 64), dtype=np.float16)
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

"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.zeros.1_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py``'s docstring for
the exact path):

- ``buf0 = spyre_constant_tensor(0.0, torch.device("spyre:0"),
  torch.float16)`` — a zero-valued "seed" tensor, dtype ``torch.float16``.
- ``buf1 = spyre_empty_with_layout((1, 8, 2048, 128), ...)`` — logical
  output ``f16[1, 8, 2048, 128]``, implementing
  ``aten.full.default([1, 8, 2048, 128], 0, dtype=torch.float16)``.
- ``triton_unk_fused_zeros_0.run(buf0, buf1, 2097152, stream=stream0)`` —
  launched with ``xnumel=2097152``.
- ``config={'XBLOCK': 65536}``, ``triton_meta={..., 'spyre_grid': (32,)}`` —
  a 32-program grid, each covering a 64-row chunk of the padded output
  tile's second axis (``32 * 64 == 2048``, exact).
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    ``xnumel``/``XBLOCK`` are accepted so the signature matches the full
    param set, but neither shapes the data (every descriptor index derives
    directly from ``tl.program_id(0)``, see ``kernel.py``): ``in_ptr0`` (the
    zero seed) is built at its full descriptor ``shape`` ``[1, 1, 1, 64]``
    (all zeros, matching the traced ``spyre_constant_tensor(0.0, ...)``),
    and ``out_ptr0`` at its full descriptor ``shape`` ``[1, 2048, 128, 64]``.
    """
    del xnumel, XBLOCK
    in_ptr0 = np.zeros((1, 1, 1, 64), dtype=np.float16)
    out_ptr0 = np.full((1, 2048, 128, 64), np.nan, dtype=np.float16)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: broadcast the zero-valued seed stick across the full
    output tile (``aten.full.default(..., 0, dtype=torch.float16)``)."""
    seed = inputs["in_ptr0"]  # [1, 1, 1, 64], all zeros
    return np.broadcast_to(seed, (1, 2048, 128, 64)).astype(np.float16)


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
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d", "broadcast"],
        "summary": (
            "Zero-fill implementing `torch.zeros.1` (`aten.full.default`) "
            "on Meta-Llama-3.1-8B-Instruct's traced op: loads a single "
            "zero-valued stick and broadcasts/stores it across the full "
            "output tile."
        ),
        "doc": (
            "`aten.full.default([1, 8, 2048, 128], 0, dtype=torch.float16)` "
            "is implemented by loading a single 64-element zero-valued "
            "stick (`in_ptr0`, the caller-side `spyre_constant_tensor(0.0, "
            "...)`) once per program, then broadcasting it (via "
            "`tl.reshape` + `tl.broadcast_to`, both no-ops on the actual "
            "element values -- just shape bookkeeping) up to that "
            "program's full `[1, 64, 128, 64]` store tile. "
            "`tl.program_id(0)` selects a 64-row chunk of the padded "
            "output's second axis (`c1 = program_id(0) * 64`); 32 programs "
            "cover all 2048 rows exactly, with nothing left over. Every "
            "load/broadcast/store shape lines up here -- no bug, unlike "
            "`torch.float.1_spyre`/`torch.index_copy_.2_spyre`."
        ),
        "kernel_fn":  kernel.triton_unk_fused_zeros_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [2097152], "XBLOCK": [65536]},
        "grid":       [32],
        # grid (32) * 64 (row step size) == 2048 (the padded axis's row
        # count): every program gets a fixed 64-row chunk with nothing left
        # over, same reasoning as torch.mul.1_spyre's
        # distribution_loop: False.
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}

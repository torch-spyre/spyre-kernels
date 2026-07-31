"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.index_copy_.3_spyre``.

Byte-for-byte the same kernel body as ``torch.index_copy_.2_spyre``
(confirmed by comparing both traces' ``output_code.py``); the shapes below
are independently re-verified against this op's own trace, not assumed from
its sibling.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py``'s docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 8, 2048, 128), (2097152, 262144, 128, 1))``
  — logical in-place target, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (1,), (1,))`` — logical index tensor, dtype
  ``torch.int64``.
- ``assert_size_stride(arg2_1, (1, 8, 1, 128), (1024, 128, 128, 1))`` —
  logical source values, dtype ``torch.float16``.
- ``triton_unk_fused_index_copy_0.run(arg1_1, arg2_1, arg0_1, 1024,
  stream=stream0)`` — launched with ``xnumel=1024``.
- ``config={'XBLOCK': 64}``, ``triton_meta={..., 'spyre_grid': (16,)}`` — a
  16-program grid.

This is a **disabled** example (see ``VARIANTS["default"]["disabled"]``):
the traced kernel never performs a real indexed copy (it ignores the
loaded/reshaped index tensor entirely) and its one real store has a
mismatched element count between the loaded value and the store's target
block. There is no meaningful NumPy oracle to write against buggy,
under-elaborated arithmetic like this, so ``run`` below is a documented
placeholder, never exercised while the variant is disabled.
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
    directly from ``tl.program_id(0)``, see ``kernel.py``): buffers are built
    at each descriptor's full ``shape`` — ``in_ptr0`` (the index tensor) at
    ``[1, 32]`` (``int64``), ``in_ptr1`` at ``[8, 1, 2, 1, 64]``, and
    ``out_ptr0`` at ``[8, 2048, 2, 1, 64]``.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.integers(0, 2048, size=(1, 32)).astype(np.int64)
    in_ptr1 = rng.standard_normal((8, 1, 2, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((8, 2048, 2, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """Placeholder oracle -- never exercised while ``disabled``. The traced
    kernel body's real store (``desc_2.store(..., tmp3)``) is a shape
    mismatch (loaded ``[1, 1, 1, 1, 64]`` vs. target block
    ``[1, 2048, 1, 1, 64]``), so there is no well-defined reference
    computation to match."""
    del inputs
    raise NotImplementedError(
        "torch.index_copy_.2_spyre's traced kernel has a shape-mismatched "
        "store and never uses its loaded index tensor; see the `disabled` "
        "reason in VARIANTS."
    )


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*i64",
    "in_ptr1":  "*fp16",
    "out_ptr0": "*fp16",
    "xnumel":   "i32",
    "XBLOCK":   "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d"],
        "summary": (
            "KV-cache-style `index_copy_` on "
            "Meta-Llama-3.1-8B-Instruct's traced `torch.index_copy_.3` op -- "
            "disabled, the traced kernel body is buggy."
        ),
        "doc": (
            "`aten.index_put_.default(arg0_1, [None, None, arg1_1], "
            "arg2_1)` is meant to scatter `arg2_1` (logical "
            "`f16[1, 8, 1, 128]`) into `arg0_1` (logical "
            "`f16[1, 8, 2048, 128]`) at the row selected by `arg1_1` "
            "(a single `int64` index). The traced kernel instead loads the "
            "index tensor (`desc_0` / `tmp0`), reshapes and broadcasts it "
            "into `tmp1`/`tmp2` -- and then never uses either: the real "
            "store (`desc_2.store(...)`) writes `tmp3`, loaded straight "
            "from `in_ptr1` via `desc_1`, unconditionally. That store is "
            "also shape-mismatched on its own terms: `tmp3`'s shape is "
            "`desc_1`'s block shape `[1, 1, 1, 1, 64]` (64 elements), but "
            "`desc_2`'s block shape is `[1, 2048, 1, 1, 64]` (131072 "
            "elements) -- the value being stored doesn't fill the target "
            "block. See `kernel.py`'s docstring for the full bug writeup."
        ),
        "kernel_fn":  kernel.triton_unk_fused_index_copy_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [1024], "XBLOCK": [64]},
        "grid":       [16],
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
        "disabled": {
            "reason": (
                "Traced kernel never performs a real indexed copy (the "
                "loaded/reshaped index tensor is dead) and its one real "
                "store is shape-mismatched: loaded value has 64 elements "
                "(`[1, 1, 1, 1, 64]`) but the store's target block expects "
                "131072 (`[1, 2048, 1, 1, 64]`)."
            ),
        },
    },
}

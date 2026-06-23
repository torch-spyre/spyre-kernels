# SPDX-License-Identifier: Apache-2.0
"""KTIR lowering driver for the rms_norm kernel variants.

Consumed by ``scripts/gen_ktir.py``, which lowers each entry in ``VARIANTS``
to ``kernels/rms_norm/<variant>.ktir``.

``VARIANTS`` maps a **variant name** (which is also the source ``.py`` module
name and the output ``.ktir`` stem) to the four things the round-trip lowering
needs:

    KERNEL      : the @triton.jit function to lower
    SIGNATURE   : dict[str, str]  arg name -> Triton type ("*fp16", "i32", ...)
    CONSTEXPRS  : dict[str, value] for every constexpr arg
    GRID        : optional list, forwarded to SpyreOptions.grid

To add another Spyre variant (e.g. a future ``spyre_aware.py``), import its
kernel and add an entry keyed by its module name.
"""

from kernels.rms_norm.tensor_descriptor import _rms_norm_kernel_td

# Concrete shapes match tests/ktir/test_rms_norm.py: 32 rows x 4096 cols.
_TD_SIGNATURE = {
    "input_ptr": "*fp16",
    "weight_ptr": "*fp16",
    "output_ptr": "*fp16",
    "n_rows": "i32",
    "n_cols": "i32",
    "eps": "fp16",
    "BLOCK_SIZE": "i32",
    "ROWS_PER_PROGRAM": "i32",
}

VARIANTS = {
    "tensor_descriptor": {
        "KERNEL": _rms_norm_kernel_td,
        "SIGNATURE": _TD_SIGNATURE,
        "CONSTEXPRS": {
            "BLOCK_SIZE": 1024,
            "ROWS_PER_PROGRAM": 1,
        },
        # 32 cores, 1D grid: cdiv(32, ROWS_PER_PROGRAM=1) = 32 programs, one/row.
        "GRID": [32],
    },
    # "spyre_aware": {
    #     "KERNEL": _rms_norm_kernel_sa,
    #     "SIGNATURE": {...},
    #     "CONSTEXPRS": {...},
    #     "GRID": [32],
    # },
}

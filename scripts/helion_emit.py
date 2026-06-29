#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Autotune-compile a Helion kernel back to Triton with the ``tensor_descriptor``
indexing lever pinned, and emit the generated Triton source.

Stage 2 of the helion-convert workflow. Imports
``kernels/<name>/helion_kernel.py``, re-decorates the kernel at runtime with
``static_shapes=False`` and ``autotune_config_overrides={"indexing":
"tensor_descriptor"}`` (so the autotuner is free on every knob *except* indexing,
which is forced to descriptors), autotunes against example args, and writes the
emitted Triton to ``kernels/<name>/triton_emitted.py``.

Must run on TMA-capable hardware (NVIDIA Hopper / H100) — ``tensor_descriptor``
lowering needs it; see the helion-convert SKILL.

Usage::

    python -m scripts.helion_emit <name>

``kernels/<name>/helion_kernel.py`` must define:
- the kernel function ``<name>_helion``
- ``example_args(dev) -> tuple`` returning representative, descriptor-eligible
  args (they drive autotuning, not correctness).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
import torch
import helion

ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = ROOT / "kernels"

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="kernel directory under kernels/")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA required (tensor_descriptor needs TMA-capable GPU)", file=sys.stderr)
        return 1

    mod = importlib.import_module(f"kernels.{args.name}.helion_kernel")
    kernel = getattr(mod, f"{args.name}_helion")
    if not hasattr(mod, "example_args"):
        print(f"error: kernels/{args.name}/helion_kernel.py must define "
              "example_args(dev) -> tuple", file=sys.stderr)
        return 1

    # Re-decorate the underlying function with the stage-2 knobs. `.fn` is the
    # original undecorated function held by the Helion Kernel object.
    rekernel = helion.kernel(
        kernel.fn,
        static_shapes=False,
        autotune_config_overrides={"indexing": "tensor_descriptor"},
    )

    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    ex = mod.example_args(dev)

    bound = rekernel.bind(ex)
    print("autotuning (indexing=tensor_descriptor pinned)...", flush=True)
    best = bound.autotune(ex, force=True)
    print(f"best config: {best!r}", flush=True)

    code = bound.to_triton_code(best)
    n_desc = code.count("make_tensor_descriptor")
    print(f"make_tensor_descriptor occurrences: {n_desc}"
          + ("  DESCRIPTORS FIRED" if n_desc else "  (pointer fallback!)"), flush=True)

    out_path = KERNELS_DIR / args.name / "triton_emitted.py"
    out_path.write_text(f"# Autotuned config:\n# {best!r}\n\n{code}\n")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

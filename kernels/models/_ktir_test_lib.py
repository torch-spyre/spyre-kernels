# SPDX-License-Identifier: Apache-2.0
"""Shared KTIR validation logic for kernels/models/{op_name}/ test dirs.

Each op dir's thin test_ktir.py calls into here. See the three op-dir
"families" (simple / bundled_linear / per_stage) documented at the call
sites below -- this module implements one generic validator per family
rather than hand-written per-op logic.
"""

import importlib.util
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter

_FUNC_RE = re.compile(r"func\.func\s+@(\S+)\(")
_MEMVIEW_RE_TEMPLATE = r"ktdp\.construct_memory_view %{arg},\s*sizes:\s*\[([\d,\s]+)\]"


def _load_wrapper_module(op_dir: Path):
    """Import op_dir/wrapper.py despite the dotted/hyphenated op-dir name.

    wrapper.py does `from . import triton_kernel`, which needs a real
    package context to resolve -- so we synthesize one rooted at op_dir.
    Mirrors scripts/_spyre/round_trip.py's load_driver().
    """
    pkg_name = "_ktir_test_lib_pkg_" + re.sub(r"\W", "_", str(op_dir))
    if pkg_name in sys.modules:
        return sys.modules[pkg_name + ".wrapper"]

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(op_dir)]
    sys.modules[pkg_name] = pkg

    spec = importlib.util.spec_from_file_location(f"{pkg_name}.wrapper", op_dir / "wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_func_name(ktir_text: str) -> str:
    return _FUNC_RE.search(ktir_text).group(1)


def _memview_sizes_for_arg(ktir_text: str, argname: str) -> tuple:
    m = re.search(_MEMVIEW_RE_TEMPLATE.format(arg=argname), ktir_text)
    return tuple(int(x) for x in m.group(1).split(","))


def classify(op_dir: Path) -> str:
    if (op_dir / "ktir_kernel.ktir").exists():
        return "simple"
    n_ktir = len(list(op_dir.glob("*.ktir")))
    if n_ktir == 2:
        return "bundled_linear"
    if n_ktir == 5:
        return "per_stage"
    raise ValueError(f"{op_dir}: unrecognized KTIR layout ({n_ktir} .ktir files)")


# ---------------------------------------------------------------------------
# Known pre-existing, unrelated bugs/limits -- documented xfail reasons,
# following the DISABLED_REASON convention in
# torch.nn.functional.linear.1_spyre/wrapper.py.
# ---------------------------------------------------------------------------

_MUL_LAYOUT_DRIFT_REASON = (
    "Pre-existing KTIR-generation bug: the generated ktir_kernel.ktir's output "
    "ktdp.construct_memory_view declares the same head-outermost sizes/strides as "
    "the input descriptor, but the kernel's own triton_kernel.py declares a "
    "genuinely different (tile-outermost) output descriptor via "
    "tl.make_tensor_descriptor(...). This is a store-descriptor lowering/round-trip "
    "bug, not a test-harness issue -- confirmed by comparing wrapper.py's run() "
    "reshape/transpose order against both descriptors directly."
)

_SCRATCHPAD_OVERFLOW_REASON = (
    "Pre-existing ktir_cpu interpreter limit: this kernel's per-program tile is "
    "524288 elements (1 MiB in f16), which exceeds ktir_cpu's simulated LX "
    "scratchpad capacity (MemoryError: LX scratchpad overflow). A genuine "
    "interpreter resource limit, not a numerical bug."
)

_LINEAR_LAYOUT_DRIFT_REASON = (
    "Pre-existing KTIR-generation bug: stage 0 (repack) completes, but stage 1's "
    "(tl.dot) result is numerically unrelated to the NumPy oracle (>99% of "
    "elements mismatched). Confirmed this is not a shape/reinterpretation issue "
    "in the test harness (reshaping the scratch buffer to stage 1's declared "
    "construct_memory_view sizes changes the result bit-for-bit not at all) -- "
    "it's the same class of store/load descriptor-layout drift bug as the "
    "torch.mul.* cases, just between the two bundled stages instead of within one "
    "kernel."
)

XFAIL_REASONS = {
    **{f"torch.mul.{i}_spyre": _MUL_LAYOUT_DRIFT_REASON for i in (2, 4, 5, 6, 7, 12, 13, 14, 15)},
    "torch.zeros.1_spyre": _SCRATCHPAD_OVERFLOW_REASON,
    **{f"torch.nn.functional.linear.{i}_spyre": _SCRATCHPAD_OVERFLOW_REASON for i in (1, 6, 8)},
    **{f"torch.nn.functional.linear.{i}_spyre": _LINEAR_LAYOUT_DRIFT_REASON for i in (2, 7)},
}


# ---------------------------------------------------------------------------
# Family 1: "simple" -- single ktir_kernel.ktir.
# ---------------------------------------------------------------------------

def validate_simple_op(op_dir: Path) -> None:
    mod = _load_wrapper_module(op_dir)

    inputs = mod.make_inputs()
    expected = mod.run(inputs)

    ktir_text = (op_dir / "ktir_kernel.ktir").read_text()
    func_name = _extract_func_name(ktir_text)

    runtime_args = [k for k in mod.SIGNATURE if k not in mod.CONSTEXPR]
    kwargs = {}
    for i, name in enumerate(runtime_args):
        if name in inputs:
            kwargs[f"arg{i}"] = inputs[name]
        else:
            const = getattr(mod, name.upper())
            kwargs[f"arg{i}"] = np.int32(const) if mod.SIGNATURE[name] == "i32" else const

    interp = KTIRInterpreter()
    interp.load(ktir_text)
    outputs = interp.execute_function(func_name, **kwargs)
    result = outputs[f"arg{runtime_args.index(mod.OUTPUT_KEY)}"]

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32), rtol=1e-2, atol=1e-2,
    )


# ---------------------------------------------------------------------------
# Family 2: "bundled_linear" -- two-stage repack + tl.dot bundle
# (torch.nn.functional.linear.*_spyre).
# ---------------------------------------------------------------------------

def validate_bundled_linear_op(op_dir: Path) -> None:
    mod = _load_wrapper_module(op_dir)

    inputs = mod.make_inputs()
    expected = mod.run(inputs)

    ktir_files = sorted(op_dir.glob("*.ktir"), key=lambda p: int(re.search(r"_(\d+)$", p.stem).group(1)))
    stage0_path, stage1_path = ktir_files

    stage0_text = stage0_path.read_text()
    interp0 = KTIRInterpreter()
    interp0.load(stage0_text)
    out0 = interp0.execute_function(
        _extract_func_name(stage0_text),
        arg0=inputs["in_ptr0"],
        arg1=inputs["out_ptr0"].copy(),
        arg2=np.int32(0),
    )
    scratch = out0["arg1"]

    stage1_text = stage1_path.read_text()
    scratch = scratch.reshape(_memview_sizes_for_arg(stage1_text, "arg1"))
    interp1 = KTIRInterpreter()
    interp1.load(stage1_text)
    out1 = interp1.execute_function(
        _extract_func_name(stage1_text),
        arg0=inputs["in_ptr1"],
        arg1=scratch,
        arg2=inputs["out_ptr1"].copy(),
        arg3=np.int32(0),
        arg4=np.int32(0),
        arg5=np.int32(0),
    )
    result = out1["arg2"]

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32), rtol=1e-2, atol=1e-2,
    )


# ---------------------------------------------------------------------------
# Family 3: "per_stage" -- independent per-stage variants
# (torch.nn.functional.silu.*_spyre).
# ---------------------------------------------------------------------------

def _discover_per_stage_suffixes(mod) -> list:
    import inspect

    suffixes = []
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(fn)
        if "kernel_fn" not in sig.parameters:
            continue
        parts = name.split("_")
        for i in range(len(parts)):
            suffix = "_".join(parts[i:])
            if hasattr(mod, f"make_inputs_{suffix}") and hasattr(mod, f"run_{suffix}"):
                suffixes.append(suffix)
                break
    return suffixes


def validate_per_stage_op_suffixes(op_dir: Path) -> list:
    mod = _load_wrapper_module(op_dir)
    return _discover_per_stage_suffixes(mod)


def validate_per_stage_op_suffix(op_dir: Path, suffix: str) -> None:
    import inspect

    mod = _load_wrapper_module(op_dir)

    make_inputs_fn = getattr(mod, f"make_inputs_{suffix}")
    run_fn = getattr(mod, f"run_{suffix}")
    inputs = make_inputs_fn()
    expected = run_fn(inputs)

    kernel_fn = None
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        sig = inspect.signature(fn)
        if "kernel_fn" not in sig.parameters:
            continue
        parts = name.split("_")
        for i in range(len(parts)):
            if "_".join(parts[i:]) == suffix:
                kernel_fn = sig.parameters["kernel_fn"].default
                break
        if kernel_fn is not None:
            break
    kfn_name = getattr(kernel_fn, "__name__", None) or kernel_fn.fn.__name__

    ktir_text = (op_dir / f"{kfn_name}.ktir").read_text()
    func_name = _extract_func_name(ktir_text)

    CONSTEXPR = getattr(mod, "CONSTEXPR", [])
    sig_dicts = [
        (attr, val) for attr in dir(mod)
        if "SIGNATURE" in attr.upper() and isinstance((val := getattr(mod, attr)), dict)
    ]
    chosen = None
    for attr_name, SIG in sig_dicts:
        tensor_keys = [k for k in SIG if k not in CONSTEXPR and SIG[k].startswith("*")]
        if set(tensor_keys) == set(inputs.keys()):
            chosen = SIG
            break
    if chosen is None:
        raise AssertionError(f"no SIGNATURE dict on {mod} matches make_inputs_{suffix}() keys {list(inputs)}")

    runtime_args = [k for k in chosen if k not in CONSTEXPR]
    kwargs = {}
    for i, argname in enumerate(runtime_args):
        if argname in inputs:
            kwargs[f"arg{i}"] = inputs[argname]
        else:
            const = getattr(mod, argname.upper())
            kwargs[f"arg{i}"] = np.int32(const) if chosen[argname] == "i32" else const

    out_candidates = [k for k in runtime_args if "out" in k]
    if len(out_candidates) != 1:
        raise AssertionError(f"expected exactly one output-like arg, got {out_candidates}")
    output_key = out_candidates[0]

    interp = KTIRInterpreter()
    interp.load(ktir_text)
    outputs = interp.execute_function(func_name, **kwargs)
    result = outputs[f"arg{runtime_args.index(output_key)}"]

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32), rtol=1e-2, atol=1e-2,
    )

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rewrite a Helion-emitted Triton kernel so baked ``tl.constexpr`` literals
become function arguments.

Helion always materializes block sizes (and any other tuning knob) as
module-level ``_BLOCK_SIZE_N = tl.constexpr(V)`` literals in its emitted Triton
— ``static_shapes`` / ``register_block_size`` do not change this. This script
post-processes that output: every module-level ``X = tl.constexpr(V)`` is moved
into the ``@triton.jit`` kernel's signature as a ``X: tl.constexpr`` parameter
and threaded through the ``_launcher(...)`` call as a keyword argument, so the
block sizes are no longer hard-wired numerics in the source.

Pure ``ast`` transform; runs locally, no GPU. Reads
``kernels/<name>/triton_emitted.py`` (the stage-2 output of ``helion_emit.py``)
and writes ``kernels/<name>/triton_helion_roundtrip.py``.

Usage::

    python -m scripts.argify_constexpr <name>

Writes ``triton_helion_roundtrip.py`` and deletes the now-redundant
``triton_emitted.py`` — the argified kernel is the deliverable.

The wrapper function keeps its local ``_BLOCK_SIZE_N = V`` assignments (the
launch grid is computed from them); the same locals are forwarded as the new
kwargs, so kernel and grid never disagree on block size.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = ROOT / "kernels"

EMITTED = "triton_emitted.py"
ARGIFIED = "triton_helion_roundtrip.py"


def _is_constexpr_call(node: ast.expr) -> bool:
    """True for ``tl.constexpr(<constant>)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "constexpr"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
    )


def _constexpr_annotation() -> ast.expr:
    """The ``tl.constexpr`` annotation node for a new parameter."""
    return ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="constexpr",
                         ctx=ast.Load())


def _has_triton_jit(fn: ast.FunctionDef) -> bool:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name) \
                and dec.value.id == "triton" and dec.attr == "jit":
            return True
    return False


def argify(source: str) -> str:
    """Return the rewritten source. Raises ValueError if the expected
    structure (constexpr defs, a @triton.jit kernel, a _launcher call) is
    missing — a layout change should fail loudly, not silently mis-rewrite."""
    tree = ast.parse(source)

    # Pass A: collect module-level `X = tl.constexpr(V)`, mark for removal.
    # Keep the literal V too — block sizes the grid doesn't use have no
    # wrapper-local def emitted, so Pass D re-adds them from these.
    consts: list[str] = []
    const_vals: dict[str, ast.expr] = {}
    new_body: list[ast.stmt] = []
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and _is_constexpr_call(stmt.value)
        ):
            name = stmt.targets[0].id
            consts.append(name)
            const_vals[name] = stmt.value.args[0]
            continue  # drop the module-level def
        new_body.append(stmt)
    if not consts:
        raise ValueError("no module-level `X = tl.constexpr(V)` defs found")
    tree.body = new_body

    # Pass B: append consts to the @triton.jit kernel signature.
    jit_fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and _has_triton_jit(n)), None)
    if jit_fn is None:
        raise ValueError("no @triton.jit kernel found")
    for name in consts:
        jit_fn.args.args.append(ast.arg(arg=name, annotation=_constexpr_annotation()))

    # Pass C: thread consts as kwargs into the _launcher(...) call.
    launcher_call = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
         and n.func.id == "_launcher"),
        None,
    )
    if launcher_call is None:
        raise ValueError("no _launcher(...) call found")
    for name in consts:
        launcher_call.keywords.append(
            ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load()))
        )

    # Pass D: the kwargs in Pass C reference wrapper-local `name = V` defs.
    # Helion only emits those for block sizes the launch grid uses — a
    # reduction-axis block (e.g. K in matmul) has no grid term, so no local,
    # so the kwarg would hit an undefined name. Re-add the missing locals from
    # the captured literals at the top of the wrapper that holds the launcher.
    wrapper_fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef)
         and any(launcher_call is c for c in ast.walk(n))),
        None,
    )
    if wrapper_fn is None:
        raise ValueError("no wrapper function containing the _launcher call found")
    assigned = {
        t.id
        for s in wrapper_fn.body
        if isinstance(s, ast.Assign) and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Name)
        for t in [s.targets[0]]
    }
    missing = [n for n in consts if n not in assigned]
    for name in reversed(missing):  # reversed so source order is preserved
        wrapper_fn.body.insert(
            0,
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                       value=const_vals[name]),
        )

    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    ast.parse(out)  # self-check: rewritten source must parse
    return out


def _demo() -> None:
    """Self-check: a reduction-axis block size (no grid term, so no
    wrapper-local in the emitted source) must still get a local def, or the
    threaded kwarg hits an undefined name. See Pass D."""
    src = (
        "import triton\nimport triton.language as tl\n"
        "_BLK_M = tl.constexpr(64)\n"
        "_BLK_K = tl.constexpr(32)\n"  # reduction axis: not in grid
        "@triton.jit\n"
        "def _k(a, m):\n    pass\n"
        "def w(a):\n    m = a.size(0)\n    _BLK_M = 64\n"
        "    _launcher(_k, ((m + _BLK_M - 1) // _BLK_M,), a, m)\n"
    )
    out = argify(src)
    # _BLK_K must be defined as a wrapper local BEFORE the launcher forwards it
    # — otherwise the kwarg `_BLK_K=_BLK_K` is a NameError at call time.
    assert "_BLK_K = 32" in out, out
    assert out.index("_BLK_K = 32") < out.index("_launcher"), out
    # And it must reach the kernel signature + the launcher kwargs.
    assert "_BLK_K: tl.constexpr" in out, out
    assert "_BLK_K=_BLK_K" in out, out
    print("argify self-check OK")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?", help="kernel directory under kernels/")
    p.add_argument("--check", action="store_true", help="run self-check and exit")
    args = p.parse_args()

    if args.check:
        _demo()
        return 0
    if not args.name:
        p.error("name is required (or pass --check)")

    d = KERNELS_DIR / args.name
    src_path = d / EMITTED
    out_path = d / ARGIFIED
    if not src_path.exists():
        print(f"error: {src_path} not found (run helion_emit.py first)", file=sys.stderr)
        return 1

    out_path.write_text(argify(src_path.read_text()))
    src_path.unlink()  # emitted is redundant once argified exists
    print(f"wrote {out_path} (removed {src_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

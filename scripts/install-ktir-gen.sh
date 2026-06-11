#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Author tier: swap Triton in the current project venv to the spyre-enabled
# from-source build, so scripts/gen_ktir.py can lower Triton -> KTIR.
#
# Usage:
#     GIT_PAT=<github token> scripts/install-ktir-gen.sh
#
# What this does and why it's a script, not pyproject config:
#   torch depends on triton transitively, and uv resolves a single triton node
#   per venv. Any git source for triton in pyproject.toml (even one scoped to an
#   extra) pins that whole node to git, forcing the spyre source build on EVERY
#   `uv sync` (which needs GIT_PAT). To keep the default `uv sync` fast and on
#   stock PyPI triton, the spyre build is installed on-demand into the same venv
#   with `uv pip install --reinstall` instead.
#
# This is imperative: a later plain `uv sync` reverts triton to the PyPI build
# (base tier). Re-run this script to switch back to the spyre build.
#
# GIT_PAT is the GitHub token used for the LLVM fetch during the build.
# TRITON_BACKENDS selects which backends are compiled into libtriton. Defaults
# to nvidia,amd,spyre so a single venv can BOTH generate KTIR (spyre) and launch
# GPU reference kernels (nvidia); amd is required only because Gluon's
# gluon_ir.cc unconditionally #includes the AMD dialect header, so a GPU build
# without it fails to compile (amd needs no AMD hardware).
set -euo pipefail

# Pinned to match docs/spyre-triton-build.md and the round-trip vendoring.
TRITON_REF="${TRITON_REF:-5b467467c883c53ec7a8a89f9e89cfd55241034b}"
TRITON_URL="git+https://github.com/torch-spyre/triton@${TRITON_REF}"

# All three backends by default (see header). Override with TRITON_BACKENDS=...
TRITON_BACKENDS="${TRITON_BACKENDS:-nvidia,amd,spyre}"

if [[ -z "${GIT_PAT:-}" ]]; then
    echo "error: GIT_PAT is not set (needed for the LLVM fetch during the build)." >&2
    echo "       Export a GitHub token: GIT_PAT=<token> $0" >&2
    exit 1
fi

# Resolve the project venv the same way uv does.
VENV="${UV_PROJECT_ENVIRONMENT:-${VIRTUAL_ENV:-.venv}}"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "error: no project venv at ${VENV}. Run \`uv sync --extra test\` first." >&2
    exit 1
fi

echo ">> Installing spyre Triton (${TRITON_REF}, backends=${TRITON_BACKENDS}) into ${VENV} ..."
# --no-cache is required: TRITON_BACKENDS is a build-time env var that uv's wheel
# cache key does NOT include. Without it, uv reuses a previously-built wheel for
# the same git rev even when TRITON_BACKENDS changed, silently keeping the old
# backend set (e.g. a prior spyre-only build).
VIRTUAL_ENV="${VENV}" UV_PROJECT_ENVIRONMENT="${VENV}" \
TRITON_BACKENDS="${TRITON_BACKENDS}" \
    uv pip install --reinstall --no-cache "triton @ ${TRITON_URL}"

echo ">> Verifying the spyre backend (and any other requested backends) import ..."
TRITON_BACKENDS="${TRITON_BACKENDS}" "${VENV}/bin/python" - <<'PY'
import os
import triton
import triton.backends as b
import triton.backends.spyre.compiler  # noqa: F401
from triton._C.libtriton import spyre  # noqa: F401

want = set(os.environ.get("TRITON_BACKENDS", "spyre").split(","))
have = set(b.backends.keys())
missing = want - have
if missing:
    raise SystemExit(f"FAIL: requested backends {sorted(missing)} not compiled in "
                     f"(have {sorted(have)})")
print(f"OK: spyre-enabled triton {triton.__version__}, backends={sorted(have)}")
PY

echo ">> Done. Regenerate KTIR with: ${VENV}/bin/python scripts/gen_ktir.py"
echo "   (plain \`uv sync\` switches back to the stock PyPI triton base tier.)"

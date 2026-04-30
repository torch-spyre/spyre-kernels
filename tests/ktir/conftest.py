# SPDX-License-Identifier: Apache-2.0
import pytest

try:
    import ktir_cpu  # noqa: F401
except ImportError:
    pytest.skip(
        "ktir-cpu not installed (install with: uv sync --extra test)",
        allow_module_level=True,
    )

# SPDX-License-Identifier: Apache-2.0
"""Project-wide test setup.

We pin the driver to ``nvidia`` for tests so GPU reference kernels launch
on CUDA, while the Spyre backend remains available for compiling kernels
to KTIR (that path constructs ``SpyreBackend`` directly and does not go
through driver selection). This must run before any ``import triton``.
"""

import os

os.environ.setdefault("TRITON_DEFAULT_BACKEND", "nvidia")

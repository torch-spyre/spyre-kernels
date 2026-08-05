# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Patch builtins.float8_info before importing original.py.
# In vLLM, float8_info is defined at module level as:
#   float8_info = torch.finfo(current_platform.fp8_dtype())
import builtins

import torch

if not hasattr(builtins, "float8_info"):
    builtins.float8_info = torch.finfo(torch.float8_e4m3fn)

import kernels.vllm.merge_attn_states.original as _orig

_orig.float8_info = builtins.float8_info

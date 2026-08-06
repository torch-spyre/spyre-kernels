# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from kernels.models._ktir_test_lib import (
    XFAIL_REASONS,
    validate_per_stage_op_suffix,
    validate_per_stage_op_suffixes,
)

OP_DIR = Path(__file__).resolve().parent

pytestmark = [pytest.mark.ktir_cpu]
if (_reason := XFAIL_REASONS.get(OP_DIR.name)):
    pytestmark.append(pytest.mark.xfail(reason=_reason, strict=True))


@pytest.mark.parametrize("suffix", validate_per_stage_op_suffixes(OP_DIR))
def test_ktir(suffix):
    validate_per_stage_op_suffix(OP_DIR, suffix)

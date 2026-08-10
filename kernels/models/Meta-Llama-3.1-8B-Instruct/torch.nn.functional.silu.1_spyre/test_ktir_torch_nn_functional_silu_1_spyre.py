# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from kernels.models._ktir_test_lib import (
    validate_per_stage_op_suffix,
    validate_per_stage_op_suffixes,
)

OP_DIR = Path(__file__).resolve().parent

pytestmark = pytest.mark.ktir_cpu


@pytest.mark.parametrize("suffix", validate_per_stage_op_suffixes(OP_DIR))
def test_ktir(suffix):
    validate_per_stage_op_suffix(OP_DIR, suffix)

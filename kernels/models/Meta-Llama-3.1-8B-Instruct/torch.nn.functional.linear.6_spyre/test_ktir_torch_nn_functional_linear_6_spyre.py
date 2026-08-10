# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from kernels.models._ktir_test_lib import validate_bundled_linear_op

OP_DIR = Path(__file__).resolve().parent

pytestmark = pytest.mark.ktir_cpu


def test_ktir():
    validate_bundled_linear_op(OP_DIR)

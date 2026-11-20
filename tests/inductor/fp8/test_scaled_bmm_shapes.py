# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP8 scaled_bmm FVT at model-derived shapes and core counts (Sections 1-5).

Shared cases, helpers and _KNOWN_FAILURES are in scaled_bmm_common.py; the other
sections are in test_scaled_bmm_semantics.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from scaled_bmm_common import (  # noqa: E402
    SHAPE_PARAMS,
    _assert_within_bound,
    _expect_known_failure,
    _run_exact,
)


class TestScaledBmmShapes:
    """Sections 1-5: FR-003 (3D) and FR-004 (4D) at model-derived shapes."""

    @pytest.mark.parametrize("fvt_id,sencores", SHAPE_PARAMS)
    def test_shapes(self, fvt_id, sencores, request):
        with _expect_known_failure(request):
            _assert_within_bound(*_run_exact(fvt_id, sencores))

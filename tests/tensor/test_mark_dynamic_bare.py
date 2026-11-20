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

"""Tests that dynamic= calls mark_dynamic bare, with no min=/max=.

Deliberately eager-only -- no torch.compile here. These tests only need
to confirm what mark_dynamic was called with, which doesn't require
tracing at all, and calling it with an explicit min=/max= installs a
StrictMinMaxConstraint that would collide with the separate granularity
check applied downstream (raising ConstraintViolationError instead of an
ordinary guard miss).
"""

import torch
from torch._dynamo.decorators import _DimRange
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401

DEVICE = torch.device("spyre")


class TestMarkDynamicBare(TestCase):
    def test_dim_is_marked_dynamic(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertIn(0, x_dev._dynamo_dynamic_indices)

    def test_mark_dynamic_called_with_no_bound(self) -> None:
        # A bare call records _DimRange(0, None, None); passing min=/max=
        # to mark_dynamic would instead record _DimRange(0, 70, 630).
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertIn(_DimRange(0, None, None), x_dev._dynamo_dynamic_range)


if __name__ == "__main__":
    run_tests()

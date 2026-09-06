# Copyright 2025 The Torch-Spyre Authors.
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

"""Inductor test suite conftest — ensures OpSpec validation for all tests.

Validation is on by default, but we set the env var explicitly here so tests
remain covered even if the production default changes in the future.  Disable
via SPYRE_VALIDATE_OP_SPECS=0 if profiling test-suite runtime.
"""

import gc
import os

import pytest
import torch


def pytest_configure(config):
    """Ensure OpSpec validation is enabled for the inductor test suite."""
    os.environ["SPYRE_VALIDATE_OP_SPECS"] = "1"


_POISON_VALUE = 1234.0

# Several tensors of varied sizes (not one big blob) so the allocator's
# best-fit free-list ends up with poisoned blocks of many different sizes
# scattered across segments -- maximizing the chance that a subsequent
# test's real allocation reuses a poisoned block instead of a lucky
# never-touched (all-zero) one. Total is a few GB: small relative to the
# ~96 GiB of Tensor-usable device HBM, but comfortably larger than any
# individual test's working set in this file.
_POISON_SHAPES_BYTES = [
    1 * 1024**3,
    512 * 1024**2,
    512 * 1024**2,
    256 * 1024**2,
    256 * 1024**2,
    128 * 1024**2,
    128 * 1024**2,
    64 * 1024**2,
]


@pytest.fixture(scope="session", autouse=True)
def _poison_device_hbm():
    """Poison device HBM with non-zero sentinel values before any test runs.

    Defeats the "virgin device HBM reads back as zero" failure mode: a
    kernel bug that reads uninitialized HBM instead of its intended operand
    silently produces a zero-padded (and often coincidentally correct-
    looking) result on a freshly-initialized device, masking the bug until
    some other test happens to leave nonzero data behind first. Poisoning
    once here means every test -- including one run alone -- sees non-zero
    garbage instead, making such bugs fail deterministically. See issue
    #3613 and test_unsqueeze_broadcast_matmul_tile_E_poisoned_correct in
    test_coarse_tile_e2e.py for the specific bug class this guards against.
    """
    poison_tensors = [
        torch.full((nbytes // 2,), _POISON_VALUE, dtype=torch.float16, device="spyre")
        for nbytes in _POISON_SHAPES_BYTES
    ]
    del poison_tensors
    gc.collect()
    yield

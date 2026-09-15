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

"""Eager-mode sanity check for the nested for_each_tile fixture's reference."""

import unittest

import torch

from tests.inductor.for_each_tile_fixtures import (
    matmul_inputs,
    nested_split_m_then_k_fn,
    nested_split_m_then_k_reference,
)


class TestNestedForEachTileFixture(unittest.TestCase):
    def test_reference_matches_plain_matmul(self):
        (X, Y), expected = matmul_inputs()
        actual = nested_split_m_then_k_reference(X, Y)
        torch.testing.assert_close(actual, expected)

    def test_eager_fn_matches_plain_matmul(self):
        (X, Y), expected = matmul_inputs()
        actual = nested_split_m_then_k_fn(X, Y)
        torch.testing.assert_close(actual, expected)

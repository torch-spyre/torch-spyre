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

"""Unit tests for scatter layout enforcement helper functions.

Tests the core layout-checking logic in enforce_indirect_access_layout.py:
- _dim_order_is_compliant: checks if indirect dim is at device position 0
- _indirect_stride_idx: finds which coordinate carries IndirectAccess
- _build_required_stl: constructs compliant layout by rotating dimensions
"""

import unittest

import sympy
import torch

from torch_spyre._C import SpyreTensorLayout, get_device_dtype
from torch_spyre._inductor.enforce_indirect_access_layout import (
    _dim_order_is_compliant,
    _indirect_stride_idx,
    _build_required_stl,
)
from torch_spyre._inductor.op_spec import IndirectAccess


class TestDimOrderCompliance(unittest.TestCase):
    """Tests for _dim_order_is_compliant."""

    def test_indirect_at_position_0_is_compliant(self):
        """Indirect dim at device position 0 (outermost): compliant."""
        stl = SpyreTensorLayout(
            device_size=[8, 2, 64, 1],
            stride_map=[128, 64, 1, 1],
            device_dtype=get_device_dtype(torch.float16),
        )
        # stride_idx from right: 3 (rightmost coordinate)
        # device_pos = 4 - 1 - 3 = 0 ✓
        self.assertTrue(_dim_order_is_compliant(stl, stride_idx=3))

    def test_indirect_at_position_1_non_compliant(self):
        """Indirect dim at device position 1: non-compliant."""
        stl = SpyreTensorLayout(
            device_size=[2, 8, 64, 1],
            stride_map=[512, 64, 1, 1],
            device_dtype=get_device_dtype(torch.float16),
        )
        # stride_idx from right: 2
        # device_pos = 4 - 1 - 2 = 1 ✗
        self.assertFalse(_dim_order_is_compliant(stl, stride_idx=2))

    def test_indirect_at_position_2_non_compliant(self):
        """Indirect dim at device position 2: non-compliant."""
        stl = SpyreTensorLayout(
            device_size=[2, 4, 64, 1],
            stride_map=[256, 64, 1, 1],
            device_dtype=get_device_dtype(torch.float16),
        )
        # stride_idx from right: 1
        # device_pos = 4 - 1 - 1 = 2 ✗
        self.assertFalse(_dim_order_is_compliant(stl, stride_idx=1))


class TestIndirectStrideIdx(unittest.TestCase):
    """Tests for _indirect_stride_idx."""

    def test_finds_indirect_access_marker(self):
        """Finds coordinate carrying IndirectAccess marker."""
        idx_sym = sympy.Symbol("idx")
        coords = [
            IndirectAccess(idx_sym),
            sympy.S(0),
            sympy.S(0),
            sympy.S(1),
        ]
        access_subs = {}
        stride_idx = _indirect_stride_idx(coords, access_subs)
        self.assertEqual(stride_idx, 3)  # rightmost is index 0, so 3 from left

    def test_finds_indirect_after_substitution(self):
        """Finds IndirectAccess after applying substitutions."""
        idx_sym = sympy.Symbol("idx")
        coords = [
            idx_sym,
            sympy.S(0),
            sympy.S(0),
            sympy.S(1),
        ]
        access_subs = {idx_sym: IndirectAccess(sympy.Symbol("index_buffer"))}
        stride_idx = _indirect_stride_idx(coords, access_subs)
        self.assertEqual(stride_idx, 3)

    def test_returns_none_no_indirect(self):
        """Returns None when no IndirectAccess found."""
        coords = [
            sympy.S(0),
            sympy.S(1),
            sympy.S(2),
            sympy.S(3),
        ]
        access_subs = {}
        stride_idx = _indirect_stride_idx(coords, access_subs)
        self.assertIsNone(stride_idx)

    def test_finds_first_indirect_from_right(self):
        """Returns stride_idx (0-indexed from right) of first IndirectAccess."""
        idx_sym = sympy.Symbol("idx")
        coords = [
            IndirectAccess(idx_sym),
            IndirectAccess(sympy.Symbol("idx2")),
            sympy.S(0),
            sympy.S(1),
        ]
        access_subs = {}
        stride_idx = _indirect_stride_idx(coords, access_subs)
        # rightmost IndirectAccess is at index 1 in original list
        # which is index 2 in reversed list (4 - 1 - 1 = 2)
        self.assertEqual(stride_idx, 2)


class TestBuildRequiredStl(unittest.TestCase):
    """Tests for _build_required_stl."""

    def test_rotate_indirect_to_position_0(self):
        """Rotates indirect dim from position 2 to position 0."""
        original_stl = SpyreTensorLayout(
            device_size=[2, 4, 8, 1],
            stride_map=[256, 64, 1, 1],
            device_dtype=get_device_dtype(torch.float16),
        )
        required_stl = _build_required_stl(original_stl, indirect_device_pos=2)

        # Should move dim 2 (size 8) to position 0
        self.assertEqual(required_stl.device_size[0], 8)
        self.assertEqual(required_stl.stride_map[0], 1)
        # Stick (pos 3) should stay at end
        self.assertEqual(required_stl.device_size[3], 1)
        self.assertEqual(required_stl.stride_map[3], 1)

    def test_already_at_position_0_unchanged(self):
        """Returns same STL when indirect already at position 0."""
        original_stl = SpyreTensorLayout(
            device_size=[8, 2, 64, 1],
            stride_map=[128, 64, 1, 1],
            device_dtype=get_device_dtype(torch.float16),
        )
        required_stl = _build_required_stl(original_stl, indirect_device_pos=0)

        self.assertEqual(required_stl.device_size, original_stl.device_size)
        self.assertEqual(required_stl.stride_map, original_stl.stride_map)

    def test_rotate_indirect_from_position_1(self):
        """Rotates indirect dim from position 1 to position 0."""
        original_stl = SpyreTensorLayout(
            device_size=[2, 8, 64, 1],
            stride_map=[512, 64, 1, 1],
            device_dtype=get_device_dtype(torch.float16),
        )
        required_stl = _build_required_stl(original_stl, indirect_device_pos=1)

        # Dim 1 (size 8) moves to position 0
        self.assertEqual(required_stl.device_size[0], 8)
        # Dim 0 (size 2) should move to position 1
        self.assertEqual(required_stl.device_size[1], 2)
        # Stick stays at end
        self.assertEqual(required_stl.device_size[3], 1)


if __name__ == "__main__":
    unittest.main()

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

"""Unit tests for the QFP8WT K constraint helpers in nonstick_dim_order.py.

The QFP8WT 2D stick pairs K with N, and readers take K from device coordinate
-2. nonstick_dim_order finds K as the matmul's reduction variable and moves
that device dim next to the stick:
- _qfp8wt_k_rotation: decides the reorder (or rejects the layout)
- _permute_stl: reorders device dims, keeping the arrangement and padding
"""

import unittest

import sympy
import torch

from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.nonstick_dim_order import _permute_stl, _qfp8wt_k_rotation

b, h, k, n = sympy.symbols("b h k n", integer=True, nonnegative=True)


def _contiguous(size):
    stride, acc = [], 1
    for s in reversed(size):
        stride.insert(0, acc)
        acc *= s
    return stride


def _qfp8wt_identity_stl(size):
    """A QFP8WT layout in identity dim order, as on main before the fix."""
    return SpyreTensorLayout(
        size,
        _contiguous(size),
        torch.float8_e4m3fn,
        list(range(len(size))),
        ElementArrangement.QFP8WT,
    )


class TestQfp8wtKRotation(unittest.TestCase):
    def test_moves_k_next_to_the_stick(self):
        """[B, H, K, N] in identity order has B next to the stick; K moves there."""
        stl = _qfp8wt_identity_stl([2, 12, 128, 128])
        self.assertEqual(list(stl.device_size), [12, 128, 1, 2, 128])
        # device dims: [H, K, N-tile, B, stick]
        coords = [h, k, sympy.floor(n / 128), b, sympy.Mod(n, 128)]

        new = _qfp8wt_k_rotation(stl, coords, k, "w")

        self.assertIsNotNone(new)
        self.assertEqual(list(new.device_size), [12, 1, 2, 128, 128])
        self.assertEqual(list(new.stride_map), [16384, 128, 196608, 128, 1])
        self.assertEqual(new.element_arrangement, ElementArrangement.QFP8WT)
        # K (host stride 128) is now device dim -2.
        self.assertEqual(new.device_size[-2], 128)
        self.assertEqual(new.stride_map[-2], 128)

    def test_k_already_next_to_the_stick_is_left_alone(self):
        """Rank 2 [K, N]: K is already device dim -2."""
        stl = _qfp8wt_identity_stl([128, 256])
        coords = [sympy.floor(n / 128), k, sympy.Mod(n, 128)]
        self.assertIsNone(_qfp8wt_k_rotation(stl, coords, k, "w"))

    def test_reduction_over_the_stick_dim_is_rejected(self):
        """A matmul reducing over the stick dim cannot be fixed by a reorder."""
        stl = _qfp8wt_identity_stl([128, 256])
        # The reduction variable n is the stick dim.
        coords = [sympy.floor(n / 128), k, sympy.Mod(n, 128)]
        with self.assertRaisesRegex(Unsupported, "reduces over its stick dim"):
            _qfp8wt_k_rotation(stl, coords, n, "w")

    def test_k_in_no_device_dim_is_rejected(self):
        stl = _qfp8wt_identity_stl([128, 256])
        coords = [sympy.floor(n / 128), b, sympy.Mod(n, 128)]
        with self.assertRaisesRegex(Unsupported, "expected exactly one"):
            _qfp8wt_k_rotation(stl, coords, k, "w")


class TestPermuteStl(unittest.TestCase):
    def test_keeps_arrangement_and_remaps_padding(self):
        """N = 64 pads the stick; valid_elements must follow the reorder."""
        stl = _qfp8wt_identity_stl([4, 128, 64])
        self.assertEqual(list(stl.device_size), [128, 1, 4, 128])
        self.assertEqual(stl.valid_elements, {(1, 3): 64, (0,): 128, (2,): 4})

        new = _permute_stl(stl, [1, 2, 0, 3])

        self.assertEqual(list(new.device_size), [1, 4, 128, 128])
        self.assertEqual(list(new.stride_map), [64, 8192, 64, 1])
        self.assertEqual(new.element_arrangement, ElementArrangement.QFP8WT)
        self.assertEqual(new.valid_elements, {(0, 3): 64, (2,): 128, (1,): 4})


if __name__ == "__main__":
    unittest.main()

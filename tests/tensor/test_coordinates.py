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

import sympy

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_spyre._inductor.views import compute_coordinates

p0, p1, p2, p3, p4, p5 = sympy.symbols("p0 p1 p2 p3 p4 p5", integer=True)


class TestCoordinates(TestCase):
    def setUp(self):
        torch.manual_seed(0xAFFE)

    def test_compute_coordinates(self):
        # B, S, E -> B, E/H, S, H
        cx = compute_coordinates(
            [2, 256, 4096],
            [1048576, 4096, 1],
            {p0: 2, p1: 32, p2: 256, p3: 128},
            1048576 * p0 + 128 * p1 + 4096 * p2 + p3,
        )
        self.assertEqual(cx, [p0, p2, 128 * p1 + p3])

        # B, S, E -> B*S, E
        cx = compute_coordinates(
            [2, 256, 4096],
            [1048576, 4096, 1],
            {p0: 512, p1: 4096},
            4096 * p0 + p1,
        )
        self.assertEqual(cx, [p0 // 256, p0 % 256, p1])

        # dim of size 1 with stride>0
        cx = compute_coordinates(
            [3, 1, 128],
            [128, 128, 1],
            {p0: 3, p1: 128},
            128 * p0 + p1,
        )
        self.assertEqual(cx, [p0, 0, p1])

        # dim of size 1 with stride<0
        cx = compute_coordinates(
            [3, 1, 128],
            [128, -1, 1],
            {p0: 3, p1: 128},
            128 * p0 + p1,
        )
        self.assertEqual(cx, [p0, 0, p1])

        # dims of size 1
        cx = compute_coordinates(
            [4, 1, 1, 3, 1, 128],
            [384, 384, -1, 128, -1, 1],
            {p0: 4, p1: 1, p2: 1, p3: 3, p4: 1, p5: 128},
            384 * p0 + 128 * p3 + p5,
        )
        self.assertEqual(cx, [p0, 0, 0, p3, 0, p5])

        # dim with stride==0
        cx = compute_coordinates(
            [3, 42, 128],
            [128, 0, 1],
            {p0: 3, p1: 42, p2: 128},
            128 * p0 + p1,
        )
        self.assertEqual(cx, [p0, 0, p1])

    def test_compute_device_coordinates(self):
        # B, S, E -> B, E/H, S, H
        cx = compute_coordinates(
            [256, 64, 2, 64],
            [4096, 64, 1048576, 1],
            {p0: 2, p1: 32, p2: 256, p3: 128},
            1048576 * p0 + 128 * p1 + 4096 * p2 + p3,
        )
        self.assertEqual(cx, [p2, 2 * p1 + p3 // 64, p0, p3 % 64])

        # B, S, E -> B*S, E
        cx = compute_coordinates(
            [256, 64, 2, 64],
            [4096, 64, 1048576, 1],
            {p0: 512, p1: 4096},
            4096 * p0 + p1,
        )
        self.assertEqual(cx, [p0 % 256, p1 // 64, p0 // 256, p1 % 64])


if __name__ == "__main__":
    run_tests()

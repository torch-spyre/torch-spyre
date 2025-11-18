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

# Owner(s): ["module: cpp"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_spyre._C import SpyreTensorLayout


class TestSpyreTensorLayout(TestCase):
    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "spyre")

    # Test generic stick shorthands
    def test_generic_stick(self):
        stl = SpyreTensorLayout([128], torch.float16)
        self.assertEqual(stl.device_size, [2, 64])
        self.assertEqual(stl.device_strides, [64, 1])
        self.assertEqual(stl.dim_map, [0, 0])
        self.assertEqual(stl.format, SpyreTensorLayout.StickFormat.Dense)
        self.assertEqual(stl.num_stick_dims, 1)

        stl = SpyreTensorLayout([512, 256], torch.float16)
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.device_strides, [32768, 64, 1])
        self.assertEqual(stl.dim_map, [1, 0, 1])
        self.assertEqual(stl.format, SpyreTensorLayout.StickFormat.Dense)
        self.assertEqual(stl.num_stick_dims, 1)

        stl = SpyreTensorLayout([512, 8, 256], torch.float16)
        self.assertEqual(stl.device_size, [4, 512, 8, 64])
        self.assertEqual(stl.device_strides, [262144, 512, 64, 1])
        self.assertEqual(stl.dim_map, [2, 0, 1, 2])
        self.assertEqual(stl.format, SpyreTensorLayout.StickFormat.Dense)
        self.assertEqual(stl.num_stick_dims, 1)

    def test_explicit_stl_constructor(self):
        stl_x = SpyreTensorLayout([512, 256], torch.float16)
        stl_y = SpyreTensorLayout(
            [4, 512, 64],
            [32768, 64, 1],
            [1, 0, 1],
            1,
            SpyreTensorLayout.StickFormat.Dense,
        )
        self.assertEqual(stl_x.format, stl_y.format)
        self.assertEqual(stl_x.num_stick_dims, stl_y.num_stick_dims)
        self.assertEqual(stl_x.dim_map, stl_y.dim_map)
        self.assertEqual(stl_x.device_strides, stl_y.device_strides)
        self.assertEqual(stl_x.device_size, stl_y.device_size)

    def test_sparse_stl_constructor(self):
        stl = SpyreTensorLayout(
            [4, 512, 64],
            [32768, 64, 1],
            [1, 0, 1],
            1,
            SpyreTensorLayout.StickFormat.Sparse,
        )
        self.assertEqual(stl.format, SpyreTensorLayout.StickFormat.Sparse)

    def test_stl_str(self):
        stl = SpyreTensorLayout([512, 256], torch.float16)
        self.assertEqual(
            str(stl),
            'SpyreTensorLayout(device_size=[4, 512, 64], device_strides=[32768, 64, 1], dim_map =[1, 0, 1], num_stick_dims=1, format="Dense")',
        )


if __name__ == "__main__":
    run_tests()

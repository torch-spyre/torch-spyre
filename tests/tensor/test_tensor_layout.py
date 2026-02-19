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
from torch_spyre._C import (
    SpyreTensorLayout,
    to_with_layout,
    get_device_dtype,
    compute_view_layout,
)


class TestSpyreTensorLayout(TestCase):
    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "spyre")

    # Test generic stick shorthands
    def test_generic_stick(self):
        stl = SpyreTensorLayout([128], torch.float16)
        self.assertEqual(stl.device_size, [2, 64])
        self.assertEqual(stl.dim_map, [0, 0])
        self.assertEqual(stl.host_stick_dim(), 0)

        stl = SpyreTensorLayout([512, 256], torch.float16)
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.dim_map, [1, 0, 1])
        self.assertEqual(stl.host_stick_dim(), 1)

        stl = SpyreTensorLayout([512, 8, 256], torch.float16)
        self.assertEqual(stl.device_size, [8, 4, 512, 64])
        self.assertEqual(stl.dim_map, [1, 2, 0, 2])
        self.assertEqual(stl.host_stick_dim(), 2)

    def test_dim_order(self):
        stl = SpyreTensorLayout([512, 256], torch.float16, [1, 0])
        self.assertEqual(stl.device_size, [8, 256, 64])
        self.assertEqual(stl.dim_map, [0, 1, 0])
        self.assertEqual(stl.host_stick_dim(), 0)

        stl = SpyreTensorLayout([512, 8, 256], torch.float16, [2, 1, 0])
        self.assertEqual(stl.device_size, [8, 8, 256, 64])
        self.assertEqual(stl.dim_map, [1, 0, 2, 0])
        self.assertEqual(stl.host_stick_dim(), 0)

    def test_explicit_stl_constructor(self):
        stl_x = SpyreTensorLayout([512, 256], torch.float16)
        stl_y = SpyreTensorLayout(
            [4, 512, 64], [1, 0, 1], get_device_dtype(torch.float16)
        )
        self.assertEqual(stl_x.dim_map, stl_y.dim_map)
        self.assertEqual(stl_x.device_size, stl_y.device_size)

    def test_sparse_dim_order(self):
        stl = SpyreTensorLayout([512, 256], torch.float16, [0, 1, -1])
        self.assertEqual(stl.device_size, [256, 512, 64])
        self.assertEqual(stl.dim_map, [1, 0, -1])
        self.assertEqual(stl.host_stick_dim(), 1)

    def test_stl_str(self):
        stl = SpyreTensorLayout([512, 256], torch.float16)
        self.assertEqual(
            str(stl),
            "SpyreTensorLayout(device_size=[4, 512, 64], dim_map =[1, 0, 1], device_dtype=DataFormats.SEN169_FP16)",
        )

    def test_device_alloc(self):
        x = torch.rand([512, 256], dtype=torch.float16).to("spyre")
        stl = x.device_tensor_layout()
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.dim_map, [1, 0, 1])

    def test_equality(self):
        x = SpyreTensorLayout([512, 256], torch.float16)
        y = SpyreTensorLayout([512, 256], torch.float16, [0, 1])
        z = SpyreTensorLayout([512, 256], torch.float16, [1, 0])
        self.assertEqual(x, y)
        self.assertNotEqual(y, z)

    def test_to_spyre_layout(self):
        x = torch.rand([512, 256], dtype=torch.float16)
        x_stl = SpyreTensorLayout([512, 256], torch.float16)
        x_dev = to_with_layout(x, x_stl)
        self.assertEqual(x, x_dev.cpu())

        y = torch.rand([512, 512], dtype=torch.float16)
        y_stl = SpyreTensorLayout(
            [8, 512, 64], [1, 0, 1], get_device_dtype(torch.float16)
        )
        y_dev = to_with_layout(y, y_stl)
        self.assertEqual(y, y_dev.cpu())

        z = torch.rand([512, 8, 256], dtype=torch.float16)
        z_stl = SpyreTensorLayout([512, 8, 256], torch.float16, [2, 1, 0])
        z_dev = to_with_layout(z, z_stl)
        self.assertEqual(z_dev, z_dev.cpu())

        w = torch.rand([512, 256], dtype=torch.float16)
        w_stl = SpyreTensorLayout([512, 256], torch.float16, [0, 1, -1])
        w_dev = to_with_layout(w, w_stl)
        self.assertEqual(w, w_dev.cpu())

    def test_to_layout_patched(self):
        x = torch.rand([512, 256], dtype=torch.float16)
        x_stl = SpyreTensorLayout([512, 256], torch.float16)
        x_dev = x.to("spyre", device_layout=x_stl)
        stl = x_dev.device_tensor_layout()
        self.assertEqual(x_dev, x_dev.cpu())
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.dim_map, [1, 0, 1])

    def test_empty_layout_patched(self):
        x_stl = SpyreTensorLayout([512, 8, 256], torch.float16, [2, 1, 0])
        x = torch.empty((512, 8, 256), device_layout=x_stl, dtype=torch.float16)
        stl = x.device_tensor_layout()
        self.assertEqual(stl.device_size, [8, 8, 256, 64])
        self.assertEqual(stl.dim_map, [1, 0, 2, 0])

    def test_to_sparse_layout_patched(self):
        x = torch.rand([512, 256], dtype=torch.float16)
        x_stl = SpyreTensorLayout([512, 256], torch.float16, [0, 1, -1])
        x_dev = x.to("spyre", device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())
        self.assertEqual(x_stl.device_size, [256, 512, 64])
        self.assertEqual(x_stl.dim_map, [1, 0, -1])

    def test_compute_view_layout(self):
        stl = SpyreTensorLayout((3, 5, 128), torch.float16)
        unsqueeze_stl = compute_view_layout((3, 5, 128), (3, 5, 1, 128), stl)
        self.assertEqual(unsqueeze_stl.device_size, stl.device_size)
        self.assertEqual(unsqueeze_stl.dim_map, [1, 3, 0, 3])
        fused_stl = compute_view_layout((3, 5, 128), (15, 1, 128), stl)
        self.assertEqual(fused_stl.device_size, stl.device_size)
        self.assertEqual(fused_stl.dim_map, [0, 2, 0, 2])


if __name__ == "__main__":
    run_tests()

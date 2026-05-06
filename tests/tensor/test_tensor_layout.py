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

import copy
import pickle
import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.spyre import SpyreTensorLayout, get_device_dtype


@instantiate_parametrized_tests
class TestSpyreTensorLayout(TestCase):
    def setUp(self):
        torch.manual_seed(0xAFFE)

    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "spyre")

    def test_default_layout(self):
        stl = SpyreTensorLayout([], torch.float16)
        self.assertEqual(stl.device_size, [1, 64])
        self.assertEqual(stl.stride_map, [-1, -1])

        stl = SpyreTensorLayout([120], torch.float16)
        self.assertEqual(stl.device_size, [2, 64])
        self.assertEqual(stl.stride_map, [64, 1])

        stl = SpyreTensorLayout([128], torch.float16)
        self.assertEqual(stl.device_size, [2, 64])
        self.assertEqual(stl.stride_map, [64, 1])

        stl = SpyreTensorLayout([512, 240], torch.float16)
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.stride_map, [64, 240, 1])

        stl = SpyreTensorLayout([512, 256], torch.float16)
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.stride_map, [64, 256, 1])

        stl = SpyreTensorLayout([512, 8, 240], torch.float16)
        self.assertEqual(stl.device_size, [8, 4, 512, 64])
        self.assertEqual(stl.stride_map, [240, 64, 1920, 1])

        stl = SpyreTensorLayout([512, 8, 256], torch.float16)
        self.assertEqual(stl.device_size, [8, 4, 512, 64])
        self.assertEqual(stl.stride_map, [256, 64, 2048, 1])

    def test_dim_order(self):
        stl = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [1, 0])
        self.assertEqual(stl.device_size, [8, 256, 64])
        self.assertEqual(stl.stride_map, [16384, 1, 256])

        stl = SpyreTensorLayout([512, 8, 256], [2048, 256, 1], torch.float16, [0, 2, 1])
        self.assertEqual(stl.device_size, [256, 1, 512, 64])
        self.assertEqual(stl.stride_map, [1, 16384, 2048, 256])

        stl = SpyreTensorLayout([512, 8, 256], [2048, 256, 1], torch.float16, [1, 0, 2])
        self.assertEqual(stl.device_size, [512, 4, 8, 64])
        self.assertEqual(stl.stride_map, [2048, 64, 256, 1])

        stl = SpyreTensorLayout([512, 8, 256], [2048, 256, 1], torch.float16, [1, 2, 0])
        self.assertEqual(stl.device_size, [256, 8, 8, 64])
        self.assertEqual(stl.stride_map, [1, 131072, 256, 2048])

        stl = SpyreTensorLayout([512, 8, 256], [2048, 256, 1], torch.float16, [2, 0, 1])
        self.assertEqual(stl.device_size, [512, 1, 256, 64])
        self.assertEqual(stl.stride_map, [2048, 16384, 1, 256])

        stl = SpyreTensorLayout([512, 8, 256], [2048, 256, 1], torch.float16, [2, 1, 0])
        self.assertEqual(stl.device_size, [8, 8, 256, 64])
        self.assertEqual(stl.stride_map, [256, 131072, 1, 2048])

    def test_explicit_stl_constructor(self):
        stl_x = SpyreTensorLayout([512, 256], torch.float16)
        stl_y = SpyreTensorLayout(
            [4, 512, 64], [64, 256, 1], get_device_dtype(torch.float16)
        )
        self.assertEqual(stl_x.stride_map, stl_y.stride_map)
        self.assertEqual(stl_x.device_size, stl_y.device_size)

    def test_sparse_dim_order(self):
        stl = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [0, 1, -1])
        self.assertEqual(stl.device_size, [256, 1, 512, 64])
        self.assertEqual(stl.stride_map, [1, -1, 256, -1])

    def test_stl_str(self):
        stl = SpyreTensorLayout([512, 256], torch.float16)
        self.assertEqual(
            str(stl),
            "SpyreTensorLayout(device_size=[4, 512, 64], stride_map =[64, 256, 1], device_dtype=DataFormats.SEN169_FP16)",
        )

    def test_device_alloc(self):
        x = torch.rand([512, 256], dtype=torch.float16).to("spyre")
        stl = x.device_tensor_layout()
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.stride_map, [64, 256, 1])

    def test_equality(self):
        x = SpyreTensorLayout([512, 256], torch.float16)
        y = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [0, 1])
        z = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [1, 0])
        self.assertEqual(x, y)
        self.assertNotEqual(y, z)

    def test_stl_pickleable(self):
        stl = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [1, 0])
        self.assertEqual(stl, pickle.loads(pickle.dumps(stl)))

    def test_stl_copyable(self):
        stl = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [1, 0])
        self.assertEqual(stl, copy.deepcopy(stl))

    def test_to_spyre_layout(self):
        x = torch.rand([512, 256], dtype=torch.float16)
        x_stl = SpyreTensorLayout([512, 256], torch.float16)
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

        y = torch.rand([512, 512], dtype=torch.float16)
        y_stl = SpyreTensorLayout(
            [8, 512, 64], [64, 512, 1], get_device_dtype(torch.float16)
        )
        y_dev = y.to(device_layout=y_stl)
        self.assertEqual(y, y_dev.cpu())

        z = torch.rand([512, 8, 256], dtype=torch.float16)
        z_stl = SpyreTensorLayout(
            [512, 8, 256], [2048, 256, 1], torch.float16, [2, 1, 0]
        )
        z_dev = z.to(device_layout=z_stl)
        self.assertEqual(z_dev, z_dev.cpu())

        w = torch.rand([512, 256], dtype=torch.float16)
        w_stl = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [0, 1, -1])
        w_dev = w.to(device_layout=w_stl)
        self.assertEqual(w, w_dev.cpu())

        w = torch.rand([512, 256, 1], dtype=torch.float16)
        w_stl = SpyreTensorLayout([512, 256, 1], [256, 1, 1], torch.float16, [0, 1, 2])
        w_dev = w.to(device_layout=w_stl)
        self.assertEqual(w, w_dev.cpu())

        w = torch.rand([512, 256], dtype=torch.float16)
        w_stl = SpyreTensorLayout([131072], [1], torch.float16, [0])
        w_dev = w.to(device_layout=w_stl)
        self.assertEqual(w, w_dev.cpu())

        w = torch.rand([512, 256], dtype=torch.float16)
        w_slice = w[256:, :]
        w_stl = SpyreTensorLayout([256, 256], [256, 1], torch.float16, [0, 1])
        w_dev = w_slice.to(device_layout=w_stl)
        self.assertEqual(w_slice, w_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([60], [1], [1, 64], [64, 1]),
            ([64], [1], [1, 64], [64, 1]),
            ([120], [1], [2, 64], [64, 1]),
            ([128], [1], [2, 64], [64, 1]),
            ([240], [1], [4, 64], [64, 1]),
            ([256], [1], [4, 64], [64, 1]),
            ([40, 60], [60, 1], [1, 60, 64], [3840, 1, 60]),
            ([40, 60], [60, 1], [1, 40, 64], [64, 60, 1]),
            ([40, 64], [64, 1], [1, 64, 64], [4096, 1, 64]),
            ([40, 64], [64, 1], [1, 40, 64], [64, 64, 1]),
            ([40, 120], [120, 1], [1, 120, 64], [7680, 1, 120]),
            ([40, 120], [120, 1], [2, 40, 64], [64, 120, 1]),
            ([40, 128], [128, 1], [1, 128, 64], [8192, 1, 128]),
            ([40, 128], [128, 1], [2, 40, 64], [64, 128, 1]),
            ([40, 240], [240, 1], [1, 240, 64], [15360, 1, 240]),
            ([40, 240], [240, 1], [4, 40, 64], [64, 240, 1]),
            ([40, 256], [256, 1], [1, 256, 64], [16384, 1, 256]),
            ([40, 256], [256, 1], [4, 40, 64], [64, 256, 1]),
        ],
    )
    def test_to_spyre_layout_explicit(self, sizes, strides, device_size, stride_map):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([40, 60], [60, 1], [1, 512, 64], [3840, 1, 60]),
            ([40, 60], [60, 1], [1, 512, 64], [64, 60, 1]),
            ([40, 64], [64, 1], [1, 512, 64], [4096, 1, 64]),
            ([40, 64], [64, 1], [1, 512, 64], [64, 64, 1]),
            ([40, 120], [120, 1], [1, 512, 64], [7680, 1, 120]),
            ([40, 120], [120, 1], [2, 512, 64], [64, 120, 1]),
            ([40, 128], [128, 1], [1, 512, 64], [8192, 1, 128]),
            ([40, 128], [128, 1], [2, 512, 64], [64, 128, 1]),
            ([40, 240], [240, 1], [1, 512, 64], [15360, 1, 240]),
            ([40, 240], [240, 1], [4, 512, 64], [64, 240, 1]),
            ([40, 256], [256, 1], [1, 512, 64], [16384, 1, 256]),
            ([40, 256], [256, 1], [4, 512, 64], [64, 256, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_padding(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([60], [1], [1, 512, 64], [64, 0, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_expanding(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([40, 60], [60, 1], [38, 64], [64, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_folding(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([60], [1], [1, 1, 64], [64, 60, 1]),
            ([60], [1], [1, 1, 1, 64], [60, 64, 60, 1]),
            ([60], [1], [1, 1, 1, 1, 64], [60, 60, 64, 60, 1]),
            ([60], [1], [1, 1, 1, 1, 1, 64], [60, 60, 60, 64, 60, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_leading_ones(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([60], [1], [1, 60, 64], [64, 1, 1]),
            ([60], [1], [1, 1, 60, 64], [1, 64, 1, 1]),
            ([60], [1], [1, 1, 1, 60, 64], [1, 1, 64, 1, 1]),
            ([60], [1], [1, 1, 1, 1, 60, 64], [1, 1, 1, 64, 1, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_trailing_ones(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([60], [1], [1, 2, 64], [64, 30, 1]),
            ([60], [1], [2, 1, 2, 64], [15, 64, 30, 1]),
            ([60], [1], [2, 1, 1, 2, 64], [15, 15, 64, 30, 1]),
            ([60], [1], [1, 2, 1, 2, 64], [30, 15, 64, 30, 1]),
            ([60], [1], [2, 2, 1, 1, 64], [30, 15, 64, 60, 1]),
            ([60], [1], [2, 1, 1, 1, 2, 64], [15, 15, 15, 64, 30, 1]),
            ([60], [1], [1, 2, 1, 1, 2, 64], [30, 15, 15, 64, 30, 1]),
            ([60], [1], [1, 1, 2, 1, 2, 64], [30, 30, 15, 64, 30, 1]),
            ([60], [1], [2, 2, 1, 1, 1, 64], [30, 15, 15, 64, 60, 1]),
            ([60], [1], [2, 1, 2, 1, 1, 64], [30, 30, 15, 64, 60, 1]),
            ([60], [1], [1, 2, 2, 1, 1, 64], [60, 30, 15, 64, 60, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_viewing(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([4800], [1], [3, 2, 14, 64], [120, 64, 360, 1]),
            ([40, 60], [60, 1], [24, 1, 2, 64], [60, 64, 1440, 1]),
            ([40, 60], [60, 1], [2, 12, 1, 2, 64], [720, 60, 64, 1200, 1]),
            ([40, 60], [60, 1], [2, 6, 2, 1, 2, 64], [720, 120, 60, 64, 1200, 1]),
            ([40, 60], [60, 1], [3, 4, 3, 1, 2, 64], [600, 180, 60, 64, 1800, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_tiling(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            # Permuted [60, 40] to [40, 60]
            ([40, 60], [1, 40], [1, 40, 64], [2560, 1, 40]),
            ([40, 60], [1, 40], [1, 60, 64], [64, 40, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_permuted(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([120], [1], [1, 64], [64, 1]),
            ([128], [1], [1, 64], [64, 1]),
            ([240], [1], [2, 64], [64, 1]),
            ([256], [1], [2, 64], [64, 1]),
            ([480], [1], [4, 64], [64, 1]),
            ([512], [1], [4, 64], [64, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_sliced_batch(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_sliced = x[sizes[0] // 2 :]
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x_sliced.to(device_layout=x_stl)
        self.assertEqual(x_sliced, x_dev.cpu())

    @parametrize(
        "sizes,strides",
        [
            ([40, 120], [120, 1]),
            ([40, 120], [120, 1]),
        ],
    )
    def test_to_spyre_sliced_other(self, sizes, strides):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_sliced = x[:, sizes[1] // 2 :]
        x_dev = x_sliced.to("spyre")
        # Sliced tensors (that are not sliced along the batch dimension) are
        # non-dense but produce dense tensors when transferred across devices.
        # This requires an update the the stride_map after the device transfer
        # for all non-dense dimensions.
        # Once this is implemented, this test should pass.
        self.assertEqual(x_sliced.contiguous(), x_dev.cpu())

    @unittest.skip(
        "Skip until device transfers are updated to account for overlapping tensors"
    )
    @parametrize(
        "sizes,strides,device_size,stride_map",
        [
            ([1, 60], [60, 1], [1, 512, 64], [64, 0, 1]),
        ],
    )
    def test_to_spyre_layout_explicit_expanded(
        self, sizes, strides, device_size, stride_map
    ):
        x = torch.empty_strided(sizes, strides, dtype=torch.float16).uniform_(0, 1)
        x_stl = SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(torch.float16)
        )
        x_dev = x.to(device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())

        expanded_sizes = sizes
        expanded_sizes[0] = 512
        x_expanded = x.expand(expanded_sizes)
        x_dev = x_expanded.to(device_layout=x_stl)
        # Expanded tensors are overlapping but produce non-overlapping tensors
        # when transferred across devices.
        # This requires an update the the stride_map after the device transfer
        # for all overlapping dimensions.
        # Once this is implemented, this test should pass.
        self.assertEqual(x_expanded.contiguous(), x_dev.cpu())

    def test_to_layout_patched(self):
        x = torch.rand([512, 256], dtype=torch.float16)
        x_stl = SpyreTensorLayout([512, 256], torch.float16)
        x_dev = x.to("spyre", device_layout=x_stl)
        stl = x_dev.device_tensor_layout()
        self.assertEqual(x_dev, x_dev.cpu())
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.stride_map, [64, 256, 1])

    def test_empty_layout_patched(self):
        x_stl = SpyreTensorLayout(
            [512, 8, 256], [2048, 256, 1], torch.float16, [2, 1, 0]
        )
        x = torch.empty((512, 8, 256), device_layout=x_stl, dtype=torch.float16)
        stl = x.device_tensor_layout()
        self.assertEqual(stl.device_size, [8, 8, 256, 64])
        self.assertEqual(stl.stride_map, [256, 131072, 1, 2048])

    def test_to_sparse_layout_patched(self):
        x = torch.rand([512, 256], dtype=torch.float16)
        x_stl = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [0, 1, -1])
        x_dev = x.to("spyre", device_layout=x_stl)
        self.assertEqual(x, x_dev.cpu())
        self.assertEqual(x_stl.device_size, [256, 1, 512, 64])
        self.assertEqual(x_stl.stride_map, [1, -1, 256, -1])

    def test_add_with_mixed_layout_dim_orders(self):
        """Compiled add where x and y have different device layouts."""
        x = torch.rand(3, 2, 2048, dtype=torch.float16)
        y = torch.rand(3, 2, 2048, dtype=torch.float16)
        cpu_result = x + y  # linter won't allow lambdas
        x_stl = SpyreTensorLayout(x.size(), x.stride(), torch.float16, [1, 0, 2])
        y_stl = SpyreTensorLayout(x.size(), x.stride(), torch.float16, [0, 1, 2])
        _ = x.to("spyre")  # required for lazy device initialization
        x_dev = x.to(device_layout=x_stl)
        y_dev = y.to(device_layout=y_stl)
        compiled = torch.compile(torch.add)
        compiled_result = compiled(x_dev, y_dev).cpu()
        torch.testing.assert_close(
            cpu_result, compiled_result, rtol=0.001, atol=0.00001
        )

    def test_spyre_tensor_layout_guard(self):
        """
        Verify that torch.compile recompiles when SpyreTensorLayout changes
        between calls. Two tensors with same shape but different layout must
        produce separate compiled graphs — regression test for issue #1297.
        """
        x = torch.rand([512, 256], dtype=torch.float16)
        stl_default = SpyreTensorLayout([512, 256], torch.float16)
        stl_custom = SpyreTensorLayout(x.size(), x.stride(), torch.float16, [1, 0])
        _ = x.to("spyre")  # required for lazy device initialization

        tensor_default = x.to(device_layout=stl_default)
        tensor_custom = x.to(device_layout=stl_custom)

        def simple_add(a):
            return a + a

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(simple_add)

        # call 1 — default layout → compiles new graph
        compiled(tensor_default)
        count_after_default = torch._dynamo.utils.counters["stats"]["calls_captured"]

        # call 2 — different layout → guard fails → recompile expected
        compiled(tensor_custom)
        count_after_custom = torch._dynamo.utils.counters["stats"]["calls_captured"]

        self.assertEqual(
            count_after_custom,
            count_after_default + 1,
            "Expected recompilation when SpyreTensorLayout changes between calls",
        )

        # call 3 — same custom layout again → cache hit → no recompile expected
        compiled(tensor_custom)
        count_after_custom_second = torch._dynamo.utils.counters["stats"][
            "calls_captured"
        ]
        self.assertEqual(
            count_after_custom_second,
            count_after_custom,
            "Expected cache hit when SpyreTensorLayout is the same as previous call",
        )


if __name__ == "__main__":
    run_tests()

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
from torch_spyre._C import DataFormats, ElementArrangement, get_device_size_in_bytes


@instantiate_parametrized_tests
class TestSpyreTensorLayout(TestCase):
    def setUp(self):
        torch.manual_seed(0xAFFE)

    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "spyre")

    @parametrize(
        "df,eps,bits",
        [
            (DataFormats.SEN169_FP16, 64, 16),
            (DataFormats.IEEE_FP32, 32, 32),
            (DataFormats.SEN143_FP8, 128, 8),
            (DataFormats.SENINT4, 256, 4),
            (DataFormats.SENINT2, 512, 2),
            (DataFormats.IEEE_INT32, 32, 32),
            (DataFormats.IEEE_INT64, 16, 64),
            (DataFormats.BOOL, 128, 8),
            (DataFormats.BFLOAT16, 64, 16),
        ],
    )
    def test_device_storage_size(self, df, eps, bits):
        self.assertEqual(df.elems_per_stick(), eps)
        for ea in ElementArrangement.__members__.values():
            # Both normal and shortened/sparse trailing extents occupy full
            # sticks. Empty outer geometry remains empty after FP8 rescaling.
            for outer, inner in [(3, eps), (3, 1), (0, eps)]:
                stl = SpyreTensorLayout([outer, inner], [eps, 1], df, ea)
                expected = outer * eps * bits // 8
                self.assertEqual(get_device_size_in_bytes(stl), expected)
                self.assertEqual(
                    get_device_size_in_bytes(stl.device_size, df), expected
                )

    @parametrize(
        "df",
        [
            DataFormats.INVALID,
            DataFormats.SEN153_FP9,
            DataFormats.SENINT24,
            DataFormats.SEN18F_FP24,
        ],
    )
    def test_unmapped_compute_format_has_no_storage_size(self, df):
        with self.assertRaisesRegex(RuntimeError, "No device stick geometry"):
            get_device_size_in_bytes([1, 64], df)

    @parametrize(
        "df,dtype",
        [(DataFormats.BFLOAT16, torch.float32), (DataFormats.IEEE_INT64, torch.int32)],
    )
    def test_explicit_transfer_storage_roundtrip(self, df, dtype):
        # Explicit transfer encodings differ from default compute storage.
        x = torch.arange(64, dtype=dtype)
        eps = df.elems_per_stick()
        stl = SpyreTensorLayout([64 // eps, eps], [eps, 1], df)
        y = x.to("spyre", device_layout=stl)
        self.assertEqual(y.device_tensor_layout().device_dtype, df)
        self.assertEqual(y.cpu(), x)

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

    def test_equality_and_hashable(self):
        x = SpyreTensorLayout([512, 256], torch.float16)
        y = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [0, 1])
        z = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [1, 0])
        z2 = SpyreTensorLayout([512, 256], torch.float32)

        self.assertEqual(hash(x), hash(x))

        self.assertEqual(x, y)
        self.assertEqual(hash(x), hash(y))

        self.assertNotEqual(y, z)
        self.assertNotEqual(hash(y), hash(z))

        self.assertNotEqual(x, z2)
        self.assertNotEqual(hash(x), hash(z2))

        # usable as dict key
        d = {x: "value"}
        self.assertEqual(d[y], "value")

        # usable in a set
        s = {x, y, z}
        self.assertEqual(len(s), 2)

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
        # Device fp16 storage rounds to ~1/2048 (~4.9e-4) granularity; near-zero
        # values can drift past assertEqual's default fp16 atol=1e-5. Empirically
        # the observed max round-trip delta on these torch.rand tensors is
        # ~4.9e-4, so atol/rtol=1e-3 sits just above the device's real precision
        # without being loose enough to mask a genuine regression.
        self.assertEqual(x, x_dev.cpu(), atol=1e-3, rtol=1e-3)

        y = torch.rand([512, 512], dtype=torch.float16)
        y_stl = SpyreTensorLayout(
            [8, 512, 64], [64, 512, 1], get_device_dtype(torch.float16)
        )
        y_dev = y.to(device_layout=y_stl)
        self.assertEqual(y, y_dev.cpu(), atol=1e-3, rtol=1e-3)

        z = torch.rand([512, 8, 256], dtype=torch.float16)
        z_stl = SpyreTensorLayout(
            [512, 8, 256], [2048, 256, 1], torch.float16, [2, 1, 0]
        )
        z_dev = z.to(device_layout=z_stl)
        self.assertEqual(z_dev, z_dev.cpu(), atol=1e-3, rtol=1e-3)

        w = torch.rand([512, 256], dtype=torch.float16)
        w_stl = SpyreTensorLayout([512, 256], [256, 1], torch.float16, [0, 1, -1])
        w_dev = w.to(device_layout=w_stl)
        self.assertEqual(w, w_dev.cpu(), atol=1e-3, rtol=1e-3)

        w = torch.rand([512, 256, 1], dtype=torch.float16)
        w_stl = SpyreTensorLayout([512, 256, 1], [256, 1, 1], torch.float16, [0, 1, 2])
        w_dev = w.to(device_layout=w_stl)
        self.assertEqual(w, w_dev.cpu(), atol=1e-3, rtol=1e-3)

        w = torch.rand([512, 256], dtype=torch.float16)
        w_stl = SpyreTensorLayout([131072], [1], torch.float16, [0])
        w_dev = w.to(device_layout=w_stl)
        self.assertEqual(w, w_dev.cpu(), atol=1e-3, rtol=1e-3)

        w = torch.rand([512, 256], dtype=torch.float16)
        w_slice = w[256:, :]
        w_stl = SpyreTensorLayout([256, 256], [256, 1], torch.float16, [0, 1])
        w_dev = w_slice.to(device_layout=w_stl)
        self.assertEqual(w_slice, w_dev.cpu(), atol=1e-3, rtol=1e-3)

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

    def test_to_same_layout_honors_a_different_explicit_device(self):
        """A matching layout must not bypass an explicit cross-device copy."""
        if torch.spyre.device_count() < 2:
            self.skipTest("requires at least two Spyre devices")

        previous = torch.spyre.current_device()
        target = torch.device("spyre", (previous + 1) % torch.spyre.device_count())
        x = torch.rand([128, 256], dtype=torch.float16)
        layout = SpyreTensorLayout(list(x.shape), x.dtype)
        source = x.to(torch.device("spyre", previous), device_layout=layout)

        result = source.to(target, device_layout=layout)

        self.assertIsNot(result, source)
        self.assertEqual(source.device, torch.device("spyre", previous))
        self.assertEqual(result.device, target)
        self.assertEqual(result.device_tensor_layout(), layout)
        round_trip = result.to(source.device, device_layout=layout, copy=True)
        torch.testing.assert_close(round_trip.cpu(), x, rtol=2e-3, atol=1e-4)
        self.assertEqual(torch.spyre.current_device(), previous)

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
        # Device fp16 storage rounds to ~1/2048 (~4.9e-4) granularity; near-zero
        # values can drift past assertEqual's default fp16 atol=1e-5. Empirically
        # the observed max round-trip delta on these torch.rand tensors is
        # ~4.9e-4, so atol/rtol=1e-3 sits just above the device's real precision
        # without being loose enough to mask a genuine regression.
        self.assertEqual(x, x_dev.cpu(), atol=1e-3, rtol=1e-3)
        self.assertEqual(x_stl.device_size, [256, 1, 512, 64])
        self.assertEqual(x_stl.stride_map, [1, -1, 256, -1])

    def test_add_with_mixed_layout_dim_orders(self):
        """Compiled add where x and y have different device layouts."""
        x = torch.randn(3, 2, 2048, dtype=torch.float16)
        y = torch.randn(3, 2, 2048, dtype=torch.float16)
        cpu_result = x + y  # linter won't allow lambdas
        x_stl = SpyreTensorLayout(x.size(), x.stride(), torch.float16, [1, 0, 2])
        y_stl = SpyreTensorLayout(x.size(), x.stride(), torch.float16, [0, 1, 2])
        _ = x.to("spyre")  # required for lazy device initialization
        x_dev = x.to(device_layout=x_stl)
        y_dev = y.to(device_layout=y_stl)
        compiled = torch.compile(torch.add)
        compiled_result = compiled(x_dev, y_dev).cpu()
        torch.testing.assert_close(cpu_result, compiled_result, rtol=0.005, atol=0.005)

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

    def test_flattened_attention_view_feeds_linear_across_compiles(self):
        """A BLHD-backed BHLD result must be safe for a later projection.

        Attention kernels naturally produce ``[B,H,L,D]`` with token-major
        backing storage. A separately compiled transpose+reshape therefore
        returns a logically contiguous ``[B,L,H*D]`` view whose device layout
        still has H and D factorized. The next compiled linear must canonicalize
        that input instead of presenting two contraction dimensions to the
        backend.
        """
        B, L, H, D = 1, 8, 32, 128
        hidden = H * D
        torch.manual_seed(0xAFFE)
        x = torch.randn(B, L, hidden, dtype=torch.float16)
        weight = torch.randn(hidden, hidden, dtype=torch.float16) / hidden**0.5
        residual = torch.randn(B, L, hidden, dtype=torch.float16)
        expected = torch.nn.functional.linear(x, weight) + residual

        def project(x, weight, residual):
            return torch.nn.functional.linear(x, weight) + residual

        factorized_layout = SpyreTensorLayout(
            [L, D // 64, H, 64],
            [hidden, 64, D, 1],
            get_device_dtype(torch.float16),
        )
        flattened = x.to(device_layout=factorized_layout)
        self.assertEqual(flattened.device_tensor_layout(), factorized_layout)
        actual = torch.compile(project, dynamic=False)(
            flattened, weight.to("spyre"), residual.to("spyre")
        ).cpu()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    @parametrize("H,N_KV,LQ", [(4, 2, 8), (32, 8, 1)])
    def test_sdpa_output_feeds_linear_in_same_graph(self, H, N_KV, LQ):
        """A fused SDPA -> BL(H*D) view -> linear gets one contraction dim."""
        B, LK, D = 1, 64, 128
        hidden = H * D
        q = torch.randn(B, H, LQ, D, dtype=torch.float16)
        k = torch.randn(B, N_KV, LK, D, dtype=torch.float16)
        v = torch.randn(B, N_KV, LK, D, dtype=torch.float16)
        weight = torch.randn(hidden, hidden, dtype=torch.float16) / hidden**0.5
        query_positions = torch.arange(LK - LQ, LK).view(1, 1, LQ, 1)
        key_positions = torch.arange(LK).view(1, 1, 1, LK)
        mask = torch.where(
            key_positions <= query_positions,
            torch.tensor(0.0, dtype=torch.float16),
            torch.tensor(torch.finfo(torch.float16).min / 2, dtype=torch.float16),
        )

        def attention_project(q, k, v, mask, weight):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=D**-0.5,
                enable_gqa=True,
            )
            out = out.transpose(1, 2).reshape(B, LQ, hidden)
            return torch.nn.functional.linear(out, weight)

        expected = attention_project(q, k, v, mask, weight)
        actual = torch.compile(attention_project, dynamic=False)(
            q.to("spyre"),
            k.to("spyre"),
            v.to("spyre"),
            mask.to("spyre"),
            weight.to("spyre"),
        ).cpu()
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)

    def test_rescale_for_dtype_rejects_inexact_stick_rescale(self):
        """A stick-indexing dim that does not hold a whole number of output
        sticks must raise, not floor.

        Widening the stick depth shrinks the num-sticks dim by the depth ratio.
        Flooring an inexact ratio drops data, and a single input stick floors to
        zero; such a layout describes no tensor, and it used to reach
        ``get_device_stride_infos``, which divided by it and killed the process
        with SIGFPE rather than raising (issue #3604). Needs no device.
        """
        from torch_spyre._C import ElementArrangement
        from torch_spyre._inductor.errors import Unsupported
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        fp32 = get_device_dtype(torch.float32)
        # One fp32 stick (32 elements): 1 * 32 // 64 == 0 going to fp16.
        one_stick = SpyreTensorLayout(
            [1, 4, 32], [32, 32, 1], fp32, ElementArrangement.STANDARD
        )
        with self.assertRaisesRegex(Unsupported, "not a whole number of 64-element"):
            rescale_stl_for_dtype(one_stick, torch.float16, ElementArrangement.STANDARD)
        # Three fp32 sticks (96 elements): flooring to one fp16 stick would
        # silently drop 32 elements.
        three_sticks = SpyreTensorLayout(
            [3, 4, 32], [32, 32, 1], fp32, ElementArrangement.STANDARD
        )
        with self.assertRaisesRegex(Unsupported, "3 stick\\(s\\) of 32 elements"):
            rescale_stl_for_dtype(
                three_sticks, torch.float16, ElementArrangement.STANDARD
            )
        # An exact ratio rescales as before.
        two_sticks = SpyreTensorLayout(
            [2, 4, 32], [32, 32, 1], fp32, ElementArrangement.STANDARD
        )
        rescaled = rescale_stl_for_dtype(
            two_sticks, torch.float16, ElementArrangement.STANDARD
        )
        self.assertEqual(list(rescaled.device_size), [1, 4, 64])
        self.assertEqual(list(rescaled.stride_map), [64, 32, 1])

    def test_qfp8ch_layout_rounds_a_partial_stick_up(self):
        """qfp8ch's fp16 -> fp8 output may end in a partially filled fp8 stick:
        one fp16 stick becomes one (half-filled) 128-element fp8 stick, never a
        size-0 dim. The fp8 -> fp16 conversion that consumes this output
        rebuilds a dense layout from the host size, so the partial stick is the
        padded case it already handles.

        Same guarantee as #3809's ``_qfp8ch_stl``, asserted on the same values
        through the shared helper that replaces it: for this direction alone,
        ceiling on the capacity and on the extent are the same function, since
        ``ceil(ceil(e / 64) / 2) == ceil(e / 128)`` for every extent. Only the
        capacity-based fallback, which cannot tell 1..64 elements from a full 64,
        has to refuse -- asserted here too, since it is why the extent has to be
        threaded through.
        """
        from torch_spyre._C import ElementArrangement
        from torch_spyre._inductor.errors import Unsupported
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        fp16 = get_device_dtype(torch.float16)
        # (input sticks, stick-axis extent, expected fp8 num-sticks)
        for in_sticks, extent, expected in ((1, 64, 1), (3, 192, 2), (2, 128, 1)):
            stl = SpyreTensorLayout(
                [in_sticks, 4, 64], [64, 64, 1], fp16, ElementArrangement.STANDARD
            )
            out = rescale_stl_for_dtype(
                stl,
                torch.float8_e4m3fn,
                ElementArrangement.QFP8CH,
                stick_extent=extent,
            )
            self.assertEqual(
                list(out.device_size),
                [expected, 4, 128],
                f"{in_sticks} fp16 stick(s), extent {extent}",
            )
            self.assertEqual(list(out.stride_map), [128, 64, 1])
            self.assertEqual(out.element_arrangement, ElementArrangement.QFP8CH)
            # Without the extent, 1 and 3 sticks are an inexact capacity rescale
            # (64 and 192 elements are not multiples of 128) and are refused
            # rather than floored to 0 or 1; 2 sticks divide exactly.
            if (in_sticks * 64) % 128:
                with self.assertRaises(Unsupported):
                    rescale_stl_for_dtype(
                        stl, torch.float8_e4m3fn, ElementArrangement.QFP8CH
                    )

    def test_explicit_layout_rejects_malformed_device_size(self):
        """The explicit (device_size, stride_map) constructor validates the one
        invariant every consumer assumes: non-negative device dims, one
        stride_map entry each. A negative dim is rejected at construction
        instead of crashing the process later. A size-0 dim is how an empty
        tensor is laid out and stays legal; the degenerate size-0 dim behind
        issue #3604 is refused by rescale_stl_for_dtype instead."""
        from torch_spyre._C import ElementArrangement

        fp16 = get_device_dtype(torch.float16)
        with self.assertRaisesRegex(
            RuntimeError, "device dimension 0 has negative size -1"
        ):
            SpyreTensorLayout(
                [-1, 4, 64], [64, 32, 1], fp16, ElementArrangement.STANDARD
            )
        empty = SpyreTensorLayout(
            [0, 4, 64], [64, 32, 1], fp16, ElementArrangement.STANDARD
        )
        self.assertEqual(list(empty.device_size), [0, 4, 64])
        self.assertEqual(get_device_size_in_bytes(empty), 0)
        with self.assertRaisesRegex(RuntimeError, "stride_map has 2 entries for 3"):
            SpyreTensorLayout([1, 4, 64], [64, 1], fp16, ElementArrangement.STANDARD)
        # -1 (size-1 / sparse) and 0 (broadcast) stride entries stay legal.
        ok = SpyreTensorLayout(
            [1, 4, 64], [-1, 0, 1], fp16, ElementArrangement.STANDARD
        )
        self.assertEqual(list(ok.device_size), [1, 4, 64])


@instantiate_parametrized_tests
class TestRescaleStlForDtype(TestCase):
    """rescale_stl_for_dtype must agree with the layout constructor (issue #4392).

    Rescaling the input's *padded stick capacity* cannot reproduce the stick
    count: capacity is identical for a 32- and a 64-element fp16 row, so it
    over-counts narrowing conversions and floors to zero sticks for a sub-stick
    widening one, which later divides by zero in the D2H copy path.
    """

    # Shapes spanning stick-aligned, sub-stick, and unaligned stick extents.
    # The higher-rank entries matter: their stride_map carries more than one dim
    # with stride == elems_per_stick, so the num-sticks dim is ambiguous by the
    # stride test alone (e.g. 2x4x8x64 fp16 -> [512, 64, 64, 2048, 1]).
    SHAPES = [
        (4, 16),
        (4, 32),
        (4, 63),
        (4, 64),
        (4, 68),
        (4, 128),
        (8, 192),
        (2, 4, 64),
        (2, 4, 8, 64),
        (2, 4, 8, 63),
        (2, 4, 8, 68),
        (1, 1, 4, 32),
    ]

    # A device stick is a fixed 1024 bits, so elems_per_stick is 1024 divided by
    # the *device* format's element width -- always a power of two, which is the
    # property that matters here: for any two widths one divides the other, so a
    # capacity exactness test can never fire in the direction that narrows the
    # element (n * 128 is divisible by 64 and by 32 for every n). That direction
    # over-counts silently instead of refusing, which is why the count has to come
    # from the extent. The extent arithmetic is width-agnostic; these pairs cover
    # every ordered pair of the three widths a dtype conversion can reach --
    # 32 (IEEE_FP32), 64 (SEN169_FP16) and 128 (SEN143_FP8).
    #
    # Do not read that as an inventory of the tree: get_elem_in_stick answers for
    # more formats than the convert paths handle, including SENUINT2 at 512 elems,
    # and the torch dtype's own width does not decide the device format's -- int8
    # maps to SENINT8 (128) but uint8 widens to SENUINT32 (32). torch.bool has no
    # width of its own at all: its format is fp16 or fp32 depending on the operand
    # that produced it (see bool_layout_dtype), so it is not rescaled by dtype and
    # does not appear below.
    DTYPE_PAIRS = [
        (torch.float32, torch.float16),
        (torch.float16, torch.float32),
        (torch.float16, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.float16),
        (torch.float32, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.float32),
    ]

    @parametrize("shape", SHAPES)
    @parametrize("src_dtype,dst_dtype", DTYPE_PAIRS)
    def test_matches_canonical_layout(self, shape, src_dtype, dst_dtype):
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout(list(shape), src_dtype)
        rescaled = rescale_stl_for_dtype(
            src_stl,
            dst_dtype,
            ElementArrangement.STANDARD,
            stick_extent=shape[-1],
        )
        expected = SpyreTensorLayout(list(shape), dst_dtype)
        self.assertEqual(
            list(rescaled.device_size),
            list(expected.device_size),
            f"{shape} {src_dtype}->{dst_dtype}: rescaled device_size "
            f"{list(rescaled.device_size)} != canonical "
            f"{list(expected.device_size)}",
        )
        # The num-sticks stride must be rescaled too, not just its extent: a stale
        # stride describes a differently-shaped tensor than device_size claims.
        self.assertEqual(
            list(rescaled.stride_map),
            list(expected.stride_map),
            f"{shape} {src_dtype}->{dst_dtype}: rescaled stride_map "
            f"{list(rescaled.stride_map)} != canonical "
            f"{list(expected.stride_map)}",
        )
        self.assertEqual(rescaled.device_dtype, get_device_dtype(dst_dtype))

    @parametrize("shape", SHAPES)
    @parametrize(
        "src_dtype,dst_dtype",
        [(torch.float32, torch.float16), (torch.float16, torch.float32)],
    )
    def test_no_zero_sized_dim_without_stick_extent(self, shape, src_dtype, dst_dtype):
        """The capacity fallback must never yield a 0 dim -- it refuses instead.

        A zero-sized device dim reaches an unguarded integer division in
        get_device_stride_infos and kills the process with SIGFPE, so where
        capacity alone cannot give an exact count the fallback raises Unsupported
        rather than flooring (to zero, or to any count it would be inventing).
        Either outcome is acceptable here; a 0 in device_size is not.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.errors import Unsupported
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout(list(shape), src_dtype)
        try:
            rescaled = rescale_stl_for_dtype(
                src_stl, dst_dtype, ElementArrangement.STANDARD
            )
        except Unsupported:
            return
        self.assertNotIn(
            0,
            list(rescaled.device_size),
            f"{shape} {src_dtype}->{dst_dtype}: zero-sized device dim "
            f"{list(rescaled.device_size)}",
        )

    def test_sub_stick_widening_does_not_floor_to_zero(self):
        """The exact reported case: one fp32 stick (capacity 32) -> fp16.

        32 // 64 == 0 under the old capacity rescale.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout([4, 32], torch.float32)
        self.assertEqual(list(src_stl.device_size), [1, 4, 32])
        rescaled = rescale_stl_for_dtype(
            src_stl,
            torch.float16,
            ElementArrangement.STANDARD,
            stick_extent=32,
        )
        self.assertEqual(list(rescaled.device_size), [1, 4, 64])

    def test_ambiguous_stride_map_picks_the_num_sticks_dim(self):
        """Several dims can share stride == elems_per_stick; only one is the
        num-sticks dim, and writing the count into the wrong one permutes the
        tensor. 2x4x8x64 fp16 has stride_map [512, 64, 64, 2048, 1]: dims 1 and 2
        both match, but dim 2 is the num-sticks dim.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout([2, 4, 8, 64], torch.float16)
        self.assertEqual(list(src_stl.stride_map), [512, 64, 64, 2048, 1])
        self.assertEqual(list(src_stl.device_size), [4, 8, 1, 2, 64])
        rescaled = rescale_stl_for_dtype(
            src_stl,
            torch.float32,
            ElementArrangement.STANDARD,
            stick_extent=64,
        )
        # dim 2 goes 1 -> 2 sticks; dim 1 (size 8) must be untouched.
        self.assertEqual(list(rescaled.device_size), [4, 8, 2, 2, 32])

    def test_sentinel_stride_map_is_left_alone(self):
        """A layout with no whole-stick stride (broadcast sentinel -1) is
        untouched apart from the stick depth, before and after the fix."""
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout([4, 1], torch.float32)
        self.assertEqual(list(src_stl.stride_map), [-1, 1, -1])
        rescaled = rescale_stl_for_dtype(
            src_stl, torch.float16, ElementArrangement.STANDARD, stick_extent=1
        )
        self.assertEqual(list(rescaled.device_size), [1, 4, 64])
        self.assertEqual(list(rescaled.stride_map), [-1, 1, -1])

    def test_sentinel_inner_stride_does_not_rescale_a_lookalike_dim(self):
        """A sentinel inner stride must not fall back to a bare ``== in_eps`` match.

        The stick axis of host (1,4,32) with dim_order [1,2,0] is the extent-1 dim,
        so ``stride_map[-1]`` is the sentinel -1 and there is no whole-stick stride
        to rescale. Device dim 1 nevertheless has stride 32 == the fp32 stick depth
        -- an ordinary outer stride that merely shares the value. Rescaling it (what
        an ``== in_eps`` fallback does) describes a differently strided tensor.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        size, strides, dim_order = [1, 4, 32], [128, 32, 1], [1, 2, 0]
        src_stl = SpyreTensorLayout(size, strides, torch.float32, dim_order)
        self.assertEqual(list(src_stl.stride_map), [1, -1, 32, -1])
        expected = SpyreTensorLayout(size, strides, torch.float16, dim_order)
        for kwargs in (
            {"stick_extent": size[dim_order[-1]]},
            {"host_size": size, "host_stride": strides},
            {},
        ):
            rescaled = rescale_stl_for_dtype(
                src_stl, torch.float16, ElementArrangement.STANDARD, **kwargs
            )
            self.assertEqual(
                list(rescaled.stride_map), list(expected.stride_map), f"{kwargs}"
            )
            self.assertEqual(
                list(rescaled.device_size), list(expected.device_size), f"{kwargs}"
            )

    # Non-canonical dim_order, i.e. the stick axis is not the last host dim.
    # host_size lets the helper find it by validation instead of assuming
    # size[-1]; these all fail if the stick count is taken from the last host dim.
    PERMUTED_SHAPES = [(4, 32), (4, 63), (4, 68), (68, 4), (128, 4), (16, 4)]

    @parametrize("shape", PERMUTED_SHAPES)
    @parametrize(
        "src_dtype,dst_dtype",
        [(torch.float32, torch.float16), (torch.float16, torch.float32)],
    )
    def test_permuted_dim_order_matches_canonical(self, shape, src_dtype, dst_dtype):
        """A column-major view resolves to the right stick axis via host_size."""
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        host_size = list(shape)
        # Column-major: host dim 0 is the contiguous one, so it is the stick axis.
        host_strides = [1, shape[0]]
        dim_order = [1, 0]
        src_stl = SpyreTensorLayout(host_size, host_strides, src_dtype, dim_order)
        expected = SpyreTensorLayout(host_size, host_strides, dst_dtype, dim_order)
        rescaled = rescale_stl_for_dtype(
            src_stl,
            dst_dtype,
            ElementArrangement.STANDARD,
            host_size=host_size,
        )
        self.assertEqual(
            list(rescaled.device_size),
            list(expected.device_size),
            f"{shape} strides={host_strides} order={dim_order} "
            f"{src_dtype}->{dst_dtype}: rescaled device_size "
            f"{list(rescaled.device_size)} != canonical "
            f"{list(expected.device_size)}",
        )

    # A staggered EA is a statement about *within-stick* order. A sparse stick --
    # ``stride_map[-1] < 0``, i.e. the stick axis is an extent-1 host dim -- holds
    # one valid host element, so there is no such order to describe and the
    # conversion output is an ordinary STANDARD stick. Reductions produce these
    # constantly: ``mean(dim=-1, keepdim=True)`` leaves the reduced axis at extent 1
    # on the stick, and downcasting that result is the RMSNorm case that used to
    # label two buffers FP32_TO_DL16. ``DtypeOpTable.ea_map`` (dtype_ops.py) keys on
    # (src dtype, dst dtype, src ea) alone and structurally cannot see a geometry,
    # so the correction lives in ``rescale_stl_for_dtype``, which holds both.
    SPARSE_STICK_SHAPES = [(1, 280, 1), (4, 1), (2, 4, 1), (1, 1)]

    # Same conversions on a stick that really does carry several host elements, so
    # the stagger is observable and must survive. Without this control the test
    # above passes for a helper that simply never stamps a staggered EA.
    DENSE_STICK_SHAPES = [(4, 32), (4, 64), (4, 68), (2, 4, 64)]

    STAGGERING_CONVERTS = [
        (torch.float32, torch.float16, ElementArrangement.FP32_TO_DL16),
        (torch.float16, torch.float32, ElementArrangement.DL16_TO_FP32),
    ]

    @parametrize("shape", SPARSE_STICK_SHAPES)
    @parametrize("src_dtype,dst_dtype,staggered", STAGGERING_CONVERTS)
    def test_sparse_stick_drops_staggered_ea(
        self, shape, src_dtype, dst_dtype, staggered
    ):
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout(list(shape), src_dtype)
        self.assertLess(
            src_stl.stride_map[-1],
            0,
            f"{shape} {src_dtype} is not a sparse stick: "
            f"stride_map={list(src_stl.stride_map)}",
        )
        rescaled = rescale_stl_for_dtype(
            src_stl, dst_dtype, staggered, stick_extent=shape[-1]
        )
        self.assertEqual(
            rescaled.element_arrangement,
            ElementArrangement.STANDARD,
            f"{shape} {src_dtype}->{dst_dtype}: a stick holding one host element "
            f"cannot carry a stagger, got {rescaled.element_arrangement}",
        )
        # The EA correction must not disturb the geometry the helper exists to
        # compute -- it stays equal to canonical, exactly as with a STANDARD ea.
        expected = SpyreTensorLayout(list(shape), dst_dtype)
        self.assertEqual(list(rescaled.device_size), list(expected.device_size))
        self.assertEqual(list(rescaled.stride_map), list(expected.stride_map))

    @parametrize("shape", DENSE_STICK_SHAPES)
    @parametrize("src_dtype,dst_dtype,staggered", STAGGERING_CONVERTS)
    def test_dense_stick_keeps_staggered_ea(
        self, shape, src_dtype, dst_dtype, staggered
    ):
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout(list(shape), src_dtype)
        self.assertGreater(
            src_stl.stride_map[-1],
            0,
            f"{shape} {src_dtype} is not a dense stick: "
            f"stride_map={list(src_stl.stride_map)}",
        )
        rescaled = rescale_stl_for_dtype(
            src_stl, dst_dtype, staggered, stick_extent=shape[-1]
        )
        self.assertEqual(
            rescaled.element_arrangement,
            staggered,
            f"{shape} {src_dtype}->{dst_dtype}: the stagger is observable on a "
            f"stick of {src_stl.device_size[-1]} elements and must be kept",
        )

    def test_last_device_dim_is_never_a_num_sticks_candidate(self):
        """The stick depth counts elements, not sticks.

        Row-major host (4,32) with dim_order [1,0] has fp32 stride_map
        [1024, 1, 32], whose *last* entry equals elems_per_stick. Selecting it
        would overwrite the rescaled stick depth with a stick count.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout([4, 32], [32, 1], torch.float32, [1, 0])
        self.assertEqual(list(src_stl.stride_map), [1024, 1, 32])
        self.assertEqual(list(src_stl.device_size), [1, 32, 32])
        self.assertEqual(len(src_stl.device_size) - 1, 2)  # the offending index
        rescaled = rescale_stl_for_dtype(
            src_stl,
            torch.float16,
            ElementArrangement.STANDARD,
            host_size=[4, 32],
        )
        # The stick depth must be the fp16 depth, not a stick count.
        self.assertEqual(rescaled.device_size[-1], 64)
        expected = SpyreTensorLayout([4, 32], [32, 1], torch.float16, [1, 0])
        self.assertEqual(list(rescaled.device_size), list(expected.device_size))

    def test_disproven_candidate_is_left_alone(self):
        """A known extent that rules out every candidate must not be estimated.

        host (2,3,32) fp32 with dim_order [2,1,0] has stride_map
        [32, 3072, 1, 96]: dim 0's stride coincides with elems_per_stick but its
        size (3) is not the input stick count for any host extent. The capacity
        estimate would write 2 there and permute the tensor; the canonical layout
        leaves it at 3.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        host_size, host_strides, dim_order = [2, 3, 32], [96, 32, 1], [2, 1, 0]
        src_stl = SpyreTensorLayout(host_size, host_strides, torch.float32, dim_order)
        self.assertEqual(list(src_stl.stride_map), [32, 3072, 1, 96])
        self.assertEqual(list(src_stl.device_size), [3, 1, 32, 32])
        rescaled = rescale_stl_for_dtype(
            src_stl,
            torch.float16,
            ElementArrangement.STANDARD,
            host_size=host_size,
        )
        expected = SpyreTensorLayout(host_size, host_strides, torch.float16, dim_order)
        self.assertEqual(list(rescaled.device_size), list(expected.device_size))
        self.assertEqual(list(rescaled.device_size), [3, 1, 32, 64])

    # Stick axis with a non-unit host stride, i.e. not host-contiguous. Its
    # num-sticks stride is host_stride[stick_dim] * elems_per_stick, so a bare
    # `stride == elems_per_stick` test matches nothing and the layout is declined
    # outright. The general factor is stride_map[-1], the inner-stick stride.
    # Row-major, so host dim 0 is the stick axis and carries host stride 4.
    NON_CONTIGUOUS_STICK_AXIS = [([68, 4], [4, 1]), ([132, 4], [4, 1])]

    @parametrize("host_size,host_strides", NON_CONTIGUOUS_STICK_AXIS)
    @parametrize(
        "src_dtype,dst_dtype",
        [(torch.float32, torch.float16), (torch.float16, torch.float32)],
    )
    def test_non_contiguous_stick_axis_matches_canonical(
        self, host_size, host_strides, src_dtype, dst_dtype
    ):
        """The num-sticks stride is host_stride[stick_dim] * elems_per_stick.

        Row-major host (68,4) with dim_order [1,0] makes host dim 0 the stick axis
        with host stride 4, so fp16 stride_map is [256, 1, 4] -- the num-sticks
        stride is 4*64, not 64. A bare ``stride == elems_per_stick`` test matches
        nothing here and the layout is declined outright. Recognising it needs
        stride_map[-1], which is already on the layout, so no host strides have to
        be threaded in.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        dim_order = [1, 0]
        src_stl = SpyreTensorLayout(host_size, host_strides, src_dtype, dim_order)
        # The stick axis is not host-contiguous: its host stride is the inner-stick
        # stride the layout records, and it is not 1.
        self.assertEqual(src_stl.stride_map[-1], host_strides[dim_order[-1]])
        self.assertNotEqual(src_stl.stride_map[-1], 1)
        rescaled = rescale_stl_for_dtype(
            src_stl,
            dst_dtype,
            ElementArrangement.STANDARD,
            host_size=host_size,
        )
        expected = SpyreTensorLayout(host_size, host_strides, dst_dtype, dim_order)
        self.assertEqual(list(rescaled.device_size), list(expected.device_size))
        self.assertEqual(list(rescaled.stride_map), list(expected.stride_map))

    def test_non_contiguous_stick_axis_exact_values(self):
        """Pins the numbers behind the parametrized case above.

        Host (68,4) needs 2 fp16 sticks for the 68-element axis and 3 fp32 ones,
        and the num-sticks stride is 4*64 == 256 at fp16, 4*32 == 128 at fp32.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout([68, 4], [4, 1], torch.float16, [1, 0])
        self.assertEqual(list(src_stl.device_size), [2, 4, 64])
        self.assertEqual(list(src_stl.stride_map), [256, 1, 4])
        rescaled = rescale_stl_for_dtype(
            src_stl,
            torch.float32,
            ElementArrangement.STANDARD,
            host_size=[68, 4],
        )
        self.assertEqual(list(rescaled.device_size), [3, 4, 32])
        self.assertEqual(list(rescaled.stride_map), [128, 1, 4])

    def test_stride_is_rescaled_even_when_the_count_is_ambiguous(self):
        """Declining the stick *count* must not leave the stick *stride* stale.

        The count needs the host extent; the stride is
        host_stride[stick_dim] * elems_per_stick and needs only stride_map[-1], so
        the two are independent. Host (4,63) column-major fp16 is one stick and both
        host extents (4 and 63) give in_sticks == 1 while implying different fp32
        counts, so the count is rightly declined -- but the canonical layout still
        halves the num-sticks stride from 64 to 32, and so must we.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        host_size, host_strides, dim_order = [4, 63], [1, 4], [1, 0]
        src_stl = SpyreTensorLayout(host_size, host_strides, torch.float16, dim_order)
        self.assertEqual(list(src_stl.device_size), [1, 63, 64])
        self.assertEqual(list(src_stl.stride_map), [64, 4, 1])
        rescaled = rescale_stl_for_dtype(
            src_stl,
            torch.float32,
            ElementArrangement.STANDARD,
            host_size=host_size,
        )
        expected = SpyreTensorLayout(host_size, host_strides, torch.float32, dim_order)
        # The count stayed at 1 (correctly, and it is what the canonical layout has)
        # while the stride went 64 -> 32.
        self.assertEqual(list(expected.device_size), [1, 63, 32])
        self.assertEqual(list(expected.stride_map), [32, 4, 1])
        self.assertEqual(list(rescaled.device_size), [1, 63, 32])
        self.assertEqual(list(rescaled.stride_map), [32, 4, 1])

    @parametrize("shape", SHAPES)
    @parametrize("src_dtype,dst_dtype", DTYPE_PAIRS)
    def test_host_stride_matches_canonical(self, shape, src_dtype, dst_dtype):
        """Host size + host strides must be as exact as an authoritative extent.

        This is the eager ``.to()`` path, which has no MemoryDep to run coordinate
        identity against but does have a real Tensor's strides. Same assertions as
        test_matches_canonical_layout, which gets the extent handed to it.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        host_strides, acc = [0] * len(shape), 1
        for i in reversed(range(len(shape))):
            host_strides[i] = acc
            acc *= shape[i]
        src_stl = SpyreTensorLayout(list(shape), src_dtype)
        rescaled = rescale_stl_for_dtype(
            src_stl,
            dst_dtype,
            ElementArrangement.STANDARD,
            host_size=list(shape),
            host_stride=host_strides,
        )
        expected = SpyreTensorLayout(list(shape), dst_dtype)
        self.assertEqual(
            list(rescaled.device_size),
            list(expected.device_size),
            f"{shape} {src_dtype}->{dst_dtype}: device_size "
            f"{list(rescaled.device_size)} != canonical "
            f"{list(expected.device_size)}",
        )
        self.assertEqual(
            list(rescaled.stride_map),
            list(expected.stride_map),
            f"{shape} {src_dtype}->{dst_dtype}: stride_map "
            f"{list(rescaled.stride_map)} != canonical "
            f"{list(expected.stride_map)}",
        )

    def test_host_stride_resolves_what_host_size_alone_cannot(self):
        """Host strides pin the stick axis; host extents only constrain it.

        Host (4,64) fp16 is one stick and both extents give in_sticks == 1, so the
        host_size search cannot tell 1 fp32 stick from 2 and declines -- leaving the
        input's count, an *under*-count, which the unfixed capacity rescale happened
        to get right. stride_map[-1] is host_stride[stick_dim], so matching the host
        strides against it names host dim 1 outright and the count is exact.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout([4, 64], torch.float16)
        self.assertEqual(list(src_stl.device_size), [1, 4, 64])
        kwargs = dict(host_size=[4, 64])
        without = rescale_stl_for_dtype(
            src_stl, torch.float32, ElementArrangement.STANDARD, **kwargs
        )
        with_strides = rescale_stl_for_dtype(
            src_stl,
            torch.float32,
            ElementArrangement.STANDARD,
            host_stride=[64, 1],
            **kwargs,
        )
        expected = SpyreTensorLayout([4, 64], torch.float32)
        self.assertEqual(list(expected.device_size), [2, 4, 32])
        # Extents alone: count declined (stale 1) and, with two dims sharing the
        # candidate stride, the stride declined too.
        self.assertEqual(list(without.device_size), [1, 4, 32])
        self.assertEqual(list(without.stride_map), [64, 64, 1])
        # Strides resolve both.
        self.assertEqual(list(with_strides.device_size), [2, 4, 32])
        self.assertEqual(list(with_strides.stride_map), list(expected.stride_map))

    def test_host_stride_tie_broken_by_dropping_size_one_dims(self):
        """A size-1 dim shares its neighbour's host stride and is never the axis.

        Host (63,1) is dense with strides [1,1], so both dims match the inner-stick
        stride. Dim 1 has extent 1 and cannot be the axis a 63-element stick axis
        is measured on, so dropping it resolves the tie to dim 0 and the count is
        exact. Without the tie-break this falls back to the extent search, which
        declines and leaves the input's single stick.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        host_size, host_strides, dim_order = [63, 1], [1, 1], [1, 0]
        src_stl = SpyreTensorLayout(host_size, host_strides, torch.float16, dim_order)
        self.assertEqual(list(src_stl.device_size), [1, 1, 64])
        rescaled = rescale_stl_for_dtype(
            src_stl,
            torch.float32,
            ElementArrangement.STANDARD,
            host_size=host_size,
            host_stride=host_strides,
        )
        expected = SpyreTensorLayout(host_size, host_strides, torch.float32, dim_order)
        self.assertEqual(list(expected.device_size), [2, 1, 32])
        self.assertEqual(list(rescaled.device_size), [2, 1, 32])
        self.assertEqual(list(rescaled.stride_map), list(expected.stride_map))

    def test_host_stride_that_names_no_dim_falls_back_to_host_size(self):
        """Strides that match no dim must degrade to the extent search, not skip it.

        A caller can hand over strides that do not correspond to this layout (a
        non-dense view whose conversion output was made contiguous, say). Naming no
        dim then has to leave the host_size behaviour untouched rather than lose the
        extents entirely and drop to the clamped estimate. Strides whose length does
        not match host_size are ignored the same way, rather than raising IndexError
        while indexing the extents by a stride dim.
        """
        from torch_spyre._inductor.constants import ElementArrangement
        from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

        src_stl = SpyreTensorLayout([4, 32], torch.float32)
        kwargs = dict(
            out_dtype=torch.float16,
            ea=ElementArrangement.STANDARD,
            host_size=[4, 32],
        )
        baseline = rescale_stl_for_dtype(src_stl, **kwargs)
        for bogus in ([100, 7], [1], [32, 1, 1]):
            got = rescale_stl_for_dtype(src_stl, host_stride=bogus, **kwargs)
            self.assertEqual(
                list(got.device_size),
                list(baseline.device_size),
                f"host_stride={bogus} changed device_size",
            )
            self.assertEqual(
                list(got.stride_map),
                list(baseline.stride_map),
                f"host_stride={bogus} changed stride_map",
            )


if __name__ == "__main__":
    run_tests()

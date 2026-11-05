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

"""Tests for torch_spyre._C.get_reserved_dims.

A read-only accessor onto a tensor's dynamic-shape reservation (set by
tensor.to("spyre", dynamic={dim: {min, max, granularity}})). Unlike most
accessors in this codebase, it must never raise: it is meant to be called
on arbitrary graph inputs, including tensors that were never reserved and
tensors that aren't even on this device.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401
from torch_spyre._C import get_reserved_dims

DEVICE = torch.device("spyre")


class TestGetReservedDims(TestCase):
    def test_reserved_tensor_returns_declared_dict(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertEqual(
            get_reserved_dims(x_dev),
            {0: {"min": 70, "max": 630, "granularity": 70}},
        )

    def test_unreserved_spyre_tensor_returns_none(self) -> None:
        plain = torch.rand(4, 4, dtype=torch.float16).to(DEVICE)
        self.assertIsNone(get_reserved_dims(plain))

    def test_cpu_tensor_returns_none_not_raises(self) -> None:
        cpu = torch.rand(4, 4, dtype=torch.float16)
        self.assertIsNone(get_reserved_dims(cpu))

    def test_detached_copy_inherits_the_map(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertEqual(get_reserved_dims(x_dev.detach()), get_reserved_dims(x_dev))

    def test_still_reserved_after_in_range_resize(self) -> None:
        # resize_ legitimately changes the shape; it must keep the contract.
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        x_dev.resize_((140, 1024))
        self.assertEqual(
            get_reserved_dims(x_dev),
            {0: {"min": 70, "max": 630, "granularity": 70}},
        )

    def test_transposed_view_is_not_reserved(self) -> None:
        # The map is keyed by absolute dim, so it must not follow an
        # axis-swapping view.
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertIsNone(get_reserved_dims(x_dev.t()))
        self.assertIsNone(get_reserved_dims(x_dev.t().detach()))

    def test_in_place_transpose_drops_the_map(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        d = x_dev.detach()
        d.t_()
        self.assertIsNone(get_reserved_dims(d))
        # The original is a different tensor object and is untouched.
        self.assertIsNotNone(get_reserved_dims(x_dev))

    def test_in_place_rank_change_drops_the_map(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        d = x_dev.detach()
        d.unsqueeze_(0)
        self.assertIsNone(get_reserved_dims(d))


if __name__ == "__main__":
    run_tests()

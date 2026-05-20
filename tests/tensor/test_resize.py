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
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
)

import torch_spyre  # noqa: F401

DTYPE = torch.float16


def _make(shape: list[int]) -> torch.Tensor:
    """Create a sequential fp16 tensor on Spyre."""
    numel = 1
    for d in shape:
        numel *= d
    return torch.arange(0, numel, dtype=DTYPE).reshape(shape).to("spyre")


@instantiate_parametrized_tests
class TestResizeSameNumel(TestCase):
    """resize_: new shape has the same number of elements as the original.

    The storage is not reallocated and all elements must be preserved in
    their original flat order.
    """

    @parametrize(
        "orig_shape,new_shape",
        [
            subtest(([16], [16]), name="1d_noop"),
            subtest(([32], [32]), name="1d_noop_larger"),
            subtest(([4, 8], [32]), name="2d_to_1d_flatten"),
            subtest(([2, 8], [16]), name="2d_to_1d_flatten_small"),
            subtest(([4, 8], [8, 4]), name="2d_transpose_shape"),
            subtest(([4, 8], [2, 16]), name="2d_regroup"),
            subtest(([2, 4, 8], [8, 8]), name="3d_to_2d_merge_all"),
            subtest(([2, 4, 8], [4, 16]), name="3d_to_2d_merge_last_two"),
        ],
    )
    def test_data_preserved(self, orig_shape, new_shape):
        t = _make(orig_shape)
        orig_flat = t.cpu().flatten()

        t.resize_(*new_shape)

        self.assertEqual(list(t.shape), new_shape)
        self.assertEqual(t.numel(), orig_flat.numel())
        torch.testing.assert_close(t.cpu().flatten(), orig_flat)


@instantiate_parametrized_tests
class TestResizeShrink(TestCase):
    """resize_: new shape has fewer elements than the original.

    Storage is not reallocated.  The first new_numel elements (in flat
    order) must equal the corresponding elements of the original tensor.
    """

    @parametrize(
        "orig_shape,new_shape",
        [
            subtest(([8], [4]), name="1d_half"),
            subtest(([8], [1]), name="1d_to_one"),
            subtest(([8], [6]), name="1d_non_power_of_two"),
            subtest(([4, 8], [16]), name="2d_to_1d_full_row"),
            subtest(([4, 8], [8]), name="2d_to_1d_half_row"),
            subtest(([4, 8], [2, 8]), name="2d_shrink_rows"),
            subtest(([4, 8], [2, 4]), name="2d_shrink_both_dims"),
            subtest(([2, 4, 8], [4, 8]), name="3d_to_2d_drop_outer"),
            subtest(([2, 4, 8], [2, 8]), name="3d_to_2d_half"),
        ],
    )
    def test_data_preserved(self, orig_shape, new_shape):
        t = _make(orig_shape)
        orig_flat = t.cpu().flatten()

        t.resize_(*new_shape)

        new_numel = t.numel()
        self.assertEqual(list(t.shape), new_shape)
        torch.testing.assert_close(t.cpu().flatten(), orig_flat[:new_numel])


@instantiate_parametrized_tests
class TestResizeExpand(TestCase):
    """resize_: new shape has more elements than the original.

    New storage is allocated and the original elements are copied into flat
    positions [0 .. old_numel-1].  Elements beyond that are uninitialised
    and are not checked.
    """

    @parametrize(
        "orig_shape,new_shape",
        [
            subtest(([8], [32]), name="1d_4x"),
            subtest(([4], [16]), name="1d_4x_small"),
            subtest(([2, 8], [32]), name="2d_to_1d_double"),
            subtest(([2, 8], [64]), name="2d_to_1d_4x"),
            subtest(([2, 8], [4, 8]), name="2d_double_rows"),
            subtest(([2, 8], [4, 16]), name="2d_double_both"),
            subtest(([2, 2, 8], [8, 8]), name="3d_to_2d_expand"),
            subtest(([2, 2, 8], [4, 16]), name="3d_to_2d_expand_regroup"),
        ],
    )
    def test_original_elements_preserved(self, orig_shape, new_shape):
        t = _make(orig_shape)
        old_numel = t.numel()
        orig_flat = t.cpu().flatten()

        t.resize_(*new_shape)

        self.assertEqual(list(t.shape), new_shape)
        # Only the first old_numel elements are guaranteed; tail is uninitialised.
        torch.testing.assert_close(t.cpu().flatten()[:old_numel], orig_flat)


if __name__ == "__main__":
    run_tests()

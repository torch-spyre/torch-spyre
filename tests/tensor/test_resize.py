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

import os
import sys
import unittest

import torch

_tests_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_tests_dir)

from inductor.utils_inductor import ParameterizedTestMeta, cached_randn  # type: ignore[attr-defined]  # noqa: E402

import torch_spyre  # noqa: F401, E402

DTYPE = torch.float16


class TestResize(unittest.TestCase, metaclass=ParameterizedTestMeta):
    PARAMS = {
        ("test_same_numel_data_preserved", "test_same_numel"): {
            "param_sets": {
                "1d_noop": ([16], [16]),
                "1d_noop_larger": ([32], [32]),
                "2d_to_1d_flatten": ([4, 8], [32]),
                "2d_to_1d_flatten_small": ([2, 8], [16]),
                "2d_transpose_shape": ([4, 8], [8, 4]),
                "2d_regroup": ([4, 8], [2, 16]),
                "3d_to_2d_merge_all": ([2, 4, 8], [8, 8]),
                "3d_to_2d_merge_last_two": ([2, 4, 8], [4, 16]),
                # multi-stick boundary cases (1 stick = 64 fp16 elements)
                "1d_2sticks_noop": ([256], [256]),
                "2d_to_1d_2sticks": ([8, 16], [128]),
            },
        },
        ("test_shrink_data_preserved", "test_shrink"): {
            "param_sets": {
                "1d_half": ([8], [4]),
                "1d_to_one": ([8], [1]),
                "1d_non_power_of_two": ([8], [6]),
                "2d_to_1d_full_row": ([4, 8], [16]),
                "2d_to_1d_half_row": ([4, 8], [8]),
                "2d_shrink_rows": ([4, 8], [2, 8]),
                "2d_shrink_both_dims": ([4, 8], [2, 4]),
                "3d_to_2d_drop_outer": ([2, 4, 8], [4, 8]),
                "3d_to_2d_half": ([2, 4, 8], [2, 8]),
                # stick-boundary cases (1 stick = 64 fp16 elements)
                "1d_2sticks_to_1stick": ([128], [64]),
                "2d_4sticks_to_1stick": ([16, 16], [8, 8]),
            },
        },
        ("test_expand_original_elements_preserved", "test_expand"): {
            "param_sets": {
                "1d_4x": ([8], [32]),
                "1d_4x_small": ([4], [16]),
                "2d_to_1d_double": ([2, 8], [32]),
                "2d_to_1d_4x": ([2, 8], [64]),
                "2d_double_rows": ([2, 8], [4, 8]),
                "2d_double_both": ([2, 8], [4, 16]),
                "3d_to_2d_expand": ([2, 2, 8], [8, 8]),
                "3d_to_2d_expand_regroup": ([2, 2, 8], [4, 16]),
                # stick-boundary cases (1 stick = 64 fp16 elements)
                "1d_1stick_to_4sticks": ([64], [256]),
                "2d_1stick_to_2sticks": ([8, 8], [8, 16]),
                "1d_large_4sticks_to_16sticks": ([256], [1024]),
            },
        },
    }

    def test_same_numel(self, orig_shape, new_shape):
        """resize_ to a shape with the same numel preserves all elements in flat order."""
        t = cached_randn(orig_shape, dtype=DTYPE).to("spyre")
        orig_flat = t.cpu().flatten()

        t.resize_(*new_shape)

        self.assertEqual(list(t.shape), new_shape)
        self.assertEqual(t.numel(), orig_flat.numel())
        torch.testing.assert_close(t.cpu().flatten(), orig_flat)

    def test_shrink(self, orig_shape, new_shape):
        """resize_ to a smaller shape preserves the first new_numel elements in flat order."""
        t = cached_randn(orig_shape, dtype=DTYPE).to("spyre")
        orig_flat = t.cpu().flatten()

        t.resize_(*new_shape)

        new_numel = t.numel()
        self.assertEqual(list(t.shape), new_shape)
        torch.testing.assert_close(t.cpu().flatten(), orig_flat[:new_numel])

    def test_expand(self, orig_shape, new_shape):
        """resize_ to a larger shape preserves original elements in flat positions [0..old_numel-1]."""
        t = cached_randn(orig_shape, dtype=DTYPE).to("spyre")
        old_numel = t.numel()
        orig_flat = t.cpu().flatten()

        t.resize_(*new_shape)

        self.assertEqual(list(t.shape), new_shape)
        torch.testing.assert_close(t.cpu().flatten()[:old_numel], orig_flat)


if __name__ == "__main__":
    unittest.main()

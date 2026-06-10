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

from inductor.utils_inductor import ParameterizedTestMeta  # type: ignore[attr-defined]  # noqa: E402

import torch_spyre  # noqa: F401, E402

DTYPE = torch.float16


def _identifiable_tensor(shape):
    """Sequential integers 0..numel-1 reshaped to `shape`.

    Unique values detect corruption after resize_() without assuming element
    ordering. All values are exact in float16 (< 2048).
    """
    numel = 1
    for dim in shape:
        numel *= dim
    return torch.arange(numel, dtype=DTYPE).reshape(shape)


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
        """Same-numel resize_ reinterprets storage in-place; full element set survives."""
        t_cpu = _identifiable_tensor(orig_shape)
        t = t_cpu.to("spyre")
        orig_numel = t.numel()

        t.resize_(*new_shape)

        self.assertEqual(list(t.shape), new_shape)
        self.assertEqual(t.numel(), orig_numel)
        result = t.cpu()
        self.assertEqual(list(result.shape), new_shape)
        # Permutation of originals — no corrupted or lost elements.
        self.assertEqual(
            sorted(result.flatten().tolist()), sorted(t_cpu.flatten().tolist())
        )

    def test_shrink(self, orig_shape, new_shape):
        """Shrink resize_ reinterprets a storage slice; surviving elements are genuine."""
        t_cpu = _identifiable_tensor(orig_shape)
        t = t_cpu.to("spyre")
        orig_numel = t.numel()
        orig_values = set(t_cpu.flatten().tolist())

        t.resize_(*new_shape)

        self.assertEqual(list(t.shape), new_shape)
        self.assertLessEqual(t.numel(), orig_numel)
        result = t.cpu()
        self.assertEqual(list(result.shape), new_shape)
        result_values = result.flatten().tolist()
        # Subset of originals — no duplicated or corrupted elements.
        self.assertEqual(len(result_values), len(set(result_values)))
        self.assertTrue(set(result_values).issubset(orig_values))

    def test_expand(self, orig_shape, new_shape):
        """Expand resize_ (D2H → CPU resize_ → H2D) preserves original elements verbatim."""
        t_cpu = _identifiable_tensor(orig_shape)
        t = t_cpu.to("spyre")
        orig_numel = t.numel()

        t.resize_(*new_shape)

        self.assertEqual(list(t.shape), new_shape)
        self.assertGreaterEqual(t.numel(), orig_numel)
        result = t.cpu()
        self.assertEqual(list(result.shape), new_shape)
        # The original elements must survive at their original flat positions.
        self.assertEqual(
            result.flatten()[:orig_numel].tolist(), t_cpu.flatten().tolist()
        )


if __name__ == "__main__":
    unittest.main()

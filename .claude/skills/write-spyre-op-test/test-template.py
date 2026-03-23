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

"""
Template for writing Spyre compiled-path operation tests.

Copy this file to tests/_inductor/ and rename it. Then:
1. Replace the op dict and param sets with your operation
2. Implement the base test functions
3. Run: python3 -m pytest tests/_inductor/<your_test>.py
"""

import unittest

import torch

from utils_inductor import (
    ParameterizedTestMeta,
    cached_randn,  # noqa: F401
    compare_with_cpu,
    make_param_dict,
)

# Define ops to test (for cross-product parameterization)
MY_OPS_DICT = {
    # "op_name": torch.op_function,
    # Example:
    # "abs": torch.abs,
    # "neg": torch.neg,
}


class TestMyOps(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        # --- Example: Unary op with ops_dict (cross product) ---
        # Generates: test_my_unary_op_{op_name}_{shape_key}
        ("test_my_unary_op", "test_unary_op"): {
            "ops_dict": MY_OPS_DICT,
            "param_sets": make_param_dict(
                [
                    ((256,),),  # 1D, stick-aligned
                    ((67, 256),),  # 2D, non-aligned first dim
                    ((67, 71, 256),),  # 3D
                    ((7, 12, 32, 64),),  # 4D
                ]
            ),
        },
        # --- Example: Concrete test without ops_dict ---
        # Generates: test_my_specific_op_{case_name}
        # ("test_my_specific_op", "test_specific_base"): {
        #     "param_sets": {
        #         "2d": (cached_randn((67, 256)),),
        #         "3d": (cached_randn((4, 67, 256)),),
        #     },
        # },
    }

    # --- Base test functions ---
    # These are referenced by PARAMS and expanded into parameterized tests.
    # They are removed from the test class after expansion.

    def test_unary_op(self, op, x):
        """Base function for unary op tests. Receives (self, op, *params)."""
        compare_with_cpu(lambda a: op(a), x)

    # def test_specific_base(self, x):
    #     """Base function for concrete tests. Receives (self, *params)."""
    #     compare_with_cpu(
    #         lambda a: torch.nn.functional.my_op(a),
    #         x,
    #     )

    # --- Non-parameterized tests ---
    # These run as-is without parameterization.

    # def test_edge_case(self):
    #     x = cached_randn((64,), abs=True)  # abs=True for positive inputs
    #     compare_with_cpu(lambda a: torch.sqrt(a), x)


if __name__ == "__main__":
    unittest.main()

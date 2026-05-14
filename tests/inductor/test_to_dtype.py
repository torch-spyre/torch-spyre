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

"""Tests for explicit dtype conversions in torch-spyre.

This test suite verifies:
1. Explicit dtype casts via .to(dtype=...) and .to(device=..., dtype=...)
2. DtypeOpTable operator mappings using PyTorch dtypes
"""

import torch
import unittest

from torch_spyre._inductor.dtype_ops import DtypeOpTable
from tests.inductor.utils_inductor import (
    compare_with_cpu,
    make_param_dict,
    ParameterizedTestMeta,
)

ALL_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.bool,
]

ALL_DTYPE_PAIRS = [(src, dst) for src in ALL_DTYPES for dst in ALL_DTYPES if src != dst]


class TestDtypeOpTable(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """Verify DtypeOpTable mappings."""

    PARAMS = {
        ("test_op_mapping", "test_op_mapping"): {
            "param_sets": {
                f"{str(src).replace('torch.', '')}_to_{str(dst).replace('torch.', '')}": (
                    src,
                    dst,
                )
                for src, dst in ALL_DTYPE_PAIRS
            },
        },
    }

    def test_op_mapping(self, src, dst):
        """Verify operator mapping."""
        result = DtypeOpTable.get_operator(src, dst)
        conversions = DtypeOpTable.get_table()
        if (src, dst) in conversions:
            expected = conversions[(src, dst)]
            assert result == expected, (
                f"Expected {expected} for {src}->{dst}, got {result}"
            )
        else:
            assert result is None, (
                f"Expected None for unsupported {src}->{dst}, got {result}"
            )


DTYPE_OPS_PAIRS = {
    f"{str(src).replace('torch.', '')}_to_{str(dst).replace('torch.', '')}": (
        src,
        dst,
    )
    for (src, dst) in DtypeOpTable.get_dtype_pairs()
    if src != torch.bfloat16
    and dst != torch.bfloat16
    and src != torch.bool
    and dst != torch.bool
    and not (src == torch.float16 and dst == torch.float32)  # Exclude fp16->fp32
}

# Small test shapes for dtype conversion tests
TEST_SHAPES = [
    (4, 2),
    # (4, 8), # FIXME deeptools accuracy issue #4261
]


class TestDtypeConversions(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """Test dtype conversions on Spyre."""

    torch.manual_seed(0xAFFE)

    PARAMS = {
        ("test_to_dtype", "test_to_dtype"): {
            "ops_dict": DTYPE_OPS_PAIRS,
            "param_sets": make_param_dict(tuple((s,) for s in TEST_SHAPES)),
        },
    }

    def test_to_dtype(self, dtype_pair, x):
        """Test dtype conversion."""
        src_dtype, dst_dtype = dtype_pair
        x_src = x.to(dtype=src_dtype)

        # FIXME: sdsc not getting compiled
        compare_with_cpu(
            lambda a: a.to(dtype=dst_dtype),
            x_src,
            cpu_compile=False,
            run_eager=False,
        )


if __name__ == "__main__":
    unittest.main()

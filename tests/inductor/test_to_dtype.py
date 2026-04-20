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
2. DtypeOpTable operator mappings
"""

import torch
import unittest

from torch_spyre._C import DataFormats
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.constants import (
    DL16TOFP32_OP,
    FP32TODL16_OP,
    DL16TOBF16_OP,
    FP8TODL16_OP,
)
from tests.inductor.utils_inductor import (
    compare_with_cpu,
    make_param_dict,
    ParameterizedTestMeta,
)

ALL_FORMATS = [
    fmt for name, fmt in DataFormats.__members__.items() if name != "INVALID"
]

FORMAT_PAIRS = [(src, dst) for src in ALL_FORMATS for dst in ALL_FORMATS if src != dst]

# Map DataFormats to PyTorch dtypes
FORMAT_TO_DTYPE = {
    DataFormats.IEEE_FP32: torch.float32,
    DataFormats.SEN169_FP16: torch.float16,
    DataFormats.BFLOAT16: torch.bfloat16,
    DataFormats.SEN143_FP8: torch.float8_e4m3fn,
    DataFormats.SEN152_FP8: torch.float8_e5m2,
}

SUPPORTED_CONVERSIONS = {
    (DataFormats.IEEE_FP32, DataFormats.SEN169_FP16): FP32TODL16_OP,
    (DataFormats.SEN169_FP16, DataFormats.IEEE_FP32): DL16TOFP32_OP,
    (DataFormats.SEN169_FP16, DataFormats.BFLOAT16): DL16TOBF16_OP,
    (DataFormats.SEN143_FP8, DataFormats.SEN169_FP16): FP8TODL16_OP,
    (DataFormats.SEN152_FP8, DataFormats.SEN169_FP16): FP8TODL16_OP,
}

# TRACE-TIME FILTERING: Only supported ops are added to the dictionary.
# This ensures pytest only collects and lists valid test cases.
SUPPORTED_OPS_DICT = {
    f"{src.name}_to_{dst.name}": (src, dst)
    for (src, dst) in SUPPORTED_CONVERSIONS.keys()
}

TEST_SHAPES = [
    (64,),
    (256,),
    (67,),
    (32, 64),
    (67, 256),
    (71, 67),
    (16, 32, 64),
    (67, 71, 256),
    (8, 16, 32, 64),
    (7, 12, 32, 64),
    (3, 5, 7, 11, 64),
]


class TestDtypeOpTable(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """Verify DtypeOpTable mappings for all format combinations."""

    PARAMS = {
        ("test_format_mapping", "test_format_mapping_base"): {
            "param_sets": {
                f"{src.name}_to_{dst.name}": (src, dst) for src, dst in FORMAT_PAIRS
            },
        },
    }

    def test_format_mapping_base(self, src, dst):
        """Validates that DtypeOpTable returns correct ops (or None) for all pairs."""
        result = DtypeOpTable.get_operator(src, dst)
        if (src, dst) in SUPPORTED_CONVERSIONS:
            expected = SUPPORTED_CONVERSIONS[(src, dst)]
            assert result == expected, (
                f"Expected {expected} for {src.name}->{dst.name}, got {result}"
            )
        else:
            assert result is None, (
                f"Expected None for unsupported {src.name}->{dst.name}, got {result}"
            )


class TestDtypeConversions(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """Parameterized Inductor tests for supported dtype conversions."""

    torch.manual_seed(0xAFFE)

    PARAMS = {
        ("test_conversion", "test_conversion_base"): {
            "ops_dict": SUPPORTED_OPS_DICT,
            "param_sets": make_param_dict(tuple((s,) for s in TEST_SHAPES)),
        },
    }

    def test_conversion_base(self, format_pair, x):
        """Execute and compare compiled Spyre conversion against CPU reference."""
        src_fmt, dst_fmt = format_pair
        src_dtype = FORMAT_TO_DTYPE[src_fmt]
        dst_dtype = FORMAT_TO_DTYPE[dst_fmt]

        x_src = x.to(dtype=src_dtype)

        # Determine tolerances based on precision targets
        # FP8 and BFLOAT16 typically require looser tolerances than FP32/FP16
        if dst_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            atol, rtol = (0.1, 0.1)
        elif dst_dtype == torch.bfloat16 or src_dtype == torch.bfloat16:
            atol, rtol = (1e-2, 1e-2)
        else:
            atol, rtol = (1e-3, 1e-2)

        compare_with_cpu(
            lambda a: a.to(dtype=dst_dtype),
            x_src,
            atol=atol,
            rtol=rtol,
            cpu_compile=True,
            run_eager=False,
        )


if __name__ == "__main__":
    unittest.main()

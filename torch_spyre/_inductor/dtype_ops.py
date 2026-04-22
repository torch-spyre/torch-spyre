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
Dtype conversion operator table for torch-spyre.

This module provides a centralized table for dtype conversion operators,
mapping PyTorch dtype pairs to Spyre hardware operators.
"""

from typing import Optional

import torch

from torch_spyre._inductor.constants import (
    IDENTITY_OP,
    # Deeptools type cast ops
    DL16TOFP32_OP,
    FP32TODL16_OP,
    # bf16 is identical to DL16 as per PR #1605
    # DL16TOBF16_OP,
    # FP8TODL16_OP,
    # TBD Deeptools type cast ops
    # FP32TOBF16_OP,
    # BF16TOFP32_OP,
    # FP32TOFP8_OP,
    # FP8TOFP32_OP,
    # BF16TODL16_OP,
    # INT32TOINT16_OP,
    # INT16TOINT32_OP,
    # INT32TOINT8_OP,
    # INT8TOINT32_OP,
)


class DtypeOpTable:
    _IDENTITY_CONVERSIONS = [
        (torch.float16, torch.bool),
        (torch.bool, torch.float16),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ]

    _FP16_TO_FP32_CONVERSIONS = [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ]

    _FP32_TO_FP16_CONVERSIONS = [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ]

    _CONVERSIONS = {
        **{pair: IDENTITY_OP for pair in _IDENTITY_CONVERSIONS},
        **{pair: FP32TODL16_OP for pair in _FP16_TO_FP32_CONVERSIONS},
        **{pair: DL16TOFP32_OP for pair in _FP32_TO_FP16_CONVERSIONS},
        # TBD Deeptools dtype cast ops
        # (torch.bfloat16, torch.float32): FP32TOBF16_OP,
        # (torch.float32, torch.bfloat16): BF16TOFP32_OP,
        # FP8 conversions (when supported)
        # (torch.float8_e4m3fn, torch.float32): FP32TOFP8_OP,
        # (torch.float8_e5m2, torch.float32): FP32TOFP8_OP,
        # (torch.float32, torch.float8_e4m3fn): FP8TOFP32_OP,
        # (torch.float32, torch.float8_e5m2): FP8TOFP32_OP,
        # (torch.float16, torch.float8_e4m3fn): FP8TODL16_OP,
        # (torch.float16, torch.float8_e5m2): FP8TODL16_OP,
        # Integer conversions (when supported)
        # (torch.int16, torch.int32): INT32TOINT16_OP,
        # (torch.int32, torch.int16): INT16TOINT32_OP,
        # (torch.int8, torch.int32): INT32TOINT8_OP,
        # (torch.int32, torch.int8): INT8TOINT32_OP,
    }

    @classmethod
    def get_operator(
        cls, src_dtype: torch.dtype, dst_dtype: torch.dtype
    ) -> Optional[str]:
        return cls._CONVERSIONS.get((src_dtype, dst_dtype))

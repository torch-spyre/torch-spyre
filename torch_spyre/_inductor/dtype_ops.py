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

from torch_spyre._C import DataFormats
from torch_spyre._inductor.constants import (
    # Deeptools type cast ops
    DL16TOFP32_OP,
    FP32TODL16_OP,
    DL16TOBF16_OP,
    FP8TODL16_OP,
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
    """Dtype conversion operator table.

    Maps (source_dtype, destination_dtype) pairs to Spyre hardware operator names.
    Provides methods to check support and retrieve stick sizes for conversions.
    """

    # Phase 0: Currently implemented conversions
    _CONVERSIONS = {
        # Deeptools dtype cast ops
        (DataFormats.IEEE_FP32, DataFormats.SEN169_FP16): FP32TODL16_OP,
        (DataFormats.SEN169_FP16, DataFormats.IEEE_FP32): DL16TOFP32_OP,
        (DataFormats.SEN169_FP16, DataFormats.BFLOAT16): DL16TOBF16_OP,
        (DataFormats.SEN143_FP8, DataFormats.SEN169_FP16): FP8TODL16_OP,
        (DataFormats.SEN152_FP8, DataFormats.SEN169_FP16): FP8TODL16_OP,
        # TBD Deeptools dtype cast ops
        # (DataFormats.IEEE_FP32, DataFormats.BFLOAT16): FP32TOBF16_OP,
        # (DataFormats.BFLOAT16, DataFormats.IEEE_FP32): BF16TOFP32_OP,
        # (DataFormats.IEEE_FP32, DataFormats.SEN143_FP8): FP32TOFP8_OP,
        # (DataFormats.IEEE_FP32, DataFormats.SEN152_FP8): FP32TOFP8_OP,
        # (DataFormats.SEN143_FP8, DataFormats.IEEE_FP32): FP8TOFP32_OP,
        # (DataFormats.SEN152_FP8, DataFormats.IEEE_FP32): FP8TOFP32_OP,
        # (DataFormats.BFLOAT16, DataFormats.SEN169_FP16): BF16TODL16_OP,
        # (DataFormats.IEEE_INT32, DataFormats.SENINT16): INT32TOINT16_OP,
        # (DataFormats.SENINT16, DataFormats.IEEE_INT32): INT16TOINT32_OP,
        # (DataFormats.IEEE_INT32, DataFormats.SENINT8): INT32TOINT8_OP,
        # (DataFormats.SENINT8, DataFormats.IEEE_INT32): INT8TOINT32_OP,
    }

    @classmethod
    def get_operator(
        cls, src_dtype: DataFormats, dst_dtype: DataFormats
    ) -> Optional[str]:
        return cls._CONVERSIONS.get((src_dtype, dst_dtype))

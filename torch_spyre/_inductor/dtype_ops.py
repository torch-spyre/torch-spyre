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

from typing import Mapping, Optional

import torch

from torch_spyre._C import DataFormats, get_device_dtype
from torch_spyre._inductor.constants import (
    IDENTITY_OP,
    DL16TOFP32_OP,
    FP32TODL16_OP,
    FP8TODL16_OP,
    FP32TOINT32_OP,
    INT32TOFP32_OP,
)
from torch_spyre._inductor.errors import Unsupported

# Spyre has no native bool: a bool tensor reuses whichever physical format
# produced it (e.g. fp16 vs fp32 comparison results). Maps that format to
# the logical dtype that natively uses it.
_BOOL_EQUIVALENT_DTYPES: Mapping[DataFormats, torch.dtype] = {
    DataFormats.SEN169_FP16: torch.float16,
    DataFormats.IEEE_FP32: torch.float32,
}


def bool_equivalent_dtype(device_dtype: DataFormats) -> Optional[torch.dtype]:
    """Logical dtype matching a bool's physical format, or None if unsupported."""
    return _BOOL_EQUIVALENT_DTYPES.get(device_dtype)


def bool_layout_dtype(
    device_dtype: DataFormats, context: str = "result"
) -> torch.dtype:
    """Return the logical dtype to build a bool tensor's layout with.

    ``get_elem_in_stick(torch.bool)`` hardcodes the SEN169_FP16 stick size of
    64, so any layout built from a bool's *logical* dtype silently assumes an
    fp16 format. Callers must instead resolve the stick size from the bool's
    real physical format via this function.

    Raises Unsupported if `device_dtype` has no bool-equivalent dtype.
    """
    dtype_for_layout = bool_equivalent_dtype(device_dtype)
    if dtype_for_layout is None:
        raise Unsupported(
            f"torch.bool {context} of operand with device format {device_dtype}"
        )
    return dtype_for_layout


def resolve_output_formats(
    output_dtype: torch.dtype,
    bool_device_dtype: Optional[DataFormats],
    context: str = "result",
) -> tuple[DataFormats, torch.dtype]:
    """Resolve an op output's ``(device_dtype, layout_dtype)`` pair.

    Non-bool outputs derive both from ``output_dtype``. For bool outputs,
    ``get_device_dtype(torch.bool)`` hardcodes SEN169_FP16 -- wrong for e.g. a
    float32 comparison result -- so the caller supplies the real on-device
    format in ``bool_device_dtype`` (read off the producing operand(s)).

    Callers needing only the layout dtype should call `bool_layout_dtype`
    directly rather than discarding half of this pair.
    """
    if output_dtype == torch.bool:
        assert bool_device_dtype is not None, "bool output needs bool_device_dtype"
        return bool_device_dtype, bool_layout_dtype(bool_device_dtype, context)
    return get_device_dtype(output_dtype), output_dtype


class DtypeOpTable:
    _IDENTITY_DTYPES = [
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bool),
        (torch.bfloat16, torch.bool),
        (torch.float32, torch.bool),
    ]

    _FP16_TO_FP32_DTYPES = [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ]

    _FP32_TO_FP16_DTYPES = [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ]

    _FP8_TO_FP16_DTYPES = [
        (torch.float8_e4m3fn, torch.float16),
    ]

    _FP32_TO_INT32_DTYPES = [
        (torch.float32, torch.int32),
    ]

    _INT32_TO_FP32_DTYPES = [
        (torch.int32, torch.float32),
    ]

    _TYPECAST_OPS_TABLE = {
        **{pair: IDENTITY_OP for pair in _IDENTITY_DTYPES},
        **{pair: DL16TOFP32_OP for pair in _FP16_TO_FP32_DTYPES},
        **{pair: FP32TODL16_OP for pair in _FP32_TO_FP16_DTYPES},
        **{pair: FP8TODL16_OP for pair in _FP8_TO_FP16_DTYPES},
        **{pair: FP32TOINT32_OP for pair in _FP32_TO_INT32_DTYPES},
        **{pair: INT32TOFP32_OP for pair in _INT32_TO_FP32_DTYPES},
    }

    _TYPECAST_OP_NAMES = set(_TYPECAST_OPS_TABLE.values())
    _TYPECAST_OP_DTYPES = set(_TYPECAST_OPS_TABLE.keys())

    # Typecast ops that change the physical stick element ordering, as opposed to
    # IDENTITY_OP (a same-width byte copy). Reading a reordering op's output back
    # over a DMA-copied host buffer with a different HBM ordering yields shuffled
    # elements, so host bool InputBuffers feeding one must fall back to CPU.
    _STICK_REORDERING_OPS = {DL16TOFP32_OP, FP32TODL16_OP, FP8TODL16_OP}

    @classmethod
    def get_operator(
        cls, src_dtype: torch.dtype, dst_dtype: torch.dtype
    ) -> Optional[str]:
        return cls._TYPECAST_OPS_TABLE.get((src_dtype, dst_dtype))

    @classmethod
    def get_bool_src_operator(
        cls, device_dtype: DataFormats, dst_dtype: torch.dtype
    ) -> Optional[str]:
        """Resolve the conversion op for a torch.bool source tensor.

        Treats the bool as its physically-equivalent dtype (from
        `device_dtype`) and resolves the op for that pair.
        """
        equivalent_src_dtype = bool_equivalent_dtype(device_dtype)
        if equivalent_src_dtype is None:
            return None
        if equivalent_src_dtype == dst_dtype:
            return IDENTITY_OP
        return cls.get_operator(equivalent_src_dtype, dst_dtype)

    @classmethod
    def bool_host_conversion_reorders_sticks(cls, dst_dtype: torch.dtype) -> bool:
        """Whether converting a host (fp16-format) bool to dst_dtype reorders sticks.

        Host bool InputBuffers are DMA-copied in the fp16 physical format
        (SEN169_FP16). A reordering typecast (e.g. DL16TOFP32 for bool->fp32)
        reads that DMA layout back shuffled, so the caller must fall back to CPU;
        an IDENTITY byte-copy (e.g. bool->fp16) is safe.
        """
        op = cls.get_bool_src_operator(DataFormats.SEN169_FP16, dst_dtype)
        return op in cls._STICK_REORDERING_OPS

    @classmethod
    def is_supported(cls, src_dtype: torch.dtype, dst_dtype: torch.dtype) -> bool:
        """Whether Spyre can natively perform this dtype conversion.

        For torch.bool sources this is a *static* gate with no access to the
        tensor's real device_dtype, so it accepts the pair if *some* physical
        format could convert to dst_dtype (see get_bool_src_operator). The
        codegen (spyre_kernel.to_dtype) instead resolves from the one real
        device_dtype.

        The looseness is structural, not an oversight: this runs from the
        convert_element_type lowering, which decides the CPU fallback before
        layout propagation has assigned any device_dtype. Keying on the real
        format would mean deferring that fallback decision to a later pass.

        This "any format" looseness is safe only while every dst reachable from
        a bool resolves to a supported op under *both* SEN169_FP16 (fp16) and
        IEEE_FP32 (fp32) -- true for the current table, so the predicate and its
        consumer never disagree. If a future op supports a dst under one bool
        equivalent but not the other, this could accept a conversion the codegen
        cannot then lower; tighten to key on the real device_dtype at that point.
        """
        if src_dtype == torch.bool:
            return any(
                cls.get_bool_src_operator(fmt, dst_dtype) is not None
                for fmt in _BOOL_EQUIVALENT_DTYPES
            )
        return cls.get_operator(src_dtype, dst_dtype) is not None

    @classmethod
    def get_table(
        cls,
    ) -> Mapping[tuple[torch.dtype, torch.dtype], str]:
        return cls._TYPECAST_OPS_TABLE

    @classmethod
    def get_dtype_pairs(cls) -> set[tuple[torch.dtype, torch.dtype]]:
        return cls._TYPECAST_OP_DTYPES

    @classmethod
    def is_dtype_op(cls, op: str) -> bool:
        return op in cls._TYPECAST_OP_NAMES

    @classmethod
    def op_names(cls) -> frozenset[str]:
        """All op names this table can produce (e.g. for op-name validation)."""
        return frozenset(cls._TYPECAST_OP_NAMES)

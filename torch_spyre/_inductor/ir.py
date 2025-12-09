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

from typing import Any, Callable, Optional, Sequence

from sympy import Expr
import torch
from torch._inductor.utils import ir_dataclass
from torch._inductor.ir import (
    FixedLayout,
    IRNode,
    Pointwise,
    Reduction,
    ReductionHint,
    TensorBox,
)
from torch_spyre._C import SpyreTensorLayout


@ir_dataclass
class SpyrePointwise(Pointwise):
    op_info: Any


@ir_dataclass
class SpyreReduction(Reduction):
    op_info: Any

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        reduction_type,
        op_info=None,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: Optional[IRNode] = None,
    ) -> TensorBox:
        return TensorBox.create(
            SpyreReduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=inner_fn,
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
                op_info=op_info,
            )
        )


class FixedTiledLayout(FixedLayout):
    device_layout: SpyreTensorLayout

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: list[Expr],
        stride: list[Expr],
        device_layout: SpyreTensorLayout,
    ) -> None:
        super().__init__(device, dtype, size, stride)
        self.device_layout = device_layout

    def __str__(self) -> str:
        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}, device_layout={self.device_layout})"
        )

    def get_allocation_size(self) -> list[Expr]:
        # TODO: Eventually this will include padding, etc.
        return self.size

    __repr__ = __str__

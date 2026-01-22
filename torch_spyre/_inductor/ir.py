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
    Reduction,
    ReductionHint,
    TensorBox,
)
from torch_spyre._C import SpyreTensorLayout


@ir_dataclass
class SpyreReduction(Reduction):
    """
    This class extends Reduction with an op_info to enable spyre-specific information
    to be passed from lowering to codegen for reduction operations.

    We believe this is needed because reduction operations do not go through the same
    virtualized ops API as pointwise operations do after lowering.
    TODO: validate this belief.
    """

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
    """
    A Tensor layout for a tensor that is on a Spyre device.
    It augments FixedLayout (the "host" tensor layout) with
    the device tensor layout and the information needed to map between them.
    """

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

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        """
        A closure containing math to read a given element.

        NOTE:   For the purposes of representing an access in the LoopLevelIR,
                we use a stride of 1 for the stick dimension.
                This is not true, because the sticks are actually tiled in memory.
                If we needed this indexer to compute the real offset in memory, the stick dimension
                compuation would actually need to be something like:
                    result = result + ((index[stick_dim] // 64) * stride[-2] + (index[stick_dim] % 64)
                However, all SpyreKernel needs from this indexer to be able to build a KernelSpec
                is for the indexer function to robustly capture the relationship between dim_map and
                the free variables in the index expression.
                By using a simpler expression it is easier to recover this relationship by stride-ordering the variables.
        """
        offset = self.offset
        stl = self.device_layout

        def indexer(index: Sequence[Expr]) -> Expr:
            for d in stl.dim_map:
                assert d < len(index)
            result = offset
            stick_dim = stl.dim_map[-1]
            for hd, stride in zip(stl.dim_map, stl.device_strides()):
                if hd != stick_dim:
                    result = result + (index[hd] * stride)
            result = result + index[stick_dim]  # stride of 1!
            return result

        return indexer

    __repr__ = __str__

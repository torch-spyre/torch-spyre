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

from typing import Optional

import sympy
from torch._inductor.codegen.wrapper import (
    BufferLike,
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.virtualized import V
from torch._inductor.sizevars import SizeVarAllocator
from .stickify import FixedTiledLayout


class SpyrePythonWrapperCodegen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
        V.graph.sizevars._simplify_loops_impl = noop_simplify_loops_impl.__get__(
            V.graph.sizevars, SizeVarAllocator
        )

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return SpyrePythonWrapperCodegen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            """
                from torch_spyre._inductor.runtime import ConstantArg, TensorArg, KernelSpec, UnimplementedOp
                from torch_spyre._inductor.runtime.async_compile import SpyreAsyncCompile
                from torch_spyre._C import SpyreTensorLayout, StickFormat, spyre_empty_with_layout
                import subprocess
            """,
            strip=True,
        )
        self.header.writeline("del async_compile")
        self.header.writeline("async_compile = SpyreAsyncCompile()")

    def make_buffer_allocation(self, buffer: BufferLike):
        layout = buffer.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            return super().make_buffer_allocation(buffer)

        name = buffer.get_name()
        codegen_shape_tuple = self.codegen_python_shape_tuple(tuple(layout.size))
        codegen_allocation_shape_tuple = self.codegen_python_shape_tuple(
            tuple(layout.get_allocation_size())
        )
        codegen_stride_tuple = self.codegen_python_shape_tuple(tuple(layout.stride))

        out = (
            f"{name} = spyre_empty_with_layout("
            f"{codegen_allocation_shape_tuple}, "
            f"{codegen_stride_tuple}, "
            f"{layout.dtype}, "
            f"{layout.device_layout!r})"
        )
        if codegen_shape_tuple != codegen_allocation_shape_tuple:
            out = out + f".as_strided({codegen_shape_tuple}, {codegen_stride_tuple})"

        return out


def noop_simplify_loops_impl(
    self, index_vars: list[sympy.Symbol], sizes, index_formulas
):
    """
    This is a noop implementation of SizeVarAllocator._simplify_loops_impl.

    We do this because the memory layout of tensors on the Spyre device is not
    entirely visible to Inductor.  Therefore Inductor's understanding of which
    tensor dimensions are actually contiguous is not accurate.
    """
    return sizes, lambda x: x, lambda x: x

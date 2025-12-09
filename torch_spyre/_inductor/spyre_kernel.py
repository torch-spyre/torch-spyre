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

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union
import regex as re

import torch
import sympy

from torch.utils._sympy.value_ranges import ValueRanges
from torch._inductor.codegen.common import (
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
)
from torch._inductor.codegen.simd import SIMDKernel
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import ReductionType, StoreMode, V


from .runtime import ConstantArg, TensorArg
from .constants import (
    MATMUL_REDUCTION_OP,
    TRANSPOSE_OP,
    SPYRE_FP32_OPS,
    BATCH_MATMUL_OP,
)
from . import Unsupported
from .opoverrides import SpyreKernelOverrides
from .opfuncs import UNIMPLEMENTED, get_spyre_op
from .ir import FixedTiledLayout


@dataclass
class TensorAccess:
    name: str
    index: sympy.Expr
    layout: FixedTiledLayout


@dataclass
class Constant:
    value: Union[bool, float, int]
    dtype: torch.dtype


@dataclass
class DimensionInfo:
    var: sympy.Symbol
    numel: int


@dataclass
class KernelSummary:
    op: list[DimensionInfo]
    scales: list[list[int]]
    arguments: list[TensorArg | ConstantArg]
    op_info: dict[str, Any]


def create_tensor_arg(
    is_input: bool, arg_index: int, layout: FixedTiledLayout
) -> TensorArg:
    return TensorArg(
        is_input, arg_index, layout.dtype, layout.size, layout.device_layout
    )


class SpyreKernelCSEVariable(CSEVariable):
    undefined_re = re.compile(r"\b(tmp\d+)\[\?\]")

    def __init__(
        self,
        name,
        bounds: ValueRanges[Any],
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(name, bounds, dtype)

    def update_on_args(self, name, args, kwargs):
        if name == "constant":
            V.kernel.compute_inputs.append(Constant(args[0], args[1]))
        else:
            V.kernel.record_compute_op(name, False)


class SpyreKernel(SIMDKernel[SpyreKernelCSEVariable]):
    overrides = SpyreKernelOverrides  # type: ignore[assignment]

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs,
    ) -> None:
        super().__init__(tiling, **kwargs)
        self.compute_op: str = ""
        self.compute_op_is_reduction = False
        self.spyre_op: str = ""
        self.op_info: dict[str, Any] = {}
        self.compute_inputs: list[TensorAccess | Constant] = []
        self.compute_output: Optional[TensorAccess] = None

    def create_cse_var(self, name, bounds=None, dtype=None):
        return SpyreKernelCSEVariable(name, bounds, dtype)

    def lookup_cse_var(self, name: str):
        return self.cse.varname_map[re.sub(r"\[.*", "", name)]

    def record_compute_op(self, op: str, is_reduction: bool):
        if V.kernel.compute_op != "":
            raise Unsupported(f"multi-op kernel: {V.kernel.compute_op} {op}")
        self.compute_op = op
        self.spyre_op = get_spyre_op(op)
        self.compute_op_is_reduction = is_reduction
        if hasattr(self.current_node.node.data, "op_info"):  # type: ignore[union-attr]
            self.op_info.update(self.current_node.node.data.op_info)  # type: ignore[union-attr]

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        self.compute_inputs.append(TensorAccess(name, index, layout))
        return self.cse.generate(self.body, f"{var}")

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: StoreMode = None,
    ) -> None:
        """Codegen a store to an OutputBuffer"""
        var = self.args.output(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        if self.compute_output is not None:
            raise Unsupported(f"multi-output kernel {self.compute_output.name} {name}")
        self.compute_output = TensorAccess(name, index, layout)
        self.body.writeline(DeferredLine(name, f"{var}"))

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        self.record_compute_op(reduction_type, True)
        return ()

    def get_strides(self, index: sympy.Expr) -> dict[sympy.Symbol, sympy.Expr]:
        """
        Compute the strides of the free variables in an index expression.
        """
        return {
            s: sympy_subs(index, {s: 1}) - sympy_subs(index, {s: 0})
            for s in index.free_symbols
        }

    def analyze_tensor_access(
        self,
        op_dimensions: Sequence[DimensionInfo],
        index: sympy.Expr,
    ) -> list[int]:
        """
        Return the scale implied by the given iteration space and indexing expression
        """
        return [1 if di.var in index.free_symbols else -1 for di in op_dimensions]

    def analyze_index_expr(self, index: sympy.Expr) -> list[DimensionInfo]:
        """
        Return the iteration space implied by the index expression
        """
        strides = self.get_strides(index)
        ordered_strides: Sequence[tuple[sympy.Symbol, sympy.Expr]] = sorted(
            strides.items(), key=lambda item: item[1], reverse=True
        )
        result = []
        var_ranges = self.var_ranges()
        for v, _ in ordered_strides:
            result.append(DimensionInfo(v, int(var_ranges[v])))
        return result

    def analyze_kernel(self) -> KernelSummary:
        """
        This method is called after the line-by-line codegen of the Kernel body has completed.
        That compilation has caused compute_op, compute_inputs, and compute_output to be populated.
        Using that information, we compute the iteration space of the op and build a KernelSummary.
        """
        if self.compute_output is None:
            raise Unsupported("kernel with no output")

        actuals = self.args.python_argdefs()[1]
        args: list[TensorArg | ConstantArg] = []
        scales = []
        if self.spyre_op == MATMUL_REDUCTION_OP:
            # MATMUL is specially constructed by our lowering operation.
            # It has exactly 2 tensor inputs and 1 tensor output.
            if (
                (not len(self.range_trees) == 2)
                or (not self.range_trees[0].name == "xindex")
                or (not len(self.range_trees[0].var_list) == 2)
                or (not self.range_trees[1].name == "r0_index")
                or (not len(self.range_trees[1].var_list) == 1)
            ):
                raise Unsupported(f"matmul range trees {self.range_trees}")
            idx_rt = self.range_trees[0]
            red_rt = self.range_trees[1]
            x_0 = idx_rt.var_list[0]
            x_1 = idx_rt.var_list[1]
            r_0 = red_rt.var_list[0]
            di = [
                DimensionInfo(x_1, int(idx_rt.var_ranges[x_1])),
                DimensionInfo(r_0, int(red_rt.var_ranges[r_0])),
                DimensionInfo(x_0, int(idx_rt.var_ranges[x_0])),
            ]

            for input in self.compute_inputs:
                if not isinstance(input, TensorAccess):
                    raise Unsupported(f"matmul: unsupported input {input}")
                scale = self.analyze_tensor_access(di, input.index)
                args.append(
                    create_tensor_arg(
                        True,
                        actuals.index(input.name),
                        input.layout,
                    )
                )
                scales.append(scale)
            scale = self.analyze_tensor_access(di, self.compute_output.index)
            args.append(
                create_tensor_arg(
                    False,
                    actuals.index(self.compute_output.name),
                    self.compute_output.layout,
                )
            )
            scales.append(scale)
            return KernelSummary(di, scales, args, self.op_info)
        elif self.spyre_op == BATCH_MATMUL_OP:
            # BATCH_MATMUL is specially constructed by our lowering operation.
            # It has exactly 2 tensor inputs and 1 tensor output.
            if (
                (not len(self.range_trees) == 2)
                or (not self.range_trees[0].name == "xindex")
                or (not len(self.range_trees[0].var_list) == 3)  # FIXME: generalize
                or (not self.range_trees[1].name == "r0_index")
                or (not len(self.range_trees[1].var_list) == 1)
            ):
                raise Unsupported(f"batchmatmul range trees {self.range_trees}")
            idx_rt = self.range_trees[0]
            red_rt = self.range_trees[1]
            x_0 = idx_rt.var_list[0]
            x_1 = idx_rt.var_list[1]
            x_2 = idx_rt.var_list[2]
            r_0 = red_rt.var_list[0]
            di = [
                DimensionInfo(x_2, int(idx_rt.var_ranges[x_2])),  # x
                DimensionInfo(x_1, int(idx_rt.var_ranges[x_1])),  # mb
                DimensionInfo(r_0, int(red_rt.var_ranges[r_0])),  # in
                DimensionInfo(x_0, int(idx_rt.var_ranges[x_0])),  # out
            ]
            args = []
            scales = []
            for input in self.compute_inputs:
                scale = self.analyze_tensor_access(di, input.index)  # type: ignore[union-attr]
                args.append(
                    create_tensor_arg(
                        True,
                        actuals.index(input.name),  # type: ignore[union-attr]
                        input.layout,  # type: ignore[union-attr]
                    )
                )
                scales.append(scale)
            scale = self.analyze_tensor_access(di, self.compute_output.index)
            args.append(
                create_tensor_arg(
                    False,
                    actuals.index(self.compute_output.name),
                    self.compute_output.layout,
                )
            )
            scales.append(scale)
            return KernelSummary(di, scales, args, self.op_info)
        elif self.compute_op_is_reduction:
            # Reductions are defined by the sole input's index
            if (not len(self.compute_inputs) == 1) or (
                not isinstance(self.compute_inputs[0], TensorAccess)
            ):
                raise Unsupported(f"reduction operands: {self.compute_inputs}")
            di = self.analyze_index_expr(self.compute_inputs[0].index)
            input = self.compute_inputs[0]
            scale = self.analyze_tensor_access(di, input.index)
            args.append(
                create_tensor_arg(
                    True,
                    actuals.index(input.name),
                    input.layout,
                )
            )
            scales.append(scale)
            scale = self.analyze_tensor_access(di, self.compute_output.index)
            args.append(
                create_tensor_arg(
                    False,
                    actuals.index(self.compute_output.name),
                    self.compute_output.layout,
                )
            )
            scales.append(scale)
            return KernelSummary(di, scales, args, self.op_info)
        elif not self.spyre_op == "":
            # Pointwise compute ops are defined by the output's index
            di = self.analyze_index_expr(self.compute_output.index)
            for input in self.compute_inputs:
                if isinstance(input, TensorAccess):
                    scale = self.analyze_tensor_access(di, input.index)
                    if self.compute_op == "layernormscale":
                        scale[-1] = -2
                    args.append(
                        create_tensor_arg(
                            True,
                            actuals.index(input.name),
                            input.layout,
                        )
                    )
                    scales.append(scale)
                else:
                    args.append(ConstantArg(input.value, input.dtype))
                    scales.append([-1] * len(di))
            scale = self.analyze_tensor_access(di, self.compute_output.index)
            args.append(
                create_tensor_arg(
                    False,
                    actuals.index(self.compute_output.name),
                    self.compute_output.layout,
                )
            )
            scales.append(scale)
            return KernelSummary(di, scales, args, self.op_info)
        else:
            # Reshapes and transposes take exactly one input and use both indexes
            if not len(self.compute_inputs) == 1:
                raise Unsupported(f"data op has {len(self.compute_inputs)} inputs")
            if not isinstance(self.compute_inputs[0], TensorAccess):
                raise Unsupported(f"data op unexpected input: {self.compute_inputs[0]}")
            self.spyre_op = TRANSPOSE_OP
            input_stride = list(
                self.get_strides(self.compute_inputs[0].index).values()
            )[0]
            output_stride = list(self.get_strides(self.compute_output.index).values())[
                0
            ]
            if input_stride == 64 and output_stride == 64:
                self.spyre_op = "swap"
            if input_stride == 64 and output_stride == 1:
                self.spyre_op = "slice"
            in_di = self.analyze_index_expr(self.compute_inputs[0].index)
            out_di = self.analyze_index_expr(self.compute_output.index)
            input = self.compute_inputs[0]
            scale = self.analyze_tensor_access(in_di, input.index)
            args.append(
                create_tensor_arg(True, actuals.index(input.name), input.layout)
            )
            scales.append(scale)
            scale = self.analyze_tensor_access(out_di, self.compute_output.index)
            args.append(
                create_tensor_arg(
                    False,
                    actuals.index(self.compute_output.name),
                    self.compute_output.layout,
                )
            )
            scales.append(scale)
            ks = KernelSummary(in_di, scales, args, self.op_info)
            if in_di != out_di:
                ks.op_info["transposed_dims"] = [
                    d for d in range(len(in_di)) if in_di[d].var != out_di[d].var
                ]
            return ks

    def codegen_kernel(self):
        """Codegen the body of this kernel by constructing its KernelSpec"""
        buf = IndentedBuffer()
        if self.spyre_op == UNIMPLEMENTED:
            buf.writeline(f"UnimplementedOp(op='{self.compute_op}')")
        else:
            ks = self.analyze_kernel()
            buf.writeline("KernelSpec(")
            with buf.indent():
                buf.writeline(f"op='{self.spyre_op}',")
                buf.writeline(f"is_reduction={self.compute_op_is_reduction},")
                buf.writeline(f"dimensions={[dmd.numel for dmd in ks.op]!r},")
                buf.writeline(f"scales={ks.scales!r},")
                buf.writeline(f"op_info={ks.op_info!r},")
                buf.writeline("args=[")
                with buf.indent():
                    for arg in ks.arguments:
                        buf.writeline(f"{arg!r},")
                        if (
                            arg.dtype == torch.float32
                            and self.spyre_op not in SPYRE_FP32_OPS
                        ):
                            raise Unsupported(f"{self.spyre_op} on {arg.dtype} dtype")
                        elif arg.dtype != torch.float16 and arg.dtype != torch.float32:
                            raise Unsupported(f"operations on {arg.dtype} dtype")
                buf.writeline("]")
            buf.writeline(")")

        return buf.getvalue()

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        call_args = []
        call_args.extend(self.args.python_argdefs()[1])
        call_args_str = ", ".join(call_args)
        wrapper.writeline(f"{name}.run({call_args_str})")

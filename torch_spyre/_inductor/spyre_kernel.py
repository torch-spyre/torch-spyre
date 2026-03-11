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

from dataclasses import dataclass, field
from typing import Any, Callable, Self, Sequence, Tuple, Union
from abc import ABC
from collections import Counter

import torch
import sympy

from torch_spyre._C import compute_view_layout

from torch._inductor.codegen.common import (
    CSEVariable,
    IndentedBuffer,
    Kernel,
)
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.codegen.simd import SIMDKernel
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import StoreMode, V

from torch_spyre.execution import OpSpec, TensorArg
from .constants import (
    MATMUL_REDUCTION_OP,
    SPYRE_FP32_OPS,
    BATCH_MATMUL_OP,
    TRANSPOSE_OP,
    CLONE_OP,
)
from .errors import Unsupported
from .ir import FixedTiledLayout
from .pass_utils import is_wildcard, map_dims_to_vars, wildcard_symbol
from .stickify import is_sparse
from .logging_utils import get_inductor_logger
import logging

logger = get_inductor_logger("spyre_kernel")


class RValue(ABC):
    """
    An RValue is an expression that can appear on the right hand side of an assignment.
    """


@dataclass
class TensorAccess(RValue):
    name: str
    index: sympy.Expr
    layout: FixedTiledLayout

    def unsqueeze_if_sparse(self):
        """
        If layout is sparse, construct a new layout that unsqueezes to a dense tensor
        """

        if is_sparse(self.layout.device_layout):
            new_size = self.layout.size + [1]
            new_stride = self.layout.stride + [1]
            new_stl = compute_view_layout(
                torch.Size(self.layout.size),
                torch.Size(new_size),
                self.layout.device_layout,
            )
            new_layout = FixedTiledLayout(
                self.layout.device, self.layout.dtype, new_size, new_stride, new_stl
            )
            return TensorAccess(self.name, self.index, new_layout)

        return self


@dataclass
class Constant(RValue):
    value: Union[bool, float, int]
    dtype: torch.dtype


@dataclass
class PointwiseOp(RValue):
    op: str
    arguments: list[RValue]
    op_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReductionOp(RValue):
    op: str
    arguments: list[RValue]
    op_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnimplementedOp(RValue):
    op: str


@dataclass(frozen=True)
class DimensionInfo:
    var: sympy.Symbol
    numel: int


class SpyreOpFuncs:
    """
    Pointwise torch ops that are directly supported by the backend compiler for the Spyre device.

    Keep these methods sorted in alphabetical order!
    """

    @staticmethod
    def abs(x):
        return PointwiseOp("abs", [x])

    @staticmethod
    def add(a, b):
        return PointwiseOp("add", [a, b])

    @staticmethod
    def clamp(x, min, max):
        op_info = {
            "constants": {
                "clipMin": min,
                "clipMax": max,
            }
        }
        return PointwiseOp("clip", [x], op_info)

    @staticmethod
    def eq(a, b):
        return PointwiseOp("equal", [a, b])

    @staticmethod
    def exp(x):
        return PointwiseOp("exp", [x])

    @staticmethod
    def exx2(a, b, c):
        return f"spyre.exx2({a} {b} {c})"

    @staticmethod
    def ge(a, b):
        return PointwiseOp("greaterequal", [a, b])

    @staticmethod
    def gelu(x):
        return PointwiseOp("gelufwd", [x])

    @staticmethod
    def layernormnorm(*args):
        return PointwiseOp("layernormnorm", list(args))

    @staticmethod
    def layernormscale(x, eps):
        op_info = {"constants": {"eps": eps}}
        return PointwiseOp("layernormscale", [x], op_info)

    @staticmethod
    def le(a, b):
        return PointwiseOp("lesserequal", [a, b])

    @staticmethod
    def log(x):
        return PointwiseOp("log", [x])

    @staticmethod
    def mul(a, b):
        return PointwiseOp("mul", [a, b])

    @staticmethod
    def ne(a, b):
        return PointwiseOp("notequal", [a, b])

    @staticmethod
    def neg(a):
        return PointwiseOp("neg", [a])

    @staticmethod
    def reciprocal(x):
        return PointwiseOp("reciprocal", [x])

    @staticmethod
    def relu(x):
        return PointwiseOp("relufwd", [x])

    @staticmethod
    def rsqrt(x):
        return PointwiseOp("rsqrt", [x])

    @staticmethod
    def slice(x):
        return PointwiseOp("slice", [x])

    @staticmethod
    def swap(x):
        return PointwiseOp("swap", [x])

    @staticmethod
    def sigmoid(x):
        return PointwiseOp("sigmoid", [x])

    @staticmethod
    def softplus(x, beta, threshold):
        op_info = {
            "constants": {
                "softplusBeta": beta,
                "softplusThresh": threshold,
            }
        }
        return PointwiseOp("softplus", [x], op_info)

    @staticmethod
    def sqrt(x):
        return PointwiseOp("sqrt", [x])

    @staticmethod
    def square(x):
        return PointwiseOp("mul", [x, x])

    @staticmethod
    def sub(a, b):
        return PointwiseOp("sub", [a, b])

    @staticmethod
    def tanh(x):
        return PointwiseOp("tanh", [x])

    @staticmethod
    def to_dtype(x, dtype, src_dtype):
        return PointwiseOp("to_dtype", [x])

    @staticmethod
    def truediv(a, b):
        return PointwiseOp("realdiv", [a, b])

    @staticmethod
    def where(x, y, z):
        return PointwiseOp("where3", [x, y, z])


class SpyreKernelOpsHandler(DefaultHandler):
    """
    This class plays the same role for SpyreKernel as common.CSEProxy does for SIMDKernel and Kernel.
    """

    name = "SpyreKernelOpsHandler"

    def __init__(self, kernel: Kernel[Any], parent_handler: SpyreOpFuncs):
        super().__init__()
        self.kernel = kernel
        self.parent_handler = parent_handler

    def _default(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> RValue:
        if hasattr(self.parent_handler, name):
            return getattr(self.parent_handler, name)(*args, **kwargs)
        else:
            return UnimplementedOp(name)

    def constant(self, value: Union[bool, float, int], dtype: torch.dtype) -> RValue:
        return Constant(value, dtype)

    def load(self, name: str, index: sympy.Expr) -> RValue:
        self.kernel.num_load += 1
        return self.kernel.load(name, index)

    def store(
        self, name: str, index: sympy.Expr, value: RValue, mode: StoreMode = None
    ) -> None:
        self.kernel.store_buffer_names.add(name)
        self.kernel.store(name, index, value, mode=mode)

    def store_reduction(
        self, name: str, index: sympy.Expr, value: ReductionOp | UnimplementedOp
    ) -> None:
        self.kernel.store_buffer_names.add(name)
        self.kernel.store_reduction(name, index, value)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: Union[RValue, tuple[RValue, ...]],
    ) -> RValue:
        self.kernel.num_reduction += 1
        if reduction_type in [
            "welford_reduce",
            "welford_combine",
            "any",
            "prod",
            "xor_sum",
        ]:
            return UnimplementedOp(reduction_type)
        elif isinstance(value, tuple):
            return ReductionOp(reduction_type, list(value))
        else:
            return ReductionOp(reduction_type, [value])

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[RValue, ...], tuple[RValue, ...]],
            tuple[RValue, ...],
        ],
        values: tuple[RValue, ...],
    ) -> tuple[RValue, ...]:
        raise NotImplementedError


def analyze_tensor_access(
    op_dimensions: Sequence[DimensionInfo],
    access: TensorAccess,
) -> list[int]:
    """
    Return the scale implied by the given iteration space and indexing expression
    """
    dim_map = map_dims_to_vars(access.layout, access.index)
    var_map = {v: k for k, v in dim_map.items()}

    # Special case: single dimension of size 1 is not elided by inductor
    if len(op_dimensions) == 1 and op_dimensions[0].numel == 1:
        return [access.layout.device_layout.dim_map[0]]

    return [var_map[di.var] if di.var in var_map else -1 for di in op_dimensions]


def create_op_spec(
    op: str,
    is_reduction: bool,
    dims: list[DimensionInfo],
    args: Sequence[TensorArg],
    op_info: dict[str, Any],
) -> OpSpec:
    for arg in args:
        if arg.dtype == torch.float32 and op not in SPYRE_FP32_OPS:
            raise Unsupported(f"{op} on {arg.dtype} dtype")
        elif arg.dtype not in [
            torch.bool,
            torch.float16,
            torch.float32,
            torch.int64,
        ]:
            raise Unsupported(f"operations on {arg.dtype} dtype")
    return OpSpec(op, is_reduction, [d.numel for d in dims], args, op_info)


class SpyreKernel(SIMDKernel[CSEVariable]):
    overrides = SpyreOpFuncs  # type: ignore[assignment]

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs,
    ) -> None:
        super().__init__(tiling, **kwargs)
        self.op_specs: list[OpSpec | UnimplementedOp] = []
        self.spyre_kernel_args: list[Tuple[str, TensorArg]] = []

    def __enter__(self) -> Self:
        super().__enter__()
        self.exit_stack.enter_context(
            V.set_ops_handler(SpyreKernelOpsHandler(self, SpyreOpFuncs()))
        )
        return self

    def create_tensor_arg(
        self, is_input: bool, name: str, tensor: TensorAccess, di: list[DimensionInfo]
    ) -> TensorArg:
        scales = analyze_tensor_access(di, tensor)
        tensor_arg = TensorArg(
            is_input,
            -1,
            tensor.layout.dtype,
            scales,
            tensor.layout.allocation,
            tensor.layout.device_layout,
        )
        self.spyre_kernel_args.append((name, tensor_arg))
        return tensor_arg

    def remove_kernel_local_buffers(self) -> None:
        """Do not remove kernel local buffers becasue we need the allocate in hbm/lx"""
        pass

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer"""
        _ = self.args.input(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"kernel_load: {name}, shape={[int(s) for s in layout.size]}, "
                f"device_size={list(layout.device_layout.device_size)}"
            )

        return TensorAccess(name, index, layout).unsqueeze_if_sparse()

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: RValue,
        mode: StoreMode = None,
    ) -> None:
        _ = self.args.output(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        dst = TensorAccess(name, index, layout).unsqueeze_if_sparse()
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        if real_dst_name != name:
            # Skip allocating an output buffer; this name is an alias to another buffer
            V.graph.removed_buffers.add(name)
        op_info = {}
        if hasattr(self.current_node, "op_dim_splits"):
            op_info["op_dim_splits"] = self.current_node.op_dim_splits  # type: ignore[union-attr]
        if hasattr(self.current_node, "n_cores_used"):
            op_info["n_cores_used"] = self.current_node.n_cores_used  # type: ignore[union-attr]

        if logger.isEnabledFor(logging.DEBUG):
            value_type = type(value).__name__
            logger.debug(
                f"kernel_store: {name} (type: {value_type}), shape={[int(s) for s in layout.size]}, "
                f"device_size={list(layout.device_layout.device_size)}, op_info={op_info}"
            )

        if isinstance(value, UnimplementedOp):
            self.op_specs.append(value)
        elif isinstance(value, PointwiseOp):
            # Pointwise compute ops are defined by the output's index
            di = self.derive_dim_info(dst)
            args: list[TensorArg] = []
            for input in value.arguments:
                if isinstance(input, TensorAccess):
                    args.append(self.create_tensor_arg(True, input.name, input, di))
                else:
                    raise Unsupported(f"unexpected argument {input} to {value.op}")
            args.append(self.create_tensor_arg(False, real_dst_name, dst, di))
            op_info.update(value.op_info)
            self.op_specs.append(create_op_spec(value.op, False, di, args, op_info))
        elif isinstance(value, TensorAccess):
            # Reshapes, transposes, and other dataops
            in_di = self.derive_dim_info(value)
            out_di = self.derive_dim_info(dst)

            args = [
                self.create_tensor_arg(True, value.name, value, in_di),
                self.create_tensor_arg(False, real_dst_name, dst, out_di),
            ]
            in_stl = args[0].device_layout  # type: ignore[union-attr]
            out_stl = args[1].device_layout  # type: ignore[union-attr]
            # Determine data op based on tensor args
            if (
                Counter(in_stl.dim_map) == Counter(out_stl.dim_map)
                and in_stl.device_size != out_stl.device_size
            ) or (Counter(in_di) == Counter(out_di) and in_di != out_di):
                # Transpose:
                #   - check that the input / output DimensionInfo are the same, but in different order.
                #   - check that the dim map has the same dimensions (no duplicate dimensions), but device size differs.
                op = TRANSPOSE_OP
            elif all(is_wildcard(d.var) for d in in_di) and not all(
                is_wildcard(d.var) for d in out_di
            ):
                # Broadcast: scalar input (all dims wildcards) expanding to non-scalar output.
                op = CLONE_OP
                in_di = out_di
                args[0] = self.create_tensor_arg(True, value.name, value, in_di)
            elif in_stl.device_size == out_stl.device_size:
                # Clone: check that device layout is the same.
                op = CLONE_OP
            else:
                # Unsupported data operation on TensorArg
                raise Unsupported(f"Data operation {args[0]})=>{args[1]}")

            op_spec = create_op_spec(op, False, in_di, args, op_info)
            if op == TRANSPOSE_OP:
                op_spec.op_info["transposed_dims"] = [
                    d for d in range(len(in_di)) if in_di[d] != out_di[d]
                ]
                # Reorder scale of the output  to implement transpositions
                (
                    op_spec.args[-1].it_dim_map[op_spec.op_info["transposed_dims"][0]],  # type: ignore[union-attr]
                    op_spec.args[-1].it_dim_map[op_spec.op_info["transposed_dims"][1]],  # type: ignore[union-attr]
                ) = (
                    op_spec.args[-1].it_dim_map[op_spec.op_info["transposed_dims"][1]],  # type: ignore[union-attr]
                    op_spec.args[-1].it_dim_map[op_spec.op_info["transposed_dims"][0]],  # type: ignore[union-attr]
                )
            self.op_specs.append(op_spec)
        else:
            raise Unsupported(f"store value of unexpected type {type(value)}")

    def store_reduction(
        self, name: str, index: sympy.Expr, value: ReductionOp | UnimplementedOp
    ) -> None:
        """Convert an RValue"""
        _ = self.args.output(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        dst = TensorAccess(name, index, layout)
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        if real_dst_name != name:
            # Skip allocating an output buffer; this name is an alias to another buffer
            V.graph.removed_buffers.add(name)

        if isinstance(value, UnimplementedOp):
            self.op_specs.append(value)
            return

        op_info = {}
        if hasattr(self.current_node.node.data, "op_info"):  # type: ignore[union-attr]
            op_info.update(self.current_node.node.data.op_info)  # type: ignore[union-attr]
        if hasattr(self.current_node, "op_dim_splits"):
            op_info["op_dim_splits"] = self.current_node.op_dim_splits  # type: ignore[union-attr]
        if hasattr(self.current_node, "n_cores_used"):
            op_info["n_cores_used"] = self.current_node.n_cores_used  # type: ignore[union-attr]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"kernel_store_reduction: {name} (op: {value.op}), shape={[int(s) for s in layout.size]}, "
                f"device_size={list(layout.device_layout.device_size)}, op_info={op_info}"
            )

        if value.op == MATMUL_REDUCTION_OP:
            if (
                len(value.arguments) != 2
                or (not isinstance(value.arguments[0], TensorAccess))
                or (not isinstance(value.arguments[1], TensorAccess))
            ):
                raise Unsupported(f"invalid matmul arguments {value.arguments}")
            x = value.arguments[0]
            y = value.arguments[1]
            di_x = self.derive_dim_info(x)
            di_y = self.derive_dim_info(y)
            if len(di_x) == 2 and len(di_y) == 2:
                di = [di_x[0], di_x[1], di_y[1]]
            elif len(di_x) == 1 and len(di_y) == 2:
                di = [di_x[0], DimensionInfo(wildcard_symbol(1), 1), di_y[1]]
                # TODO:  The OpSpec we generate is correct, but the SDSC we generate
                # will not compute the correct result.  Raise Unsupported to make this explicit.
                raise Unsupported(f"matmul requires padding support: {value.arguments}")
            elif len(di_x) == 2 and len(di_y) == 1:
                di = [di_x[0], di_x[1], DimensionInfo(wildcard_symbol(1), 1)]
                # TODO:  The OpSpec we generate is correct, but the SDSC we generate
                # will not compute the correct result.  Raise Unsupported to make this explicit.
                raise Unsupported(f"matmul requires padding support: {value.arguments}")
            else:
                raise Unsupported(f"degenerate matmul: {value.arguments}")
            args = [
                self.create_tensor_arg(True, x.name, x, di),
                self.create_tensor_arg(True, y.name, y, di),
                self.create_tensor_arg(False, real_dst_name, dst, di),
            ]
            self.op_specs.append(create_op_spec(value.op, True, di, args, op_info))
        elif value.op == BATCH_MATMUL_OP:
            if (
                len(value.arguments) != 2
                or (not isinstance(value.arguments[0], TensorAccess))
                or (not isinstance(value.arguments[1], TensorAccess))
            ):
                raise Unsupported(f"invalid batchmatmul arguments {value.arguments}")
            x = value.arguments[0]
            y = value.arguments[1]
            di_x = self.derive_dim_info(x)
            di_y = self.derive_dim_info(y)
            if len(di_x) == 4 and len(di_y) == 4:
                di = di_x[0:3] + di_y[2:]
            elif len(di_x) == 3 and len(di_y) == 3:
                di = di_x[0:3] + di_y[2:]
            elif len(di_x) == 2 and len(di_y) == 3:
                if di_x == di_y[0:2]:
                    di = [
                        di_x[0],
                        DimensionInfo(wildcard_symbol(1), 1),
                        di_x[1],
                        di_y[2],
                    ]
                elif di_x[0] == di_y[0]:
                    di = [
                        di_x[0],
                        di_x[1],
                        DimensionInfo(wildcard_symbol(1), 1),
                        di_y[2],
                    ]
                else:
                    di = [
                        DimensionInfo(wildcard_symbol(1), 1),
                        di_x[0],
                        di_x[1],
                        di_y[2],
                    ]
            elif len(di_x) == 3 and len(di_y) == 2:
                if di_x[:2] == di_y:
                    di = [di_x[0], di_x[1], di_x[2], DimensionInfo(self.wildcard, 1)]
                elif di_x[2] == di_y[0]:
                    di = [di_x[0], di_x[1], di_x[2], di_y[1]]
                else:
                    raise Unsupported(f"malformed bmm {di_x} {di_y}")
            elif len(di_x) == 2 and len(di_y) == 2:
                if di_x == di_y:
                    di = [
                        di_x[0],
                        DimensionInfo(wildcard_symbol(1), 1),
                        di_x[1],
                        DimensionInfo(wildcard_symbol(2), 1),
                    ]
                elif di_x[0] == di_y[0]:
                    di = [
                        di_x[0],
                        di_x[1],
                        DimensionInfo(wildcard_symbol(1), 1),
                        di_y[1],
                    ]
                else:
                    di = [
                        DimensionInfo(wildcard_symbol(1), 1),
                        di_x[0],
                        di_x[1],
                        di_y[1],
                    ]
            else:
                raise Unsupported(f"malformed bmm {di_x} {di_y}")
            args = [
                self.create_tensor_arg(True, x.name, x, di),
                self.create_tensor_arg(True, y.name, y, di),
                self.create_tensor_arg(False, real_dst_name, dst, di),
            ]
            self.op_specs.append(create_op_spec(value.op, True, di, args, op_info))
        else:
            # All other reductions have exactly one input which is a tensor
            if (not len(value.arguments) == 1) or (
                not isinstance(value.arguments[0], TensorAccess)
            ):
                raise Unsupported(f"reduction operands: {value.arguments}")
            x = value.arguments[0]
            di = self.derive_dim_info(x)
            args = [
                self.create_tensor_arg(True, x.name, x, di),
                self.create_tensor_arg(False, real_dst_name, dst, di),
            ]
            self.op_specs.append(create_op_spec(value.op, True, di, args, op_info))

    def derive_dim_info(self, access: TensorAccess) -> list[DimensionInfo]:
        """
        Return the iteration space implied by the tensor access
        """
        var_ranges = self.var_ranges()
        if var_ranges:
            dim_map = map_dims_to_vars(access.layout, access.index)
            return [
                DimensionInfo(dim_map[v], int(var_ranges.get(dim_map[v], 1)))
                for v in sorted(dim_map)
            ]
        else:
            return [DimensionInfo(wildcard_symbol(0), 1)]

    def codegen_kernel(self):
        """Codegen the body of this kernel by pretty printing its list of OpSpecs"""

        # Now that all loads/stores have been processed we know the final kernel_args and can map names to indices
        actuals = self.args.python_argdefs()[1]
        for name, tensor_arg in self.spyre_kernel_args:
            tensor_arg.arg_index = actuals.index(name)

        buf = IndentedBuffer()
        buf.writeline("[")
        with buf.indent():
            for op_spec in self.op_specs:
                if logger.isEnabledFor(logging.DEBUG):
                    if isinstance(op_spec, UnimplementedOp):
                        logger.debug(f"op_spec: UnimplementedOp({op_spec.op})")
                    else:
                        logger.debug(
                            f"op_spec: {op_spec.op}, is_reduction={op_spec.is_reduction}, "
                            f"iteration_space={op_spec.iteration_space}, op_info={op_spec.op_info}"
                        )

                if isinstance(op_spec, UnimplementedOp):
                    buf.writeline(f"UnimplementedOp(op='{op_spec.op}')")
                else:
                    buf.writeline("OpSpec(")
                    with buf.indent():
                        buf.writeline(f"op='{op_spec.op}',")
                        buf.writeline(f"is_reduction={op_spec.is_reduction},")
                        buf.writeline(f"iteration_space={op_spec.iteration_space!r},")
                        buf.writeline(f"op_info={op_spec.op_info!r},")
                        buf.writeline("args=[")
                        with buf.indent():
                            for arg in op_spec.args:
                                buf.writeline(f"{arg!r},")
                        buf.writeline("]")
                    buf.writeline("),")
        buf.writeline("]")
        return buf.getvalue()

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        call_args = []
        call_args.extend(self.args.python_argdefs()[1])
        call_args_str = ", ".join(call_args)
        wrapper.writeline(f"{name}.run({call_args_str})")

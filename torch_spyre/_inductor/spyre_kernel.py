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

import torch
import sympy

from torch_spyre._C import DataFormats

from torch._inductor.codegen.common import (
    CSEVariable,
    Kernel,
)
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch._inductor.ops_handler import DefaultHandler, StoreMode
from torch._inductor.utils import IndentedBuffer, sympy_index_symbol, sympy_subs
from torch._inductor.virtualized import V

from .constants import (
    SPYRE_FP32_OPS,
    BATCH_MATMUL_OP,
    BATCH_MATMUL_FP8_OP,
    IDENTITY_OP,
    RESTICKIFY_OP,
    SEGMENT_OFFSETS,
    SHARED_WEIGHT_UNIT_BMM_INFO_KEY,
)
from .errors import Unsupported
from .ir import FixedTiledLayout
from .pass_utils import (
    concretize_expr,
    concretize_index,
    apply_splits_from_index_coeff,
    iteration_space,
    indirect_access_subs_from_kernel,
)
from .views import compute_coordinates, align_tensors
from .logging_utils import get_inductor_logger
from .op_spec import (
    IndirectAccess,
    LoopSpec,
    OpSpec,
    TensorArg,
    UnimplementedOp as OpSpecUnimplementedOp,
)
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


def _preserve_shared_weight_unit_bmm_dim(
    op: str,
    it_space: dict[sympy.Symbol, tuple[sympy.Expr, int]],
    args: Sequence[TensorArg],
    op_info: dict[str, Any],
) -> dict[sympy.Symbol, tuple[sympy.Expr, int]]:
    # TensorArg layout is normalized in-place below to match the surrounding
    # OpSpec construction helpers.
    if SHARED_WEIGHT_UNIT_BMM_INFO_KEY not in op_info:
        return it_space
    if op not in [BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP]:
        return it_space
    if len(it_space) != 3 or len(args) < 3:
        return it_space
    info = op_info.get(SHARED_WEIGHT_UNIT_BMM_INFO_KEY)
    if not isinstance(info, dict) or info.get("batch_dim") != 0:
        return it_space

    unit_sym = sympy.Symbol("_spyre_bmm_unit")
    suffix = 0
    while unit_sym in it_space:
        suffix += 1
        unit_sym = sympy.Symbol(f"_spyre_bmm_unit_{suffix}")

    def _unit_indices(arg: TensorArg) -> list[int]:
        return [
            idx
            for idx, (size, coord) in enumerate(
                zip(arg.device_size[:-1], arg.device_coordinates[:-1])
            )
            if concretize_expr(size) == 1 and coord == 0
        ]

    target_args = (args[0], args[-1])
    unit_idxs_by_arg = [_unit_indices(arg) for arg in target_args]

    if all(len(unit_idxs) == 0 for unit_idxs in unit_idxs_by_arg):
        for arg in target_args:
            if len(arg.device_size) < 2:
                return it_space
            insert_at = len(arg.device_size) - 1
            arg.device_size.insert(insert_at, 1)
            arg.device_coordinates.insert(insert_at, sympy.S.Zero)
        unit_idxs_by_arg = [_unit_indices(arg) for arg in target_args]

    if not all(len(unit_idxs) == 1 for unit_idxs in unit_idxs_by_arg):
        return it_space

    rewrite_targets = [
        (arg, unit_idxs[0]) for arg, unit_idxs in zip(target_args, unit_idxs_by_arg)
    ]

    for arg, unit_idx in rewrite_targets:
        arg.device_coordinates[unit_idx] = unit_sym
        nonstick = list(range(len(arg.device_size) - 1))
        order = [unit_idx] + [i for i in reversed(nonstick) if i != unit_idx]
        order.append(len(arg.device_size) - 1)
        arg.device_size[:] = [arg.device_size[i] for i in order]
        arg.device_coordinates[:] = [arg.device_coordinates[i] for i in order]

    logger.info("Preserving shared-weight unit BMM dim %s", unit_sym)
    return {unit_sym: (sympy.S.One, 1), **it_space}


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


def _serialize_value(v):
    """Serialize a value for code generation, handling symbolic expressions.

    Produces valid Python source text that can appear in the generated kernel
    wrapper code.  All sympy expressions—including symbolic ones with free
    symbols—are concretized to Python ``int`` / ``float`` so the generated
    code never depends on sympy names (``Mul``, ``Float``, ``Pow``, etc.)
    being in scope.

    This is needed because ``op_info`` dicts may contain symbolic scalars
    (e.g. ``1.0 / s97``) that came from Inductor's symbolic analysis.

    TODO(issue#220): once SDSC generation produces symbolic JSON
    (``symbolDefinitions_``), this function should emit symbolic references
    rather than concretizing.
    """
    if isinstance(v, sympy.Integer):
        return repr(int(v))
    elif isinstance(v, sympy.Basic):
        # Concretize: first try direct float conversion for concrete numerics,
        # then fall back to substituting size_hints for symbolic expressions.
        if hasattr(v, "free_symbols") and v.free_symbols:
            # Substitute each symbol individually (size_hint handles simple
            # Symbol lookups reliably), then evaluate.  This works for float
            # expressions like 1.0/s97 where size_hint on the whole expression
            # might not handle the float division correctly.
            subs = {s: V.graph.sizevars.size_hint(s) for s in v.free_symbols}
            concrete = float(v.subs(subs))
            return repr(concrete)
        try:
            return repr(float(v))
        except (TypeError, ValueError):
            return repr(V.graph.sizevars.size_hint(v))
    elif isinstance(v, dict):
        items = ", ".join(f"{repr(k)}: {_serialize_value(val)}" for k, val in v.items())
        return "{" + items + "}"
    else:
        return repr(v)


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
    def floor(x):
        return PointwiseOp("floor", [x])

    @staticmethod
    def ge(a, b):
        return PointwiseOp("greaterequal", [a, b])

    @staticmethod
    def gelu(x):
        return PointwiseOp("gelufwd", [x])

    @staticmethod
    def gt(a, b):
        return PointwiseOp("greaterthan", [a, b])

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
    def logical_and(x, y):
        return PointwiseOp("mul", [x, y])

    @staticmethod
    def lt(a, b):
        return PointwiseOp("lesserthan", [a, b])

    @staticmethod
    def maximum(a, b):
        return PointwiseOp("maximum", [a, b])

    @staticmethod
    def minimum(a, b):
        return PointwiseOp("minimum", [a, b])

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
    def qfp8ch(x):
        return PointwiseOp("qfp8ch", [x])

    @staticmethod
    def relu(x):
        return PointwiseOp("relufwd", [x])

    @staticmethod
    def rsqrt(x):
        return PointwiseOp("rsqrt", [x])

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
        assert dtype != src_dtype

        op = DtypeOpTable.get_operator(src_dtype, dtype)
        if op is None:
            raise Unsupported(f"type conversion from {src_dtype} to {dtype}")

        return PointwiseOp(op, [x])

    @staticmethod
    def truediv(a, b):
        return PointwiseOp("realdiv", [a, b])

    @staticmethod
    def silu(a):
        return PointwiseOp("silu", [a])

    @staticmethod
    def where(x, y, z):
        return PointwiseOp("where3", [x, y, z])


class SpyreKernelOpsHandler(DefaultHandler):
    """
    This class plays the same role for SpyreKernel as common.CSEProxy does for Kernel.
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

    def indirect_indexing(
        self,
        index_var: Any,
        size: Any,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        if isinstance(index_var, TensorAccess):
            sym = sympy_index_symbol(f"indirect{self.kernel._indirect_var_count}")
            self.kernel._indirect_var_count += 1
            self.kernel.indirect_vars[sym] = index_var
            self.kernel.indirect_sizes[sym] = int(size)
            return sym
        return sympy_index_symbol(str(index_var))

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


class SpyreKernel(Kernel[CSEVariable]):
    overrides = SpyreOpFuncs  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.op_specs: list[OpSpec | UnimplementedOp | LoopSpec] = []
        self.spyre_kernel_args: list[Tuple[str, TensorArg]] = []
        self.indirect_vars: dict[sympy.Symbol, TensorAccess] = {}
        self.indirect_sizes: dict[sympy.Symbol, int] = {}
        self._indirect_var_count: int = 0

    def indirect_var_names(self) -> "frozenset[str] | None":
        if not self.indirect_vars:
            return None
        return frozenset(t.name for t in self.indirect_vars.values())

    def __enter__(self) -> Self:
        super().__enter__()
        self.exit_stack.enter_context(
            V.set_ops_handler(SpyreKernelOpsHandler(self, SpyreOpFuncs()))
        )
        return self

    def create_tensor_arg(
        self,
        is_input: bool,
        name: str,
        tensor: TensorAccess,
        opspec_name: "str | None" = None,
    ) -> TensorArg:
        it_space = iteration_space(self.current_node)
        # With dynamic=True the host index may contain symbolic strides
        # (e.g. x0*s1+x1).  Concretize size symbols so normalize_coordinates
        # can correctly isolate each loop variable's contribution.

        index = concretize_index(tensor.index, set(it_space.keys()))
        device_coords = compute_coordinates(
            tensor.layout.device_layout.device_size,
            tensor.layout.device_layout.stride_map,
            it_space,
            index,
            self.indirect_sizes,
        )
        tensor_arg = TensorArg(
            is_input,
            -1,
            tensor.layout.device_layout.device_dtype,
            tensor.layout.device_layout.device_size,
            device_coords,
            tensor.layout.allocation,
            per_tile_fixed=getattr(tensor.layout, "per_tile_fixed", False),
            name=opspec_name,
        )
        if (
            "lx" not in tensor.layout.allocation
            and "pool" not in tensor.layout.allocation
        ):
            self.spyre_kernel_args.append((name, tensor_arg))
        return tensor_arg

    def create_op_spec(
        self,
        op: str,
        is_reduction: bool,
        args: Sequence[TensorArg],
        op_info: dict[str, Any],
        indirect_var_names: "frozenset[str] | None" = None,
    ) -> OpSpec:
        from torch_spyre._inductor.constants import SPYRE_FP8_OPS

        for arg in args:
            if _is_indirect_index_arg(arg, indirect_var_names):
                continue
            # Check if operation supports the argument's dtype
            if not (
                op == IDENTITY_OP
                or DtypeOpTable.is_dtype_op(op)
                or (op in SPYRE_FP32_OPS and arg.device_dtype == DataFormats.IEEE_FP32)
                or arg.device_dtype == DataFormats.SEN169_FP16
                or (
                    op in SPYRE_FP8_OPS
                    and arg.device_dtype
                    in [DataFormats.SEN143_FP8, DataFormats.SEN152_FP8]
                )
            ):
                raise Unsupported(f"{op} on {arg.device_dtype}")

        it_space = iteration_space(self.current_node)

        ir_node = self.current_node.node  # ComputedBuffer
        work_division: dict[sympy.Symbol, int] = {}
        if hasattr(ir_node, "op_it_space_splits"):
            write_index = next(iter(self.current_node.read_writes.writes)).index
            read_index = next(iter(self.current_node.read_writes.reads)).index
            work_division = apply_splits_from_index_coeff(
                ir_node.op_it_space_splits,
                write_index,
                read_index,
                it_space,
            )

        it_space_extended = {
            k: (v, work_division.get(k, 1)) for k, v in it_space.items()
        }
        it_space_extended = _preserve_shared_weight_unit_bmm_dim(
            op, it_space_extended, args, op_info
        )

        # If this op is inside a coarse-tiling loop, identify which iteration-space
        # symbols are tiled by the enclosing loop(s).  loop_tiled_dims is a
        # list[list[int]] (nested multi-level, outermost first).  Flatten all
        # levels so that tiled_symbols covers every loop variable from outermost
        # to innermost — matching the loop_vars ordering in bundle.py _emit_specs.
        li = getattr(ir_node, "loop_info", None)
        raw_tiled_dims: list[list[int]] = li.loop_tiled_dims if li is not None else []
        raw_tiled_red_dims: list[list[int]] = (
            li.loop_tiled_reduction_dims if li is not None else []
        )
        all_tiled_dims = [d for level in raw_tiled_dims for d in level]
        all_tiled_red_dims = [d for level in raw_tiled_red_dims for d in level]
        it_space_keys = list(it_space.keys())
        tiled_syms = [
            it_space_keys[i] for i in all_tiled_dims if i < len(it_space_keys)
        ]
        # For reduction ops tiled over a reduction dimension, it_space (from
        # reads.ranges) has output-dim symbols first, then reduction-dim symbols.
        # loop_tiled_reduction_dims indices are 0-based into the reduction portion,
        # so offset them by the number of output-space symbols.
        if all_tiled_red_dims:
            write_dep = next(iter(self.current_node.read_writes.writes), None)
            n_output_syms = len(write_dep.ranges) if write_dep is not None else 0
            tiled_syms += [
                it_space_keys[n_output_syms + r]
                for r in all_tiled_red_dims
                if n_output_syms + r < len(it_space_keys)
            ]

        return OpSpec(
            op,
            is_reduction,
            it_space_extended,
            args,
            op_info,
            tiled_symbols=tiled_syms,
        )

    def remove_kernel_local_buffers(self) -> None:
        """Remove buffers that have a scratchpad or temporary allocation from the kernel's arg list."""
        for name in list(self.store_buffer_names):
            buf = V.graph.get_buffer(name)
            if buf is None:
                continue
            layout = buf.get_layout()
            if isinstance(layout, FixedTiledLayout) and (
                "lx" in layout.allocation or "pool" in layout.allocation
            ):
                self.remove_buffer(name)

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer"""
        scheduler = getattr(V.graph, "scheduler", None)
        if scheduler is not None:
            name = scheduler.mutation_real_name.get(name, name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        if "lx" not in layout.allocation and "pool" not in layout.allocation:
            _ = self.args.input(name)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"kernel_load: {name}, shape={[concretize_expr(s) for s in layout.size]}, "
                f"device_size={list(layout.device_layout.device_size)}"
            )

        return TensorAccess(name, index, layout)

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: RValue,
        mode: StoreMode = None,
    ) -> None:
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        # Pool buffers are intermediates whose address is baked into the TensorArg
        # allocation dict; registering them as outputs would overflow SEGMENT_OFFSETS.
        # (lx buffers are already excluded from spyre_kernel_args in _tensor_arg.)
        if "pool" not in layout.allocation:
            _ = self.args.output(name)
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        dst = TensorAccess(name, index, layout)
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        if real_dst_name != name:
            # Skip allocating an output buffer; this name is an alias to another buffer
            V.graph.removed_buffers.add(name)
        op_info: dict[str, Any] = {}
        if logger.isEnabledFor(logging.DEBUG):
            value_type = type(value).__name__
            logger.debug(
                f"kernel_store: {name} (type: {value_type}), shape={[concretize_expr(s) for s in layout.size]}, "
                f"device_size={list(layout.device_layout.device_size)}, op_info={op_info}"
            )

        if isinstance(value, UnimplementedOp):
            self.op_specs.append(value)
        elif isinstance(value, PointwiseOp):
            # Pointwise compute ops
            args: list[TensorArg] = []
            indirect_syms = _indirect_syms_used(value, self.indirect_vars)
            if indirect_syms:
                args += [
                    self.create_tensor_arg(
                        True,
                        idx_tensor.name,
                        idx_tensor,
                        opspec_name=idx_tensor.name,
                    )
                    for sym in sorted(indirect_syms, key=str)
                    for idx_tensor in [self.indirect_vars[sym]]
                ]
            for input in value.arguments:
                if isinstance(input, TensorAccess):
                    args.append(self.create_tensor_arg(True, input.name, input))
                else:
                    raise Unsupported(f"unexpected argument {input} to {value.op}")
            args.append(self.create_tensor_arg(False, real_dst_name, dst))
            op_info.update(value.op_info)
            self.op_specs.append(
                self.create_op_spec(
                    value.op, False, args, op_info, self.indirect_var_names()
                )
            )
        elif isinstance(value, TensorAccess):
            # Reshapes, transposes, and other dataops.
            if self.indirect_vars:
                # Gather/scatter: coordinates are built with raw indirect symbols here;
                # indirect_access_subs is applied later in codegen_kernel → simplify_op_spec.
                # TODO: scatter codegen (IndirectAccess on output TensorArg → SuperDSC) not yet wired up.
                args = [
                    self.create_tensor_arg(
                        True,
                        idx_tensor.name,
                        idx_tensor,
                        opspec_name=idx_tensor.name,
                    )
                    for idx_tensor in sorted(
                        self.indirect_vars.values(),
                        key=lambda t: t.name,
                    )
                ]
                args += [
                    self.create_tensor_arg(True, value.name, value),
                    self.create_tensor_arg(False, real_dst_name, dst),
                ]
            else:
                args = [
                    self.create_tensor_arg(True, value.name, value),
                    self.create_tensor_arg(False, real_dst_name, dst),
                ]
            in_coords = args[-2].device_coordinates
            out_coords = args[-1].device_coordinates
            if all(e == 0 for e in in_coords) and not all(e == 0 for e in out_coords):
                # Broadcast: scalar input expanding to non-scalar output.
                op = IDENTITY_OP
            elif in_coords[-1].free_symbols != out_coords[-1].free_symbols:
                op = RESTICKIFY_OP
            else:
                op = IDENTITY_OP
            op_spec = self.create_op_spec(
                op, False, args, op_info, self.indirect_var_names()
            )
            self.op_specs.append(op_spec)
        else:
            raise Unsupported(f"store value of unexpected type {type(value)}")

    def store_reduction(
        self, name: str, index: sympy.Expr, value: ReductionOp | UnimplementedOp
    ) -> None:
        """Convert an RValue"""
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        # Pool buffers are intermediates whose address is baked into the TensorArg
        # allocation dict; registering them as outputs would overflow SEGMENT_OFFSETS.
        # (lx buffers are already excluded from spyre_kernel_args in _tensor_arg.)
        if "pool" not in layout.allocation:
            _ = self.args.output(name)
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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"kernel_store_reduction: {name} (op: {value.op}), shape={[concretize_expr(s) for s in layout.size]}, "
                f"device_size={list(layout.device_layout.device_size)}, op_info={op_info}"
            )

        if value.op in [BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP]:
            if (
                len(value.arguments) != 2
                or (not isinstance(value.arguments[0], TensorAccess))
                or (not isinstance(value.arguments[1], TensorAccess))
            ):
                raise Unsupported(f"invalid {value.op} arguments {value.arguments}")
            x = value.arguments[0]
            y = value.arguments[1]
            args = [
                self.create_tensor_arg(True, x.name, x),
                self.create_tensor_arg(True, y.name, y),
                self.create_tensor_arg(False, real_dst_name, dst),
            ]
            self.op_specs.append(self.create_op_spec(value.op, True, args, op_info))
        else:
            # All other reductions have exactly one input which is a tensor
            if (not len(value.arguments) == 1) or (
                not isinstance(value.arguments[0], TensorAccess)
            ):
                raise Unsupported(f"reduction operands: {value.arguments}")
            x = value.arguments[0]
            args = [
                self.create_tensor_arg(True, x.name, x),
                self.create_tensor_arg(False, real_dst_name, dst),
            ]
            self.op_specs.append(self.create_op_spec(value.op, True, args, op_info))

    def wrap_op_specs_in_loop(
        self, count: sympy.Expr, tiled_symbols: list | None = None
    ) -> None:
        """Replace the current op_specs list with a single LoopSpec of the given count."""
        body = self.op_specs
        self.op_specs = [
            LoopSpec(
                count=count,
                body=body,
                tiled_symbols=tiled_symbols if tiled_symbols is not None else [],
            )
        ]

    def codegen_kernel(self):
        """Codegen the body of this kernel by pretty printing its list of OpSpecs"""

        indirect_access_subs = (
            indirect_access_subs_from_kernel(self.indirect_vars)
            if self.indirect_vars
            else None
        )
        for op_spec in _iter_op_specs(self.op_specs):
            simplify_op_spec(op_spec, self.indirect_sizes, indirect_access_subs)

        def sympy_str(x: sympy.Expr) -> str:
            if isinstance(x, IndirectAccess):
                name_sym = x.args[0]
                return f"IndirectAccess('{name_sym}')"
            return "sympify('" + str(x) + "')"

        # Now that all loads/stores have been processed we know the final kernel_args and can map names to indices
        actuals = self.args.python_argdefs()[1]
        pool_size = getattr(V.graph, "pool_size", 0)
        has_pool_allocations = pool_size > 0

        for name, tensor_arg in self.spyre_kernel_args:
            tensor_arg.arg_index = actuals.index(name)
            tensor_arg.allocation["hbm"] = SEGMENT_OFFSETS[
                tensor_arg.arg_index + 1
                if has_pool_allocations
                else tensor_arg.arg_index
            ]

        buf = IndentedBuffer()
        buf.writeline("[")
        with buf.indent():
            _codegen_op_spec_list(self.op_specs, buf, sympy_str)
        buf.writeline("]")
        return buf.getvalue()

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        call_args = []

        if getattr(V.graph, "pool_size", 0) > 0:
            call_args.append("_pool")

        # Add remaining kernel arguments
        call_args.extend(self.args.python_argdefs()[1])

        call_args_str = ", ".join(call_args)
        wrapper.writeline(f"{name}.run({call_args_str})")


def _indirect_syms_used(
    value: "PointwiseOp", indirect_vars: "dict[sympy.Symbol, TensorAccess]"
) -> "set[sympy.Symbol]":
    """Return the subset of indirect_vars keys that appear in value's argument indices."""
    return {
        s
        for inp in value.arguments
        if isinstance(inp, TensorAccess)
        for s in inp.index.free_symbols
        if s in indirect_vars
    }


def _is_indirect_index_arg(
    arg: TensorArg, indirect_var_names: "frozenset[str] | None"
) -> bool:
    """Return True if arg is an indirect index tensor (i.e. a gather index buffer).

    Uses the kernel-level indirect_var_names set, which is populated before
    create_op_spec is called and is always ground truth regardless of whether
    IndirectAccess substitution has run.
    """
    return arg.name is not None and bool(
        indirect_var_names and arg.name in indirect_var_names
    )


def _iter_op_specs(specs):
    """Yield every OpSpec in a (possibly nested) op-spec list, depth-first."""
    for item in specs:
        if isinstance(item, LoopSpec):
            yield from _iter_op_specs(item.body)
        elif isinstance(item, OpSpec):
            yield item


def _codegen_op_spec_list(specs, buf: IndentedBuffer, sympy_str) -> None:
    """Emit Python source for a list of OpSpec / UnimplementedOp / LoopSpec entries."""
    for op_spec in specs:
        if isinstance(op_spec, LoopSpec):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"op_spec: LoopSpec(count={op_spec.count})")
            buf.writeline("LoopSpec(")
            with buf.indent():
                buf.writeline(f"count={sympy_str(op_spec.count)},")
                buf.writeline("body=[")
                with buf.indent():
                    _codegen_op_spec_list(op_spec.body, buf, sympy_str)
                buf.writeline("],")
                if op_spec.tiled_symbols:
                    buf.writeline(
                        "tiled_symbols=["
                        + ", ".join(sympy_str(s) for s in op_spec.tiled_symbols)
                        + "],"
                    )
            buf.writeline("),")
        elif isinstance(op_spec, (UnimplementedOp, OpSpecUnimplementedOp)):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"op_spec: UnimplementedOp({op_spec.op})")
            buf.writeline(f"UnimplementedOp(op='{op_spec.op}')")
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"op_spec: {op_spec.op}, is_reduction={op_spec.is_reduction}, "
                    f"iteration_space={op_spec.iteration_space}, op_info={op_spec.op_info}"
                )
            buf.writeline("OpSpec(")
            with buf.indent():
                buf.writeline(f"op='{op_spec.op}',")
                buf.writeline(f"is_reduction={op_spec.is_reduction},")
                buf.writeline(
                    "iteration_space={"
                    + ", ".join(
                        [
                            sympy_str(k)
                            + ": ("
                            + sympy_str(v[0])
                            + ", "
                            + str(v[1])
                            + ")"
                            for k, v in op_spec.iteration_space.items()
                        ]
                    )
                    + "},"
                )
                buf.writeline(f"op_info={_serialize_value(op_spec.op_info)},")
                if op_spec.tiled_symbols:
                    buf.writeline(
                        "tiled_symbols=["
                        + ", ".join(sympy_str(s) for s in op_spec.tiled_symbols)
                        + "],"
                    )
                buf.writeline("args=[")
                with buf.indent():
                    for arg in op_spec.args:
                        buf.writeline("TensorArg(")
                        with buf.indent():
                            buf.writeline(
                                f"is_input={arg.is_input}, arg_index={arg.arg_index}, device_dtype={arg.device_dtype},"
                            )
                            buf.writeline(f"device_size={arg.device_size},")
                            buf.writeline(
                                "device_coordinates=["
                                + ", ".join(
                                    [sympy_str(e) for e in arg.device_coordinates]
                                )
                                + "],"
                            )
                            buf.writeline(f"allocation={arg.allocation!r},")
                            if arg.per_tile_fixed:
                                buf.writeline("per_tile_fixed=True,")
                            if arg.name is not None:
                                buf.writeline(f"name={arg.name!r},")
                        buf.writeline("),")
                buf.writeline("]")
            buf.writeline("),")


def simplify_op_spec(op_spec, indirect_sizes=None, indirect_access_subs=None):
    # Both parameters must be provided together for gather kernels — indirect_sizes
    # decomposes symbols in align_tensors; indirect_access_subs replaces them with IndirectAccess.
    it_space = op_spec.iteration_space

    new_op_space_splits, new_tensors = align_tensors(
        it_space,
        [
            {"size": arg.device_size, "coordinates": arg.device_coordinates}
            for arg in op_spec.args
        ],
        indirect_sizes,
    )
    op_spec.iteration_space = new_op_space_splits

    for arg, t in zip(op_spec.args, new_tensors):
        arg.device_size = t["size"]
        arg.device_coordinates = t["coordinates"]

        # Apply indirect_access_subs after align_tensors, so that indirect symbols
        # are decomposed as regular variables before substitution.
        if indirect_access_subs:
            arg.device_coordinates = [
                c.xreplace(indirect_access_subs) for c in arg.device_coordinates
            ]

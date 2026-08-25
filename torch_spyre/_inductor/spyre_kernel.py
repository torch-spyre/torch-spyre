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
import math
from typing import Any, Callable, Self, Sequence, Tuple, Union
from abc import ABC
import itertools

import torch
import sympy

from torch_spyre._C import DataFormats, ElementArrangement

from torch._inductor.codegen.common import (
    CSEVariable,
    Kernel,
)
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ops_handler import DefaultHandler, StoreMode
from torch._inductor.utils import IndentedBuffer, sympy_index_symbol, sympy_subs
from torch._inductor.virtualized import V


from .constants import (
    SPYRE_FP32_OPS,
    BATCH_MATMUL_OP,
    BATCH_MATMUL_FP8_OP,
    CONV_OPS,
    DEPTHWISE_CONV_REDUCTION_OPS,
    IDENTITY_OP,
    POOL_OPS,
    RESTICKIFY_OP,
    SEGMENT_OFFSETS,
    TWO_INPUT_REDUCTION_OPS,
    SHARED_WEIGHT_UNIT_BMM_INFO_KEY,
)
from . import config as _spyre_config
from .errors import Unsupported
from .ir import FixedTiledLayout
from .scratchpad.lx_relayout import work_division_from_view
from .pass_utils import (
    concretize_expr,
    concretize_index,
    compute_symbolic_bounds,
    finite_upper_or_none,
    apply_splits_from_index_coeff,
    iteration_space,
    indirect_access_subs_from_kernel,
    is_restickify_coords,
)
from .views import compute_coordinates, align_tensors, tiling_expr_to_device_expr
from .logging_utils import get_inductor_logger
from .op_spec import (
    IndirectAccess,
    LoopSpec,
    OpSpec,
    TensorArg,
    TensorWorkDivision,
    UnimplementedOp as OpSpecUnimplementedOp,
    format_op_spec_list,
    is_lx_relayout_identity,
)
from .op_spec_validation import validate_op_specs
from torch_spyre._inductor.provenance import build_debug_handle
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
    # The unit-BMM marker is only valid for plain rank-3 BMMs after layout
    # construction. If a target tensor still carries extra physical axes, such
    # as attention heads from SDPA, rewriting one axis into the BMM iteration
    # space can produce an illegal SDSC layout.
    if any(len(arg.device_size) > 4 for arg in target_args):
        return it_space
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
        # then fall back to substituting hints for symbolic expressions.
        if hasattr(v, "free_symbols") and v.free_symbols:
            # Substitute each symbol individually (guarding_hint_or_throw handles
            # simple Symbol lookups reliably), then evaluate.  This works for float
            # expressions like 1.0/s97 where a hint on the whole expression might
            # not handle the float division correctly.  This value is baked into
            # generated kernel source, so use the strict hint: resolve backed
            # symbols to their true value and raise on unbacked ones rather than
            # emitting an optimization fallback.
            subs = {
                s: V.graph.sizevars.guarding_hint_or_throw(s) for s in v.free_symbols
            }
            concrete = float(v.subs(subs))
            return repr(concrete)
        try:
            return repr(float(v))
        except (TypeError, ValueError):
            return repr(V.graph.sizevars.guarding_hint_or_throw(v))
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
    def qfp8wt(x):
        return PointwiseOp("qfp8wt", [x])

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
    def to_dtype(x, dtype, src_dtype, use_compute_types=False):
        # PT 2.12 passes a new `use_compute_types` kwarg through OpsHandler.
        # Spyre maps directly to fixed hardware ops via DtypeOpTable and
        # cannot honor compute-type promotion, so accept and ignore.
        assert dtype != src_dtype

        if src_dtype == torch.bool:
            # A bool's physical format (fp16 vs fp32) depends on how it was
            # produced, so resolve the op from its propagated device dtype.
            op = DtypeOpTable.get_bool_src_operator(
                x.layout.device_layout.device_dtype, dtype
            )
        else:
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


def _tile_advance_expr_from_dep(
    dep: MemoryDep,
    tiled_dim_extents: dict[int, sympy.Expr],
) -> sympy.Expr:
    """Element-offset advance of ``dep`` for one step of each tiled dim.

    ``tiled_dim_extents`` maps a host-range positional index (the same
    indices used in ``loop_tiled_dims`` / ``loop_tiled_reduction_dims``) to
    the tile extent one loop step advances that dim's ``d{i}`` symbol by
    (the product of ``loop_count`` for every level tiling that dim).

    Built by substituting, in ``dep.index``, each tiled dim's ``d{dim}``
    with ``extent * d{dim}`` (staying symbolic in ``d{dim}`` -- the
    expression is meant to stay unevaluated until a later compilation stage
    substitutes a concrete tile-index value for each ``d{dim}``) and every
    other free symbol in ``dep.index`` with ``0`` (dims that never advance:
    untiled dims, and broadcast dims ``dep`` does not depend on).  Because
    this is a direct substitution rather than coefficient extraction,
    ``dep.index`` need not be affine in the tiled symbols -- a tiled dim
    wrapped in ``Mod``/``FloorDiv``/``ModularIndexing`` (reshape-split or
    gather/indirect-indexing dims; ``_loop_var_to_ranges_pos`` only checks
    for a single free symbol, so such a dim still reaches this function)
    produces the exact non-linear term rather than an approximation.
    Returns ``sympy.Integer(0)`` when every tiled dim's substituted term
    also evaluates to ``0`` (this dependency never advances).

    Moved here (from coarse_tile.py) so index substitution happens once
    the tensor's get_read_writes().index is guaranteed final -- capturing
    the advance expression any earlier risks reading a stale, not-yet-final
    index.
    """
    subs = {
        sympy_index_symbol(f"d{dim}"): extent * sympy_index_symbol(f"d{dim}")
        for dim, extent in tiled_dim_extents.items()
    }
    subs.update(
        {sym: sympy.Integer(0) for sym in dep.index.free_symbols if sym not in subs}
    )
    return dep.index.subs(subs)


class SpyreKernel(Kernel[CSEVariable]):
    overrides = SpyreOpFuncs  # type: ignore[assignment]

    def __init__(self, pool_size: int = 0) -> None:
        super().__init__()
        self.op_specs: list[OpSpec | UnimplementedOp | LoopSpec] = []
        self.spyre_kernel_args: list[Tuple[str, TensorArg]] = []
        self.indirect_vars: dict[sympy.Symbol, TensorAccess] = {}
        self.indirect_sizes: dict[sympy.Symbol, int] = {}
        self._indirect_var_count: int = 0
        self._general_tile_advance_seen: dict[str, int] = {}
        self._tile_advance_symbols: dict[int, sympy.Symbol] = {}
        self.pool_size: int = pool_size

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

    def _get_or_mint_level_symbol(self, level_idx: int, op_name: str) -> sympy.Symbol:
        """Fresh, distinct symbol for one nesting level of the current op.

        Minted once per (op, level) and cached in self._tile_advance_symbols
        (reset at the start of every op by store()/store_reduction()) so
        every TensorArg built for this op, plus create_op_spec's own
        tiled_symbols/tiled_symbol_trip_counts, resolve to the exact same
        Symbol object for a given level -- required for
        device_tile_advance_expr.coeff(sym) lookups in compute_ops.py /
        superdsc.py to find the term each level contributed.

        Distinct from the real Inductor d{i} symbols by construction (the
        name embeds the op name and level index, which never collides with
        Inductor's own "d0", "d1", ... convention) -- this is what fixes
        the same-real-symbol-tiled-at-2+-levels collapse bug: two levels
        tiling the same host dim get two different minted symbols here,
        so their sympy.Add terms never combine into one coefficient.
        """
        if level_idx not in self._tile_advance_symbols:
            self._tile_advance_symbols[level_idx] = sympy.Symbol(
                f"_tile_adv_{op_name}_lvl{level_idx}"
            )
        return self._tile_advance_symbols[level_idx]

    @staticmethod
    def _host_dim_to_index_symbol(ir_node: Any, dim: int) -> sympy.Symbol:
        """Map a host-range positional dim index to its dep.index d{i} symbol.

        CoarseTileInfo.tiled_dims_per_read/output_tiled_dims store *host-range*
        positional indices (indices into op.data.ranges, or
        n_output_dims + reduction_pos for reduction dims -- see loop_info.py).
        But dep.index's own free symbols are minted by Inductor's
        extract_read_writes -> index_vars_squeeze, which -- via
        SqueezeView.squeezer -- drops unit-size (==1) dims from op.data.ranges/
        reduction_ranges before numbering d0, d1, ... densely over the
        *remaining* dims. So d{dim} in dep.index is only the correct symbol
        when no unit dims precede it; whenever a unit dim is skipped, the
        host-range index and the d{i} symbol index diverge (e.g. BHLD with
        B=1: host-range dim 1 (H) is squeezed d0, not d1).

        This mirrors create_op_spec's own host_to_it construction (same
        squeeze arithmetic, same n_output_it_syms reduction-dim offset).
        """
        n_output_dims = 0
        it_idx = 0
        mapped: "int | None" = None
        if hasattr(ir_node, "data") and hasattr(ir_node.data, "ranges"):
            for host_idx, r in enumerate(ir_node.data.ranges):
                if int(r) != 1:
                    if host_idx == dim:
                        mapped = it_idx
                    it_idx += 1
            n_output_dims = it_idx
        else:
            mapped = dim
        if mapped is None and hasattr(ir_node, "data"):
            # Reduction dim: dim >= n_output_dims: offset is n_output_dims
            # (post-squeeze) plus the squeezed position within
            # reduction_ranges.
            reduction_pos = dim - (
                len(ir_node.data.ranges) if hasattr(ir_node.data, "ranges") else 0
            )
            reduction_ranges = getattr(ir_node.data, "reduction_ranges", None)
            if reduction_ranges is not None and reduction_pos >= 0:
                red_it_idx = 0
                for host_idx, r in enumerate(reduction_ranges):
                    if int(r) != 1:
                        if host_idx == reduction_pos:
                            mapped = n_output_dims + red_it_idx
                            break
                        red_it_idx += 1
        if mapped is None:
            # Fallback: identity mapping (no unit dims to skip, or ir_node
            # lacks a `data` attribute -- e.g. in unit tests using bare
            # fixtures).
            mapped = dim
        return sympy_index_symbol(f"d{mapped}")

    def _general_tile_advance(
        self, tensor: TensorAccess, is_input: bool, name: str
    ) -> "sympy.Expr | None":
        """This arg's device-element tile-advance expr.

        Re-derives dep.index at this call (guaranteed final -- every pass
        that could rewrite it via WrapperHandler has already run), builds
        one substituted term per nesting level using that level's own
        minted symbol (via _get_or_mint_level_symbol) and
        CoarseTileInfo.tiled_dims_per_read/output_tiled_dims's per-level
        (dim, extent) decision, reprojects each level's host-space term to
        device-element space via views.tiling_expr_to_device_expr, and sums
        every level into one combined Expr -- preserving the single-Expr-
        per-arg contract compute_ops.py/superdsc.py/bundle.py depend on.

        A read dim tiled down to extent 1 in dep's own iteration space has
        no d{i} symbol at all (Inductor's SqueezeView.squeezer drops it
        unconditionally -- see CoarseTileInfo.squeezed_advance_per_read's
        docstring), so it cannot contribute via substitution into dep.index
        like every other tiled dim here. Its (host_stride, extent) pairs
        are added as independent terms (level_symbol * extent * host_stride,
        already in host-element space) through the same
        tiling_expr_to_device_expr projection instead.

        This is the sole tile-advance mechanism. Returns None for ops
        without loop_info/coarse tiling.
        """
        ir_node = self.current_node.node
        loop_info = getattr(ir_node, "loop_info", None)
        if loop_info is None:
            return None

        op_name = ir_node.get_operation_name()
        squeezed_advance_per_level: list[list[tuple[sympy.Expr, sympy.Expr]]] = []

        if is_input:
            read_deps = [
                dep
                for dep in ir_node.get_read_writes().reads
                if isinstance(dep, MemoryDep)
            ]
            matching_idx = [i for i, dep in enumerate(read_deps) if dep.name == name]
            if not matching_idx:
                return None
            # Positional tie-break: if this buffer is read more than once,
            # consume matches in order across successive calls for the same
            # name within this op's TensorArg construction. store()/
            # store_reduction() reset this dict at the start of each op.
            consumed = self._general_tile_advance_seen.get(name, 0)
            if consumed >= len(matching_idx):
                return None
            self._general_tile_advance_seen[name] = consumed + 1
            dep_idx = matching_idx[consumed]
            if dep_idx >= len(loop_info.tiled_dims_per_read):
                return None
            dep = read_deps[dep_idx]
            per_level_dims = loop_info.tiled_dims_per_read[dep_idx]
            squeezed_advance_per_read = getattr(
                loop_info, "squeezed_advance_per_read", None
            )
            if squeezed_advance_per_read and dep_idx < len(squeezed_advance_per_read):
                squeezed_advance_per_level = squeezed_advance_per_read[dep_idx]
        else:
            write_deps = [
                dep
                for dep in ir_node.get_read_writes().writes
                if isinstance(dep, MemoryDep)
            ]
            if not write_deps:
                return None
            dep = write_deps[0]
            per_level_dims = loop_info.output_tiled_dims
            squeezed_advance_per_level = (
                getattr(loop_info, "squeezed_advance_output", None) or []
            )

        if not per_level_dims and not any(squeezed_advance_per_level):
            return None

        device_size = tensor.layout.device_layout.device_size
        stride_map = tensor.layout.device_layout.stride_map

        total_device_expr: "sympy.Expr | None" = None
        n_levels = max(len(per_level_dims), len(squeezed_advance_per_level))
        for level_idx in range(n_levels):
            dim_extent_pairs = (
                per_level_dims[level_idx] if level_idx < len(per_level_dims) else []
            )
            squeezed_pairs = (
                squeezed_advance_per_level[level_idx]
                if level_idx < len(squeezed_advance_per_level)
                else []
            )
            if not dim_extent_pairs and not squeezed_pairs:
                continue
            level_symbol = self._get_or_mint_level_symbol(level_idx, op_name)
            host_expr = sympy.S.Zero
            if dim_extent_pairs:
                tiled_dim_extents = {
                    self._host_dim_to_index_symbol(ir_node, d): extent * level_symbol
                    for d, extent in dim_extent_pairs
                }
                subs = dict(tiled_dim_extents)
                subs.update(
                    {
                        sym: sympy.Integer(0)
                        for sym in dep.index.free_symbols
                        if sym not in subs
                    }
                )
                host_expr += dep.index.subs(subs)
            for host_stride, extent in squeezed_pairs:
                host_expr += level_symbol * extent * host_stride
            device_expr = tiling_expr_to_device_expr(device_size, stride_map, host_expr)
            total_device_expr = (
                device_expr
                if total_device_expr is None
                else total_device_expr + device_expr
            )

        return total_device_expr

    def create_tensor_arg(
        self,
        is_input: bool,
        name: str,
        tensor: TensorAccess,
        opspec_name: "str | None" = None,
    ) -> TensorArg:
        # OpSpec->KTIR needs a stable per-buffer identity for register-threaded
        # fused intermediates (all arg_index == -1): _buf_id keys on TensorArg.name,
        # which is serialized into the emitted op-spec literal and read back by
        # generate_ktir.  The SDSC/flex literal identifies buffers by arg_index +
        # allocation address and only needs name for gather indices, so populate it
        # from the buffer name only when the KTIR emitter is enabled
        # (config.ktir_emitter, i.e. TORCH_SPYRE_KTIR=1) -- leaving the default
        # SDSC literal byte-identical.
        if opspec_name is None and _spyre_config.ktir_emitter:
            opspec_name = name
        it_space = iteration_space(self.current_node)
        # With dynamic=True the host index may contain symbolic strides
        # (e.g. x0*s1+x1).  Concretize size symbols so normalize_coordinates
        # can correctly isolate each loop variable's contribution.

        index = concretize_index(tensor.index, set(it_space.keys()))

        # insert_post_mutation_restickify may override the input layout for this input tensor.
        # Restore it here because the tensor data was uploaded as orig_stl.
        if is_input:
            overrides = getattr(self.current_node.node, "_input_layout_overrides", {})
            if (layout := overrides.get(name)) is not None:
                tensor.layout = layout

        device_coords = compute_coordinates(
            tensor.layout.device_layout.device_size,
            tensor.layout.device_layout.stride_map,
            it_space,
            index,
            self.indirect_sizes,
        )
        work_division = work_division_from_view(
            tensor.layout.lx_view if "lx" in tensor.layout.allocation else None,
            device_coords,
            tuple(it_space),
        )
        device_tile_advance_expr = self._general_tile_advance(tensor, is_input, name)
        tensor_arg = TensorArg(
            is_input,
            -1,
            tensor.layout.device_layout.device_dtype,
            tensor.layout.device_layout.device_size,
            device_coords,
            tensor.layout.allocation,
            element_arrangement=tensor.layout.device_layout.element_arrangement,
            name=opspec_name,
            device_tile_advance_expr=device_tile_advance_expr,
            work_division=work_division,
        )
        if (
            "lx" not in tensor.layout.allocation
            and "hbm_pool" not in tensor.layout.allocation
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
            # Match the encoding in work_division.apply_splits: an indirect
            # (gather) read carries data-dependent symbols whose coefficients are
            # not a stable identity key, so prefer the first non-indirect read as
            # the reduction-split reference index.
            reads = self.current_node.read_writes.reads
            read_dep = next(
                (d for d in reads if isinstance(d, MemoryDep) and not d.is_indirect()),
                next(iter(reads), None),
            )
            read_index = read_dep.index if read_dep is not None else write_index
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

        # Build per-level tiled_symbols (innermost first) for this op.
        # loop_tiled_dims / loop_tiled_reduction_dims are lists of per-level
        # dim-index lists, outermost first — so we build outermost-first then
        # reverse to get innermost-first for tiled_symbols storage.
        #
        # Each level that tiles at least one dim (output or reduction) gets a
        # single minted symbol via _get_or_mint_level_symbol, shared with the
        # same (op, level) symbol _general_tile_advance mints when building
        # each TensorArg's device_tile_advance_expr -- so
        # device_tile_advance_expr.coeff(sym) resolves to that level's own
        # contribution regardless of how many host dims it tiles.
        li = getattr(ir_node, "loop_info", None)
        raw_tiled_dims: list[list[int]] = li.loop_tiled_dims if li is not None else []
        raw_tiled_red_dims: list[list[int]] = (
            li.loop_tiled_reduction_dims if li is not None else []
        )
        # CoarseTileInfo always constructs loop_tiled_dims and
        # loop_tiled_reduction_dims with the same length (one sublist per
        # nesting level), so max() is just a safety net; in practice both
        # lists have the same length and the per-level loop below never
        # silently drops an entry from the shorter one.
        n_levels = max(len(raw_tiled_dims), len(raw_tiled_red_dims))

        tiled_symbol_trip_counts: dict = {}
        tiled_syms: list[list] = []
        if n_levels > 0:
            loop_count = li.loop_count if li is not None else []

            op_name = ir_node.get_operation_name()
            tiled_syms_per_level_outermost: list[list] = []
            for lvl in range(n_levels):
                level_syms: list = []
                has_any_tiled_dim = (
                    lvl < len(raw_tiled_dims) and raw_tiled_dims[lvl]
                ) or (lvl < len(raw_tiled_red_dims) and raw_tiled_red_dims[lvl])
                if has_any_tiled_dim:
                    level_syms.append(self._get_or_mint_level_symbol(lvl, op_name))
                tiled_syms_per_level_outermost.append(level_syms)
                if lvl < len(loop_count):
                    trip_count = int(loop_count[lvl])
                    for sym in level_syms:
                        tiled_symbol_trip_counts[sym] = trip_count
            # Reverse so index 0 = innermost level.
            tiled_syms = list(reversed(tiled_syms_per_level_outermost))

        # Collect (max, granularity) bounds for any symbolic iteration-space
        # dims. These are passed through OpSpec so SDSC codegen can emit
        # symbolicDimInfo_ without needing the live ShapeEnv (which is gone
        # during the codegen phase).
        symbolic_dim_bounds: dict[str, tuple[int, int]] = {}
        for _, (size_expr, _) in it_space_extended.items():
            if not (hasattr(size_expr, "free_symbols") and size_expr.free_symbols):
                continue
            if finite_upper_or_none(size_expr) is None:
                logger.debug(
                    f"[work_division/symbolic] skipping auto-dynamic symbol "
                    f"{size_expr}; use mark_dynamic(max=...) to enable symbolic planning"
                )
                continue
            bounds = compute_symbolic_bounds(size_expr)
            if bounds is not None:
                symbolic_dim_bounds[str(size_expr)] = bounds

        # Provenance is a debug-only feature: a failure building the handle must
        # never break a compile. build_debug_handle is best-effort, but guard the
        # call site too so an unexpected node shape can't fail create_op_spec.
        try:
            debug_handle = build_debug_handle(ir_node)
        except Exception:  # noqa: BLE001 - provenance must never fail the build
            logger.warning(
                "debug_handle construction failed for op %s; continuing without "
                "provenance",
                op,
                exc_info=True,
            )
            debug_handle = None

        # Carry the node's full logical output ranges (NCHW, incl. unit dims)
        # so codegen can derive surviving dim roles and the channel count from
        # live IR instead of a lowering-time size snapshot.  Store raw ranges
        # (no int(): ranges may be symbolic); consumers convert only static
        # dims.  Populated for pools and convs (forward + depthwise) — the only
        # consumers — so other kernels' generated source is unchanged.
        node_output_ranges = (
            tuple(ir_node.data.ranges)
            if op in POOL_OPS | CONV_OPS
            and hasattr(ir_node, "data")
            and hasattr(ir_node.data, "ranges")
            else None
        )
        if not is_lx_relayout_identity(op, args):
            for arg in args:
                arg.work_division = None

        return OpSpec(
            op,
            is_reduction,
            it_space_extended,
            args,
            op_info,
            tiled_symbols=tiled_syms,
            tiled_symbol_trip_counts=tiled_symbol_trip_counts,
            symbolic_dim_bounds=symbolic_dim_bounds,
            node_output_ranges=node_output_ranges,
            debug_handle=debug_handle,
        )

    def remove_kernel_local_buffers(self) -> None:
        """Remove buffers that have a scratchpad or temporary allocation from the kernel's arg list."""
        for name in list(self.store_buffer_names):
            buf = V.graph.get_buffer(name)
            if buf is None:
                continue
            layout = buf.get_layout()
            if isinstance(layout, FixedTiledLayout) and (
                "lx" in layout.allocation or "hbm_pool" in layout.allocation
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
        if "lx" not in layout.allocation and "hbm_pool" not in layout.allocation:
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
        self._general_tile_advance_seen = {}
        self._tile_advance_symbols = {}
        # mutation_real_name maps mutation aliases to their real destination buffer. Resolve that here,
        # and mark the buf name as removed so the wrapper does not allocate it separately.
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        if real_dst_name != name:
            V.graph.removed_buffers.add(name)
        buf = V.graph.get_buffer(real_dst_name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{real_dst_name} does not have FixedTiledLayout")
        # HBM-pool buffers are intermediates whose address is baked into the TensorArg
        # allocation dict; registering them as outputs would overflow SEGMENT_OFFSETS.
        # (lx buffers are already excluded from spyre_kernel_args in _tensor_arg.)
        # Also skip buffers marked as removed by Inductor's optimizer (e.g., by LX )
        # This can occur when SDPA decomposition creates intermediate
        # buffers that later get marked as dead code.
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        is_removed = real_dst_name in V.graph.removed_buffers
        if "hbm_pool" not in layout.allocation and not is_removed:
            # Pass the alias here, not real_dst_name: args.output resolves the
            # mutation alias internally. (load() passes the pre-resolved real
            # name to args.input, which does not resolve.)
            _ = self.args.output(name)
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        dst = TensorAccess(name, index, layout)
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
            # Compute which indirect variables THIS operation actually uses:
            # - For gather: check source index for indirect symbols
            # - For scatter: check destination index for indirect symbols
            # Use the same filtering logic as PointwiseOp to avoid duplication.
            indirect_syms_used = (
                _indirect_syms_used(
                    value,
                    self.indirect_vars,
                    src_index=value.index,
                    dst_index=dst.index,
                )
                if self.indirect_vars
                else set()
            )

            if indirect_syms_used:
                # Gather/scatter: coordinates are built with raw indirect symbols here;
                # indirect_access_subs is applied later in codegen_kernel → simplify_op_spec.
                # Only add the indirect tensors that this specific operation uses.
                args = [
                    self.create_tensor_arg(
                        True,
                        idx_tensor.name,
                        idx_tensor,
                        opspec_name=idx_tensor.name,
                    )
                    for sym in sorted(indirect_syms_used, key=str)
                    for idx_tensor in [self.indirect_vars[sym]]
                ]
                args += [
                    self.create_tensor_arg(True, value.name, value),
                    self.create_tensor_arg(False, real_dst_name, dst),
                ]
                # Only pass indirect var names that this operation uses
                op_indirect_var_names = frozenset(
                    self.indirect_vars[sym].name for sym in indirect_syms_used
                )
            else:
                args = [
                    self.create_tensor_arg(True, value.name, value),
                    self.create_tensor_arg(False, real_dst_name, dst),
                ]
                op_indirect_var_names = None
            in_coords = args[-2].device_coordinates
            out_coords = args[-1].device_coordinates
            if is_restickify_coords(in_coords, out_coords):
                op = RESTICKIFY_OP
            else:
                op = IDENTITY_OP
            op_spec = self.create_op_spec(
                op, False, args, op_info, op_indirect_var_names
            )
            self.op_specs.append(op_spec)
        else:
            raise Unsupported(f"store value of unexpected type {type(value)}")

    def store_reduction(
        self, name: str, index: sympy.Expr, value: ReductionOp | UnimplementedOp
    ) -> None:
        """Convert an RValue"""
        self._general_tile_advance_seen = {}
        self._tile_advance_symbols = {}
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        # HBM-pool buffers are intermediates whose address is baked into the TensorArg
        # allocation dict; registering them as outputs would overflow SEGMENT_OFFSETS.
        # (lx buffers are already excluded from spyre_kernel_args in _tensor_arg.)
        if "hbm_pool" not in layout.allocation:
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

        if value.op in TWO_INPUT_REDUCTION_OPS:
            # Two-input reductions: matmul (activation @ weight) and conv2d
            # (activation * weight, reduced over in/ki/kj). Both build
            # [input, weight, output] tensor args.
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
        elif value.op in DEPTHWISE_CONV_REDUCTION_OPS:
            if (
                len(value.arguments) < 2
                or (not isinstance(value.arguments[0], TensorAccess))
                or (not isinstance(value.arguments[1], TensorAccess))
            ):
                raise Unsupported(
                    f"invalid depthwiseconv2dnative arguments {value.arguments}"
                )
            x = value.arguments[0]
            w = value.arguments[1]
            args = [
                self.create_tensor_arg(True, x.name, x),
                self.create_tensor_arg(True, w.name, w),
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

    def wrap_op_specs_in_loop(self, count: sympy.Expr) -> None:
        """Replace the current op_specs list with a single LoopSpec of the given count."""
        body = self.op_specs
        self.op_specs = [LoopSpec(count=count, body=body)]

    def codegen_kernel(self):
        """Codegen the body of this kernel by pretty printing its list of OpSpecs"""

        indirect_access_subs = (
            indirect_access_subs_from_kernel(self.indirect_vars)
            if self.indirect_vars
            else None
        )

        if _spyre_config.validate_op_specs:
            validate_op_specs(self.op_specs, stage="after_creation_loop_wrapping")
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "OP SPECS AFTER CREATION/LOOP-WRAPPING\n%s",
                format_op_spec_list(self.op_specs),
            )

        for op_spec in _iter_op_specs(self.op_specs):
            simplify_op_spec(op_spec, self.indirect_sizes, indirect_access_subs)

        if _spyre_config.validate_op_specs:
            validate_op_specs(self.op_specs, stage="after_simplification")
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "OP SPECS AFTER SIMPLIFICATION\n%s",
                format_op_spec_list(self.op_specs),
            )

        def sympy_str(x: sympy.Expr) -> str:
            if isinstance(x, IndirectAccess):
                name_sym = x.args[0]
                return f"IndirectAccess('{name_sym}')"
            return "sympify('" + str(x) + "')"

        # Now that all loads/stores have been processed we know the final kernel_args and can map names to indices
        actuals = self.args.python_argdefs()[1]
        has_pool_allocations = self.pool_size > 0

        for name, tensor_arg in self.spyre_kernel_args:
            tensor_arg.arg_index = actuals.index(name)
            if _spyre_config.bundle_symbolic_args:
                # On the symbolic path the HBM address is provided at runtime
                # via input_arg_extract; start_address is never used as a
                # literal address.  Use the arg_index itself as a small,
                # positive, human-readable sentinel that is clearly not a
                # real HBM address (which are O(16 GB) apart).
                tensor_arg.allocation["hbm"] = tensor_arg.arg_index
            else:
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

    def _kernel_uses_hbm_pool(self) -> bool:
        """Return True if any op in this kernel references an HBM-pool-allocated tensor."""
        return uses_hbm_pool(self.op_specs)

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel. This kernel's own HBM pool tensor
        (if any) is allocated by the generated MLIR itself, via
        sdscbundle.device_mem_allocate -- there is no Python-side pool
        tensor to allocate or free here."""
        wrapper = V.graph.wrapper_code
        call_args = []

        # Add kernel arguments, deduplicating tensors that appear as both
        # input and output (e.g. in-place ops like x *= 2).  With symbolic
        # args the MLIR bundle emits one !sdscbundle.input_arg<index> per unique
        # arg_index; passing the same tensor twice would cause a runtime
        # "Number of inputs mismatches" error in processComputeOnHostCommand.
        seen: set[str] = set()
        for arg in self.args.python_argdefs()[1]:
            if arg not in seen:
                seen.add(arg)
                call_args.append(arg)

        call_args_str = ", ".join(call_args)
        wrapper.writeline(f"{name}.run({call_args_str})")

    def emit_layout_restores(self, restores) -> None:
        """Emit set_spyre_tensor_layout wrapper calls after this kernel's run.

        The scheduler selects and dedups the restores; this kernel just writes
        them into the wrapper alongside its own call, using the same wrapper.
        """
        wrapper = V.graph.wrapper_code
        for target_name, alt_stl in restores:
            wrapper.writeline(f"set_spyre_tensor_layout({target_name}, {alt_stl!r})")


def _indirect_syms_used(
    value,
    indirect_vars: "dict[sympy.Symbol, TensorAccess]",
    src_index: "sympy.Expr | None" = None,
    dst_index: "sympy.Expr | None" = None,
) -> "set[sympy.Symbol]":
    """Return the subset of indirect_vars keys that appear in value's indices.

    For PointwiseOp: checks all argument indices (via value.arguments).
    If src_index is provided (for gather source indices), also checks it.
    If dst_index is provided (for scatter destination indices), also checks it.
    """
    syms = set()
    if hasattr(value, "arguments"):
        syms = {
            s
            for inp in value.arguments
            if isinstance(inp, TensorAccess)
            for s in inp.index.free_symbols
            if s in indirect_vars
        }
    if src_index is not None:
        syms.update(s for s in src_index.free_symbols if s in indirect_vars)
    if dst_index is not None:
        syms.update(s for s in dst_index.free_symbols if s in indirect_vars)
    return syms


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


def uses_hbm_pool(specs) -> bool:
    """Return True if any op in ``specs`` references an HBM-pool-allocated tensor.

    This decides whether ``call_kernel`` passes the pool ahead of the kernel
    args, so anything reading a kernel's ``.run()`` arguments has to agree with
    it -- hence a shared function rather than a copy per caller.
    """
    return any(
        "hbm_pool" in arg.allocation
        for op in _iter_op_specs(specs)
        for arg in op.args
        if isinstance(arg, TensorArg)
    )


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
                        + ", ".join(
                            "[" + ", ".join(sympy_str(s) for s in level) + "]"
                            for level in op_spec.tiled_symbols
                        )
                        + "],"
                    )
                if op_spec.tiled_symbol_trip_counts:
                    trip_counts_str = ", ".join(
                        f"{sympy_str(sym)}: {count}"
                        for sym, count in op_spec.tiled_symbol_trip_counts.items()
                    )
                    buf.writeline(f"tiled_symbol_trip_counts={{{trip_counts_str}}},")
                buf.writeline(
                    f"symbolic_dim_bounds={_serialize_value(op_spec.symbolic_dim_bounds)},"
                )
                if op_spec.node_output_ranges is not None:
                    # Must survive the OpSpec -> generated-source -> exec
                    # round-trip: pool/conv codegen reads it to align dim labels
                    # and the channel-count fallback.  Ranges are sympy Exprs;
                    # sympy_str emits eval-able sympify(...) calls.
                    buf.writeline(
                        "node_output_ranges=("
                        + "".join(
                            sympy_str(r) + ", " for r in op_spec.node_output_ranges
                        )
                        + "),"
                    )
                if op_spec.debug_handle is not None:
                    # Source-to-kernel provenance must survive the OpSpec ->
                    # generated-source -> exec round-trip. DebugHandle/SourceLoc
                    # are frozen dataclasses, so repr() is eval-able; the
                    # generated wrapper header imports both names.
                    buf.writeline(f"debug_handle={op_spec.debug_handle!r},")
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
                            if arg.name is not None:
                                buf.writeline(f"name={arg.name!r},")
                            if arg.device_tile_advance_expr is not None:
                                buf.writeline(
                                    "device_tile_advance_expr="
                                    f"{sympy_str(arg.device_tile_advance_expr)},"
                                )
                            if arg.element_arrangement != ElementArrangement.STANDARD:
                                buf.writeline(
                                    f"element_arrangement={arg.element_arrangement},"
                                )
                            if arg.work_division is not None:
                                splits = ", ".join(
                                    f"{sympy_str(dim)}: {split}"
                                    for dim, split in arg.work_division.work_slices.items()
                                )
                                core_map = ", ".join(
                                    f"{sympy_str(dim)}: {sympy_str(slot)}"
                                    for dim, slot in arg.work_division.core_id_to_work_slice.items()
                                )
                                buf.writeline(
                                    "work_division=TensorWorkDivision("
                                    f"work_slices={{{splits}}}, "
                                    f"core_id_to_work_slice={{{core_map}}}),"
                                )
                        buf.writeline("),")
                buf.writeline("]")
            buf.writeline("),")


def _remap_work_division(arg: TensorArg, work_division_remap) -> None:
    """Carry tensor ownership through iteration-space normalization."""

    if arg.work_division is None:
        return
    new_splits: dict[sympy.Symbol, int] = {}
    new_core_map: dict[sympy.Symbol, sympy.Expr] = {}
    for old_dim, split in arg.work_division.work_slices.items():
        new_dims = work_division_remap[old_dim]
        remaining_split = int(split)
        split_factors = []
        if len(new_dims) == 1:
            split_factors = [(new_dims[0][0], remaining_split)]
            remaining_split = 1
        else:
            for new_dim, basis in reversed(new_dims):
                factor = math.gcd(remaining_split, basis)
                split_factors.append((new_dim, factor))
                remaining_split //= factor
            split_factors.reverse()
        if remaining_split != 1:
            raise ValueError(f"cannot normalize {split}-way split on {old_dim}")

        slot = arg.work_division.core_id_to_work_slice[old_dim]
        slot_stride = 1
        for new_dim, factor in split_factors:
            if factor == 1:
                continue
            new_slot = sympy.Mod(sympy.floor(slot / slot_stride), factor)
            if new_dim in new_splits and (
                new_splits[new_dim],
                new_core_map[new_dim],
            ) != (factor, new_slot):
                raise ValueError(f"conflicting normalized ownership on {new_dim}")
            new_splits[new_dim] = factor
            new_core_map[new_dim] = new_slot
            slot_stride *= factor
    arg.work_division = TensorWorkDivision(new_splits, new_core_map)


def _restickify_restore_elided_dim(op_spec) -> None:
    """Restore a restickify's elided size-1 dim BEFORE align_tensors (in place).

    A restickify swaps which host dim lands inside the 128-byte stick.  When the
    dim on EITHER side of the swap has host size 1, upstream Inductor squeezes it
    away and never emits a loop symbol for it, so exactly one operand's
    within-stick (last) coordinate collapses to the constant ``0`` -- the
    "elided" operand (the other, unaffected operand is "intact"). With no
    iteration symbol the two operands disagree on which dim carries the stick and
    the backend cannot build a dimension mapping.

    align_tensors matches operands by shared symbol, so we restore the dim here,
    just before align runs -- creating one fresh symbol ``new_sym`` shared by both
    operands reduces the size-1 case to the ordinary N>=2 path where both carry a
    within-stick symbol. Doing it later (e.g. at SDSC time, or in the scheduler's
    ``mark_run``) is too late to affect the descriptor align has already built.
    The two operands are rebuilt to share ``new_sym`` (64 = fp16 stick elements):

    - ELIDED operand: its stick is rebuilt as
      ``[floor(new_sym/64)] + real_dims + [Mod(new_sym, 64)]``.
    - INTACT operand: ``new_sym`` binds to the outermost size-64 gap dim the
      padding pass (``_pad_elided_dim``) prepended to cover the 64-plane sweep
      (see the reuse site below).  ``new_sym`` has iteration RANGE 1, so it only
      ever takes the value 0: SDSC codegen's back-gap mechanism absorbs the
      size-64-vs-range-1 gap and it contributes no real stride to either operand.
    """
    assert len(op_spec.args) == 2, f"restickify op_spec has {len(op_spec.args)} args"
    in_arg, out_arg = op_spec.args[0], op_spec.args[1]

    def _stick_sym(arg):
        syms = tuple(arg.device_coordinates[-1].free_symbols)
        assert len(syms) <= 1, f"expected 0 or 1 free symbols, got {len(syms)}"
        return syms[0] if syms else None

    in_sym = _stick_sym(in_arg)
    out_sym = _stick_sym(out_arg)
    # Both-intact is the ordinary N>=2 case; nothing to restore.
    if in_sym is not None and out_sym is not None:
        return
    # Both-elided would mean neither operand's within-stick coord carries a
    # free symbol, contradicting is_restickify_coords's own free-symbol-mismatch test.
    assert not (in_sym is None and out_sym is None), "both operands elided"

    stick_size = in_arg.device_dtype.elems_per_stick()

    def _restore(new_sym, elided_arg, intact_arg) -> None:
        # Rebuild the elided stick as [floor(new_sym/64)] + reals + [Mod(new_sym, 64)].
        elided_coords = list(elided_arg.device_coordinates)
        elided_size = list(elided_arg.device_size)
        real_coords, real_sizes = [], []
        for i in range(len(elided_coords) - 1):  # exclude within-stick
            if elided_coords[i].free_symbols:
                real_coords.append(elided_coords[i])
                real_sizes.append(elided_size[i])
        new_elided_coords = (
            [sympy.floor(new_sym / stick_size)]
            + real_coords
            + [sympy.Mod(new_sym, stick_size)]
        )
        new_elided_size = [1] + real_sizes + [stick_size]

        # Bind new_sym to the size-64 dim _pad_elided_dim prepended, so the
        # descriptor's total size matches the grown allocation. The grow always
        # targets the intact operand, so that dim is present here: outermost
        # size-64 with coordinate 0 (asserted before we overwrite it).
        intact_coords = list(intact_arg.device_coordinates)
        intact_size = list(intact_arg.device_size)
        assert intact_size[0] == stick_size and intact_coords[0] == 0, (
            f"restickify restore: expected padding-prepended size-{stick_size} "
            f"gap dim on the intact operand, got size={intact_size[0]} "
            f"coord={intact_coords[0]}"
        )
        intact_coords[0] = new_sym

        # Range 1, not 64: new_sym only ever takes value 0, so it contributes
        # no real stride and the size-64 device slot is just back-gap padding.
        op_spec.iteration_space = {new_sym: (stick_size, 1), **op_spec.iteration_space}
        elided_arg.device_coordinates = new_elided_coords
        elided_arg.device_size = new_elided_size
        intact_arg.device_coordinates = intact_coords
        intact_arg.device_size = intact_size

    # Pick an unused name; new_sym is shared by both operands below so align
    # matches them as the same iteration var.
    used = set(op_spec.iteration_space.keys())
    for idx in itertools.count():
        new_sym = sympy.Symbol(f"rs{idx}")
        if new_sym not in used:
            break

    if in_sym is None:
        _restore(new_sym, in_arg, out_arg)
    else:
        # out_sym is None
        _restore(new_sym, out_arg, in_arg)


def simplify_op_spec(op_spec, indirect_sizes=None, indirect_access_subs=None):
    # Both parameters must be provided together for gather kernels — indirect_sizes
    # decomposes symbols in align_tensors; indirect_access_subs replaces them with IndirectAccess.

    if op_spec.op == RESTICKIFY_OP:
        # Restore a restickify's elided size-1 stick, creating a shared iteration
        # symbol on both operands, so align_tensors matches them by that symbol.
        _restickify_restore_elided_dim(op_spec)

    it_space = op_spec.iteration_space
    new_op_space_splits, new_tensors, work_division_remap = align_tensors(
        it_space,
        [
            {"size": arg.device_size, "coordinates": arg.device_coordinates}
            for arg in op_spec.args
        ],
        indirect_sizes,
    )
    op_spec.iteration_space = new_op_space_splits

    for arg, t in zip(op_spec.args, new_tensors):
        _remap_work_division(arg, work_division_remap)
        arg.device_size = t["size"]
        arg.device_coordinates = t["coordinates"]

        # Apply indirect_access_subs after align_tensors, so that indirect symbols
        # are decomposed as regular variables before substitution.
        if indirect_access_subs:
            arg.device_coordinates = [
                c.xreplace(indirect_access_subs) for c in arg.device_coordinates
            ]

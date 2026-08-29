# Copyright 2026 The Torch-Spyre Authors.
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

"""Op-specific work-division constraints, collected in one place.

work_division.py's core algorithm (span reduction, priority-based
distribution, the matmul cost model) is generic over the iteration space. A
few ops/layouts additionally forbid splitting specific dims, or force a dim's
split to an exact value, for reasons the generic algorithm has no way to know
about — e.g. the backend cannot coordinate-mask a dim spread over cores, or a
QFP8WT tensor's second stick dimension must stay whole.
``collect_work_division_constraints`` calls each rule and merges the results,
so work_division.py's call sites only need one call instead of hand-invoking
every rule.

"""

import dataclasses
import typing
from sympy import Expr, Symbol, divisors

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, Pointwise, Reduction
from torch_spyre._C import ElementArrangement

from .constants import (
    BATCH_MATMUL_OP,
    BATCH_MATMUL_FP8_OP,
    CONV2D_FWD_OP,
    DEPTHWISE_CONV2D_OP,
    KEEP_BY_INDEX_OP,
    POOL_OPS,
    _MAX_K_PER_CORE,
    TOPK_MAX_K_PER_CORE,
    TOPK_OPS,
)
from .errors import Unsupported
from .pass_utils import (
    concretize_expr,
    indirect_forbidden_split_syms,
    is_restickify_coords,
    op_read_writes,
)
from .logging_utils import get_inductor_logger
from . import config

if typing.TYPE_CHECKING:
    # Deferred to avoid a circular import: work_division.py imports from this
    # module, so TensorDep can only be used here as a string annotation.
    from .work_division import TensorDep

logger = get_inductor_logger("work_division_constraints")


@dataclasses.dataclass
class WorkDivConstraintContext:
    """Everything a constraint needs to decide which dims it restricts."""

    op: ComputedBuffer
    it_space: dict[Symbol, Expr]
    it_space_adjusted: dict[Symbol, Expr]
    output_td: "TensorDep"
    input_tds: "list[TensorDep]"
    stick_vars: dict[Symbol, int]
    reduction_vars: list[Symbol]
    committed_splits: dict[Symbol, int]


@dataclasses.dataclass
class ConstraintResult:
    """A constraint's verdict on the iteration space in a WorkDivConstraintContext.

    ``blocked`` dims must remain unsplit (composes by union across
    constraints). ``allowed_splits`` maps each dim to its hard legal factors
    (composes by intersection).
    """

    blocked: set[Symbol] = dataclasses.field(default_factory=set)
    allowed_splits: dict[Symbol, frozenset[int]] = dataclasses.field(
        default_factory=dict
    )


def collect_work_division_constraints(
    ctx: WorkDivConstraintContext,
) -> ConstraintResult:
    """Run every constraint below against ``ctx`` and merge the results.

    Raises Unsupported if a blocked dimension or hard domain conflicts with a
    prior span-limit commitment, or intersections between domains are empty.
    """
    blocked: set[Symbol] = set()
    allowed_splits: dict[Symbol, frozenset[int]] = {}
    for constraint in (
        carried_reduction_pinned_row,
        coordinate_mask_blocked_vars,
        conv_spatial_blocked_vars,
        reduction_window_blocked_vars,
        restickify_padding_blocked_vars,
        qfp8wt_split_domains,
        qfp8wt_matmul_k_split_domains,
        topk_split_domains,
        keep_by_index_k_split_constraint,
        keep_by_index_pinned_search_space_vars,
        indirect_access_split_domains,
    ):
        result = constraint(ctx)

        forced = {s for s in result.blocked if ctx.committed_splits.get(s, 1) > 1}
        if forced:
            raise Unsupported(
                f"{ctx.op.get_name()}: blocked dim(s) "
                f"{sorted(str(s) for s in forced)} conflict with hardware "
                f"memory-span split(s) "
                f"{[(str(s), ctx.committed_splits[s]) for s in forced]} "
                f"({constraint.__name__})."
            )
        blocked |= result.blocked

        for sym, allowed in result.allowed_splits.items():
            allowed = frozenset(allowed)
            if not allowed:
                raise Unsupported(
                    f"{ctx.op.get_name()}: empty legal split domain for {sym} "
                    f"({constraint.__name__})."
                )
            if sym in allowed_splits:
                allowed &= allowed_splits[sym]
                if not allowed:
                    raise Unsupported(
                        f"{ctx.op.get_name()}: conflicting legal split domains "
                        f"for {sym} ({constraint.__name__})."
                    )
            committed_split = ctx.committed_splits.get(sym)
            if committed_split is not None and committed_split not in allowed:
                raise Unsupported(
                    f"{ctx.op.get_name()}: legal split domain for {sym} is "
                    f"{sorted(allowed)} ({constraint.__name__}), but hardware "
                    f"memory-span limit committed {committed_split}."
                )
            allowed_splits[sym] = allowed

    return ConstraintResult(blocked=blocked, allowed_splits=allowed_splits)


def carried_reduction_pinned_row(
    ctx: WorkDivConstraintContext,
) -> ConstraintResult:
    """Keep every stage of a carried sum on its declared output-row split."""

    record = getattr(ctx.op, "_carried_reduction_record", None)
    if record is None:
        return ConstraintResult()

    loop_var_dims = getattr(ctx.op, "work_div_loop_info", {})
    candidates = [
        sym
        for sym in ctx.it_space_adjusted
        if record.row_dim_name in loop_var_dims.get(sym, [])
    ]
    if len(candidates) != 1:
        raise Unsupported(
            f"{ctx.op.get_name()}: carried reduction row "
            f"{record.row_dim_name!r} resolved to {candidates}"
        )
    return ConstraintResult(
        allowed_splits={
            candidates[0]: frozenset({record.required_row_split}),
        }
    )


def coordinate_mask_blocked_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Block reduction stick vars that cannot be split across cores.

    The backend cannot coordinate-mask a dim spread over cores (mirrors
    ``_get_coordinate_mask`` in codegen/superdsc.py). ``ctx.it_space`` must be
    the element-valued iteration space, since padding is defined on element
    counts.
    """
    blocked = {
        v
        for v in ctx.reduction_vars
        if v in ctx.stick_vars
        and concretize_expr(ctx.it_space[v]) % ctx.stick_vars[v] != 0
    }
    return ConstraintResult(blocked=blocked)


def conv_spatial_blocked_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Block output image dims for strided convolutions.

    Splitting spatial dims produces incorrect per-core DSM addressing. Span-limit
    commitments win, handled uniformly by ``collect_work_division_constraints``.
    """
    if not config.disable_conv2d_spatial_split:
        return ConstraintResult()

    op_info = getattr(ctx.op.data, "op_info", None)
    if not isinstance(op_info, dict):
        return ConstraintResult()
    conv_params = op_info.get("conv_params")
    if not isinstance(conv_params, dict):
        return ConstraintResult()
    # Depthwise conv2d (#3510) records stride as stride_i/stride_j; forward
    # conv2d (#3284) records it as stride_h/stride_w. Accept either spelling so
    # the strided-spatial-split block covers both direct-conv paths.
    stride_i = conv_params.get("stride_i", conv_params.get("stride_h", 1))
    stride_j = conv_params.get("stride_j", conv_params.get("stride_w", 1))
    if (stride_i or 1) <= 1 and (stride_j or 1) <= 1:
        return ConstraintResult()

    write = typing.cast(MemoryDep, next(iter(op_read_writes(ctx.op).writes)))
    blocked = {
        sym
        for sym in list(write.ranges)[-2:]
        if isinstance(sym, Symbol)
        and sym in ctx.it_space
        and concretize_expr(ctx.it_space[sym]) > 1
    }
    return ConstraintResult(blocked=blocked)


def reduction_window_blocked_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Keep pooling and convolution kernel windows local to each core."""

    if not isinstance(ctx.op.data, Reduction):
        return ConstraintResult()
    op = ctx.op.data.reduction_type
    if op in POOL_OPS:
        window_dims = ctx.reduction_vars
    elif op == CONV2D_FWD_OP:
        op_info = getattr(ctx.op.data, "op_info", None)
        conv_params = (
            op_info.get("conv_params", {}) if isinstance(op_info, dict) else {}
        )
        kernel_dims = sum(
            int(conv_params.get(name, 1)) > 1 for name in ("kernel_h", "kernel_w")
        )
        window_dims = ctx.reduction_vars[-kernel_dims:] if kernel_dims else []
    elif op == DEPTHWISE_CONV2D_OP:
        # Depthwise reduction order is kh, kw, then optional group. Unlike the
        # forward-conv path, a group dimension may therefore follow the window.
        window_dims = ctx.reduction_vars[:2]
    else:
        return ConstraintResult()

    return ConstraintResult(blocked=set(window_dims))


def restickify_padding_blocked_vars(
    ctx: WorkDivConstraintContext,
) -> ConstraintResult:
    """Keep an unaligned restickify stick dimension on one core."""

    if (
        not isinstance(ctx.op.data, Pointwise)
        or len(ctx.input_tds) != 1
        or not is_restickify_coords(
            ctx.input_tds[0].device_coords, ctx.output_td.device_coords
        )
    ):
        return ConstraintResult()

    padded = {
        dim
        for dim, stick_size in ctx.stick_vars.items()
        if concretize_expr(ctx.it_space[dim]) % stick_size
    }
    return ConstraintResult(blocked=padded)


def has_qfp8wt_tensor(tds: "list[TensorDep]") -> bool:
    return any(
        hasattr(td.layout.device_layout, "element_arrangement")
        and td.layout.device_layout.element_arrangement == ElementArrangement.QFP8WT
        for td in tds
    )


def qfp8wt_split_domains(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Restrict QFP8WT tensors' second stick dimension to split=1.

    QFP8WT uses a 2D stick layout (2x64 elements, 128 bytes); both stick dims
    must stay atomic 128-byte units, so any iteration var indexing the second
    stick coordinate of the matmul kernel tensor (second input) or the output
    has the singleton legal domain ``{1}``.
    """
    all_tds = ctx.input_tds + [ctx.output_td]
    if not has_qfp8wt_tensor(all_tds):
        return ConstraintResult()

    allowed_splits: dict[Symbol, frozenset[int]] = {}

    if len(ctx.input_tds) > 1:
        kernel_td = ctx.input_tds[1]
        if len(kernel_td.device_coords) > 1 and has_qfp8wt_tensor([kernel_td]):
            for var in kernel_td.device_coords[-2].free_symbols:
                if isinstance(var, Symbol):
                    allowed_splits[var] = frozenset({1})

    if len(ctx.output_td.device_coords) > 1 and has_qfp8wt_tensor([ctx.output_td]):
        for var in ctx.output_td.device_coords[-2].free_symbols:
            if isinstance(var, Symbol):
                allowed_splits[var] = frozenset({1})

    return ConstraintResult(allowed_splits=allowed_splits)


def qfp8wt_matmul_k_split_domains(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Restrict reduction K to split=1 for QFP8WT batchmatmul.

    Splitting K would require partial-sum accumulation across cores, which the
    QFP8WT matmul kernel does not support.
    """
    if not isinstance(ctx.op.data, Reduction):
        return ConstraintResult()
    if ctx.op.data.reduction_type not in (BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP):
        return ConstraintResult()

    all_tds = ctx.input_tds + [ctx.output_td]
    if not has_qfp8wt_tensor(all_tds):
        return ConstraintResult()

    return ConstraintResult(
        allowed_splits={v: frozenset({1}) for v in ctx.reduction_vars}
    )


def _topk_output_k_var(ctx: WorkDivConstraintContext) -> Symbol | None:
    """Return TopK k var, absent from every input index expression."""
    input_vars = {
        var
        for td in ctx.input_tds
        for var in td.dep.index.free_symbols
        if isinstance(var, Symbol)
    }
    output_vars = {
        var for var in ctx.output_td.dep.index.free_symbols if isinstance(var, Symbol)
    }
    candidates = output_vars - input_vars
    return next(iter(candidates)) if len(candidates) == 1 else None


def topk_split_domains(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Restrict TopK search-space and result dims to supported factors.

    TopK hardware requires at most ``TOPK_MAX_K_PER_CORE`` result rows per
    core. Although larger divisors also meet that limit, the 4D ``k=32``
    result-axis regression showed they produce incorrect output mapping. Keep
    only the smallest sufficient K split until larger factors have codegen
    support and regression coverage.
    """
    if (
        not isinstance(ctx.op.data, Reduction)
        or ctx.op.data.reduction_type not in TOPK_OPS
    ):
        return ConstraintResult()

    allowed_splits = {var: frozenset({1}) for var in ctx.reduction_vars}
    k_var = _topk_output_k_var(ctx)
    if k_var is None:
        return ConstraintResult(allowed_splits=allowed_splits)

    k_size = concretize_expr(ctx.it_space[k_var])
    legal_k_splits = frozenset(
        split
        for split in divisors(k_size)
        if split <= config.sencores and k_size // split <= TOPK_MAX_K_PER_CORE
    )
    if not legal_k_splits:
        raise Unsupported(
            f"topk(k={k_size}): no divisor within {config.sencores} cores gives "
            f"k_per_core <= {TOPK_MAX_K_PER_CORE}."
        )
    allowed_splits[k_var] = frozenset({min(legal_k_splits)})
    return ConstraintResult(allowed_splits=allowed_splits)


def _keep_by_index_axes(ctx: WorkDivConstraintContext) -> set[Symbol] | None:
    """Return the index-only K axes of a keep_by_index op."""
    if not (
        isinstance(ctx.op.data, Reduction)
        and ctx.op.data.reduction_type == KEEP_BY_INDEX_OP
    ):
        return None
    writes = op_read_writes(ctx.op).writes
    if not writes:
        return None
    iteration_vars = set(ctx.it_space)
    output_vars = {
        sym
        for sym in next(iter(writes)).index.free_symbols
        if isinstance(sym, Symbol) and sym in iteration_vars
    }
    # The indices input is the one that introduces K, a symbol absent from the
    # values/output index. This is structural rather than name-based: argument
    # names are scheduler-generated and therefore not a stable identifier.
    index_vars = set().union(
        *(
            {
                sym
                for sym in td.dep.index.free_symbols
                if isinstance(sym, Symbol) and sym in iteration_vars
            }
            for td in ctx.input_tds
            if td.dep.index.free_symbols & (iteration_vars - output_vars)
        )
    )
    if not index_vars:
        index_vars = set(ctx.reduction_vars)
    return index_vars - output_vars


def keep_by_index_k_split_constraint(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin index-only K to the smallest split that leaves at most four results/core."""
    axes = _keep_by_index_axes(ctx)
    if axes is None:
        return ConstraintResult()
    allowed_splits = {}
    for axis in axes:
        size = concretize_expr(ctx.it_space[axis])
        legal = [
            split
            for split in divisors(size)
            if split <= config.sencores and size // split <= _MAX_K_PER_CORE
        ]
        if not legal:
            raise Unsupported(
                f"keep_by_index(k={size}): no divisor within {config.sencores} "
                f"cores gives k_per_core <= {_MAX_K_PER_CORE}."
            )
        allowed_splits[axis] = frozenset({min(legal)})
    return ConstraintResult(allowed_splits=allowed_splits)


def keep_by_index_pinned_search_space_vars(
    ctx: WorkDivConstraintContext,
) -> ConstraintResult:
    """Keep one keep_by_index full-search output axis on each core.

    A broadcast indices input can omit unrelated output/batch axes. Preserve the
    prior coordinate-based policy: select one simplest output coordinate absent
    from the semantic indices operand rather than pinning every absent symbol.
    """
    if (
        not (
            isinstance(ctx.op.data, Reduction)
            and ctx.op.data.reduction_type == KEEP_BY_INDEX_OP
        )
        or len(ctx.input_tds) < 2
    ):
        return ConstraintResult()

    index_coords = ctx.input_tds[1].device_coords
    candidates = [
        coord
        for coord in ctx.output_td.device_coords
        if coord.free_symbols and not any(coord.equals(index) for index in index_coords)
    ]
    if not candidates:
        return ConstraintResult()

    search_coord = min(
        candidates, key=lambda coord: (len(coord.free_symbols), str(coord))
    )
    search_axis = next(
        (axis for axis in ctx.it_space if axis in search_coord.free_symbols), None
    )
    return (
        ConstraintResult(allowed_splits={search_axis: frozenset({1})})
        if search_axis is not None
        else ConstraintResult()
    )


def indirect_access_split_domains(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Keep indirect shared-data and unsafe partial-stick dims unsplit.

    A gather value table and scatter destination have one shared base on every
    core. Their data dims must therefore stay at split=1. A partial index stick
    also stays unsplit unless gather-output padding made its entry slices
    stick-aligned. Other index-entry dims remain available for multicore work.
    """
    return ConstraintResult(
        allowed_splits={
            sym: frozenset({1}) for sym in indirect_forbidden_split_syms(ctx.op)
        }
    )

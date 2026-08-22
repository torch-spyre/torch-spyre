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
from sympy import Expr, Symbol

from torch._inductor.ir import ComputedBuffer, Reduction
from torch_spyre._C import ElementArrangement

from .constants import BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP
from .errors import Unsupported
from .pass_utils import (
    concretize_expr,
    indirect_forbidden_split_syms,
    indirect_store_entry_syms,
    is_topk,
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

    ``blocked`` dims must not be split beyond whatever split they already
    carry (composes by union across constraints). ``pinned`` dims must equal
    exactly the given split (composes by equality; two constraints pinning the
    same dim to different values is a modeling conflict, not something to
    silently resolve — see collect_work_division_constraints).

    ``forbidden`` dims must NEVER be split anywhere — a hard correctness rule
    stronger than ``blocked``. Unlike ``blocked`` (which the distribution passes
    honour but span reduction may still override to satisfy the memory-span
    limit), a ``forbidden`` dim is filtered out of the span-reduction candidate
    set too, so it is never split under any circumstance. Used for shared
    gather/scatter table data dims.

    ``force_output`` dims are promoted to output-split priority even when they
    don't appear in the output coordinates (a scatter's index-entry dim, whose
    destination row is runtime-chosen). Composes by union.
    """

    blocked: set[Symbol] = dataclasses.field(default_factory=set)
    pinned: dict[Symbol, int] = dataclasses.field(default_factory=dict)
    forbidden: set[Symbol] = dataclasses.field(default_factory=set)
    force_output: set[Symbol] = dataclasses.field(default_factory=set)


def collect_work_division_constraints(
    ctx: WorkDivConstraintContext,
) -> ConstraintResult:
    """Run every constraint below against ``ctx`` and merge the results.

    A blocked dim that ``ctx.committed_splits`` has already split beyond 1 is
    dropped from the result (with a warning): a mandatory prior commitment —
    e.g. span_reduction satisfying the hardware span limit — outranks a
    constraint's preference not to split that dim further.

    Raises Unsupported if a pin conflicts with a prior span-limit commitment,
    or if two constraints pin the same dim to different values.
    """
    blocked: set[Symbol] = set()
    pinned: dict[Symbol, int] = {}
    forbidden: set[Symbol] = set()
    force_output: set[Symbol] = set()
    for constraint in (
        coordinate_mask_blocked_vars,
        conv_spatial_blocked_vars,
        qfp8wt_pinned_vars,
        qfp8wt_matmul_k_pinned,
        topk_pinned_search_space_vars,
        topk_k_split_constraint,
        indirect_access_constraints,
    ):
        result = constraint(ctx)

        forced = {s for s in result.blocked if ctx.committed_splits.get(s, 1) > 1}
        if forced:
            logger.warning(
                f"{ctx.op.get_name()}: constraint {constraint.__name__} would "
                f"block dim(s) {sorted(str(s) for s in forced)} from being "
                f"split, but the hardware memory-span limit already committed "
                f"split(s) {[(str(s), ctx.committed_splits[s]) for s in forced]}; "
                f"the constraint is not honoured for those dims."
            )
        blocked |= result.blocked - forced
        forbidden |= result.forbidden
        force_output |= result.force_output

        for sym, split in result.pinned.items():
            committed_split = ctx.committed_splits.get(sym)
            if committed_split is not None and committed_split != split:
                raise Unsupported(
                    f"{ctx.op.get_name()}: pinned split for {sym} is {split} "
                    f"({constraint.__name__}), but hardware memory-span limit "
                    f"committed {committed_split}."
                )
            if sym in pinned and pinned[sym] != split:
                raise Unsupported(
                    f"{ctx.op.get_name()}: conflicting pinned split for {sym}: "
                    f"{pinned[sym]} (from an earlier constraint) vs {split} "
                    f"(from {constraint.__name__})."
                )
            pinned[sym] = split

    return ConstraintResult(
        blocked=blocked,
        pinned=pinned,
        forbidden=forbidden,
        force_output=force_output,
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

    write_ranges = list(next(iter(op_read_writes(ctx.op).writes)).ranges)
    blocked = {
        sym
        for sym in write_ranges[-2:]
        if sym in ctx.it_space and concretize_expr(ctx.it_space[sym]) > 1
    }
    return ConstraintResult(blocked=blocked)


def has_qfp8wt_tensor(tds: "list[TensorDep]") -> bool:
    return any(
        hasattr(td.layout.device_layout, "element_arrangement")
        and td.layout.device_layout.element_arrangement == ElementArrangement.QFP8WT
        for td in tds
    )


def qfp8wt_pinned_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin QFP8WT tensors' second stick dimension to split=1.

    QFP8WT uses a 2D stick layout (2x64 elements, 128 bytes); both stick dims
    must stay atomic 128-byte units, so any iteration var indexing the second
    stick coordinate of the matmul kernel tensor (second input) or the output
    is pinned to exactly 1.
    """
    all_tds = ctx.input_tds + [ctx.output_td]
    if not has_qfp8wt_tensor(all_tds):
        return ConstraintResult()

    pinned: dict[Symbol, int] = {}

    if len(ctx.input_tds) > 1:
        kernel_td = ctx.input_tds[1]
        if len(kernel_td.device_coords) > 1 and has_qfp8wt_tensor([kernel_td]):
            for var in kernel_td.device_coords[-2].free_symbols:
                pinned[var] = 1

    if len(ctx.output_td.device_coords) > 1 and has_qfp8wt_tensor([ctx.output_td]):
        for var in ctx.output_td.device_coords[-2].free_symbols:
            pinned[var] = 1

    return ConstraintResult(pinned=pinned)


def qfp8wt_matmul_k_pinned(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin the reduction (K) dim to split=1 for batchmatmulfp8 with a QFP8WT kernel.

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

    return ConstraintResult(pinned={v: 1 for v in ctx.reduction_vars})


def topk_pinned_search_space_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin the search-space (reduction) dim to split=1 for topk ops.

    The topk hardware op searches the full dimension on one core to compute
    the top-k results. Splitting the search-space would require merging
    partial top-k results across cores, which the hardware does not support.
    """
    if not is_topk(ctx.op):
        return ConstraintResult()

    return ConstraintResult(pinned={v: 1 for v in ctx.reduction_vars})


def topk_k_split_constraint(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin k to the smallest valid split for topk ops.

    Each core can produce at most 4 top-k results per pass. The smallest valid
    k-split is ceil(k / 4), chosen to minimize core usage while satisfying the
    hardware constraint. This is pinned as a hard constraint to ensure the
    work_distribution planner picks the minimal k-split rather than a
    larger one that leaves more cores for other dims.
    """
    from sympy import divisors

    if not is_topk(ctx.op):
        return ConstraintResult()

    # Find k's symbol (output dim absent from every input's device coords).
    coord_vars = {
        s for td in ctx.input_tds for e in td.device_coords[:-1] for s in e.free_symbols
    }
    output_vars = {s for e in ctx.output_td.device_coords[:-1] for s in e.free_symbols}
    k_sym_candidates = [s for s in output_vars if s not in coord_vars]
    if len(k_sym_candidates) != 1:
        # k=1 or malformed; no constraint needed.
        return ConstraintResult()

    k_sym = k_sym_candidates[0]
    k_val = concretize_expr(ctx.it_space[k_sym])

    # Find the smallest divisor d of k such that k / d <= 4.
    _TOPK_MAX_K_PER_CORE = 4
    max_cores = config.sencores
    min_k_split = None
    for d in sorted(divisors(k_val)):
        if k_val // d <= _TOPK_MAX_K_PER_CORE and d <= max_cores:
            min_k_split = d
            break

    if min_k_split is None:
        raise Unsupported(
            f"topk(k={k_val}): no divisor of k in [1, {max_cores}] gives "
            f"k_per_core <= {_TOPK_MAX_K_PER_CORE}, so k cannot be split "
            f"across at most {max_cores} cores"
        )

    if min_k_split > 1:
        return ConstraintResult(pinned={k_sym: min_k_split})

    return ConstraintResult()


def indirect_access_constraints(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Split rules for indirect (gather/scatter-style) access. Empty for other ops.

    ``forbidden`` — a gather value table / scatter destination stays at the same
    base address on every core, its row chosen at runtime by IndirectAccess.
    Splitting a data (non-row) dimension would give each core a different base
    into that shared table and miscompile, so those dims must NEVER be split — a
    hard constraint : span reduction must not split them either. Also covers the
    partial-last-stick index-entry dim.

    ``force_output`` — a scatter's destination row is runtime-chosen, so its
    index-entry dim never appears in the output coordinates and would otherwise
    be classed as a reduction dim and left unsplit. Promoting it lets each core
    write a disjoint set of source rows in parallel.

    Together these supersede the old blanket single-core pin: the entry/output
    dims stay splittable, enabling multicore indirect access.
    """
    forbidden = indirect_forbidden_split_syms(ctx.op)
    return ConstraintResult(
        forbidden=forbidden,
        force_output=indirect_store_entry_syms(ctx.op, forbidden),
    )

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
from .pass_utils import concretize_expr, indirect_info_from_op, op_read_writes
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
    """

    blocked: set[Symbol] = dataclasses.field(default_factory=set)
    pinned: dict[Symbol, int] = dataclasses.field(default_factory=dict)


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
    for constraint in (
        coordinate_mask_blocked_vars,
        conv_spatial_blocked_vars,
        qfp8wt_pinned_vars,
        qfp8wt_matmul_k_pinned,
        indirect_access_pinned_vars,
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

    return ConstraintResult(blocked=blocked, pinned=pinned)


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


def indirect_access_pinned_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin every dim to split=1 for ops with indirect (gather/scatter-style) access.

    The backend's indirect-addressing path runs single-core: an indexed
    dimension's coordinate depends on runtime data, not a static per-core
    offset, so the generic per-core span/coordinate arithmetic does not apply.
    """
    dep_names, _, _ = indirect_info_from_op(ctx.op)
    if not dep_names:
        return ConstraintResult()
    return ConstraintResult(pinned={v: 1 for v in ctx.it_space_adjusted})

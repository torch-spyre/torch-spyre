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

"""Declarative coarse tiling, applied inside the scratchpad planning pass.

The tiling is stated as data (a
:class:`~torch_spyre._inductor.scratchpad.plan_solver.TileSpec` per op) and
*applied* to a real graph through a :class:`ScratchpadOptimizationPass`. The
tiling is an input here, not a search -- candidate enumeration and the solver
that chooses among tilings live elsewhere.

The pass mints hint ids and a group-id offset from bases derived off the graph
(never a reserved constant), so a tiling applied here cannot collide with a
hint-driven group already stamped pre-stickification at pass 430. It reuses the
existing ``coarse_tile`` machinery verbatim; the only new work is lowering a
``TileSpec`` to per-op ``DimHint``s and deriving groups as consecutive runs of
ops that share a spec.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import sympy

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation, Reduction

from ..errors import Unsupported
from ..logging_utils import get_inductor_logger
from ..pass_utils import iteration_space_from_op, op_out_coords
from ..propagate_hints import DimHint
from ..wsr.coarse_tile import (
    _loop_var_to_reduction_ranges_pos,
    coarse_tile_post_stickify,
    reduction_loop_vars,
    validate_coarse_tile_groups,
)
from .allocator import ScratchpadOptimizationPass
from .plan_solver import TileSpec

logger = get_inductor_logger("scratchpad.coarse_tiling")


def _get_out_var(
    op: ComputedBuffer,
    out_coords: list[sympy.Expr],
    iter_space: Mapping[sympy.Symbol, sympy.Expr],
    host_dim: int,
) -> tuple[sympy.Symbol | None, str | None]:
    """``(loop_var, None)`` for output ``host_dim``, or ``(None, reason)``.

    The output-axis step of :func:`try_resolve_tile_axis_loop_vars`.
    ``host_dim`` indexes ``out_coords``, the ``op_out_coords(op)`` the caller
    computes once for all of a spec's axes -- exactly ``_dims_to_hints`` (span
    overflow). Rejects a ``host_dim`` past the end of ``out_coords``, and a
    coordinate that is not a function of exactly one loop variable -- a
    constant, or several vars folded into one host dim -- since there is then no
    single loop to tile.

    Also rejects a coordinate whose one free symbol is not in ``iter_space``,
    the op's own iteration variables. ``op_out_coords`` evaluates the write
    index against indirect-index sizes and enclosing ``for_each_tile`` loop
    ranges, so a coordinate can be an indirect-index symbol or an enclosing
    loop's variable -- a symbol the op does not loop over, so there is no loop
    of its own to tile.
    """
    if host_dim >= len(out_coords):
        return None, (
            f"coarse tiling: host_dim={host_dim} is out of bounds "
            f"for {len(out_coords)} output coordinates on {op.get_name()}."
        )
    coord = out_coords[host_dim]
    free_symbols = coord.free_symbols
    if len(free_symbols) != 1:
        return None, (
            f"coarse tiling: host_dim={host_dim} output coordinate "
            f"{coord} on {op.get_name()} has {len(free_symbols)} free "
            "symbols; expected exactly one loop var."
        )
    loop_var = next(iter(free_symbols))
    if loop_var not in iter_space:
        return None, (
            f"coarse tiling: host_dim={host_dim} output coordinate {coord} on "
            f"{op.get_name()} is not one of its iteration variables "
            f"{list(iter_space)}."
        )
    return loop_var, None


def _get_red_var(
    op: ComputedBuffer, host_dim: int
) -> tuple[sympy.Symbol | None, str | None]:
    """``(loop_var, None)`` for reduction ``host_dim``, or ``(None, reason)``.

    The reduction-axis step of :func:`try_resolve_tile_axis_loop_vars`, the
    inverse of :func:`reduction_loop_vars`: ``host_dim`` positionally indexes
    the op's ordered reduction loop variables, which Inductor *squeezes* -- a
    size-1 reduction dim carries no loop variable and has no position.

    Rejects, in order: an op that is not a ``Reduction``; one with no write dep
    to derive loop variables from; one with no reduction loop variables, because
    no read dep carries one (``reduction_loop_vars`` returns none when the op
    loads no buffer); a ``host_dim`` past the end of its loop variables; and a
    loop variable the applier cannot place in ``reduction_ranges``.

    The last is not a lowering limit but an applier one. The applier picks the
    ``reduction_ranges`` entry to divide with
    ``_loop_var_to_reduction_ranges_pos``, which pairs the squeezed loop
    variables with the dims whose extent is not 1 and gives up when they do not
    pair one-to-one (say, a broadcast symbol leaks into the loop variables).
    Asking it here keeps lowering from handing the applier a hint it cannot
    place.
    """
    if not isinstance(op.data, Reduction):
        return None, (
            f"coarse tiling: reduction axis host_dim={host_dim} "
            f"requested on non-Reduction op {op.get_name()}."
        )
    try:
        red_vars = reduction_loop_vars(op)
    except StopIteration:
        return None, (
            f"coarse tiling: {op.get_name()} has no write dep to derive "
            "reduction loop variables from."
        )
    if not red_vars:
        return None, (
            f"coarse tiling: reduction host_dim={host_dim} on {op.get_name()}: "
            "no read dep carries a reduction loop variable, so there is no "
            "reduction loop to tile."
        )
    if host_dim >= len(red_vars):
        return None, (
            f"coarse tiling: reduction host_dim={host_dim} is out "
            f"of bounds for {len(red_vars)} reduction loop variables on "
            f"{op.get_name()}."
        )
    loop_var = red_vars[host_dim]
    if _loop_var_to_reduction_ranges_pos(op, loop_var) is None:
        return None, (
            f"coarse tiling: reduction host_dim={host_dim} on {op.get_name()}: "
            f"its {len(red_vars)} reduction loop variables do not pair "
            "one-to-one with the non-size-1 dims of reduction ranges "
            f"{list(op.data.reduction_ranges)}, so the applier cannot place "
            f"{loop_var} in them."
        )
    return loop_var, None


def try_resolve_tile_axis_loop_vars(
    op: ComputedBuffer, spec: TileSpec
) -> tuple[list[sympy.Symbol] | None, str | None]:
    """``(loop_vars, None)``, one per axis of ``spec``, or ``(None, reason)``.

    The single authority on which loop variable each :class:`TileAxis` names on
    ``op`` and on whether ``spec`` can be applied at all. It reports rather than
    raises so a caller weighing candidates nobody has committed to can treat a
    rejection as ordinary pruning; :func:`tile_spec_to_dim_hints` lowers a spec
    the planner has committed to, so it raises ``Unsupported`` with the
    ``reason``.

    ``host_dim`` is positional within one of two frames, selected by
    ``is_reduction``: ``op_out_coords(op)`` for an output axis
    (:func:`_get_out_var`), the squeezed reduction loop variables for a
    reduction axis (:func:`_get_red_var`).
    """
    out_coords = op_out_coords(op)
    iter_space = iteration_space_from_op(op)
    loop_vars: list[sympy.Symbol] = []
    for axis in spec.axes:
        if axis.is_reduction:
            loop_var, reason = _get_red_var(op, axis.host_dim)
        else:
            loop_var, reason = _get_out_var(op, out_coords, iter_space, axis.host_dim)
        if loop_var is None:
            return None, reason
        loop_vars.append(loop_var)
    return loop_vars, None


def tile_spec_to_dim_hints(
    op: ComputedBuffer,
    spec: TileSpec,
    hint_ids: Sequence[int],
) -> list[DimHint]:
    """Lower a :class:`TileSpec` into per-op :class:`DimHint`s.

    Each :class:`TileAxis` becomes one ``DimHint`` carrying the axis's split
    count and the op's *own* loop variable for that axis, paired with the group's
    ``hint_id`` for that level. ``hint_ids`` has one entry per axis, outermost
    first, matching the group's ``levels``.

    Which loop variable an axis names, and every ``Unsupported`` a spec can earn,
    belongs to :func:`try_resolve_tile_axis_loop_vars`; only the ``hint_ids``
    pairing lives here.
    """
    if len(hint_ids) != len(spec.axes):
        raise ValueError(
            f"tile_spec_to_dim_hints: {len(hint_ids)} hint_ids for "
            f"{len(spec.axes)} axes on {op.get_name()}"
        )
    loop_vars, reason = try_resolve_tile_axis_loop_vars(op, spec)
    if loop_vars is None:
        raise Unsupported(reason)
    return [
        DimHint(
            dim_names=["_coarse_tile"],
            split_count=axis.count,
            loop_var=loop_var,
            is_reduction=axis.is_reduction,
            hint_id=hint_id,
        )
        for axis, loop_var, hint_id in zip(spec.axes, loop_vars, hint_ids)
    ]


@dataclasses.dataclass(frozen=True)
class PrescribedRegion:
    """The ops one outermost ``for_each_tile`` loop covers.

    ``for_each_tile`` loops are authoritative: their axes, nesting and trip
    counts are the user's, so no compiler tiling may re-tile an op inside one.

    ``start``/``stop`` bound the region in ``graph.operations`` (``stop`` is
    exclusive), and ``names`` holds every operation name in that slice.
    ``unstamped`` names the ones no loop level stamped.  Lowering can place an
    op that runs once, outside the loop, between the loop's body ops (the
    buffer a map loop writes its tiles into, for one), and a later pass can
    insert an op inside the loop without copying its metadata.  Both are held
    with the region: tiling either would start a loop group in the middle of
    the user's loop.
    """

    loop_group_id: int
    start: int
    stop: int
    names: frozenset[str]
    unstamped: tuple[str, ...]


def _is_spliced(op: Operation) -> bool:
    """Whether ``splice_while_loops`` stamped ``op`` as part of a loop level.

    ``_stamp_direct_loop_info`` appends a ``DimHint`` with ``loop_var_range``
    set to every op of each level; no other tiling source sets that field
    (see ``loop_var_ranges_from_dim_hints`` in pass_utils.py).
    """
    return getattr(op, "loop_info", None) is not None and any(
        h.loop_var is not None and h.loop_var_range is not None
        for h in getattr(op, "dim_hints", None) or []
    )


def prescribed_regions(operations: Sequence[Operation]) -> list[PrescribedRegion]:
    """Every ``for_each_tile`` region in ``operations``, in operation order.

    A region is seeded by the spliced ops sharing an outermost
    ``loop_group_id``, and extends over every op between the first and last of
    them.  Membership is by position rather than by attribute alone, so an op
    sitting between the loop's body ops is covered whether or not a loop level
    stamped it.
    """
    spans: dict[int, list[int]] = {}
    for pos, op in enumerate(operations):
        if not _is_spliced(op):
            continue
        outer = op.loop_info.loop_group_id[0]  # type: ignore[attr-defined]
        span = spans.setdefault(outer, [pos, pos])
        span[1] = pos
    regions: list[PrescribedRegion] = []
    for group_id, (first, last) in sorted(spans.items(), key=lambda kv: kv[1][0]):
        members = operations[first : last + 1]
        unstamped = tuple(
            op.get_operation_name() for op in members if not _is_spliced(op)
        )
        if unstamped:
            logger.debug(
                "for_each_tile region %d also holds %s, which no loop level stamped",
                group_id,
                ", ".join(unstamped),
            )
        regions.append(
            PrescribedRegion(
                loop_group_id=group_id,
                start=first,
                stop=last + 1,
                names=frozenset(op.get_operation_name() for op in members),
                unstamped=unstamped,
            )
        )
    return regions


def _host_dim_walk(
    op: ComputedBuffer, dep: MemoryDep, host_dim: int
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """``(stride, extent)`` of ``dep`` along the loop var at output ``host_dim``
    of ``op``, or ``None`` where that is not a single affine loop var."""
    coords = op_out_coords(op)
    if host_dim >= len(coords) or len(coords[host_dim].free_symbols) != 1:
        return None
    var = next(iter(coords[host_dim].free_symbols))
    stride = sympy.diff(dep.index, var)
    extent = dict(zip(dep.var_names, dep.size)).get(var)
    if stride == 0 or stride.free_symbols or extent is None:
        return None
    return stride, extent


def tile_aligned_host_dims(
    op: ComputedBuffer, producer: ComputedBuffer
) -> frozenset[int] | None:
    """The output host dims of ``producer`` that ``op`` walks, in every read of
    it, along the same buffer dim at the same extent as ``producer`` writes
    them; ``None`` where no spec is read as written (a read or the write that
    is not a :class:`MemoryDep`).

    ``TileAxis.host_dim`` is positional, so a shared spec is not a shared
    logical dim: across ``[A, S, D] -> [S, A, D]`` host dim 0 is ``A`` on one
    side and ``S`` on the other, and the reader would index the writer's
    per-tile scratch as though it held a tile of ``S``. Anything not provably
    aligned counts as misaligned.
    """
    name = producer.get_name()
    write = next(
        (
            w
            for w in producer.get_read_writes().writes
            if isinstance(w, MemoryDep) and w.name == name
        ),
        None,
    )
    if write is None:
        return None
    dims = set(range(len(op_out_coords(producer))))
    for read in op.get_read_writes().reads:
        if read.name != name:
            continue
        if not isinstance(read, MemoryDep):
            return None
        dims = {
            d
            for d in dims
            if (walk := _host_dim_walk(producer, write, d)) is not None
            and walk == _host_dim_walk(op, read, d)
        }
    return frozenset(dims)


def _reads_tiles_as_written(
    op: Operation,
    by_buf: Mapping[str, ComputedBuffer],
    spec: TileSpec,
) -> bool:
    """Whether ``op`` reads each op of ``by_buf`` it reads with every tiled
    output axis of ``spec`` walked as that op writes it."""
    if not isinstance(op, ComputedBuffer):
        return True
    for name in {read.name for read in op.get_read_writes().reads}:
        producer = by_buf.get(name)
        if producer is not None and not spec.read_as_written(
            tile_aligned_host_dims(op, producer)
        ):
            return False
    return True


def derive_tiling_groups(
    graph: GraphLowering,
    choices: Mapping[str, TileSpec],
) -> list[tuple[list[Operation], TileSpec]]:
    """Group consecutive ops that share the same non-empty :class:`TileSpec`.

    Mirrors ``hints_to_coarse_tile_groups``' consecutive-run shape with the hint
    key replaced by the chosen ``TileSpec``: a run breaks whenever an op is
    untiled (absent from ``choices`` or mapped to the empty spec) or its spec
    differs from the run's. Contiguity is a hard requirement, not an
    optimization -- ``validate_coarse_tile_groups`` and ``_apply_plan`` both rely
    on each group occupying one contiguous stretch of the operation list.

    Two non-adjacent runs carrying the same spec are therefore **two groups**,
    each minting its own hint ids and group id. They are not the same group and
    not an error: ``TileSpec`` equality is structural, so unrelated regions
    anywhere in the graph collide on a small alphabet (~6 counts per axis over
    at most two dims), and refusing them would refuse ordinary graphs.

    **Precondition on the caller, which this signature cannot check.** Ops meant
    to tile together have to be contiguous in ``graph.operations``. A chooser
    walking producer/consumer reachability is not walking contiguity: an op it
    could not tile -- a menu-backed one, or one already carrying ``dim_hints``
    -- sitting in the middle of a region leaves the second half reading the
    first half's *full* extent while the chooser priced both at the per-tile
    footprint. That is a mispricing rather than an illegal graph, and a
    name->spec map carries no region identity to detect it with, so it belongs
    to whoever builds ``choices``. ``_validate_contiguous`` remains the backstop
    for the illegal case.

    A run also breaks at an op that reads an op of its *stretch* -- the ops
    since the spec last changed, not only its group -- along a different
    logical dim than the spec tiles it by (:func:`tile_aligned_host_dims`):
    one group would hand it the wrong slice, two make it a cross-group read of
    the full buffer. Taking the stretch over-breaks only where that op is
    already in an earlier group, and makes where a group starts depend on the
    specs between an op and what it reads alone, which the SA co-optimizer
    re-derives per move (``SaCoOptimizingSolver._run_bounds``).

    ``choices`` is keyed by operation name (``op.get_operation_name()``).
    """
    groups: list[tuple[list[Operation], TileSpec]] = []
    current_ops: list[Operation] = []
    current_spec: TileSpec | None = None
    stretch_by_buf: dict[str, ComputedBuffer] = {}
    for op in graph.operations:
        spec = choices.get(op.get_operation_name())
        if spec is not None and spec.is_untiled:
            spec = None
        if spec != current_spec:
            stretch_by_buf = {}
        if (
            spec is not None
            and spec == current_spec
            and _reads_tiles_as_written(op, stretch_by_buf, spec)
        ):
            current_ops.append(op)
        else:
            if current_ops:
                assert current_spec is not None
                groups.append((current_ops, current_spec))
            current_ops = [op] if spec is not None else []
            current_spec = spec
        if spec is not None and isinstance(op, ComputedBuffer):
            stretch_by_buf[op.get_name()] = op
    if current_ops:
        assert current_spec is not None
        groups.append((current_ops, current_spec))
    return groups


def _derive_hint_id_base(graph: GraphLowering) -> int:
    """``max(hint_id present in the graph, default=-1) + 1``.

    Derived, never a reserved constant: whatever hint ids a pre-stickification
    hint-driven group already minted, this pass mints strictly above them, so
    ``validate_coarse_tile_groups`` can never see a hint id in two groups.
    """
    ids = [h.hint_id for op in graph.operations for h in getattr(op, "dim_hints", [])]
    return max(ids, default=-1) + 1


def _derive_group_idx_offset(graph: GraphLowering) -> int:
    """``max(loop_group_id[0] present, default=-1) + 1`` -- the same derivation
    ``_maybe_coarse_tile_span_overflow`` uses to avoid a ``loop_group_id``
    collision with a hint-driven group stamped pre-stickification."""
    used = [
        op.loop_info.loop_group_id[0]
        for op in graph.operations
        if getattr(op, "loop_info", None) is not None
    ]
    return max(used, default=-1) + 1


class CoarseTilingPass(ScratchpadOptimizationPass):
    """Apply a declared coarse tiling to a graph, inside the scratchpad pass.

    The tiling is an *input* (``choices``: operation name -> TileSpec),
    not a search. Consecutive ops sharing a non-empty spec form one loop group;
    the pass mints hint ids and a group-id offset from bases derived off the
    graph, stamps each op's ``dim_hints``, validates group contiguity, then calls
    ``coarse_tile``. With empty (or all-untiled) ``choices`` it is a no-op and
    the op count is unchanged -- which is what keeps it inert until a solver
    hands it real choices.
    """

    def __init__(self, choices: Mapping[str, TileSpec]):
        self._choices = dict(choices)

    def apply_pass(self, graph: GraphLowering) -> None:
        groups_specs = derive_tiling_groups(graph, self._choices)
        if not groups_specs:
            return
        # A for_each_tile region's tiling is the user's and already stamped;
        # re-tiling one of its ops would overwrite that op's dim_hints and
        # loop_info.  Candidate selection is expected to hold region ops
        # untiled, so reaching this is a bug upstream of the pass.
        region_ops = {
            name
            for region in prescribed_regions(graph.operations)
            for name in region.names
        }
        for group_ops, spec in groups_specs:
            clash = [
                op.get_operation_name()
                for op in group_ops
                if op.get_operation_name() in region_ops
            ]
            if clash:
                raise Unsupported(
                    f"coarse tiling: {spec} would re-tile {', '.join(clash)}, "
                    "which a for_each_tile loop already tiles."
                )
        # Both bases are derived off the graph *before* this pass stamps any of
        # its own hints/groups, so pre-existing (hint-driven) ids are avoided
        # and the ids this pass mints increase monotonically.
        next_hint_id = _derive_hint_id_base(graph)
        group_idx_offset = _derive_group_idx_offset(graph)
        groups: list[tuple] = []
        for group_ops, spec in groups_specs:
            hint_ids = list(range(next_hint_id, next_hint_id + len(spec.axes)))
            next_hint_id += len(spec.axes)
            levels = [
                (hint_id, sympy.Integer(axis.count))
                for hint_id, axis in zip(hint_ids, spec.axes)
            ]
            for op in group_ops:
                op.dim_hints = tile_spec_to_dim_hints(op, spec, hint_ids)
            groups.append((group_ops, levels))
        validate_coarse_tile_groups(groups)
        # This pass runs inside scratchpad/LX planning -- after stickification
        # (insert_restickify) and the post-stickify span-overflow WSR pass -- so
        # every op already carries a committed FixedTiledLayout. Use the
        # post-stickify entry point (run_read_copies=False): a read copy-in here
        # would only be a useless HBM-to-HBM copy, exactly as the sibling
        # post-stickify consumer (_maybe_coarse_tile_span_overflow) does.
        coarse_tile_post_stickify(
            graph, groups=groups, group_idx_offset=group_idx_offset
        )

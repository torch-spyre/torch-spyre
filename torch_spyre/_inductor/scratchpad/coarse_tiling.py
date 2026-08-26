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

This is stage 1 of the coarse-tiling optimization: a tiling stated as data (a
:class:`~torch_spyre._inductor.scratchpad.plan_solver.TileSpec` per op) and
*applied* to a real graph through a :class:`ScratchpadOptimizationPass`. The
tiling is an input here, not a search -- candidate enumeration and the solver
that chooses among tilings arrive in later stages.

The pass mints hint ids and a group-id offset from bases derived off the graph
(never a reserved constant), so a tiling applied here cannot collide with a
hint-driven group already stamped pre-stickification at pass 430. It reuses the
existing ``coarse_tile`` machinery verbatim; the only new work is lowering a
``TileSpec`` to per-op ``DimHint``s and deriving groups as consecutive runs of
ops that share a spec.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import sympy

from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation, Reduction

from ..errors import Unsupported
from ..pass_utils import op_out_coords
from ..propagate_hints import DimHint
from ..wsr.coarse_tile import (
    _loop_var_to_ranges_pos,
    _loop_var_to_reduction_ranges_pos,
    coarse_tile_post_stickify,
    reduction_loop_vars,
    validate_coarse_tile_groups,
)
from .passes import ScratchpadOptimizationPass
from .plan_solver import TileAxis, TileSpec


def tile_spec_to_dim_hints(
    op: ComputedBuffer,
    spec: TileSpec,
    hint_ids: Sequence[int],
    dim_names: Sequence[Sequence[str]] | None = None,
) -> list[DimHint]:
    """Lower a :class:`TileSpec` into per-op :class:`DimHint`s.

    Each :class:`TileAxis` becomes one ``DimHint`` carrying the axis's split
    count and the op's *own* loop variable for that axis, paired with the group's
    ``hint_id`` for that level. ``hint_ids`` has one entry per axis, outermost
    first, matching the group's ``levels``.

    ``dim_names`` defaults to the discovered-tiling label ``["_coarse_tile"]`` on
    every axis. A caller re-applying a *carried* pin passes the pin's own names
    per axis so the applied level keeps the caller's dim name (e.g. ``"S"``)
    rather than being relabelled as compiler-discovered -- the identity a
    consumer keying on the pin (debug output, the hint-preservation tests) reads.

    The output-axis case is exactly ``_dims_to_hints`` (span overflow): resolve
    the loop var from ``op_out_coords(op)[host_dim]``. The reduction-axis case is
    the inverse of :func:`reduction_loop_vars` -- ``host_dim`` positionally
    indexes the op's ordered reduction loop variables.
    """
    if len(hint_ids) != len(spec.axes):
        raise ValueError(
            f"tile_spec_to_dim_hints: {len(hint_ids)} hint_ids for "
            f"{len(spec.axes)} axes on {op.get_name()}"
        )
    if dim_names is not None and len(dim_names) != len(spec.axes):
        raise ValueError(
            f"tile_spec_to_dim_hints: {len(dim_names)} dim_names for "
            f"{len(spec.axes)} axes on {op.get_name()}"
        )
    out_coords = op_out_coords(op)
    red_vars: list[sympy.Symbol] | None = None
    hints: list[DimHint] = []
    for i, (axis, hint_id) in enumerate(zip(spec.axes, hint_ids)):
        if axis.is_reduction:
            if not isinstance(op.data, Reduction):
                raise Unsupported(
                    f"coarse tiling: reduction axis host_dim={axis.host_dim} "
                    f"requested on non-Reduction op {op.get_name()}."
                )
            if red_vars is None:
                red_vars = reduction_loop_vars(op)
            if axis.host_dim >= len(red_vars):
                raise Unsupported(
                    f"coarse tiling: reduction host_dim={axis.host_dim} is out "
                    f"of bounds for {len(red_vars)} reduction loop variables on "
                    f"{op.get_name()}."
                )
            loop_var = red_vars[axis.host_dim]
        else:
            if axis.host_dim >= len(out_coords):
                raise Unsupported(
                    f"coarse tiling: host_dim={axis.host_dim} is out of bounds "
                    f"for {len(out_coords)} output coordinates on {op.get_name()}."
                )
            coord = out_coords[axis.host_dim]
            free_symbols = coord.free_symbols
            if len(free_symbols) != 1:
                raise Unsupported(
                    f"coarse tiling: host_dim={axis.host_dim} output coordinate "
                    f"{coord} on {op.get_name()} has {len(free_symbols)} free "
                    "symbols; expected exactly one loop var."
                )
            loop_var = next(iter(free_symbols))
        hints.append(
            DimHint(
                dim_names=(
                    ["_coarse_tile"] if dim_names is None else list(dim_names[i])
                ),
                split_count=axis.count,
                loop_var=loop_var,
                is_reduction=axis.is_reduction,
                hint_id=hint_id,
            )
        )
    return hints


def dim_hints_to_tile_spec(
    op: ComputedBuffer,
    dim_hints: Sequence[DimHint],
) -> TileSpec:
    """Lift an op's :class:`DimHint`s back into the :class:`TileSpec` they tile.

    The inverse of :func:`tile_spec_to_dim_hints`: where that lowers each
    :class:`TileAxis` to the loop var it tiles, this recovers the axis's
    positional ``host_dim`` from the hint's ``loop_var``. Both directions go
    through the same resolvers -- ``_loop_var_to_ranges_pos`` for an output axis,
    ``_loop_var_to_reduction_ranges_pos`` for a reduction one -- so a spec
    round-trips through a lowering and back unchanged.

    Only hints that actually produce a loop level become axes, applying the same
    two filters ``_hints_levels`` uses to decide a group's nest: a hint the op is
    broadcast against (``loop_var is None``) and a split of 1 both tile nothing.
    What survives is ordered outermost-first by ``hint_id`` (``spyre_hint``'s
    counter increases inwards), which is exactly the order a :class:`TileSpec`
    nests its axes in.

    This is how a pin can reach the tile search as a *constraint* rather than a
    mutation: the pin lives on the op as a ``DimHint`` (minted by
    ``assign_dim_hints``, never applied to the graph), and the solver reads its
    ``TileSpec`` here to restrict the op's candidates to the tilings that honor
    it. The ``hint_id`` is not carried onto the axis -- ``TileAxis`` has no field
    for it -- but the axis order preserves it positionally, so a caller that must
    re-stamp the caller's original id on apply can recover it by pairing the
    sorted hints back against ``spec.axes``.

    Raises ``Unsupported`` if a level-producing hint's ``loop_var`` cannot be
    placed on ``op``: a hint naming an axis the op does not carry is not a tiling
    this op can take, and silently dropping it would understate the nest.
    """
    leveled = sorted(
        (h for h in dim_hints if h.loop_var is not None and h.split_count != 1),
        key=lambda h: h.hint_id,
    )
    if not leveled:
        return TileSpec()
    out_coords = op_out_coords(op)
    axes: list[TileAxis] = []
    for h in leveled:
        if h.is_reduction:
            if not isinstance(op.data, Reduction):
                raise Unsupported(
                    f"dim_hints_to_tile_spec: reduction hint_{h.hint_id} on "
                    f"non-Reduction op {op.get_name()}."
                )
            host_dim = _loop_var_to_reduction_ranges_pos(op, h.loop_var)
        else:
            host_dim = _loop_var_to_ranges_pos(out_coords, h.loop_var)
        if host_dim is None:
            kind = "reduction" if h.is_reduction else "output"
            raise Unsupported(
                f"dim_hints_to_tile_spec: hint_{h.hint_id}'s loop var "
                f"{h.loop_var} is not an {kind} axis of {op.get_name()}."
            )
        axes.append(
            TileAxis(
                host_dim=host_dim, count=h.split_count, is_reduction=h.is_reduction
            )
        )
    return TileSpec(axes=tuple(axes))


def _carried_pins(op: Operation) -> dict[tuple[int, int, bool], DimHint]:
    """Index ``op``'s un-applied hints by the axis each tiles.

    The key is ``(host_dim, count, is_reduction)`` -- the identity a chosen
    :class:`TileAxis` matches on -- and the value is the ``DimHint`` it came
    from, so :class:`CoarseTilingPass` can recover a carried pin's ``hint_id``
    and dim name when it applies a spec that contains that axis. Reuses
    :func:`dim_hints_to_tile_spec`'s own filter and ordering, then pairs each
    resolved axis back with its origin hint. Empty when the op carries no
    level-producing hint (the unhinted / discovered-only cases) or when a hint
    fails to resolve on the op -- either way the reuse it feeds is inert.
    """
    dim_hints = list(getattr(op, "dim_hints", None) or [])
    leveled = sorted(
        (h for h in dim_hints if h.loop_var is not None and h.split_count != 1),
        key=lambda h: h.hint_id,
    )
    if not leveled:
        return {}
    try:
        spec = dim_hints_to_tile_spec(op, dim_hints)
    except Unsupported:
        return {}
    return {
        (axis.host_dim, axis.count, axis.is_reduction): h
        for axis, h in zip(spec.axes, leveled)
    }


def _find_carried_pin(
    pins_by_op: Mapping[str, Mapping[tuple[int, int, bool], DimHint]],
    group_ops: Sequence[Operation],
    axis: TileAxis,
) -> DimHint | None:
    """The carried pin some group op holds for ``axis``, or ``None``.

    A group shares one spec, so any member that carried the pin identifies it;
    the first match wins.
    """
    key = (axis.host_dim, axis.count, axis.is_reduction)
    for op in group_ops:
        pin = pins_by_op.get(op.get_name(), {}).get(key)
        if pin is not None:
            return pin
    return None


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
    on each group occupying one contiguous stretch of the operation list, so a
    connected component that skipped an intervening untiled op would be rejected
    at apply time.

    ``choices`` is keyed by operation name (``op.get_operation_name()``).
    """
    groups: list[tuple[list[Operation], TileSpec]] = []
    current_ops: list[Operation] = []
    current_spec: TileSpec | None = None
    for op in graph.operations:
        spec = choices.get(op.get_operation_name())
        if spec is not None and spec.is_untiled:
            spec = None
        if spec is not None and spec == current_spec:
            current_ops.append(op)
        else:
            if current_ops:
                assert current_spec is not None
                groups.append((current_ops, current_spec))
            current_ops = [op] if spec is not None else []
            current_spec = spec
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

    Stage 1: the tiling is an *input* (``choices``: operation name -> TileSpec),
    not a search. Consecutive ops sharing a non-empty spec form one loop group;
    the pass mints hint ids and a group-id offset from bases derived off the
    graph, stamps each op's ``dim_hints``, validates group contiguity, then calls
    ``coarse_tile``. With empty (or all-untiled) ``choices`` it is a no-op and
    the op count is unchanged -- which is what keeps it inert while
    ``auto_coarse_tiling`` is off.
    """

    def __init__(self, choices: Mapping[str, TileSpec]):
        self._choices = dict(choices)

    def apply_pass(self, graph: GraphLowering) -> None:
        groups_specs = derive_tiling_groups(graph, self._choices)
        if not groups_specs:
            return
        # Snapshot each op's carried pins before the loop below overwrites any
        # dim_hints. A carried pin is a caller hint that reached the solve as
        # data (``_solver_owns_tiling``) instead of a pre-stickification tiling;
        # reusing its id/name below is what lets a pin keep its identity after
        # being applied here rather than pre-stickification.
        pins_by_op = {
            op.get_name(): _carried_pins(op)
            for group_ops, _ in groups_specs
            for op in group_ops
        }
        # Fresh ids for compiler-discovered levels start above every id already
        # on the graph -- the carried pins included, so a reused pin id (small)
        # never clashes with a minted one. The group-id offset is derived the
        # same way, both *before* this pass stamps anything of its own.
        next_fresh_id = _derive_hint_id_base(graph)
        group_idx_offset = _derive_group_idx_offset(graph)
        # A pin can spread across ops the solve split into different spec-groups,
        # but ``validate_coarse_tile_groups`` forbids one hint id in two groups.
        # So a pin's own id is reused by the *first* group that carries its axis
        # and every later group mints a fresh id for that axis instead: the pin
        # still surfaces under its own id (and name) at least once -- enough to
        # identify it -- while the rest reads as discovered. This assumes a pin
        # nests outermost (its small id sorts outer), which holds for every pin
        # the caller places today; a pin used as an inner level would want its id
        # ordered against the discovered ones, not simply reused.
        claimed_pin_ids: set[int] = set()
        groups: list[tuple] = []
        for group_ops, spec in groups_specs:
            axis_ids: list[int] = []
            axis_names: list[list[str]] = []
            for axis in spec.axes:
                pin = _find_carried_pin(pins_by_op, group_ops, axis)
                if pin is not None and pin.hint_id not in claimed_pin_ids:
                    claimed_pin_ids.add(pin.hint_id)
                    axis_ids.append(pin.hint_id)
                    axis_names.append(list(pin.dim_names))
                else:
                    axis_ids.append(next_fresh_id)
                    axis_names.append(["_coarse_tile"])
                    next_fresh_id += 1
            levels = [
                (hint_id, sympy.Integer(axis.count))
                for hint_id, axis in zip(axis_ids, spec.axes)
            ]
            for op in group_ops:
                op.dim_hints = tile_spec_to_dim_hints(op, spec, axis_ids, axis_names)
            groups.append((group_ops, levels))
        validate_coarse_tile_groups(groups)
        coarse_tile_post_stickify(
            graph, groups=groups, group_idx_offset=group_idx_offset
        )

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
    coarse_tile_post_stickify,
    reduction_loop_vars,
    validate_coarse_tile_groups,
)
from .passes import ScratchpadOptimizationPass
from .plan_solver import TileSpec


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
    out_coords = op_out_coords(op)
    red_vars: list[sympy.Symbol] | None = None
    hints: list[DimHint] = []
    for axis, hint_id in zip(spec.axes, hint_ids):
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
                dim_names=["_coarse_tile"],
                split_count=axis.count,
                loop_var=loop_var,
                is_reduction=axis.is_reduction,
                hint_id=hint_id,
            )
        )
    return hints


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

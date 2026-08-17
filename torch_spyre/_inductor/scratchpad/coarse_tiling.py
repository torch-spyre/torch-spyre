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

"""Apply a declaratively-stated coarse tiling inside the scratchpad pass.

This is the apply half of the coarse-tiling optimization
(``docs/source/rfcs/draft-unified-tiling-implementation-plan.md``, stage 1):
a tiling stated as :class:`TileSpec` data is lowered to the same per-op
``DimHint``/group inputs the existing hint- and span-overflow-driven paths
feed ``coarse_tile`` — no new application path. The *choice* of tiling is an
input here; enumerating and solving for it arrive in later stages.

Run as a :class:`ScratchpadOptimizationPass` pre-optimization pass, the
mutation happens before ``ScratchpadAllocator._prepare_buffers``, so the
buffer set the placement solve sees is built from the already-tiled graph:
sizes and lifetimes are measured, not predicted, and the ops tiling inserts
are enumerated and divided like any others.

There are no reserved id namespaces here. Hint ids and the group index
offset are both minted from a base derived off the graph
(``max(id present, default=-1) + 1``), so a hint-driven group stamped
pre-stickification can never collide with one applied by this pass.
"""

import sympy

from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation

from ..errors import Unsupported
from ..logging_utils import get_inductor_logger
from ..propagate_hints import DimHint
from ..wsr.coarse_tile import (
    coarse_tile,
    reduction_loop_vars,
    validate_coarse_tile_groups,
)
from ..wsr.coarse_tile_span_overflow import _dims_to_hints
from .passes import ScratchpadOptimizationPass
from .plan_solver import TileSpec

logger = get_inductor_logger("scratchpad.coarse_tiling")

# Provenance marker stamped into DimHint.dim_names for hints this pass mints.
_COARSE_TILING_DIM_NAME = "_coarse_tiling"


def tile_spec_to_dim_hints(
    op: ComputedBuffer,
    spec: TileSpec,
    hint_ids: list[int],
) -> list[DimHint]:
    """Lower ``spec`` to per-op ``DimHint``s, one per tile level.

    ``hint_ids`` pairs positionally with ``spec.axes`` (outermost first) and
    must be ascending: outer hint ids sort smaller, which is how
    ``_hints_levels`` recovers the level order.

    An output axis resolves its ``loop_var`` from the op's own output
    coordinates (via the span-overflow path's ``_dims_to_hints``, which is
    already exactly this lowering); a reduction axis indexes the op's ordered
    reduction loop variables — the inverse of :func:`reduction_loop_vars`.
    """
    assert len(hint_ids) == spec.depth, (
        f"need one hint id per tile level: {len(hint_ids)} ids for "
        f"depth-{spec.depth} spec"
    )
    assert list(hint_ids) == sorted(hint_ids), (
        f"hint ids must ascend outermost-first, got {hint_ids}"
    )

    out_pairs = [
        (ax, hid) for ax, hid in zip(spec.axes, hint_ids) if not ax.is_reduction
    ]
    red_pairs = [(ax, hid) for ax, hid in zip(spec.axes, hint_ids) if ax.is_reduction]

    hints: list[DimHint] = []
    if out_pairs:
        dims = tuple((ax.host_dim, ax.count, False) for ax, _ in out_pairs)
        hints.extend(
            _dims_to_hints(
                op,
                dims,
                [hid for _, hid in out_pairs],
                dim_name=_COARSE_TILING_DIM_NAME,
            )
        )
    if red_pairs:
        rvars = reduction_loop_vars(op)
        for ax, hid in red_pairs:
            if ax.host_dim >= len(rvars):
                raise Unsupported(
                    f"Cannot lower tiling for {op.get_name()}: reduction "
                    f"host_dim={ax.host_dim} is out of bounds for "
                    f"{len(rvars)} reduction loop variables."
                )
            hints.append(
                DimHint(
                    dim_names=[_COARSE_TILING_DIM_NAME],
                    split_count=ax.count,
                    loop_var=rvars[ax.host_dim],
                    is_reduction=True,
                    hint_id=hid,
                )
            )
    return hints


def derive_tiling_groups(
    operations: list[Operation],
    choices: dict[str, TileSpec],
) -> list[tuple[list[Operation], TileSpec]]:
    """Split the chosen tilings into contiguous groups of identically-tiled ops.

    A group is a consecutive run over ``operations`` sharing one
    :class:`TileSpec`, breaking on an untiled op, a spec change, or any op
    that is not a ``ComputedBuffer``. Consecutive-run, not connected
    components: contiguity is a hard requirement of ``_apply_plan`` and
    ``validate_coarse_tile_groups``, so this is
    ``hints_to_coarse_tile_groups``'s own shape with the hint-id key replaced
    by the chosen spec.
    """
    groups: list[tuple[list[Operation], TileSpec]] = []
    current_ops: list[Operation] = []
    current_spec: TileSpec | None = None

    def _flush() -> None:
        nonlocal current_ops, current_spec
        if current_ops:
            assert current_spec is not None
            groups.append((current_ops, current_spec))
        current_ops = []
        current_spec = None

    for op in operations:
        spec = choices.get(op.get_name()) if isinstance(op, ComputedBuffer) else None
        if spec is not None and spec.is_untiled:
            spec = None
        if spec is None:
            _flush()
        elif spec == current_spec:
            current_ops.append(op)
        else:
            _flush()
            current_ops = [op]
            current_spec = spec
    _flush()
    return groups


def _derived_hint_id_base(operations: list[Operation]) -> int:
    """First hint id free in this graph: max present + 1, or 0."""
    present = [h.hint_id for op in operations for h in getattr(op, "dim_hints", [])]
    return max(present, default=-1) + 1


def _derived_group_idx_offset(operations: list[Operation]) -> int:
    """First outer loop-group id free in this graph: max stamped + 1, or 0.

    Same derivation ``_maybe_coarse_tile_span_overflow`` uses, so groups this
    pass applies cannot collide with hint-driven groups stamped earlier.
    """
    used = [
        op.loop_info.loop_group_id[0]
        for op in operations
        if getattr(op, "loop_info", None) is not None
    ]
    return max(used, default=-1) + 1


class CoarseTilingPass(ScratchpadOptimizationPass):
    """Apply chosen per-op tilings to the graph, through ``coarse_tile``.

    ``choices`` maps buffer name -> :class:`TileSpec`. Empty specs and names
    absent from the graph are inert; everything else is lowered to
    ``DimHint``s, grouped by consecutive spec equality, and applied. The pass
    is a pure applier: it never decides a tiling, so a solve that fails must
    simply not construct it (keeping the ``SolveError`` fallback upstream of
    any IR mutation).
    """

    def __init__(self, choices: dict[str, TileSpec]):
        self.choices = {
            name: spec for name, spec in choices.items() if not spec.is_untiled
        }

    def apply_pass(self, graph: GraphLowering) -> None:
        if not self.choices:
            return
        operations = graph.operations
        tiling_groups = derive_tiling_groups(operations, self.choices)
        if not tiling_groups:
            return

        hint_id_base = _derived_hint_id_base(operations)
        group_idx_offset = _derived_group_idx_offset(operations)

        groups: list[tuple] = []
        for group_ops, spec in tiling_groups:
            hint_ids = list(range(hint_id_base, hint_id_base + spec.depth))
            hint_id_base += spec.depth
            for op in group_ops:
                assert not getattr(op, "dim_hints", []), (
                    f"CoarseTilingPass must not re-tile already-hinted op "
                    f"{op.get_name()}"
                )
                op.dim_hints = tile_spec_to_dim_hints(op, spec, hint_ids)
            levels = [
                (hid, sympy.Integer(ax.count)) for hid, ax in zip(hint_ids, spec.axes)
            ]
            groups.append((group_ops, levels))
            logger.debug(
                "coarse_tiling: group ops=%s spec=%s hint_ids=%s",
                [op.get_name() for op in group_ops],
                spec.label,
                hint_ids,
            )

        validate_coarse_tile_groups(groups)
        coarse_tile(graph, groups=groups, group_idx_offset=group_idx_offset)

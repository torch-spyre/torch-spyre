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

"""Enforce non-stick dimension ordering required by indirect-access ops.

The three-pass restickify pipeline (propagate_layouts -> optimize_restickify ->
insert_restickify) resolves stick-dimension layout constraints. Indirect-access
ops (gather/scatter) impose an additional requirement on non-stick dimension
ordering: the indexed dimension must be outermost in the value tensor's device
layout, based on their coordinate access patterns.

This pass runs after insert_restickify, once every op has a committed
FixedTiledLayout. For indirect-access ops, it checks whether the value tensor's
current dim_order matches this requirement; if not, either rewrites the
producer's output layout in place (if the producer is a ComputedBuffer and not a
graph output) or inserts a spyre.restickify copy in the required layout.
"""

import sympy

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, MutationLayoutSHOULDREMOVE

from torch_spyre._C import SpyreTensorLayout

from .constants import ELIDED_COPY_BACK_ATTR
from .insert_restickify import _fixed_tiled, insert_restickify_on_node_inputs
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess
from .pass_utils import device_coordinates, indirect_info_from_op

logger = get_inductor_logger("enforce_indirect_access_layout")


def _real_layout(buf) -> FixedTiledLayout:
    layout = buf.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        assert getattr(buf, ELIDED_COPY_BACK_ATTR, False), (
            f"unexpected mutation layout on {buf.get_name()!r}"
        )
        layout = layout.real_layout()
    return layout


def _indirect_stride_idx(coords: list[sympy.Expr], access_subs: dict) -> int | None:
    """Return the stride_idx (from right, 0-indexed) of the first IndirectAccess
    coordinate, or None if coords carry no indirect symbol."""
    for idx, coord in enumerate(reversed(coords)):
        substituted = coord.xreplace(access_subs) if access_subs else coord
        if hasattr(substituted, "has") and substituted.has(IndirectAccess):
            return idx
    return None


def _dim_order_is_compliant(value_stl: SpyreTensorLayout, stride_idx: int) -> bool:
    """Check if indirect access is at the outermost (leftmost) device position.

    For indirect access to work correctly, the IndirectAccess coordinate must
    be at device position 0 (the outermost dimension before non-stick and stick).
    This corresponds to stride_idx being positioned such that the indirect
    dimension is leftmost.
    """
    v_n = len(value_stl.stride_map)
    v_indirect_pos = v_n - 1 - stride_idx

    # Compliant if indirect is at position 0 (outermost device dim)
    compliant = v_indirect_pos == 0

    logger.debug(
        "_dim_order_is_compliant: v_n=%d, stride_idx=%d, indirect_pos=%d, compliant=%s",
        v_n,
        stride_idx,
        v_indirect_pos,
        compliant,
    )
    return compliant


def _build_required_stl(
    value_stl: SpyreTensorLayout,
    indirect_device_pos: int,
) -> SpyreTensorLayout:
    """Build a new STL with the indirect coordinate rotated to device position 0.

    Takes the current device layout and rotates it so the indirect coordinate
    (at indirect_device_pos) moves to position 0, while keeping the stick
    (at position -1) at the end. Returns a new STL with the rotated layout.
    """
    device_size = list(value_stl.device_size)
    stride_map = list(value_stl.stride_map)
    n = len(device_size)
    stick_pos = n - 1

    # If indirect is already at position 0, no change needed
    if indirect_device_pos == 0:
        return value_stl

    # Rotate: move indirect_device_pos to position 0, keep stick at end
    order = (
        [indirect_device_pos]
        + [i for i in range(n) if i != indirect_device_pos and i != stick_pos]
        + [stick_pos]
    )

    new_device_size = [device_size[i] for i in order]
    new_stride_map = [stride_map[i] for i in order]

    return SpyreTensorLayout(
        device_size=new_device_size,
        stride_map=new_stride_map,
        device_dtype=value_stl.device_dtype,
    )


def _can_mutate_producer_in_place(value_buf, output_names: set[str]) -> bool:
    """Check if a value buffer's producer layout can be rewritten in place.

    Producer layout can be rewritten if the buffer is a ComputedBuffer (not
    a graph input), not a mutation layout, and not a graph output. Multiple
    consumers are fine — we're rewriting the producer's output, which all
    consumers will see.
    """
    if not isinstance(value_buf, ComputedBuffer):
        return False
    if isinstance(value_buf.layout, MutationLayoutSHOULDREMOVE):
        return False
    if value_buf.get_name() in output_names:
        return False
    return True


def _rewrite_producer_layout(value_buf, required_stl: SpyreTensorLayout) -> None:
    value_buf.layout = _fixed_tiled(value_buf.get_layout(), required_stl)
    logger.info(
        "enforce_indirect_access_layout: rewrote producer %s layout in place -> %s",
        value_buf.get_name(),
        list(required_stl.stride_map),
    )


def _insert_relayout_copy(
    graph: GraphLowering,
    consumer_op: ComputedBuffer,
    value_buf,
    required_layout: FixedTiledLayout,
) -> ComputedBuffer:
    """Insert a spyre.restickify copy of value_buf in required_layout ahead of
    consumer_op, and patch consumer_op's inner_fn to read the new buffer.

    Returns the reconstructed ComputedBuffer that replaced consumer_op in
    graph.operations (insert_restickify_on_node_inputs invalidates the
    original instance).
    """
    operations = graph.operations
    arg_name = value_buf.get_name()
    consumer_name = consumer_op.get_name()
    insert_restickify_on_node_inputs(
        consumer_op,
        [{"arg_name": arg_name, "target_layout": required_layout}],
        operations,
    )
    logger.info(
        "enforce_indirect_access_layout: inserted relayout copy of %s before %s",
        arg_name,
        consumer_name,
    )
    return next(
        o
        for o in operations
        if isinstance(o, ComputedBuffer) and o.get_name() == consumer_name
    )


def _value_bufs_for_op(
    graph: GraphLowering,
    op: ComputedBuffer,
    access_subs: dict,
    sizes: dict | None,
) -> list:
    """Return the value-tensor buffers this op indirectly reads (gather:
    any read dep whose device_coordinates contain an IndirectAccess)."""
    value_bufs: list = []
    for dep in op.get_read_writes().reads:
        if not isinstance(dep, MemoryDep):
            continue
        buf = graph.get_buffer(dep.name)
        layout = _real_layout(buf)
        if not isinstance(layout, FixedTiledLayout):
            continue
        coords = [
            c.xreplace(access_subs)
            for c in device_coordinates(layout.device_layout, dep, sizes)
        ]
        if any(hasattr(c, "has") and c.has(IndirectAccess) for c in coords):
            value_bufs.append(buf)
    return value_bufs


def _get_indirect_access_dim_order_requirements(
    op: ComputedBuffer,
) -> tuple[set[str], dict, dict[sympy.Symbol, int] | None] | None:
    """Extract non-stick dimension ordering requirements from an indirect-access op.

    Returns (dep_names, access_subs, sizes) if the op has requirements, else None.
    """
    dep_names, access_subs, sizes = indirect_info_from_op(op)
    if dep_names:
        logger.debug(
            "enforce_indirect_access_layout: op %s has dim-order requirements "
            "from %d deps",
            op.get_name(),
            len(dep_names),
        )
        return dep_names, access_subs, sizes
    return None


def enforce_indirect_access_layout(graph: GraphLowering) -> None:
    """Reorder non-stick dimensions to satisfy indirect-access ops' requirements.

    Runs after insert_restickify: every op's layout is a committed
    FixedTiledLayout at this point. For each indirect-access op (gather/scatter),
    checks whether the value tensor's current non-stick dim_order puts the
    indexed dimension outermost as required; if not, either rewrites the
    producer's layout in place (single-consumer, non-mutation, non-graph-output
    case) or inserts a spyre.restickify copy node ahead of the consumer.
    """
    for original_op in list(graph.operations):
        if not isinstance(original_op, ComputedBuffer):
            continue
        requirement = _get_indirect_access_dim_order_requirements(original_op)
        if not requirement:
            continue
        dep_names, access_subs, sizes = requirement

        op = original_op
        value_bufs = _value_bufs_for_op(graph, op, access_subs, sizes)
        for value_buf in value_bufs:
            value_layout = _real_layout(value_buf)
            if not isinstance(value_layout, FixedTiledLayout):
                continue
            value_stl = value_layout.device_layout

            value_dep = next(
                d
                for d in op.get_read_writes().reads
                if isinstance(d, MemoryDep) and d.name == value_buf.get_name()
            )
            value_coords = device_coordinates(value_stl, value_dep, sizes)
            stride_idx = _indirect_stride_idx(value_coords, access_subs)
            if stride_idx is None:
                continue

            index_names = {
                sym.args[0].name
                for sym in access_subs.values()
                if isinstance(sym, IndirectAccess)
            }
            index_name = next(iter(index_names & dep_names), None)
            if index_name is None:
                continue
            index_dep = next(
                (
                    d
                    for d in op.get_read_writes().reads
                    if isinstance(d, MemoryDep) and d.name == index_name
                ),
                None,
            )
            if index_dep is None:
                continue
            index_layout = _real_layout(graph.get_buffer(index_name))
            if not isinstance(index_layout, FixedTiledLayout):
                continue
            if _dim_order_is_compliant(value_stl, stride_idx):
                continue

            # Rotate the device layout to put the indirect coordinate at position 0
            indirect_device_pos = len(value_stl.stride_map) - 1 - stride_idx
            required_stl = _build_required_stl(value_stl, indirect_device_pos)

            if _can_mutate_producer_in_place(value_buf, graph.get_output_names()):
                _rewrite_producer_layout(value_buf, required_stl)
            else:
                required_layout = _fixed_tiled(value_layout, required_stl)
                op = _insert_relayout_copy(graph, op, value_buf, required_layout)

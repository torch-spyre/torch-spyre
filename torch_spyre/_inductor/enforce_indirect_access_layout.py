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
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    ReinterpretView,
    Scatter,
    StorageBox,
)
from torch_spyre._C import SpyreTensorLayout

from .constants import ELIDED_COPY_BACK_ATTR
from .insert_restickify import (
    _create_restickify_node,
    _fixed_tiled,
    insert_restickify_on_node_inputs,
)
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess
from .pass_utils import (
    _build_indirect_store_subs,
    device_coordinates,
    indirect_info_from_op,
)

logger = get_inductor_logger("enforce_indirect_access_layout")


def _scatter_access_subs_and_sizes(
    scatter_op: ComputedBuffer, output_layout, write_dep: MemoryDep
) -> tuple[dict, dict]:
    """Build access substitutions and sizes for scatter op index symbols.

    _build_indirect_store_subs keys its result by the scatter index symbol
    itself (e.g. tmp0 in `d1 + 1024*tmp0`), mapping it to the index buffer's
    IndexedBase access. Extract the actual scatter index symbols from
    write_dep.index (symbols that appear in the write but NOT in write_dep's
    loop ranges) and build IndirectAccess mappings for each one.

    For each scatter symbol, compute its device dimension via device_coordinates
    to find the corresponding size in output_layout.device_size.
    Returns ({sym: IndirectAccess(...)}, {sym: size}).
    """

    store_subs, _ = _build_indirect_store_subs(scatter_op)

    # Extract the actual scatter index symbols from write_dep.index.
    # These are symbols that appear in the write but NOT in write_dep.ranges.
    all_syms = write_dep.index.free_symbols
    loop_syms = set(write_dep.ranges.keys())
    scatter_syms = all_syms - loop_syms

    access_subs: dict = {
        sym: IndirectAccess(sympy.Symbol(store_subs[sym].base.name))
        for sym in scatter_syms
        if sym in store_subs and hasattr(store_subs[sym], "base")
    }

    # Compute device coordinates to find the actual device dimension for each
    # scatter symbol, then look up its size from output_layout.device_size.
    sizes: dict[sympy.Symbol, int] = {}
    if output_layout.size and output_layout.stride:
        # For each scatter symbol, find which host dimension it multiplies
        # (by matching the stride of that dimension in output_layout.stride).
        for sym in access_subs:
            # Extract the coefficient of this symbol in write_dep.index
            coeff = write_dep.index.coeff(sym)
            if coeff is not None:
                # Find which host dimension has this stride
                for dim_idx, stride in enumerate(output_layout.stride):
                    if stride == coeff:
                        if 0 <= dim_idx < len(output_layout.size):
                            sizes[sym] = output_layout.size[dim_idx]
                        break

    return access_subs, sizes


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


def _output_real_layout(op: ComputedBuffer) -> FixedTiledLayout:
    """Resolve an op's committed output layout, unwrapping a genuine mutation
    target (unlike _real_layout, which asserts mutation layouts only appear on
    elided copy-backs — an op's own MutationLayoutSHOULDREMOVE is expected)."""
    layout = op.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    return layout


def _resolve_mutation_target(op: ComputedBuffer) -> tuple[str, object]:
    """Unwrap a MutationLayoutSHOULDREMOVE op's layout to (target_name, target_buf).

    Mirrors propagate_layouts.py's MutationLayoutSHOULDREMOVE branch, which
    unwraps ReinterpretView chains the same way to resolve the mutation target.
    """
    assert isinstance(op.layout, MutationLayoutSHOULDREMOVE)
    target = op.layout.target
    while isinstance(target, ReinterpretView):
        target = target.data
    return target.get_name(), target


def _insert_mutation_relayout_copy(
    graph: GraphLowering,
    mutation_op: ComputedBuffer,
    write_dep: MemoryDep,
    access_subs: dict,
    sizes: dict | None,
) -> None:
    """Fix a non-compliant indirect-write layout on a MutationLayoutSHOULDREMOVE op.

    Inserts a copy-in / retarget / copy-back sequence around the mutation.
    For scatter ops, uses buf_tmp as the metadata source for copy-back to
    avoid inheriting the index tensor dependency from the scatter op.
    """
    is_scatter_op = (
        any(isinstance(v, IndirectAccess) for v in access_subs.values())
        if access_subs
        else False
    )
    is_scatter_op = is_scatter_op or isinstance(mutation_op.data, Scatter)

    output_stl = _output_real_layout(mutation_op).device_layout

    write_stride_idx: int | None = None
    if is_scatter_op:
        logger.debug(
            "enforce_indirect_layout: scatter op device_size=%s, stride_map=%s",
            output_stl.device_size,
            output_stl.stride_map,
        )
        # For scatter, get access subs and sizes, then find indirect in write coords
        scatter_access_subs, scatter_sizes = _scatter_access_subs_and_sizes(
            mutation_op, _output_real_layout(mutation_op), write_dep
        )
        write_stride_idx = _indirect_stride_idx(
            device_coordinates(output_stl, write_dep, scatter_sizes),
            scatter_access_subs,
        )
    else:
        write_stride_idx = _indirect_stride_idx(
            device_coordinates(output_stl, write_dep, sizes), access_subs
        )
        assert write_stride_idx is not None, (
            f"expected an IndirectAccess write coordinate on {mutation_op.get_name()!r}"
        )
    assert write_stride_idx is not None
    output_indirect_pos = len(output_stl.stride_map) - 1 - write_stride_idx
    required_stl = _build_required_stl(output_stl, output_indirect_pos)

    target_name, target_buf = _resolve_mutation_target(mutation_op)
    if target_buf is None:
        raise AssertionError(
            f"mutation target resolved to None for {mutation_op.get_name()}"
        )
    target_layout = target_buf.get_layout()
    if target_layout is None:
        raise AssertionError(
            f"mutation target {target_name!r} has None layout for {mutation_op.get_name()}"
        )
    assert isinstance(target_layout, FixedTiledLayout), (
        f"expected FixedTiledLayout on mutation target {target_name!r}, "
        f"got {type(target_layout).__name__}"
    )

    buf_tmp_layout = _fixed_tiled(target_layout, required_stl)
    orig_stl_layout = target_layout

    # Step 1: copy-in: target (current layout) -> buf_tmp (required_stl).
    _, buf_tmp = _create_restickify_node(
        {"arg_name": target_name, "target_layout": buf_tmp_layout}, mutation_op
    )
    buf_tmp_name = buf_tmp.get_name()
    buf_tmp._input_layout_overrides = {target_name: orig_stl_layout}

    # Step 2: retarget the mutation to buf_tmp, preserving any slice offset.
    mutation_name = mutation_op.get_name()
    original_layout = mutation_op.layout
    assert isinstance(original_layout, MutationLayoutSHOULDREMOVE)
    slice_layout = original_layout.target.get_layout()
    if isinstance(original_layout.target, ReinterpretView) and slice_layout.offset != 0:
        mutation_op.layout = MutationLayoutSHOULDREMOVE(
            ReinterpretView(data=StorageBox(buf_tmp), layout=slice_layout)
        )
    else:
        mutation_op.layout = MutationLayoutSHOULDREMOVE(buf_tmp)

    operations = graph.operations
    mutation_op_index = operations.index(mutation_op)
    operations.remove(buf_tmp)
    operations.insert(mutation_op_index, buf_tmp)

    # Step 3: copy-back: buf_tmp (required_stl) -> target_buf (original layout)
    buf_copyback_layout = _fixed_tiled(target_layout, required_stl)
    # For scatter, use buf_tmp as metadata source to avoid inheriting index tensor dependency
    copyback_metadata_op = buf_tmp if is_scatter_op else mutation_op
    _, buf_copyback = _create_restickify_node(
        {"arg_name": buf_tmp_name, "target_layout": buf_copyback_layout},
        copyback_metadata_op,
    )
    buf_copyback.layout = MutationLayoutSHOULDREMOVE(target_buf)
    operations.remove(buf_copyback)
    operations.insert(mutation_op_index + 2, buf_copyback)

    logger.info(
        "enforce_indirect_layout: inserted mutation relayout copy for %s "
        "(copy-in %s -> %s, copy-back %s -> %s)",
        mutation_name,
        target_name,
        buf_tmp_name,
        buf_tmp_name,
        target_name,
    )


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


def _enforce_scatter_destination_layout(
    graph: GraphLowering,
    scatter_op: ComputedBuffer,
    requirement: tuple[set[str], dict, dict[sympy.Symbol, int] | None] | None,
) -> None:
    """Ensure a scatter's destination has its scattered dim outermost.

    Unlike gather, whose read side is a plain ComputedBuffer, a scatter's
    write side is expressed as a MutationLayoutSHOULDREMOVE on the value
    tensor -- so the "does the indexed dim sit outermost" check that the main
    pass loop already does for read coordinates has to be redone here against
    write coordinates, against the *target's* committed layout rather than
    the op's own.

    Preferring a producer rewrite over a destination copy: if the mutation
    target is itself a ComputedBuffer we can still rewrite in place (not a
    graph output, not already a mutation), retargeting its layout to match
    the scatter output's layout is free -- no new copy node -- and makes the
    destination trivially compliant, since target and output share one
    layout. Only fall back to inserting a copy-in/copy-back pair (via
    _insert_mutation_relayout_copy) when that rewrite isn't available and the
    destination's own layout doesn't already satisfy the requirement.
    """
    write_dep = next(
        (d for d in scatter_op.get_read_writes().writes if isinstance(d, MemoryDep)),
        None,
    )
    if write_dep is None:
        return
    output_layout = _output_real_layout(scatter_op)
    if isinstance(output_layout, FixedTiledLayout):
        output_stl = output_layout.device_layout
    else:
        # For non-tiled layouts (e.g., FixedLayout), skip enforcement.
        return

    # The scatter mutates its input (the value tensor); the mutation target is
    # the value producer. Resolve and look up the actual buffer.
    target_name, _ = _resolve_mutation_target(scatter_op)
    target_buf = graph.get_buffer(target_name)
    logger.debug(
        "scatter_destination_check: target_name=%r, target_buf type=%s",
        target_name,
        type(target_buf).__name__ if target_buf else "None",
    )
    value_producer_rewritten = False
    if isinstance(target_buf, ComputedBuffer) and _can_mutate_producer_in_place(
        target_buf, graph.get_output_names()
    ):
        value_layout = _real_layout(target_buf)
        if isinstance(value_layout, FixedTiledLayout):
            value_stl = value_layout.device_layout
            if value_stl != output_stl:
                _rewrite_producer_layout(target_buf, output_stl)
                value_producer_rewritten = True
                logger.info(
                    "scatter_value_check: rewrote mutation target %s layout "
                    "to match scatter output layout",
                    target_buf.get_name(),
                )

    if value_producer_rewritten:
        # Target and output now share a layout, so the destination check
        # below is automatically satisfied -- skip it.
        return

    # Check scatter destination compliance: scatter index dimensions must be outermost.
    # Detect scatter index symbols (non-loop symbols in write_dep.index).
    all_write_syms = write_dep.index.free_symbols
    loop_syms = set(write_dep.ranges.keys())
    scatter_syms = all_write_syms - loop_syms

    if not scatter_syms:
        # No scatter index symbols found (shouldn't happen for a real scatter).
        logger.debug(
            "scatter_destination_check: no scatter symbols found for %s",
            scatter_op.get_name(),
        )
        return

    # Build substitutions mapping scatter symbols to IndirectAccess markers.
    scatter_access_subs = {sym: IndirectAccess(sym) for sym in scatter_syms}

    # For scatter destination compliance, check against the *target's* layout,
    # not the output layout. The write side must conform to the target's committed
    # device layout, which is where the scatter actually writes.
    target_layout = target_buf.get_layout()
    target_stl = None
    if isinstance(target_layout, FixedTiledLayout):
        target_stl = target_layout.device_layout
    else:
        # For non-tiled layouts (e.g., FixedLayout on graph inputs), try to get
        # the SpyreTensorLayout from the TensorBox's .layouts attribute.
        layouts = getattr(target_buf, "layouts", None)
        if layouts:
            target_stl = next(iter(layouts))
        else:
            # Target is a graph input with no device layout propagated yet,
            # or some other non-tiled layout we cannot enforce on.
            logger.debug(
                "scatter_destination_check: skipping %s: mutation target %r has %s "
                "with no device layout (cannot enforce)",
                scatter_op.get_name(),
                target_name,
                type(target_layout).__name__,
            )
            return

    if target_stl is None:
        # Target is a graph input (FixedLayout) or other non-tiled layout.
        # We cannot modify graph inputs, so skip enforcement.
        logger.debug(
            "scatter_destination_check: skipping %s: mutation target %r has %s "
            "(not FixedTiledLayout, cannot enforce)",
            scatter_op.get_name(),
            target_name,
            type(target_layout).__name__,
        )
        return

    # Compute write coordinates against the target's layout and find positions
    # of IndirectAccess markers.
    write_coords = device_coordinates(target_stl, write_dep, None)
    indirect_stride_idxs = []
    for idx, coord in enumerate(reversed(write_coords)):
        substituted = coord.xreplace(scatter_access_subs)
        if hasattr(substituted, "has") and substituted.has(IndirectAccess):
            indirect_stride_idxs.append(idx)

    # Check compliance: all indirect positions must be at the front (positions 0, 1, ...).
    is_compliant = False
    if indirect_stride_idxs:
        # Convert stride_idx (from right) to device_pos (from left).
        indirect_device_pos = sorted(
            len(target_stl.stride_map) - 1 - idx for idx in indirect_stride_idxs
        )
        # Indirect dims must occupy the first N positions (0, 1, ..., N-1).
        expected_pos = list(range(len(indirect_stride_idxs)))
        is_compliant = indirect_device_pos == expected_pos
        logger.debug(
            "scatter_destination_check: %s indirect_device_pos=%s, expected=%s, compliant=%s",
            scatter_op.get_name(),
            indirect_device_pos,
            expected_pos,
            is_compliant,
        )

    if not is_compliant:
        logger.info(
            "scatter_destination_check: inserting mutation relayout copy for %s",
            scatter_op.get_name(),
        )
        _insert_mutation_relayout_copy(graph, scatter_op, write_dep, {}, None)


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
        is_scatter = isinstance(original_op.data, Scatter)
        is_mutation = isinstance(original_op.layout, MutationLayoutSHOULDREMOVE)
        if is_scatter or is_mutation:
            logger.debug(
                "enforce_indirect_access_layout: checking op %s, is_scatter=%s, "
                "is_mutation=%s",
                original_op.get_name(),
                is_scatter,
                is_mutation,
            )
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

        # For scatter ops, ensure scatter destination dimension 0 (scatter index) is outermost
        if is_scatter and is_mutation:
            _enforce_scatter_destination_layout(graph, original_op, requirement)

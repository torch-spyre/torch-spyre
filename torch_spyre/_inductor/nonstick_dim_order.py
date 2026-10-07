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

"""Reorder non-stick device dimensions for correctness and performance.

Runs after optimize_restickify_locations and before finalize_layouts. At that
point each buffer has a single committed_stl chosen by the beam optimizer;
stick choices are final and these rewrites affect only non-stick dim ordering.

Two phases, in order:

  Phase 1 — constraint transforms (run first, establish pinned_dims):
    gather IA constraint
      The indirectly-indexed dimension of a gather value tensor must sit at
      device position 0.  This is a hardware requirement, not a hint.
    scatter IA constraint
      The scattered dimension of a scatter destination must sit at device
      position 0.  Same reason.

  Phase 2 — performance transforms (run second, respect pinned_dims):
    matmul perf reorder
      For factorised-stick layouts, swaps the largest non-pinned nonstick dim
      into the slot between the two stick dims (outer_stick+1) so the widest
      loop variable carries the most work.

Each transform is tried independently per phase; constraint transforms run
first and record which device positions they have fixed (pinned_dims).  The
performance transform runs second and respects those pins.  This is correct
today because the constraint and performance transforms target structurally
different dimensions — the IA constraint fixes the indexed dim at position 0,
while the matmul reorder moves the *largest remaining* dim into a different
slot.  If a future use case requires more complex composition, extend
pinned_dims or replace the two-phase structure with explicit dim-order
constraint solving.
pinned_dims: dict[str, set[int]]  (local to reorder_nonstick_dims, never on graph)
"""

import sympy

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Reduction,
    ReinterpretView,
    Scatter,
    StorageBox,
)
from torch._inductor.virtualized import V
from torch_spyre._C import ElementArrangement, SpyreTensorLayout

from .constants import MATMUL_REDUCTION_OPS
from .errors import Unsupported
from .insert_restickify import (
    _create_restickify_node,
    _fixed_tiled,
    RestickifyArgInfo,
)
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess
from .pass_utils import (
    device_coordinates,
    indirect_info_from_op,
    loop_var_ranges_from_dim_hints,
    try_device_coordinates,
)

# Import helpers that remain in enforce_indirect_access_layout and are used by
# the mutation-target dim-order enforcement functions moved here in Task 4.
# enforce_indirect_access_layout does NOT import nonstick_dim_order, so this
# import is safe at module level.
from .enforce_indirect_access_layout import (
    _dim_order_is_compliant,
    _get_indirect_access_dim_order_requirements,
    _insert_relayout_copy,
    _output_real_layout,
    _real_layout,
    _resolve_mutation_target,
    _scatter_access_subs_and_sizes,
)

logger = get_inductor_logger("nonstick_dim_order")


def _buf_stl(buf) -> SpyreTensorLayout | None:
    """Return the committed STL for any buffer type.

    For ComputedBuffer: buf.committed_stl (set by optimize_restickify_locations,
    deleted by insert_restickify — only valid between those two passes).
    For InputBuffer (graph inputs): buf.committed_stl (set by
    optimize_restickify_locations, retained through insert_restickify).
    For other buffers without committed_stl: read from the FixedTiledLayout
    directly (e.g. restickify nodes inserted by insert_restickify).
    Returns None if no STL is available.
    """
    if hasattr(buf, "committed_stl"):
        return buf.committed_stl
    layout = _real_layout(buf)
    if isinstance(layout, FixedTiledLayout):
        return layout.device_layout
    return None


def _matmul_reorder_stl(
    stl: SpyreTensorLayout,
    dep: MemoryDep,
    name: str = "",
    pinned: set[int] | None = None,
) -> SpyreTensorLayout | None:
    """Swap the largest non-pinned nonstick dim into the outer_stick+1 slot.

    Returns a new STL if a swap was made, or None if no change is needed.
    pinned is a set of device positions that must not be moved or displaced.
    """
    if stl.element_arrangement != ElementArrangement.STANDARD:
        return None
    device_size = list(stl.device_size)
    stride_map = list(stl.stride_map)
    n = len(device_size)
    if n <= 2:
        return None

    idc = try_device_coordinates(stl, dep, {})
    if idc is None:
        return None

    stick_syms = idc[-1].free_symbols
    if not stick_syms:
        return None

    outer_stick = None
    for i in range(n - 2, -1, -1):
        if idc[i].free_symbols & stick_syms:
            outer_stick = i
            break
    if outer_stick is None:
        return None

    slot = outer_stick + 1
    if slot >= n - 1:
        return None

    # If the slot itself is pinned, we cannot move anything into it.
    if pinned and slot in pinned:
        logger.debug("nonstick_dim_order: skipping %s — slot %d is pinned", name, slot)
        return None

    # Only consider dims before outer_stick with real (non-constant) coordinates
    # that are not pinned.
    candidates = [
        d
        for d in range(outer_stick)
        if idc[d].free_symbols and (not pinned or d not in pinned)
    ]
    if not candidates:
        return None
    largest = max(candidates, key=lambda d: device_size[d])
    if device_size[largest] <= device_size[slot]:
        return None

    new_order = list(range(n))
    new_order[slot], new_order[largest] = new_order[largest], new_order[slot]
    new_device_size = [device_size[d] for d in new_order]
    new_stride_map = [stride_map[d] for d in new_order]
    logger.debug(
        "[NDO] %s  %s -> %s  stride_map %s -> %s",
        name,
        device_size,
        new_device_size,
        list(stl.stride_map),
        new_stride_map,
    )
    return SpyreTensorLayout(
        device_size=new_device_size,
        stride_map=new_stride_map,
        device_dtype=stl.device_dtype,
    )


def _try_matmul_perf_reorder(
    buf: ComputedBuffer,
    pinned: set[int],
) -> SpyreTensorLayout | None:
    """Return a reordered STL for buf if matmul perf reorder applies, else None.

    Uses the buffer's own write dep to compute device coordinates, matching
    the index expressions the buffer was actually written with.
    """
    if not hasattr(buf, "committed_stl"):
        return None
    write_dep = next(iter(buf.get_read_writes().writes), None)
    if write_dep is None:
        return None
    return _matmul_reorder_stl(buf.committed_stl, write_dep, buf.get_name(), pinned)


def _indirect_stride_idx(
    coords: list,
    access_subs: dict,
) -> int | None:
    """Return the stride_idx (from right, 0-indexed) of the first IndirectAccess
    coordinate, or None if coords carry no indirect symbol."""
    for idx, coord in enumerate(reversed(coords)):
        substituted = coord.xreplace(access_subs) if access_subs else coord
        if hasattr(substituted, "has") and substituted.has(IndirectAccess):
            return idx
    return None


def _ia_rotate_stl(
    stl: SpyreTensorLayout,
    indirect_device_pos: int,
) -> SpyreTensorLayout:
    """Build a new STL with the indirect coordinate rotated to device position 0."""
    device_size = list(stl.device_size)
    stride_map = list(stl.stride_map)
    n = len(device_size)
    stick_pos = n - 1

    if indirect_device_pos == 0:
        return stl

    order = (
        [indirect_device_pos]
        + [i for i in range(n) if i != indirect_device_pos and i != stick_pos]
        + [stick_pos]
    )
    return SpyreTensorLayout(
        device_size=[device_size[i] for i in order],
        stride_map=[stride_map[i] for i in order],
        device_dtype=stl.device_dtype,
    )


def _try_gather_ia_constraint(
    buf: ComputedBuffer,
    dep: MemoryDep,
    op: ComputedBuffer,
) -> tuple[SpyreTensorLayout, set[int]] | None:
    """Rotate the indirectly-indexed dim of a gather value tensor to device position 0.

    buf is the value tensor (indirectly-indexed buffer); dep is its read dep.
    Returns (new_stl, {0}) if a rotation is needed, None if already compliant
    or not applicable.
    """
    if not hasattr(buf, "committed_stl"):
        return None
    _, access_subs, sizes = indirect_info_from_op(op)
    if not access_subs:
        return None
    stl = buf.committed_stl
    try:
        coords = device_coordinates(stl, dep, sizes, op=op)
    except (Unsupported, Exception):
        return None
    coords_substituted = [c.xreplace(access_subs) for c in coords]
    stride_idx = _indirect_stride_idx(coords_substituted, access_subs)
    if stride_idx is None:
        return None
    indirect_device_pos = len(stl.stride_map) - 1 - stride_idx
    if indirect_device_pos == 0:
        return None
    new_stl = _ia_rotate_stl(stl, indirect_device_pos)
    logger.info(
        "nonstick_dim_order: gather IA constraint on %s — indirect dim %d -> pos 0",
        buf.get_name(),
        indirect_device_pos,
    )
    return new_stl, {0}


def _try_scatter_ia_constraint(
    buf: ComputedBuffer,
    dep: MemoryDep,
    op: ComputedBuffer,
) -> tuple[SpyreTensorLayout, set[int]] | None:
    """Rotate the scattered dim of a scatter destination to device position 0.

    buf is the scatter destination buffer (resolved by the caller);
    dep is the write dep. Does not re-resolve the destination.
    Returns (new_stl, {0}) if rotation needed, None if compliant or not applicable.
    """
    if not hasattr(buf, "committed_stl"):
        return None
    stl = buf.committed_stl

    # Extract scatter index symbols: symbols in dep.index that are not loop
    # range keys and not WhileLoop splice vars.
    all_write_syms = dep.index.free_symbols
    loop_syms = set(dep.ranges.keys())
    loop_syms |= set(loop_var_ranges_from_dim_hints(op))
    scatter_syms = all_write_syms - loop_syms
    if not scatter_syms:
        return None

    scatter_access_subs = {sym: IndirectAccess(sym) for sym in scatter_syms}

    try:
        write_coords = device_coordinates(stl, dep, None)
    except (Unsupported, Exception):
        return None

    indirect_stride_idxs = []
    for idx, coord in enumerate(reversed(write_coords)):
        substituted = coord.xreplace(scatter_access_subs)
        if hasattr(substituted, "has") and substituted.has(IndirectAccess):
            indirect_stride_idxs.append(idx)

    if not indirect_stride_idxs:
        return None

    indirect_device_pos = sorted(
        len(stl.stride_map) - 1 - idx for idx in indirect_stride_idxs
    )
    expected_pos = list(range(len(indirect_stride_idxs)))
    if indirect_device_pos == expected_pos:
        return None  # already compliant

    # Rotate the first indirect dim to position 0.
    new_stl = _ia_rotate_stl(stl, indirect_device_pos[0])
    logger.info(
        "nonstick_dim_order: scatter IA constraint on %s — indirect dim %d -> pos 0",
        buf.get_name(),
        indirect_device_pos[0],
    )
    return new_stl, {0}


def reorder_nonstick_dims(graph: GraphLowering) -> None:
    """Reorder non-stick dims for correctness (phase 1) and performance (phase 2).

    Phase 1 (one backward walk) applies constraint transforms: gather IA and
    scatter IA constraints pin device positions in pinned_dims.
    Phase 2 (second backward walk) applies performance transforms: matmul perf
    reorder respects pinned_dims set by phase 1.
    """
    V.graph.nonstick_reorder_log = {}
    log: dict[str, SpyreTensorLayout] = {}
    pinned_dims: dict[str, set[int]] = {}
    graph_inputs = set(V.graph.graph_input_names)

    # Phase 1: constraint transforms (gather IA, scatter IA).
    seen_p1: set[tuple[str, str]] = set()
    for op in reversed(graph.operations):
        if not isinstance(op, ComputedBuffer):
            continue
        if not hasattr(op, "data"):
            continue

        # Gather IA: find value tensor deps with an IndirectAccess coordinate.
        dep_names, access_subs, sizes = indirect_info_from_op(op)
        if dep_names and not isinstance(op.data, Scatter):
            for dep in op.get_read_writes().reads:
                if not isinstance(dep, MemoryDep):
                    continue
                buf = V.graph.get_buffer(dep.name)
                if not isinstance(buf, ComputedBuffer):
                    continue
                if not hasattr(buf, "committed_stl"):
                    continue
                try:
                    coords = device_coordinates(buf.committed_stl, dep, sizes)
                except Exception:
                    continue
                coords_sub = [c.xreplace(access_subs) for c in coords]
                if not any(
                    hasattr(c, "has") and c.has(IndirectAccess) for c in coords_sub
                ):
                    continue
                key = (dep.name, op.get_name())
                if key in seen_p1:
                    continue
                seen_p1.add(key)
                result = _try_gather_ia_constraint(buf, dep, op)
                if result is not None:
                    new_stl, pinned = result
                    buf.committed_stl = new_stl
                    log[buf.get_name()] = new_stl
                    pinned_dims.setdefault(buf.get_name(), set()).update(pinned)

        # Scatter IA: resolve destination and apply constraint.
        if isinstance(op.data, Scatter) and isinstance(
            op.layout, MutationLayoutSHOULDREMOVE
        ):
            write_deps = [
                d for d in op.get_read_writes().writes if isinstance(d, MemoryDep)
            ]
            if write_deps:
                write_dep = write_deps[0]
                target = op.layout.target
                while isinstance(target, ReinterpretView):
                    target = target.data
                dest_buf = target if isinstance(target, ComputedBuffer) else None
                if dest_buf is not None and hasattr(dest_buf, "committed_stl"):
                    result = _try_scatter_ia_constraint(dest_buf, write_dep, op)
                    if result is not None:
                        new_stl, pinned = result
                        dest_buf.committed_stl = new_stl
                        log[dest_buf.get_name()] = new_stl
                        pinned_dims.setdefault(dest_buf.get_name(), set()).update(
                            pinned
                        )

    # Phase 2: performance transforms (matmul perf reorder).
    seen_p2: set[str] = set()
    for op in reversed(graph.operations):
        if not isinstance(op, ComputedBuffer):
            continue
        if not hasattr(op, "data"):
            continue
        if not (
            isinstance(op.data, Reduction)
            and op.data.reduction_type in MATMUL_REDUCTION_OPS
        ):
            continue
        for dep in op.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            if dep.name in graph_inputs:
                continue
            buf = V.graph.get_buffer(dep.name)
            if not isinstance(buf, ComputedBuffer):
                continue
            name = buf.get_name()
            if name in seen_p2:
                continue
            seen_p2.add(name)
            pinned = pinned_dims.get(name, set())
            reordered_stl = _try_matmul_perf_reorder(buf, pinned)
            if reordered_stl is not None:
                buf.committed_stl = reordered_stl
                log[name] = reordered_stl
                logger.info("nonstick_dim_order: reordered %s", name)

    V.graph.nonstick_reorder_log = log


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
        "nonstick_dim_order: rewrote producer %s layout in place -> %s",
        value_buf.get_name(),
        list(required_stl.stride_map),
    )


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
            "nonstick_dim_order: scatter op device_size=%s, stride_map=%s",
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
    required_stl = _ia_rotate_stl(output_stl, output_indirect_pos)

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
        RestickifyArgInfo(
            arg_name=target_name,
            dep_index=None,
            occurrence=0,
            target_layout=buf_tmp_layout,
        ),
        mutation_op,
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
        RestickifyArgInfo(
            arg_name=buf_tmp_name,
            dep_index=None,
            occurrence=0,
            target_layout=buf_copyback_layout,
        ),
        copyback_metadata_op,
    )
    buf_copyback.layout = MutationLayoutSHOULDREMOVE(target_buf)
    operations.remove(buf_copyback)
    operations.insert(mutation_op_index + 2, buf_copyback)

    logger.info(
        "nonstick_dim_order: inserted mutation relayout copy for %s "
        "(copy-in %s -> %s, copy-back %s -> %s)",
        mutation_name,
        target_name,
        buf_tmp_name,
        buf_tmp_name,
        target_name,
    )


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
    #
    # A WhileLoop-splice per-iteration loop_var (e.g. u0, see
    # wsr/for_each_tile_lowering.py's _synthesize_dim_hints_for_group) is
    # deliberately folded into write_dep.index without ever being a
    # write_dep.ranges key -- see pass_utils.py's
    # loop_var_ranges_from_dim_hints -- so it looks exactly like a scatter
    # index symbol by this "not a loop range key" test alone. Excluding it
    # explicitly matches _build_indirect_store_subs's identical exclusion
    # for the read-side version of this same inference; without it, a
    # scatter nested inside a spliced WhileLoop body would misclassify its
    # own tile-advancing loop_var as a scatter index and corrupt the
    # dim-order compliance check below.
    all_write_syms = write_dep.index.free_symbols
    loop_syms = set(write_dep.ranges.keys())
    loop_syms |= set(loop_var_ranges_from_dim_hints(scatter_op))
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
    target_fixed_tiled_layout = None
    if isinstance(target_layout, FixedTiledLayout):
        target_stl = target_layout.device_layout
        target_fixed_tiled_layout = target_layout
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

    # Compute write coordinates against target layout. Sizes must be resolved
    # against target's strides (not output's), since coordinates are computed
    # against target_stl.
    if target_fixed_tiled_layout is not None:
        subs_from_op, scatter_sizes = _scatter_access_subs_and_sizes(
            scatter_op, target_fixed_tiled_layout, write_dep
        )
        if subs_from_op:
            scatter_access_subs = subs_from_op
    else:
        logger.debug(
            "scatter_destination_check: skipping %s: target is non-FixedTiledLayout",
            scatter_op.get_name(),
        )
        return
    try:
        write_coords = device_coordinates(target_stl, write_dep, scatter_sizes)
    except Unsupported as e:
        logger.debug(
            "scatter_destination_check: skipping %s: could not resolve sizes: %s",
            scatter_op.get_name(),
            str(e),
        )
        return
    indirect_stride_idxs = []
    for idx, coord in enumerate(reversed(write_coords)):
        substituted = coord.xreplace(scatter_access_subs)
        if hasattr(substituted, "has") and substituted.has(IndirectAccess):
            indirect_stride_idxs.append(idx)

    is_compliant = False
    if indirect_stride_idxs:
        indirect_device_pos = sorted(
            len(target_stl.stride_map) - 1 - idx for idx in indirect_stride_idxs
        )
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


def reorder_nonstick_dims_mutation(graph: GraphLowering) -> None:
    """Second phase of nonstick dim reordering for mutation targets.

    reorder_nonstick_dims (before finalize_layouts) cannot handle mutation
    targets because finalize_layouts skips MutationLayoutSHOULDREMOVE ops.
    This pass runs after insert_restickify when every buffer has a committed
    FixedTiledLayout and inserts copy-in/copy-back pairs for scatter
    destinations whose dim order does not satisfy the IA constraint.

    Mirrors the propagate_layouts / propagate_mutation_layouts pattern.
    """
    for op in list(graph.operations):
        if not isinstance(op, ComputedBuffer):
            continue
        if not isinstance(op.data, Scatter):
            continue
        if not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            continue
        requirement = _get_indirect_access_dim_order_requirements(op)
        if not requirement:
            continue
        _enforce_scatter_destination_layout(graph, op, requirement)

    # Handle gather graph-input value tensors post-insert_restickify.
    # Phase 1 of reorder_nonstick_dims skips graph inputs (no committed_stl).
    # Here every buffer has a FixedTiledLayout so we can insert relayout copies.
    for op in list(graph.operations):
        if not isinstance(op, ComputedBuffer):
            continue
        if isinstance(op.data, Scatter):
            continue  # scatter handled above
        dep_names, access_subs, sizes = indirect_info_from_op(op)
        if not dep_names:
            continue
        for dep in op.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            buf = graph.get_buffer(dep.name)
            if buf is None:
                continue
            if isinstance(buf, ComputedBuffer):
                continue  # handled by phase 1 of reorder_nonstick_dims
            layout = _real_layout(buf)
            if not isinstance(layout, FixedTiledLayout):
                continue
            value_stl = layout.device_layout
            try:
                coords = device_coordinates(value_stl, dep, sizes)
            except Exception:
                continue
            coords_sub = [c.xreplace(access_subs) for c in coords]
            stride_idx = _indirect_stride_idx(coords_sub, {})
            if stride_idx is None:
                continue
            if _dim_order_is_compliant(value_stl, stride_idx):
                continue
            indirect_device_pos = len(value_stl.stride_map) - 1 - stride_idx
            required_stl = _ia_rotate_stl(value_stl, indirect_device_pos)
            required_layout = _fixed_tiled(layout, required_stl)
            op = _insert_relayout_copy(graph, op, buf, required_layout)

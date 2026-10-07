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

Two-pass design: ``reorder_nonstick_dims`` (pass 1) makes all dim-order
decisions and records deferred work; ``reorder_nonstick_dims_mutation``
(pass 2) executes that work.

Pass 1 — ``reorder_nonstick_dims``
  Runs after optimize_restickify_locations and before finalize_layouts. At
  that point each buffer has a single committed_stl chosen by the beam
  optimizer; stick choices are final and these rewrites affect only non-stick
  dim ordering.

  Internally two phases, in order:

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

  Decisions that cannot be executed in pass 1 (mutation targets, graph
  inputs) are stored on V.graph.nonstick_deferred as _DeferredReorder entries.

Pass 2 — ``reorder_nonstick_dims_mutation``
  Runs after insert_restickify, when every buffer has a committed
  FixedTiledLayout. Iterates V.graph.nonstick_deferred and executes each
  entry by calling ``_rewrite_producer_layout`` or inserting copy nodes via
  ``_insert_relayout_copy`` / ``_insert_mutation_relayout_copy``. No
  analysis logic — purely mechanical execution of pass 1 decisions.

pinned_dims: dict[str, set[int]]  (local to reorder_nonstick_dims, never on graph)
"""

from typing import Literal, NamedTuple


from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Reduction,
    ReinterpretView,
    Scatter,
    StorageBox,
    TensorBox,
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
    try_device_coordinates,
)

# Helpers shared with enforce_indirect_access_layout.
# enforce_indirect_access_layout does NOT import nonstick_dim_order, so this
# import is safe at module level.
from .enforce_indirect_access_layout import (
    _insert_relayout_copy,
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


class _DeferredReorder(NamedTuple):
    op: ComputedBuffer  # the op that reads or writes buf
    buf_name: str  # name of buffer that needs reordering
    required_stl: SpyreTensorLayout
    kind: Literal["copy", "producer_rewrite"]


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
    coordinate, or None if coords carry no indirect symbol.
    """
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
    except Unsupported:
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
    buf: ComputedBuffer | InputBuffer,
    op: ComputedBuffer,
) -> tuple[SpyreTensorLayout, set[int]] | None:
    """Rotate the scattered dim of a scatter destination to device position 0.

    buf is the scatter destination buffer (ComputedBuffer or InputBuffer); op is
    the scatter op. Uses buf.get_layout() (not _real_layout) because at phase 1
    time buf may still have FlexibleLayout or FixedLayout — both have concrete
    .size and .stride sufficient to extract scatter symbol sizes via
    _scatter_access_subs_and_sizes. Returns (new_stl, {0}) if rotation needed,
    None if already compliant or not applicable.
    """
    stl = _buf_stl(buf)
    if stl is None:
        return None
    write_deps = [d for d in op.get_read_writes().writes if isinstance(d, MemoryDep)]
    if not write_deps:
        return None
    write_dep = write_deps[0]
    # Use get_layout() rather than _real_layout() because at phase 1 time
    # (before finalize_layouts) dest_buf has FlexibleLayout, not FixedTiledLayout.
    # FlexibleLayout has concrete .size and .stride, which is all we need to
    # extract scatter symbol sizes from write_dep.index coefficients.
    buf_layout = buf.get_layout()
    if buf_layout is None or not (
        getattr(buf_layout, "size", None) and getattr(buf_layout, "stride", None)
    ):
        return None
    try:
        access_subs, sizes = _scatter_access_subs_and_sizes(op, buf_layout, write_dep)
    except Unsupported:
        return None
    if not access_subs:
        return None
    try:
        write_coords = device_coordinates(stl, write_dep, sizes)
    except Unsupported:
        return None
    indirect_stride_idxs = []
    for idx, coord in enumerate(reversed(write_coords)):
        substituted = coord.xreplace(access_subs)
        if hasattr(substituted, "has") and substituted.has(IndirectAccess):
            indirect_stride_idxs.append(idx)
    if not indirect_stride_idxs:
        return None
    indirect_device_pos = sorted(
        len(stl.stride_map) - 1 - idx for idx in indirect_stride_idxs
    )
    expected_pos = list(range(len(indirect_stride_idxs)))
    if indirect_device_pos == expected_pos:
        return None
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
    V.graph.nonstick_deferred = []  # list[_DeferredReorder]
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
                    # Graph input or other non-ComputedBuffer: use _buf_stl,
                    # cannot update in place — record for phase 2 execution.
                    stl = _buf_stl(buf)
                    if stl is None:
                        continue
                    try:
                        coords = device_coordinates(stl, dep, sizes, op=op)
                    except Unsupported:
                        continue
                    coords_sub = [c.xreplace(access_subs) for c in coords]
                    stride_idx = _indirect_stride_idx(coords_sub, access_subs)
                    if stride_idx is None:
                        continue
                    indirect_device_pos = len(stl.stride_map) - 1 - stride_idx
                    if indirect_device_pos == 0:
                        continue  # already compliant
                    required_stl = _ia_rotate_stl(stl, indirect_device_pos)
                    key = (dep.name, op.get_name())
                    if key in seen_p1:
                        continue
                    seen_p1.add(key)
                    V.graph.nonstick_deferred.append(
                        _DeferredReorder(op, dep.name, required_stl, "copy")
                    )
                    continue
                # ComputedBuffer path continues below.
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
            target = op.layout.target
            while isinstance(target, (ReinterpretView, TensorBox, StorageBox)):
                target = target.data
            dest_buf = (
                target if isinstance(target, (ComputedBuffer, InputBuffer)) else None
            )
            if dest_buf is not None:
                dest_stl = _buf_stl(dest_buf)
                if dest_stl is not None:
                    result = _try_scatter_ia_constraint(dest_buf, op)
                    if result is not None:
                        new_stl, pinned = result
                        # Do NOT update dest_buf.committed_stl here: the scatter
                        # destination's layout transformation is applied at phase 2
                        # via _insert_mutation_relayout_copy, which needs to see the
                        # original (pre-rotation) target layout from finalize_layouts.
                        # Updating committed_stl now would cause finalize_layouts to
                        # create a FixedTiledLayout with the rotated STL, making
                        # _insert_mutation_relayout_copy a no-op copy.
                        pinned_dims.setdefault(dest_buf.get_name(), set()).update(
                            pinned
                        )
                        # Decide execution strategy for phase 2.
                        if _can_mutate_producer_in_place(
                            dest_buf, set(graph.get_output_names())
                        ):
                            V.graph.nonstick_deferred.append(
                                _DeferredReorder(
                                    op,
                                    dest_buf.get_name(),
                                    new_stl,
                                    "producer_rewrite",
                                )
                            )
                        else:
                            V.graph.nonstick_deferred.append(
                                _DeferredReorder(
                                    op, dest_buf.get_name(), new_stl, "copy"
                                )
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
    required_stl: SpyreTensorLayout,
) -> None:
    """Insert copy-in / retarget / copy-back for a scatter mutation op.

    required_stl is the layout the destination must have for the scatter to be
    hardware-compliant. Determined by reorder_nonstick_dims (phase 1).
    Uses buf_tmp as the metadata source for copy-back to avoid inheriting the
    index tensor dependency from the scatter op.
    """
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

    # Step 3: copy-back: buf_tmp (required_stl) -> target_buf (original layout).
    buf_copyback_layout = _fixed_tiled(target_layout, required_stl)
    _, buf_copyback = _create_restickify_node(
        RestickifyArgInfo(
            arg_name=buf_tmp_name,
            dep_index=None,
            occurrence=0,
            target_layout=buf_copyback_layout,
        ),
        buf_tmp,  # use buf_tmp as metadata source, not mutation_op
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


def reorder_nonstick_dims_mutation(graph: GraphLowering) -> None:
    """Execute deferred dim-order reorders recorded by reorder_nonstick_dims.

    Runs after insert_restickify when every buffer has a committed FixedTiledLayout.
    Iterates V.graph.nonstick_deferred and inserts copy nodes or rewrites producer
    layouts as decided by phase 1.
    """
    for entry in getattr(V.graph, "nonstick_deferred", []):
        buf = graph.try_get_buffer(entry.buf_name)
        if buf is None:
            continue
        if entry.kind == "producer_rewrite":
            _rewrite_producer_layout(buf, entry.required_stl)
        elif entry.kind == "copy":
            if isinstance(entry.op.layout, MutationLayoutSHOULDREMOVE):
                _insert_mutation_relayout_copy(graph, entry.op, entry.required_stl)
            else:
                layout = _real_layout(buf)
                if isinstance(layout, FixedTiledLayout):
                    required_layout = _fixed_tiled(layout, entry.required_stl)
                    _insert_relayout_copy(graph, entry.op, buf, required_layout)

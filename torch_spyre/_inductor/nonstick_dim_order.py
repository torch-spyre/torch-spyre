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

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Reduction,
    ReinterpretView,
    Scatter,
)
from torch._inductor.virtualized import V
from torch_spyre._C import ElementArrangement, SpyreTensorLayout

from .constants import MATMUL_REDUCTION_OPS
from .errors import Unsupported
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess
from .pass_utils import (
    device_coordinates,
    indirect_info_from_op,
    loop_var_ranges_from_dim_hints,
    try_device_coordinates,
)

logger = get_inductor_logger("nonstick_dim_order")


def _reorder_stl(
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
    return _reorder_stl(buf.committed_stl, write_dep, buf.get_name(), pinned)


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


def _build_required_stl(
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
    new_stl = _build_required_stl(stl, indirect_device_pos)
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

    buf is the scatter destination buffer (resolved by _collect_triples);
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
    new_stl = _build_required_stl(stl, indirect_device_pos[0])
    logger.info(
        "nonstick_dim_order: scatter IA constraint on %s — indirect dim %d -> pos 0",
        buf.get_name(),
        indirect_device_pos[0],
    )
    return new_stl, {0}


def _collect_triples(
    graph: GraphLowering,
) -> list[tuple[ComputedBuffer, MemoryDep, ComputedBuffer]]:
    """Walk graph; collect (buf, dep, op) triples for constraint and perf transforms.

    Returns triples where:
      buf — the ComputedBuffer to potentially reorder
      dep — the MemoryDep describing the access from op to buf
      op  — the ComputedBuffer that reads buf

    Collects:
      - Indirect-access gather ops: value tensor deps where dep.name in dep_names.
      - Matmul ops: all ComputedBuffer inputs.
    """
    seen: set[tuple[str, str]] = set()
    triples: list[tuple[ComputedBuffer, MemoryDep, ComputedBuffer]] = []
    graph_inputs = set(V.graph.graph_input_names)
    for op in reversed(graph.operations):
        if not isinstance(op, ComputedBuffer):
            continue
        if not hasattr(op, "data"):
            continue

        # Indirect-access ops (gather): collect value tensor deps.
        dep_names, access_subs, sizes = indirect_info_from_op(op)
        is_indirect = bool(dep_names) and not isinstance(op.data, Scatter)
        if is_indirect:
            for dep in op.get_read_writes().reads:
                if not isinstance(dep, MemoryDep):
                    continue
                if dep.name not in dep_names:
                    continue
                buf = V.graph.get_buffer(dep.name)
                if not isinstance(buf, ComputedBuffer):
                    continue
                key = (dep.name, op.get_name())
                if key in seen:
                    continue
                seen.add(key)
                triples.append((buf, dep, op))

        # Indirect-access ops (scatter): collect destination buf.
        is_scatter = isinstance(op.data, Scatter)
        if is_scatter and isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            write_deps = [
                d for d in op.get_read_writes().writes if isinstance(d, MemoryDep)
            ]
            if write_deps:
                write_dep = write_deps[0]
                # Resolve destination from MutationLayoutSHOULDREMOVE target.
                target = op.layout.target
                while isinstance(target, ReinterpretView):
                    target = target.data
                dest_buf = target if isinstance(target, ComputedBuffer) else None
                if dest_buf is not None and hasattr(dest_buf, "committed_stl"):
                    triples.append((dest_buf, write_dep, op))

        # Matmul ops: collect all ComputedBuffer inputs.
        is_matmul = (
            isinstance(op.data, Reduction)
            and op.data.reduction_type in MATMUL_REDUCTION_OPS
        )
        if is_matmul:
            for dep in op.get_read_writes().reads:
                if not isinstance(dep, MemoryDep):
                    continue
                if dep.name in graph_inputs:
                    continue
                buf = V.graph.get_buffer(dep.name)
                if not isinstance(buf, ComputedBuffer):
                    continue
                key = (dep.name, op.get_name())
                if key in seen:
                    continue
                seen.add(key)
                triples.append((buf, dep, op))

    return triples


def reorder_nonstick_dims(graph: GraphLowering) -> None:
    """Reorder non-stick dims for correctness (phase 1) and performance (phase 2).

    Phase 1 applies constraint transforms (gather IA).
    Phase 2 applies performance transforms (matmul perf reorder), respecting
    pinned_dims set by phase 1.
    """
    V.graph.nonstick_reorder_log = {}
    log: dict[str, SpyreTensorLayout] = {}

    triples = _collect_triples(graph)

    # Phase 1: constraint transforms.
    pinned_dims: dict[str, set[int]] = {}
    for buf, dep, op in triples:
        result = _try_gather_ia_constraint(buf, dep, op)
        if result is None and isinstance(op.data, Scatter):
            result = _try_scatter_ia_constraint(buf, dep, op)
        if result is not None:
            new_stl, pinned = result
            buf.committed_stl = new_stl
            log[buf.get_name()] = new_stl
            pinned_dims.setdefault(buf.get_name(), set()).update(pinned)

    # Phase 2: performance transforms.
    # Collect unique bufs that appeared in matmul triples for perf reorder.
    seen_bufs: set[str] = set()
    for buf, dep, op in triples:
        name = buf.get_name()
        if name in seen_bufs:
            continue
        seen_bufs.add(name)
        pinned = pinned_dims.get(name, set())
        reordered_stl = _try_matmul_perf_reorder(buf, pinned)
        if reordered_stl is not None:
            buf.committed_stl = reordered_stl
            log[name] = reordered_stl
            logger.info("nonstick_dim_order: reordered %s", name)

    V.graph.nonstick_reorder_log = log

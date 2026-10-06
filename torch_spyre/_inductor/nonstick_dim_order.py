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
from torch._inductor.ir import ComputedBuffer, Reduction
from torch._inductor.virtualized import V
from torch_spyre._C import ElementArrangement, SpyreTensorLayout

from .constants import MATMUL_REDUCTION_OPS
from .logging_utils import get_inductor_logger
from .pass_utils import try_device_coordinates

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


def _collect_matmul_input_bufs(
    graph: GraphLowering,
) -> list[ComputedBuffer]:
    """Walk graph backward; collect ComputedBuffers that feed matmul ops.

    Returns a deduplicated list of ComputedBuffer instances that are read by
    any matmul op and are not graph inputs.

    Indirect-access (gather/scatter) candidates will be added in follow-up tasks.
    """
    seen: set[str] = set()
    result: list[ComputedBuffer] = []
    graph_inputs = set(V.graph.graph_input_names)
    for op in reversed(graph.operations):
        if not isinstance(op, ComputedBuffer):
            continue
        if not hasattr(op, "data"):
            continue

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
                if dep.name in seen:
                    continue
                buf = V.graph.get_buffer(dep.name)
                if not isinstance(buf, ComputedBuffer):
                    continue
                seen.add(dep.name)
                result.append(buf)

    return result


def reorder_nonstick_dims(graph: GraphLowering) -> None:
    """Reorder non-stick dims on matmul inputs for better work division.

    Phase 1 (constraint transforms) is a placeholder for now — gather/scatter
    IA constraints will be added in a follow-up task.  Phase 2 (matmul perf
    reorder) runs with an empty pinned_dims dict.
    """
    V.graph.nonstick_reorder_log = {}
    log: dict[str, SpyreTensorLayout] = {}

    candidates = _collect_matmul_input_bufs(graph)

    # Phase 1: constraint transforms (gather IA, scatter IA).
    # pinned_dims maps buf_name -> set of device positions fixed by constraints.
    pinned_dims: dict[str, set[int]] = {}
    # (Constraint transforms will be added in the next task.)

    # Phase 2: performance transforms.
    for buf in candidates:
        pinned = pinned_dims.get(buf.get_name(), set())
        new_stl = _try_matmul_perf_reorder(buf, pinned)
        if new_stl is not None:
            buf.committed_stl = new_stl
            log[buf.get_name()] = new_stl
            logger.info("nonstick_dim_order: reordered %s", buf.get_name())

    V.graph.nonstick_reorder_log = log

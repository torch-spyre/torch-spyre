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

"""Reorder non-stick device dimensions for better work division.

Runs between propagate_layouts and optimize_restickify. Walks the graph
backward to find matmul inputs, then swaps the largest non-stick device dim
into the slot between the two stick dims (outer_stick+1), so the most work
is parallelised across the widest dimension.
"""

from torch._inductor.graph import GraphLowering
from torch._inductor.dependencies import MemoryDep
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
) -> SpyreTensorLayout:
    """Swap the largest non-stick dim into the slot between the two stick dims.

    A factorised stick produces two dims that share the same loop variable:
    floor(d/64) at position outer_stick and Mod(d/64) at the last position.
    This function moves the largest remaining dim into outer_stick+1 — the
    slot between them — so the compiler assigns the most iterations to the
    widest loop variable.
    """
    # Non-STANDARD element arrangements have hardware-defined dimension
    # semantics; reordering them corrupts the DDL template matching.
    if stl.element_arrangement != ElementArrangement.STANDARD:
        return stl
    device_size = list(stl.device_size)
    stride_map = list(stl.stride_map)
    n = len(device_size)
    if n <= 2:
        return stl

    idc = try_device_coordinates(stl, dep, {})
    if idc is None:
        return stl

    # Find the stick variable from the last dim's coordinate.
    stick_syms = idc[-1].free_symbols
    if not stick_syms:
        # Degenerate/broadcast stick (constant 0): nothing to do.
        return stl

    # Find the outer stick dim: the non-last dim that shares the stick variable.
    outer_stick = None
    for i in range(n - 2, -1, -1):
        if idc[i].free_symbols & stick_syms:
            outer_stick = i
            break
    if outer_stick is None:
        return stl  # unsplit stick, no slot to fill

    slot = outer_stick + 1
    logger.debug(
        "nonstick_dim_order: %s idc=%s outer_stick=%d slot=%d n=%d",
        name,
        [str(x) for x in idc],
        outer_stick,
        slot,
        n,
    )
    if slot >= n - 1:
        logger.debug(
            "nonstick_dim_order: skipping %s — no room between stick dims"
            " (outer_stick=%d, n=%d)",
            name,
            outer_stick,
            n,
        )
        return stl

    # Only move dims from outside (before outer_stick) into the slot,
    # and only if the largest outside dim is bigger than what's already there.
    candidates = list(range(outer_stick))
    if not candidates:
        return stl
    largest = max(candidates, key=lambda d: device_size[d])
    if device_size[largest] <= device_size[slot]:
        return stl  # already optimal or nothing to gain

    # Swap largest into slot.
    new_order = list(range(n))
    new_order[slot], new_order[largest] = new_order[largest], new_order[slot]
    new_device_size = [device_size[d] for d in new_order]
    new_stride_map = [stride_map[d] for d in new_order]
    return SpyreTensorLayout(
        device_size=new_device_size,
        stride_map=new_stride_map,
        device_dtype=stl.device_dtype,
    )


def _backward_pass(graph: GraphLowering) -> set[str]:
    """Walk graph in reverse; collect names of matmul inputs to reorder."""
    targets: set[str] = set()
    graph_inputs = set(V.graph.graph_input_names)
    for op in reversed(graph.operations):
        if not hasattr(op, "data"):
            continue
        if not isinstance(op.data, Reduction):
            continue
        if op.data.reduction_type not in MATMUL_REDUCTION_OPS:
            continue
        out_name = op.get_name()
        # if out_name not in graph_inputs:
        #     targets.add(out_name)
        for dep in op.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            if dep.name in graph_inputs:
                continue
            targets.add(dep.name)
            logger.debug(
                "nonstick_dim_order: matmul %s requests reorder on %s",
                out_name,
                dep.name,
            )
    return targets


def _forward_pass(targets: set[str]) -> None:
    """Reorder candidate STLs for each target buffer.

    Writes V.graph.nonstick_reorder_log: dict[str, list[SpyreTensorLayout]]
    mapping each reordered buffer name to its new layouts list.
    """
    log: dict[str, list] = {}
    for name in targets:
        buf = V.graph.get_buffer(name)
        if not hasattr(buf, "layouts"):
            logger.debug(
                "nonstick_dim_order: skipping %s — no .layouts attribute", name
            )
            continue
        # Only reorder ComputedBuffer outputs. ExternKernel outputs (FallbackKernel,
        # MultiOutput, etc.) have placeholder layouts whose actual arrangement is
        # determined by the external op; reordering them corrupts DDL template matching.
        if not isinstance(buf, ComputedBuffer):
            logger.debug(
                "nonstick_dim_order: skipping %s — not a ComputedBuffer (%s)",
                name,
                type(buf).__name__,
            )
            continue
        # Use the buffer's write dep to compute device coordinates for each
        # candidate STL, so we can identify the outer stick dim.
        write_dep = next(iter(buf.get_read_writes().writes), None)
        new_layouts = [
            _reorder_stl(stl, write_dep, name) if write_dep is not None else stl
            for stl in buf.layouts
        ]
        changed = any(
            list(a.device_size) != list(b.device_size)
            for a, b in zip(buf.layouts, new_layouts)
        )
        if changed:
            for i, (old, new) in enumerate(zip(buf.layouts, new_layouts)):
                if list(old.device_size) != list(new.device_size):
                    logger.debug(
                        "[NDO] %s[%d]  %s -> %s  stride_map %s -> %s",
                        name,
                        i,
                        list(old.device_size),
                        list(new.device_size),
                        list(old.stride_map),
                        list(new.stride_map),
                    )
            buf.layouts[:] = new_layouts
            log[name] = list(buf.layouts)
            logger.info(
                "nonstick_dim_order: reordered %s (%d candidates)",
                name,
                len(buf.layouts),
            )
    V.graph.nonstick_reorder_log = log


def reorder_nonstick_dims(graph: GraphLowering) -> None:
    """Reorder non-stick dims on matmul inputs for better work division."""
    targets = _backward_pass(graph)
    if targets:
        _forward_pass(targets)

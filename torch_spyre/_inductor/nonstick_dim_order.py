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

Runs after optimize_restickify and before finalize_layouts. Walks the graph
backward over MATMUL_REDUCTION_OPS and rewrites buf.committed_stl on their
inputs. optimize_restickify has already committed its stick choices, so these
rewrites affect only non-stick dim ordering — they do not influence stick
decisions.

Two transforms, tried in order:

  flat-M projection
    A fused SDPA output may have a higher-rank host view (e.g. [B, L, H, D])
    even though the downstream projection matmul reads it as a flat 2-D matrix
    [B*L, H*D]. Retaining those outer axes causes the backend to treat the op
    as a BMM rather than a flat MM. This transform collapses committed_stl to
    the canonical flat-M shape [M, K] when the access pattern is provably a
    single dense row-major 2-D matrix with a rank-2 weight.

  dim reorder
    For factorised-stick layouts, the stick variable occupies two dims:
    floor(d/64) at outer_stick and Mod(d, 64) at the last position. The slot
    between them (outer_stick+1) is where the work-division loop variable
    should sit for maximum parallelism. This transform swaps the largest
    non-constant outer dim into that slot.

Flat-M projection takes priority: if it fires, dim reorder is skipped for
that buffer, since the collapsed layout already has the right shape.
"""

import math
from typing import TYPE_CHECKING

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FixedLayout, Reduction
from torch._inductor.virtualized import V
from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout

if TYPE_CHECKING:
    from torch._inductor.ir import Operation

from .constants import BATCH_MATMUL_OP, MATMUL_REDUCTION_OPS
from .errors import Unsupported
from .logging_utils import get_inductor_logger
from .pass_utils import (
    concretize_expr,
    find_reduction_var,
    get_matmul_m_size,
    get_matmul_n_size,
    try_device_coordinates,
)
from .propagate_layouts import PropArg

logger = get_inductor_logger("nonstick_dim_order")


def _flat_dense_projection_x_layout(
    x: PropArg,
    y: PropArg,
    output: FixedLayout,
    output_dep: MemoryDep,
    reduction_var: sympy.Symbol,
    m_size: int,
    n_size: int,
) -> SpyreTensorLayout | None:
    """Return a canonical flat-M STL for a logically 2-D dense projection, or None.

    A fused attention producer can retain a higher-rank contiguous host view
    such as ``[B, L, H, D]`` even though the projection reads it as the logical
    matrix ``[B*L, H*D]``. Preserving those physical outer axes makes the
    backend encode the shared-weight projection as a BMM, which is much slower
    than the equivalent flat MM. Collapse only when the complete access is
    provably one dense row-major 2-D matrix and the weight is rank-2.
    """
    if (
        m_size <= 1
        or n_size <= 1
        or len(output.size) != 2
        or len(x.layout.size) <= 2
        or len(y.layout.size) != 2
        or not x.layouts
        or x.layouts[0].element_arrangement != ElementArrangement.STANDARD
        or x.layouts[0].device_dtype != DataFormats.SEN169_FP16
    ):
        return None

    x_size = [concretize_expr(s) for s in x.layout.size]
    x_stride = [concretize_expr(s) for s in x.layout.stride]
    y_size = [concretize_expr(s) for s in y.layout.size]
    out_size = [concretize_expr(s) for s in output.size]
    out_stride = [concretize_expr(s) for s in output.stride]

    def is_dense_contiguous(size: list[int], stride: list[int]) -> bool:
        expected = 1
        for dim_size, dim_stride in zip(reversed(size), reversed(stride)):
            if dim_size != 1 and dim_stride != expected:
                return False
            expected *= dim_size
        return True

    if not is_dense_contiguous(x_size, x_stride) or not is_dense_contiguous(
        out_size, out_stride
    ):
        return None

    active_x_vars = set(x.dep.index.free_symbols) & set(x.dep.ranges)
    row_vars = active_x_vars - {reduction_var}
    if reduction_var not in active_x_vars or len(row_vars) != 1:
        return None
    (row_var,) = row_vars

    row_size = concretize_expr(x.dep.ranges[row_var])
    reduction_size = concretize_expr(x.dep.ranges[reduction_var])
    active_out_vars = set(output_dep.index.free_symbols) & set(output_dep.ranges)
    generated_vars = active_out_vars - {row_var}
    if row_var not in active_out_vars or len(generated_vars) != 1:
        return None
    (generated_var,) = generated_vars
    generated_size = concretize_expr(output_dep.ranges[generated_var])

    if (
        row_size != m_size
        or generated_size != n_size
        or math.prod(x_size) != row_size * reduction_size
        or math.prod(y_size) != reduction_size * generated_size
        or math.prod(out_size) != m_size * n_size
        or row_var in y.dep.index.free_symbols
        or {reduction_var, generated_var}
        != (set(y.dep.index.free_symbols) & set(y.dep.ranges))
    ):
        return None

    expected_index = reduction_size * row_var + reduction_var
    expected_output_index = generated_size * row_var + generated_var
    if (
        sympy.simplify(x.dep.index - expected_index) != 0
        or sympy.simplify(output_dep.index - expected_output_index) != 0
    ):
        return None

    return SpyreTensorLayout(
        [row_size, reduction_size],
        [reduction_size, 1],
        x.layout.dtype,
        [0, 1],
        ElementArrangement.STANDARD,
    )


def _try_flat_m_projection(
    buf: ComputedBuffer,
    x_dep: MemoryDep,
    op: "Operation",
) -> SpyreTensorLayout | None:
    """Return a flat-M committed_stl for buf, or None if not applicable.

    Requires the full matmul context (both operands and the output dep) to
    verify the access is a dense 2-D matrix. Returns None if the layout does
    not qualify or if shape helpers raise Unsupported (e.g. dynamic shapes).
    """
    return None  # Temp: disabled for performance comparison
    if op.data.reduction_type != BATCH_MATMUL_OP:
        return None
    if len(buf.get_layout().size) <= 2:
        return None
    reads = [r for r in op.get_read_writes().reads if isinstance(r, MemoryDep)]
    y_deps = [r for r in reads if r.name != x_dep.name]
    out_deps = list(op.get_read_writes().writes)

    if len(y_deps) < 1 or len(out_deps) < 1:
        return None

    y_dep = y_deps[0]
    out_dep = out_deps[0]
    y_buf = V.graph.get_buffer(y_dep.name)
    out_buf = V.graph.get_buffer(op.get_name())
    if not hasattr(y_buf, "committed_stl"):
        return None

    x_prop = PropArg(x_dep, buf.get_layout(), [buf.committed_stl])
    y_prop = PropArg(y_dep, y_buf.get_layout(), [y_buf.committed_stl])
    out_host = out_buf.get_layout()
    try:
        reduction_var = find_reduction_var((x_dep,), out_dep)
        m_size = get_matmul_m_size(op)
        n_size = get_matmul_n_size(op)
        flat_stl = _flat_dense_projection_x_layout(
            x_prop, y_prop, out_host, out_dep, reduction_var, m_size, n_size
        )
    except Unsupported:
        return None

    if flat_stl is None:
        return None

    logger.info(
        "nonstick_dim_order: flat-M projection on %s — %s -> %s",
        x_dep.name,
        list(buf.committed_stl.device_size),
        list(flat_stl.device_size),
    )
    return flat_stl


def _move_largest_dim_between_sticks(
    stl: SpyreTensorLayout,
    dep: MemoryDep,
    name: str = "",
) -> SpyreTensorLayout:
    """Move the largest non-stick dim into the slot between the two stick dims.

    A factorised stick occupies two positions: floor(d/64) at outer_stick and
    Mod(d, 64) at the last position. Moving the largest non-constant outer dim
    into outer_stick+1 puts the widest loop variable between the sticks, which
    maximises the number of iterations assigned to that work-division slot.
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
    # Exclude dims with constant (zero free-symbol) coordinates — these are
    # padding/gap dims prepended by restickify/compact and must not be moved.
    candidates = [d for d in range(outer_stick) if idc[d].free_symbols]
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


def _reorder_for_matmul_perf(
    buf: ComputedBuffer,
    x_dep: MemoryDep,
) -> SpyreTensorLayout | None:
    """Reorder committed_stl to move the largest dim between the sticks.

    Uses the buffer's own write dep (not the matmul's read dep) so the
    coordinate expression reflects how the buffer itself writes its elements.
    Returns None if the layout does not change.
    """
    write_dep = next(iter(buf.get_read_writes().writes), None)
    if write_dep is None:
        return None
    new_stl = _move_largest_dim_between_sticks(buf.committed_stl, write_dep, x_dep.name)
    if list(new_stl.device_size) == list(buf.committed_stl.device_size):
        return None
    logger.debug(
        "[NDO] %s  %s -> %s  stride_map %s -> %s",
        x_dep.name,
        list(buf.committed_stl.device_size),
        list(new_stl.device_size),
        list(buf.committed_stl.stride_map),
        list(new_stl.stride_map),
    )
    return new_stl


def _compute_nonstick_layout(
    buf: ComputedBuffer,
    x_dep: MemoryDep,
    op: "Operation",
) -> SpyreTensorLayout | None:
    """Return a replacement committed_stl for buf, or None if no change is needed.

    Tries flat-M projection first; falls through to dim reorder if it does not
    apply. Returns None if neither transform produces a change.
    """
    if not hasattr(buf, "committed_stl"):
        return None
    return _try_flat_m_projection(buf, x_dep, op) or _reorder_for_matmul_perf(
        buf, x_dep
    )


def reorder_nonstick_dims(graph: GraphLowering) -> None:
    """Reorder non-stick dims on matmul inputs for better work division.

    Walks the graph backward so that a buffer produced by one matmul and
    consumed by another is visited in consumer-first order, letting each
    consumer's preferred shape propagate toward its producer.

    Graph inputs (weights, activations) are skipped — their layouts are owned
    by the caller and cannot be changed here.

    Writes the replacement STL into buf.committed_stl in place. This pass
    runs after optimize_restickify, so stick choices are already committed and
    these rewrites affect only non-stick dim ordering.

    Results are saved to V.graph.nonstick_reorder_log for test inspection.
    """
    log: dict[str, SpyreTensorLayout] = {}
    graph_inputs = set(V.graph.graph_input_names)

    for op in reversed(graph.operations):
        if not isinstance(getattr(op, "data", None), Reduction):
            continue
        if op.data.reduction_type not in MATMUL_REDUCTION_OPS:
            continue

        for dep in op.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            if dep.name in graph_inputs:
                continue
            buf = V.graph.get_buffer(dep.name)
            if not isinstance(buf, ComputedBuffer):
                logger.debug(
                    "nonstick_dim_order: skipping %s — not a ComputedBuffer", dep.name
                )
                continue

            new_stl = _compute_nonstick_layout(buf, dep, op)
            if new_stl is not None:
                buf.committed_stl = new_stl
                log[dep.name] = new_stl
                logger.info(
                    "nonstick_dim_order: reordered %s",
                    dep.name,
                )

    V.graph.nonstick_reorder_log = log

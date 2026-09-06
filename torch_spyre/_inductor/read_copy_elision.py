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

"""Remove read copies only after the final pre-scheduling plan proves safety."""

from __future__ import annotations

import dataclasses

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    InputBuffer,
    Operation,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V

from . import config
from .constants import MATMUL_REDUCTION_OPS
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .loop_info import CoarseTileInfo, ReadCopyElisionRecord, copy_op_metadata
from .pass_utils import (
    _per_core_view_on_buf,
    device_coordinates,
    find_matmul_generated_var,
    identify_matmul_inputs,
    replace_computed_buffer_body,
)
from .wsr.coarse_tile import (
    validate_reader_tile_advance,
    validate_writer_tile_advance,
)

logger = get_inductor_logger("read_copy_elision")


def _memory_deps(deps) -> list[MemoryDep]:
    return [dep for dep in deps if isinstance(dep, MemoryDep)]


def _one_memory_dep(deps) -> MemoryDep | None:
    result = _memory_deps(deps)
    return result[0] if len(result) == 1 else None


def _unwrap_buffer(buf):
    if isinstance(buf, TensorBox):
        buf = buf.data
    if isinstance(buf, StorageBox):
        buf = buf.data
    return buf


def _static_int(expr: sympy.Expr) -> int | None:
    expr = sympy.sympify(expr)
    if not expr.is_number or expr.is_integer is False:
        return None
    try:
        return int(expr)
    except TypeError:
        return None


def _affine_bounds(dep: MemoryDep) -> tuple[int, int] | None:
    """Inclusive min/max source indexes for one invocation, in elements."""
    expr = sympy.expand(dep.index)
    zero_subs = {var: sympy.Integer(0) for var in dep.var_names}
    constant = sympy.expand(expr.subs(zero_subs))
    lo = _static_int(constant)
    if lo is None:
        return None
    hi = lo
    residual = expr - constant
    for var, size_expr in zip(dep.var_names, dep.size):
        coeff = sympy.expand(expr).coeff(var)
        coeff_i = _static_int(coeff)
        size_i = _static_int(size_expr)
        if coeff_i is None or size_i is None or size_i < 1:
            return None
        residual -= coeff * var
        delta = coeff_i * (size_i - 1)
        lo += min(0, delta)
        hi += max(0, delta)
    if sympy.simplify(residual) != 0:
        return None
    return lo, hi


def _loop_advance_bound(
    loop_info: CoarseTileInfo, dep_idx: int
) -> tuple[int, int] | None:
    """Inclusive address change across loop trips, in source elements."""
    if dep_idx >= len(loop_info.tiled_dims_per_read):
        return None
    if any(loop_info.tiled_dims_per_read[dep_idx]):
        return None
    squeezed = (
        loop_info.squeezed_advance_per_read[dep_idx]
        if dep_idx < len(loop_info.squeezed_advance_per_read)
        else []
    )
    if not any(squeezed):
        return None

    lo = 0
    hi = 0
    for level_idx, trip_expr in enumerate(loop_info.loop_count):
        trip_count = _static_int(trip_expr)
        if trip_count is None or trip_count < 1:
            return None
        pairs = squeezed[level_idx] if level_idx < len(squeezed) else []
        step = 0
        for stride_expr, extent_expr in pairs:
            stride = _static_int(stride_expr)
            extent = _static_int(extent_expr)
            if stride is None or extent is None:
                return None
            step += stride * extent
        delta = step * (trip_count - 1)
        lo += min(0, delta)
        hi += max(0, delta)
    return lo, hi


def _copy_readers(operations: list[Operation], copy_name: str) -> list[ComputedBuffer]:
    readers = []
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        if any(
            dep.name == copy_name for dep in _memory_deps(op.get_read_writes().reads)
        ):
            readers.append(op)
    return readers


def _clone_direct_consumer(
    consumer: ComputedBuffer, record: ReadCopyElisionRecord
) -> ComputedBuffer:
    direct_data = dataclasses.replace(consumer.data, inner_fn=record.direct_inner_fn)
    direct_op = ComputedBuffer(
        name=consumer.get_name(),
        layout=consumer.layout,
        data=direct_data,
        _split_size=consumer._split_size,
        _original_inner_fn=consumer._original_inner_fn,
        _original_ranges=consumer._original_ranges,
        _original_reduction_ranges=consumer._original_reduction_ranges,
    )
    direct_op.operation_name = consumer.operation_name
    direct_op.origins = consumer.origins
    copy_op_metadata(consumer, direct_op)
    for attr in ("layouts", "restick_cost_fn", "op_it_space_splits"):
        if hasattr(consumer, attr):
            setattr(direct_op, attr, getattr(consumer, attr))
    return direct_op


def _prove_matmul_direct_read(
    consumer: ComputedBuffer,
    copy_op: ComputedBuffer,
    record: ReadCopyElisionRecord,
) -> tuple[ComputedBuffer, str] | tuple[None, str]:
    """Build, but do not install, a direct-read matmul when every fact holds."""
    if consumer.get_name() != record.consumer_name:
        return None, "consumer identity changed"
    if not isinstance(consumer.data, Reduction):
        return None, "consumer is not a reduction"
    if consumer.data.reduction_type not in MATMUL_REDUCTION_OPS:
        return None, "consumer is not a matmul"

    copy_layout = copy_op.get_layout()
    if not isinstance(copy_layout, FixedTiledLayout):
        return None, "copy has no final device layout"
    if "lx" in copy_layout.allocation:
        return None, "LX planning retained the copy"

    direct_op = _clone_direct_consumer(consumer, record)
    direct_rw = direct_op.get_read_writes()
    direct_reads = _memory_deps(direct_rw.reads)
    output_dep = _one_memory_dep(direct_rw.writes)
    if output_dep is None or len(direct_reads) != 2:
        return None, "matmul dependencies are ambiguous"
    x_dep, weight_dep = identify_matmul_inputs(direct_reads, output_dep)
    if x_dep is None or weight_dep is None or weight_dep.name != record.source_name:
        return None, "saved source is not the matmul weight"
    direct_source_idx = direct_reads.index(weight_dep)

    current_rw = consumer.get_read_writes()
    current_reads = _memory_deps(current_rw.reads)
    current_output = _one_memory_dep(current_rw.writes)
    copy_deps = [dep for dep in current_reads if dep.name == record.copy_name]
    if current_output is None or len(copy_deps) != 1:
        return None, "consumer no longer reads the saved copy exactly once"
    copy_dep = copy_deps[0]
    current_non_weight = [dep for dep in current_reads if dep != copy_dep]
    direct_non_weight = [dep for dep in direct_reads if dep != weight_dep]
    if current_non_weight != direct_non_weight:
        return None, "another matmul input changed after recording"
    if current_output != output_dep or consumer.get_layout() != direct_op.get_layout():
        return None, "output layout or index changed"

    copy_reads = _memory_deps(copy_op.get_read_writes().reads)
    copy_source_indices = [
        idx for idx, dep in enumerate(copy_reads) if dep.name == record.source_name
    ]
    copy_loop_info = getattr(copy_op, "loop_info", None)
    current_loop_info = getattr(consumer, "loop_info", None)
    if (
        len(copy_source_indices) != 1
        or not isinstance(copy_loop_info, CoarseTileInfo)
        or not isinstance(current_loop_info, CoarseTileInfo)
    ):
        return None, "copy has no complete loop-address record"
    copy_source_idx = copy_source_indices[0]
    if copy_source_idx >= len(copy_loop_info.tiled_dims_per_read):
        return None, "copy has no tiled-dimension record for its source"

    tiled_dims = [
        [list(level) for level in per_read]
        for per_read in current_loop_info.tiled_dims_per_read
    ]
    squeezed = [
        [list(level) for level in per_read]
        for per_read in current_loop_info.squeezed_advance_per_read
    ]
    if direct_source_idx >= len(tiled_dims):
        return None, "direct-read metadata does not match its dependencies"
    squeezed.extend([] for _ in range(len(direct_reads) - len(squeezed)))
    tiled_dims[direct_source_idx] = [
        list(level) for level in copy_loop_info.tiled_dims_per_read[copy_source_idx]
    ]
    copy_squeezed = (
        copy_loop_info.squeezed_advance_per_read[copy_source_idx]
        if copy_source_idx < len(copy_loop_info.squeezed_advance_per_read)
        else []
    )
    squeezed[direct_source_idx] = [list(level) for level in copy_squeezed]
    resolved_loop_info = dataclasses.replace(
        current_loop_info,
        tiled_dims_per_read=tiled_dims,
        squeezed_advance_per_read=squeezed,
    )
    direct_op.loop_info = resolved_loop_info  # type: ignore[attr-defined]

    source = _unwrap_buffer(V.graph.get_buffer(record.source_name))
    if not isinstance(source, InputBuffer):
        return None, "source is not a graph input"
    source_layout = source.get_layout()
    if not isinstance(source_layout, FixedTiledLayout):
        return None, "source has no final device layout"

    try:
        generated_var = find_matmul_generated_var(
            weight_dep, x_dep, output_dep, direct_op
        )
        source_stick = device_coordinates(
            source_layout.device_layout, weight_dep, None
        )[-1]
    except Exception as exc:
        return None, f"source layout is not directly readable: {exc}"
    if generated_var not in source_stick.free_symbols:
        return None, "source stick dimension is not the matmul output dimension"

    local_bounds = _affine_bounds(weight_dep)
    advance_bounds = _loop_advance_bound(resolved_loop_info, direct_source_idx)
    source_numel = _static_int(sympy.prod(source_layout.size))
    if local_bounds is None or advance_bounds is None or source_numel is None:
        return None, "source element range is not statically provable"
    lo = local_bounds[0] + advance_bounds[0]
    hi = local_bounds[1] + advance_bounds[1]
    if lo < 0 or hi >= source_numel:
        return None, f"source element range [{lo}, {hi}] exceeds [0, {source_numel})"

    old_view, _, old_ok = _per_core_view_on_buf(
        consumer, current_output, current_output.name
    )
    new_view, _, new_ok = _per_core_view_on_buf(direct_op, output_dep, output_dep.name)
    copy_view, _, copy_ok = _per_core_view_on_buf(consumer, copy_dep, record.copy_name)
    source_view, _, source_ok = _per_core_view_on_buf(
        direct_op, weight_dep, record.source_name
    )
    if not old_ok or not new_ok or old_view != new_view:
        return None, "output core ownership changed"
    if not copy_ok or not source_ok:
        return None, "weight core ownership is not representable"
    # Representability alone is insufficient: both views may be legal while
    # assigning different source slices to the same core.  The staging copy
    # is the behavior being replaced, so the direct HBM read must preserve
    # its exact physical core-to-slice map.
    if copy_view != source_view:
        return None, (
            "direct source ownership does not match the staged copy: "
            f"{source_view} != {copy_view}"
        )
    if getattr(consumer, "op_it_space_splits", None) != getattr(
        direct_op, "op_it_space_splits", None
    ):
        return None, "work division changed"

    return direct_op, "proved"


def _validate_proposal(
    operations: list[Operation],
    consumer: ComputedBuffer,
    direct_op: ComputedBuffer,
    copy_op: ComputedBuffer,
) -> str | None:
    proposed = [
        direct_op if op is consumer else op for op in operations if op is not copy_op
    ]
    try:
        validate_writer_tile_advance(proposed)
        validate_reader_tile_advance(proposed)
    except Exception as exc:
        return f"coarse-tile validation failed: {exc}"
    return None


def elide_proven_read_copies(graph: GraphLowering) -> None:
    """Remove only copies whose post-allocation direct-read proof succeeds."""
    if not config.read_copy_elision:
        return

    operations = graph.operations
    for consumer in list(operations):
        record = getattr(consumer, "_read_copy_elision_record", None)
        if not isinstance(consumer, ComputedBuffer) or not isinstance(
            record, ReadCopyElisionRecord
        ):
            continue
        copy_op = next(
            (
                op
                for op in operations
                if isinstance(op, ComputedBuffer) and op.get_name() == record.copy_name
            ),
            None,
        )
        if copy_op is None:
            logger.info(
                "read-copy elision declined for %s: copy is absent",
                record.consumer_name,
            )
            continue
        if _copy_readers(operations, record.copy_name) != [consumer]:
            logger.info(
                "read-copy elision declined for %s: copy has other readers",
                record.consumer_name,
            )
            continue

        direct_op, detail = _prove_matmul_direct_read(consumer, copy_op, record)
        if direct_op is None:
            logger.info(
                "read-copy elision declined for %s: %s",
                record.consumer_name,
                detail,
            )
            continue
        validation_error = _validate_proposal(operations, consumer, direct_op, copy_op)
        if validation_error is not None:
            logger.info(
                "read-copy elision declined for %s: %s",
                record.consumer_name,
                validation_error,
            )
            continue

        replacement = replace_computed_buffer_body(
            consumer,
            direct_op.data,
            operations,
            pass_name="read_copy_elision",
            reason="read advancing matmul weights directly from HBM",
        )
        replacement.loop_info = direct_op.loop_info  # type: ignore[attr-defined]
        for attr in ("layouts", "restick_cost_fn", "op_it_space_splits"):
            if hasattr(direct_op, attr):
                setattr(replacement, attr, getattr(direct_op, attr))
        if hasattr(replacement, "_read_copy_elision_record"):
            del replacement._read_copy_elision_record  # type: ignore[attr-defined]
        V.graph.name_to_buffer[replacement.get_name()] = replacement
        graph.removed_buffers.add(copy_op.get_name())
        operations.remove(copy_op)
        logger.info(
            "removed read copy %s; %s reads %s directly",
            copy_op.get_name(),
            replacement.get_name(),
            record.source_name,
        )

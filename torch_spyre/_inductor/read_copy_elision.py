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
import copy

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    InputBuffer,
    Operation,
    Pointwise,
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
    commit_iteration_space_ownership,
    device_coordinates,
    find_matmul_generated_var,
    identify_matmul_inputs,
    per_core_views_equal,
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
    op: ComputedBuffer,
    dep: MemoryDep,
    loop_info: CoarseTileInfo,
    dep_idx: int,
) -> tuple[int, int] | None:
    """Inclusive address change across loop trips, in source elements."""
    if dep_idx >= len(loop_info.tiled_dims_per_read):
        return None
    tiled = loop_info.tiled_dims_per_read[dep_idx]
    squeezed = (
        loop_info.squeezed_advance_per_read[dep_idx]
        if dep_idx < len(loop_info.squeezed_advance_per_read)
        else []
    )
    if not any(tiled) and not any(squeezed):
        return None

    ranges = list(getattr(op.data, "ranges", None) or [])
    reduction_ranges = list(getattr(op.data, "reduction_ranges", None) or [])
    raw_to_symbol: dict[int, sympy.Symbol] = {}
    squeezed_idx = 0
    for raw_idx, extent in enumerate(ranges):
        if _static_int(extent) != 1:
            if squeezed_idx >= len(dep.var_names):
                return None
            raw_to_symbol[raw_idx] = dep.var_names[squeezed_idx]
            squeezed_idx += 1
    for red_idx, extent in enumerate(reduction_ranges):
        if _static_int(extent) != 1:
            if squeezed_idx >= len(dep.var_names):
                return None
            raw_to_symbol[len(ranges) + red_idx] = dep.var_names[squeezed_idx]
            squeezed_idx += 1

    lo = 0
    hi = 0
    for level_idx, trip_expr in enumerate(loop_info.loop_count):
        trip_count = _static_int(trip_expr)
        if trip_count is None or trip_count < 1:
            return None
        dim_pairs = tiled[level_idx] if level_idx < len(tiled) else []
        zero_subs = {symbol: sympy.Integer(0) for symbol in dep.index.free_symbols}
        step_expr = sympy.Integer(0)
        for raw_dim, extent_expr in dim_pairs:
            symbol = raw_to_symbol.get(raw_dim)
            extent = _static_int(extent_expr)
            if symbol is None or extent is None:
                return None
            level_subs = dict(zero_subs)
            level_subs[symbol] = sympy.Integer(extent)
            step_expr += sympy.expand(dep.index.subs(level_subs))
            step_expr -= sympy.expand(dep.index.subs(zero_subs))
        pairs = squeezed[level_idx] if level_idx < len(squeezed) else []
        step = 0
        for stride_expr, extent_expr in pairs:
            stride = _static_int(stride_expr)
            extent = _static_int(extent_expr)
            if stride is None or extent is None:
                return None
            step += stride * extent
        tiled_step = _static_int(step_expr)
        if tiled_step is None:
            return None
        step += tiled_step
        delta = step * (trip_count - 1)
        lo += min(0, delta)
        hi += max(0, delta)
    return lo, hi


def _copy_readers(operations: list[Operation], copy_name: str) -> list[Operation]:
    readers = []
    for op in operations:
        if any(
            getattr(dep, "name", None) == copy_name
            for dep in op.get_read_writes().reads
        ):
            readers.append(op)
    return readers


def _same_storage_encoding(
    source_layout: FixedTiledLayout, copy_layout: FixedTiledLayout
) -> bool:
    """Whether bypassing ``copy_layout`` preserves the stored value encoding."""
    source_stl = source_layout.device_layout
    copy_stl = copy_layout.device_layout
    return (
        source_layout.dtype == copy_layout.dtype
        and source_stl.device_dtype == copy_stl.device_dtype
        and source_stl.element_arrangement == copy_stl.element_arrangement
    )


def _logical_split_ownership(
    op: ComputedBuffer, dep: MemoryDep
) -> tuple[tuple[sympy.Symbol, int, sympy.Expr], ...] | None:
    """Logical tensor slices this operation assigns to each physical core.

    A staged copy may permute physical device axes, so comparing its
    :class:`PerCoreView` directly with a graph input can reject equivalent
    accesses.  Both dependencies are read by the same operation, however; if
    they depend on exactly the same split iteration symbols, the operation's
    committed owner formula assigns the same logical slice to every core.
    """
    ownership = getattr(op, "iteration_space_ownership", None)
    if ownership is None:
        return None
    if set(ownership.work_slices) != set(ownership.core_id_to_work_slice):
        return None
    return tuple(
        sorted(
            [
                (
                    symbol,
                    int(split),
                    sympy.simplify(ownership.core_id_to_work_slice[symbol]),
                )
                for symbol, split in ownership.work_slices.items()
                if int(split) > 1 and dep.index.has(symbol)
            ],
            key=lambda item: str(item[0]),
        )
    )


def _direct_source_ownership_matches(
    consumer: ComputedBuffer,
    direct_op: ComputedBuffer,
    copy_dep: MemoryDep,
    source_dep: MemoryDep,
    copy_view,
    source_view,
) -> bool:
    """Whether bypassing a staged copy preserves each core's logical slice."""
    if per_core_views_equal(copy_view, source_view):
        return True
    copy_ownership = _logical_split_ownership(consumer, copy_dep)
    source_ownership = _logical_split_ownership(direct_op, source_dep)
    return copy_ownership is not None and copy_ownership == source_ownership


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
    """Build, but do not install, a direct-read op when every fact holds."""
    if consumer.get_name() != record.consumer_name:
        return None, "consumer identity changed"
    is_matmul = isinstance(consumer.data, Reduction) and (
        consumer.data.reduction_type in MATMUL_REDUCTION_OPS
    )
    if not is_matmul and not isinstance(consumer.data, Pointwise):
        return None, "consumer is not pointwise or a matmul"

    copy_layout = copy_op.get_layout()
    if not isinstance(copy_layout, FixedTiledLayout):
        return None, "copy has no final device layout"

    direct_op = _clone_direct_consumer(consumer, record)
    direct_rw = direct_op.get_read_writes()
    direct_reads = _memory_deps(direct_rw.reads)
    output_dep = _one_memory_dep(direct_rw.writes)
    direct_source_deps = [dep for dep in direct_reads if dep.name == record.source_name]
    if output_dep is None or len(direct_source_deps) != 1:
        return None, "direct source dependency is ambiguous"
    source_dep = direct_source_deps[0]
    direct_source_idx = direct_reads.index(source_dep)

    if is_matmul:
        if len(direct_reads) != 2:
            return None, "matmul dependencies are ambiguous"
        x_dep, weight_dep = identify_matmul_inputs(direct_reads, output_dep)
        if x_dep is None or weight_dep != source_dep:
            return None, "saved source is not the matmul weight"

    current_rw = consumer.get_read_writes()
    current_reads = _memory_deps(current_rw.reads)
    current_output = _one_memory_dep(current_rw.writes)
    copy_deps = [dep for dep in current_reads if dep.name == record.copy_name]
    if current_output is None or len(copy_deps) != 1:
        return None, "consumer no longer reads the saved copy exactly once"
    copy_dep = copy_deps[0]
    current_other_reads = [dep for dep in current_reads if dep != copy_dep]
    direct_other_reads = [dep for dep in direct_reads if dep != source_dep]
    if current_other_reads != direct_other_reads:
        return None, (
            "another input changed after recording: "
            f"current={current_other_reads}, direct={direct_other_reads}"
        )
    if current_output != output_dep or consumer.get_layout() != direct_op.get_layout():
        return None, "output layout or index changed"

    copy_reads = _memory_deps(copy_op.get_read_writes().reads)
    copy_source_indices = [
        idx for idx, dep in enumerate(copy_reads) if dep.name == record.source_name
    ]
    copy_loop_info = getattr(copy_op, "loop_info", None)
    current_loop_info = getattr(consumer, "loop_info", None)
    if len(copy_source_indices) > 1 or not isinstance(
        current_loop_info, CoarseTileInfo
    ):
        return None, "copy has no complete loop-address record"
    recorded_tiled = record.direct_tiled_dims_per_level
    recorded_squeezed = record.direct_squeezed_advance_per_level
    has_recorded_address = recorded_tiled is not None and recorded_squeezed is not None
    if not has_recorded_address and (
        len(copy_source_indices) != 1 or not isinstance(copy_loop_info, CoarseTileInfo)
    ):
        return None, "copy has no complete loop-address record"
    copy_source_idx = copy_source_indices[0] if copy_source_indices else None
    if (
        isinstance(copy_loop_info, CoarseTileInfo)
        and copy_source_idx is not None
        and copy_source_idx >= len(copy_loop_info.tiled_dims_per_read)
    ):
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
    if recorded_tiled is None or recorded_squeezed is None:
        assert isinstance(copy_loop_info, CoarseTileInfo)
        assert copy_source_idx is not None
        recorded_tiled = tuple(
            tuple(tuple(pair) for pair in level)
            for level in copy_loop_info.tiled_dims_per_read[copy_source_idx]
        )
        copy_squeezed = (
            copy_loop_info.squeezed_advance_per_read[copy_source_idx]
            if copy_source_idx < len(copy_loop_info.squeezed_advance_per_read)
            else []
        )
        recorded_squeezed = tuple(
            tuple(tuple(pair) for pair in level) for level in copy_squeezed
        )
    tiled_dims[direct_source_idx] = [
        [tuple(pair) for pair in level] for level in recorded_tiled
    ]
    squeezed[direct_source_idx] = [
        [tuple(pair) for pair in level] for level in recorded_squeezed
    ]
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
    if not _same_storage_encoding(source_layout, copy_layout):
        return None, "source and staged copy use different storage encodings"

    try:
        source_stick = device_coordinates(
            source_layout.device_layout, source_dep, None
        )[-1]
    except Exception as exc:
        return None, f"source layout is not directly readable: {exc}"
    if isinstance(consumer.data, Pointwise):
        try:
            copy_stick = device_coordinates(copy_layout.device_layout, copy_dep, None)[
                -1
            ]
        except Exception as exc:
            return None, f"staged copy layout is not directly readable: {exc}"
        if source_stick.free_symbols != copy_stick.free_symbols:
            return None, (
                "direct source changes the pointwise stick variables: "
                f"{source_stick} != {copy_stick}"
            )
    if is_matmul:
        try:
            generated_var = find_matmul_generated_var(
                source_dep, x_dep, output_dep, direct_op
            )
        except Exception as exc:
            return None, f"matmul source role is not directly readable: {exc}"
        if generated_var not in source_stick.free_symbols:
            return None, "source stick dimension is not the matmul output dimension"

    local_bounds = _affine_bounds(source_dep)
    advance_bounds = _loop_advance_bound(
        direct_op, source_dep, resolved_loop_info, direct_source_idx
    )
    if (
        advance_bounds is None
        and isinstance(copy_loop_info, CoarseTileInfo)
        and copy_source_idx is not None
        and (
            record.direct_tiled_dims_per_level is not None
            or record.direct_squeezed_advance_per_level is not None
        )
    ):
        # Rebasing a spliced-loop source removes the induction variable from
        # its load index.  Usually the recorded pre-rebase metadata still
        # describes the step.  A size-one tiled dimension can already have
        # been squeezed out, however, leaving that metadata unable to recover
        # the source stride (for example, one expert per loop trip).  The
        # staging copy was built before that information was lost and carries
        # the authoritative equivalent advance; use it as a proof/codegen
        # fallback rather than declining an otherwise safe direct read.
        tiled_dims[direct_source_idx] = [
            list(level) for level in copy_loop_info.tiled_dims_per_read[copy_source_idx]
        ]
        copy_squeezed = (
            copy_loop_info.squeezed_advance_per_read[copy_source_idx]
            if copy_source_idx < len(copy_loop_info.squeezed_advance_per_read)
            else []
        )
        squeezed[direct_source_idx] = [list(level) for level in copy_squeezed]
        fallback_loop_info = dataclasses.replace(
            current_loop_info,
            tiled_dims_per_read=tiled_dims,
            squeezed_advance_per_read=squeezed,
        )
        fallback_bounds = _loop_advance_bound(
            direct_op, source_dep, fallback_loop_info, direct_source_idx
        )
        if fallback_bounds is not None:
            resolved_loop_info = fallback_loop_info
            direct_op.loop_info = resolved_loop_info  # type: ignore[attr-defined]
            advance_bounds = fallback_bounds
    # A non-contiguous graph-input view can address a larger backing span than
    # its logical numel (for example, an active KV-cache prefix whose head
    # stride still reflects the maximum cache length).  Bound the generated
    # element index against that physical span, including layout.offset,
    # rather than rejecting valid addresses in its inter-head gaps.
    source_numel = _static_int(source_layout.storage_size())
    logger.debug(
        "direct-read bounds for %s <- %s: source_dep=%s tiled=%s "
        "squeezed=%s local=%s advance=%s numel=%s",
        consumer.get_name(),
        record.source_name,
        source_dep,
        resolved_loop_info.tiled_dims_per_read[direct_source_idx],
        resolved_loop_info.squeezed_advance_per_read[direct_source_idx],
        local_bounds,
        advance_bounds,
        source_numel,
    )
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
        direct_op, source_dep, record.source_name
    )
    if not old_ok or not new_ok or not per_core_views_equal(old_view, new_view):
        return None, "output core ownership changed"
    if not copy_ok:
        return None, "staged-copy core ownership is not representable"
    if not source_ok:
        return None, "direct-source core ownership is not representable"
    # Representability alone is insufficient: both views may be legal while
    # assigning different source slices to the same core.  The staging copy
    # is the behavior being replaced, so the direct HBM read must preserve
    # its exact physical core-to-slice map.
    if not _direct_source_ownership_matches(
        consumer, direct_op, copy_dep, source_dep, copy_view, source_view
    ):
        copy_ownership = _logical_split_ownership(consumer, copy_dep)
        source_ownership = _logical_split_ownership(direct_op, source_dep)
        return None, (
            "direct source ownership does not match the staged copy: "
            f"physical={source_view} != {copy_view}; "
            f"logical={source_ownership} != {copy_ownership}"
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


def project_transport_read_copies(graph, division_splits, *, relayout_sources=()):
    """Non-mutating cost view of transport copies removable for every candidate.

    The allocator still plans the original buffers, and the normal late pass
    still validates and performs removal. Pricing a removable staging copy's
    own division is misleading: the executed direct read uses its consumer's
    division instead. Reuse the full address/ownership/loop proof for *every*
    candidate, declining the projection if any candidate or graph check fails.
    No assumption about a preferred split or final LX placement is needed.
    """
    operations = list(graph.operations)
    if not config.read_copy_elision:
        return operations
    graph_output_names = set(graph.get_output_names())
    for consumer in list(operations):
        record = getattr(consumer, "_read_copy_elision_record", None)
        if (
            not isinstance(consumer, ComputedBuffer)
            or not isinstance(record, ReadCopyElisionRecord)
            or not isinstance(consumer.data, Pointwise)
            or set(consumer.data.inner_fn_opcount().used_ops) != {"load"}
        ):
            continue
        candidates = division_splits.get(consumer.get_name(), ())
        # A later input clone or LX relayout can redirect this read and invalidate
        # its saved record. Keep those allocation-dependent cases authoritative.
        if (
            not candidates
            or record.source_name in division_splits
            or record.copy_name in relayout_sources
            or record.copy_name in graph_output_names
        ):
            continue
        copy_op = next(
            (op for op in operations if op.get_name() == record.copy_name), None
        )
        if not isinstance(copy_op, ComputedBuffer):
            continue
        if _copy_readers(operations, record.copy_name) != [consumer]:
            continue
        direct_op = None
        for splits in candidates:
            candidate = copy.copy(consumer)
            commit_iteration_space_ownership(candidate, splits)
            direct_op, _ = _prove_matmul_direct_read(candidate, copy_op, record)
            if direct_op is None:
                break
            if _validate_proposal(operations, consumer, direct_op, copy_op) is not None:
                direct_op = None
                break
        if direct_op is not None:
            operations = [
                direct_op if op is consumer else op
                for op in operations
                if op is not copy_op
            ]
    return operations


def elide_proven_read_copies(graph: GraphLowering) -> None:
    """Remove only copies whose post-allocation direct-read proof succeeds."""
    if not config.read_copy_elision:
        return

    operations = graph.operations
    graph_output_names = set(graph.get_output_names())
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
            logger.debug(
                "read-copy elision declined for %s: copy is absent",
                record.consumer_name,
            )
            continue
        if record.copy_name in graph_output_names:
            logger.debug(
                "read-copy elision declined for %s: copy is a graph output",
                record.consumer_name,
            )
            continue
        if _copy_readers(operations, record.copy_name) != [consumer]:
            logger.debug(
                "read-copy elision declined for %s: copy has other readers",
                record.consumer_name,
            )
            continue

        direct_op, detail = _prove_matmul_direct_read(consumer, copy_op, record)
        if direct_op is None:
            logger.debug(
                "read-copy elision declined for %s: %s",
                record.consumer_name,
                detail,
            )
            continue
        validation_error = _validate_proposal(operations, consumer, direct_op, copy_op)
        if validation_error is not None:
            logger.debug(
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
            reason="read advancing input directly from HBM",
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
        for orphan_name in record.orphaned_copy_names:
            orphan = next(
                (
                    op
                    for op in operations
                    if isinstance(op, ComputedBuffer) and op.get_name() == orphan_name
                ),
                None,
            )
            if (
                orphan is not None
                and orphan_name not in graph_output_names
                and not _copy_readers(operations, orphan_name)
            ):
                graph.removed_buffers.add(orphan_name)
                operations.remove(orphan)
        logger.info(
            "removed read copy %s; %s reads %s directly",
            copy_op.get_name(),
            replacement.get_name(),
            record.source_name,
        )

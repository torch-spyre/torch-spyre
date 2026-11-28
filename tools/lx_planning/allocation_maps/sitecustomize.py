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

"""Capture graph-operation and LX-allocation records from allocator solves."""

from __future__ import annotations
# ruff: noqa: E402, I001, RUF100

import json
import linecache
import math
import os
import threading
import time
import traceback
from pathlib import Path

import regex

_capture_dir_raw = os.environ.get("SPYRE_GRAPH_CAPTURE_DIR")
if not _capture_dir_raw:
    raise RuntimeError("SPYRE_GRAPH_CAPTURE_DIR must name an output directory")
_CAPTURE_DIR = Path(_capture_dir_raw).resolve()
_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()
_STATUS_PATH = _CAPTURE_DIR / f"processes.{os.getpid()}.jsonl"
with _STATUS_PATH.open("a", encoding="utf-8") as stream:
    stream.write(
        json.dumps({"event": "sitecustomize_started", "pid": os.getpid()}) + "\n"
    )

# Import torch before torch-spyre. torch backend autoload resolves torch-spyre.
import torch
from torch._functorch._aot_autograd.logging_utils import (
    get_aot_compilation_context,
    get_aot_graph_name,
)

from torch_spyre._inductor.pass_utils import op_read_writes, op_short_name
from torch_spyre._inductor.scratchpad import allocator as allocator_module
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver

_FRAME_RE = regex.compile(r'File "([^"]+)", line (\d+)')
_ORIGINAL_POST_SOLVE = allocator_module.CoOptimizingAllocator._post_solve
_ORIGINAL_SOLVE = allocator_module.CoOptimizingAllocator._solve
_ORIGINAL_EXTRACT = CpSatLayoutSolver._extract


def _capture_extract(self, solver, bufs):
    result = _ORIGINAL_EXTRACT(self, solver, bufs)
    # The model places boxes in slots (2*i is operation i's write, 2*i+1
    # its optional copy), not in the original buffer's operation indices.
    self._capture_slot_spans = {
        name: tuple(
            bound if isinstance(bound, int) else solver.Value(bound)
            for bound in sb.slot_span
        )
        for name, sb in bufs.items()
        if sb.slot_span is not None
    }
    return result


def _capture_solve(self, solver, graph):
    allocation = _ORIGINAL_SOLVE(self, solver, graph)
    self._capture_slot_spans = (
        getattr(solver, "_capture_slot_spans", {} if not allocation else None)
        if isinstance(solver, CpSatLayoutSolver)
        else None
    )
    return allocation


CpSatLayoutSolver._extract = _capture_extract
allocator_module.CoOptimizingAllocator._solve = _capture_solve


def _node_record(node: torch.fx.Node) -> dict:
    meta = getattr(node, "meta", None) or {}
    frames = _FRAME_RE.findall(meta.get("stack_trace") or "")
    user_frames = [frame for frame in frames if "/torch/" not in frame[0]]
    frame = (user_frames or frames)[-1] if (user_frames or frames) else None
    source = None
    if frame is not None:
        source_file, line_text = frame
        line = int(line_text)
        source = {
            "file": source_file,
            "line": line,
            "code": linecache.getline(source_file, line).strip() or None,
        }
    module_stack = []
    raw_stack = meta.get("nn_module_stack") or {}
    if isinstance(raw_stack, dict):
        for key, entry in raw_stack.items():
            module = entry[1] if len(entry) > 1 else None
            module_stack.append(
                {
                    "node": str(key),
                    "fqn": str(entry[0]) if entry else None,
                    "type": getattr(module, "__qualname__", type(module).__qualname__),
                }
            )
    aten = meta.get("original_aten")
    return {
        "name": node.name,
        "fx_op": node.op,
        "target": str(node.target),
        "original_aten": str(aten) if aten is not None else None,
        "source": source,
        "module_stack": module_stack,
    }


def _origins(graph, operation) -> list[dict]:
    nodes = []
    for carrier in (operation, getattr(operation, "data", None)):
        if carrier is None:
            continue
        nodes.extend(
            node
            for node in getattr(carrier, "origins", ()) or ()
            if isinstance(node, torch.fx.Node)
        )
        origin = getattr(carrier, "origin_node", None)
        if isinstance(origin, torch.fx.Node):
            nodes.append(origin)

    try:
        buffer_name = operation.get_name()
    except (AttributeError, RuntimeError):
        buffer_name = getattr(operation, "name", None)
    if buffer_name:
        for node, tensor in graph.env.items():
            if not isinstance(node, torch.fx.Node):
                continue
            try:
                if tensor.get_name() == buffer_name:
                    nodes.append(node)
            except (AttributeError, RuntimeError, TypeError):
                continue
    unique = {node.name: node for node in nodes}
    return [_node_record(unique[name]) for name in sorted(unique)]


def _optional_value(obj, attribute):
    try:
        return getattr(obj, attribute, None)
    except Exception:  # noqa: BLE001 - optional compiler metadata is best effort
        # Capture must not turn a missing/deferred compiler layout into a failure.
        return None


def _dimensions(values):
    if values is None:
        return None
    try:
        result = []
        for value in values:
            if isinstance(value, int) or (
                getattr(value, "is_number", False)
                and getattr(value, "is_integer", False)
            ):
                result.append(int(value))
            else:
                result.append(str(value))
        return result
    except Exception:  # noqa: BLE001 - symbolic dimensions may not resolve
        return None


def _spyre_layout(layout):
    if layout is None:
        return dict.fromkeys(
            (
                "device_size",
                "stride_map",
                "device_dtype",
                "element_arrangement",
                "valid_elements",
            )
        )
    valid = _optional_value(layout, "valid_elements")
    try:
        valid_elements = (
            [
                {"device_dims": list(dims), "count": int(count)}
                for dims, count in sorted(valid.items())
            ]
            if valid is not None
            else None
        )
    except Exception:  # noqa: BLE001 - optional layout metadata is best effort
        valid_elements = None
    dtype = _optional_value(layout, "device_dtype")
    arrangement = _optional_value(layout, "element_arrangement")
    return {
        "device_size": _dimensions(_optional_value(layout, "device_size")),
        "stride_map": _dimensions(_optional_value(layout, "stride_map")),
        "device_dtype": _optional_value(dtype, "name"),
        "element_arrangement": _optional_value(arrangement, "name"),
        "valid_elements": valid_elements,
    }


def _lx_view(view):
    if view is None:
        return None
    try:
        return {
            "work_slice_dims": [list(pair) for pair in view.work_slice_dims],
            "core_to_slot": [
                {"device_dim": dim, "slot": str(slot)}
                for dim, slot in view.core_to_slot
            ],
            "num_cores": view.num_cores,
        }
    except Exception:  # noqa: BLE001 - optional LX view is best effort
        return None


def _tensor_metadata(graph, name):
    try:
        tensor = graph.try_get_buffer(name)
        layout = tensor.get_layout() if tensor is not None else None
    except Exception:  # noqa: BLE001 - absent/deferred layout is allowed
        layout = None
    shape = _dimensions(_optional_value(layout, "size"))
    strides = _dimensions(_optional_value(layout, "stride"))
    dtype = _optional_value(layout, "dtype")
    dtype_name = str(dtype) if dtype is not None else None
    device_layout = _optional_value(layout, "device_layout")
    physical = _spyre_layout(device_layout)
    if (
        shape is None
        and strides is None
        and dtype_name is None
        and device_layout is None
    ):
        status = "unavailable"
    elif (
        shape is not None
        and strides is not None
        and dtype_name is not None
        and all(value is not None for value in physical.values())
    ):
        status = "complete"
    else:
        status = "partial"
    try:
        logical_bytes = (
            math.prod(shape) * dtype.itemsize
            if shape is not None
            and all(type(dim) is int and dim >= 0 for dim in shape)
            and dtype is not None
            else None
        )
    except Exception:  # noqa: BLE001 - logical footprint can be symbolic
        logical_bytes = None
    return {
        "host_shape": shape,
        "host_strides": strides,
        "host_dtype": dtype_name,
        "spyre_tensor_layout": physical,
        "logical_tensor_bytes": logical_bytes,
        "metadata_status": status,
    }


def _graph_edges(graph):
    producers = {}
    consumers = {}
    readable = True
    for index, op in enumerate(graph.operations):
        name = _optional_value(op, "name")
        if name is not None:
            producers[name] = index
        try:
            for dep in op_read_writes(op).reads:
                if dep.name is not None:
                    consumers.setdefault(dep.name, set()).add(index)
        except Exception:  # noqa: BLE001 - partial graphs have unknown readers
            # A partial read graph cannot prove a buffer has no consumers.
            readable = False
    return producers, consumers, readable


def _capture_pre_commit(allocator, graph, allocation, accepted_lx_relayouts):
    try:
        aot_path, model_name, nth_graph = get_aot_compilation_context()
        aot_graph_name = get_aot_graph_name()
    except (AttributeError, RuntimeError):
        aot_path, model_name, nth_graph, aot_graph_name = [], None, None, None

    operations = []
    for index, operation in enumerate(graph.operations):
        try:
            name = operation.get_name()
        except (AttributeError, RuntimeError):
            name = getattr(operation, "name", None)
        try:
            operation_name = operation.get_operation_name()
        except (AttributeError, RuntimeError):
            operation_name = None
        operations.append(
            {
                "op_index": index,
                "buffer_name": name,
                "operation_name": operation_name,
                "op_type": op_short_name(operation),
                "class": type(operation).__name__,
                "origins": _origins(graph, operation),
            }
        )
    producers, consumers, readable = _graph_edges(graph)
    graph_input_names = set(graph.graph_input_names)
    plans_by_destination = {
        plan.destination_name: plan for plan in accepted_lx_relayouts
    }
    # CP-SAT boxes use solver slots: operation i writes at 2*i and a cut's
    # optional copy executes at 2*i+1. These are not buffer.uses indices.
    spans = allocator._capture_slot_spans
    if spans is None or set(spans) != {buffer.name for buffer in allocation}:
        raise ValueError("CP-SAT solved slot spans unavailable for capture")

    buffers = []
    resident = []
    for buffer in allocation:
        division = getattr(buffer, "chosen_division", None)
        reserved_size = (
            buffer.per_core_size(division) if division is not None else buffer.size
        )
        synthetic = buffer.name.startswith("__spyre_lx_relayout__")
        candidates = getattr(buffer, "candidates", ())
        source_name = getattr(buffer, "relayout_parent", None)
        plan = plans_by_destination.get(buffer.name)
        if source_name is None and synthetic and not candidates:
            # Fixed-division destination names carry their source, even when
            # the plan was rejected after solving.
            parts = buffer.name.split(":", 2)
            source_name = parts[1] if len(parts) == 3 else None
        if plan is None and source_name and candidates:
            plan = next(
                (
                    accepted
                    for accepted in accepted_lx_relayouts
                    if accepted.source_name == source_name
                    and accepted.destination_view == candidates[0].destination_view
                    and accepted.destination_address == buffer.address
                ),
                None,
            )
        if plan is not None:
            source_name = plan.source_name
        source_metadata = _tensor_metadata(graph, source_name) if source_name else None
        if synthetic:
            # The source layout is NOT the physical layout of the LX copy.
            metadata = {
                "host_shape": None,
                "host_strides": None,
                "host_dtype": None,
                "spyre_tensor_layout": _spyre_layout(None),
                "logical_tensor_bytes": None,
                "metadata_status": "synthetic_relayout_destination_unknown",
            }
            consumer_indices = (
                sorted({producers[name] for name in plan.consumer_names})
                if plan is not None
                and all(name in producers for name in plan.consumer_names)
                else None
            )
            producer_index = None  # Shuffle is at a transfer tick, not a graph op.
            lx_view = (
                plan.destination_view
                if plan is not None
                else getattr(buffer, "lx_view", None)
            )
        else:
            metadata = _tensor_metadata(graph, buffer.name)
            producer_index = (
                None if buffer.name in graph_input_names else producers.get(buffer.name)
            )
            # Graph readers remain graph readers even when the planner
            # substitutes an LX relayout copy at separate transfer ticks.
            consumer_indices = (
                sorted(consumers.get(buffer.name, ())) if readable else None
            )
            lx_view = getattr(buffer, "lx_view", None)
            if lx_view is None:
                lx_view = next(
                    (
                        item.source_view
                        for item in accepted_lx_relayouts
                        if item.source_name == buffer.name
                    ),
                    None,
                )
        record = {
            "name": buffer.name,
            # Legacy v2 name: this is solver allocation footprint, NOT logical bytes.
            "logical_size_bytes": buffer.size,
            "allocated_size_bytes": buffer.size,
            "reserved_size_bytes": reserved_size,
            "start_time": spans[buffer.name][0],
            "end_time": spans[buffer.name][1],
            "uses": [2 * tick for tick in buffer.uses],
            "address_bytes": buffer.address,
            "chosen_division": division,
            "chosen_division_label": (
                buffer.core_divisions[division].label if division is not None else None
            ),
            "spill_reason": buffer.spill_reason,
            "residency_reason": buffer.residency_reason,
            "is_synthetic_relayout": synthetic,
            **metadata,
            "producer_op_index": producer_index,
            "consumer_op_indices": consumer_indices,
            "source_name": source_name,
            "source_metadata": source_metadata,
            "source": (
                next(
                    (
                        origin["source"]
                        for origin in operations[producer_index]["origins"]
                        if origin["source"] is not None
                    ),
                    None,
                )
                if producer_index is not None
                else None
            ),
            "lx_view": _lx_view(lx_view),
        }
        buffers.append(record)
        if buffer.address is not None:
            resident.append(record)

    timeline = []
    for operation in operations:
        index = 2 * operation["op_index"]
        active = [
            buffer
            for buffer in resident
            if buffer["start_time"] <= index < buffer["end_time"]
        ]
        watermark = max(
            (
                buffer["address_bytes"] + buffer["reserved_size_bytes"]
                for buffer in active
            ),
            default=0,
        )
        timeline.append(
            {
                **operation,
                "allocation_tick": index,
                "watermark_bytes": watermark,
                "capacity_bytes": allocator.size,
                "active_resident_buffers": [buffer["name"] for buffer in active],
                "buffers_at_watermark": [
                    buffer["name"]
                    for buffer in active
                    if buffer["address_bytes"] + buffer["reserved_size_bytes"]
                    == watermark
                ]
                if watermark
                else [],
            }
        )

    global_watermark = max(
        (
            buffer["address_bytes"] + buffer["reserved_size_bytes"]
            for buffer in resident
        ),
        default=0,
    )
    record = {
        "schema": "spyre-lx-allocation-graph-trace-v2",
        "metadata_capture_phase": "planner_pre_commit",
        "allocation_time_unit": "solver_slot",
        "pid": os.getpid(),
        "timestamp_unix": time.time(),
        "aot_graph_name": aot_graph_name,
        "aot_graph_path": list(aot_path) if aot_path is not None else None,
        "aot_model_name": model_name,
        "aot_nth_graph": nth_graph,
        "inductor_graph_name": getattr(graph, "name", None),
        "inductor_graph_id": getattr(graph, "graph_id", None),
        "inductor_post_grad_graph_id": getattr(graph, "post_grad_graph_id", None),
        "allocator": type(allocator).__name__,
        "solver": (
            allocator.layout_planning.__name__
            if allocator.layout_planning is not None
            else None
        ),
        "capacity_bytes": allocator.size,
        "global_watermark_bytes": global_watermark,
        "max_operation_watermark_bytes": max(
            (row["watermark_bytes"] for row in timeline), default=0
        ),
        "timeline_matches_global_watermark": (
            max((row["watermark_bytes"] for row in timeline), default=0)
            == global_watermark
        ),
        "all_resident_spans_within_capacity": all(
            buffer["address_bytes"] + buffer["reserved_size_bytes"] <= allocator.size
            for buffer in resident
        ),
        "operation_count": len(operations),
        "buffer_count": len(buffers),
        "resident_count": len(resident),
        "accepted_lx_relayouts": [repr(plan) for plan in accepted_lx_relayouts],
        "operations": operations,
        "timeline": timeline,
        "buffers": buffers,
    }
    path = _CAPTURE_DIR / f"allocations.{os.getpid()}.jsonl"
    with _LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")
        stream.flush()


def _capture_post_solve(allocator, graph, allocation, accepted_lx_relayouts):
    try:
        _capture_pre_commit(allocator, graph, allocation, accepted_lx_relayouts)
    except Exception:  # noqa: BLE001 - log capture errors; never break compilation
        # Inspection is best effort, but an omitted record must be visible.
        with _LOCK, _STATUS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "capture_failed",
                        "pid": os.getpid(),
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
    return _ORIGINAL_POST_SOLVE(allocator, graph, allocation, accepted_lx_relayouts)


allocator_module.CoOptimizingAllocator._post_solve = _capture_post_solve

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

"""Report extracted cost features and keep the latest diagnostic totals."""

import os

from .cost_model import explain
from .cost_features import (
    _int as _int,
    _prod_ints as _prod_ints,
    _op_name as _op_name,
    _work_slices as _work_slices,
    _resolved_work_slices as _resolved_work_slices,
    _cores as _cores,
    _replication as _replication,
    _real_layout as _real_layout,
    _mem_of_layout as _mem_of_layout,
    _device_dims as _device_dims,
    _input_traffic as _input_traffic,
    _loop_features as _loop_features,
    _tiled_symbols_per_level as _tiled_symbols_per_level,
    _dep_index as _dep_index,
    _level_loop_vars as _level_loop_vars,
    _stamped_advances as _stamped_advances,
    _loop_var_advances as _loop_var_advances,
    _loop_factor_for_index as _loop_factor_for_index,
    _tiled_flag as _tiled_flag,
    _candidate_tiling as _candidate_tiling,
    _tile_count as _tile_count,
    _per_tile_elems as _per_tile_elems,
    _row_split as _row_split,
    _contiguous_device_run as _contiguous_device_run,
    _transport_read_geometry as _transport_read_geometry,
    _split_stick_axes as _split_stick_axes,
    _operand_device_run as _operand_device_run,
    _operand_read_geometry as _operand_read_geometry,
    _read_run_bytes as _read_run_bytes,
    _matmul_features as _matmul_features,
    _hbm_pattern as _hbm_pattern,
    _per_core_run as _per_core_run,
    governing_run_split as governing_run_split,
    relayout_cost_features as relayout_cost_features,
    _relayout_features as _relayout_features,
    _graph_boundary_names as _graph_boundary_names,
    _writes_graph_output as _writes_graph_output,
    _stored_elems as _stored_elems,
    _unit_stride_stick_var as _unit_stride_stick_var,
    _indirect_write_elems as _indirect_write_elems,
    _relayout_logger as _relayout_logger,
    _has_partitioning_candidate as _has_partitioning_candidate,
    extract_op_features as extract_op_features,
    extract_features as extract_features,
)


def cost_dump_enabled() -> bool:
    return os.environ.get("SPYRE_DUMP_COST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Totals + per-arg detail from the most recent extraction, using the DEVICE-layout
# byte accounting. Tools (e.g. examples/profile_ops.py) read this to get the model's
# I/O size and verify BW = hbm_bytes / kernel_time, without re-parsing the printed dump.
# LAST_FEATS holds the raw OpFeatures so a tool can call cost_model.predict_ops() to get
# the model's estimated kernel time.
LAST_IO: dict = {}
LAST_FEATS: list = []


def _record_last_io(feats: list) -> None:
    global LAST_IO, LAST_FEATS
    LAST_FEATS = list(feats)
    ops = []
    for o in feats:
        args = []
        for a in o.args:
            bs = a.elems * o.dtype_bytes
            # Same accounting as ``hbm_bytes()`` below, so the per-arg breakdown sums
            # to the total: own size x loop_factor for an HBM arg (L for a per-tile
            # accumulator re-accessed each loop iteration, 1 otherwise), the small
            # one-load size for a broadcast operand, ~free for LX -- except a graph
            # output's write, which stays charged despite LX, and the clone-in load of
            # a resident graph input whose clone this bundle pays for.
            counted = (a.hbm_elems() + a.clone_in_elems()) * o.dtype_bytes
            args.append(
                {
                    "name": a.name,
                    "role": a.role,
                    "mem": a.mem,
                    "dims": list(a.dims) if a.dims else [a.elems],
                    "logical": list(a.logical),
                    "elems": a.elems,
                    "loop_factor": a.loop_factor,
                    "bytes": bs,
                    "hbm_counted": counted,
                    "broadcast": a.broadcast,
                }
            )
        ops.append({"name": o.name, "is_reduction": o.is_reduction, "args": args})
    LAST_IO = {
        "hbm_bytes": sum(o.hbm_bytes() for o in feats),
        "lx_bytes": sum(o.lx_bytes() for o in feats),
        "ops": ops,
    }


def dump_cost_model(operations: list) -> None:
    """Print per-op cost features + predicted latency; no-op unless SPYRE_DUMP_COST.

    Treats the whole op list as one bundle (matching full fusion, e.g. softmax);
    for single-op example programs this is just that op.
    """
    if not cost_dump_enabled():
        return
    from .dump_common import banner, emit

    try:
        feats = extract_features(operations)
        _record_last_io(feats)
        bar = banner("Cost model features + prediction (after pre-scheduling)")
        emit(f"{bar}\n{explain(feats)}\n")
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        emit(f"[SPYRE_DUMP_COST] failed: {exc!r}")

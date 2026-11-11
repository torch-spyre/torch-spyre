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

"""Per-kernel cost sidecar (``TORCH_SPYRE_KERNEL_COST_DIR``).

Joins each compiled kernel's provenance key to the cost model's prediction for the
group of ops it was built from, as one JSON line in ``kernel_cost.<pid>.jsonl``.
Everything is read from the ``CostReport`` of the graph being compiled; nothing is
re-priced. A kernel replayed from the FX graph cache has no report and gets
``"cost": null``.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from typing import Any

from . import config, cost_model_pass
from .cost_model import CostParams
from .kernel_provenance import KernelProvenanceDescriptor
from .logging_utils import get_inductor_logger
from .op_spec import LoopSpec, OpSpec

logger = get_inductor_logger("kernel_cost")

_lock = threading.Lock()
_written: set[tuple[int, str]] = set()


def _buffers(specs: Sequence[Any]) -> set[str]:
    names: set[str] = set()
    for spec in specs:
        if isinstance(spec, LoopSpec):
            names |= _buffers(spec.body)
        elif (
            isinstance(spec, OpSpec)
            and spec.debug_handle is not None
            and spec.debug_handle.ir_chain
        ):
            names.add(spec.debug_handle.ir_chain[-1])
    return names


def _cost(buffers: set[str]) -> dict | None:
    report = cost_model_pass.LAST_REPORT
    groups = [g for g in report.groups if buffers & set(g.buffers)] if report else []
    if not groups:
        return None
    params = CostParams()
    return {
        "predicted_us": sum(g.predicted_us for g in groups),
        "hbm_read_bytes": sum(g.hbm_read_bytes for g in groups),
        "hbm_write_bytes": sum(g.hbm_write_bytes for g in groups),
        "lx_bytes": sum(o.lx_bytes for g in groups for o in g.ops),
        "macs": sum(g.macs for g in groups),
        "cores": max(g.cores for g in groups),
        "groups": [g.index for g in groups],
        "floor_inputs": {
            "bw_peak_gbps": params.bw_peak_gbps,
            "mac_peak_per_core_ns": params.mac_peak_per_core_ns,
            "fill_ns": params.fill_ns,
        },
    }


def record_kernel_cost(
    kernel_name: str,
    specs: Sequence[Any],
    provenance: KernelProvenanceDescriptor | None,
) -> None:
    """Append this kernel's record once per process. Never raises."""
    out_dir = config.kernel_cost_dir
    if not out_dir or provenance is None:
        return
    pid = os.getpid()
    try:
        with _lock:
            if (pid, provenance.key) in _written:
                return
            _written.add((pid, provenance.key))
        record = {
            "v": 1,
            "provenance_key": provenance.key,
            "kernel": kernel_name,
            "aten_ops": list(provenance.aten_ops),
            "cost": _cost(_buffers(specs)),
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"kernel_cost.{pid}.jsonl"), "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001 - instrumentation must not fail a compile
        logger.warning("kernel cost record for %s skipped", kernel_name, exc_info=True)

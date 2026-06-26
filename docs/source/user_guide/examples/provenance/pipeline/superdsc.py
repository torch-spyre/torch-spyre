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
"""
pipeline/superdsc.py  —  Stage 6 (SuperDSC JSON) reader.

Reads the emitted ``sdsc_*.json`` bundles from the *exact* per-kernel output
directories captured in-process at compile time (captures.stage6_kernels.
output_dirs), keyed by ``kernel_name``. No /tmp globbing, no mtime guessing,
no iteration-space heuristics — the kernel→dir map is recorded by the compiler
itself ([async_compile.get_output_dir]).

For each kernel we report how many ``sdsc_*.json`` files (one per OpSpec) it
emitted and whether any provenance field appears anywhere in the serialized
JSON. Confirmed absence is the Stage-6 finding.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

# Provenance field names we scan the serialized SDSC JSON for (issue #2574).
# The first six are PyTorch's real FX node.meta / Inductor IR provenance keys;
# "provenance" and "debug_handle" are forward-looking names for what Phase 2
# (#2575) will thread through OpSpec/SDSC (debug_handle mirrors upstream's
# _inductor_kernel_provenance_debug_handle).
PROVENANCE_FIELD_NAMES = [
    "stack_trace",  # FX meta
    "original_aten",  # FX meta
    "from_node",  # IR attribute
    "origins",  # IR attribute
    "origin_node",  # IR attribute
    "traceback",  # FX meta
    "provenance",  # forward-looking
    "debug_handle",  # forward-looking
]


def _is_populated(val: Any) -> bool:
    """A provenance value counts as carried only if non-empty: ``0`` is
    populated (a valid handle); ``None`` / empty collection / ``""`` are not.
    Matches the population rule used at every other stage."""
    if val is None:
        return False
    if isinstance(val, (list, tuple, set, dict, str)) and len(val) == 0:
        return False
    return True


def _scan_json_for_provenance(obj: Any) -> dict[str, bool]:
    # Population, not mere key presence: a key whose value is null/empty does
    # NOT count (a serialized "debug_handle": null is not carried provenance).
    found = {name: False for name in PROVENANCE_FIELD_NAMES}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in found and _is_populated(v):
                    found[k] = True
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return found


def read_bundle(kernel_name: str, output_dir: str) -> dict[str, Any]:
    """Read all sdsc_*.json under one kernel's output_dir."""
    d = pathlib.Path(output_dir)
    rec: dict[str, Any] = {
        "kernel_name": kernel_name,
        "output_dir": output_dir,
        "exists": d.is_dir(),
        "sdsc_files": [],
        "provenance_present": False,
    }
    if not d.is_dir():
        return rec

    for jf in sorted(d.glob("sdsc_*.json")):
        try:
            text = jf.read_text()
            data = json.loads(text)  # validate
        except Exception as e:
            rec["sdsc_files"].append({"file": jf.name, "parse_error": str(e)})
            continue
        hits = _scan_json_for_provenance(data)
        rec["sdsc_files"].append(
            {
                "file": jf.name,
                "provenance_fields": {k: v for k, v in hits.items() if v},
                "has_provenance": any(hits.values()),
            }
        )

    rec["provenance_present"] = any(f.get("has_provenance") for f in rec["sdsc_files"])
    return rec


def run(output_dirs: dict[str, str]) -> dict[str, Any]:
    """Read every captured kernel bundle.

    Args:
        output_dirs: kernel_name -> exact output directory, as captured by the
            in-process hook on async_compile.get_output_dir during this run.

    Returns:
        {
          "kernels": [read_bundle(...), ...],   # one per kernel_name
          "total_kernels": int,
          "total_sdsc_files": int,
          "provenance_present": bool,            # any field in any bundle
        }
    """
    kernels = [read_bundle(name, d) for name, d in sorted(output_dirs.items())]
    total_files = sum(len(k["sdsc_files"]) for k in kernels)
    return {
        "kernels": kernels,
        "total_kernels": len(kernels),
        "total_sdsc_files": total_files,
        "provenance_present": any(k["provenance_present"] for k in kernels),
    }

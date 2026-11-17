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
pipeline/fields.py — single source of truth for the provenance field names and
the shared helpers that key off them.

Both the capture layer (captures.py) and the report layer (report.py) use these,
so they are defined once here to prevent drift.
"""

from __future__ import annotations

from typing import Any

# FX ``node.meta`` provenance fields (issue #2574), in the order the issue
# lists them.
FX_FIELDS = [
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "original_aten",
    "from_node",
]

# Inductor IR / LoopLevelIR provenance attributes stored on the
# ``ComputedBuffer`` (what the capture reads directly).
IR_ATTR_FIELDS = ["origins", "origin_node", "traceback"]

# IR provenance shown as matrix columns: the stored attributes plus the derived
# ``get_stack_traces()`` accessor (not a stored attribute, so it is separate
# from IR_ATTR_FIELDS).
IR_FIELDS = IR_ATTR_FIELDS + ["get_stack_traces"]

# Provenance fields an ``OpSpec`` instance may carry. Phase 2a (#2575) adds
# ``debug_handle``; the IR attrs / accessor are checked too so the OpSpec column
# can show partial population once a field is declared but not on every op.
OPSPEC_PROVENANCE_FIELDS = IR_ATTR_FIELDS + ["get_stack_traces", "debug_handle"]


def is_populated(val: Any) -> bool:
    """The shared population rule: a value counts as carried only if non-empty.
    ``0`` is populated (a valid handle/id); ``None``, an empty collection, or
    ``""`` are not. Used at every capture/report stage.
    """
    if val is None:
        return False
    if isinstance(val, (list, tuple, set, dict, str)) and len(val) == 0:
        return False
    return True


def source_loc_str(src: Any) -> str | None:
    """Render a structured ``SourceLoc`` dict (``{file, start_line, end_line?}``)
    as ``basename:line`` (or ``basename:line-end`` for a range). Returns None
    when there is no resolvable source (``src`` is not a dict, or has no file /
    start line). Shared by the capture (Stage 5) and report (Stage 6) layers so
    their source columns stay in sync.
    """
    if not isinstance(src, dict):
        return None
    file = src.get("file")
    start = src.get("start_line")
    if not file or start is None:
        return None
    base = str(file).rsplit("/", 1)[-1]
    end = src.get("end_line")
    return f"{base}:{start}-{end}" if end and end != start else f"{base}:{start}"

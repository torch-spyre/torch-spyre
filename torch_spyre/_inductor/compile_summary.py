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

"""Per-process compile summary (``TORCH_SPYRE_COMPILE_SUMMARY_DIR``).

Written to ``compile_summary.<pid>.json`` at exit, or by ``dump()``::

    {"total_s": ..., "passes": {"<Pipeline>:<pass>": self_s}, "cache": {...}}

Derived from the timing recorder's events and the kernel cache registry.
"""

from __future__ import annotations

import atexit
import json
import os
from typing import Optional

from . import config
from .logging_utils import get_inductor_logger
from .timing_recorder import RECORDER

logger = get_inductor_logger("compile_summary")

_COMPILE = "stage:compile_fx:spyre_compile"
_PASS = "pass:"


def build() -> dict:
    from torch_spyre.execution.kernel_cache import get_kernel_registry

    RECORDER.finalize()
    events = [e for e in RECORDER.events if e.is_closed]
    by_ordinal = {e.ordinal: e for e in events}

    def nested_compile(event) -> bool:
        parent = event.parent_ordinal
        while parent is not None:
            ancestor = by_ordinal.get(parent)
            if ancestor is None:
                return False
            if ancestor.name == _COMPILE:
                return True
            parent = ancestor.parent_ordinal
        return False

    passes: dict[str, float] = {}
    for e in events:
        if e.name.startswith(_PASS):
            key = e.name[len(_PASS) :]
            passes[key] = passes.get(key, 0.0) + e.self_ns / 1e9
    entries = get_kernel_registry().all_entries().values()
    return {
        "total_s": sum(
            e.inclusive_ns
            for e in events
            if e.name == _COMPILE and not nested_compile(e)
        )
        / 1e9,
        "passes": passes,
        "cache": {
            "entries": len(entries),
            "hits": sum(m["hit_count"] for m in entries),
            "misses": sum(m["miss_count"] for m in entries),
        },
    }


def dump(directory: Optional[str] = None) -> Optional[str]:
    """Write this process's summary. Returns the path, or None when unconfigured."""
    directory = directory or config.compile_summary_dir
    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"compile_summary.{os.getpid()}.json")
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(build(), f, indent=1)
    os.replace(tmp, path)
    return path


@atexit.register
def _dump_at_exit() -> None:
    try:
        dump()
    except Exception as exc:  # noqa: BLE001 - a summary must not fail the run
        logger.warning("compile summary not written: %s: %s", type(exc).__name__, exc)

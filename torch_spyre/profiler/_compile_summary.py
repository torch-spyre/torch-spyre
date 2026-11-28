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

Written to ``compile_summary.<pid>.json`` at exit, or by :func:`dump`::

    {"v": 1, "total_s": ..., "passes": {"<Pipeline>:<pass>": self_s},
     "cache": {"entries": ..., "hits": ..., "misses": ...}}

``total_s`` and ``passes`` come from the frontend timing recorder, so they need
``TORCH_SPYRE_TIMING=1`` as well; :func:`dump` says so when the recorder is off.
``cache`` comes from the kernel cache registry and is always populated.

Off by default: unconfigured, this module reads one environment variable at
import and registers nothing. Both sources are read, never driven, so nothing
here changes what a compile does.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
from typing import Optional

_log = logging.getLogger(__name__)

DIR_VAR = "TORCH_SPYRE_COMPILE_SUMMARY_DIR"
TIMING_VAR = "TORCH_SPYRE_TIMING"

# The timing recorder's name for one outermost Spyre compile, and the prefix every
# per-pass region carries (see torch_spyre/_inductor/timing_recorder.py).
_COMPILE = "stage:compile_fx:spyre_compile"
_PASS = "pass:"


def configured_dir() -> str:
    """The destination directory, or "" when the sidecar is off."""
    return os.environ.get(DIR_VAR, "")


def build() -> dict:
    """This process's summary. Reads the timing recorder and the kernel registry."""
    # Imported here, not at module scope: this module is imported eagerly with
    # torch_spyre, and the compiler must not be pulled in behind it.
    from torch_spyre._inductor.timing_recorder import RECORDER
    from torch_spyre.execution.kernel_cache import get_kernel_registry

    RECORDER.finalize()
    events = [e for e in RECORDER.events if e.is_closed]
    by_ordinal = {e.ordinal: e for e in events}

    def nested_compile(event) -> bool:
        """Whether a compile region sits under another, so only the outermost counts."""
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
        "v": 1,
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
    directory = directory or configured_dir()
    if not directory:
        return None
    _warn_if_untimed()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"compile_summary.{os.getpid()}.json")
    # Written through a temporary file: a reader globbing the directory while the
    # process exits must never see a half-written summary.
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(build(), f, indent=1)
    os.replace(tmp, path)
    return path


def _warn_if_untimed() -> None:
    """Say why a summary carries no compile time, rather than writing zeros silently."""
    from torch_spyre._inductor import config

    if not config.timing:
        _log.warning(
            "%s is set but the timing recorder is off, so total_s and passes will be "
            "empty; set %s=1 as well. Cache counters are unaffected.",
            DIR_VAR,
            TIMING_VAR,
        )


def _dump_at_exit() -> None:
    try:
        dump()
    except Exception as exc:  # noqa: BLE001 - a summary must not fail the run
        _log.warning("compile summary not written: %s: %s", type(exc).__name__, exc)


# Registered only when configured, so an ordinary process carries no exit hook.
if configured_dir():
    atexit.register(_dump_at_exit)

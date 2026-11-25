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

"""Run context beside a vLLM bench's results, as benchmark_runs.props in the engine's vocabulary.

`env.json`, `phases.json`, `host_resources.json`, `outputs.json` and `compile_summary.json`
are read from `<results>/<test_name>/<kind>.json`, else `<results>/<test_name>.<kind>.json`, else a
shared `<results>/<kind>.json`. `repeats/` is never read: only the first run is ingested.
Each is optional and a malformed one is skipped: they describe a run, they never decide
whether it is ingested.

Keys (every one passes through v_bench_attrs unchanged):
  env.json          run.node, run.host.*, run.env.<VAR>, subject.image, subject.artifact_id12
  phases.json       run.phase.<name>_s
  host_resources    run.res.<phase>.<scalar>
  outputs.json      run.output_digest, run.output_digest.<k>, run.output_greedy
  compile_summary   run.compile.total_s, run.compile.cache_{entries,hits,misses}
`env.fp.*` and `env.label.*` are not written: they come from environment_fingerprints, whose env_id
only the engine's probe derives, so a sidecar cannot claim one.
"""

import hashlib
import json
import logging
import math
import os
import regex as re
from typing import Any

from .offline import is_sidecar  # noqa: F401

log = logging.getLogger(__name__)

_MAX_KNOBS = 40
_MAX_RES = 200
_MAX_DIGESTS = 16
# Over-long keys and values are skipped, not truncated, so a stored value is never a prefix of the real one.
_MAX_KEY = 128
_MAX_VALUE = 512
_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9_]+")
_MALFORMED = (
    TypeError,
    ValueError,
    OverflowError,
    AttributeError,
    RecursionError,
    MemoryError,
)


def _load(results_dir: str, test_name: str, kind: str) -> dict[str, Any]:
    if not test_name or os.path.isabs(test_name) or ".." in test_name.split("/"):
        return {}
    # <test_name>.host_resources.json is already the sampler's per-record file.
    names = [os.path.join(test_name, f"{kind}.json"), f"{kind}.json"]
    if kind != "host_resources":
        names.insert(1, f"{test_name}.{kind}.json")
    for name in names:
        path = os.path.join(results_dir, name)
        if not os.path.isfile(path):
            continue
        # A symlink (file or directory) that leaves the results dir is never read.
        root = os.path.realpath(results_dir)
        if os.path.commonpath([root, os.path.realpath(path)]) != root:
            log.warning("%s leaves the results dir — skipped", name)
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError, RecursionError):
            log.warning("Unreadable %s — skipped", name)
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _put(out: dict[str, str], key: str, value: Any) -> None:
    if value is None or isinstance(value, (dict, list)):
        return
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return
    if str(value) != "" and len(key) <= _MAX_KEY and len(str(value)) <= _MAX_VALUE:
        out[key] = str(value)


def _num(v: Any) -> float | None:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    return float(v) if math.isfinite(v) else None


def _name(v: Any) -> str:
    return _KEY_UNSAFE.sub("_", str(v))


def _env(d: dict[str, Any], out: dict[str, str]) -> None:
    cpu = d.get("cpu") if isinstance(d.get("cpu"), dict) else {}
    cg = d.get("cgroup") if isinstance(d.get("cgroup"), dict) else {}
    cards = [c for c in d.get("cards") or [] if isinstance(c, dict)]
    for key, value in (
        ("run.node", d.get("host")),
        ("run.host.cpu_model", cpu.get("model")),
        ("run.host.cpu_threads", cpu.get("threads")),
        ("run.host.cgroup_cpu_max", cg.get("cpu_max")),
        ("run.host.cgroup_memory_max", cg.get("memory_max")),
        ("run.host.mem_total_b", d.get("mem_total_b")),
        ("run.host.kernel", d.get("kernel")),
        ("run.host.os", d.get("os")),
        ("run.host.card_kind", d.get("card_kind")),
        ("run.host.card_ids", ",".join(str(c["pci"]) for c in cards if c.get("pci"))),
        ("subject.image", d.get("image")),
    ):
        _put(out, key, value)
    art = d.get("artifact") if isinstance(d.get("artifact"), dict) else {}
    _put(out, "subject.artifact_id12", art.get("id12") or str(art.get("id") or "")[:12])
    knobs = d.get("knobs") if isinstance(d.get("knobs"), dict) else {}
    for k in [k for k in sorted(knobs) if _name(k)][:_MAX_KNOBS]:
        _put(out, "run.env." + _name(k), knobs[k])


def _phases(d: dict[str, Any], out: dict[str, str]) -> None:
    _put(out, "run.phase.server_startup_s", d.get("server_startup_sec"))
    walls: dict[str, float] = {}
    for p in d.get("phases") or []:
        if not isinstance(p, dict) or not isinstance(p.get("name"), str):
            continue
        start, end = _num(p.get("t_start")), _num(p.get("t_end"))
        if start is not None and end is not None and end >= start:
            walls[p["name"]] = walls.get(p["name"], 0.0) + end - start
    for name, wall in walls.items():
        out[f"run.phase.{_name(name)}_s"] = f"{wall:.3f}"


def _host_resources(d: dict[str, Any], out: dict[str, str]) -> None:
    phases = d.get("phases") if isinstance(d.get("phases"), dict) else d
    for name, scalars in phases.items():
        if not isinstance(scalars, dict):
            continue
        for k, v in scalars.items():
            if len(out) >= _MAX_RES:
                return
            if _num(v) is not None:
                _put(out, f"run.res.{_name(name)}.{_name(k)}", v)


def _outputs(d: dict[str, Any], out: dict[str, str]) -> None:
    _put(out, "run.output_greedy", d.get("greedy"))
    hashes = d.get("token_sha256")
    if isinstance(hashes, list) and hashes:
        out["run.output_digest"] = hashlib.sha256(
            "\n".join(map(str, hashes)).encode()
        ).hexdigest()
        for k, h in enumerate(hashes[:_MAX_DIGESTS]):
            _put(out, f"run.output_digest.{k}", h)


def _compile(d: dict[str, Any], out: dict[str, str]) -> None:
    _put(out, "run.compile.total_s", d.get("total_s"))
    cache = d.get("cache") if isinstance(d.get("cache"), dict) else {}
    for k in ("entries", "hits", "misses"):
        _put(out, f"run.compile.cache_{k}", cache.get(k))


_READERS = {
    "env": _env,
    "phases": _phases,
    "host_resources": _host_resources,
    "outputs": _outputs,
    "compile_summary": _compile,
}


def flatten(kind: str, data: dict[str, Any]) -> dict[str, str]:
    """The run props one sidecar `kind` contributes; empty when `data` is malformed."""
    got: dict[str, str] = {}
    try:
        _READERS[kind](data, got)
    except _MALFORMED:
        log.warning("Malformed %s data — skipped", kind)
        return {}
    return got


def props_for(results_dir: str, test_name: str) -> dict[str, str]:
    """The run props of the sidecar files that apply to `test_name`."""
    out: dict[str, str] = {}
    for kind in _READERS:
        data = _load(results_dir, test_name, kind)
        if data:
            out.update(flatten(kind, data))
    return out

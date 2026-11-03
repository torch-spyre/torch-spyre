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

"""Run context beside a vLLM bench's results, flattened to benchmark_runs.props.

`env.json`, `phases.json`, `host_resources.json`, `outputs.json` and `compile_summary.json`
are read from `<results>/<test_name>/<kind>.json`, else `<results>/<test_name>.<kind>.json`, else a
shared `<results>/<kind>.json`. `repeats/` is never read: only the first run is ingested.
Each is optional and a malformed one is skipped: they describe a run, they never decide
whether it is ingested.
"""

import hashlib
import json
import logging
import os
import regex as re
from typing import Any

log = logging.getLogger(__name__)

KINDS = ("env", "phases", "host_resources", "outputs", "compile_summary")
SIDECAR_FILES = frozenset(f"{k}.json" for k in KINDS)
_MAX_KNOBS = 40
_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9_]+")


def is_sidecar(filename: str) -> bool:
    """A `<kind>.json` or `<test_name>.<kind>.json` file; not a benchmark record file."""
    return filename in SIDECAR_FILES or any(
        filename.endswith(f".{k}.json") and k != "host_resources" for k in KINDS
    )


def _load(results_dir: str, test_name: str, kind: str) -> dict[str, Any]:
    # <test_name>.host_resources.json is already the sampler's per-record file.
    names = [os.path.join(test_name, f"{kind}.json"), f"{kind}.json"]
    if kind != "host_resources":
        names.insert(1, f"{test_name}.{kind}.json")
    for name in names:
        path = os.path.join(results_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            log.warning("Unreadable %s — skipped", name)
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _scalar(v: Any) -> str | None:
    if v is None or isinstance(v, (dict, list)):
        return None
    return str(v)


def _put(out: dict[str, str], key: str, value: Any) -> None:
    s = _scalar(value)
    if s not in (None, ""):
        out[key] = s


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _env(d: dict[str, Any], out: dict[str, str]) -> None:
    cpu = d.get("cpu") if isinstance(d.get("cpu"), dict) else {}
    cg = d.get("cgroup") if isinstance(d.get("cgroup"), dict) else {}
    cards = [c for c in d.get("cards") or [] if isinstance(c, dict)]
    for key, value in (
        ("env_host", d.get("host")),
        ("env_arch", d.get("arch")),
        ("env_cpu_model", cpu.get("model")),
        ("env_cpu_threads", cpu.get("threads")),
        ("env_cgroup_cpu_max", cg.get("cpu_max")),
        ("env_cgroup_memory_max", cg.get("memory_max")),
        ("env_mem_total_b", d.get("mem_total_b")),
        ("env_kernel", d.get("kernel")),
        ("env_os", d.get("os")),
        ("env_image", d.get("image")),
        ("env_card_kind", d.get("card_kind")),
        ("env_card_ids", ",".join(str(c["pci"]) for c in cards if c.get("pci"))),
    ):
        _put(out, key, value)
    art = d.get("artifact") if isinstance(d.get("artifact"), dict) else {}
    _put(out, "artifact_id12", art.get("id12") or str(art.get("id") or "")[:12])
    knobs = d.get("knobs") if isinstance(d.get("knobs"), dict) else {}
    for k in sorted(knobs)[:_MAX_KNOBS]:
        _put(out, "env_knob_" + _KEY_UNSAFE.sub("_", k), knobs[k])


def _phases(d: dict[str, Any], out: dict[str, str]) -> None:
    _put(out, "server_startup_sec", d.get("server_startup_sec"))
    walls: dict[str, float] = {}
    for p in d.get("phases") or []:
        if not isinstance(p, dict) or not isinstance(p.get("name"), str):
            continue
        start, end = _num(p.get("t_start")), _num(p.get("t_end"))
        if start is not None and end is not None and end >= start:
            walls[p["name"]] = walls.get(p["name"], 0.0) + end - start
    for name, wall in walls.items():
        out[f"phase_{_KEY_UNSAFE.sub('_', name)}_s"] = f"{wall:.3f}"


def _host_resources(d: dict[str, Any], out: dict[str, str]) -> None:
    phases = d.get("phases") if isinstance(d.get("phases"), dict) else d
    for name, scalars in phases.items():
        if not isinstance(scalars, dict):
            continue
        for k, v in scalars.items():
            if _num(v) is not None:
                out[f"res_{_KEY_UNSAFE.sub('_', name)}_{k}"] = str(v)


def _outputs(d: dict[str, Any], out: dict[str, str]) -> None:
    _put(out, "outputs_prompts", d.get("prompts"))
    _put(out, "outputs_greedy", d.get("greedy"))
    hashes = d.get("token_sha256")
    if isinstance(hashes, list) and hashes:
        out["outputs_token_sha256"] = hashlib.sha256(
            "\n".join(map(str, hashes)).encode()
        ).hexdigest()


def _compile(d: dict[str, Any], out: dict[str, str]) -> None:
    _put(out, "compile_total_s", d.get("total_s"))
    cache = d.get("cache") if isinstance(d.get("cache"), dict) else {}
    for k in ("entries", "hits", "misses"):
        _put(out, f"compile_cache_{k}", cache.get(k))


_READERS = {
    "env": _env,
    "phases": _phases,
    "host_resources": _host_resources,
    "outputs": _outputs,
    "compile_summary": _compile,
}


def flatten(kind: str, data: dict[str, Any]) -> dict[str, str]:
    """The run props one sidecar `kind` contributes, for a caller that already holds its data."""
    out: dict[str, str] = {}
    _READERS[kind](data, out)
    return out


def props_for(results_dir: str, test_name: str) -> dict[str, str]:
    """The flat string props of the sidecar files that apply to `test_name`."""
    out: dict[str, str] = {}
    for kind, reader in _READERS.items():
        data = _load(results_dir, test_name, kind)
        if data:
            reader(data, out)
    return out

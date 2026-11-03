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

"""The perf suite's report_sendnn.txt -> spyre_v2 benchmark_runs (report_kind 'perf-suite-sendnn-txt').

sendnn results only ever arrive in an uploaded bundle: no CI leg runs the sendnn stack. Records take
parse_benchmark_xml's shape and go through the same _bench_entries / BenchmarkWriter, so a sendnn
number lands on the benchmark_id of torch-spyre's own measurement of that op, as backend `sendnn`.

Reads the single-stack report (`Metric sendnn`) and the merged one (`Metric torch-spyre sendnn ...`),
taking the sendnn column either way.
"""

from datetime import datetime
from pathlib import Path

import regex as re

REPORT_KIND = "perf-suite-sendnn-txt"
COMPONENT = "torch-spyre"

# report row -> the measurement key report.xml's ingest gives the same quantity
METRICS = {
    "wall_clock_ms.mean_ms": "total_duration_ms",
    "cpu_ms.mean_ms": "cpu_ms",
    "spyre_ms.mean_ms": "spyre_ms",
    "kernel_ms.mean_ms": "kernel_mean_ms",
    "memory_transfer_ms.mean_ms": "memory_transfer_mean_ms",
    "compile_ms.mean_ms": "compile_ms",
    "compiler_ms.mean_ms": "compile_ms",
    "runtime_ms.mean_ms": "runtime_ms",
    "mem_size.MB": "mem_size_mb",
    "mem_size_MB.mean_ms": "mem_size_mb",
    "pt_util%": "pt_util_percent",
}

_OP_RE = re.compile(r"^Op:\s+(?P<op>\S+)\s+Shape:\s+(?P<shape>.+?)\s*$")
_GRANITE_RE = re.compile(r"^Granite Config:\s+(?P<config>\S+)\s*$")
_GRANITE_CONFIG_RE = re.compile(r"bs(?P<batch_size>\d+)(?:_pl(?P<prompt_length>\d+))?")


def _section(line: str) -> dict | None:
    m = _OP_RE.match(line)
    if m:
        return {"record_type": "op", "operation_name": m["op"], "input_shapes": m["shape"],
                "run_mode": "op_benchmark"}  # fmt: skip
    m = _GRANITE_RE.match(line)
    if m:
        cfg = _GRANITE_CONFIG_RE.search(m["config"])
        return {
            "record_type": "model",
            "operation_name": "granite",
            "config_name": m["config"],
            "batch_size": int(cfg["batch_size"]) if cfg else None,
            "prompt_length": int(cfg["prompt_length"])
            if cfg and cfg["prompt_length"]
            else None,
        }
    return None


def parse(path: Path) -> list:
    """One record per section that has a sendnn value; `metric` selects backend sendnn."""
    records, rec, col = [], None, None
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        found = _section(line)
        if found is not None:
            rec, col = {**found, "metric": "sendnn_ms"}, None
            records.append(rec)
            continue
        tokens = line.split()
        if not tokens or rec is None or line.startswith("-"):
            continue
        if tokens[0] == "Metric":
            col = tokens[1:].index("sendnn") if "sendnn" in tokens[1:] else None
            continue
        key = METRICS.get(tokens[0])
        if key is None or col is None or col + 1 >= len(tokens):
            continue
        try:
            rec[key] = float(tokens[col + 1])
        except ValueError:  # N/A: this stack did not measure it
            pass
    return [r for r in records if any(k in r for k in set(METRICS.values()))]


def run_utc(started_at: str) -> str:
    """bundle.json started_at (ISO-8601) as the 'YYYY-MM-DD HH:MM:SS' UTC the run props carry."""
    from datetime import UTC

    ts = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    return ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def write(client, db: str, run_id: str, path: Path, run_props: dict) -> int:
    """benchmarks + benchmark_runs for one report; nothing when this run already has it."""
    from .results import _bench_entries
    from .writer import BenchmarkWriter

    path = Path(path)
    if BenchmarkWriter.already_ingested(
        client, db, run_id, COMPONENT, REPORT_KIND, path.name
    ):
        return 0
    entries = _bench_entries(parse(path))
    for e in entries:
        e["run_props"] = run_props
    return BenchmarkWriter.insert(
        client, db, COMPONENT, run_id, entries, REPORT_KIND, path.name
    )

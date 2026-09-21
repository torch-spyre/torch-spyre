#!/usr/bin/env python3
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


"""Turn a directory of timing records into a comparable summary.

    python3 tools/frontend_timing/summarize.py /tmp/records
    python3 tools/frontend_timing/summarize.py /tmp/records --passes --csv out.csv

Median across samples, never mean: pod wall time has a long tail, and one contended
sample should not move the number. Frontend time is a subtraction -- the compile region
minus the backend invocations inside it -- because the backend runs per kernel from
within codegen rather than after it.

Records that never reached a compile are reported and excluded rather than averaged in:
a process that died before compiling looks like a fast one otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any

COMPILE_EVENT = "stage:compile_fx:spyre_compile"
BACKEND_SUFFIX = ":backend_compile"
GRAPH_PIPELINE = "pipeline:CustomPreSchedulingPasses"

#: Metadata keys that describe the run rather than the point being measured.
_NON_PARAM_KEYS = frozenset(
    {
        "sample",
        "cold",
        "cache_dir",
        "spyre_config",
        "workload",
        "recorder_version",
        "clock",
        "pid",
        "git_sha",
        "torch_spyre_path",
        "torch_spyre_version",
        "torch_version",
        "python_version",
        "frontend_only",
        "backend_skipped_kernels",
    }
)


@dataclass
class Record:
    """One process's timing record."""

    path: str
    meta: dict[str, Any]
    events: list[dict[str, Any]]

    @property
    def workload(self) -> str:
        return str(self.meta.get("workload", "unknown"))

    @property
    def params(self) -> dict[str, Any]:
        return {k: v for k, v in self.meta.items() if k not in _NON_PARAM_KEYS}

    def total_ns(self, name: str) -> int:
        return sum(e["inclusive_ns"] for e in self.events if e["name"] == name)

    def backend_ns(self) -> int:
        return sum(
            e["inclusive_ns"] for e in self.events if e["name"].endswith(BACKEND_SUFFIX)
        )

    def graph_operations(self) -> int:
        """Largest pre-scheduling graph this process compiled."""
        sizes = [
            e.get("meta", {}).get("input_operations", 0)
            for e in self.events
            if e["name"] == GRAPH_PIPELINE
        ]
        return max(sizes) if sizes else 0

    def by_prefix(self, prefix: str) -> dict[str, int]:
        totals: dict[str, int] = {}
        for event in self.events:
            if event["name"].startswith(prefix):
                totals[event["name"]] = totals.get(event["name"], 0) + int(
                    event["inclusive_ns"]
                )
        return totals


@dataclass
class PointSummary:
    """Medians for one workload at one set of parameters."""

    point: str
    workload: str
    params: dict[str, Any]
    samples: int
    graph_operations: int
    total_ms: float
    backend_ms: float
    buckets_ms: dict[str, float] = field(default_factory=dict)
    passes_ms: dict[str, float] = field(default_factory=dict)

    @property
    def frontend_ms(self) -> float:
        return self.total_ms - self.backend_ms


def point_name(workload: str, params: dict[str, Any]) -> str:
    if not params:
        return workload
    return f"{workload}-" + "_".join(f"{k}{params[k]}" for k in sorted(params))


def load_records(directory: str) -> tuple[list[Record], list[str]]:
    """Load every record in ``directory``. Returns (usable, problems).

    Subdirectories are not walked, which is what keeps the driver's discarded warmup --
    written to ``warmup/`` -- out of the summary.
    """
    records: list[Record] = []
    problems: list[str] = []
    if not os.path.isdir(directory):
        raise SystemExit(f"no such records directory: {directory}")

    for entry in sorted(os.listdir(directory)):
        if not entry.endswith(".json"):
            continue
        path = os.path.join(directory, entry)
        try:
            with open(path) as handle:
                payload = json.load(handle)
            record = Record(
                path=path, meta=payload["meta"], events=payload.get("events", [])
            )
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"{entry}: unreadable ({type(exc).__name__})")
            continue

        if not record.total_ns(COMPILE_EVENT):
            problems.append(f"{entry}: no {COMPILE_EVENT} event, excluded")
            continue
        if any(e.get("open") for e in record.events):
            problems.append(f"{entry}: written mid-region, excluded")
            continue
        failed = [e["name"] for e in record.events if e.get("error")]
        if failed:
            # The recorder dumps at process exit whether or not the compile succeeded,
            # so a failed sample leaves a record whose times measure a failure.
            problems.append(f"{entry}: compile failed in {failed[0]}, excluded")
            continue
        records.append(record)
    return records, problems


def _median_by_key(per_record: list[dict[str, int]]) -> dict[str, float]:
    """Median ns per key, in milliseconds, over records that carry that key."""
    keys = {key for record in per_record for key in record}
    out: dict[str, float] = {}
    for key in keys:
        values = [record[key] for record in per_record if key in record]
        out[key] = statistics.median(values) / 1e6
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _buckets(group: list[Record]) -> dict[str, float]:
    """Stage medians without the compile root, which ``total_ms`` already carries."""
    buckets = _median_by_key([r.by_prefix("stage:") for r in group])
    buckets.pop(COMPILE_EVENT, None)
    return buckets


def summarize(records: list[Record]) -> list[PointSummary]:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        key = point_name(record.workload, record.params)
        grouped.setdefault(key, []).append(record)

    summaries = []
    for point, group in sorted(grouped.items()):
        summaries.append(
            PointSummary(
                point=point,
                workload=group[0].workload,
                params=group[0].params,
                samples=len(group),
                graph_operations=max(r.graph_operations() for r in group),
                total_ms=statistics.median(r.total_ns(COMPILE_EVENT) for r in group)
                / 1e6,
                backend_ms=statistics.median(r.backend_ns() for r in group) / 1e6,
                buckets_ms=_buckets(group),
                passes_ms=_median_by_key([r.by_prefix("pass:") for r in group]),
            )
        )
    return summaries


def render_markdown(summaries: list[PointSummary], passes: bool = False) -> str:
    lines = [
        "| point | ops | frontend ms | backend ms | total ms | n |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.point} | {s.graph_operations} | {s.frontend_ms:.1f} | "
            f"{s.backend_ms:.1f} | {s.total_ms:.1f} | {s.samples} |"
        )
    if not passes:
        return "\n".join(lines)

    for s in summaries:
        lines += ["", f"### {s.point}", "", "| region | ms |", "|---|--:|"]
        for name, ms in s.buckets_ms.items():
            lines.append(f"| {name} | {ms:.1f} |")
        for name, ms in list(s.passes_ms.items())[:12]:
            lines.append(f"| {name} | {ms:.1f} |")
    return "\n".join(lines)


def write_csv(summaries: list[PointSummary], path: str) -> None:
    """One row per region, carrying the point's graph size.

    Graph size travels with every row so per-region time can be plotted against it
    without joining back to another table.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["point", "workload", "params", "graph_operations", "region", "median_ms"]
        )
        for s in summaries:
            params = json.dumps(s.params, sort_keys=True)
            rows = [("total", s.total_ms), ("frontend", s.frontend_ms)]
            rows += list(s.buckets_ms.items()) + list(s.passes_ms.items())
            for region, ms in rows:
                writer.writerow(
                    [
                        s.point,
                        s.workload,
                        params,
                        s.graph_operations,
                        region,
                        f"{ms:.3f}",
                    ]
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", help="directory of timing records")
    parser.add_argument("--passes", action="store_true", help="per-region detail")
    parser.add_argument("--csv", help="also write plot-ready rows here")
    args = parser.parse_args(argv)

    records, problems = load_records(args.records)
    for problem in problems:
        print(f"skipped {problem}", file=sys.stderr)
    if not records:
        print("no usable records", file=sys.stderr)
        return 1

    summaries = summarize(records)
    print(render_markdown(summaries, passes=args.passes))
    if args.csv:
        write_csv(summaries, args.csv)
        print(f"\ncsv -> {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

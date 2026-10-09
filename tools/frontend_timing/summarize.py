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

Medians across samples. Frontend time is the compile region minus the backend
invocations inside it. Records that did not finish a compile are reported and excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

COMPILE_EVENT = "stage:compile_fx:spyre_compile"
BACKEND_SUFFIX = ":backend_compile"
GRAPH_PIPELINE = "pipeline:CustomPreSchedulingPasses"

#: Metadata keys that describe the run rather than the point being measured.
_NON_PARAM_KEYS = frozenset(
    "sample cold cache_dir spyre_config workload recorder_version clock pid git_sha"
    " torch_spyre_path torch_spyre_version torch_version python_version frontend_only"
    " backend_skipped_kernels env_arm compile_wall_ms peak_rss_kb"
    " kernels_skipped".split()
)

#: Event ``meta`` keys that describe the graph. Any other number on a pass event is a
#: counter, so a counter added to the compiler shows up here with no edit.
_GRAPH_META_KEYS = frozenset(
    "input_operations output_operations input_nodes output_nodes passes".split()
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
    def arm(self) -> str:
        """The A/B arm this record was taken under; empty for the default."""
        return str(self.meta.get("env_arm", ""))

    @property
    def params(self) -> dict[str, Any]:
        return {k: v for k, v in self.meta.items() if k not in _NON_PARAM_KEYS}

    def total_ns(self, name: str) -> int:
        return sum(e["inclusive_ns"] for e in self.events if e["name"] == name)

    def backend_ns(self) -> int:
        return sum(
            e["inclusive_ns"] for e in self.events if e["name"].endswith(BACKEND_SUFFIX)
        )

    def graph_operations(self) -> int | None:
        """Largest pre-scheduling graph compiled; None, not 0, if there was none."""
        sizes = [
            e.get("meta", {}).get("input_operations", 0)
            for e in self.events
            if e["name"] == GRAPH_PIPELINE
        ]
        return max(sizes) if sizes else None

    def graph_nodes(self) -> int | None:
        """Largest FX graph compiled, or None if no pipeline reported one."""
        sizes = [
            (e.get("meta") or {})["input_nodes"]
            for e in self.events
            if e["name"].startswith("pipeline:")
            and "input_nodes" in (e.get("meta") or {})
        ]
        return max(sizes) if sizes else None

    def counters(self) -> dict[str, int]:
        """Analysis-call counts summed over every pass.

        Only pass events: a pipeline repeats its own passes' totals. Summed per compile
        rather than kept per pass, so the metric name set stays small.
        """
        totals: dict[str, int] = {}
        for event in self.events:
            if not event["name"].startswith("pass:"):
                continue
            for key, value in (event.get("meta") or {}).items():
                if key in _GRAPH_META_KEYS or not isinstance(value, (int, float)):
                    continue
                if isinstance(value, bool):
                    continue
                totals[key] = totals.get(key, 0) + value
        return totals

    def run_metrics(self) -> dict[str, float]:
        """Run-level numbers the child recorded alongside the events."""
        out: dict[str, float] = {}
        for key in ("compile_wall_ms", "peak_rss_kb", "kernels_skipped"):
            value = self.meta.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
        return out

    def by_prefix(self, prefix: str) -> dict[str, int]:
        totals: dict[str, int] = {}
        for event in self.events:
            if event["name"].startswith(prefix):
                totals[event["name"]] = totals.get(event["name"], 0) + int(
                    event["inclusive_ns"]
                )
        return totals


#: Rows-file version; a consumer that does not know it should refuse the file.
ROWS_SCHEMA = "frontend-timing-rows/1"


def metric_key(event_name: str) -> str:
    """``stage:Owner:what`` -> ``stage.Owner.what_ms``, safe as a warehouse Map key."""
    return event_name.replace(":", ".") + "_ms"


@dataclass
class PointSummary:
    """One workload at one set of parameters under one arm, with its samples."""

    point: str
    workload: str
    params: dict[str, Any]
    arm: str
    samples: int
    measurements: dict[str, list[float]] = field(default_factory=dict)

    def median(self, key: str) -> float:
        values = self.measurements.get(key)
        return statistics.median(values) if values else 0.0

    @property
    def graph_operations(self) -> int:
        return int(self.median("graph_operations"))

    @property
    def total_ms(self) -> float:
        return self.median("total_ms")

    @property
    def backend_ms(self) -> float:
        return self.median("backend_ms")

    @property
    def frontend_ms(self) -> float:
        # The median of per-sample differences, not the difference of medians: both
        # sides of the subtraction must come from one process.
        return self.median("frontend_ms")

    def regions_ms(self, prefix: str) -> dict[str, float]:
        """Median ms per region name under ``prefix``, largest first."""
        medians = {
            key: self.median(key)
            for key in self.measurements
            if key.startswith(prefix) and key.endswith("_ms")
        }
        return dict(sorted(medians.items(), key=lambda kv: -kv[1]))


def point_name(workload: str, params: dict[str, Any], arm: str = "") -> str:
    name = workload
    if params:
        name += "-" + "_".join(f"{k}{params[k]}" for k in sorted(params))
    if arm:
        name += f"+{arm}"
    return name


def load_records(directory: str) -> tuple[list[Record], list[str]]:
    """Load every record in ``directory``. Returns (usable, problems).

    Not recursive, which keeps the driver's discarded ``warmup/`` out of the summary.
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
            # The recorder writes at exit even when the compile raised.
            problems.append(f"{entry}: compile failed in {failed[0]}, excluded")
            continue
        records.append(record)
    return records, problems


def measurements(group: list[Record]) -> dict[str, list[float]]:
    """Every metric for one point, as per-sample values; an absent one is not a zero."""
    series: dict[str, list[float]] = {}

    def add(key: str, value: float) -> None:
        series.setdefault(key, []).append(float(value))

    for record in group:
        total = record.total_ns(COMPILE_EVENT)
        backend = record.backend_ns()
        add("total_ms", total / 1e6)
        add("backend_ms", backend / 1e6)
        add("frontend_ms", (total - backend) / 1e6)
        for key, size in (
            ("graph_operations", record.graph_operations()),
            ("graph_nodes", record.graph_nodes()),
        ):
            if size is not None:
                add(key, size)
        for name, ns in record.by_prefix("stage:").items():
            # Already total_ms, so a sum over stage.* must not see it again.
            if name == COMPILE_EVENT:
                continue
            add(metric_key(name), ns / 1e6)
        for name, ns in record.by_prefix("pass:").items():
            add(metric_key(name), ns / 1e6)
        for name, count in record.counters().items():
            add(f"counter.{name}", count)
        for name, value in record.run_metrics().items():
            add(name, value)
    return series


def summarize(records: list[Record]) -> list[PointSummary]:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        key = point_name(record.workload, record.params, record.arm)
        grouped.setdefault(key, []).append(record)
    return [
        PointSummary(
            point=point,
            workload=group[0].workload,
            params=group[0].params,
            arm=group[0].arm,
            samples=len(group),
            measurements=measurements(group),
        )
        for point, group in sorted(grouped.items())
    ]


def provenance(records: list[Record]) -> dict[str, Any]:
    """Build identity, read off the first record: it may be summarized elsewhere."""
    first = records[0].meta if records else {}
    keys = "git_sha torch_version torch_spyre_version python_version recorder_version"
    return {key: first.get(key) for key in keys.split() if first.get(key) is not None}


def render_markdown(
    summaries: list[PointSummary], passes: bool = False, top: int = 12
) -> str:
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
        for name, ms in s.regions_ms("stage.").items():
            lines.append(f"| {name} | {ms:.1f} |")
        ranked = list(s.regions_ms("pass.").items())
        for name, ms in ranked[:top]:
            lines.append(f"| {name} | {ms:.1f} |")
        if len(ranked) > top:
            lines.append(f"| ...{len(ranked) - top} more passes (see --csv/--json) | |")
    return "\n".join(lines)


def write_csv(summaries: list[PointSummary], path: str) -> None:
    """One row per metric, each with the point's graph size so plots need no join."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = "point workload params arm graph_operations region median_ms samples"
        writer.writerow(header.split())
        for s in summaries:
            params = json.dumps(s.params, sort_keys=True)
            for region in sorted(s.measurements):
                values = s.measurements[region]
                row = [s.point, s.workload, params, s.arm, s.graph_operations, region]
                median = f"{statistics.median(values):.3f}"
                writer.writerow(row + [median, len(values)])


def write_json(
    summaries: list[PointSummary],
    records: list[Record],
    path: str,
    tier: str | None = None,
) -> None:
    """The rows file: one object per point, every metric as its per-sample values."""
    payload = {
        "schema": ROWS_SCHEMA,
        "meta": {
            **provenance(records),
            "tier": tier,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "points": [
            {
                "name": s.point,
                "workload": s.workload,
                "params": s.params,
                "arm": s.arm,
                "samples": s.samples,
                "measurements": {k: s.measurements[k] for k in sorted(s.measurements)},
            }
            for s in summaries
        ],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", help="directory of timing records")
    parser.add_argument("--passes", action="store_true", help="per-region detail")
    parser.add_argument("--csv", help="also write plot-ready rows here")
    parser.add_argument("--json", help="also write the dashboard rows file here")
    parser.add_argument(
        "--tier", help="tier label to record in the rows file's metadata"
    )
    parser.add_argument(
        "--top", type=int, default=12, help="passes shown per point with --passes (12)"
    )
    args = parser.parse_args(argv)

    records, problems = load_records(args.records)
    for problem in problems:
        print(f"skipped {problem}", file=sys.stderr)
    if not records:
        print("no usable records", file=sys.stderr)
        return 1

    summaries = summarize(records)
    print(render_markdown(summaries, passes=args.passes, top=args.top))
    if args.csv:
        write_csv(summaries, args.csv)
        print(f"\ncsv -> {args.csv}", file=sys.stderr)
    if args.json:
        write_json(summaries, records, args.json, tier=args.tier)
        print(f"rows -> {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

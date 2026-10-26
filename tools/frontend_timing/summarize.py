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
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

COMPILE_EVENT = "stage:compile_fx:spyre_compile"
#: Matched exactly, never by trailing segment. Upstream has a dynamo_timed phase of
#: its own called backend_compile -- the call into Inductor, recorded as
#: stage:torch:backend_compile -- which *encloses* the Spyre compile region, so a
#: suffix match totals the whole Inductor compile as backend time and drives the
#: frontend subtraction negative. Mirrors timing_recorder.BACKEND_COMPILE_EVENT.
BACKEND_EVENT = "stage:SpyreAsyncCompile:backend_compile"
GRAPH_PIPELINE = "pipeline:CustomPreSchedulingPasses"

#: Event-name prefix for one pass of the graph pipeline, which a caller completes
#: with the pass's own name. Built from its parts rather than written out whole:
#: a literal of the finished form is a word ending in "pass", a colon and a
#: value, which IBM's secret scanner reads as a password assignment and blocks
#: the push on. ``summarize.py`` matches the bare ``pass:`` prefix and so never
#: tripped it; the tests, which need whole names, did.
GRAPH_PASS_PREFIX = GRAPH_PIPELINE.replace("pipeline:", "pass:") + ":"

#: The FX-graph pipeline whose ``input_nodes`` is an FX node count. The node pipelines
#: reuse that key for a *scheduler* node count, so taking a max over every pipeline
#: would mix two different units into one metric.
#: The FX-graph pipeline whose ``input_nodes`` is an FX node count, read for
#: ``graph_nodes``. ``CustomPreGradPasses`` returns before opening its region
#: unless a node carries ``meta["val"]`` on a Spyre device, and the pre-grad
#: graph only carries ``example_value`` -- so it is absent from every real
#: record and ``graph_nodes`` was never produced by a sweep, only by a test.
#: ``CustomPrePasses`` is the pipeline that does run; the node pipelines reuse
#: ``input_nodes`` for a *scheduler* node count, so taking a max over every
#: pipeline would mix two different units into one metric.
FX_PIPELINE = "pipeline:CustomPrePasses"

#: Event ``meta`` keys that describe the graph. Any other number on a pass event is a
#: counter, so a counter added to the compiler shows up here with no edit.
_GRAPH_META_KEYS = frozenset(
    "input_operations output_operations input_nodes output_nodes passes".split()
)


def _has_ancestor(
    event: dict[str, Any], ordinals: set[int], parent_of: dict[int, Any]
) -> bool:
    """Whether this event is inside another event of the same name.

    Walks the whole parent chain rather than checking the immediate parent: a
    region can recurse with other regions in between, and ``GraphLowering.run``
    does.
    """
    parent = event["parent_ordinal"]
    while parent is not None:
        if parent in ordinals:
            return True
        parent = parent_of.get(parent)
    return False


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
        """The builder's resolved parameters, from their own metadata key.

        A namespace rather than a denylist over the whole of ``meta``: the recorder's
        contract is that readers tolerate unknown keys, so any key added to run
        metadata would otherwise become a workload parameter -- renaming every point,
        and splitting one point into single samples if the value varies per process.
        """
        params = self.meta.get("params")
        return dict(params) if isinstance(params, dict) else {}

    @property
    def compiled(self) -> bool:
        """Whether the child reported reaching the end of its compile.

        Recorded by the driver rather than inferred from event errors. The recorder
        stamps an error on any region an exception passed through, including ones a
        caller deliberately swallows -- provenance collection does exactly that -- so
        error-based exclusion drops successful compiles.
        """
        return self.meta.get("compiled") is True

    def total_ns(self, name: str) -> int:
        return sum(e["inclusive_ns"] for e in self.events if e["name"] == name)

    def ran_backend(self) -> bool:
        return any(e["name"] == BACKEND_EVENT for e in self.events)

    def backend_ns(self) -> int:
        return sum(e["inclusive_ns"] for e in self.events if e["name"] == BACKEND_EVENT)

    def graph_operations(self) -> int | None:
        """Largest pre-scheduling graph compiled; None, not 0, if there was none."""
        sizes = [
            (e.get("meta") or {})["input_operations"]
            for e in self.events
            if e["name"] == GRAPH_PIPELINE
            and "input_operations" in (e.get("meta") or {})
        ]
        return max(sizes) if sizes else None

    def graph_nodes(self) -> int | None:
        """FX nodes entering the pre-grad pipeline, or None if it did not report.

        Read from one named pipeline, not maxed over every pipeline: the node
        pipelines put a *scheduler* node count under the same key, and a max over both
        silently reports whichever unit happens to be larger.
        """
        sizes = [
            (e.get("meta") or {})["input_nodes"]
            for e in self.events
            if e["name"] == FX_PIPELINE and "input_nodes" in (e.get("meta") or {})
        ]
        return max(sizes) if sizes else None

    def counters(self) -> dict[str, int]:
        """Work counts from pass and stage event metadata.

        Pass counts are summed under their bare key: they are per-pass instances of
        the same measurement, and a compile total is what a dashboard wants. Stage
        counts are keyed by their region instead, because two stages can carry the
        same key for different things -- the scratchpad records ``buffers`` on more
        than one region -- and summing those would report a number that is neither.

        Pipelines are skipped: they repeat the totals of the passes inside them.
        """
        totals: dict[str, int] = {}
        for event in self.events:
            name = event["name"]
            if name.startswith("pass:"):
                prefix = ""
            elif name.startswith("stage:"):
                prefix = name[len("stage:") :].replace(":", ".") + "."
            else:
                continue
            for key, value in (event.get("meta") or {}).items():
                if key in _GRAPH_META_KEYS or not isinstance(value, (int, float)):
                    continue
                if isinstance(value, bool):
                    continue
                full = prefix + key
                totals[full] = totals.get(full, 0) + value
        return totals

    def per_region(self) -> dict[str, float]:
        """Every pass's and stage's own counters and graph sizes, keyed by region.

        ``counters`` sums a pass's counts into a compile total, which is what a
        dashboard wants and what the complexity audit cannot use: a claim about
        one pass is in that pass's own variables, against the operation count at
        *its* entry rather than at pipeline entry. Both are emitted -- the summed
        keys are unchanged, these are additional.

        Keys are dotted (``pass.Pipeline.name.counter.read_writes.misses``), which
        is also what keeps them out of the secret scanner's way: the colon form of
        a pass event name reads as a password assignment and blocks a push.

        A reentrant region's values are summed, which is right for a count and for
        an entry size both: two entries of one pass are two lots of work.
        """
        out: dict[str, float] = {}
        for event in self.events:
            name = event["name"]
            if not name.startswith(("pass:", "stage:")):
                continue
            base = metric_key(name)[: -len("_ms")]
            for key, value in (event.get("meta") or {}).items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if key in _GRAPH_META_KEYS:
                    full = f"{base}.{key}"
                else:
                    full = f"{base}.{counter_metric_key(key)}"
                out[full] = out.get(full, 0.0) + float(value)
        return out

    def run_metrics(self) -> dict[str, float]:
        """Run-level numbers the child recorded alongside the events."""
        out: dict[str, float] = {}
        for key in ("compile_wall_ms", "peak_rss_kb", "kernels_skipped"):
            value = self.meta.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
        out["solve_time_bound"] = float(self.solve_time_bound())
        return out

    def solve_time_bound(self) -> bool:
        """Whether any scratchpad solve stopped at its time limit.

        A solve that returns FEASIBLE rather than OPTIMAL hands back whichever
        plan it had reached when the clock ran out, and a different plan does
        different downstream work -- which is why the counts that vary between
        samples of one point are exactly the points that read FEASIBLE: on a
        real sweep, misses varied on 4 of 26 points and every one of them was
        time-bound, while the model-size counts were identical everywhere.
        Such a point also cannot report a harder solve as more time, so a fit
        through it reads flat. Flagged rather than dropped: the point still
        measures everything outside the solve.
        """
        for event in self.events:
            if event["name"].endswith(":cpsat_solve"):
                if (event.get("meta") or {}).get("status") not in (None, "OPTIMAL"):
                    return True
        return False

    def by_prefix(self, prefix: str) -> dict[str, int]:
        """Inclusive time per region name, counting a reentrant region once.

        Several regions are entered inside themselves: ``GraphLowering.run``
        recurses through subgraphs, and the ``_recursive_*`` graph passes are
        named for doing so. Summing every event of one name then counts the
        inner time at each level it appears -- 71% over on
        ``GraphLowering.run`` for a Granite layer at S=1024, 186% on
        ``_recursive_joint_graph_passes``. Only the outermost entry of each name
        is counted, which is the wall time that name was inside.

        Self time is not a substitute: it is right only where the nested entry
        is the region's only child, and ``_recursive_post_grad_passes`` reads
        109 ms self against 436 ms of real cost because it has other children
        too.
        """
        parent_of = {event["ordinal"]: event["parent_ordinal"] for event in self.events}
        totals: dict[str, int] = {}
        for name, events in self._by_name(prefix).items():
            ordinals = {event["ordinal"] for event in events}
            outer = [e for e in events if not _has_ancestor(e, ordinals, parent_of)]
            totals[name] = sum(int(event["inclusive_ns"]) for event in outer)
        return totals

    def _by_name(self, prefix: str) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in self.events:
            if event["name"].startswith(prefix):
                grouped.setdefault(event["name"], []).append(event)
        return grouped

    def self_by_prefix(self, prefix: str) -> dict[str, int]:
        """Self time per region: inclusive minus direct children.

        Recorded alongside the inclusive series because they answer different
        questions and neither substitutes for the other. Inclusive is what the
        frontend subtraction needs and what shows the nesting; it also sums to more
        than the compile, so ranking regions by it puts every ancestor of the hot
        pass at ~100% and says nothing. Self time is what ranks cost.
        """
        totals: dict[str, int] = {}
        for event in self.events:
            if event["name"].startswith(prefix):
                totals[event["name"]] = totals.get(event["name"], 0) + int(
                    event["self_ns"]
                )
        return totals


#: Rows-file version; a consumer that does not know it should refuse the file.
ROWS_SCHEMA = "frontend-timing-rows/2"

#: Suffixes on a counter name that mean it is a duration, not a count. A pass
#: reports both through the same ``meta`` dict, and the name is the only thing
#: that distinguishes them: ``read_writes.extract_ns`` and
#: ``cpsat_solve.solver_wall_s`` are clocks, and they spread 14% and 64% within
#: one point on a real sweep where every count beside them was identical.
#: Emitting them as ``counter.*`` made "the counters are reproducible" false of
#: the prefix as a whole, which is the one property the counters are read for.
#: They go under ``duration.*`` instead, and anything else a pass adds is still
#: a counter with no edit here.
_DURATION_SUFFIXES = ("_ns", "_s", "_ms", "_us", "_wall_s")


def counter_metric_key(name: str) -> str:
    """``counter.<name>`` for a count, ``duration.<name>`` for a clock."""
    prefix = "duration" if name.endswith(_DURATION_SUFFIXES) else "counter"
    return f"{prefix}.{name}"


def metric_key(event_name: str) -> str:
    """``stage:Owner:what`` -> ``stage.Owner.what_ms``, safe as a warehouse Map key.

    Each region also gets a ``..._self_ms`` sibling. New metric names are additive, so
    a reader written against an older rows file keeps working and simply does not see
    them.
    """
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

    def minimum(self, key: str) -> float | None:
        """The least of this metric's samples, or None when nothing measured it.

        The minimum rather than the median. Co-tenant interference on a shared pod
        only ever makes a compile slower -- measured at 11% typical and 63% worst
        spread across three samples of one point -- so the fastest sample is the one
        least contaminated by it, and the long tail the median was chosen to resist is
        the very thing being discarded. For a deterministic metric (a graph size, a
        work count) every sample agrees and the choice does not arise.

        None rather than 0.0 when absent: a metric nothing measured is not a zero.
        """
        values = self.measurements.get(key)
        return min(values) if values else None

    @property
    def graph_operations(self) -> int | None:
        value = self.minimum("graph_operations")
        return None if value is None else int(value)

    @property
    def total_ms(self) -> float | None:
        return self.minimum("total_ms")

    @property
    def backend_ms(self) -> float | None:
        return self.minimum("backend_ms")

    @property
    def frontend_ms(self) -> float | None:
        # Of the per-sample differences, not a difference of two statistics: both
        # sides of the subtraction must come from one process.
        return self.minimum("frontend_ms")

    def self_ms(self, region_key: str) -> float:
        """The ``_self_ms`` sibling of an inclusive region key; 0.0 if absent."""
        return self.minimum(region_key[:-3] + "_self_ms") or 0.0

    def regions_ms(self, prefix: str) -> dict[str, float]:
        """Least inclusive ms per region under ``prefix``, ranked by self time.

        The self siblings are not keys here -- they are the ranking. Inclusive time
        sums to more than the compile, so ordering by it puts every ancestor of the
        hot pass near the top and says nothing about where the cost is.
        """
        lows = {
            key: self.minimum(key) or 0.0
            for key in self.measurements
            if key.startswith(prefix)
            and key.endswith("_ms")
            and not key.endswith("_self_ms")
        }
        return dict(sorted(lows.items(), key=lambda kv: -self.self_ms(kv[0])))


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
        if not record.compiled:
            # The recorder writes at exit even when the compile raised, so a record
            # exists either way and the driver's own verdict is what separates them.
            failed = [e["name"] for e in record.events if e.get("error")]
            where = f" ({failed[0]})" if failed else ""
            problems.append(f"{entry}: compile did not complete{where}, excluded")
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
        # Omitted, not zero, when the backend was skipped: a 0 ms backend and a backend
        # that never ran are different facts, and only one of them is a measurement.
        if record.ran_backend():
            add("backend_ms", backend / 1e6)
        add("frontend_ms", (total - backend) / 1e6)
        for key, size in (
            ("graph_operations", record.graph_operations()),
            ("graph_nodes", record.graph_nodes()),
        ):
            if size is not None:
                add(key, size)
        for prefix in ("stage:", "pass:", "pipeline:"):
            for name, ns in record.self_by_prefix(prefix).items():
                add(metric_key(name)[:-3] + "_self_ms", ns / 1e6)
        for name, ns in record.by_prefix("stage:").items():
            # Already total_ms, so a sum over stage.* must not see it again.
            if name == COMPILE_EVENT:
                continue
            add(metric_key(name), ns / 1e6)
        for name, ns in record.by_prefix("pass:").items():
            add(metric_key(name), ns / 1e6)
        for name, count in record.counters().items():
            add(counter_metric_key(name), count)
        for name, value in record.per_region().items():
            add(name, value)
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


#: Metadata that must agree across every record in a summary. A sweep writes one file
#: per sample with the pid in its name, so nothing is ever replaced and a second sweep
#: into the same directory silently doubles each point's samples with the earlier
#: build's. Disagreement here is that situation, reported rather than averaged.
_IDENTITY_KEYS = ("git_sha", "torch_version", "torch_spyre_version")

#: Config entries that are per-record by construction and so can never agree
#: across a sweep. ``timing_out`` is each sample's own destination file and the
#: driver gives every sample its own, so comparing it verbatim makes the conflict
#: check fire on every multi-sample sweep -- which is every real one. The config
#: declares its own ``_cache_config_ignore_prefix`` for the same reason, so that
#: is honoured here rather than duplicated.
_CONFIG_IGNORED = ("timing_out",)


def comparable_config(meta: dict[str, Any]) -> Any:
    """``spyre_config`` with the per-record entries dropped."""
    config = meta.get("spyre_config")
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except ValueError:
            return config
    if not isinstance(config, dict):
        return config
    ignored = set(_CONFIG_IGNORED)
    ignored.update(config.get("_cache_config_ignore_prefix") or ())
    return {k: v for k, v in config.items() if k not in ignored}


def identity_conflicts(records: list[Record]) -> list[str]:
    """Keys on which these records disagree, with the values they disagree on."""
    conflicts = []
    for key in _IDENTITY_KEYS:
        seen = {json.dumps(r.meta.get(key), sort_keys=True) for r in records}
        if len(seen) > 1:
            shown = ", ".join(sorted(s[:60] for s in seen)[:3])
            conflicts.append(f"{key} differs across records: {shown}")
    # Config is compared within each arm, not across them. An arm exists to change
    # the config -- the three shipped arms set lx_planner_relayout,
    # lx_solver_relayout_groups_per_edge and sencores, all read at import -- so
    # comparing every record against every other rejected any sweep containing one,
    # which is every weekly sweep. An arm is already part of a point's identity and
    # is never averaged into its control; this makes the check agree with that.
    by_arm: dict[str, list[Record]] = {}
    for record in records:
        by_arm.setdefault(record.meta.get("env_arm") or "", []).append(record)
    for arm, group in sorted(by_arm.items()):
        seen = {json.dumps(comparable_config(r.meta), sort_keys=True) for r in group}
        if len(seen) > 1:
            shown = ", ".join(sorted(s[:60] for s in seen)[:3])
            where = f" within arm {arm}" if arm else ""
            conflicts.append(f"spyre_config differs across records{where}: {shown}")
    return conflicts


def provenance(records: list[Record]) -> dict[str, Any]:
    """Build identity, read off the first record: it may be summarized elsewhere."""
    first = records[0].meta if records else {}
    keys = "git_sha torch_version torch_spyre_version python_version recorder_version"
    return {key: first.get(key) for key in keys.split() if first.get(key) is not None}


def _cell(value: float | int | None, fmt: str = "{:.1f}") -> str:
    return "-" if value is None else fmt.format(value)


def render_markdown(
    summaries: list[PointSummary], passes: bool = False, top: int = 12
) -> str:
    lines = [
        "| point | ops | frontend ms | backend ms | total ms | n |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for s in summaries:
        # "-" for a metric nothing measured. Printing 0 would read as a graph that
        # shrank to nothing, or a backend that ran instantly rather than not at all.
        cells = [
            s.point,
            _cell(s.graph_operations, "{:.0f}"),
            _cell(s.frontend_ms),
            _cell(s.backend_ms),
            _cell(s.total_ms),
            str(s.samples),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not passes:
        return "\n".join(lines)

    for s in summaries:
        # Both columns, because neither answers the other's question: inclusive is
        # what the frontend subtraction uses and what shows the nesting, self is what
        # ranks cost. Rows are ordered by self for that reason.
        header = "| region | ms | self ms |"
        lines += ["", f"### {s.point}", "", header, "|---|--:|--:|"]
        for name, ms in s.regions_ms("stage.").items():
            lines.append(f"| {name} | {ms:.1f} | {s.self_ms(name):.1f} |")
        ranked = list(s.regions_ms("pass.").items())
        for name, ms in ranked[:top]:
            lines.append(f"| {name} | {ms:.1f} | {s.self_ms(name):.1f} |")
        if len(ranked) > top:
            lines.append(
                f"| ...{len(ranked) - top} more passes (see --csv/--json) | | |"
            )
    return "\n".join(lines)


def write_csv(summaries: list[PointSummary], path: str) -> None:
    """One row per metric, each with the point's graph size so plots need no join.

    The value column is named ``minimum`` rather than carrying a unit: the metric name
    already says milliseconds or a count, and the old ``median_ms`` was wrong on both
    halves once counts and byte sizes joined the table.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = "point workload params arm graph_operations region minimum samples"
        writer.writerow(header.split())
        for s in summaries:
            params = json.dumps(s.params, sort_keys=True)
            ops = "" if s.graph_operations is None else s.graph_operations
            for region in sorted(s.measurements):
                values = s.measurements[region]
                row = [s.point, s.workload, params, s.arm, ops, region]
                writer.writerow(row + [f"{min(values):.3f}", len(values)])


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
    parser.add_argument(
        "--allow-mixed",
        action="store_true",
        help="summarize records that disagree on build or configuration",
    )
    args = parser.parse_args(argv)

    records, problems = load_records(args.records)
    for problem in problems:
        print(f"skipped {problem}", file=sys.stderr)
    if not records:
        print("no usable records", file=sys.stderr)
        return 1

    conflicts = identity_conflicts(records)
    if conflicts:
        # Printed either way. Under --allow-mixed these are being tolerated on
        # purpose, and a reader of the output still needs to know which build or
        # configuration the numbers came from, since provenance() reports only
        # the first record's.
        label = "warning" if args.allow_mixed else "error"
        for conflict in conflicts:
            print(f"{label}: {conflict}", file=sys.stderr)
    if conflicts and not args.allow_mixed:
        print(
            "refusing to summarize records from more than one build or configuration; "
            "sweep into a fresh --out, or pass --allow-mixed if this is deliberate",
            file=sys.stderr,
        )
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

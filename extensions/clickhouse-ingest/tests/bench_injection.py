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

"""Regression-injection harness: synthetic nightly history plus a deliberately slowed run.

Used by tests/test_bench_verdict_injection.py on chdb. As a script it runs the same scenario in a
personal sandbox database (a name starting `sandbox_`; it refuses anything else):

    python tests/bench_injection.py --database sandbox_<you>__f1d
"""

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

# (name, metric, base value, higher_is_better); one benchmark per row, synthetic names only.
BENCHMARKS = (
    ("op-alpha", "kernel_mean_ms", 10.0, False),
    ("op-beta", "kernel_mean_ms", 4.0, False),
    ("op-gamma", "total_duration_ms", 22.0, False),
    ("op-delta", "output_throughput", 900.0, True),
)
COMPONENT = "synth-comp"
BACKEND = "synth-backend"
ARCH = "x86_64"
NIGHTLY_JOB = "synth-nightly"
PR_JOB = "synth-pr"
START = datetime(2026, 10, 1, 2, 0, 0)
NOISE_CV = 0.015
SAMPLES = 5
SCHEMA_FILES = (
    "10-functional-tests",
    "20-artifacts",
    "30-benchmarks",
    "49-result-uploads",
    "50-artifact-views",
    "60-benchmark-views",
    "62-benchmark-verdicts",
    "81-bench-environments",
    "82-run-placements",
    "83-result-invalidations",
    "84-bench-views",
    "85-benchmark-verdict-views",
)
NS = uuid.UUID("5b5b5b5b-0000-5000-8000-000000000000")


def bench_id(name):
    return str(uuid.uuid5(NS, f"bench:{name}"))


def run_uuid(n):
    return str(uuid.uuid5(NS, f"run:{n}"))


class Injection:
    """Seeds a database through `db.insert(table, *rows)` and `db.rows(sql)`."""

    def __init__(self, db, seed=7):
        self.db = db
        self.rng = random.Random(seed)
        self.next_run = 0

    def register_benchmarks(self):
        self.db.insert(
            "benchmarks",
            *[
                {
                    "benchmark_id": bench_id(n),
                    "component": COMPONENT,
                    "name": n,
                    "tags": [],
                    "props": {"record_type": "op", "input_shapes": "[1]"},
                }  # fmt: skip
                for n, *_ in BENCHMARKS
            ],
        )

    def run(self, slow=None, job=NIGHTLY_JOB, repeats=3, props=None, day=None):
        """Insert one run; `slow` maps benchmark name to a worsening factor (0.25 = 25 % worse)."""
        n = self.next_run
        self.next_run += 1
        when = START + timedelta(days=n if day is None else day)
        stamp = when.strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for name, metric, base, higher in BENCHMARKS:
            worse = (slow or {}).get(name, 0.0)
            level = base * (1 - worse if higher else 1 + worse)
            for rep in range(repeats):
                samples = [
                    round(self.rng.gauss(level, level * NOISE_CV), 4)
                    for _ in range(SAMPLES)
                ]
                rows.append(
                    {
                        "ts": stamp,
                        "run_id": run_uuid(n),
                        "benchmark_id": bench_id(name),
                        "component": COMPONENT,
                        "backend": BACKEND,
                        "measurements": {metric: samples},
                        "iterations": SAMPLES,
                        "props": {
                            "source_file": f"{job}/results.json",
                            "arch": ARCH,
                            "run_utc": stamp,
                            "run.repeat": str(rep),
                            "run.trust": "internal",
                            "run.validity": "valid",
                            "run.purpose": "nightly",
                            **(props or {}),
                        },
                    }  # fmt: skip
                )
        self.db.insert("benchmark_runs", *rows)
        return run_uuid(n)

    def history(self, nights=22):
        self.register_benchmarks()
        return [self.run() for _ in range(nights)]

    def invalidate(self, run_id, ts, n=1, reason_code="bad_card", state="invalidated"):
        self._event(run_id, ts, n, reason_code, state, "run")

    def invalidate_repeat(self, run_id, repeat, ts, n=9):
        """One repeat of every benchmark of the run (a run_benchmark event per benchmark)."""
        for k, (name, *_) in enumerate(BENCHMARKS):
            self._event(
                run_id,
                ts,
                n * 10 + k,
                "bad_card",
                "invalidated",
                "run_benchmark",
                bench_id(name),
                repeat,
            )

    def _event(
        self, run_id, ts, n, reason_code, state, scope, benchmark=None, repeat=None
    ):
        self.db.insert(
            "result_invalidations",
            {**({"benchmark_id": benchmark, "repeat": repeat} if benchmark else {}),"invalidation_id": str(uuid.uuid5(NS, f"inv:{n}")), "ts": ts, "scope_kind": scope, "run_id": run_id,
             "state": state, "reason_code": reason_code, "actor": "synthetic@example.com"},
        )  # fmt: skip

    def verdicts(self, run_id):
        """{(name, metric): (status, reason, n_baseline)} straight from the verdict view."""
        out = self.db.rows(
            "SELECT name, metric, status, reason, n_baseline FROM v_benchmark_metric_verdicts "
            f"WHERE run_id = toUUID('{run_id}')"
        )
        return {
            (r["name"], r["metric"]): (r["status"], r["reason"], r["n_baseline"])
            for r in out
        }

    def baseline_ids(self, run_id):
        """Run ids in the rolling baseline window of this run's first benchmark row."""
        out = self.db.rows(
            "SELECT arrayMap(x -> toString(x), baseline_run_ids) AS ids FROM v_benchmark_metric_verdicts "
            f"WHERE run_id = toUUID('{run_id}') LIMIT 1"
        )
        return out[0]["ids"]


DATETIME_COLUMNS = ("ts", "window_from", "window_to", "audit_timestamp")


def to_datetime(value):
    """'YYYY-MM-DD HH:MM:SS' as a UTC datetime, independent of the local timezone."""
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    return value


class ClickHouseSandbox:
    """The same insert/rows surface over clickhouse_connect, for a sandbox database only."""

    def __init__(self, client, database):
        if not database.startswith("sandbox_"):
            raise SystemExit(
                f"refusing to write to {database!r}: only sandbox_* databases are allowed"
            )
        self.client, self.database = client, database

    def insert(self, table, *rows):
        cols = sorted({k for r in rows for k in r})
        self.client.insert(
            f"{self.database}.{table}",
            [
                [
                    to_datetime(r.get(c)) if c in DATETIME_COLUMNS else r.get(c)
                    for c in cols
                ]
                for r in rows
            ],
            column_names=cols,
        )

    def rows(self, sql):
        return list(self.client.query(sql).named_results())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--database", required=True)
    args = ap.parse_args(argv)
    from spyre_clickhouse_ingest.client import ClickHouse

    sink = ClickHouseSandbox(ClickHouse.connect(database=args.database), args.database)
    inj = Injection(sink)
    inj.history()
    inj.run(slow={"op-alpha": 0.25})
    bad = inj.run(slow={"op-alpha": 0.25})
    print(json.dumps({"|".join(k): v for k, v in inj.verdicts(bad).items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

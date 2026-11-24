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

"""Synthetic nightly history with injectable slowdowns, for the verdict view on chdb."""

import random
import uuid
from datetime import datetime, timedelta

# (name, metric, base value, higher_is_better); synthetic names only.
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
START = datetime(2026, 10, 1, 2, 0, 0)
NOISE_CV = 0.015
SAMPLES = 5
NS = uuid.UUID("5b5b5b5b-0000-5000-8000-000000000000")


def bench_id(name):
    return str(uuid.uuid5(NS, f"bench:{name}"))


def run_uuid(n):
    return str(uuid.uuid5(NS, f"run:{n}"))


class Fixture:
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
        """Insert one run; `slow` maps a benchmark name to a worsening factor (0.25 = 25 % worse).

        A `props` value of None removes that key, so old-style rows can be written."""
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
                p = {
                    "source_file": f"{job}/results.json",
                    "arch": ARCH,
                    "run_utc": stamp,
                    "run.repeat": str(rep),
                    "run.purpose": "nightly",
                    **(props or {}),
                }  # fmt: skip
                rows.append(
                    {
                        "ts": stamp,
                        "run_id": run_uuid(n),
                        "benchmark_id": bench_id(name),
                        "component": COMPONENT,
                        "backend": BACKEND,
                        "measurements": {metric: samples},
                        "iterations": SAMPLES,
                        "props": {k: v for k, v in p.items() if v is not None},
                    }  # fmt: skip
                )
        self.db.insert("benchmark_runs", *rows)
        return run_uuid(n)

    def history(self, nights=22, **kw):
        self.register_benchmarks()
        return [self.run(**kw) for _ in range(nights)]

    def event(self, run_id, state, n, ts, benchmark=None, repeat=None):
        """One invalidation event: whole run, or one benchmark (and repeat) of it."""
        row = {
            "invalidation_id": str(uuid.uuid5(NS, f"inv:{n}")),
            "ts": ts,
            "scope_kind": "run" if benchmark is None else "run_benchmark",
            "run_id": run_id,
            "state": state,
            "reason_code": "bad_card",
            "actor": "synthetic@example.com",
        }  # fmt: skip
        if benchmark is not None:
            row["benchmark_id"] = bench_id(benchmark)
            if repeat is not None:
                row["repeat"] = repeat
        self.db.insert("result_invalidations", row)

    def invalidate(self, run_id, n=1, ts="2026-12-01 00:00:00", **kw):
        self.event(run_id, "invalidated", n, ts, **kw)

    def suspect(self, run_id, n=1, ts="2026-12-01 00:00:00", **kw):
        self.event(run_id, "suspect", n, ts, **kw)

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

    def baseline(self, run_id, name="op-alpha"):
        """(baseline run ids, baseline median) of one benchmark in a run's verdict row."""
        out = self.db.rows(
            "SELECT arrayMap(x -> toString(x), baseline_run_ids) AS ids, baseline_value AS med, n_current "
            f"FROM v_benchmark_metric_verdicts WHERE run_id = toUUID('{run_id}') AND name = '{name}'"
        )
        return out[0]["ids"], out[0]["med"], out[0]["n_current"]

    def baseline_is_its_runs(self, run_id, name="op-alpha"):
        """True when the baseline median is the median of exactly the samples of its listed runs."""
        ids, med, _ = self.baseline(run_id, name)
        metric = next(m for n, m, *_ in BENCHMARKS if n == name)
        listed = ", ".join(f"toUUID('{i}')" for i in ids)
        out = self.db.rows(
            f"SELECT round(median(s), 4) AS m FROM benchmark_runs ARRAY JOIN measurements['{metric}'] AS s "
            f"WHERE benchmark_id = toUUID('{bench_id(name)}') AND run_id IN ({listed}) AND s != 0"
        )
        return out[0]["m"] == med

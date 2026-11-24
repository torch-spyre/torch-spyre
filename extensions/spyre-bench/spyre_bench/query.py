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

"""`spyre-bench query`: ask the store what it already holds.

`verdict` answers for one run. This answers the questions that come up between runs: is anything
regressed, which series are stale, how has this benchmark moved. The queries themselves are the
ingest's own read API (`spyre_clickhouse_ingest.perf_api`), which is where the SQL, the windows and
the invalidation rules live; this verb only resolves names, routes and prints, so the terminal and a
dashboard cannot drift apart in what they report.

A series is addressed by `benchmark_id`, which nobody wants to carry by hand, so `series` also takes
a case name and derives the id through the catalog the same way `ids` does.

Needs `spyre-bench[clickhouse]`; `series --case` additionally needs `spyre-bench[ingest]` to derive
an ingest-mode id.
"""

import argparse
import json

from spyre_bench import cliutil, store
from spyre_bench.cliutil import UsageError

ROUTES = {
    "overview": "/api/v2/perf/overview",
    "suites": "/api/v2/perf/suites",
    "compare": "/api/v2/perf/compare",
    "references": "/api/v2/perf/references",
}
PARAMS = ("arch", "component", "days", "metric", "backend", "source_job", "limit", "status", "q", "kind", "cursor")


def api():
    try:
        from spyre_clickhouse_ingest import perf_api
    except ImportError as e:
        raise UsageError(
            f"the read API is not importable ({e}); install spyre-bench[clickhouse] alongside spyre-clickhouse-ingest"
        ) from None
    return perf_api


def runner():
    """A `run(sql, params)` callable on the configured store."""
    try:
        client = store.ClickHouseStore().client()
    except store.StoreError as e:
        raise UsageError(str(e)) from None
    return api().clickhouse_runner(client)


def benchmark_id_of(a):
    """The series id to query: an explicit id, or the one a case in the catalog derives."""
    if a.benchmark_id:
        return a.benchmark_id
    if not a.case:
        raise UsageError("series needs either a BENCHMARK_ID or --case NAME")
    cat = cliutil.open_catalog(a.catalog, a.central, a.pack)
    case = next((c for c in cat.cases if c["name"] == a.case), None)
    if case is None:
        raise UsageError(f"no case {a.case!r} in the catalog")
    return cat.ids(case)["benchmark_id"]


def query_params(a):
    return {k: getattr(a, k) for k in PARAMS if getattr(a, k, None) not in (None, "")}


def call(path, params, run=None):
    perf_api = api()
    try:
        status, body = perf_api.route(run or runner(), path, params)
    except perf_api.ApiError as e:
        raise UsageError(f"{e.status} {e.code}: {e.message}") from None
    if status >= 400:
        raise UsageError(f"{status}: {json.dumps(body)}")
    return body


def print_table(rows, columns):
    """Columns that at least one row carries, padded; a value the rows do not have is left out."""
    if not rows:
        print("(no rows)")
        return
    present = [c for c in columns if any(r.get(c) is not None for r in rows)]
    if not present:
        print("(no rows)")
        return
    widths = {c: max([len(c)] + [len(_cell(r.get(c))) for r in rows]) for c in present}
    print("  ".join(c.upper().ljust(widths[c]) for c in present).rstrip())
    for r in rows:
        print("  ".join(_cell(r.get(c)).ljust(widths[c]) for c in present).rstrip())


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


TABLES = {
    "overview": (
        "tiles",
        ("arch", "component", "benchmarks_in_window", "runs_in_window", "invalidated_runs_in_window"),
    ),
    "suites": ("items", ("verdict", "name", "arch", "backend", "primary_metric", "worst_delta_pct", "run_ts")),
    "references": ("items", ("key", "version", "state", "metric", "value")),
}


def show(verb, body, as_json):
    if as_json:
        print(json.dumps(body, indent=1, default=str))
        return 0
    data = body.get("data") or {}
    meta = body.get("meta") or {}
    if meta.get("no_data"):
        print(f"no data yet{': ' + meta['reason'] if meta.get('reason') else ''}")
        return 0
    key, columns = TABLES.get(verb, (None, ()))
    rows = data.get(key) if key else None
    if isinstance(rows, list):
        print_table(rows, columns)
        return 0
    print(json.dumps(data, indent=1, default=str))  # compare and anything without a table shape
    return 0


def show_series(body, as_json):
    if as_json:
        print(json.dumps(body, indent=1, default=str))
        return 0
    data = body.get("data") or {}
    series, summary = data.get("series") or {}, data.get("summary") or {}
    direction = "higher is better" if data.get("higher_is_better") else "lower is better"
    print(f"{data.get('benchmark', data.get('benchmark_id'))}  {data.get('metric')}  ({direction})")
    if series:
        print(f"  series: arch={series.get('arch')} backend={series.get('backend')} job={series.get('source_job')}")
    if summary:
        print(
            f"  {summary.get('n_points')} points, median {_cell(summary.get('median_value'))}, "
            f"noise CV {_cell(summary.get('median_noise_cv_pct'))}%, last {summary.get('last_status')}"
        )
    points = data.get("points") or []
    rows = [
        {
            "ts": p.get("ts"),
            "value": p.get("value"),
            "verdict": (p.get("verdict") or {}).get("status"),
            "delta_pct": (p.get("verdict") or {}).get("delta_pct"),
            "n": p.get("n_samples"),
            "run_id": p.get("run_id"),
        }
        for p in points
    ]
    print_table(rows, ("ts", "value", "verdict", "delta_pct", "n", "run_id"))
    return 0


def execute(a):
    if a.verb == "series":
        # The id is derived before connecting, so a catalog mistake fails without a round trip.
        path = f"/api/v2/perf/benchmarks/{benchmark_id_of(a)}/series"
        return show_series(call(path, query_params(a)), a.json)
    return show(a.verb, call(ROUTES[a.verb], query_params(a)), a.json)


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench query", description=__doc__)
    ap.add_argument("verb", choices=sorted([*ROUTES, "series"]), help="which view to read")
    ap.add_argument("benchmark_id", nargs="?", help="series: the benchmark_id (or use --case)")
    ap.add_argument("--case", help="series: a case name, whose benchmark_id is derived from the catalog")
    cliutil.add_catalog_flags(ap)
    ap.add_argument("--arch")
    ap.add_argument("--component")
    ap.add_argument("--days", type=int, help="window, in days; each view has its own default and cap")
    ap.add_argument("--metric", help="series: which metric (default: the series' primary one)")
    ap.add_argument("--backend")
    ap.add_argument("--source-job", dest="source_job")
    ap.add_argument("--status", help="suites: fail, warn, no_baseline or pass")
    ap.add_argument("--kind", help="compare: derive, effect, runs or reference")
    ap.add_argument("-q", dest="q", help="suites: name substring")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cursor", help="continue a listing from meta.next_cursor")
    ap.add_argument("--json", action="store_true", help="the API body verbatim")
    return ap


def main(argv=None):
    return cliutil.run_main(execute, parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())

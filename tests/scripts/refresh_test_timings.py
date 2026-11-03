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

"""Regenerate .github/test_timings.json: median test minutes per config, from ClickHouse.

filter_configs.py --timings packs short configs into shared CI jobs from this file (see
tests/oot_framework/utils/test_layout.py). It is generated, never hand-edited: run this on a
schedule and review the data diff in a PR.

Each config's time is the median, over the window, of its summed test-case durations per run, on
first attempts. Keys are config paths relative to tests/configs/torch_spyre_tests/, the same as
.github/runner_overrides.yaml. Every config with at least one run is written with its run count;
the grouping trusts a median only from the policy's min_runs on, and runs a config with fewer (or
with no entry at all, i.e. new) as its own job until then.

Connection: by default the repo's ClickHouse client (CLICKHOUSE_HOST / CLICKHOUSE_USER /
CLICKHOUSE_PASS, see extensions/clickhouse-ingest). For a laptop, --http-url plus --token-file
uses ClickHouse's HTTP interface instead. The query is the same either way.

Usage:
  python tests/scripts/refresh_test_timings.py --out .github/test_timings.json
  python tests/scripts/refresh_test_timings.py --http-url https://host/ch/ --token-file ~/.ch-token
"""

import argparse
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

CONFIG_ROOT = Path("tests/configs/torch_spyre_tests")

# One row per (run, config): summed case time per run, then the median across runs. A config that
# ran alone reports <config stem>.xml; a bucket of short configs reports bucket_<name>.xml, so its
# cases are credited back to a config by their test file (the module in the JUnit classname), via
# {file_owner}, built from the configs on disk. The grouping never buckets a config whose test file
# another config also lists, so that lookup is unambiguous. A pod-level retry overwrites its report,
# so first attempts and retries are not double-counted.
QUERY = """
SELECT stem, round(quantile(0.5)(run_min), 3) AS p50_min, round(quantile(0.9)(run_min), 3) AS p90_min,
       count() AS runs, round(quantile(0.5)(cases)) AS cases
FROM (
  SELECT if(startsWith(src, 'bucket_'),
            {file_owner}[extract(c.classname, '(?:^|\\\\.)(test_[A-Za-z0-9_]+?)(?:__oot_wrapper)?(?:\\\\.|$)')],
            src) AS stem,
         r.run_id AS run_id, sum(r.duration_s) / 60 AS run_min, count() AS cases
  FROM (SELECT *, replaceRegexpOne(props['source_file'], '\\\\.xml$', '') AS src
        FROM {db}.test_case_runs
        WHERE component = 'torch-spyre' AND ts > now() - INTERVAL {days} DAY
          AND props['source_file'] != '' AND duration_s >= 0) AS r
  JOIN {db}.test_cases AS c ON c.test_case_id = r.test_case_id
  GROUP BY stem, run_id
  HAVING stem != ''
)
GROUP BY stem
FORMAT JSONEachRow
"""


def file_owners() -> dict:
    """{test file stem: config stem} for test files exactly one config lists."""
    import yaml

    owners: dict = {}
    for cfg in CONFIG_ROOT.rglob("*.yaml"):
        tsc = (yaml.safe_load(cfg.read_text()) or {}).get("test_suite_config") or {}
        for f in tsc.get("files") or []:
            owners.setdefault(Path(str(f.get("path", ""))).stem, set()).add(cfg.stem)
    return {f: next(iter(c)) for f, c in sorted(owners.items()) if len(c) == 1}


def sql_map(d: dict) -> str:
    """A ClickHouse map literal; the names are file stems, so a quote is the only hazard."""
    q = lambda x: "'" + x.replace("\\", "").replace("'", "") + "'"  # noqa: E731
    return "map(" + ", ".join(f"{q(k)}, {q(v)}" for k, v in d.items()) + ")"


def query_http(url: str, token: str, sql: str) -> list:
    req = urllib.request.Request(url, data=sql.encode(), method="POST")
    req.add_header(
        "Authorization",
        "Basic " + __import__("base64").b64encode(f"default:{token}".encode()).decode(),
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return [
            json.loads(line)
            for line in resp.read().decode().splitlines()
            if line.strip()
        ]


def query_client(sql: str) -> list:
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "extensions" / "clickhouse-ingest")
    )
    from spyre_clickhouse_ingest.client import ClickHouse  # noqa: E402

    client = ClickHouse.connect()
    result = client.query(sql.replace("FORMAT JSONEachRow", ""))
    cols = result.column_names
    return [dict(zip(cols, row)) for row in result.result_rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=".github/test_timings.json")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--db", default="spyre_v2")
    ap.add_argument(
        "--http-url", default="", help="ClickHouse HTTP endpoint (laptop use)"
    )
    ap.add_argument("--token-file", default="", help="file holding the HTTP password")
    args = ap.parse_args(argv)

    sql = QUERY.format(db=args.db, days=args.days, file_owner=sql_map(file_owners()))
    if args.http_url:
        token = Path(args.token_file).expanduser().read_text().strip()
        rows = query_http(args.http_url, token, sql)
    else:
        rows = query_client(sql)
    by_stem = {r["stem"]: r for r in rows}

    configs = {}
    for cfg in sorted(CONFIG_ROOT.rglob("*.yaml")):
        r = by_stem.get(cfg.stem)
        if r is None:
            continue
        configs[str(cfg.relative_to(CONFIG_ROOT))] = {
            "p50_min": float(r["p50_min"]),
            "p90_min": float(r["p90_min"]),
            "runs": int(r["runs"]),
            "cases": int(r["cases"]),
        }
    out = {
        "version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_days": args.days,
        "source": f"ClickHouse {args.db}.test_case_runs, median of per-run summed case minutes",
        "configs": configs,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out}: {len(configs)} configs with timings", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

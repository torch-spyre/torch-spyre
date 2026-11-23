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

"""Data layer behind the Perf Lab drill-down (overview -> suites -> series -> run -> environment -> steps -> spans,
plus compare and references). Every function takes a `run(sql, params) -> list[dict]` callable (see
`clickhouse_runner`) and returns {"data": ..., "meta": ...}; failures raise ApiError. Queries are read-only,
parameterised and bounded by a time window, a run id or a LIMIT. The contract is docs/perf-api.md."""

import base64
import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone

NIL = "00000000-0000-0000-0000-000000000000"
EXCL = "(SELECT record_type FROM benchmark_excluded_record_types)"
GATE_RANK = {"fail": 0, "warn": 1, "no_baseline": 2, "pass": 3}
RESERVED = ("step_index", "step_kind", "cold")
POLICY_LABEL = "cpu_pinning"
MAX_WINDOW_DAYS = {"overview": 30, "suites": 30, "series": 400, "effect": 120}


class ApiError(Exception):
    def __init__(self, status, code, message, **details):
        super().__init__(message)
        self.status, self.code, self.message, self.details = (
            status,
            code,
            message,
            details,
        )

    def body(self):
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


def clickhouse_runner(client):
    def run(sql, params=None):
        return list(client.query(sql, parameters=params or {}).named_results())

    return run


def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def naive(now):
    return (
        now_utc()
        if now is None
        else now.astimezone(timezone.utc).replace(tzinfo=None)
        if now.tzinfo
        else now
    )


def iso(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.replace(tzinfo=None).isoformat(sep="T", timespec="seconds") + "Z"
    s = str(v)
    return s.replace(" ", "T")[:19] + "Z" if len(s) >= 19 else s


def to_dt(v):
    return (
        None
        if v is None
        else v.replace(tzinfo=None)
        if isinstance(v, datetime)
        else parse_time(iso(v), "ts")
    )


def ch_time(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_time(v, name):
    if v is None or isinstance(v, datetime):
        return v.replace(tzinfo=None) if v else None
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError as e:
        raise ApiError(
            400,
            "bad_parameter",
            f"{name} must be ISO 8601",
            parameter=name,
            value=str(v),
        ) from e
    return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d


def window(now, t_from, t_to, default_days, max_days, name="window"):
    end = parse_time(t_to, "to") or now
    start = parse_time(t_from, "from") or end - timedelta(days=default_days)
    if start >= end:
        raise ApiError(
            400,
            "bad_window",
            "from must be before to",
            **{"from": iso(start), "to": iso(end)},
        )
    if end - start > timedelta(days=max_days):
        raise ApiError(
            400,
            "window_too_large",
            f"{name} spans more than {max_days} days",
            max_days=max_days,
        )
    return start, end


def bounded(v, name, default, lo, hi):
    if v is None:
        return default
    try:
        n = int(v)
    except (TypeError, ValueError) as e:
        raise ApiError(
            400, "bad_parameter", f"{name} must be an integer", parameter=name
        ) from e
    if not lo <= n <= hi:
        raise ApiError(
            400, "bad_parameter", f"{name} must be {lo}..{hi}", parameter=name, value=n
        )
    return n


def uuid_arg(v, name):
    s = str(v or "").lower()
    ok = len(s) == 36 and all(c in "0123456789abcdef-" for c in s) and s.count("-") == 4
    if not ok:
        raise ApiError(
            400, "bad_parameter", f"{name} must be a UUID", parameter=name, value=str(v)
        )
    return s


def encode_cursor(obj):
    return (
        base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )


def decode_cursor(c):
    if not c:
        return {}
    try:
        out = json.loads(base64.urlsafe_b64decode(c + "=" * (-len(c) % 4)))
    except ValueError as e:
        raise ApiError(400, "bad_cursor", "cursor is not valid") from e
    if not isinstance(out, dict):
        raise ApiError(400, "bad_cursor", "cursor is not valid")
    return out


def envelope(data, now, cache_s, **meta):
    body = json.dumps(data, sort_keys=True, default=str)
    return {
        "data": data,
        "meta": {"as_of": iso(now), "cache": {"max_age_s": cache_s, "etag": hashlib.sha1(body.encode()).hexdigest()[:16]}, **meta},
    }  # fmt: skip


def num(v):
    return None if v is None else float(v)


def median(xs):
    return statistics.median(xs) if xs else None


def cv_pct(xs):
    return (
        statistics.stdev(xs) / abs(statistics.mean(xs)) * 100
        if len(xs) > 1 and statistics.mean(xs)
        else None
    )


def rnd(v, n=4):
    return None if v is None else round(v, n)


# ---------------------------------------------------------------------------------------------------------------
# shared reads


def latest_verdicts(extra="", cols=""):
    """Stored verdicts at their newest policy_version, one row per run x series x metric (62's benchmark_metric_verdicts)."""
    return f"""
SELECT run_id, benchmark_id, component, backend, arch, source_job, metric,
       argMax(ts, evaluated_at) AS ts, argMax(tier, evaluated_at) AS tier,
       argMax(baseline_run_ids, evaluated_at) AS baseline_run_ids, argMax(n_baseline, evaluated_at) AS n_baseline,
       argMax(n_current, evaluated_at) AS n_current, argMax(baseline_value, evaluated_at) AS baseline_value,
       argMax(current_value, evaluated_at) AS current_value, argMax(delta_pct, evaluated_at) AS delta_pct,
       argMax(worse_pct, evaluated_at) AS worse_pct, argMax(score, evaluated_at) AS score,
       argMax(floor_pct, evaluated_at) AS floor_pct, argMax(status, evaluated_at) AS status,
       argMax(reason, evaluated_at) AS reason, argMax(policy_version, evaluated_at) AS policy_version{cols}
FROM (SELECT * FROM benchmark_metric_verdicts WHERE ts >= {{t_from:DateTime}} AND ts < {{t_to:DateTime}} {extra})
GROUP BY run_id, benchmark_id, component, backend, arch, source_job, metric"""


def gate_rows(extra="", cols=""):
    """v_benchmark_gate's rollup over the stored verdicts: one row per run x benchmark x backend x arch x source_job."""
    return f"""
SELECT run_id, benchmark_id, component, backend, arch, source_job, max(ts) AS gts,
       multiIf(countIf(status = 'regressed' AND tier = 'blocking') > 0, 'fail',
               countIf(status IN ('regressed', 'suspected') AND tier IN ('blocking', 'advisory')) > 0, 'warn',
               countIf(status = 'no_baseline') = count(), 'no_baseline', 'pass') AS gate,
       countIf(status = 'improved') > 0 AS improved,
       groupArrayIf(metric, status = 'regressed' AND tier = 'blocking') AS regressed_metrics,
       groupArrayIf(metric, status IN ('regressed', 'suspected') AND tier IN ('blocking', 'advisory')
                            AND NOT (status = 'regressed' AND tier = 'blocking')) AS warn_metrics,
       argMaxIf(metric, worse_pct, status IN ('regressed', 'suspected')) AS primary_metric,
       maxIf(worse_pct, status IN ('regressed', 'suspected')) AS worst_delta_pct,
       min(n_baseline) AS min_baseline
FROM ({latest_verdicts(extra, cols)})
GROUP BY run_id, benchmark_id, component, backend, arch, source_job"""


def anchored_read(run, sql, params):
    """Rows of the anchored views (schema 88), or None when the database predates them."""
    try:
        return run(sql, params)
    except (
        Exception
    ) as e:  # clickhouse_connect and chdb word the missing-object error differently
        if "v_benchmark_anchor" in str(e) and any(
            m in str(e)
            for m in (
                "UNKNOWN_TABLE",
                "Unknown table",
                "doesn't exist",
                "does not exist",
            )
        ):
            return None
        raise


def anchor_status(r):
    """The point-level anchored verdict from a v_benchmark_anchored_verdicts row."""
    return {
        "status": r["status"], "reason": r["reason"], "anchor_value": rnd(num(r["anchor_value"])), "delta_pct": rnd(num(r["delta_pct"]), 1),
        "worse_pct": rnd(num(r["worse_pct"]), 1), "floor_pct": rnd(num(r["floor_pct"]), 1), "since_ts": iso(r["since_ts"]) if r["since_ts"] else None,
        "n_exceeding_runs": int(r["n_exceeding_runs"] or 0), "days_exceeding": int(r["days_exceeding"] or 0), "version": r["version"] or None,
    }  # fmt: skip


def invalidations(run, run_ids):
    """{run_id: [hit]} from v_invalidated_runs (run, run_benchmark, node/device/placement scopes, retracted uploads)."""
    if not run_ids:
        return {}
    rows = run(
        """SELECT i.run_id AS run_id, i.benchmark_id AS benchmark_id, i.repeat AS repeat, i.state AS state,
                  i.reason_code AS reason_code, i.invalidation_id AS invalidation_id,
                  s.reason AS reason, s.scope_kind AS scope_kind, s.actor AS actor, s.updated_at AS updated_at
           FROM v_invalidated_runs AS i
           LEFT JOIN v_invalidation_state AS s ON s.invalidation_id = i.invalidation_id
           WHERE i.run_id IN {ids:Array(UUID)}
           ORDER BY i.run_id, i.invalidation_id LIMIT 20000""",
        {"ids": sorted(run_ids)},
    )
    out = {}
    for r in rows:
        out.setdefault(str(r["run_id"]), []).append({
            "invalidation_id": str(r["invalidation_id"]), "state": r["state"], "reason_code": r["reason_code"],
            "reason": r.get("reason") or "", "scope": r.get("scope_kind") or "retracted_upload",
            "benchmark_id": None if r["benchmark_id"] is None else str(r["benchmark_id"]),
            "repeat": None if r["repeat"] is None else int(r["repeat"]), "actor": r.get("actor") or "",
            "updated_at": iso(r.get("updated_at")),
        })  # fmt: skip
    return out


def covers(hit, benchmark_id=None, repeat=None):
    return (hit["benchmark_id"] in (None, benchmark_id)) and (
        hit["repeat"] in (None, repeat)
    )


def invalid_repeat(hits, benchmark_id, repeat):
    return any(
        h["state"] == "invalidated" and covers(h, benchmark_id, repeat) for h in hits
    )


def flag(v, default=True):
    return default if v is None else str(v).lower() not in ("0", "false", "no", "off")


def run_state(hits, repeats=None):
    """invalidated (whole run, or every repeat of it), partial (a benchmark or some repeats), suspect, or valid."""
    live = [h for h in hits if h["state"] == "invalidated"]
    if any(h["benchmark_id"] is None and h["repeat"] is None for h in live):
        return "invalidated"
    if repeats and repeats <= {
        h["repeat"]
        for h in live
        if h["benchmark_id"] is None and h["repeat"] is not None
    }:
        return "invalidated"
    if live:
        return "partial"
    return "suspect" if any(h["state"] == "suspect" for h in hits) else "valid"


def subjects(run, run_ids):
    """{run_id: subject} for performance runs: the latest published verdict's artifact, its identity and newest tag."""
    if not run_ids:
        return {}
    links = run(
        """SELECT run_id, argMax(artifact_id, (ts, audit_timestamp)) AS artifact_id, argMax(arch, (ts, audit_timestamp)) AS arch,
                  argMax(state, (ts, audit_timestamp)) AS state, max(ts) AS result_ts, argMax(props['run_url'], (ts, audit_timestamp)) AS run_url
           FROM v_artifact_results_published
           WHERE result_kind = 'performance' AND run_id IN {ids:Array(UUID)}
           GROUP BY run_id LIMIT 100000""",
        {"ids": sorted(run_ids)},
    )
    aids = sorted({str(r["artifact_id"]) for r in links})
    arts = {}
    if aids:
        for a in run(
            """SELECT artifact_id, component, artifact_name, kind, props['id12'] AS id12, props['build'] AS build
               FROM v_artifacts WHERE artifact_id IN {ids:Array(UUID)} LIMIT 100000""",
            {"ids": aids},
        ):
            arts[str(a["artifact_id"])] = a
        for t in run(
            """SELECT artifact_id, argMax(tag, ts) AS latest_tag, argMax(tag_family, ts) AS latest_family
               FROM artifact_tags WHERE artifact_id IN {ids:Array(UUID)} AND tag_family != 'reference'
               GROUP BY artifact_id LIMIT 100000""",
            {"ids": aids},
        ):
            arts.setdefault(str(t["artifact_id"]), {}).update(
                tag=t["latest_tag"], tag_family=t["latest_family"]
            )
    out = {}
    for r in links:
        a = arts.get(str(r["artifact_id"]), {})
        out[str(r["run_id"])] = {
            "artifact_id": str(r["artifact_id"]), "name": a.get("artifact_name"), "component": a.get("component"),
            "id12": a.get("id12") or str(r["artifact_id"]).replace("-", "")[:12], "build": a.get("build") or None,
            "tag": a.get("tag") or None, "tag_family": a.get("tag_family") or None, "arch": r["arch"],
            "verdict_state": r["state"], "result_ts": iso(r["result_ts"]), "run_url": r.get("run_url") or None,
        }  # fmt: skip
    return out


def policy(run):
    return {
        r["metric"]: r
        for r in run(
            "SELECT metric, tier, higher_is_better, floor_pct FROM benchmark_metric_policy LIMIT 1000"
        )
    }


def higher_is_better(pol, metric):
    p = pol.get(metric)
    if p:
        return bool(p["higher_is_better"])
    return (
        "throughput" in metric
        or metric.endswith("_per_second")
        or metric == "pt_util_percent"
    )


def bench_meta(run, ids):
    if not ids:
        return {}
    rows = run(
        """SELECT benchmark_id, any(component) AS component, any(name) AS name, any(tags) AS tags, any(props) AS props
           FROM benchmarks WHERE benchmark_id IN {ids:Array(UUID)} GROUP BY benchmark_id LIMIT 100000""",
        {"ids": sorted(ids)},
    )
    return {str(r["benchmark_id"]): r for r in rows}


# ---------------------------------------------------------------------------------------------------------------
# L0 overview


def overview(
    run,
    now=None,
    arch=None,
    component=None,
    days=7,
    lookback_days=30,
    stale_after_hours=36,
    as_of=None,
    **_,
):
    now = parse_time(as_of, "as_of") or naive(now)
    days = bounded(days, "days", 7, 1, MAX_WINDOW_DAYS["overview"])
    lookback = bounded(lookback_days, "lookback_days", 30, days, 90)
    stale_h = bounded(stale_after_hours, "stale_after_hours", 36, 1, 24 * 14)
    win_from = now - timedelta(days=days)
    flt = (" AND arch = {arch:String}" if arch else "") + (
        " AND component = {component:String}" if component else ""
    )
    p = {
        "t_from": ch_time(now - timedelta(days=lookback)),
        "t_to": ch_time(now),
        "w_from": ch_time(win_from),
        "stale": stale_h,
    }
    if arch:
        p["arch"] = arch
    if component:
        p["component"] = component
    # series level: the newest valid run per (component, arch, benchmark, backend, source_job) decides the tile counts
    series = run(
        f"""WITH
              inv AS (SELECT DISTINCT run_id FROM v_invalidated_runs
                      WHERE state = 'invalidated' AND benchmark_id IS NULL AND repeat IS NULL AND run_id IS NOT NULL),
              g AS ({gate_rows(flt)})
            SELECT arch, component,
                   count() AS n_known,
                   countIf(last_ts >= {{w_from:DateTime}}) AS n_in_window,
                   countIf(last_ts >= {{w_from:DateTime}} AND s_gate = 'fail') AS n_fail,
                   countIf(last_ts >= {{w_from:DateTime}} AND s_gate = 'warn') AS n_warn,
                   countIf(last_ts >= {{w_from:DateTime}} AND s_gate = 'pass') AS n_pass,
                   countIf(last_ts >= {{w_from:DateTime}} AND s_gate = 'no_baseline') AS n_no_baseline,
                   countIf(last_ts >= {{w_from:DateTime}} AND s_gate = 'pass' AND s_improved) AS n_improved,
                   max(last_ts) AS last_run_ts
            FROM (SELECT arch, component, benchmark_id, backend, source_job, max(gts) AS last_ts,
                         argMax(gate, gts) AS s_gate, argMax(improved, gts) AS s_improved
                  FROM g WHERE run_id NOT IN inv
                  GROUP BY arch, component, benchmark_id, backend, source_job)
            GROUP BY arch, component ORDER BY arch, component LIMIT 500""",
        p,
    )
    runs = run(
        f"""SELECT arch, component, uniqExactIf(run_id, ts >= {{w_from:DateTime}}) AS n_runs
            FROM ({latest_verdicts(flt)}) GROUP BY arch, component LIMIT 500""",
        p,
    )
    # schema 85 drops an invalidated run's samples, so such runs have no verdict rows: count them from the runs themselves
    bad = run(
        f"""SELECT if(props['arch'] != '', props['arch'], props['platform']) AS run_arch, component AS run_component,
                   uniqExact(run_id) AS n
            FROM benchmark_runs
            WHERE ts >= {{w_from:DateTime}} AND ts < {{t_to:DateTime}} {"AND component = {component:String}" if component else ""}
              AND run_id IN (SELECT run_id FROM v_invalidated_runs
                             WHERE state = 'invalidated' AND benchmark_id IS NULL AND repeat IS NULL AND run_id IS NOT NULL)
            GROUP BY run_arch, run_component LIMIT 500""",
        p,
    )
    bmap = {(r["run_arch"], r["run_component"]): int(r["n"]) for r in bad}
    anchored, persistent, anote = _overview_anchored(run, p, flt)
    rmap = {(r["arch"], r["component"]): r for r in runs}
    tiles = []
    for s in series:
        r = rmap.get((s["arch"], s["component"]), {})
        last = s["last_run_ts"]
        age_h = (
            None
            if last is None
            else round((now - parse_time(iso(last), "ts")).total_seconds() / 3600, 1)
        )
        tiles.append({
            "arch": s["arch"], "component": s["component"],
            "verdicts": {k: int(s[f"n_{k}"]) for k in ("fail", "warn", "pass", "no_baseline", "improved")},
            "benchmarks_in_window": int(s["n_in_window"]),
            "freshness": {"last_run_ts": iso(last), "age_hours": age_h, "stale": age_h is None or age_h > stale_h},
            "coverage": {"seen_in_window": int(s["n_in_window"]), "known_in_lookback": int(s["n_known"]),
                         "ratio": rnd(int(s["n_in_window"]) / int(s["n_known"]), 3) if int(s["n_known"]) else None},
            "runs_in_window": int(r.get("n_runs", 0)), "invalidated_runs_in_window": bmap.get((s["arch"], s["component"]), 0),
            "anchored": anchored.get((s["arch"], s["component"]), dict(ANCHORED_ZERO)),
        })  # fmt: skip
    return envelope({"tiles": tiles, "persistent_regressions": persistent}, now, 60,
                    window={"from": iso(win_from), "to": iso(now), "lookback_days": lookback}, no_data=not tiles,
                    **({"notes": anote} if anote else {}))  # fmt: skip


ANCHORED_ZERO = {
    "regressed": 0,
    "suspected": 0,
    "improved": 0,
    "unchanged": 0,
    "no_anchor": 0,
}


def _overview_anchored(run, p, flt, limit=50):
    """Per tile: series by their newest anchored verdict (blocking and advisory metrics only), and the persistent regressions."""
    rows = anchored_read(
        run,
        f"""SELECT arch, component, benchmark_id, any(name) AS name, backend, source_job, metric, any(tier) AS tier,
                   max(run_ts) AS last_ts, argMax(status, (run_ts, run_id)) AS st, argMax(worse_pct, (run_ts, run_id)) AS worse_pct,
                   argMax(since_ts, (run_ts, run_id)) AS since_ts, argMax(n_exceeding_runs, (run_ts, run_id)) AS n_exceeding_runs,
                   argMax(days_exceeding, (run_ts, run_id)) AS days_exceeding, argMax(version, (run_ts, run_id)) AS version
            FROM v_benchmark_anchored_verdicts
            WHERE run_ts >= {{t_from:DateTime}} AND run_ts < {{t_to:DateTime}} {flt}
            GROUP BY arch, component, benchmark_id, backend, source_job, metric
            HAVING last_ts >= {{w_from:DateTime}} AND tier IN ('blocking', 'advisory')
            ORDER BY arch, component, benchmark_id, backend, source_job, metric LIMIT 20000""",
        p,
    )
    if rows is None:
        return {}, [], ["schema 88 (anchored verdicts) is not applied"]
    note = (
        ["anchored counts are truncated at 20000 series x metrics"]
        if len(rows) >= 20000
        else None
    )
    order = ("regressed", "suspected", "improved", "unchanged", "no_anchor")
    series = {}
    for r in rows:
        k = (
            r["arch"],
            r["component"],
            str(r["benchmark_id"]),
            r["backend"],
            r["source_job"],
        )
        series.setdefault(k, []).append(r)
    tiles, persistent = {}, []
    for k, rs in series.items():
        worst = min((r["st"] for r in rs), key=order.index)
        tiles.setdefault((k[0], k[1]), dict(ANCHORED_ZERO))[worst] += 1
        for r in rs:
            if r["st"] == "regressed":
                persistent.append({
                    "arch": k[0], "component": k[1], "benchmark_id": k[2], "name": r["name"], "backend": k[3], "source_job": k[4],
                    "metric": r["metric"], "tier": r["tier"], "worse_pct": rnd(num(r["worse_pct"]), 1), "since_ts": iso(r["since_ts"]),
                    "n_exceeding_runs": int(r["n_exceeding_runs"]), "days_exceeding": int(r["days_exceeding"]), "anchor_version": r["version"],
                })  # fmt: skip
    persistent.sort(key=lambda x: (-x["days_exceeding"], x["name"] or "", x["metric"]))
    return tiles, persistent[:limit], note


# ---------------------------------------------------------------------------------------------------------------
# L1 suites


def suites(run, now=None, arch=None, component=None, days=7, group_by="suite", status=None, q=None, limit=100, cursor=None,
           as_of=None, **_):  # fmt: skip
    cur = decode_cursor(cursor)
    now = (
        parse_time(cur.get("to"), "cursor") or parse_time(as_of, "as_of") or naive(now)
    )
    days = bounded(days, "days", 7, 1, MAX_WINDOW_DAYS["suites"])
    limit = bounded(limit, "limit", 100, 1, 500)
    if group_by not in ("suite", "tier", "none"):
        raise ApiError(
            400,
            "bad_parameter",
            "group_by must be suite, tier or none",
            parameter="group_by",
        )
    if status and status not in GATE_RANK:
        raise ApiError(
            400,
            "bad_parameter",
            f"status must be one of {sorted(GATE_RANK)}",
            parameter="status",
        )
    flt = (" AND arch = {arch:String}" if arch else "") + (
        " AND component = {component:String}" if component else ""
    )
    p = {"t_from": ch_time(now - timedelta(days=days)), "t_to": ch_time(now)}
    for k, v in (("arch", arch), ("component", component)):
        if v:
            p[k] = v
    rows = run(
        f"""WITH inv AS (SELECT DISTINCT run_id FROM v_invalidated_runs
                         WHERE state = 'invalidated' AND benchmark_id IS NULL AND repeat IS NULL AND run_id IS NOT NULL)
            SELECT component, arch, benchmark_id, backend, source_job,
                   argMax(run_id, gts) AS l_run_id, max(gts) AS l_ts, argMax(gate, gts) AS l_gate, argMax(improved, gts) AS l_improved,
                   argMax(primary_metric, gts) AS l_primary, argMax(worst_delta_pct, gts) AS l_worst,
                   argMax(regressed_metrics, gts) AS l_regressed, argMax(warn_metrics, gts) AS l_warn,
                   argMax(min_baseline, gts) AS l_min_baseline, count() AS l_runs
            FROM ({gate_rows(flt)}) WHERE run_id NOT IN inv
            GROUP BY component, arch, benchmark_id, backend, source_job LIMIT 20000""",
        p,
    )
    meta = bench_meta(run, {str(r["benchmark_id"]) for r in rows})
    items = []
    for r in rows:
        b = meta.get(str(r["benchmark_id"]), {})
        name = b.get("name") or str(r["benchmark_id"])
        if q and q.lower() not in name.lower():
            continue
        if status and r["l_gate"] != status:
            continue
        items.append({
            "benchmark_id": str(r["benchmark_id"]), "name": name, "component": r["component"], "arch": r["arch"],
            "backend": r["backend"], "source_job": r["source_job"], "run_id": str(r["l_run_id"]), "run_ts": iso(r["l_ts"]),
            "verdict": r["l_gate"], "improved": bool(r["l_improved"]), "primary_metric": r["l_primary"] or None,
            "worst_delta_pct": rnd(num(r["l_worst"]), 1) if r["l_gate"] in ("fail", "warn") else None,
            "regressed_metrics": list(r["l_regressed"]), "warn_metrics": list(r["l_warn"]),
            "baseline_runs": int(r["l_min_baseline"]), "runs_in_window": int(r["l_runs"]),
            "record_type": (b.get("props") or {}).get("record_type") or None, "tags": list(b.get("tags") or []),
        })  # fmt: skip
    items.sort(
        key=lambda i: (
            GATE_RANK[i["verdict"]],
            i["name"],
            i["benchmark_id"],
            i["arch"],
            i["backend"],
        )
    )
    total = len(items)
    off = bounded(cur.get("offset"), "cursor", 0, 0, 10_000_000)
    page = items[off : off + limit]
    _decorate_suites(run, page, group_by, now, days)
    counts = {k: sum(1 for i in items if i["verdict"] == k) for k in GATE_RANK}
    nxt = (
        encode_cursor({"to": ch_time(now), "offset": off + limit})
        if off + limit < total
        else None
    )
    return envelope({"items": page, "counts": counts}, now, 60, total=total, next_cursor=nxt, group_by=group_by,
                    window={"from": p["t_from"], "to": p["t_to"]}, no_data=total == 0)  # fmt: skip


def _decorate_suites(run, page, group_by, now, days):
    """Group value (a run prop as of the run) and the last 10 verdicts per series, for the page only."""
    if not page:
        return
    for comp in sorted({i["component"] for i in page}):
        sub = [i for i in page if i["component"] == comp]
        props = run(
            """SELECT run_id, benchmark_id, argMax(props, ts) AS props FROM benchmark_runs
               WHERE component = {c:String} AND run_id IN {runs:Array(UUID)} AND benchmark_id IN {bids:Array(UUID)}
               GROUP BY run_id, benchmark_id LIMIT 100000""",
            {
                "c": comp,
                "runs": sorted({i["run_id"] for i in sub}),
                "bids": sorted({i["benchmark_id"] for i in sub}),
            },
        )
        pm = {(str(r["run_id"]), str(r["benchmark_id"])): r["props"] for r in props}
        hist = run(
            f"""SELECT benchmark_id, arch, backend, source_job, arraySlice(arrayReverse(arraySort(x -> x.1, groupArray((gts, gate)))), 1, 10) AS h
                FROM ({gate_rows(" AND component = {c:String} AND benchmark_id IN {bids:Array(UUID)}")})
                WHERE run_id NOT IN (SELECT run_id FROM v_invalidated_runs
                                     WHERE state = 'invalidated' AND benchmark_id IS NULL AND repeat IS NULL AND run_id IS NOT NULL)
                GROUP BY benchmark_id, arch, backend, source_job LIMIT 100000""",
            {"c": comp, "bids": sorted({i["benchmark_id"] for i in sub}), "t_from": ch_time(now - timedelta(days=max(days, 30))),
             "t_to": ch_time(now)},
        )  # fmt: skip
        hm = {
            (str(r["benchmark_id"]), r["arch"], r["backend"], r["source_job"]): r["h"]
            for r in hist
        }
        for i in sub:
            pr = pm.get((i["run_id"], i["benchmark_id"]), {})
            i["group"] = (
                None
                if group_by == "none"
                else (
                    pr.get("run.suite.name")
                    if group_by == "suite"
                    else pr.get("run.tier")
                )
                or None
            )
            i["purpose"] = pr.get("run.purpose") or None
            i["history"] = [
                {"ts": iso(t), "verdict": g}
                for t, g in hm.get(
                    (i["benchmark_id"], i["arch"], i["backend"], i["source_job"]), []
                )
            ]


# ---------------------------------------------------------------------------------------------------------------
# L2 series


def series(run, benchmark_id, now=None, metric=None, arch=None, backend=None, source_job=None, t_from=None, t_to=None,
           days=90, limit=200, include_invalid=True, reference_key=None, **_):  # fmt: skip
    now = naive(now)
    bid = uuid_arg(benchmark_id, "benchmark_id")
    start, end = window(
        now,
        t_from,
        t_to,
        bounded(days, "days", 90, 1, MAX_WINDOW_DAYS["series"]),
        MAX_WINDOW_DAYS["series"],
    )
    limit = bounded(limit, "limit", 200, 1, 1000)
    bm = bench_meta(run, {bid}).get(bid)
    if not bm:
        raise ApiError(404, "unknown_benchmark", "no such benchmark", benchmark_id=bid)
    rtype = (bm["props"] or {}).get("record_type", "")
    if rtype in {
        "reference",
        "step_trace",
        "span",
        "trace_phase",
        "trace_step",
        "trace_kernel",
        "compile_pass",
    }:
        raise ApiError(
            422,
            "not_a_measurement_series",
            f"record_type {rtype} has no verdict series",
            record_type=rtype,
        )
    comp = bm["component"]
    p = {"t_from": ch_time(start), "t_to": ch_time(end), "c": comp, "bid": bid}
    opts = run(
        f"""SELECT arch, backend, source_job, metric, count() AS n, max(ts) AS last_ts
            FROM ({latest_verdicts(" AND component = {c:String} AND benchmark_id = {bid:UUID}")})
            GROUP BY arch, backend, source_job, metric ORDER BY n DESC, arch, backend, source_job, metric LIMIT 200""",
        p,
    )
    pol = policy(run)
    tier_rank = {"blocking": 0, "advisory": 1, "informational": 2}
    options = [{"arch": o["arch"], "backend": o["backend"], "source_job": o["source_job"], "metric": o["metric"],
                "runs": int(o["n"]), "last_ts": iso(o["last_ts"])} for o in opts]  # fmt: skip
    if not options:
        return envelope({"benchmark": _bench_head(bm, bid), "points": [], "subject_changes": [], "env_changes": []}, now, 120,
                        no_data=True, series_options=[], window={"from": iso(start), "to": iso(end)})  # fmt: skip
    pick = [o for o in options if (not arch or o["arch"] == arch) and (not backend or o["backend"] == backend)
            and (not source_job or o["source_job"] == source_job) and (not metric or o["metric"] == metric)]  # fmt: skip
    if not pick:
        raise ApiError(
            404,
            "unknown_series",
            "no verdict series matches the filters",
            options=options[:20],
        )
    if not metric:
        pick.sort(
            key=lambda o: (
                tier_rank.get(
                    (pol.get(o["metric"]) or {}).get("tier", "informational"), 2
                ),
                -o["runs"],
                o["metric"],
            )
        )
    sel = pick[0]
    sel_arch, sel_backend, sel_job, sel_metric = (
        sel["arch"],
        sel["backend"],
        sel["source_job"],
        sel["metric"],
    )
    vrows = run(
        f"""{latest_verdicts(" AND component = {c:String} AND benchmark_id = {bid:UUID} AND arch = {a:String} AND backend = {b:String} AND source_job = {j:String} AND metric = {m:String}")}
            ORDER BY ts DESC, run_id LIMIT {{lim:UInt32}}""",
        {
            **p,
            "a": sel_arch,
            "b": sel_backend,
            "j": sel_job,
            "m": sel_metric,
            "lim": limit,
        },
    )
    vmap = {str(v["run_id"]): v for v in vrows}
    # the measurements behind each point, for the same runs; a run without a verdict row yet is shown as pending
    meas = run(
        """SELECT run_id, toUInt16OrZero(props['run.repeat']) AS repeat, min(ts) AS first_ts, any(props) AS rprops,
                  arrayFlatten(groupArray(measurements[{m:String}])) AS samples
           FROM benchmark_runs
           WHERE component = {c:String} AND benchmark_id = {bid:UUID} AND backend = {b:String}
             AND ts >= {t_from:DateTime} AND ts < {t_to:DateTime} AND mapContains(measurements, {m:String})
             AND props['run.ref_only'] != '1' AND props['run.record_type'] != 'reference'
           GROUP BY run_id, repeat ORDER BY first_ts DESC, run_id LIMIT 20000""",
        {**p, "b": sel_backend, "m": sel_metric},
    )
    by_run = {}
    for r in meas:
        by_run.setdefault(str(r["run_id"]), []).append(r)
    ids = set(vmap) | {
        k
        for k, rs in by_run.items()
        if _row_job(rs[0]["rprops"]) == sel_job and _row_arch(rs[0]["rprops"], sel_arch)
    }
    ids &= set(by_run)  # a point needs its measurements
    ordered = sorted(
        ids, key=lambda k: (min(to_dt(r["first_ts"]) for r in by_run[k]), k)
    )[-limit:]
    inv = invalidations(run, ordered)
    subj = subjects(run, ordered)
    points, subject_changes, env_changes = [], [], []
    prev_subj = prev_env = None
    hib = higher_is_better(pol, sel_metric)
    reps_of = {k: {r["repeat"] for r in rs} for k, rs in by_run.items()}
    invalid_ids = {
        k for k, h in inv.items() if run_state(h, reps_of.get(k)) == "invalidated"
    }
    base_ids = {str(b) for v in vmap.values() for b in v["baseline_run_ids"]} - set(
        ordered
    )
    for k, h in invalidations(run, base_ids).items():
        if run_state(h) == "invalidated":
            invalid_ids.add(k)
    for rid in ordered:
        all_rs = sorted(by_run[rid], key=lambda r: r["repeat"])
        rs = [
            r for r in all_rs if not invalid_repeat(inv.get(rid, []), bid, r["repeat"])
        ]
        samples = [float(x) for r in rs for x in r["samples"] if x != 0]
        reps = [median([float(x) for x in r["samples"] if x != 0]) for r in rs]
        props = all_rs[0]["rprops"]
        v = vmap.get(rid)
        state = run_state(inv.get(rid, []), reps_of.get(rid))
        env_id = props.get("env.id") or None
        s = subj.get(rid)
        pt = {
            "run_id": rid, "ts": iso(parse_time(props.get("run_utc"), "run_utc") if props.get("run_utc") else all_rs[0]["first_ts"]),
            "value": rnd(median(samples)), "n_samples": len(samples), "n_repeats": len(rs),
            "repeat_medians": [rnd(x) for x in reps if x is not None], "noise_cv_pct": rnd(cv_pct(samples), 2),
            "min": rnd(min(samples)) if samples else None, "max": rnd(max(samples)) if samples else None,
            "subject_id": s["artifact_id"] if s else None, "env_id": env_id, "purpose": props.get("run.purpose") or None,
            "validity": props.get("run.validity") or None, "invalidation_state": state, "invalidations": inv.get(rid, []),
            "verdict": None,
        }  # fmt: skip
        if v:
            base = num(v["baseline_value"])
            fl = num(v["floor_pct"])
            pt["verdict"] = {
                "status": v["status"], "reason": v["reason"] or None, "tier": v["tier"], "baseline_value": rnd(base),
                "delta_pct": rnd(num(v["delta_pct"]), 1), "worse_pct": rnd(num(v["worse_pct"]), 1),
                "floor_pct": rnd(fl, 1), "n_baseline": int(v["n_baseline"]), "score": rnd(num(v["score"])),
                "band": None if not base or v["status"] == "no_baseline" else
                    {"low": rnd(base * (1 - fl / 100)), "high": rnd(base * (1 + fl / 100)), "center": rnd(base)},
                "baseline_includes_invalidated": sorted(str(b) for b in v["baseline_run_ids"] if str(b) in invalid_ids),
                "policy_version": v["policy_version"],
            }  # fmt: skip
            if state == "invalidated":
                pt["verdict"]["excluded"] = True
        else:
            pt["verdict"] = (
                {"status": "excluded", "reason": "invalidated", "excluded": True}
                if state == "invalidated"
                else {
                    "status": "pending",
                    "reason": "no stored verdict yet (refresh runs every 30 minutes)",
                }
            )
        if s and s["artifact_id"] != prev_subj and prev_subj is not None:
            subject_changes.append(
                {
                    "at_run_id": rid,
                    "ts": pt["ts"],
                    "from": prev_subj,
                    "to": s["artifact_id"],
                    "tag": s["tag"],
                    "id12": s["id12"],
                }
            )
        if env_id != prev_env and prev_env is not None:
            env_changes.append(
                {"at_run_id": rid, "ts": pt["ts"], "from": prev_env, "to": env_id}
            )
        prev_subj = s["artifact_id"] if s else prev_subj
        prev_env = env_id if env_id else prev_env
        pt["subject_changed"] = bool(
            subject_changes and subject_changes[-1]["at_run_id"] == rid
        )
        pt["env_changed"] = bool(env_changes and env_changes[-1]["at_run_id"] == rid)
        pt["subject"] = s
        pt.update(
            arch=sel_arch,
            backend=sel_backend,
            component=comp,
            artifact_id=pt["subject_id"],
            artifact_name=(s or {}).get("name"),
            tags=[t for t in [(s or {}).get("tag")] if t],
            run_url=(s or {}).get("run_url"),
            highlight=False,
        )
        points.append(pt)
    anchor_line, anchor_notes = _attach_anchor(
        run, points, p, comp, bid, sel, start, end
    )
    if not flag(include_invalid):
        points = [pt for pt in points if pt["invalidation_state"] != "invalidated"]
    cvs = [pt["noise_cv_pct"] for pt in points if pt["noise_cv_pct"] is not None]
    head = _bench_head(bm, bid)
    data = {
        "benchmark_id": bid,
        "metric": sel_metric,
        "higher_is_better": hib,
        "count": len(points),
        "benchmark": head,
        "series": {
            "arch": sel_arch,
            "backend": sel_backend,
            "source_job": sel_job,
            "metric": sel_metric,
            "tier": (pol.get(sel_metric) or {}).get("tier", "informational"),
            "higher_is_better": hib,
        },  # fmt: skip
        "points": points,
        "subject_changes": subject_changes,
        "env_changes": env_changes,
        "summary": {
            "n_points": len(points),
            "median_value": rnd(
                median([x["value"] for x in points if x["value"] is not None])
            ),
            "median_noise_cv_pct": rnd(median(cvs), 2),
            "last_status": (points[-1]["verdict"] or {}).get("status")
            if points
            else None,
            "last_anchored_status": ((anchor_line or {}).get("latest") or {}).get(
                "status"
            ),
        },  # fmt: skip
        "anchor": anchor_line,
    }
    if reference_key:
        try:
            data["reference"] = _reference_line(run, reference_key, bid, sel_metric)
        except ApiError as e:
            data["reference"] = {"key": reference_key, "error": e.code}
    return envelope(
        data,
        now,
        120,
        window={"from": iso(start), "to": iso(end)},
        series_options=options,
        truncated=len(vrows) >= limit,
        **({"notes": anchor_notes} if anchor_notes else {}),
    )


def _attach_anchor(run, points, p, comp, bid, sel, start, end):
    """The series' current anchor and each point's anchored verdict (schema 88); no anchor is a state, not an error."""
    q = {
        **p,
        "a": sel["arch"],
        "b": sel["backend"],
        "j": sel["source_job"],
        "m": sel["metric"],
    }
    keyed = " AND component = {c:String} AND benchmark_id = {bid:UUID} AND arch = {a:String} AND backend = {b:String} AND metric = {m:String}"
    for pt in points:
        pt["anchored"] = None
    anc = anchored_read(
        run,
        f"""SELECT anchor_artifact_id, ref_key, version, anchor_from, accepted_by, accepted_at, from_run_id, n_samples, anchor_value,
                   anchor_mad, source_job
            FROM v_benchmark_anchor WHERE source_job IN ({{j:String}}, '') {keyed}
            ORDER BY source_job DESC LIMIT 1""",
        q,
    )
    if anc is None:
        return {"state": "unavailable"}, [
            "schema 88 (anchored verdicts) is not applied"
        ]
    if not anc:
        return {"state": "none"}, None
    a = anc[0]
    rows = anchored_read(
        run,
        f"""SELECT run_id, status, reason, anchor_value, delta_pct, worse_pct, floor_pct, since_ts, n_exceeding_runs, days_exceeding, version
            FROM v_benchmark_anchored_verdicts
            WHERE source_job = {{j:String}} {keyed} AND run_ts >= {{t_from:DateTime}} AND run_ts < {{t_to:DateTime}}
            ORDER BY run_ts DESC, run_id LIMIT 20000""",
        q,
    )
    by = {str(r["run_id"]): r for r in rows or []}
    for pt in points:
        r = by.get(pt["run_id"])
        pt["anchored"] = anchor_status(r) if r else None
    base = num(a["anchor_value"])
    floor = next(
        (num(r["floor_pct"]) for r in rows or [] if r["status"] != "no_anchor"),
        None,
    )
    newest = next(
        (pt for pt in reversed(points) if pt["invalidation_state"] != "invalidated"),
        None,
    )
    last = (
        newest["anchored"] if newest else None
    )  # no anchored row yet is unknown, not the run before
    return {
        "state": "anchored", "artifact_id": str(a["anchor_artifact_id"]), "ref_key": a["ref_key"], "version": a["version"],
        "value": rnd(base), "n_samples": int(a["n_samples"]), "from": iso(a["anchor_from"]), "accepted_by": a["accepted_by"] or None,
        "accepted_at": a["accepted_at"] or None, "from_run_id": a["from_run_id"] or None, "source_job": a["source_job"] or None,
        "floor_pct": rnd(floor, 1),
        "band": None if not base or floor is None else
            {"low": rnd(base * (1 - floor / 100)), "high": rnd(base * (1 + floor / 100)), "center": rnd(base)},
        "latest": last,
        "persistent": bool(last and last["status"] == "regressed"),
    }, None  # fmt: skip


def _bench_head(bm, bid):
    pr = bm["props"] or {}
    return {"benchmark_id": bid, "name": bm["name"], "component": bm["component"], "tags": list(bm["tags"] or []),
            "record_type": pr.get("record_type") or None, "config_name": pr.get("config_name") or None, "props": dict(pr)}  # fmt: skip


def _row_job(props):
    return (props.get("source_file") or "").split("/")[0]


def _row_arch(props, want):
    return (props.get("arch") or props.get("platform") or want) == want


def _reference_line(run, key, bid, metric):
    ref = _ref_values(run, key, None)
    vals = [
        v for v in ref["values"] if v["benchmark_id"] == bid and v["metric"] == metric
    ]
    return {"key": key, "version": ref["reference"]["version"], "state": ref["reference"]["state"],
            "value": vals[0]["value"] if vals else None, "label": ref["reference"]["label"]}  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# L3 run detail

RUN_PROP_GROUPS = ("run.purpose", "run.trust", "run.tier", "run.invoker", "run.trigger", "run.validity", "run.validity_policy",
                   "run.suite.name", "run.profile_level", "run.on_behalf_of", "run.preempted", "run.cache")  # fmt: skip


def _run_rows(run, run_id, bids=None, valid_only=False):
    """Measured rows of one run (reference and trace record types excluded): the repeats behind L3 and compare.
    valid_only drops repeats an invalidation covers."""
    rows = _run_rows_all(run, run_id, bids)
    if not valid_only:
        return rows
    hits = invalidations(run, {run_id}).get(str(run_id), [])
    return [
        r for r in rows if not invalid_repeat(hits, str(r["benchmark_id"]), r["repeat"])
    ]


def _run_rows_all(run, run_id, bids=None):
    return run(
        f"""SELECT r.benchmark_id AS benchmark_id, r.backend AS backend, r.component AS component, r.ts AS ts,
                   toUInt16OrZero(r.props['run.repeat']) AS repeat, r.measurements AS measurements, r.iterations AS iterations,
                   r.props AS props
            FROM benchmark_runs AS r
            INNER JOIN (SELECT benchmark_id, any(props['record_type']) AS record_type FROM benchmarks GROUP BY benchmark_id) AS b
              ON b.benchmark_id = r.benchmark_id
            WHERE r.run_id = {{run_id:UUID}} AND r.props['run.ref_only'] != '1' AND r.props['run.record_type'] != 'reference'
              AND b.record_type NOT IN {EXCL} {"AND r.benchmark_id IN {bids:Array(UUID)}" if bids else ""}
            ORDER BY r.benchmark_id, r.backend, repeat LIMIT 200000""",
        {"run_id": run_id, **({"bids": sorted(bids)} if bids else {})},
    )


def summarize_repeats(rows):
    """{metric: stats} over a benchmark's repeat rows: per-repeat medians, their median, spread, and the all-sample CV."""
    out = {}
    for m in sorted({k for r in rows for k in r["measurements"]}):
        reps = [
            (r["repeat"], [float(x) for x in r["measurements"][m] if x != 0])
            for r in rows
            if m in r["measurements"]
        ]
        reps = [(i, xs) for i, xs in reps if xs]
        if not reps:
            continue
        meds = [median(xs) for _, xs in reps]
        allx = [x for _, xs in reps for x in xs]
        out[m] = {
            "value": rnd(median(allx)), "median_of_repeats": rnd(median(meds)), "min": rnd(min(allx)), "max": rnd(max(allx)),
            "n_repeats": len(reps), "n_samples": len(allx), "cv_pct": rnd(cv_pct(allx), 2), "repeat_cv_pct": rnd(cv_pct(meds), 2),
            "repeats": [{"repeat": i, "median": rnd(median(xs)), "n": len(xs), "samples": [rnd(x) for x in xs]} for i, xs in reps],
        }  # fmt: skip
    return out


def run_detail(run, run_id, now=None, limit=50, cursor=None, benchmark_id=None, **_):
    now = naive(now)
    rid = uuid_arg(run_id, "run_id")
    limit = bounded(limit, "limit", 50, 1, 200)
    off = bounded(decode_cursor(cursor).get("offset"), "cursor", 0, 0, 10_000_000)
    rows = _run_rows(
        run, rid, {uuid_arg(benchmark_id, "benchmark_id")} if benchmark_id else None
    )
    subj = subjects(run, {rid}).get(rid)
    kinds = run(
        """SELECT b.record_type AS record_type, count() AS n
           FROM benchmark_runs AS r
           INNER JOIN (SELECT benchmark_id, any(props['record_type']) AS record_type FROM benchmarks GROUP BY benchmark_id) AS b
             ON b.benchmark_id = r.benchmark_id
           WHERE r.run_id = {run_id:UUID} GROUP BY record_type LIMIT 100""",
        {"run_id": rid},
    )
    kmap = {k["record_type"]: int(k["n"]) for k in kinds}
    if not rows and not subj and not kmap:
        raise ApiError(404, "unknown_run", "no such run", run_id=rid)
    inv = invalidations(run, {rid}).get(rid, [])
    pl = run(
        "SELECT repeat, env_id, nodes, clean FROM v_run_placements_effective WHERE run_id = {run_id:UUID} ORDER BY repeat LIMIT 1000",
        {"run_id": rid},
    )
    comps = sorted({r["component"] for r in rows}) or (
        [subj["component"]] if subj and subj["component"] else []
    )
    vr = []
    if comps:
        t_from = min(
            (to_dt(r["ts"]) for r in rows), default=now - timedelta(days=1)
        ) - timedelta(days=3)
        vr = run(
            f"""{latest_verdicts(" AND component IN {cs:Array(String)} AND run_id = {rid:UUID}")} LIMIT 20000""",
            {
                "cs": comps,
                "rid": rid,
                "t_from": ch_time(t_from),
                "t_to": ch_time(now + timedelta(days=1)),
            },
        )
    vmap = {}
    for v in vr:
        vmap.setdefault((str(v["benchmark_id"]), v["backend"]), {})[v["metric"]] = {
            "status": v["status"], "reason": v["reason"] or None, "tier": v["tier"], "baseline_value": rnd(num(v["baseline_value"])),
            "current_value": rnd(num(v["current_value"])), "delta_pct": rnd(num(v["delta_pct"]), 1), "worse_pct": rnd(num(v["worse_pct"]), 1),
            "floor_pct": rnd(num(v["floor_pct"]), 1), "n_baseline": int(v["n_baseline"]),
        }  # fmt: skip
    groups = {}
    for r in rows:
        groups.setdefault((str(r["benchmark_id"]), r["backend"]), []).append(r)
    meta = bench_meta(run, {k[0] for k in groups})
    ordered = sorted(
        groups, key=lambda k: ((meta.get(k[0]) or {}).get("name", ""), k[0], k[1])
    )
    first = rows[0]["props"] if rows else {}
    benches = []
    for k in ordered[off : off + limit]:
        rs = groups[k]
        b = meta.get(k[0], {})
        reps = sorted({r["repeat"] for r in rs})
        hit = [h for h in inv if covers(h, k[0], None) and h["state"] != "restored"]
        benches.append({
            "benchmark_id": k[0], "name": b.get("name"), "backend": k[1], "component": b.get("component"),
            "record_type": (b.get("props") or {}).get("record_type") or None, "repeats": reps,
            "metrics": summarize_repeats([r for r in rs if not invalid_repeat(inv, k[0], r["repeat"])]), "verdicts": vmap.get(k, {}),
            "invalidation_state": run_state(hit), "invalidated_repeats": sorted({h["repeat"] for h in inv if h["state"] == "invalidated" and h["repeat"] is not None}),  # fmt: skip
        })  # fmt: skip
    hdr_props = {
        k: first.get(k) for k in RUN_PROP_GROUPS if first.get(k) not in (None, "")
    }
    clean = {k: v for k, v in first.items() if k.startswith("run.clean.")}
    cvs = [
        m["cv_pct"]
        for b in benches
        for m in b["metrics"].values()
        if m["cv_pct"] is not None
    ]
    n_rep = max([len(b["repeats"]) for b in benches], default=0)
    validity, reasons, source = _validity(
        first, clean, n_rep, [c for c in cvs], pl, inv
    )
    data = {
        "run": {
            "run_id": rid,
            "arch": (first.get("arch") or (subj or {}).get("arch") or None),
            "ts": iso(
                first.get("run_utc")
                and parse_time(first["run_utc"], "run_utc")
                or (rows[0]["ts"] if rows else None)
            ),
            "components": comps,
            "props": hdr_props,
            "clean": clean,
            "env_id": first.get("env.id") or None,
            "source_job": _row_job(first) or None,
            "subject": subj,
            "invalidation_state": run_state(inv, {r["repeat"] for r in rows}),
            "invalidations": inv,
            "validity": validity,
            "validity_reasons": reasons,
            "validity_reasons_source": source,
            "trust": first.get("run.trust") or None,
        },  # fmt: skip
        "benchmarks": benches,
        "availability": {
            "benchmarks": len(groups),
            "environment": bool(first.get("env.id")),
            "placements": bool(pl),
            "steps": kmap.get("step_trace", 0) > 0,
            "spans": kmap.get("span", 0) > 0,
            "step_rows": kmap.get("step_trace", 0),
            "span_rows": kmap.get("span", 0),
            "verdicts": bool(vr),
        },  # fmt: skip
    }
    nxt = encode_cursor({"offset": off + limit}) if off + limit < len(ordered) else None
    return envelope(data, now, 60, total=len(ordered), next_cursor=nxt)


def _validity(first, clean, n_rep, cvs, placements, inv):
    stored = first.get("run.validity_reasons")
    state = first.get("run.validity") or "unknown"
    if stored:
        return state, [s for s in stored.split(",") if s], "stored"
    reasons = []
    if 0 < n_rep < 3:
        reasons.append("few_repeats")
    if cvs and max(cvs) > 5.0:
        reasons.append("cv_above_limit")
    if clean.get("run.clean.exclusive") not in (None, "enforced"):
        reasons.append("shared_env")
    pin = first.get("run.exec.cpu_pinning", "none")
    if pin not in ("none", "") and first.get("run.exec.pinning_verified") != "true":
        reasons.append(
            "pinning_violated"
            if first.get("run.exec.pinning_verified") == "false"
            else "pinning_unverified"
        )
    if first.get("run.trust") == "external":
        reasons.append("external")
    if any(h["state"] == "invalidated" for h in inv):
        reasons.append("invalidated")
    return state, reasons, "derived"


# ---------------------------------------------------------------------------------------------------------------
# L4 environment


def environment(run, run_id, now=None, **_):
    now = naive(now)
    rid = uuid_arg(run_id, "run_id")
    reps = run(
        """SELECT toUInt16OrZero(props['run.repeat']) AS repeat, min(ts) AS ts, any(props) AS rprops
           FROM benchmark_runs WHERE run_id = {run_id:UUID} AND props['run.ref_only'] != '1'
           GROUP BY repeat ORDER BY repeat LIMIT 1000""",
        {"run_id": rid},
    )
    pl = run(
        """SELECT repeat, started_at, env_id, placement_id, nodes, devices, links, clean
           FROM v_run_placements_effective WHERE run_id = {run_id:UUID} ORDER BY repeat LIMIT 1000""",
        {"run_id": rid},
    )
    if not reps and not pl:
        raise ApiError(404, "unknown_run", "no such run", run_id=rid)
    first = reps[0]["rprops"] if reps else {}
    run_ts = (
        parse_time(first.get("run_utc"), "run_utc")
        if first.get("run_utc")
        else (reps[0]["ts"] if reps else now)
    )
    env_ids = sorted(
        {
            str(x)
            for x in [first.get("env.id")] + [p["env_id"] for p in pl]
            if x and str(x) != NIL
        }
    )
    nodes = sorted({n for p in pl for n in p["nodes"]})
    fps = {}
    if env_ids:
        fps = {str(f["env_id"]): f for f in run(
            "SELECT env_id, fingerprint_version, labels, fingerprint, members, devices FROM v_environment_fingerprints WHERE env_id IN {ids:Array(UUID)} LIMIT 1000",
            {"ids": env_ids})}  # fmt: skip
    hist = []
    if nodes:
        hist = run(
            "SELECT node, env_id, node_class, valid_from, valid_to FROM v_environment_nodes WHERE node IN {n:Array(String)} ORDER BY node, valid_from LIMIT 5000",
            {"n": nodes},
        )
    quals = []
    if nodes and env_ids:
        quals = run(
            "SELECT node, env_id, outcome, cv_pct, qualified_at, expires_at, is_current FROM v_environment_qualified WHERE node IN {n:Array(String)} AND env_id IN {e:Array(UUID)} LIMIT 5000",
            {"n": nodes, "e": env_ids},
        )
    classes = sorted({h["node_class"] for h in hist})
    ncs = {}
    if classes:
        ncs = {c["name"]: c for c in run(
            "SELECT name, host_profile, qualification FROM v_node_classes WHERE name IN {n:Array(String)} LIMIT 100", {"n": classes})}  # fmt: skip
    req = None
    if first.get("env.name"):
        rr = run(
            "SELECT name, spec_digest, labels, devices, host, policy FROM v_environments WHERE name = {n:String} LIMIT 5",
            {"n": first["env.name"]},
        )
        req = dict(rr[0]) if rr else None
    nodeinfo = {}
    for n in nodes:
        iv = [h for h in hist if h["node"] == n]
        cur = [
            h
            for h in iv
            if to_dt(h["valid_from"]) <= to_dt(run_ts)
            and (h["valid_to"] is None or to_dt(h["valid_to"]) > to_dt(run_ts))
        ]
        nodeinfo[n] = {
            "node_class": cur[0]["node_class"] if cur else None,
            "env_id_at_run": str(cur[0]["env_id"]) if cur else None,
            "env_history": [
                {
                    "env_id": str(h["env_id"]),
                    "node_class": h["node_class"],
                    "valid_from": iso(h["valid_from"]),
                    "valid_to": iso(h["valid_to"]),
                }
                for h in iv
            ],  # fmt: skip
        }
    qual = []
    for qr in quals:
        in_force = qr["outcome"] == "pass" and to_dt(qr["qualified_at"]) <= to_dt(
            run_ts
        ) < to_dt(qr["expires_at"])
        qual.append({"node": qr["node"], "env_id": str(qr["env_id"]), "outcome": qr["outcome"], "cv_pct": rnd(num(qr["cv_pct"]), 2),
                     "qualified_at": iso(qr["qualified_at"]), "expires_at": iso(qr["expires_at"]), "in_force_at_run": in_force,
                     "current_now": bool(qr["is_current"])})  # fmt: skip
    placements = []
    by_rep = {int(r["repeat"]): r["rprops"] for r in reps}
    for p in pl:
        pr = by_rep.get(int(p["repeat"]), {})
        placements.append({
            "repeat": int(p["repeat"]), "started_at": iso(p["started_at"]), "env_id": str(p["env_id"]), "placement_id": str(p["placement_id"]),
            "nodes": list(p["nodes"]), "links": list(p["links"]),
            "devices": [d if isinstance(d, dict) else dict(zip(("node", "serial", "pci", "firmware", "link"), d)) for d in p["devices"]],
            "clean": dict(p["clean"]), "pinning": _pinning(pr),
        })  # fmt: skip
    if (
        not placements
    ):  # a run recorded before placement capture still has per-repeat pinning props
        placements = []
    pinning = [{"repeat": r, **_pinning(pr)} for r, pr in sorted(by_rep.items())]
    env_main = env_ids[0] if env_ids else None
    fp = fps.get(env_main) if env_main else None
    labels = dict(fp["labels"]) if fp else {}
    in_force = any(q["in_force_at_run"] and q["env_id"] == env_main for q in qual)
    data = {
        "run_id": rid,
        "env_id": env_main,
        "env_ids": env_ids,
        "fingerprint": None
        if not fp
        else {
            "version": int(fp["fingerprint_version"]),
            "values": dict(fp["fingerprint"]),
            "members": [str(m) for m in fp["members"]],
        },  # fmt: skip
        "labels": labels,
        "pinning_label": labels.get(POLICY_LABEL) or "none",
        "node_class": next(
            (v["node_class"] for v in nodeinfo.values() if v["node_class"]), None
        ),  # fmt: skip
        "node_class_spec": {
            k: {
                "host_profile": dict(v["host_profile"]),
                "qualification": dict(v["qualification"]),
            }
            for k, v in ncs.items()
        },  # fmt: skip
        "request": req,
        "nodes": nodeinfo,
        "qualification": {"entries": qual, "in_force_at_run": in_force},
        "clean": {
            "run": {k: v for k, v in first.items() if k.startswith("run.clean.")},
            "per_repeat": [{"repeat": p["repeat"], **p["clean"]} for p in placements],
        },  # fmt: skip
        "placements": placements,
        "pinning": pinning,
        "availability": {
            "fingerprint": fp is not None,
            "placements": bool(pl),
            "qualification": bool(qual),
            "request": req is not None,
            "node_history": bool(hist),
        },  # fmt: skip
    }
    return envelope(data, now, 300)


def _pinning(props):
    return {"policy": props.get("run.exec.cpu_pinning") or None, "cpu_set": props.get("run.exec.cpu_set") or None,
            "cpu_set_source": props.get("run.exec.cpu_set_source") or None, "verified": props.get("run.exec.pinning_verified") or None,
            "violations": props.get("run.exec.pinning_violations") or None}  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# L5 steps


def steps(
    run,
    run_id,
    now=None,
    benchmark_id=None,
    repeat=None,
    prompt=None,
    signal=None,
    outlier_z=3.5,
    max_rows=100000,
    _views=True,
    **_,
):
    now = naive(now)
    rid = uuid_arg(run_id, "run_id")
    cap = bounded(max_rows, "max_rows", 100000, 1, 1000000)
    extra = (
        (" AND t.benchmark_id = {bid:UUID}" if benchmark_id else "")
        + (" AND t.repeat = {rep:UInt16}" if repeat is not None else "")
        + (" AND t.prompt = {pr:String}" if prompt is not None else "")
    )
    raw_extra = (
        (" AND benchmark_id = {bid:UUID}" if benchmark_id else "")
        + (
            " AND toUInt16OrZero(props['run.repeat']) = {rep:UInt16}"
            if repeat is not None
            else ""
        )
        + (" AND props['run.prompt'] = {pr:String}" if prompt is not None else "")
    )
    qp = {
        "run_id": rid,
        **({"bid": uuid_arg(benchmark_id, "benchmark_id")} if benchmark_id else {}),
        **(
            {"rep": bounded(repeat, "repeat", 0, 0, 65535)}
            if repeat is not None
            else {}
        ),
        **({"pr": str(prompt)} if prompt is not None else {}),
    }
    rows = run(
        f"""SELECT t.benchmark_id AS benchmark_id, t.backend AS backend, t.repeat AS repeat, t.rank AS rank, t.prompt AS prompt, t.pos AS pos,
                   t.step_index AS step_index, t.step_kind AS step_kind, t.cold AS cold, t.metrics AS metrics
            FROM v_bench_step_records AS t
            WHERE t.run_id = {{run_id:UUID}}{extra}
            ORDER BY benchmark_id, backend, repeat, prompt, rank, pos LIMIT {{cap:UInt32}}""",
        {**qp, "cap": cap + 1},
    )
    if len(rows) > cap:
        raise ApiError(
            413,
            "result_too_large",
            "the run has more step rows than max_rows; narrow with benchmark_id, repeat or prompt",
            max_rows=cap,
        )
    if not rows:
        probe = run(
            f"""SELECT 1 AS x FROM benchmark_runs WHERE run_id = {{run_id:UUID}} AND mapContains(measurements, 'step_kind')
                {raw_extra} LIMIT 1""",
            qp,
        )
        want_bid = str(uuid_arg(benchmark_id, "benchmark_id")) if benchmark_id else None
        want_rep = None if repeat is None else int(repeat)
        hits = [
            h
            for h in invalidations(run, {rid}).get(rid, [])
            if h["state"] == "invalidated"
            and (want_bid is None or h["benchmark_id"] in (None, want_bid))
            and (want_rep is None or h["repeat"] in (None, want_rep))
        ]
        if probe and hits:
            raise ApiError(
                404,
                "no_matching_steps",
                "no step records match (or the matching repeats are invalidated)",
                run_id=rid,
            )
        if probe:
            return envelope(
                {"records": [], "ranks": [], "steps": []},
                now,
                300,
                no_data=True,
                reason="step rows exist but none is readable (arrays of unequal length, non-finite or duplicate indexes are skipped)",
            )
        return envelope(
            {"records": [], "ranks": [], "steps": []},
            now,
            300,
            no_data=True,
            reason="no step_trace rows for this run",
        )
    keys = sorted(
        {
            (str(r["benchmark_id"]), int(r["repeat"]), r["prompt"], r["backend"])
            for r in rows
        }
    )
    chosen, backend = keys[0][:3], keys[0][3]
    sel = [
        r
        for r in rows
        if (str(r["benchmark_id"]), int(r["repeat"]), r["prompt"], r["backend"])
        == keys[0]
    ]
    metas = {
        (str(m["benchmark_id"]), int(m["repeat"]), m["prompt"], int(m["rank"])): m
        for m in run(
            """SELECT r.benchmark_id AS benchmark_id, toUInt16OrZero(r.props['run.repeat']) AS repeat,
                      toUInt16OrZero(r.props['run.rank']) AS rank, r.props['run.prompt'] AS prompt,
                      argMax(r.props, (r.ts, r.audit_timestamp, r.audit_uuid)) AS pr,
                      mapFilter((k, v) -> startsWith(k, 'summary.'),
                                argMax(r.measurements, (r.ts, r.audit_timestamp, r.audit_uuid))) AS summ
               FROM benchmark_runs AS r
               WHERE r.run_id = {run_id:UUID} AND r.benchmark_id = {bid:UUID} AND r.backend = {be:String}
                 AND mapContains(r.measurements, 'step_kind')
               GROUP BY benchmark_id, repeat, rank, prompt LIMIT 10000""",
            {"run_id": rid, "bid": chosen[0], "be": backend},
        )
    }
    props = metas[(chosen[0], chosen[1], chosen[2], int(sel[0]["rank"]))]["pr"]
    roles = {r: props.get(f"role.{r}") for r in ("total", "wait", "compute", "busy")}
    sig_total = signal or roles["total"]
    signals = sorted({k for r in sel for k in r["metrics"]})
    if sig_total not in signals:
        raise ApiError(
            400,
            "unknown_signal",
            "signal not present in the step records",
            signals=signals,
        )
    ranks = sorted({int(r["rank"]) for r in sel})
    stepl = sorted({(int(r["step_index"]), int(r["step_kind"])) for r in sel})
    names = {}
    cold = {}
    kind = {}
    cell = {}
    for r in sel:
        s = (int(r["step_index"]), int(r["step_kind"]))
        names[s] = "ttft" if r["step_kind"] == 0 else f"itl_{int(r['step_index']):04d}"
        kind[s] = "ttft" if r["step_kind"] == 0 else "itl"
        cold[s] = max(cold.get(s, 0), int(r["cold"]))
        cell[(int(r["rank"]), s)] = {k: float(v) for k, v in r["metrics"].items()}

    def mat(sg):
        return [
            [None if (rk, s) not in cell else cell[(rk, s)].get(sg) for s in stepl]
            for rk in ranks
        ]

    declared = bool(props.get(f"agg.{sig_total}"))
    rule = props.get(f"agg.{sig_total}") or (
        "max" if sig_total == roles["total"] else "mean"
    )
    agg = {
        "max": max,
        "min": min,
        "sum": sum,
        "mean": statistics.mean,
        "median": statistics.median,
    }[rule if rule in ("max", "min", "sum", "median") else "mean"]
    total = mat(sig_total)
    per_step, lag = [], [[None] * len(stepl) for _ in ranks]
    for j, s in enumerate(stepl):
        col = [
            (ranks[i], total[i][j])
            for i in range(len(ranks))
            if total[i][j] is not None
        ]
        if not col:
            per_step.append(None)
            continue
        vmin = min(v for _, v in col)
        for i in range(len(ranks)):
            if total[i][j] is not None:
                lag[i][j] = rnd(total[i][j] - vmin, 3)
        per_step.append({"step": names[s], "value": rnd(agg([v for _, v in col]), 3), "slowest_rank": max(col, key=lambda x: (x[1], -x[0]))[0],
                         "fastest_rank": min(col, key=lambda x: (x[1], x[0]))[0], "cold": bool(cold[s])})  # fmt: skip
    headline = {}
    for kd in ("ttft", "itl"):
        warm = [
            per_step[j]["value"]
            for j, s in enumerate(stepl)
            if per_step[j] and kind[s] == kd and not cold[s]
        ]
        rank_med = {
            ranks[i]: median(
                [
                    total[i][j]
                    for j, s in enumerate(stepl)
                    if kind[s] == kd and not cold[s] and total[i][j] is not None
                ]
            )
            for i in range(len(ranks))
        }
        valid = {k: v for k, v in rank_med.items() if v is not None}
        headline[kd] = {"value": rnd(median(warm), 3) if warm else None, "n_warm_steps": len(warm), "rule": rule, "rule_declared": declared,
                        "rank_medians": {str(k): rnd(v, 3) for k, v in valid.items()},
                        "rank_skew": rnd(max(valid.values()) - min(valid.values()), 3) if valid else None,
                        "slowest_rank": max(valid, key=lambda k: (valid[k], -k)) if valid else None}  # fmt: skip
    wait = mat(roles["wait"]) if roles["wait"] in signals else None
    awaited = None
    if wait:
        awaited = [None if all(wait[i][j] is None for i in range(len(ranks))) else
                   ranks[min((i for i in range(len(ranks)) if wait[i][j] is not None), key=lambda i: (wait[i][j], i))] for j in range(len(stepl))]  # fmt: skip
    outliers = _outliers(
        ranks,
        stepl,
        kind,
        cold,
        total,
        wait,
        mat(roles["compute"]) if roles["compute"] in signals else None,
        names,
        float(outlier_z),
    )
    if _views and sig_total == roles["total"] and float(outlier_z) == 3.5:
        headline, outliers = _headline_outliers(
            run, rid, chosen, backend, rnd, rule, declared
        )
    ws = props.get("run.world_size") or None
    consistency = []
    for key, kd in (("summary.itl." + sig_total, "itl"),):
        for rk in ranks:
            summ = metas.get((chosen[0], chosen[1], chosen[2], rk), {}).get("summ", {})
            if summ.get(key):
                steps_v = [
                    cell[(rk, s)].get(sig_total)
                    for s in stepl
                    if kind[s] == kd and cell.get((rk, s))
                ]
                steps_v = [x for x in steps_v if x is not None]
                if steps_v:
                    rep = float(summ[key][0])
                    mean = statistics.mean(steps_v)
                    consistency.append({"rank": rk, "signal": sig_total, "reported": rnd(rep, 3), "from_steps": rnd(mean, 3), "rel_diff_pct": rnd((rep - mean) / mean * 100, 2) if mean else None})  # fmt: skip
    data = {
        "selection": {"benchmark_id": chosen[0], "repeat": chosen[1], "prompt": chosen[2], "backend": backend, "available": [{"benchmark_id": k[0], "repeat": k[1], "prompt": k[2], "backend": k[3]} for k in keys]},  # fmt: skip
        "roles": roles, "signals": signals, "signal": sig_total, "ranks": ranks, "steps": [names[s] for s in stepl], "cold": [bool(cold[s]) for s in stepl],
        "values": [[rnd(v, 3) for v in rowv] for rowv in total], "cross_rank": per_step, "lag": lag, "wait": None if not wait else [[rnd(v, 3) for v in rowv] for rowv in wait],  # fmt: skip
        "awaited_rank": awaited, "headline": headline, "outliers": outliers, "consistency": consistency,
        "rank_coverage": {"present": len(ranks), "world_size": None if ws is None else int(ws), "complete": ws is not None and len(ranks) == int(ws)},
        "reporter": {"version": props.get("run.reporter_version") or None, "source": props.get("run.reporter_version_source") or None},
        "trace_uri": props.get("run.trace_uri") or None,
    }  # fmt: skip
    return envelope(data, now, 300)


def _headline_outliers(run, rid, key, backend, rnd, rule, declared):
    """The headline per kind and the outlier steps of one (benchmark, repeat, prompt), from schema 87 (role total, z > 3.5)."""
    p = {"run_id": rid, "bid": key[0], "rep": key[1], "pr": key[2], "be": backend}
    where = "run_id = {run_id:UUID} AND benchmark_id = {bid:UUID} AND backend = {be:String} AND repeat = {rep:UInt16} AND prompt = {pr:String}"
    headline = {
        kd: {
            "value": None,
            "n_warm_steps": 0,
            "rule": rule,
            "rule_declared": declared,
            "rank_medians": {},
            "rank_skew": None,
            "slowest_rank": None,
        }
        for kd in ("ttft", "itl")
    }
    for h in run(f"SELECT * FROM v_bench_step_headline WHERE {where}", p):
        med = {str(k): rnd(v, 3) for k, v in sorted((h["rank_medians"] or {}).items())}
        headline[h["kind"]] = {"value": rnd(h["headline"], 3), "n_warm_steps": int(h["n_warm_steps"]), "rule": h["rule"],
                               "rule_declared": bool(h["rule_declared"]), "rank_medians": med,
                               "rank_skew": rnd(h["rank_skew"], 3), "slowest_rank": h["slowest_rank"]}  # fmt: skip
    out = []
    for o in run(f"SELECT * FROM v_bench_step_outliers WHERE {where} LIMIT 10000", p):
        both = o["excess_wait"] is None and o["excess_compute"] is None
        out.append({"rank": int(o["rank"]), "step": o["step"], "value": rnd(o["value"], 3), "median": rnd(o["median"], 3), "z": rnd(o["z"], 1),
                    "excess": rnd(o["excess"], 3), "excess_wait": rnd(o["excess_wait"], 3), "excess_compute": rnd(o["excess_compute"], 3),
                    "excess_host": None if both else rnd(o["excess_host"], 3)})  # fmt: skip
    out.sort(key=lambda o: (-o["z"], o["rank"], o["step"]))
    return headline, out


def _outliers(ranks, stepl, kind, cold, total, wait, comp, names, zcut):
    out = []
    for i, rk in enumerate(ranks):
        for kd in ("ttft", "itl"):
            idx = [
                j
                for j, s in enumerate(stepl)
                if kind[s] == kd and not cold[s] and total[i][j] is not None
            ]
            xs = [total[i][j] for j in idx]
            if len(xs) < 5:
                continue
            med = statistics.median(xs)
            mad = statistics.median([abs(x - med) for x in xs])
            if mad == 0:
                continue
            for j in idx:
                z = 0.6745 * (total[i][j] - med) / mad
                if z > zcut:
                    ex = total[i][j] - med
                    w = (
                        (wait[i][j] - median([wait[i][k] for k in idx]))
                        if wait
                        else None
                    )
                    c = (
                        (comp[i][j] - median([comp[i][k] for k in idx]))
                        if comp
                        else None
                    )
                    out.append({"rank": rk, "step": names[stepl[j]], "value": rnd(total[i][j], 3), "median": rnd(med, 3), "z": rnd(z, 1), "excess": rnd(ex, 3),
                                "excess_wait": rnd(w, 3), "excess_compute": rnd(c, 3),
                                "excess_host": rnd(ex - (w or 0) - (c or 0), 3) if wait is not None or comp is not None else None})  # fmt: skip
    out.sort(key=lambda o: (-o["z"], o["rank"], o["step"]))
    return out


# ---------------------------------------------------------------------------------------------------------------
# L6 spans

TRACE_TYPES = {
    "reference",
    "step_trace",
    "span",
    "trace_phase",
    "trace_step",
    "trace_kernel",
    "compile_pass",
}


def _span_source(run, rid):
    """v_bench_spans_fast (schema 90) when it holds exactly the spans the run's rows declare in `iterations`, else v_bench_spans.

    Any mismatch (a partial backfill, a skipped malformed row, rows that do not state their length) takes the slow path."""
    try:
        fast = run(
            "SELECT count() AS n FROM bench_span_values WHERE run_id = {run_id:UUID}",
            {"run_id": rid},
        )
        if not fast or not int(fast[0]["n"]):
            return "v_bench_spans"
        raw = run(
            "SELECT sum(iterations) AS n FROM benchmark_runs WHERE run_id = {run_id:UUID} AND mapContains(measurements, 'dur_ns')",
            {"run_id": rid},
        )
    except Exception as err:  # noqa: BLE001 -- only a missing optional table is expected
        if "bench_span_values" not in str(err) or not any(
            m in str(err).lower()
            for m in ("unknown table", "doesn't exist", "unknown_table")
        ):
            raise
        return "v_bench_spans"
    return (
        "v_bench_spans_fast"
        if raw and int(raw[0]["n"] or 0) == int(fast[0]["n"])
        else "v_bench_spans"
    )


def spans(run, run_id, now=None, view="summary", rank=None, step=None, repeat=None, prompt=None, lane=None, layer=None,
          limit=2000, offset=0, min_dur_ns=0, top=50, **_):  # fmt: skip
    now = naive(now)
    rid = uuid_arg(run_id, "run_id")
    if view not in ("summary", "kernels", "timeline"):
        raise ApiError(
            400,
            "bad_parameter",
            "view must be summary, kernels or timeline",
            parameter="view",
        )
    limit = bounded(limit, "limit", 2000, 1, 20000)
    top = bounded(top, "top", 50, 1, 1000)
    offset = bounded(offset, "offset", 0, 0, 10_000_000)
    conds, p = [], {"run_id": rid}
    for name, val, col, typ in (("rank", rank, "g.rank", "UInt16"), ("step", step, "step", "UInt32"), ("repeat", repeat, "g.repeat", "UInt16"),
                                ("layer", layer, "layer", "Int16"), ("lane", lane, "g.lane", "String"), ("prompt", prompt, "g.prompt", "String")):  # fmt: skip
        if val is not None:
            conds.append(f"{col} = {{{name}:{typ}}}")
            p[name] = val if typ == "String" else int(val)
    if min_dur_ns:
        conds.append("dur_ns >= {mind:Int64}")
        p["mind"] = int(min_dur_ns)
    where = "".join(f" AND {c}" for c in conds)
    frm = f"FROM {_span_source(run, rid)} AS g WHERE g.run_id = {{run_id:UUID}}"
    avail = run(
        f"SELECT count() AS n, uniqExact(g.rank) AS ranks, uniqExact(step) AS steps, groupUniqArray(g.lane) AS lanes {frm}",
        {"run_id": rid},
    )
    if not avail or int(avail[0]["n"]) == 0:
        whole = any(
            h["state"] == "invalidated"
            and h["benchmark_id"] is None
            and h["repeat"] is None
            for h in invalidations(run, {rid}).get(rid, [])
        )
        if whole and run(
            "SELECT 1 AS x FROM benchmark_runs WHERE run_id = {run_id:UUID} AND mapContains(measurements, 'dur_ns') LIMIT 1",
            {"run_id": rid},
        ):
            raise ApiError(
                410,
                "run_invalidated",
                "the run is invalidated; its spans are not served",
                run_id=rid,
            )
        return envelope(
            {"lanes": [], "kernels": [], "spans": []},
            now,
            300,
            no_data=True,
            reason="no span rows for this run",
        )
    data = {"available": {"spans": int(avail[0]["n"]), "ranks": int(avail[0]["ranks"]), "steps": int(avail[0]["steps"]), "lanes": sorted(avail[0]["lanes"])},
            "filters": {k: v for k, v in p.items() if k not in ("run_id", "bad_repeats")}}  # fmt: skip
    if view == "timeline":
        if rank is None or step is None:
            raise ApiError(400, "bad_parameter", "view=timeline needs rank and step")
        rows = run(
            f"""SELECT g.rank AS rank, g.repeat AS repeat, step, seq, parent_seq, depth, layer, g.lane AS lane, g.name AS name, start_ns, dur_ns
                {frm} {where} ORDER BY start_ns, seq, g.repeat, g.prompt, g.rank LIMIT {{lim:UInt32}} OFFSET {{off:UInt32}}""",
            {**p, "lim": limit + 1, "off": offset},
        )
        data["spans"] = [
            {
                k: (int(v) if isinstance(v, (int, float)) and k != "name" else v)
                for k, v in r.items()
            }
            for r in rows[:limit]
        ]
        return envelope(data, now, 300, truncated=len(rows) > limit)
    lanes = run(
        f"""SELECT g.rank AS rank, g.lane AS lane, count() AS n, sum(dur_ns) AS total_ns, max(dur_ns) AS max_ns, avg(dur_ns) AS mean_ns,
                   quantile(0.95)(dur_ns) AS p95_ns {frm} {where} GROUP BY rank, lane ORDER BY rank, lane LIMIT 1000""", p)  # fmt: skip
    kern = run(
        f"""SELECT g.lane AS lane, g.name AS name, count() AS n, sum(dur_ns) AS total_ns, avg(dur_ns) AS mean_ns, max(dur_ns) AS max_ns,
                   quantile(0.95)(dur_ns) AS p95_ns, argMax(g.rank, dur_ns) AS slowest_rank, uniqExact(g.rank) AS ranks
            {frm} {where} GROUP BY lane, name ORDER BY total_ns DESC, lane, name LIMIT {{lim:UInt32}} OFFSET {{off:UInt32}}""",
        {
            **p,
            "lim": (top if view == "summary" else limit) + 1,
            "off": 0 if view == "summary" else offset,
        },
    )
    cap = top if view == "summary" else limit
    grand = sum(int(r["total_ns"]) for r in lanes) or 1
    data["lanes"] = [{"rank": int(r["rank"]), "lane": r["lane"], "n": int(r["n"]), "total_ns": int(r["total_ns"]), "mean_ns": rnd(num(r["mean_ns"]), 1),
                      "max_ns": int(r["max_ns"]), "p95_ns": rnd(num(r["p95_ns"]), 1)} for r in lanes]  # fmt: skip
    data["kernels"] = [{"lane": r["lane"], "name": r["name"], "n": int(r["n"]), "total_ns": int(r["total_ns"]), "share": rnd(int(r["total_ns"]) / grand, 4),
                        "mean_ns": rnd(num(r["mean_ns"]), 1), "max_ns": int(r["max_ns"]), "p95_ns": rnd(num(r["p95_ns"]), 1),
                        "slowest_rank": int(r["slowest_rank"]), "ranks": int(r["ranks"])} for r in kern[:cap]]  # fmt: skip
    if view == "summary":
        per_rank = {}
        for r in data["lanes"]:
            per_rank.setdefault(r["rank"], {})[r["lane"]] = r["total_ns"]
        dev = {k: v.get("device", 0) for k, v in per_rank.items()}
        col = {k: v.get("collective", 0) for k, v in per_rank.items()}
        data["rank_summary"] = {
            "per_rank_ns": {str(k): v for k, v in sorted(per_rank.items())},
            "slowest_device_rank": max(dev, key=lambda k: (dev[k], -k)) if dev else None,
            "longest_collective_rank": max(col, key=lambda k: (col[k], -k)) if col else None,
            "shortest_collective_rank": min(col, key=lambda k: (col[k], k)) if col else None,
        }  # fmt: skip
        by_layer = run(
            f"""SELECT g.rank AS rank, layer, sum(dur_ns) AS total_ns {frm} {where} AND g.lane = 'collective' AND layer >= 0
                GROUP BY rank, layer ORDER BY rank, layer LIMIT 100000""", p)  # fmt: skip
        data["collective_by_layer"] = [
            {
                "rank": int(r["rank"]),
                "layer": int(r["layer"]),
                "total_ns": int(r["total_ns"]),
            }
            for r in by_layer
        ]
    return envelope(data, now, 300, truncated=len(kern) > cap)


# ---------------------------------------------------------------------------------------------------------------
# references


def references(
    run,
    now=None,
    key=None,
    state="current",
    ref_class=None,
    version=None,
    include_values=False,
    limit=100,
    cursor=None,
    **_,
):
    now = naive(now)
    limit = bounded(limit, "limit", 100, 1, 500)
    if state not in ("current", "retired", "superseded", "untagged", "all"):
        raise ApiError(
            400,
            "bad_parameter",
            "state must be current, retired, superseded, untagged or all",
            parameter="state",
        )
    if key:
        ref = _ref_values(run, key, version)
        return envelope(ref, now, 300)
    off = bounded(decode_cursor(cursor).get("offset"), "cursor", 0, 0, 10_000_000)
    conds = (
        ["1"]
        + (["state = {state:String}"] if state != "all" else [])
        + (["ref_class = {rc:String}"] if ref_class else [])
    )
    rows = run(
        f"""SELECT artifact_id, ref_key, version, ref_class, origin, label, arch, as_of, approved_by, approved_at, submitted_by, run_id,
                   comparability, state, registered_at
            FROM v_bench_references WHERE {" AND ".join(conds)}
            ORDER BY ref_key, toUInt32OrZero(version) DESC LIMIT {{lim:UInt32}} OFFSET {{off:UInt32}}""",
        {"state": state, "rc": ref_class or "", "lim": limit + 1, "off": off},
    )
    items = [_ref_head(r) for r in rows[:limit]]
    nxt = encode_cursor({"offset": off + limit}) if len(rows) > limit else None
    return envelope({"items": items}, now, 300, next_cursor=nxt, no_data=not items)


def _ref_head(r):
    comp = {
        k.removeprefix("ref.comparability."): v
        for k, v in dict(r["comparability"]).items()
    }
    vals = set(comp.values())
    return {"artifact_id": str(r["artifact_id"]), "key": r["ref_key"], "version": r["version"], "class": r["ref_class"], "origin": r["origin"],
            "label": r["label"], "arch": r["arch"], "as_of": r["as_of"] or None, "approved_by": r["approved_by"] or None, "approved_at": r["approved_at"] or None,
            "submitted_by": r["submitted_by"], "state": r["state"], "run_id": str(r["run_id"]), "comparability": comp,
            "comparability_verdict": "not_comparable" if "incompatible" in vals else "partial" if "different" in vals else "comparable" if comp else "unknown"}  # fmt: skip


def _ref_values(run, key, version):
    heads = run(
        """SELECT artifact_id, ref_key, version, ref_class, origin, label, arch, as_of, approved_by, approved_at, submitted_by, run_id, comparability, state, registered_at
           FROM v_bench_references WHERE ref_key = {k:String} AND ({v:String} = '' OR version = {v:String})
           ORDER BY (state = 'current') DESC, toUInt32OrZero(version) DESC LIMIT 1""",
        {"k": key, "v": str(version or "")},
    )
    if not heads:
        raise ApiError(
            404, "unknown_reference", "no such reference (or not published)", key=key
        )
    head = _ref_head(heads[0])
    rows = run(
        """SELECT r.benchmark_id AS benchmark_id, r.backend AS backend, r.measurements AS measurements, r.iterations AS iterations, r.props AS props
           FROM benchmark_runs AS r WHERE r.run_id = {rid:UUID} AND r.props['run.ref_only'] = '1' ORDER BY benchmark_id, backend LIMIT 20000""",
        {"rid": head["run_id"]},
    )
    meta = bench_meta(run, {str(r["benchmark_id"]) for r in rows})
    vals = []
    for r in rows:
        for m, xs in sorted(r["measurements"].items()):
            xs = [float(x) for x in xs]
            vals.append({"benchmark_id": str(r["benchmark_id"]), "name": (meta.get(str(r["benchmark_id"])) or {}).get("name"), "backend": r["backend"],
                         "metric": m, "value": rnd(median(xs)), "n": len(xs), "terms": r["props"].get("ref.terms") or None})  # fmt: skip
    return {"reference": head, "values": vals}


# ---------------------------------------------------------------------------------------------------------------
# compare


def compare(run, now=None, kind="runs", **kw):
    now = naive(now)
    fns = {
        "runs": _cmp_runs,
        "derive": _cmp_derive,
        "effect": _cmp_effect,
        "reference": _cmp_reference,
    }
    if kind not in fns:
        raise ApiError(
            400, "bad_parameter", f"kind must be one of {sorted(fns)}", parameter="kind"
        )
    data, meta = fns[kind](run, now, **kw)
    return envelope(data, now, 120, kind=kind, **meta)


def _side(run, now, prefix, kw):
    """A compare side: a run id, or the newest run of a subject (+arch) or of an env_id for one benchmark."""
    if kw.get(f"{prefix}_run"):
        return uuid_arg(kw[f"{prefix}_run"], f"{prefix}_run"), {"by": "run"}
    if kw.get(f"{prefix}_subject"):
        aid = uuid_arg(kw[f"{prefix}_subject"], f"{prefix}_subject")
        rows = run(
            """SELECT run_id, max(ts) AS ts FROM v_artifact_results_published
               WHERE artifact_id = {a:UUID} AND result_kind = 'performance' AND ({arch:String} = '' OR arch = {arch:String})
               GROUP BY run_id ORDER BY ts DESC, run_id LIMIT 1""",
            {"a": aid, "arch": kw.get("arch") or ""},
        )
        if not rows:
            raise ApiError(
                404,
                "no_run_for_subject",
                "the subject has no performance run",
                subject=aid,
            )
        return str(rows[0]["run_id"]), {"by": "subject", "subject_id": aid}
    if kw.get(f"{prefix}_env"):
        eid = uuid_arg(kw[f"{prefix}_env"], f"{prefix}_env")
        bid = uuid_arg(kw.get("benchmark_id"), "benchmark_id")
        start, end = window(now, kw.get("from"), kw.get("to"), 30, 120)
        rows = run(
            """SELECT run_id, min(ts) AS first_ts FROM benchmark_runs
               WHERE benchmark_id = {b:UUID} AND ts >= {t_from:DateTime} AND ts < {t_to:DateTime} AND props['env.id'] = {e:String}
                 AND props['run.ref_only'] != '1' GROUP BY run_id ORDER BY first_ts DESC, run_id LIMIT 1""",
            {"b": bid, "e": eid, "t_from": ch_time(start), "t_to": ch_time(end)},
        )
        if not rows:
            raise ApiError(
                404,
                "no_run_for_env",
                "no run of that benchmark on the environment in the window",
                env_id=eid,
            )
        return str(rows[0]["run_id"]), {"by": "environment", "env_id": eid}
    raise ApiError(
        400, "bad_parameter", f"give {prefix}_run, {prefix}_subject or {prefix}_env"
    )


def _run_card(run, rid):
    first = run(
        "SELECT anyIf(props, props['env.id'] != '') AS rprops FROM benchmark_runs WHERE run_id = {r:UUID} AND props['run.ref_only'] != '1'",
        {"r": rid},
    )
    props = first[0]["rprops"] if first else {}
    env = props.get("env.id")
    labels = {}
    if env:
        f = run(
            "SELECT labels FROM v_environment_fingerprints WHERE env_id = {e:UUID}",
            {"e": env},
        )
        labels = dict(f[0]["labels"]) if f else {}
    s = subjects(run, {rid}).get(rid)
    inv = invalidations(run, {rid}).get(rid, [])
    return {"run_id": rid, "arch": props.get("arch") or (s or {}).get("arch"), "ts": props.get("run_utc"), "subject": s, "env_id": env, "labels": labels,
            "pinning": labels.get(POLICY_LABEL) or props.get("run.exec.cpu_pinning") or "none", "purpose": props.get("run.purpose"),
            "clean": props.get("run.clean.exclusive"), "invalidation_state": run_state(inv)}  # fmt: skip


def _cmp_runs(run, now, benchmark_id=None, metric=None, limit=200, **kw):
    limit = bounded(limit, "limit", 200, 1, 1000)
    a, ra = _side(run, now, "a", {**kw, "benchmark_id": benchmark_id})
    b, rb = _side(run, now, "b", {**kw, "benchmark_id": benchmark_id})
    bids = {uuid_arg(benchmark_id, "benchmark_id")} if benchmark_id else None
    va, vb = _vals(run, a, bids, metric), _vals(run, b, bids, metric)
    meta = bench_meta(run, {k[0] for k in {**va, **vb}})
    pol = policy(run)
    rows = []
    for k in sorted(
        set(va) | set(vb), key=lambda k: ((meta.get(k[0]) or {}).get("name", ""), k)
    ):
        x, y = va.get(k), vb.get(k)
        d = (
            None
            if not (x and y and x["value"])
            else (y["value"] - x["value"]) / abs(x["value"]) * 100
        )
        noise = max(
            [c for c in (x and x["cv_pct"], y and y["cv_pct"]) if c is not None],
            default=None,
        )
        hib = higher_is_better(pol, k[2])
        rows.append({"benchmark_id": k[0], "name": (meta.get(k[0]) or {}).get("name"), "backend": k[1], "metric": k[2], "a": x, "b": y,
                     "delta_pct": rnd(d, 2), "worse_pct": rnd(None if d is None else (-d if hib else d), 2), "noise_pct": rnd(noise, 2),
                     "within_noise": None if d is None or noise is None else abs(d) <= 2 * noise, "higher_is_better": hib})  # fmt: skip
    ca, cb = _run_card(run, a), _run_card(run, b)
    same = {"subject": (ca["subject"] or {}).get("artifact_id") == (cb["subject"] or {}).get("artifact_id"), "env_id": ca["env_id"] == cb["env_id"],
            "arch": ca["arch"] == cb["arch"], "pinning": ca["pinning"] == cb["pinning"], "clean": ca["clean"] == cb["clean"]}  # fmt: skip
    return {"a": {**ca, "resolved": ra}, "b": {**cb, "resolved": rb}, "comparable": same, "fair": all(same[k] for k in ("env_id", "arch", "pinning")),
            "rows": rows[:limit]}, {"truncated": len(rows) > limit}  # fmt: skip


def _vals(run, rid, bids, metric):
    out = {}
    groups = {}
    for r in _run_rows(run, rid, bids, valid_only=True):
        groups.setdefault((str(r["benchmark_id"]), r["backend"]), []).append(r)
    for (bid, be), rs in groups.items():
        for m, st in summarize_repeats(rs).items():
            if metric in (None, m):
                out[(bid, be, m)] = {
                    "value": st["value"],
                    "cv_pct": st["cv_pct"],
                    "n_repeats": st["n_repeats"],
                    "n_samples": st["n_samples"],
                    "min": st["min"],
                    "max": st["max"],
                }
    return out


def _cmp_derive(
    run,
    now,
    derive="decode_per_token",
    run_ids=None,
    n_benchmark=None,
    one_benchmark=None,
    n_tokens=None,
    metric="latency_ms",
    **_,
):
    if derive != "decode_per_token":
        raise ApiError(
            400, "bad_parameter", "derive must be decode_per_token", parameter="derive"
        )
    ids = [
        uuid_arg(x, "run_ids")
        for x in (
            run_ids if isinstance(run_ids, list) else str(run_ids or "").split(",")
        )
        if x
    ]
    if not ids or len(ids) > 50:
        raise ApiError(
            400, "bad_parameter", "run_ids must list 1..50 runs", parameter="run_ids"
        )
    nb, ob = (
        uuid_arg(n_benchmark, "n_benchmark"),
        uuid_arg(one_benchmark, "one_benchmark"),
    )
    n = (
        bounded(n_tokens, "n_tokens", None, 2, 10_000_000)
        if n_tokens is not None
        else None
    )
    if n is None:
        raise ApiError(
            400,
            "bad_parameter",
            "n_tokens is required (the N of latency@N)",
            parameter="n_tokens",
        )
    out, refused = [], []
    pol_cards = {}
    for rid in ids:
        rows = _run_rows(run, rid, {nb, ob}, valid_only=True)
        env = {
            str(r["benchmark_id"]): {
                x["props"].get("env.id")
                for x in rows
                if str(x["benchmark_id"]) == str(r["benchmark_id"])
            }
            for r in rows
        }
        if not rows or nb not in env or ob not in env:
            refused.append(
                {
                    "run_id": rid,
                    "reason": "missing_case",
                    "detail": "the run lacks latency@N or latency@1",
                }
            )
            continue
        if env[nb] != env[ob] or len(env[nb]) != 1 or None in env[nb] or "" in env[nb]:
            refused.append(
                {
                    "run_id": rid,
                    "reason": "env_id_differs",
                    "env_ids": sorted(map(str, env[nb] | env[ob])),
                }
            )
            continue
        vals = {}
        for bid in (nb, ob):
            st = summarize_repeats(
                [r for r in rows if str(r["benchmark_id"]) == bid]
            ).get(metric)
            vals[bid] = st
        if not vals[nb] or not vals[ob]:
            refused.append(
                {"run_id": rid, "reason": "missing_metric", "detail": metric}
            )
            continue
        card = pol_cards.setdefault(rid, _run_card(run, rid))
        per_rep = []
        mn = {x["repeat"]: x["median"] for x in vals[nb]["repeats"]}
        mo = {x["repeat"]: x["median"] for x in vals[ob]["repeats"]}
        for rp in sorted(set(mn) & set(mo)):
            per_rep.append(rnd((mn[rp] - mo[rp]) / (n - 1), 4))
        out.append({"run_id": rid, "value": rnd((vals[nb]["value"] - vals[ob]["value"]) / (n - 1), 4), "latency_n": vals[nb]["value"],
                    "latency_1": vals[ob]["value"], "n_tokens": n, "per_repeat": per_rep, "cv_pct": rnd(cv_pct(per_rep), 2) if len(per_rep) > 1 else None,
                    "env_id": card["env_id"], "subject_id": (card["subject"] or {}).get("artifact_id"), "pinning": card["pinning"]})  # fmt: skip
    if not out:
        raise ApiError(
            422,
            "derive_refused",
            "no run qualifies for the derived metric",
            refused=refused,
        )
    return {
        "derive": "decode_per_token",
        "formula": "(latency@N - latency@1) / (N - 1)",
        "metric": metric,
        "results": out,
        "refused": refused,
    }, {}


def _cmp_effect(run, now, benchmark_id=None, metric=None, arch=None, label=POLICY_LABEL, group_by="node_class", subject_id=None, purpose=None,
                source_job=None, noise_k=2.0, **kw):  # fmt: skip
    bid = uuid_arg(benchmark_id, "benchmark_id")
    if not metric:
        raise ApiError(400, "bad_parameter", "metric is required", parameter="metric")
    start, end = window(
        now, kw.get("from"), kw.get("to"), 60, MAX_WINDOW_DAYS["effect"]
    )
    bm = bench_meta(run, {bid}).get(bid)
    if not bm:
        raise ApiError(404, "unknown_benchmark", "no such benchmark", benchmark_id=bid)
    if (bm["props"] or {}).get("record_type", "") in TRACE_TYPES:
        raise ApiError(
            422,
            "not_a_measurement_series",
            "record_type has no measured series",
            record_type=bm["props"].get("record_type"),
        )
    rows = run(
        """SELECT run_id, toUInt16OrZero(props['run.repeat']) AS repeat, any(props) AS rprops,
                  arrayFlatten(groupArray(measurements[{m:String}])) AS samples
           FROM benchmark_runs
           WHERE component = {c:String} AND benchmark_id = {b:UUID} AND ts >= {t_from:DateTime} AND ts < {t_to:DateTime}
             AND mapContains(measurements, {m:String}) AND props['run.ref_only'] != '1' AND props['run.record_type'] != 'reference'
           GROUP BY run_id, repeat LIMIT 20000""",
        {
            "c": bm["component"],
            "b": bid,
            "m": metric,
            "t_from": ch_time(start),
            "t_to": ch_time(end),
        },
    )
    rows = [r for r in rows if (not arch or _row_arch(r["rprops"], arch)) and (not purpose or r["rprops"].get("run.purpose") == purpose)
            and (not source_job or _row_job(r["rprops"]) == source_job)]  # fmt: skip
    env_ids = sorted(
        {r["rprops"].get("env.id") for r in rows if r["rprops"].get("env.id")}
    )
    fps = {}
    if env_ids:
        fps = {
            str(f["env_id"]): dict(f["labels"])
            for f in run(
                "SELECT env_id, labels FROM v_environment_fingerprints WHERE env_id IN {e:Array(UUID)} LIMIT 5000",
                {"e": env_ids},
            )
        }
    ids = sorted({str(r["run_id"]) for r in rows})
    inv = invalidations(run, set(ids))
    subj = subjects(run, set(ids))
    merged = {}
    for r in rows:
        rid = str(r["run_id"])
        if invalid_repeat(inv.get(rid, []), bid, r["repeat"]):
            continue
        m = merged.setdefault(
            rid, {"run_id": rid, "rprops": r["rprops"], "samples": []}
        )
        m["samples"].extend(r["samples"])
    cells = {}
    for r in merged.values():
        rid = str(r["run_id"])
        if subject_id and (subj.get(rid) or {}).get("artifact_id") != subject_id:
            continue
        lab = fps.get(r["rprops"].get("env.id"), {})
        xs = [float(x) for x in r["samples"] if x != 0]
        if not xs:
            continue
        gk = lab.get(group_by) or "unknown"
        pv = lab.get(label) or r["rprops"].get("run.exec.cpu_pinning") or "none"
        cells.setdefault((gk, pv), []).append(
            (median(xs), (subj.get(rid) or {}).get("artifact_id"))
        )
    groups = {}
    for (gk, pv), vs in cells.items():
        vals = [v for v, _ in vs]
        groups.setdefault(gk, {})[pv] = {
            "runs": len(vals),
            "median": rnd(median(vals)),
            "cv_pct": rnd(cv_pct(vals), 2),
            "subjects": len({s for _, s in vs}),
        }
    effects = []
    pol = policy(run)
    for gk in sorted(groups):
        base = groups[gk].get("none")
        for pv in sorted(groups[gk]):
            if pv == "none":
                continue
            cur = groups[gk][pv]
            if not base:
                effects.append(
                    {
                        group_by: gk,
                        label: pv,
                        "effect_pct": None,
                        "reason": "no_unpinned_baseline",
                    }
                )
                continue
            d = (cur["median"] - base["median"]) / abs(base["median"]) * 100
            noise = max(
                c for c in (cur["cv_pct"], base["cv_pct"], 0.0) if c is not None
            )
            effects.append({group_by: gk, label: pv, "baseline_median": base["median"], "median": cur["median"], "effect_pct": rnd(d, 2),
                            "better_pct": rnd(-d if not higher_is_better(pol, metric) else d, 2), "noise_pct": rnd(noise, 2),
                            "within_noise": abs(d) <= float(noise_k) * noise, "n_baseline": base["runs"], "n": cur["runs"],
                            "mixed_subjects": cur["subjects"] > 1 or base["subjects"] > 1})  # fmt: skip
    return {"benchmark": _bench_head(bm, bid), "metric": metric, "label": label, "group_by": group_by, "groups": groups, "effects": effects,
            "rule": f"within_noise = |effect| <= {noise_k} x max(CV of the two groups' run medians)"}, {"window": {"from": iso(start), "to": iso(end)}}  # fmt: skip


def _cmp_reference(run, now, run_id=None, key=None, version=None, metric=None, **_):
    rid = uuid_arg(run_id, "run_id")
    if not key:
        raise ApiError(400, "bad_parameter", "key is required", parameter="key")
    ref = _ref_values(run, key, version)
    ours = _vals(run, rid, {v["benchmark_id"] for v in ref["values"]} or None, metric)
    pol = policy(run)
    rows = []
    for v in ref["values"]:
        if metric and v["metric"] != metric:
            continue
        mine = next(
            (
                x
                for k, x in ours.items()
                if k[0] == v["benchmark_id"] and k[2] == v["metric"]
            ),
            None,
        )
        d = (
            None
            if not mine or not v["value"]
            else (mine["value"] - v["value"]) / abs(v["value"]) * 100
        )
        rows.append({"benchmark_id": v["benchmark_id"], "name": v["name"], "metric": v["metric"], "reference": v["value"], "reference_backend": v["backend"],
                     "ours": None if not mine else mine["value"], "delta_pct": rnd(d, 2), "higher_is_better": higher_is_better(pol, v["metric"])})  # fmt: skip
    return {"reference": ref["reference"], "run": _run_card(run, rid), "rows": rows,
            "note": "a reference is a comparison target, never a baseline; comparability_verdict says how far the numbers are comparable"}, {}  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# routing


def route(run, path, params=None, now=None):
    """Dispatch a GET of /api/v2/perf/... to a function: returns (http_status, body)."""
    params = dict(params or {})
    parts = [x for x in path.strip("/").split("/")]
    try:
        if parts[:3] != ["api", "v2", "perf"]:
            raise ApiError(404, "not_found", "unknown route")
        rest = parts[3:]
        if rest == ["overview"]:
            return 200, overview(run, now, **params)
        if rest == ["suites"]:
            return 200, suites(run, now, **params)
        if rest == ["compare"]:
            return 200, compare(run, now, **params)
        if rest == ["references"]:
            return 200, references(run, now, **params)
        if len(rest) == 3 and rest[0] == "benchmarks" and rest[2] == "series":
            return 200, series(
                run,
                rest[1],
                now,
                **{
                    ("t_from" if k == "from" else "t_to" if k == "to" else k): v
                    for k, v in params.items()
                },
            )
        if len(rest) >= 2 and rest[0] == "runs":
            fn = {
                None: run_detail,
                "environment": environment,
                "steps": steps,
                "spans": spans,
            }.get(rest[2] if len(rest) == 3 else None)
            if fn and len(rest) in (2, 3):
                return 200, fn(run, rest[1], now, **params)
        raise ApiError(404, "not_found", "unknown route")
    except ApiError as e:
        return e.status, e.body()
    except (ValueError, TypeError) as e:
        return 400, ApiError(
            400, "bad_parameter", f"a parameter has the wrong type: {e}"
        ).body()

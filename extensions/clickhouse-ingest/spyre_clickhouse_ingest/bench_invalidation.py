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

"""Invalidation events: preview what a scope hits, and write result_invalidations rows.

`bench-invalidation` bundles (`results --from-bundle`) and in-process callers (the dashboard backend) both go through
`apply`, so the threshold, the state machine and the ordering rule have one implementation. The matching itself is
the SQL view v_bench_scope_hits; nothing here re-implements it.
Bundle: bundle.json {"kind": "bench-invalidation"} and results/invalidation.json {"events": [...]}.
Exit codes as the other kinds: 0 ingested or already ingested, 1 failed (retry), 2 rejected.
"""

import hashlib
import json
import os
import regex as re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .offline import (
    BUNDLE_FILE,
    FAILED,
    BundleError,
    bundle_files,
    current_user,
    digest,
    sha256,
)

KIND = "bench-invalidation"
EVENTS_FILE = "results/invalidation.json"
SCOPES = ("run", "run_benchmark", "node", "device", "placement")
STATES = ("suspect", "invalidated", "restored")
REASON_CODES = (
    "faulty_hardware",
    "measurement_bug",
    "bad_config",
    "under_investigation",
    "other",
)
THRESHOLD_ENV = "BENCH_INVALIDATION_CONFIRM_ABOVE"
DEFAULT_THRESHOLD = 50
SAMPLE_RUNS = 20
DETAIL_ROWS = 5000
CHANGE_ROWS = 200
NIL = "00000000-0000-0000-0000-000000000000"
WINDOWED = ("node", "device", "placement")
# Which states may follow which; no state may repeat (that is a no-op, not a transition).
NEXT = {
    None: ("suspect", "invalidated"),
    "suspect": ("invalidated", "restored"),
    "invalidated": ("suspect", "restored"),
    "restored": ("suspect", "invalidated"),
}
SCOPE_COLUMNS = (
    "scope_kind",
    "run_id",
    "benchmark_id",
    "repeat",
    "node",
    "device_serial",
    "placement_id",
    "window_from",
    "window_to",
)
COLUMNS = (
    "ts", "invalidation_id", "scope_kind", "run_id", "benchmark_id", "repeat", "node", "device_serial", "placement_id",
    "window_from", "window_to", "matched_runs", "state", "reason_code", "reason", "actor", "evidence_url", "props",
)  # fmt: skip
HITS = (
    "v_bench_scope_hits(scope_kind = {sk:String}, run_id = {run:String}, benchmark_id = {bm:String}, "
    "repeat = {rep:Int32}, node = {node:String}, device_serial = {dev:String}, placement_id = {pl:String}, "
    "window_from = {wf:String}, window_to = {wt:String})"
)
VERDICT_VIEW = "v_benchmark_metric_verdicts"
SERIES = "component, benchmark_id, backend, arch, source_job"


class InvalidationError(ValueError):
    """The event is wrong, or needs a confirmation it does not carry: nothing was written."""


def threshold(value=None):
    raw = value if value is not None else os.environ.get(THRESHOLD_ENV, "")
    try:
        return int(raw) if str(raw).strip() else DEFAULT_THRESHOLD
    except ValueError:
        raise InvalidationError(f"{THRESHOLD_ENV} is not a number: {raw!r}") from None


def _uuid(value, what):
    try:
        u = uuid.UUID(str(value))
    except (ValueError, AttributeError):
        raise InvalidationError(f"{what} is not a UUID: {value!r}") from None
    if str(u) == NIL:
        raise InvalidationError(f"{what} is the nil UUID")
    return str(u)


def _when(value, what):
    if value in (None, ""):
        return None
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise InvalidationError(
            f"{what} is not an ISO-8601 timestamp: {value!r}"
        ) from None
    d = (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    return d.replace(microsecond=d.microsecond // 1000 * 1000)


def normalize(ev):
    """One event dict (bundle shape: scope nested) as canonical fields; raises InvalidationError."""
    if not isinstance(ev, dict):
        raise InvalidationError("an event is a JSON object")
    scope = ev.get("scope") or {}
    if not isinstance(scope, dict):
        raise InvalidationError("scope is an object")
    out = {"invalidation_id": _uuid(ev.get("invalidation_id"), "invalidation_id")}
    out["scope_kind"] = ev.get("scope_kind") or ""
    if out["scope_kind"] == "query":
        raise InvalidationError("scope_kind query is reserved and not enabled")
    if out["scope_kind"] not in SCOPES:
        raise InvalidationError(
            f"scope_kind {out['scope_kind']!r}: one of {', '.join(SCOPES)}"
        )
    unknown = sorted(
        set(scope)
        - {"run_id", "benchmark_id", "repeat", "node", "device_serial", "placement_id"}
    )
    if unknown:
        raise InvalidationError(f"unknown scope field(s): {unknown}")
    kind = out["scope_kind"]
    out["run_id"] = (
        _uuid(scope["run_id"], "scope.run_id") if scope.get("run_id") else None
    )
    out["benchmark_id"] = (
        _uuid(scope["benchmark_id"], "scope.benchmark_id")
        if scope.get("benchmark_id")
        else None
    )
    rep = scope.get("repeat")
    if rep is not None and (
        isinstance(rep, bool) or not isinstance(rep, int) or not 0 <= rep < 65536
    ):
        raise InvalidationError(f"scope.repeat is not a repeat number: {rep!r}")
    out["repeat"] = rep
    out["node"] = str(scope.get("node") or "").strip()
    out["device_serial"] = str(scope.get("device_serial") or "").strip()
    out["placement_id"] = (
        _uuid(scope["placement_id"], "scope.placement_id")
        if scope.get("placement_id")
        else ""
    )
    need = {
        "run": ("run_id",), "run_benchmark": ("run_id", "benchmark_id"), "node": ("node",),
        "device": ("device_serial",), "placement": ("placement_id",),
    }[kind]  # fmt: skip
    miss = [k for k in need if not out[k]]
    if miss:
        raise InvalidationError(
            f"scope_kind {kind} needs scope.{', scope.'.join(miss)}"
        )
    extra = [
        k
        for k in SCOPE_COLUMNS[1:7]
        if out[k] not in (None, "")
        and k not in need
        and not (k == "repeat" and kind == "run_benchmark")
    ]
    if extra:
        raise InvalidationError(
            f"scope_kind {kind} does not take scope.{', scope.'.join(extra)}"
        )
    out["window_from"], out["window_to"] = (
        _when(ev.get("window_from"), "window_from"),
        _when(ev.get("window_to"), "window_to"),
    )
    if (out["window_from"] or out["window_to"]) and kind not in WINDOWED:
        raise InvalidationError(
            f"a window applies to node, device and placement scopes, not {kind}"
        )
    if (
        out["window_from"]
        and out["window_to"]
        and out["window_from"] > out["window_to"]
    ):
        raise InvalidationError("window_from is after window_to")
    out["state"] = ev.get("state") or ""
    if out["state"] not in STATES:
        raise InvalidationError(f"state {out['state']!r}: one of {', '.join(STATES)}")
    out["reason_code"] = ev.get("reason_code") or ""
    if out["reason_code"] not in REASON_CODES:
        raise InvalidationError(
            f"reason_code {out['reason_code']!r}: one of {', '.join(REASON_CODES)}"
        )
    out["reason"] = str(ev.get("reason") or "").strip()
    if not out["reason"]:
        raise InvalidationError("a reason is required")
    out["evidence_url"] = str(ev.get("evidence_url") or "").strip()
    if out["evidence_url"] and not re.match(r"^https?://\S+$", out["evidence_url"]):
        raise InvalidationError("evidence_url is an http(s) URL")
    out["confirmed"] = ev.get("confirmed") is True
    out["confirm_token"] = str(ev.get("confirm_token") or "")
    out["ts"] = _when(ev.get("ts"), "ts")
    return out


def event_from_flags(**kw):
    """The bundle-shaped event for scope/state flags (used by the CLI): scope fields are those of SCOPE_COLUMNS."""
    scope = {
        k: kw[k]
        for k in (
            "run_id",
            "benchmark_id",
            "repeat",
            "node",
            "device_serial",
            "placement_id",
        )
        if kw.get(k) not in (None, "")
    }
    return {
        "invalidation_id": kw.get("invalidation_id") or str(uuid.uuid4()),
        "scope_kind": kw.get("scope_kind"), "scope": scope, "window_from": kw.get("window_from"), "window_to": kw.get("window_to"),
        "state": kw.get("state"), "reason_code": kw.get("reason_code"), "reason": kw.get("reason"),
        "evidence_url": kw.get("evidence_url"), "confirmed": bool(kw.get("confirmed")), "confirm_token": kw.get("confirm_token"),
        "ts": kw.get("ts"),
    }  # fmt: skip


def _rows(client, sql, parameters=None):
    result = client.query(sql, parameters=parameters or {})
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def hit_params(ev):
    iso = lambda d: d.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if d else ""  # noqa: E731 - ClickHouse parses this form
    return {
        "sk": ev["scope_kind"], "run": ev["run_id"] or "", "bm": ev["benchmark_id"] or "",
        "rep": -1 if ev["repeat"] is None else ev["repeat"], "node": ev["node"], "dev": ev["device_serial"],
        "pl": ev["placement_id"], "wf": iso(ev["window_from"]), "wt": iso(ev["window_to"]),
    }  # fmt: skip


def current_states(client, ids):
    """{invalidation_id: row of v_invalidation_state plus the id's newest event ts}."""
    if not ids:
        return {}
    rows = _rows(
        client,
        "SELECT toString(s.invalidation_id) AS invalidation_id, s.state AS state, s.scope_kind AS scope_kind, "
        "toString(s.run_id) AS run_id, toString(s.benchmark_id) AS benchmark_id, s.repeat AS repeat, s.node AS node, "
        "s.device_serial AS device_serial, s.placement_id AS placement_id, "
        "toUnixTimestamp64Milli(s.window_from) AS window_from_ms, toUnixTimestamp64Milli(s.window_to) AS window_to_ms, "
        "toUnixTimestamp64Milli(s.updated_at) AS updated_ms "
        "FROM v_invalidation_state AS s WHERE s.invalidation_id IN {ids:Array(UUID)}",
        {"ids": sorted(set(ids))},
    )
    return {r["invalidation_id"]: r for r in rows}


def _stored_scope(row):
    """A stored creation event as normalize() canonical scope fields."""
    out = {
        k: row[k]
        for k in ("scope_kind", "node", "device_serial", "placement_id", "repeat")
    }
    out["run_id"] = row["run_id"] if row["run_id"] and row["run_id"] != "None" else None
    out["benchmark_id"] = (
        row["benchmark_id"]
        if row["benchmark_id"] and row["benchmark_id"] != "None"
        else None
    )
    out["placement_id"] = (out["placement_id"] or "").lower()
    for k in (
        "window_from",
        "window_to",
    ):  # epoch ms: a client may read a DateTime64 in its own zone
        out[k] = _utc_ms(row[f"{k}_ms"])
    return out


def _utc_ms(ms):
    return None if ms is None else datetime.fromtimestamp(int(ms) / 1000, timezone.utc)


def _same_scope(ev, stored):
    mine = {k: ev[k] for k in stored}
    mine["placement_id"] = (mine["placement_id"] or "").lower()
    return mine == stored


def hits(client, ev):
    """(run ids, repeats, detail rows, truncated) a scope hits right now; detail is capped at DETAIL_ROWS."""
    p = hit_params(ev)
    head = _rows(
        client,
        f"SELECT uniqExact(run_id) AS runs, uniqExact(run_id, repeat) AS repeats, count() AS n_rows FROM {HITS}",
        p,
    )[0]
    detail = _rows(
        client,
        "SELECT toString(run_id) AS run_id, toString(benchmark_id) AS benchmark_id, repeat, component, backend, arch, "
        f"source_job, name, toString(run_ts) AS run_ts_s, is_series FROM {HITS} ORDER BY run_ts, run_id, benchmark_id, repeat LIMIT {DETAIL_ROWS}",
        p,
    )
    return (
        int(head["runs"]),
        int(head["repeats"]),
        detail,
        int(head["n_rows"]) > len(detail),
    )


def run_summary(client, ev):
    """(run count, hash of the exact run set, first runs): the set itself never leaves the server."""
    row = _rows(
        client,
        "SELECT uniqExact(run_id) AS runs, hex(sipHash128(arraySort(groupUniqArray(run_id)))) AS h, "
        f"arraySlice(arraySort(groupUniqArray(toString(run_id))), 1, {SAMPLE_RUNS}) AS sample FROM {HITS}",
        hit_params(ev),
    )[0]
    return int(row["runs"]), row["h"], list(row["sample"])


def confirm_token(ev, runs_hash):
    """Binds a confirmation to the scope, the target state and the exact set of runs the preview showed."""
    body = json.dumps(
        [
            ev["scope_kind"],
            [ev[k] for k in SCOPE_COLUMNS[1:7]],
            str(ev["window_from"]),
            str(ev["window_to"]),
            ev["state"],
            runs_hash,
        ]
    )
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def _verdict_sql(client):
    """The verdict view's SELECT with its read of v_invalidated_runs left to the caller."""
    rows = _rows(
        client,
        "SELECT as_select FROM system.tables WHERE database = currentDatabase() AND name = {n:String}",
        {"n": VERDICT_VIEW},
    )
    if not rows:
        raise InvalidationError(f"{VERDICT_VIEW} is not in this database")
    text, n = re.subn(
        r"(?:`?\w+`?\.)?`?v_invalidated_runs`?(\s+AS\s+x\b)",
        r"({hit_rows})\1",
        rows[0]["as_select"],
    )
    if n != 1:
        raise InvalidationError(
            f"{VERDICT_VIEW} reads v_invalidated_runs {n} times; cannot rewrite it"
        )
    return text


def _after_rows(ev, state_after):
    """The v_invalidated_runs rows with this invalidation's replaced by what it would be in `state_after`."""
    kind = ev["scope_kind"]
    cols = {
        "run": "run_id, CAST(NULL AS Nullable(UUID)), CAST(NULL AS Nullable(UInt16))",
        "run_benchmark": "run_id, CAST(benchmark_id AS Nullable(UUID)), CAST(if({rep:Int32} < 0, NULL, {rep:Int32}) AS Nullable(UInt16))",
    }.get(
        kind, "run_id, CAST(NULL AS Nullable(UUID)), CAST(repeat AS Nullable(UInt16))"
    )
    keep = (
        "SELECT invalidation_id, CAST(run_id AS Nullable(UUID)) AS run_id, CAST(benchmark_id AS Nullable(UUID)) AS benchmark_id, "
        "CAST(repeat AS Nullable(UInt16)) AS repeat, CAST(state AS String) AS state, CAST(reason_code AS String) AS reason_code "
        "FROM v_invalidated_runs WHERE invalidation_id != toUUID({inv:String})"
    )
    if state_after == "restored":
        return keep
    return f"{keep} UNION ALL SELECT DISTINCT toUUID({{inv:String}}), {cols}, {{st:String}}, 'preview' FROM {HITS}"


def verdict_changes(client, ev, state_after):
    """Verdict rows of the affected series that change when this invalidation is in `state_after`."""
    base = _verdict_sql(client)
    keys = f"({SERIES}) IN (SELECT {SERIES} FROM {HITS} WHERE is_series)"
    cols = f"run_id, benchmark_id, metric, name, status, reason, toString(ts) AS ts_s, {SERIES}"
    params = {**hit_params(ev), "inv": ev["invalidation_id"], "st": state_after}
    before = _rows(client, f"SELECT {cols} FROM {VERDICT_VIEW} WHERE {keys}", params)
    after = _rows(
        client,
        f"SELECT {cols} FROM ({base.replace('{hit_rows}', _after_rows(ev, state_after))}) WHERE {keys}",
        params,
    )
    key = lambda r: (str(r["run_id"]), str(r["benchmark_id"]), r["metric"])  # noqa: E731
    b, a = {key(r): r for r in before}, {key(r): r for r in after}
    changes = []
    for k in sorted(set(a) | set(b), key=lambda k: ((b.get(k) or a.get(k))["ts_s"], k)):
        was, now = b.get(k), a.get(k)
        was_s, now_s = (was or {}).get("status"), (now or {}).get("status")
        if was_s == now_s and (was or {}).get("reason") == (now or {}).get("reason"):
            continue
        row = was or now
        changes.append({
            "run_id": k[0], "benchmark_id": k[1], "metric": k[2], "name": row["name"], "ts": row["ts_s"],
            "before": was_s or "absent", "after": now_s or "absent",
            "before_reason": (was or {}).get("reason", ""), "after_reason": (now or {}).get("reason", ""),
        })  # fmt: skip
    moves = {}
    for c in changes:
        moves[f"{c['before']}->{c['after']}"] = (
            moves.get(f"{c['before']}->{c['after']}", 0) + 1
        )
    return {
        "available": True,
        "total": len(changes),
        "transitions": moves,
        "rows": changes[:CHANGE_ROWS],
        "truncated": len(changes) > CHANGE_ROWS,
    }


def preview(client, ev, limit=None, with_verdicts=True):
    """What the bundle-shaped event `ev` would hit and change, from the store; writes nothing."""
    ev = normalize(ev)
    state = current_states(client, [ev["invalidation_id"]]).get(ev["invalidation_id"])
    if state:
        stored = _stored_scope(state)
        if not _same_scope(ev, stored):
            raise InvalidationError(
                f"{ev['invalidation_id']} exists with a different scope; an event repeats its invalidation's scope"
            )
    current = state["state"] if state else None
    noop = ev["state"] == current
    if not noop and ev["state"] not in NEXT[current]:
        raise InvalidationError(
            f"{current or 'a new invalidation'} cannot become {ev['state']}"
        )
    runs, repeats, detail, truncated = hits(client, ev)
    limit = threshold(limit)
    _, runs_hash, sample = run_summary(client, ev)
    series, benchmarks, by_run = {}, {}, {}
    for r in detail:
        by_run.setdefault(
            r["run_id"],
            {
                "run_id": r["run_id"],
                "run_ts": r["run_ts_s"],
                "repeats": set(),
                "benchmarks": set(),
            },
        )
        by_run[r["run_id"]]["repeats"].add(r["repeat"])
        by_run[r["run_id"]]["benchmarks"].add(r["benchmark_id"])
        benchmarks[r["benchmark_id"]] = {
            "benchmark_id": r["benchmark_id"],
            "name": r["name"],
            "component": r["component"],
        }
        if r["is_series"]:
            k = (
                r["component"],
                r["benchmark_id"],
                r["backend"],
                r["arch"],
                r["source_job"],
            )
            series.setdefault(
                k,
                {
                    "component": k[0],
                    "benchmark_id": k[1],
                    "backend": k[2],
                    "arch": k[3],
                    "source_job": k[4],
                    "name": r["name"],
                    "runs": set(),
                },
            )["runs"].add(r["run_id"])
    out = {
        "invalidation_id": ev["invalidation_id"], "scope_kind": ev["scope_kind"], "state_before": current, "state_after": ev["state"],
        "noop": noop, "matched_runs": runs, "matched_repeats": repeats, "truncated": truncated, "threshold": limit,
        "needs_confirm": runs > limit, "confirm_token": confirm_token(ev, runs_hash),
        "benchmarks": sorted(benchmarks.values(), key=lambda b: (b["name"], b["benchmark_id"])),
        "series": sorted(({**s, "runs": len(s["runs"])} for s in series.values()), key=lambda s: (s["component"], s["name"], s["benchmark_id"])),
        "runs": [{**r, "repeats": sorted(r["repeats"]), "benchmarks": len(r["benchmarks"])} for r in by_run.values()][: DETAIL_ROWS],
        "sample_runs": sample,
    }  # fmt: skip
    out["verdict_changes"] = {"available": False, "reason": "skipped"}
    if with_verdicts and not noop and runs:
        try:
            out["verdict_changes"] = verdict_changes(client, ev, ev["state"])
        except Exception as e:  # noqa: BLE001 - a preview without the verdict diff is still a preview
            out["verdict_changes"] = {
                "available": False,
                "reason": f"{type(e).__name__}: {e}",
            }
    return out


def next_ts(requested, last, now=None):
    """Strictly after `last` (the id's newest event), in ClickHouse's millisecond resolution."""
    clock = now or datetime.now(
        timezone.utc
    )  # zone-aware: a naive value is read in the client's zone
    t = (
        requested if requested and requested <= clock else clock
    )  # a future time would pin the id's clock
    t = t.replace(microsecond=t.microsecond // 1000 * 1000)
    if last is not None:
        last = last if last.tzinfo else last.replace(tzinfo=timezone.utc)
        if t <= last:
            t = last + timedelta(milliseconds=1)
    return t


def _replayed(client, evs, props):
    """(id, state) pairs this bundle already recorded, so re-running an old bundle cannot undo a later restore."""
    sha = str((props or {}).get("bundle_sha256") or "")
    if not sha:
        return set()
    rows = _rows(
        client,
        "SELECT DISTINCT toString(invalidation_id) AS i, state FROM result_invalidations WHERE props['bundle_sha256'] = {sha:String}",
        {"sha": sha},
    )
    return {(r["i"], r["state"]) for r in rows}


def apply(
    client,
    events,
    *,
    actor,
    channel="api",
    props=None,
    confirm_above=None,
    dry_run=False,
    now=None,
):
    """Validate and write events in order; returns one result per event. Nothing is written if any event is refused.

    Per event: a state equal to the id's latest is a no-op; a window or query covering more runs than `confirm_above`
    needs `confirmed: true` (and, when it names one, the preview's confirm_token); `matched_runs` is recomputed here.
    """
    if not actor:
        raise InvalidationError("an actor is required")
    limit = threshold(confirm_above)
    evs = [normalize(e) for e in events]
    if not evs:
        raise InvalidationError("no events")
    states = current_states(client, [e["invalidation_id"] for e in evs])
    last_ts, latest, results, rows, first_scope = {}, {}, [], [], {}
    replayed = _replayed(client, evs, props)
    for e in evs:
        i = e["invalidation_id"]
        st = states.get(i)
        stored = _stored_scope(st) if st else first_scope.get(i)
        first_scope.setdefault(
            i,
            {k: e[k] for k in SCOPE_COLUMNS}
            | {"placement_id": (e["placement_id"] or "").lower()},
        )
        if i not in last_ts and st:
            last_ts[i] = _utc_ms(st["updated_ms"])
        cur = latest.get(i, st["state"] if st else None)
        if stored and not _same_scope(e, stored):
            raise InvalidationError(
                f"{i} exists with a different scope; an event repeats its invalidation's scope"
            )
        if e["state"] == cur or (i, e["state"]) in replayed:
            results.append(
                {
                    "invalidation_id": i,
                    "state": e["state"],
                    "action": "noop",
                    "matched_runs": 0,
                }
            )
            continue
        if e["state"] not in NEXT[cur]:
            raise InvalidationError(
                f"{i}: {cur or 'a new invalidation'} cannot become {e['state']}"
            )
        n, runs_hash, sample = run_summary(client, e)
        if n > limit:
            if not e["confirmed"]:
                raise InvalidationError(
                    f"{i}: {n} runs exceed the confirmation threshold {limit}; preview it and confirm"
                )
            if e["confirm_token"] and e["confirm_token"] != confirm_token(e, runs_hash):
                raise InvalidationError(
                    f"{i}: the runs it covers changed since the preview (confirm token mismatch); preview again"
                )
        ts = next_ts(e["ts"], last_ts.get(i), now)
        last_ts[i], latest[i] = ts, e["state"]
        results.append(
            {
                "invalidation_id": i,
                "state": e["state"],
                "action": "write",
                "ts": ts.isoformat(),
                "matched_runs": n,
                "sample_runs": sample,
            }
        )
        rows.append([
            ts, uuid.UUID(i), e["scope_kind"], uuid.UUID(e["run_id"]) if e["run_id"] else None,
            uuid.UUID(e["benchmark_id"]) if e["benchmark_id"] else None, e["repeat"], e["node"], e["device_serial"],
            e["placement_id"], e["window_from"], e["window_to"], n, e["state"], e["reason_code"], e["reason"],
            actor, e["evidence_url"], {"channel": channel, **{k: str(v) for k, v in (props or {}).items()}},
        ])  # fmt: skip
    if rows and not dry_run:
        client.insert("result_invalidations", rows, column_names=list(COLUMNS))
    return results


# --- reading the log ----------------------------------------------------------------------------


def list_invalidations(client, state="", limit=100):
    """Invalidations at their latest state with the size of what they cover, newest update first."""
    if state and state not in STATES:
        raise InvalidationError(f"state {state!r}: one of {', '.join(STATES)}")
    rows = _rows(
        client,
        "SELECT toString(invalidation_id) AS invalidation_id, state, scope_kind, scope_key, reason_code, reason, actor, "
        "toString(created_at) AS created_s, toString(updated_at) AS updated_s, toString(window_from) AS from_s, "
        "toString(window_to) AS to_s, runs, repeats, benchmarks FROM v_bench_invalidation_summary "
        "WHERE ({st:String} = '' OR state = {st:String}) ORDER BY updated_at DESC, invalidation_id LIMIT {n:UInt32}",
        {"st": state, "n": int(limit)},
    )
    names = {
        "created_s": "created_at",
        "updated_s": "updated_at",
        "from_s": "window_from",
        "to_s": "window_to",
    }
    return [
        {
            names.get(k, k): (int(v) if k in ("runs", "repeats", "benchmarks") else v)
            for k, v in r.items()
        }
        for r in rows
    ]


def history(client, invalidation_id):
    """Every event of one invalidation, oldest first."""
    rows = _rows(
        client,
        "SELECT toString(ts) AS ts_s, state, actor, reason_code, reason, evidence_url, matched_runs, scope_kind, props "
        "FROM result_invalidations WHERE invalidation_id = {inv:UUID} ORDER BY ts, audit_timestamp, audit_uuid",
        {"inv": _uuid(invalidation_id, "invalidation_id")},
    )
    return [{("ts" if k == "ts_s" else k): v for k, v in r.items()} for r in rows]


def impact(client, invalidation_id, limit=DETAIL_ROWS):
    """The runs an invalidation covers now (whatever its state): one row per run with repeat and benchmark counts."""
    rows = _rows(
        client,
        "SELECT toString(run_id) AS run_id, toString(min(run_ts)) AS run_ts_s, uniqExact(repeat) AS repeats, uniqExact(benchmark_id) AS benchmarks "
        "FROM v_bench_invalidation_impact WHERE invalidation_id = {inv:UUID} GROUP BY run_id ORDER BY min(run_ts), run_id LIMIT {n:UInt32}",
        {"inv": _uuid(invalidation_id, "invalidation_id"), "n": int(limit)},
    )
    return [
        {**{k: v for k, v in r.items() if k != "run_ts_s"}, "run_ts": r["run_ts_s"]}
        for r in rows
    ]


def stored_event(client, invalidation_id):
    """The creation event of an id as a bundle-shaped event (scope only), or None."""
    row = current_states(client, [_uuid(invalidation_id, "invalidation_id")]).get(
        str(uuid.UUID(str(invalidation_id)))
    )
    if not row:
        return None
    s = _stored_scope(row)
    iso = lambda d: d.isoformat() if d else None  # noqa: E731
    scope = {
        k: s[k]
        for k in (
            "run_id",
            "benchmark_id",
            "repeat",
            "node",
            "device_serial",
            "placement_id",
        )
        if s[k] not in (None, "")
    }
    return {"invalidation_id": row["invalidation_id"], "scope_kind": s["scope_kind"], "scope": scope,
            "window_from": iso(s["window_from"]), "window_to": iso(s["window_to"]), "state": row["state"]}  # fmt: skip


# --- the bundle kind ----------------------------------------------------------------------------


def is_bundle(root):
    try:
        meta = json.loads((Path(root) / BUNDLE_FILE).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and meta.get("kind") == KIND


def load(root):
    """The bundle's events after the structural checks; the content checks are normalize()'s."""
    root = Path(root)
    meta = json.loads((root / BUNDLE_FILE).read_text())
    present = bundle_files(root)
    if present != [EVENTS_FILE]:
        raise BundleError(f"a {KIND} bundle holds only {EVENTS_FILE}: {present}")
    listed = meta.get("files")
    if listed is not None:
        if [f.get("path") if isinstance(f, dict) else None for f in listed] != [
            EVENTS_FILE
        ] or listed[0].get("sha256") != sha256(root / EVENTS_FILE):
            raise BundleError(
                "files[] does not match results/invalidation.json (sha256)"
            )
    try:
        doc = json.loads((root / EVENTS_FILE).read_text())
    except (ValueError, UnicodeDecodeError) as err:
        raise BundleError(f"{EVENTS_FILE} is not JSON: {err}") from None
    events = doc.get("events") if isinstance(doc, dict) else None
    if not isinstance(events, list) or not events:
        raise BundleError(
            f'{EVENTS_FILE} is {{"events": [...]}} with at least one event'
        )
    return events


def write_bundle(directory, events, uploader=""):
    """A sealed bundle directory for `events` (bundle-shaped dicts); returns its path."""
    directory = Path(directory)
    (directory / "results").mkdir(parents=True, exist_ok=True)
    (directory / EVENTS_FILE).write_text(
        json.dumps({"events": events}, indent=1, sort_keys=True) + "\n"
    )
    meta = {
        "schema_version": 1,
        "kind": KIND,
        "files": [{"path": EVENTS_FILE, "sha256": sha256(directory / EVENTS_FILE)}],
    }
    if uploader:
        meta["uploader"] = uploader
    (directory / BUNDLE_FILE).write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n"
    )
    return directory


def ingest(root, args, client=None):
    """`results --from-bundle` for this kind; returns the report dict (`status` ingested | duplicate | would-ingest)."""
    from .client import ClickHouse, ClickHouseEnv

    events = load(root)
    extra = dict(getattr(args, "bundle_prop", None) or [])
    actor = getattr(args, "uploader", "") or current_user()
    sha = digest(Path(root))
    if client is None:
        db = ClickHouseEnv.target_database()
        if not db:
            raise BundleError("no database: set CLICKHOUSE_DB_V2", FAILED)
        try:
            client = ClickHouse.connect(database=db)
            if args.dry_run:
                client.set_client_setting("readonly", "2")
        except Exception as err:  # noqa: BLE001 -- unreachable is a retry, not a verdict on the bundle
            raise BundleError(f"spyre_v2 unreachable: {err}", FAILED) from None
    try:
        results = apply(
            client,
            events,
            actor=actor,
            channel="bundle",
            props={**extra, "bundle_sha256": sha},
            dry_run=args.dry_run,
        )
    except InvalidationError as err:
        raise BundleError(str(err)) from None
    except BundleError:
        raise
    except Exception as err:  # noqa: BLE001 -- the store failing is a retry
        raise BundleError(
            f"spyre_v2 failed: {type(err).__name__}: {err}", FAILED
        ) from None
    wrote = any(r["action"] == "write" for r in results)
    status = (
        "would-ingest"
        if args.dry_run and wrote
        else ("ingested" if wrote else "duplicate")
    )
    return {"status": status, "kind": KIND, "bundle_sha256": sha, "events": results}

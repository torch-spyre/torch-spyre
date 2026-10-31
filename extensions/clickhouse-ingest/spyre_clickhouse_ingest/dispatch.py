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

"""Artifact dispatch: subscriptions, requests and the dispatch ledger of 25-artifact-dispatch.sql.

    python -m spyre_clickhouse_ingest dispatch request --subscription <id> --artifact <spec> \\
        [--arch <a>] [--tag <t>] [--param K=V ...] [--requested-by user|pipeline] [--dry-run]
    python -m spyre_clickhouse_ingest dispatch preview --subscription <id> [--days N]
        [--artifact <spec>]

The spyre-frameworks artifact-dispatch job owns the trigger loop; this module is what it,
`preview` and vars/v2Dispatch.groovy must agree on: the derived ids, template rendering, the
rate-limit and coalescing decision, and the reads and writes of the three tables. Matching
itself is SQL (v_artifact_subscribers), read here.
"""

import argparse
import json
import sys
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import regex as re

from .identity import DerivedId
from .schema import REQUESTED_BY_VALUES, ArtifactDispatches, DispatchRequests

NIL_UUID = "00000000-0000-0000-0000-000000000000"
# Rows the dispatcher planned but never confirmed (it died between plan and record) are
# re-planned after this; a trigger is therefore at-least-once.
STALE_QUEUED = timedelta(minutes=30)
REQUEST_WINDOW = timedelta(days=7)
# What a gha_dispatch / webhook body may carry under "artifact"; empty payload_fields = DEFAULT_PAYLOAD.
PAYLOAD_FIELDS = (
    "artifact_id",
    "digest",
    "digest_bare",
    "pull_spec",
    "tag",
    "tag_family",
    "component",
    "arch",
    "artifact_name",
    "verdicts",
)
DEFAULT_PAYLOAD = ("artifact_id", "tag")


class DispatchId(DerivedId):
    """uuid5 of what a dispatch is FOR, so a re-run finds the row it wrote."""

    @classmethod
    def auto(
        cls, subscription_id: str, artifact_id: str, event_type: str, event_key: str
    ) -> str:
        return cls.hash(
            "auto", subscription_id, str(artifact_id), event_type, event_key
        )

    @classmethod
    def request(cls, request_id: str) -> str:
        return cls.hash("request", str(request_id))


class Template:
    """`{name}` placeholders filled from the artifact; an unknown name is an error, not ''."""

    FIELDS = (
        "artifact_id",
        "digest",
        "digest_bare",
        "pull_spec",
        "ref",
        "tag",
        "tag_family",
        "component",
        "arch",
        "kind",
        "artifact_name",
        "id12",
        "dispatch_id",
        "subscription_id",
        "requested_by",
        "event_type",
        "event_key",
        "test_type",
        "state",
    )
    PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")

    @classmethod
    def render(cls, params: Mapping[str, str], context: Mapping[str, str]) -> dict:
        """params with every placeholder filled; KeyError names the first unknown or empty one."""

        def fill(m):
            name = m.group(1)
            value = context.get(name, "") if name in cls.FIELDS else None
            if not value:
                raise KeyError(name)
            return str(value)

        return {k: cls.PLACEHOLDER.sub(fill, str(v)) for k, v in (params or {}).items()}


def _utc(value) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class DispatchStore:
    """The dispatcher's reads and writes against one v2 database."""

    EVENT_COLUMNS = (
        "event_type",
        "event_key",
        "event_ts",
        "tag",
        "tag_family",
        "test_type",
        "state",
        "artifact_id",
        "component",
        "arch",
        "kind",
        "artifact_name",
        "subscription_id",
        "reason",
        "max_per_hour",
        "max_per_day",
        "coalesce_minutes",
        "target_type",
        "target",
        "params",
        "payload_fields",
        "credential_id",
        "dispatch_state",
    )

    def __init__(self, client, db: str):
        self.client, self.db = client, db

    def _rows(self, sql: str, **params) -> list[dict]:
        # The server runs in UTC but returns naive datetimes, which an insert would read as local time.
        res = self.client.query(sql, parameters=params)
        return [
            {
                k: _utc(v) if isinstance(v, datetime) else v
                for k, v in zip(res.column_names, r)
            }
            for r in res.result_rows
        ]

    def watermark(self) -> datetime | None:
        """event_ts of the newest event already dispatched, or None before the first."""
        ((n, newest),) = self.client.query(
            f"SELECT count(), max(event_ts) FROM {self.db}.artifact_dispatches "
            "WHERE requested_by = 'auto'"
        ).result_rows
        return _utc(newest) if n else None

    def events(
        self,
        since: datetime,
        *,
        due_only: bool = True,
        subscription_id: str = "",
        artifact_id: str = "",
    ) -> list[dict]:
        """Events at or after `since` with their subscribers; due_only = would fire, undispatched."""
        where = ["event_ts >= {since:DateTime64(3, 'UTC')}"]
        if due_only:
            where.append("auto AND dispatch_state = ''")
        if subscription_id:
            where.append("subscription_id = {sub:String}")
        if artifact_id:
            where.append("artifact_id = {aid:UUID}")
        return self._rows(
            f"SELECT {', '.join(self.EVENT_COLUMNS)} FROM {self.db}.v_artifact_subscribers "
            f"WHERE {' AND '.join(where)} ORDER BY event_ts, subscription_id",
            since=since,
            sub=subscription_id,
            aid=artifact_id or NIL_UUID,
        )

    def counts(self, now: datetime) -> dict[str, tuple[int, int]]:
        """subscription_id -> automatic dispatches sent in the last hour and day."""
        rows = self._rows(
            "SELECT subscription_id, countIf(requested_at >= {hour:DateTime64(3, 'UTC')}) AS h, "
            f"count() AS d FROM {self.db}.artifact_dispatches FINAL "
            "WHERE requested_by = 'auto' AND state != 'skipped' "
            "AND requested_at >= {day:DateTime64(3, 'UTC')} GROUP BY subscription_id",
            hour=now - timedelta(hours=1),
            day=now - timedelta(days=1),
        )
        return {r["subscription_id"]: (int(r["h"]), int(r["d"])) for r in rows}

    def requests(self, now: datetime) -> list[dict]:
        """Requests of the last REQUEST_WINDOW with no dispatch; matched = the artifact fits."""
        rows = self._rows(
            "SELECT r.request_id AS request_id, r.requested_at AS requested_at, "
            "r.subscription_id AS subscription_id, r.artifact_id AS artifact_id, r.tag AS tag, "
            "r.requested_by AS requested_by, r.requester AS requester, "
            "r.params AS request_params, "
            "ifNull(m.component, '') AS component, ifNull(m.arch, '') AS arch, "
            "ifNull(m.kind, '') AS kind, ifNull(m.artifact_name, '') AS artifact_name, "
            "ifNull(s.target_type, '') AS target_type, ifNull(s.target, '') AS target, "
            "s.params AS params, s.payload_fields AS payload_fields, "
            "ifNull(s.credential_id, '') AS credential_id, "
            "ifNull(m.subscription_id, '') != '' AS matched "
            f"FROM {self.db}.dispatch_requests AS r "
            f"LEFT JOIN {self.db}.v_subscription_artifacts AS m "
            "ON m.subscription_id = r.subscription_id AND m.artifact_id = r.artifact_id "
            f"LEFT JOIN (SELECT * FROM {self.db}.artifact_subscriptions FINAL) AS s "
            "ON s.subscription_id = r.subscription_id "
            "WHERE r.requested_at >= {since:DateTime64(3, 'UTC')} "
            f"AND r.request_id NOT IN (SELECT request_id FROM {self.db}.artifact_dispatches "
            "WHERE requested_by != 'auto') ORDER BY r.requested_at",
            since=now - REQUEST_WINDOW,
        )
        for r in rows:
            r["matched"] = bool(r["matched"])
            r["params"] = r["params"] or {}
            r["payload_fields"] = r["payload_fields"] or []
        return rows

    def retries(self, now: datetime, max_attempts: int) -> list[dict]:
        """Failed dispatches due another attempt, and planned ones never confirmed."""
        return self._rows(
            f"SELECT * EXCEPT (audit_uuid, audit_timestamp) FROM {self.db}.artifact_dispatches FINAL "
            "WHERE (state = 'failed' AND attempts < {max:UInt16} "
            "AND next_attempt_at <= {now:DateTime64(3, 'UTC')}) "
            "OR (state = 'queued' AND updated_at < {stale:DateTime64(3, 'UTC')}) "
            "ORDER BY updated_at",
            max=max_attempts,
            now=now,
            stale=now - STALE_QUEUED,
        )

    def facts(self, artifact_ids: Iterable[str]) -> dict[str, dict]:
        """artifact_id -> id12, the immutable pull spec and digest, and its shortest ref."""
        ids = sorted({str(a) for a in artifact_ids})
        if not ids:
            return {}
        # Aliases differ from column names: a `ref` alias would shadow the column it aggregates.
        # The shortest pinned pullspec is the bare `<repo>@<digest>`, not `<repo>:<tag>@<digest>`.
        pinned = "x.method = 'container-pull' AND position(x.ref, '@') > 0"
        rows = self._rows(
            "SELECT toString(a.artifact_id) AS artifact_id, a.props['id12'] AS id12, "
            "ifNull(r.pin, '') AS pull_spec, ifNull(r.dig, '') AS digest, "
            "ifNull(r.shortest, '') AS ref "
            f"FROM {self.db}.v_artifacts AS a LEFT JOIN ("
            "SELECT x.artifact_id AS artifact_id, "
            f"argMinIf(x.ref, length(x.ref), {pinned}) AS pin, "
            "argMinIf(if(x.content_digest != '', x.content_digest, splitByChar('@', x.ref)[-1]), "
            f"length(x.ref), {pinned}) AS dig, "
            "argMin(x.ref, length(x.ref)) AS shortest "
            f"FROM {self.db}.artifact_refs AS x FINAL WHERE x.artifact_id IN {{ids:Array(UUID)}} "
            "GROUP BY x.artifact_id) AS r ON r.artifact_id = a.artifact_id "
            "WHERE a.artifact_id IN {ids:Array(UUID)}",
            ids=ids,
        )
        out = {r["artifact_id"]: {**r, "verdicts": []} for r in rows}
        for v in self._rows(
            "SELECT toString(artifact_id) AS artifact_id, toString(run_id) AS run_id, test_type, "
            f"state, arch FROM {self.db}.v_dispatch_verdicts "
            "WHERE artifact_id IN {ids:Array(UUID)} ORDER BY ts",
            ids=ids,
        ):
            if v["artifact_id"] in out:
                out[v.pop("artifact_id")]["verdicts"].append(v)
        return out

    def record(self, rows: list[dict]) -> int:
        return ArtifactDispatches.insert(self.client, rows, db=self.db)


def context(row: Mapping[str, Any], facts: Mapping[str, str], **extra) -> dict:
    """The render context for one match or request row and its artifact's facts."""
    digest = facts.get("digest", "")
    return {
        "artifact_id": str(row["artifact_id"]),
        "digest": digest,
        "digest_bare": digest.split(":", 1)[-1],
        "pull_spec": facts.get("pull_spec", ""),
        "ref": facts.get("ref", ""),
        "id12": facts.get("id12", ""),
        **{
            k: str(row.get(k) or "")
            for k in (
                "tag",
                "tag_family",
                "component",
                "arch",
                "kind",
                "artifact_name",
                "subscription_id",
                "event_type",
                "event_key",
                "test_type",
                "state",
            )
        },
        "verdicts": facts.get("verdicts", []),
        **extra,
    }


def payload(fields, ctx: Mapping[str, Any]) -> dict:
    """The "artifact" object of a gha_dispatch / webhook body; an unknown field is a ValueError."""
    fields = list(fields or DEFAULT_PAYLOAD)
    unknown = [f for f in fields if f not in PAYLOAD_FIELDS]
    if unknown:
        raise ValueError(
            f"unknown payload field(s) {unknown}; allowed: {list(PAYLOAD_FIELDS)}"
        )
    return {f: ctx.get(f, "") for f in fields}


def decide(
    rows: list[dict], counts: Mapping[str, tuple[int, int]], now: datetime
) -> list[tuple[dict, str, str]]:
    """(row, decision, detail) per due event, oldest first: fire, coalesced (detail = the event
    that won), coalescing (its window is still open) or throttled (deferred to a later sweep)."""
    rows = sorted(rows, key=lambda r: r["event_ts"])
    sent = {k: list(v) for k, v in counts.items()}
    out = []
    for r in rows:
        window = timedelta(minutes=int(r.get("coalesce_minutes") or 0))
        key = (r["subscription_id"], r["component"], r["arch"])
        if window:
            later = [
                x
                for x in rows
                if (x["subscription_id"], x["component"], x["arch"]) == key
                and r["event_ts"] < x["event_ts"] <= r["event_ts"] + window
            ]
            if later:
                w = later[-1]
                out.append(
                    (
                        r,
                        "coalesced",
                        f"{w['artifact_id']} {w['event_key'] or w['event_type']}",
                    )
                )
                continue
            if now < r["event_ts"] + window:
                out.append(
                    (
                        r,
                        "coalescing",
                        f"fires at {r['event_ts'] + window:%Y-%m-%d %H:%M}Z",
                    )
                )
                continue
        hour, day = sent.setdefault(r["subscription_id"], [0, 0])
        per_hour, per_day = (
            int(r.get("max_per_hour") or 0),
            int(r.get("max_per_day") or 0),
        )
        if (per_hour and hour >= per_hour) or (per_day and day >= per_day):
            out.append((r, "throttled", f"{hour}/h, {day}/day sent"))
            continue
        sent[r["subscription_id"]] = [hour + 1, day + 1]
        out.append((r, "fire", ""))
    return out


def preview(
    store: DispatchStore,
    now: datetime,
    *,
    subscription_id: str = "",
    artifact_id: str = "",
    days: float = 7,
) -> list[dict]:
    """What the dispatcher would do with each recent event, and why; writes nothing."""
    rows = store.events(
        now - timedelta(days=days),
        due_only=False,
        subscription_id=subscription_id,
        artifact_id=artifact_id,
    )
    due = [r for r in rows if r["reason"] == "" and r["dispatch_state"] == ""]
    verdict = {
        id(r): (d, detail) for r, d, detail in decide(due, store.counts(now), now)
    }
    out = []
    for r in rows:
        if r["dispatch_state"]:
            outcome, why = f"dispatched ({r['dispatch_state']})", ""
        elif id(r) in verdict:
            d, why = verdict[id(r)]
            outcome = "would fire" if d == "fire" else d
        else:
            outcome, why = "not dispatched", r["reason"]
        out.append({**r, "outcome": outcome, "why": why})
    return out


def dispatch_row(**fields) -> dict:
    """An artifact_dispatches row with every column the caller left out at its default."""
    now = datetime.now(timezone.utc)
    row = {
        "updated_at": now,
        "event_type": "",
        "event_key": "",
        "tag": "",
        "event_ts": now,
        "requester": "",
        "request_id": NIL_UUID,
        "requested_at": now,
        "params": {},
        "payload": "",
        "target_url": "",
        "build_url": "",
        "error": "",
        "attempts": 0,
        "next_attempt_at": now,
        "dispatcher_url": "",
        "props": {},
    }
    row.update(fields)
    return row


def _pairs(values: list[str]) -> dict:
    out = {}
    for v in values:
        k, sep, val = v.partition("=")
        if not sep or not k:
            raise SystemExit(f"[error] --param {v!r} is not K=V")
        out[k] = val
    return out


def request(
    client,
    db: str,
    *,
    subscription_id: str,
    artifact_id: str,
    tag: str = "",
    requested_by: str = "pipeline",
    requester: str = "",
    params=None,
    reason: str = "",
    dry_run: bool = False,
) -> dict:
    """Write one dispatch_requests row, refusing a subscription the artifact does not fit."""
    ((fits,),) = client.query(
        f"SELECT count() FROM {db}.v_subscription_artifacts "
        "WHERE subscription_id = {s:String} AND artifact_id = {a:UUID}",
        parameters={"s": subscription_id, "a": artifact_id},
    ).result_rows
    if not fits:
        raise ValueError(
            f"subscription {subscription_id!r} does not exist or does not match artifact {artifact_id}"
        )
    row = {
        "request_id": str(uuid.uuid4()),
        "requested_at": datetime.now(timezone.utc),
        "subscription_id": subscription_id,
        "artifact_id": artifact_id,
        "tag": tag,
        "requested_by": requested_by,
        "requester": requester,
        "params": dict(params or {}),
        "reason": reason,
        "props": {},
    }
    if not dry_run:
        DispatchRequests.insert(client, [row], db=db)
    out = {
        k: str(v) if k in ("request_id", "requested_at") else v for k, v in row.items()
    }
    return {
        **out,
        "dispatch_id": DispatchId.request(row["request_id"]),
        "written": not dry_run,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="", help="default: $CLICKHOUSE_DB_V2")
    sub = parser.add_subparsers(dest="cmd", required=True)
    req = sub.add_parser(
        "request", help="ask the dispatcher to run one subscription (JSON)"
    )
    req.add_argument("--subscription", required=True)
    req.add_argument(
        "--artifact", required=True, help="any `artifacts resolve` spec, or id:<uuid>"
    )
    req.add_argument("--arch", default="")
    req.add_argument("--tag", default="")
    req.add_argument(
        "--param", action="append", default=[], help="K=V override, repeatable"
    )
    req.add_argument(
        "--requested-by", default="pipeline", choices=sorted(REQUESTED_BY_VALUES)
    )
    req.add_argument("--requester", default="")
    req.add_argument("--reason", default="")
    req.add_argument(
        "--dry-run", action="store_true", help="check and print, write nothing"
    )
    pre = sub.add_parser(
        "preview", help="which recent events would fire, and why others would not"
    )
    pre.add_argument("--subscription", default="")
    pre.add_argument("--artifact", default="", help="any `artifacts resolve` spec")
    pre.add_argument("--arch", default="")
    pre.add_argument("--days", type=float, default=7)
    pre.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from .client import ClickHouse, ClickHouseEnv
    from .resolver import resolve

    db = args.database or ClickHouseEnv.target_database()
    if not db:
        sys.exit("[error] no database: pass --database or set CLICKHOUSE_DB_V2")
    client = ClickHouse.connect(database=db)
    if args.cmd == "preview":
        client.set_client_setting("readonly", "2")
        aid = ""
        if args.artifact:
            found = resolve(
                args.artifact,
                args.arch,
                client=client,
                db=db,
                lookup="only",
                registry="off",
            )
            if found is None:
                sys.exit(f"[error] {args.artifact!r} names no recorded artifact")
            aid = str(found["artifact_id"])
        if not (args.subscription or aid):
            sys.exit("[error] pass --subscription and/or --artifact")
        rows = preview(
            DispatchStore(client, db),
            datetime.now(timezone.utc),
            subscription_id=args.subscription,
            artifact_id=aid,
            days=args.days,
        )
        for r in rows:
            if args.json:
                print(json.dumps(r, sort_keys=True, default=str))
            else:
                print(
                    f"{r['event_ts']:%Y-%m-%d %H:%M} {r['subscription_id']} {r['event_type']} "
                    f"{r['event_key'] or '-'} {r['artifact_name']}/{r['arch']} {r['artifact_id']}: "
                    f"{r['outcome']}" + (f" ({r['why']})" if r["why"] else "")
                )
        return
    r = resolve(
        args.artifact, args.arch, client=client, db=db, lookup="only", registry="off"
    )
    if r is None:
        sys.exit(f"[error] {args.artifact!r} names no recorded artifact")
    try:
        out = request(
            client,
            db,
            subscription_id=args.subscription,
            artifact_id=str(r["artifact_id"]),
            tag=args.tag,
            requested_by=args.requested_by,
            requester=args.requester,
            params=_pairs(args.param),
            reason=args.reason,
            dry_run=args.dry_run,
        )
    except ValueError as err:
        sys.exit(f"[error] {err}")
    print(json.dumps(out, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

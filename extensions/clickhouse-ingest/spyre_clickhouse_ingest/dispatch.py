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

The spyre-frameworks artifact-dispatch job owns the trigger loop; this module is what it and
vars/v2Dispatch.groovy must agree on: the derived ids, template rendering, and the reads and
writes of the three tables. Matching itself is SQL (v_artifact_subscribers), read here.
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


class DispatchId(DerivedId):
    """uuid5 of what a dispatch is FOR, so a re-run finds the row it wrote."""

    @classmethod
    def auto(cls, subscription_id: str, artifact_id: str, tag: str) -> str:
        return cls.hash("auto", subscription_id, str(artifact_id), tag)

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

    MATCH_COLUMNS = (
        "tag",
        "tag_family",
        "artifact_id",
        "tag_ts",
        "component",
        "arch",
        "kind",
        "artifact_name",
        "subscription_id",
        "target_type",
        "target",
        "params",
        "credential_id",
    )

    def __init__(self, client, db: str):
        self.client, self.db = client, db

    def _rows(self, sql: str, **params) -> list[dict]:
        res = self.client.query(sql, parameters=params)
        return [dict(zip(res.column_names, r)) for r in res.result_rows]

    def watermark(self) -> datetime | None:
        """tag_ts of the newest tag match already dispatched, or None before the first."""
        ((n, newest),) = self.client.query(
            f"SELECT count(), max(tag_ts) FROM {self.db}.artifact_dispatches "
            "WHERE requested_by = 'auto'"
        ).result_rows
        return _utc(newest) if n else None

    def matches(self, since: datetime) -> list[dict]:
        """Automatic matches first seen at or after `since` that have no dispatch yet."""
        cols = ", ".join(self.MATCH_COLUMNS)
        return self._rows(
            f"SELECT {cols} FROM {self.db}.v_artifact_subscribers "
            "WHERE auto AND dispatch_state = '' AND tag_ts >= {since:DateTime64(3, 'UTC')} "
            "ORDER BY tag_ts, subscription_id",
            since=since,
        )

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
            "s.params AS params, ifNull(s.credential_id, '') AS credential_id, "
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
        return {r["artifact_id"]: r for r in rows}

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
            )
        },
        **extra,
    }


def dispatch_row(**fields) -> dict:
    """An artifact_dispatches row with every column the caller left out at its default."""
    now = datetime.now(timezone.utc)
    row = {
        "updated_at": now,
        "tag": "",
        "tag_ts": now,
        "requester": "",
        "request_id": NIL_UUID,
        "requested_at": now,
        "params": {},
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
    args = parser.parse_args(argv)

    from .client import ClickHouse, ClickHouseEnv
    from .resolver import resolve

    db = args.database or ClickHouseEnv.target_database()
    if not db:
        sys.exit("[error] no database: pass --database or set CLICKHOUSE_DB_V2")
    client = ClickHouse.connect(database=db)
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

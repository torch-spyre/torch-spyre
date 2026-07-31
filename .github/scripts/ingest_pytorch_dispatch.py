#!/usr/bin/env python3
"""
Classifies a pytorch/pytorch CI-relay repository_dispatch payload and
inserts one row into ClickHouse (pytorch_ci_dispatches table).

pytorch/pytorch closes a PR on both "closed without merging" and "merged",
and `pull_request.merged` / `merge_commit_sha` are not reliably populated on
repository_dispatch-forwarded payloads. The only trustworthy signal for a
merge is the "Merged" label pytorch's own CI applies to the PR — this
mirrors the workaround already used elsewhere for the same relay.

Usage (called by the GHA workflow):
    python3 ingest_pytorch_dispatch.py \
        --payload-file payload.json \
        --event-type pull_request \
        --run-id    "74526099734" \
        --run-link  "https://github.com/org/repo/actions/runs/74526099734"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import clickhouse_connect

# ---------------------------------------------------------------------------
# ClickHouse DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pytorch_ci_dispatches
(
    delivery_id     String,

    event_type      LowCardinality(String) DEFAULT '',
    action          LowCardinality(String) DEFAULT '',
    pr_event        LowCardinality(String) DEFAULT '',

    pr_number       UInt32 DEFAULT 0,
    pr_title        String DEFAULT '',
    pr_base_ref     LowCardinality(String) DEFAULT '',
    pr_head_ref     LowCardinality(String) DEFAULT '',
    pr_head_sha     String DEFAULT '',
    pr_html_url     String DEFAULT '',

    sender_login    LowCardinality(String) DEFAULT '',
    repo_full_name  LowCardinality(String) DEFAULT '',

    gha_run_id      UInt64 DEFAULT 0,
    gha_run_link    String DEFAULT '',

    -- Populated by a later stage that decides whether to run tests for this
    -- dispatch and, if so, tracks that job's lifecycle. job_name is part of
    -- the sort key since one dispatch can fan out into several jobs.
    job_name        LowCardinality(String) DEFAULT '',
    progress        LowCardinality(String) DEFAULT '',
    result_payload  String DEFAULT '',

    received_at     DateTime DEFAULT now(),
    updated_at      DateTime DEFAULT now(),
    raw_payload     String DEFAULT ''
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (delivery_id, job_name)
"""

# ---------------------------------------------------------------------------
# ClickHouse connection client
# ---------------------------------------------------------------------------


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Actions that map 1:1 onto a pr_event name.
_DIRECT_PR_EVENTS = {
    "opened": "opened",
    "synchronize": "new_commit",
    "reopened": "reopened",
}


def classify_pr_event(action: str, pull_request: dict) -> str:
    """Map a pull_request payload's `action` to a stable pr_event label.

    A PR is "merged" if either `pull_request.merged` is `true` or it carries
    the "Merged" label pytorch's own CI applies — check both, since neither
    field is reliably populated on every repository_dispatch-forwarded
    payload on its own.
    """
    if action in _DIRECT_PR_EVENTS:
        return _DIRECT_PR_EVENTS[action]

    if action == "closed":
        labels = pull_request.get("labels") or []
        label_names = {lbl.get("name") for lbl in labels if isinstance(lbl, dict)}
        is_merged = pull_request.get("merged") is True or "Merged" in label_names
        return "merged" if is_merged else "closed"

    # Anything else (edited, labeled, assigned, ...) is passed through
    # verbatim so nothing is silently dropped from the dashboard.
    return action or "unknown"


def resolve_event_type(cli_event_type: str, payload_event_type: str) -> str:
    """The repository_dispatch event type, recorded as-is (e.g. pull_request | push)."""
    return _str(cli_event_type) or _str(payload_event_type)


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def _str(val, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip()


def _int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def build_row(
    client_payload: dict, args, raw_payload_text: str, result_payload_text: str
) -> dict:
    payload = client_payload.get("payload") or {}
    action = _str(payload.get("action"))
    pull_request = payload.get("pull_request") or {}
    event_type = resolve_event_type(
        args.event_type, _str(client_payload.get("event_type"))
    )
    pr_event = (
        classify_pr_event(action, pull_request) if event_type == "pull_request" else ""
    )

    return {
        "delivery_id": _str(client_payload.get("delivery_id")),
        "event_type": event_type,
        "action": action,
        "pr_event": pr_event,
        "pr_number": _int(payload.get("number") or pull_request.get("number")),
        "pr_title": _str(pull_request.get("title")),
        "pr_base_ref": _str((pull_request.get("base") or {}).get("ref")),
        "pr_head_ref": _str((pull_request.get("head") or {}).get("ref")),
        "pr_head_sha": _str((pull_request.get("head") or {}).get("sha")),
        "pr_html_url": _str(
            (pull_request.get("_links") or {}).get("html", {}).get("href")
        ),
        "sender_login": _str((payload.get("sender") or {}).get("login")),
        "repo_full_name": _str((payload.get("repository") or {}).get("full_name")),
        "gha_run_id": _int(args.run_id),
        "gha_run_link": _str(args.run_link),
        "job_name": _str(args.job_name),
        "progress": _str(args.progress),
        "result_payload": result_payload_text,
        "raw_payload": raw_payload_text,
    }


# Valid values for --progress / the `progress` column.
PROGRESS_STATES = ("", "in_progress", "completed", "rejected")

COLUMN_NAMES = [
    "delivery_id",
    "event_type",
    "action",
    "pr_event",
    "pr_number",
    "pr_title",
    "pr_base_ref",
    "pr_head_ref",
    "pr_head_sha",
    "pr_html_url",
    "sender_login",
    "repo_full_name",
    "gha_run_id",
    "gha_run_link",
    "job_name",
    "progress",
    "result_payload",
    "raw_payload",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a pytorch CI-relay dispatch payload -> ClickHouse pytorch_ci_dispatches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--payload-file",
        required=True,
        help="Path to the JSON file holding the repository_dispatch client_payload",
    )
    parser.add_argument(
        "--event-type",
        default="",
        help="repository_dispatch event type (github.event.action), e.g. pull_request | push",
    )
    parser.add_argument(
        "--run-id", default="0", help="GHA run ID of the ingesting workflow"
    )
    parser.add_argument("--run-link", default="", help="URL to the ingesting GHA run")
    parser.add_argument(
        "--job-name",
        default="",
        help="Name of the job run for this dispatch (set by a later stage, once decided)",
    )
    parser.add_argument(
        "--progress",
        default="",
        choices=PROGRESS_STATES,
        help="Lifecycle state of --job-name: in_progress | completed | rejected",
    )
    parser.add_argument(
        "--result-payload-file",
        default="",
        help="Optional path to a JSON file with the job's result payload",
    )
    args = parser.parse_args()

    payload_path = Path(args.payload_file)
    if not payload_path.exists():
        print(f"[error] File not found: {payload_path}", file=sys.stderr)
        sys.exit(1)

    raw_payload_text = payload_path.read_text()
    try:
        client_payload = (
            json.loads(raw_payload_text) if raw_payload_text.strip() else {}
        )
    except json.JSONDecodeError as exc:
        print(f"[error] Could not parse payload JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    result_payload_text = ""
    if args.result_payload_file:
        result_payload_path = Path(args.result_payload_file)
        if not result_payload_path.exists():
            print(f"[error] File not found: {result_payload_path}", file=sys.stderr)
            sys.exit(1)
        result_payload_text = result_payload_path.read_text()

    row = build_row(client_payload, args, raw_payload_text, result_payload_text)

    if not row["delivery_id"]:
        print(
            "[error] Payload has no delivery_id — refusing to ingest.", file=sys.stderr
        )
        sys.exit(1)

    print("[info] Classified dispatch:")
    print(f"[info]   delivery_id : {row['delivery_id']}")
    print(f"[info]   event_type  : {row['event_type']}")
    print(f"[info]   action      : {row['action']}")
    print(f"[info]   pr_event    : {row['pr_event']}")
    print(f"[info]   pr_number   : {row['pr_number']}")
    print(f"[info]   job_name    : {row['job_name']}")
    print(f"[info]   progress    : {row['progress']}")

    print(
        f"[info] Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("[info] Connected.")

    client.command(_CREATE_TABLE_SQL)

    client.insert(
        table="pytorch_ci_dispatches",
        data=[[row[col] for col in COLUMN_NAMES]],
        column_names=COLUMN_NAMES,
    )
    print("[info] Inserted 1 row into pytorch_ci_dispatches")


if __name__ == "__main__":
    main()

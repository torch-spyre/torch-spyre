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

"""Write one orchestrator run's timeline (a JSON batch) into ci_run_timings, a row per entry."""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .identity import DerivedId
from .schema import TIMING_TEST_STATE_VALUES, CiRunTimings

# The keys each batch section may carry; an unknown key is refused so a typo cannot read as NULL.
ENTRY_KEYS = {
    "component",
    "artifact_name",
    "arch",
    "id12",
    "kind",
    "attempt",
    "state",
    "result",
    "url",
    "agent",
    "queued_at",
    "started_at",
    "ended_at",
    "failure_reason",
    "failed_stage",
}
BATCH_KEYS = {
    "run": {
        "run_key",
        "trigger_kind",
        "trigger_source",
        "preset",
        "build_mode",
        "trigger_pr",
        "pr_components",
        "sha",
        "base_ref",
        "build_url",
        "verdict",
        "result",
        "superseded",
        "pickup_path",
        "comment_at",
        "picked_up_at",
        "scheduled_at",
        "started_at",
        "pr_queued_at",
        "pr_running_at",
        "ended_at",
        "props",
    },
    "builds": ENTRY_KEYS,
    "tests": ENTRY_KEYS | {"modes", "gating", "runner_died", "exec"},
    "exec": {
        "executor",
        "provision_started_at",
        "provision_ended_at",
        "dispatched_at",
        "started_at",
        "ended_at",
        "runs",
        "jobs",
        "result",
        "urls",
        "run_keys",
        "cards",
    },
}
# run_started_at is the partition key, so a run without a start cannot be written.
RUN_REQUIRED = ("run_key", "started_at")
# Run fields copied as given.
RUN_TEXT = (
    "trigger_kind",
    "trigger_source",
    "preset",
    "build_mode",
    "sha",
    "base_ref",
    "build_url",
    "pickup_path",
)

# A test leg's Jenkins result; anything unlisted (ABORTED, NOT_BUILT) gave no test signal.
RESULT_STATES = {"success": "passed", "failure": "failed", "unstable": "failed"}


def check_batch(batch: Mapping[str, Any]) -> None:
    """Raise ValueError naming every unknown section or key in `batch`, and a missing run key."""
    run = batch.get("run") or {}
    bad = [f"section {k!r}" for k in batch if k not in ("run", "builds", "tests")]
    bad += [f"run.{k}" for k in run if k not in BATCH_KEYS["run"]]
    for section in ("builds", "tests"):
        for i, e in enumerate(batch.get(section) or []):
            bad += [f"{section}[{i}].{k}" for k in e if k not in BATCH_KEYS[section]]
    for i, t in enumerate(batch.get("tests") or []):
        bad += [
            f"tests[{i}].exec.{k}"
            for k in t.get("exec") or {}
            if k not in BATCH_KEYS["exec"]
        ]
    bad += [f"run.{k} (missing)" for k in RUN_REQUIRED if not run.get(k)]
    if bad:
        raise ValueError("bad batch key(s): " + ", ".join(bad))


def ts(value: Any) -> datetime | None:
    """Epoch milliseconds or ISO-8601 as an aware UTC datetime; None when unknown."""
    if value in (None, "", 0, "0"):
        return None
    if isinstance(value, (int, float)) or str(value).strip().isdigit():
        ms = int(value)
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc) if ms > 0 else None
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _flag(value: Any) -> bool:
    """A JSON bool, or the 'true'/'false' string a Groovy map serializes to."""
    return value is True or _text(value).lower() in ("true", "1")


def _list(value: Any) -> list[str]:
    """A JSON array or a comma-separated string, blanks and repeats dropped."""
    items = value.split(",") if isinstance(value, str) else (value or [])
    return list(dict.fromkeys(s for s in (_text(v) for v in items) if s))


def split_trigger_pr(trigger_pr: str) -> tuple[str, int]:
    """'<host>/<owner>/<repo>#<n>' -> (repo, n); ('', 0) when it is not that shape."""
    path, _, num = (trigger_pr or "").strip().rpartition("#")
    repo = path.rstrip("/").rsplit("/", 1)[-1]
    return (repo, int(num)) if path and num.isdigit() else ("", 0)


def entry_state(entry: str, e: Mapping[str, Any]) -> str:
    """The entry's state; a test leg without one takes it from its Jenkins result."""
    state = _text(e.get("state")).lower()
    if entry == "build" or (state and state in TIMING_TEST_STATE_VALUES):
        return state
    result = _text(e.get("result")).lower()
    return RESULT_STATES.get(result, "error") if result else ""


def _run_fields(run: Mapping[str, Any]) -> dict[str, Any]:
    trigger_pr = _text(run.get("trigger_pr"))
    repo, pr_number = split_trigger_pr(trigger_pr)
    return {
        "run_key": _text(run["run_key"]),
        **{k: _text(run.get(k)) for k in RUN_TEXT},
        "trigger_pr": trigger_pr,
        "repo": repo,
        "pr_number": pr_number,
        "verdict": _text(run.get("verdict")).lower(),
        "run_result": _text(run.get("result")).lower(),
        "superseded": _flag(run.get("superseded")),
        "comment_at": ts(run.get("comment_at")),
        "picked_up_at": ts(run.get("picked_up_at")),
        "run_scheduled_at": ts(run.get("scheduled_at")),
        "run_started_at": ts(run.get("started_at")),
        "pr_queued_at": ts(run.get("pr_queued_at")),
        "pr_running_at": ts(run.get("pr_running_at")),
        "run_ended_at": ts(run.get("ended_at")),
        "props": {str(k): str(v) for k, v in (run.get("props") or {}).items()},
    }


def _entry_fields(entry: str, e: Mapping[str, Any], prs: set[str]) -> dict[str, Any]:
    x = e.get("exec") or {}
    component = _text(e.get("component"))
    return {
        "entry": entry,
        "component": component,
        "artifact_name": _text(e.get("artifact_name")),
        "arch": DerivedId.arch(e.get("arch") or ""),
        "id12": _text(e.get("id12")),
        "kind": _text(e.get("kind")) or "image",
        "test_modes": sorted(_list(e.get("modes"))),
        "attempt": int(e.get("attempt") or 1),
        "is_pr_component": component in prs,
        "state": entry_state(entry, e),
        "result": _text(e.get("result")).lower(),
        "gating": _text(e.get("gating")).lower(),
        "url": _text(e.get("url")),
        "agent": _text(e.get("agent")),
        "queued_at": ts(e.get("queued_at")),
        "started_at": ts(e.get("started_at")),
        "ended_at": ts(e.get("ended_at")),
        "executor": _text(x.get("executor")).lower(),
        "provision_started_at": ts(x.get("provision_started_at")),
        "provision_ended_at": ts(x.get("provision_ended_at")),
        "exec_dispatched_at": ts(x.get("dispatched_at")),
        "exec_started_at": ts(x.get("started_at")),
        "exec_ended_at": ts(x.get("ended_at")),
        "exec_runs": int(x.get("runs") or 0),
        "exec_jobs": int(x.get("jobs") or 0),
        # run_integration_tests.py reports a workflow that never started as __never_started__.
        "exec_result": _text(x.get("result")).lower().strip("_"),
        "exec_urls": _list(x.get("urls")),
        "exec_run_keys": _list(x.get("run_keys")),
        "cards": _list(x.get("cards")),
        "runner_died": _flag(e.get("runner_died")),
        "failure_reason": _text(e.get("failure_reason")),
        "failed_stage": _text(e.get("failed_stage")),
    }


def _key(row: Mapping[str, Any]) -> tuple:
    """The row's ORDER BY key past run_key; two rows sharing it would replace each other."""
    cols = ("entry", "component", "artifact_name", "arch")
    return (*(row[c] for c in cols), ",".join(row["test_modes"]), row["attempt"])


def build_rows(
    batch: Mapping[str, Any], now: datetime | None = None
) -> list[dict[str, Any]]:
    """The ci_run_timings rows of one batch: one per build and per test leg, in batch order."""
    check_batch(batch)
    run = _run_fields(batch["run"])
    # The run's PRs: the trigger and any Test-With companions, as PR refs or component names.
    refs = [run["trigger_pr"], *_list(batch["run"].get("pr_components"))]
    prs = {split_trigger_pr(p)[0] or p for p in refs if p}
    updated_at = now or datetime.now(timezone.utc)
    rows = [
        {**run, "updated_at": updated_at, **_entry_fields(entry, e, prs)}
        for entry, section in (("build", "builds"), ("test", "tests"))
        for e in batch.get(section) or []
        if _text(e.get("component"))
    ]
    dups = [k for k, n in Counter(map(_key, rows)).items() if n > 1]
    if dups:
        raise ValueError(
            f"entries repeat a key (give each try its own attempt): {dups}"
        )
    return rows


def write_batch(client, db: str, batch: Mapping[str, Any]) -> int:
    """Insert every row of `batch`; returns how many were written."""
    return CiRunTimings.insert(client, build_rows(batch), db=db)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="", help="default: $CLICKHOUSE_DB_V2")
    sub = parser.add_subparsers(dest="cmd", required=True)
    wr = sub.add_parser("write", help="one run's timeline batch (JSON)")
    wr.add_argument("batch", type=Path)
    wr.add_argument(
        "--dry-run", action="store_true", help="print the rows instead of writing them"
    )
    args = parser.parse_args(argv)

    batch = json.loads(args.batch.read_text())
    if args.dry_run:
        for row in build_rows(batch):
            # The same validation an insert applies, so a dry run fails where a write would.
            CiRunTimings.row(row)
            print(json.dumps(row, default=str, sort_keys=True))
        return

    from .client import ClickHouse, ClickHouseEnv

    db = args.database or ClickHouseEnv.target_database()
    if not db:
        sys.exit("[error] no database: pass --database or set CLICKHOUSE_DB_V2")
    client = ClickHouse.connect(database=db)
    done = write_batch(client, db, batch)
    print(f"[info] {db}: {done} ci_run_timings row(s) for {batch['run']['run_key']}")


if __name__ == "__main__":
    main()

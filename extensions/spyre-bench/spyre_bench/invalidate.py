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

"""spyre-bench invalidate: preview, record, restore and review invalidations of stored results.

  spyre-bench invalidate (--run ID [--benchmark ID [--repeat N]] | --node N | --device SERIAL | --placement ID)
        [--from TS] [--to TS] --reason-code C --reason TEXT [--state suspect|invalidated] [--apply [--confirm [TOKEN]]]
  spyre-bench invalidate restore ID --reason-code C --reason TEXT [--apply [--confirm [TOKEN]]]
  spyre-bench invalidate set ID --state suspect|invalidated --reason-code C --reason TEXT [--apply ...]
  spyre-bench invalidate list [--state S]        spyre-bench invalidate show ID

Without --apply nothing is written: the store is read and the runs, series, benchmarks, repeats and verdict changes
the event would cause are printed. --apply records one event through a `bench-invalidation` bundle and the ingest.
A window or query covering more runs than the threshold (BENCH_INVALIDATION_CONFIRM_ABOVE, default 50) also needs
--confirm; pass the preview's token to bind the confirmation to the runs you saw. The matching and the write rules
(state machine, distinct event times, the threshold) live in the ingest, which this verb requires (spyre-bench[ingest]).
"""

import argparse
import getpass
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from spyre_bench import cliutil, gen_schemas, ingest_adapter, planfile, store as stores
from spyre_bench.cliutil import UsageError

ACTIONS = ("create", "restore", "set", "list", "show")
REASON_CODES = tuple(gen_schemas.INVALIDATION_SHAPE["properties"]["code"]["enum"])
SETTABLE = ("suspect", "invalidated")
SHOWN_RUNS = 20
SHOWN_CHANGES = 20
# An invalidation is an admin event: the ingest takes it only from a ci or lab channel; a person at a terminal is lab.
INGEST_ARGV = [
    sys.executable,
    "-I",
    "-m",
    "spyre_clickhouse_ingest",
    "results",
    "--channel-trust",
    "lab",
    "--from-bundle",
]
INGEST_TIMEOUT = 300


def backend_module():
    """The ingest's invalidation module, or a UsageError that says how to install it."""
    try:
        return ingest_adapter.invalidation()
    except ingest_adapter.IngestUnavailable as e:
        raise UsageError(f"invalidate needs the ingest: {e}") from e


def actor(a):
    return cliutil.pick(a.actor, "BENCH_ACTOR") or getpass.getuser()


def scope_of(a):
    """(scope_kind, scope dict) from the scope flags; exactly one scope family."""
    families = [
        n
        for n, given in (
            ("run", a.run and not a.benchmark),
            ("run_benchmark", a.run and a.benchmark),
            ("node", a.node),
            ("device", a.device),
            ("placement", a.placement),
        )
        if given
    ]
    if len(families) != 1:
        raise UsageError("give exactly one scope: --run [--benchmark ID [--repeat N]], --node, --device or --placement")
    kind = families[0]
    if a.benchmark and not a.run:
        raise UsageError("--benchmark needs --run")
    if a.repeat is not None and kind != "run_benchmark":
        raise UsageError("--repeat needs --run and --benchmark")
    if (a.window_from or a.window_to) and kind in ("run", "run_benchmark"):
        raise UsageError("--from/--to bound a node, device or placement scope, not a run")
    scope = {
        "run_id": a.run,
        "benchmark_id": a.benchmark,
        "repeat": a.repeat,
        "node": a.node,
        "device_serial": a.device,
        "placement_id": a.placement,
    }
    return kind, {k: v for k, v in scope.items() if v not in (None, "")}


def reason_fields(a):
    if a.reason_code not in REASON_CODES:
        raise UsageError(f"--reason-code is required: one of {', '.join(REASON_CODES)}")
    if not (a.reason or "").strip():
        raise UsageError("--reason is required: say why, for the next person who reads the history")
    return {"reason_code": a.reason_code, "reason": a.reason.strip(), "evidence_url": a.evidence_url or ""}


def build_event(a, be, client):
    """The bundle-shaped event for the action; restore and set take their scope from the stored invalidation."""
    if a.action == "create":
        kind, scope = scope_of(a)
        if a.state not in (None, *SETTABLE):
            raise UsageError(f"--state {a.state}: use `restore` to restore")
        ev = {
            "invalidation_id": a.invalidation_id or str(uuid.uuid4()), "scope_kind": kind, "scope": scope,
            "window_from": a.window_from, "window_to": a.window_to, "state": a.state or "invalidated",
        }  # fmt: skip
    else:
        stored = be.stored_event(client, a.id)
        if stored is None:
            raise UsageError(f"no invalidation {a.id}")
        ev = {k: stored[k] for k in ("invalidation_id", "scope_kind", "scope", "window_from", "window_to")}
        ev["state"] = "restored" if a.action == "restore" else a.state
        if ev["state"] not in SETTABLE and a.action == "set" or ev["state"] is None:
            raise UsageError(f"--state is required: one of {', '.join(SETTABLE)}")
    return {**ev, **reason_fields(a)}


def render_preview(p, ev):
    s = p["series"]
    lines = [
        f"{ev['scope_kind']} scope {json.dumps(ev['scope'], sort_keys=True)}"
        + (
            f" window {ev['window_from'] or '...'} .. {ev['window_to'] or '...'}"
            if ev.get("window_from") or ev.get("window_to")
            else ""
        ),
        f"invalidation {p['invalidation_id']}: {p['state_before'] or 'new'} -> {p['state_after']}"
        + ("  (already in that state: nothing to do)" if p["noop"] else ""),
        f"covers {p['matched_runs']} runs, {p['matched_repeats']} repeats, {len(p['benchmarks'])} benchmarks, {len(s)} series"
        + ("  (detail truncated)" if p["truncated"] else ""),
    ]
    lines += [
        f"  run {r['run_id']}  {r['run_ts']}  repeats {r['repeats']}  benchmarks {r['benchmarks']}"
        for r in p["runs"][:SHOWN_RUNS]
    ]
    if len(p["runs"]) > SHOWN_RUNS:
        lines.append(f"  ... {p['matched_runs'] - SHOWN_RUNS} more runs (--json lists them)")
    lines += [
        f"  series {x['component']} {x['name']} [{x['backend']} {x['arch']} {x['source_job']}]  runs {x['runs']}"
        for x in s[:SHOWN_RUNS]
    ]
    vc = p["verdict_changes"]
    if p["noop"]:
        pass
    elif not vc["available"]:
        lines.append(f"verdict changes: not computed ({vc['reason']})")
    else:
        moves = ", ".join(f"{k} {v}" for k, v in sorted(vc["transitions"].items())) or "none"
        lines.append(f"verdict changes: {vc['total']} ({moves})")
        lines += [
            f"  {c['ts']}  {c['name']}  {c['metric']}  {c['before']} -> {c['after']}"
            for c in vc["rows"][:SHOWN_CHANGES]
        ]
    if not p["matched_runs"]:
        lines.append("warning: this scope matches no runs (check the id, node or serial and the window)")
    if p["needs_confirm"]:
        lines.append(
            f"confirmation required: {p['matched_runs']} runs is more than {p['threshold']}; token {p['confirm_token']}"
        )
    return "\n".join(lines)


def ingest_bundle(bundle_dir, who, runner=subprocess.run):
    """Run the ingest on a bundle; returns (exit code, its JSON report or {})."""
    env = planfile.ingest_env(os.environ, (), tempfile.gettempdir())
    db = cliutil.env("BENCH_CLICKHOUSE_DB") or cliutil.env("CLICKHOUSE_DB_V2") or cliutil.env("CLICKHOUSE_DB")
    if db:  # the database the preview read is the one the write goes to
        env["CLICKHOUSE_DB_V2"] = db
    if cliutil.env("BENCH_INVALIDATION_CONFIRM_ABOVE"):
        env["BENCH_INVALIDATION_CONFIRM_ABOVE"] = cliutil.env("BENCH_INVALIDATION_CONFIRM_ABOVE")
    try:
        proc = runner(
            [*INGEST_ARGV, str(bundle_dir), "--uploader", who],
            env=env,
            capture_output=True,
            text=True,
            timeout=INGEST_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return -1, {"reason": f"the ingest exceeded {INGEST_TIMEOUT}s; the event may or may not be recorded"}
    report = {}
    for line in reversed((proc.stdout or "").splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict):
            report = doc
            break
    if proc.returncode != 0 and not report.get("reason"):
        report["reason"] = ((proc.stderr or "").strip().splitlines() or ["no output"])[-1][:300]
    return proc.returncode, report


def do_write(a, ev, p, be, runner):
    if p["noop"]:
        print(f"nothing to do: {p['invalidation_id']} is already {p['state_after']}")
        return cliutil.EXIT_OK
    if p["needs_confirm"]:
        if a.confirm is None:
            print(
                f"error: {p['matched_runs']} runs exceeds the threshold {p['threshold']}: re-run with --confirm {p['confirm_token']}",
                file=sys.stderr,
            )
            return cliutil.EXIT_INVALID
        if a.confirm not in ("", p["confirm_token"]):
            print(
                f"error: --confirm token does not match the preview (the runs covered changed); expected {p['confirm_token']}",
                file=sys.stderr,
            )
            return cliutil.EXIT_INVALID
        ev = {**ev, "confirmed": True, "confirm_token": p["confirm_token"]}
    out = (
        Path(a.out)
        if a.out
        else Path(cliutil.env("BENCH_OUT", "results/bench"))
        / "invalidations"
        / f"{cliutil.utc_stamp().replace(':', '')}-{p['invalidation_id'][:8]}-{ev['state']}"
    )
    be.write_bundle(out, [ev], uploader=actor(a))
    note = sys.stderr if a.json else sys.stdout
    print(f"invalidation {p['invalidation_id']}; bundle: {out}", file=note)
    if a.bundle_only:
        print(
            "not ingested (--bundle-only); record it with: python -m spyre_clickhouse_ingest results --from-bundle "
            + str(out),
            file=note,
        )
        return cliutil.EXIT_OK
    rc, report = ingest_bundle(out, actor(a), runner)
    print(json.dumps(report, sort_keys=True))
    if rc == 0 and report.get("status") not in ("ingested", "duplicate"):
        rc = 1
        report.setdefault("reason", "the ingest reported no status")
    if rc != 0:
        print(f"error: the ingest exited {rc}: {report.get('reason', 'see its output')}", file=sys.stderr)
        return cliutil.EXIT_INVALID if rc == 2 else cliutil.EXIT_STORE
    return cliutil.EXIT_OK


def do_change(a, be, st, runner):
    client = st.client()
    ev = build_event(a, be, client)
    p = be.preview(client, ev)
    print(json.dumps({**p, "event": ev}, indent=1, default=str) if a.json else render_preview(p, ev))
    if not a.apply:
        if not p["noop"] and not a.json:
            print(
                "dry run: nothing written; add --apply"
                + (f" --confirm {p['confirm_token']}" if p["needs_confirm"] else "")
                + " to record it"
            )
        return cliutil.EXIT_OK
    return do_write(a, ev, p, be, runner)


def do_list(a, be, st):
    rows = be.list_invalidations(st.client(), a.state or "", a.limit)
    if a.json:
        print(json.dumps(rows, indent=1, default=str))
        return cliutil.EXIT_OK
    for r in rows:
        print(
            f"{r['invalidation_id']}  {r['state']:<11} {r['scope_kind']:<13} {str(r['scope_key'])[:24]:<24} runs {r['runs']:<5} {r['reason_code']}  {r['actor']}  {r['updated_at']}"
        )
    if not rows:
        print("no invalidations")
    return cliutil.EXIT_OK


def do_show(a, be, st):
    client = st.client()
    events = be.history(client, a.id)
    if not events:
        print(f"error: no invalidation {a.id}", file=sys.stderr)
        return cliutil.EXIT_INVALID
    runs = be.impact(client, a.id)
    if a.json:
        print(json.dumps({"invalidation_id": a.id, "events": events, "runs": runs}, indent=1, default=str))
        return cliutil.EXIT_OK
    print(f"invalidation {a.id}: {events[-1]['state']}, {events[0]['scope_kind']} scope, covers {len(runs)} runs")
    for e in events:
        print(
            f"  {e['ts']}  {e['state']:<11} {e['actor']}  {e['reason_code']}: {e['reason']}"
            + (f"  <{e['evidence_url']}>" if e["evidence_url"] else "")
            + f"  [{e['matched_runs']} runs at the time]"
        )
    for r in runs[:SHOWN_RUNS]:
        print(f"  run {r['run_id']}  {r['run_ts']}  repeats {r['repeats']}  benchmarks {r['benchmarks']}")
    if len(runs) > SHOWN_RUNS:
        print(f"  ... {len(runs) - SHOWN_RUNS} more runs (--json lists them)")
    return cliutil.EXIT_OK


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench invalidate", description=__doc__.split("\n")[0])
    ap.add_argument("action", nargs="?", choices=ACTIONS, default="create")
    ap.add_argument("id", nargs="?", help="restore, set and show: the invalidation id")
    ap.add_argument("--run", help="a stored run UUID")
    ap.add_argument("--benchmark", help="with --run: one stored benchmark_id of it")
    ap.add_argument("--repeat", type=int, help="with --run --benchmark: one repeat number")
    ap.add_argument("--node", help="every run that executed on this node")
    ap.add_argument("--device", help="every run that used the device with this serial")
    ap.add_argument("--placement", help="every run with this placement id")
    ap.add_argument(
        "--from", dest="window_from", help="start of the window (ISO-8601, UTC if no zone), by repeat start time"
    )
    ap.add_argument("--to", dest="window_to", help="end of the window")
    ap.add_argument("--state", help="create/set: suspect | invalidated; list: filter")
    ap.add_argument("--reason-code", help=f"one of {', '.join(REASON_CODES)}")
    ap.add_argument("--reason", help="why (required)")
    ap.add_argument("--evidence-url", help="link to the incident")
    ap.add_argument("--apply", action="store_true", help="record the event (default: preview only)")
    ap.add_argument("--dry-run", action="store_true", help="preview only (the default)")
    ap.add_argument(
        "--confirm",
        nargs="?",
        const="",
        default=None,
        metavar="TOKEN",
        help="needed above the threshold; TOKEN is the preview's",
    )
    ap.add_argument("--invalidation-id", help="create: use this UUID (a retry with the same id is a no-op)")
    ap.add_argument("--out", help="where to write the bundle (default <BENCH_OUT>/invalidations/...)")
    ap.add_argument("--bundle-only", action="store_true", help="write the bundle, do not ingest it")
    ap.add_argument("--actor", help="who is acting (BENCH_ACTOR, default the login name); the ingest records it")
    ap.add_argument("--limit", type=int, default=50, help="list: how many")
    ap.add_argument("--store", choices=("clickhouse",), help="BENCH_STORE, default clickhouse")
    ap.add_argument("--json", action="store_true")
    return ap


def execute(a, st=None, backend=None, runner=subprocess.run):
    if a.apply and a.dry_run:
        raise UsageError("--apply and --dry-run exclude each other")
    if a.action in ("restore", "set", "show") and not a.id:
        raise UsageError(f"{a.action} needs the invalidation id")
    if a.action == "create" and a.id:
        raise UsageError("create takes no positional id (--invalidation-id names one); restore, set and show take it")
    if cliutil.pick(a.store, "BENCH_STORE", "clickhouse") != "clickhouse":
        raise UsageError("invalidate needs a store (--store clickhouse)")
    be = backend or backend_module()
    try:
        st = st or stores.open_store("clickhouse")
        if a.action == "list":
            return do_list(a, be, st)
        if a.action == "show":
            return do_show(a, be, st)
        return do_change(a, be, st, runner)
    except be.InvalidationError as e:
        print(f"error: {e}", file=sys.stderr)
        return cliutil.EXIT_INVALID
    except stores.StoreError as e:
        print(f"error: store: {e}", file=sys.stderr)
        return cliutil.EXIT_STORE


def main(argv=None, st=None, backend=None, runner=subprocess.run):
    return cliutil.run_main(lambda a: execute(a, st, backend, runner), parser(), argv)


if __name__ == "__main__":
    sys.exit(main())

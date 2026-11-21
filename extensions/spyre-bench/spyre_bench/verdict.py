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

"""spyre-bench verdict: the stored verdict of a run, per benchmark and metric. Advisory; it gates only on --fail-on.

  spyre-bench verdict (--run-id UUID | --artifact UUID) [--store clickhouse] [--fail-on regressed] [--json]

Reads the SQL verdict view; the engine never decides a verdict itself. Output and exit codes: docs/README.md.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

from spyre_bench import cliutil, engine, store as stores
from spyre_bench.cliutil import UsageError

VERDICT_SCHEMA = "bench.verdict/1"
SEVERITY = ("regressed", "suspected", "improved", "no_baseline", "pass")
VALUE_FIELDS = ("baseline_value", "current_value", "delta_pct", "worse_pct", "floor_pct")
FIELDS = (
    "benchmark_id",
    "name",
    "component",
    "backend",
    "arch",
    "metric",
    "tier",
    "status",
    "reason",
    "baseline_kind",
    "n_baseline",
    "n_current",
    *VALUE_FIELDS,
)


def normal(row, quiet):
    out = {k: row.get(k) for k in FIELDS}
    out["benchmark_id"] = row.get("benchmark_key", row.get("benchmark_id"))  # the store aliases the string forms
    out["status"] = stores.STATUS_OF_VIEW.get(out["status"], out["status"])
    if out["status"] not in engine.VERDICTS:
        raise stores.StoreError(f"verdict view returned an unknown status {out['status']!r}")
    for k in ("n_baseline", "n_current"):
        out[k] = int(out[k] or 0)
    if quiet:
        for k in VALUE_FIELDS:
            out.pop(k)
    return out


def worst(statuses):
    return min(statuses, key=SEVERITY.index) if statuses else None


def build(rows, run_ids, artifact_id, store_name, quiet):
    verdicts = sorted((normal(r, quiet) for r in rows), key=lambda v: (v["name"] or "", v["metric"] or ""))
    per = {}
    for v in verdicts:
        per.setdefault((v["benchmark_id"], v["name"]), []).append(v["status"])
    summary = {"total": len(verdicts), **{s: sum(v["status"] == s for v in verdicts) for s in engine.VERDICTS}}
    summary["worst"] = worst([v["status"] for v in verdicts])
    return {
        "verdict_schema": VERDICT_SCHEMA,
        "run_id": run_ids[0],
        "run_ids": run_ids,
        "artifact_id": artifact_id,
        "store": store_name,
        "summary": summary,
        "benchmarks": [
            {"benchmark_id": b, "name": n, "status": worst(s), "metrics": len(s)}
            for (b, n), s in sorted(per.items(), key=lambda kv: kv[0][1] or "")
        ],
        "verdicts": verdicts,
    }


def text(doc):
    s = doc["summary"]
    head = (
        f"run {', '.join(doc['run_ids'])}: " + ", ".join(f"{s[k]} {k}" for k in SEVERITY) + f" ({s['total']} metrics)"
    )
    lines = [head]
    if not doc["verdicts"]:
        lines.append(f"no verdict rows for {', '.join(doc['run_ids'])}")
    elif doc.get("empty_run_ids"):
        lines.append(f"no verdict rows for {', '.join(doc['empty_run_ids'])}")
    for v in doc["verdicts"]:
        lines.append(f"{v['status']:<11} {v['name']}  {v['metric']}  [{v['tier']}]  {v['reason']}".rstrip())
    return "\n".join(lines)


def manifest_of(run_dir):
    """run.json of a results directory, or None when it has none (an explicit --run-dir must have one)."""
    path = Path(run_dir) / "run.json"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def resolve_ids(a):
    """(run ids, note): --run-dir means the stored ids in its run.json; an engine run id of a known run.json is translated.

    Explicit flags always beat the environment (--run-id > --run-dir > --artifact among flags): an ambient
    RUN_ID/BENCH_RUN_ID never overrides --run-dir or --artifact, an ambient BENCH_RUN_DIR never overrides --run-id, and
    an ambient directory that does not exist is ignored when a run id is given.
    """
    env_dir = cliutil.env("BENCH_RUN_DIR")
    explicit_dir = a.run_dir
    if a.run_id:
        given, run_dir = a.run_id, a.run_dir or env_dir
    elif a.run_dir:
        given, run_dir = None, a.run_dir
    elif a.artifact:
        return [], ""
    else:
        given, run_dir = cliutil.env("BENCH_RUN_ID"), env_dir
        if not given:
            env_id = cliutil.env("RUN_ID")
            given = env_id if env_id and uuid_ok(env_id) else None
    default_dir = cliutil.env("RESULTS_DIR", cliutil.env("BENCH_OUT", "results/bench"))
    manifest = manifest_of(run_dir or default_dir)
    if manifest is None and (explicit_dir or (run_dir and not given)):
        raise UsageError(f"--run-dir {run_dir}: no readable run.json")
    stored = (manifest or {}).get("stored_run_ids") or []
    if run_dir and not given:
        if not stored:
            raise UsageError(f"{run_dir}/run.json lists no stored run ids (was the run ingested?)")
        return list(stored), f"stored run ids of {run_dir}/run.json"
    if given and manifest and manifest.get("engine_run_id") == given and stored and stored != [given]:
        return list(stored), f"{given} is an engine run id; using the stored run ids {', '.join(stored)}"
    return ([given] if given else []), ""


def uuid_ok(value):
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


def execute(a, st=None):
    run_ids, note = resolve_ids(a)
    artifact = None if run_ids else cliutil.pick(a.artifact, "BENCH_SUBJECT_ID")
    if not run_ids and not artifact:
        raise UsageError("give --run-dir, --run-id or --artifact (BENCH_RUN_DIR, BENCH_RUN_ID, BENCH_SUBJECT_ID)")
    fail_on = cliutil.csv(a.fail_on)
    bad = [s for s in fail_on if s not in engine.VERDICTS]
    if bad:
        raise UsageError(f"--fail-on {bad}: one of {', '.join(engine.VERDICTS)}")
    store_name = cliutil.pick(a.store, "BENCH_STORE", "clickhouse")
    if store_name == "none":
        raise UsageError("verdict needs a store (--store clickhouse)")
    try:
        st = st or stores.open_store(store_name)
        if not run_ids:
            latest = st.latest_run(artifact)
            if not latest:
                raise UsageError(f"artifact {artifact} has no run with verdict rows")
            run_ids = [latest]
        by_id = {rid: st.verdict_rows(rid) for rid in run_ids}
        rows = [r for rid in run_ids for r in by_id[rid]]
        doc = build(rows, run_ids, artifact, store_name, cliutil.flag_on(a.quiet, "BENCH_QUIET"))
        doc["empty_run_ids"] = [rid for rid in run_ids if not by_id[rid]]
    except stores.StoreError as e:
        print(f"error: store: {e}", file=sys.stderr)
        return cliutil.EXIT_INVALID
    if note:
        print(f"note: {note}", file=sys.stderr)
    print(json.dumps(doc, indent=1) if a.json else text(doc))
    for rid in doc["empty_run_ids"]:
        print(f"warning: no verdict rows for {rid}", file=sys.stderr)
    if doc["empty_run_ids"] and a.require_rows:
        return cliutil.EXIT_FAILED
    hit = [v for v in doc["verdicts"] if v["status"] in fail_on]
    return cliutil.EXIT_FAILED if hit else cliutil.EXIT_OK


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench verdict", description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", help="a results directory: its run.json names the stored run ids (BENCH_RUN_DIR)")
    ap.add_argument("--run-id", help="a stored run UUID, or the engine id of a run in --run-dir (BENCH_RUN_ID, RUN_ID)")
    ap.add_argument("--require-rows", action="store_true", help="exit 1 when there are no verdict rows")
    ap.add_argument("--artifact", help="the subject artifact UUID: its newest run (BENCH_SUBJECT_ID)")
    ap.add_argument("--store", choices=("none", "clickhouse"), help="BENCH_STORE, default clickhouse")
    ap.add_argument("--fail-on", action="append", default=[], help="exit 1 if any metric has one of these statuses")
    ap.add_argument("--quiet", action="store_true", help="omit measured values (BENCH_QUIET=1)")
    ap.add_argument("--json", action="store_true")
    return ap


def main(argv=None, st=None):
    return cliutil.run_main(lambda a: execute(a, st), parser(), argv)


if __name__ == "__main__":
    sys.exit(main())

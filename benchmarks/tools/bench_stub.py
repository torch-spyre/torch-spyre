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

"""Card-free stand-in for the plan, run, verdict and report verbs of spyre-bench.

Exercises the Make targets and the bench-on-demand workflow end to end without hardware or a store: it plans the
cases of benchmarks/, "runs" them without measuring anything and writes the plan.json / run.json shapes the real
verbs produce, so nothing can be ingested. Selected with `BENCH="python3 benchmarks/tools/bench_stub.py"`
(what `make bench-nightly-smoke` does).
"""

import argparse
import itertools
import json
import os
import sys
import uuid
from pathlib import Path

VERBS = {
    "plan": "plan the request in BENCH_* (stub: every selected case runs)",
    "run": "run a plan without measuring (stub: writes run.json, no raw data)",
    "verdict": "print the verdict words (stub: no store, so none)",
    "report": "counts of the result directory",
}


def cases(catalog, selection, wanted):
    import yaml

    docs = [
        d
        for p in sorted(Path(catalog).rglob("*.yaml"))
        for d in yaml.safe_load_all(p.read_text())
        if d
    ]
    names = []
    for d in docs:
        if d.get("kind") != "Benchmark":
            continue
        base, spec = d["metadata"]["name"], d.get("spec") or {}
        matrix = spec.get("matrix") or {}
        axes = {
            k: [t["token"] if isinstance(t, dict) else t for t in v]
            for k, v in (matrix.get("axes") or {}).items()
        }
        if not axes:
            names += [v["name"] for v in spec.get("variants") or [{"name": base}]]
            continue
        template = matrix.get("name_template") or base + "".join(
            "_{%s}" % a for a in axes
        )
        names += [
            template.format(**dict(zip(axes, combo)))
            for combo in itertools.product(*axes.values())
        ]
    sel = next(
        (
            d
            for d in docs
            if d.get("kind") == "Selection" and d["metadata"]["name"] == selection
        ),
        None,
    )
    listed = ((sel or {}).get("spec") or {}).get("cases")
    if listed:
        names = [n for n in names if n in listed]
    if wanted:
        names = [n for n in names if n in wanted]
    return names


def cmd_plan(args):
    wanted = os.environ.get("BENCH_CASES", "").replace(",", " ").split()
    names = cases(args.catalog, os.environ.get("BENCH_SELECTION", ""), wanted)
    plan = {
        "plan_schema": "bench.plan/1",
        "stub": True,
        "purpose": os.environ.get("BENCH_PURPOSE", ""),
        "summary": {"cases": len(names), "run": len(names), "reuse": 0, "skip": 0},
        "cases": [
            {"name": n, "action": "run", "reason": "", "repeats": 1} for n in names
        ],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    print(f"plan: run {len(names)}, reuse 0, skip 0 (stub)")
    return 0


def cmd_run(args):
    if not args.plan:
        print("run: --plan is required", file=sys.stderr)
        return 2
    plan = json.loads(Path(args.plan).read_text())
    out = Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    cs = [
        {"name": c["name"], "status": "ok", "reason": "", "dir": f"cases/{c['name']}"}
        for c in plan["cases"]
        if c.get("action") == "run"
    ]
    run = {
        "run_schema": "bench.run/1",
        "stub": True,
        "run_id": os.environ.get("BENCH_RUN_ID") or str(uuid.uuid4()),
        "counts": {
            "planned": len(plan["cases"]),
            "run": len(cs),
            "ok": len(cs),
            "failed": 0,
        },
        "cases": cs,
        "exit_code": 0,
        "exit_reason": "ok",
    }
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(f"run: {len(cs)} ok, 0 failed (stub, nothing measured)")
    return 0


def cmd_verdict(args):
    if args.json:
        print(json.dumps({"stub": True, "summary": {"total": 1}}))
        return 0
    print("verdict: none (stub has no store)")
    return 0


def cmd_report(args):
    run = Path(args.dir or ".") / "run.json"
    if not run.is_file():
        print(f"report: no {run}", file=sys.stderr)
        return 2
    print(f"report: {len(json.loads(run.read_text())['cases'])} cases (stub)")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: bench_stub VERB [args]\n")
        for verb, doc in VERBS.items():
            print(f"  {verb:<10} {doc}")
        return 0
    if argv[0] not in VERBS:
        print(f"unknown verb {argv[0]}", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog=f"bench_stub {argv[0]}")
    ap.add_argument("--catalog", default="benchmarks")
    ap.add_argument("--central", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--plan", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("dir", nargs="?", default="")
    args, _ = ap.parse_known_args(argv[1:])
    return {
        "plan": cmd_plan,
        "run": cmd_run,
        "verdict": cmd_verdict,
        "report": cmd_report,
    }[argv[0]](args)


if __name__ == "__main__":
    sys.exit(main())

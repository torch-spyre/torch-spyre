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

"""spyre-bench plan: expand a request into cases, look up stored results, mark each case run | reuse | skip.

  spyre-bench plan --purpose P [--catalog DIR] [--select NAME[,NAME]] [--subject-id UUID] [--reuse MODE]
                   [--store none|clickhouse] [--out plan.json] [--json]

Flags, BENCH_* twins, the plan.json layout and exit codes: docs/README.md ("Planning from the command line").
"""

import argparse
import math
import sys
from datetime import datetime, timezone

import yaml

from spyre_bench import cliutil, engine, ingest_adapter, pinning_plan, planfile, store as stores
from spyre_bench.cliutil import UsageError

REUSE_MODES = ("never", "if_valid", "prefer", "only")


class Unbounded(dict):
    """A leg whose capacity was not stated: any device class has plenty."""

    def get(self, key, default=None):
        return 10**9


def parse_sets(items):
    """`run.repeats=3` style overrides -> the engine's overrides map."""
    allowed = {"run.repeats": "repeats", "run.cache": "cache", "run.profile_level": "profile_level"}
    out = {}
    for item in cliutil.csv(items):
        key, sep, value = item.partition("=")
        if not sep or key not in allowed:
            raise UsageError(f"--set {item!r}: use one of {sorted(allowed)} as key=value")
        out[allowed[key]] = int(value) if key == "run.repeats" and value.isdigit() else value
    if "repeats" in out and (not isinstance(out["repeats"], int) or out["repeats"] < 1):
        raise UsageError("--set run.repeats needs a positive integer")
    if out.get("cache", "warm") not in ("warm", "cold", "both"):
        raise UsageError("--set run.cache must be warm, cold or both")
    if out.get("profile_level", "off") not in ("off", "light", "full"):
        raise UsageError("--set run.profile_level must be off, light or full")
    return out


def build_request(a):
    """The engine request from the request file, BENCH_* env and flags (flag > env > file)."""
    req = {}
    path = cliutil.pick(a.request, "BENCH_REQUEST")
    if path:
        try:
            with open(path) as f:
                req = yaml.safe_load(f) or {}
        except OSError as e:
            raise UsageError(f"request file {path}: {e}") from None
        if not isinstance(req, dict):
            raise UsageError(f"request file {path}: expected a mapping")
        sel = req.get("selection") or {}
        if "name" in sel:
            sel = {**{k: v for k, v in sel.items() if k != "name"}, "cases": [sel["name"]]}
        req["selection"] = sel
        budget = req.get("budget") or {}
        if "accelerator_hours" in budget:
            req["budget"] = {"device_hours": budget["accelerator_hours"]}
    purpose = cliutil.pick(a.purpose, "BENCH_PURPOSE", req.get("purpose"))
    if not purpose:
        raise UsageError("a purpose is required (--purpose or BENCH_PURPOSE)")
    req["purpose"] = purpose
    environment = cliutil.pick(a.environment, "BENCH_ENVIRONMENT", req.get("environment"))
    if environment:
        req["environment"] = environment
    subject = cliutil.pick(a.subject_id, "BENCH_SUBJECT_ID")
    if subject:
        req["subject"] = {**(req.get("subject") or {}), "artifact_id": subject}
    sel = dict(req.get("selection") or {})
    if a.select or a.cases:  # a flag beats both env twins; they are not merged
        names, cases = cliutil.csv(a.select), cliutil.csv(a.cases)
    else:
        names, cases = cliutil.csv(cliutil.env("BENCH_SELECTION", "")), cliutil.csv(cliutil.env("BENCH_CASES", ""))
    if names or cases:
        sel["cases"] = names + cases
    tier = cliutil.pick(a.tier, "BENCH_TIER")
    suites = cliutil.csv(cliutil.pick(a.suite, "BENCH_SUITES", ""))
    if tier:
        sel["tier"] = tier
    if suites:
        sel["suites"] = suites
    kind = cliutil.pick(a.type, "BENCH_TYPE")
    if kind:
        sel["type"] = kind
    req["selection"] = sel
    mode = cliutil.pick(a.reuse, "BENCH_REUSE")
    force = cliutil.csv(cliutil.pick(a.force, "BENCH_FORCE", ""))
    max_age = cliutil.pick(a.max_age, "BENCH_MAX_AGE")
    if max_age and not engine.seconds(max_age):
        raise UsageError(f"--max-age {max_age!r}: use a positive duration such as 7d, 12h or 30m")
    if mode or force or max_age:
        if mode and mode not in REUSE_MODES:
            raise UsageError(f"--reuse {mode!r}: one of {', '.join(REUSE_MODES)}")
        base = dict(req.get("reuse") or {})
        if not (mode or base.get("mode")):
            raise UsageError("--force and --max-age need a reuse mode: add --reuse (or a reuse block in the request)")
        req["reuse"] = {
            **base,
            "mode": mode or base["mode"],
            **({"force": force} if force else {}),
            **({"max_age": max_age} if max_age else {}),
            "override": True,
        }
    budget = cliutil.pick(a.budget, "BENCH_BUDGET")
    if budget is not None:
        try:
            req["budget"] = {"device_hours": float(budget)}
        except ValueError:
            raise UsageError(f"--budget {budget!r} is not a number of device-hours") from None
    pin = build_pinning(a)
    if pin:
        req["cpu_pinning"] = pin
    sets = parse_sets([cliutil.env("BENCH_SET", "")] + (a.set or []))
    if sets:
        req["overrides"] = {**(req.get("overrides") or {}), **sets}
    check_budget(req)
    return req


PIN_OPT_INT = ("cores_per_device", "reserve")
PIN_OPT_ASSIGN = ("cores_per_device", "spread", "smt", "reserve", "reserve_cpus", "spread_scope", "llc_align")
PIN_OPT_TOP = ("layout", "launcher")


def build_pinning(a):
    """The request-level cpu_pinning block from --cpu-pinning/--cpu-set/--pin-opt and their BENCH_* twins, or None.

    It replaces the Environment's own host.cpu_pinning for this plan. Options are KEY=VALUE: layout, launcher, the
    generator knobs (cores_per_device, spread, smt, reserve, reserve_cpus, spread_scope, llc_align) and cpus_by_device
    (CPU lists separated by `;`).
    """
    policy = cliutil.pick(a.cpu_pinning, "BENCH_CPU_PINNING")
    cpus = cliutil.pick(a.cpu_set, "BENCH_CPU_SET")
    opts = cliutil.csv(cliutil.env("BENCH_PIN_OPTS", "")) + list(a.pin_opt or [])
    required = cliutil.flag_on(a.require_pinning, "BENCH_REQUIRE_PINNING")
    if not policy:
        if cpus or opts or required:
            raise UsageError("--cpu-set, --pin-opt and --require-pinning need --cpu-pinning (BENCH_CPU_PINNING)")
        return None
    if policy not in pinning_plan.POLICIES:
        raise UsageError(f"--cpu-pinning {policy!r}: one of {', '.join(pinning_plan.POLICIES)}")
    pin = {"policy": policy}
    if cpus:
        pin["cpus"] = cpus
    assign = {}
    for item in opts:
        key, sep, value = item.partition("=")
        if not sep:
            raise UsageError(f"--pin-opt {item!r}: use KEY=VALUE")
        if key in PIN_OPT_ASSIGN:
            if key in PIN_OPT_INT:
                if not value.isdigit():
                    raise UsageError(f"--pin-opt {key}: an integer")
                value = int(value)
            elif key == "llc_align":
                if value not in ("true", "false"):
                    raise UsageError("--pin-opt llc_align: true or false")
                value = value == "true"
            assign[key] = value
        elif key in PIN_OPT_TOP:
            pin[key] = value
        elif key == "cpus_by_device":
            pin[key] = [x for x in value.split(";") if x]
        else:
            raise UsageError(f"--pin-opt {key!r}: unknown option")
    if assign:
        pin["assign"] = assign
    if required:
        pin["require_verified"] = True
    return pin


def check_budget(req):
    """A budget is a finite, non-negative number of device-hours (nan would disable the check)."""
    if "budget" not in req:
        return
    try:
        hours = float((req["budget"] or {}).get("device_hours"))
    except (TypeError, ValueError, AttributeError):
        raise UsageError("budget: device_hours must be a number") from None
    if not math.isfinite(hours) or hours < 0:
        raise UsageError(f"budget {hours!r}: a finite number of device-hours >= 0")
    req["budget"] = {"device_hours": hours}


def parse_devices(text):
    out = {}
    for item in cliutil.csv(text or ""):
        key, sep, n = item.partition("=")
        if not sep or not n.isdigit():
            raise UsageError(f"--devices {item!r}: use CLASS=COUNT (for example accel/a=8)")
        out[key] = int(n)
    return out


def build_leg(a):
    devices = parse_devices(cliutil.pick(a.devices, "BENCH_DEVICES"))
    nodes = cliutil.pick(a.nodes, "BENCH_NODES")
    if nodes is not None and not (str(nodes).isdigit() and int(nodes) > 0):
        raise UsageError(f"--nodes {nodes!r}: a positive integer")
    leg = {
        "arch": cliutil.pick(a.arch, "BENCH_ARCH", cliutil.host_arch()),
        "devices": devices,
        "nodes": int(nodes) if nodes else None,
        "reference_node": cliutil.flag_on(a.reference_node, "BENCH_REFERENCE_NODE"),
        "env_id": cliutil.pick(a.env_id, "BENCH_ENV_ID"),
    }
    return leg


def engine_leg(leg):
    """The leg as engine.plan reads it: stated capacity is checked, unstated capacity is unbounded."""
    return {
        **leg,
        "devices": leg["devices"] or Unbounded(),
        "nodes": leg["nodes"] if leg["nodes"] else 10**9,
    }


def case_values(cat, c):
    ident = cat._identity(c)
    values = {k: planfile.token_value(v) for k, v in {**ident["props"], **ident["disc"]}.items()}
    values.update(case=c["name"], benchmark=c["benchmark"], component=cat.component, environment=c["environment"] or "")
    return values


def decorate(cat, rec, errors, request=None):
    """Add what `run` needs beyond the engine record: ids' profile, labels, pinning, the command templates."""
    c = next(x for x in cat.cases if x["name"] == rec["name"])
    ingest = cat.ingest_inputs(c)
    env = cat.effective("Environment", rec["environment"]) if rec.get("environment") else None
    if env is not None:
        env = engine.overlay_pinning(env, (request or {}).get("cpu_pinning"))
    command = c["run"].get("command")
    if command:
        command, errs = planfile.render_case_tokens(command, case_values(cat, c), f"case {c['name']}")
        errors += errs
    action, reason = rec["action"], rec["reason"]
    if action == "missing":
        action, reason = "skip", f"missing:{reason}"
    return {
        **rec,
        "action": action,
        "reason": reason,
        "benchmark": c["benchmark"],
        "id_mode": rec.get("id_mode", "engine"),
        "id_profile": ingest["profile"] if ingest else None,
        **({"test_type": c["type"]} if c["type"] != "perf" else {}),
        "labels": cat.env_labels(env) if env else {},
        "pinning": engine.pin_block(env),
        "command": command or None,
        "ingest": c["run"].get("ingest") or None,
        "allow_empty": bool(c["run"].get("allow_empty")),
    }


def make_plan(cat, request, leg, st, now, store_name="none"):
    """The plan document. Raises engine.PlanError, IngestUnavailable or UsageError; StoreError when a lookup fails."""
    eleg = engine_leg(leg)
    # The budget is judged on the final pass only: a case that will be reused costs nothing.
    probe = {k: v for k, v in request.items() if k != "budget"}
    first = engine.plan(cat, probe, eleg, (), (), now)
    wanted = [r for r in first["cases"] if "reuse_key" in r]
    existing = []
    if wanted:
        existing = st.stored_results(
            sorted({r["benchmark_id"] for r in wanted}), (request.get("subject") or {}).get("artifact_id")
        )
    body = engine.plan(cat, request, eleg, existing, st.invalidations([r["run_id"] for r in existing]), now)
    errors = []
    records = [decorate(cat, r, errors, request) for r in body["cases"]]
    if errors:
        raise UsageError("\n".join(dict.fromkeys(errors)))
    counts = {"run": 0, "reuse": 0, "skip": 0}
    reasons = {}
    for r in records:
        counts[r["action"]] += 1
        if r["action"] == "skip":
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    ingest = ingest_adapter.info()
    shown_leg = {**leg, "devices": leg["devices"] or None}
    return {
        "plan_schema": planfile.PLAN_SCHEMA,
        "producer": {
            "engine": planfile.producer(),
            "ingest": {"available": ingest["available"], "version": ingest["version"]},
        },
        "created_at": cliutil.utc_stamp(now),
        "purpose": request.get("purpose"),
        "request": request,
        "leg": shown_leg,
        "catalog": {"component": cat.component, "cases_total": len(cat.cases)},
        "store": store_name,
        "summary": {"cases": len(records), **counts, "reasons": dict(sorted(reasons.items()))},
        "estimate": body["estimate"],
        "cases": records,
    }


def table(plan):
    rows = [("ACTION", "CASE", "REPEATS", "REASON")] + [
        (r["action"], r["name"], str(r["repeats"]), r["reason"]) for r in plan["cases"]
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row[:3], widths)) + "  " + row[3] for row in rows]
    s = plan["summary"]
    lines.append(f"plan: run {s['run']}, reuse {s['reuse']}, skip {s['skip']}")
    return "\n".join(line.rstrip() for line in lines)


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench plan", description=__doc__.split("\n")[0])
    cliutil.add_catalog_flags(ap)
    ap.add_argument("--request", help="a Run request YAML file (BENCH_REQUEST)")
    ap.add_argument("--purpose", help="BENCH_PURPOSE")
    ap.add_argument("--environment", help="BENCH_ENVIRONMENT")
    ap.add_argument("--select", "--selection", dest="select", help="Selection or case names, comma separated")
    ap.add_argument("--cases", help="case names, comma separated (BENCH_CASES)")
    ap.add_argument("--tier", help="BENCH_TIER")
    ap.add_argument("--suite", action="append", help="BENCH_SUITES")
    ap.add_argument("--type", help="perf (default) or quality (BENCH_TYPE)")
    ap.add_argument("--subject-id", help="the subject artifact id (BENCH_SUBJECT_ID)")
    ap.add_argument("--reuse", help="never | if_valid | prefer | only; overrides the Selection's own rule")
    ap.add_argument("--force", action="append", help="case names to run despite a stored result, or all")
    ap.add_argument("--max-age", help="oldest reusable result, such as 7d")
    ap.add_argument(
        "--set", action="append", help="run.repeats=N, run.cache=C, run.profile_level=L (env twin merged per key)"
    )
    ap.add_argument("--budget", help="device-hours the plan may need")
    ap.add_argument("--arch", help="arch of this leg (default: the host's)")
    ap.add_argument("--devices", help="CLASS=COUNT[,CLASS=COUNT]: capacity of this leg (default: unchecked)")
    ap.add_argument("--nodes", help="nodes of this leg (default: unchecked)")
    ap.add_argument("--reference-node", action="store_true", help="this leg is a reference node")
    ap.add_argument("--env-id", help="env_id of this leg, required for exact-environment reuse")
    ap.add_argument("--store", choices=("none", "clickhouse"), help="reuse lookup source (BENCH_STORE, default none)")
    ap.add_argument(
        "--cpu-pinning",
        choices=pinning_plan.POLICIES,
        help="request this CPU pinning policy, replacing the Environment's own (BENCH_CPU_PINNING)",
    )
    ap.add_argument("--cpu-set", help="CPU list for --cpu-pinning explicit (BENCH_CPU_SET)")
    ap.add_argument(
        "--pin-opt",
        action="append",
        help="KEY=VALUE for the pinning request: layout, launcher, cores_per_device, spread, smt, ... (BENCH_PIN_OPTS)",
    )
    ap.add_argument(
        "--require-pinning",
        action="store_true",
        help="cap the result at suspect unless verified (BENCH_REQUIRE_PINNING)",
    )
    ap.add_argument("--out", help="write plan.json here (BENCH_PLAN_OUT)")
    ap.add_argument("--json", action="store_true", help="print the plan document instead of the table")
    ap.add_argument("--now", help=argparse.SUPPRESS)
    return ap


def execute(a, st=None):
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else datetime.now(timezone.utc)
    cat = cliutil.open_catalog(a.catalog, a.central, a.pack)
    request, leg = build_request(a), build_leg(a)
    store_name = cliutil.pick(a.store, "BENCH_STORE", "none")
    try:
        st = st or stores.open_store(store_name)
        plan = make_plan(cat, request, leg, st, now, store_name)
    except engine.PlanError as e:
        raise UsageError(str(e)) from None
    except ingest_adapter.IngestUnavailable as e:
        raise UsageError(f"a stored benchmark_id needs the ingest derivation: {e}") from None
    except stores.StoreError as e:
        print(f"error: store: {e}", file=sys.stderr)
        return cliutil.EXIT_STORE
    out = cliutil.pick(a.out, "BENCH_PLAN_OUT")
    if out:
        with open(out, "w") as f:
            f.write(planfile.dump(plan))
    print(planfile.dump(planfile.redact_plan(plan)) if a.json else table(plan))
    return cliutil.EXIT_NOTHING if plan["summary"]["skip"] == plan["summary"]["cases"] else cliutil.EXIT_OK


def main(argv=None, st=None):
    return cliutil.run_main(lambda a: execute(a, st), parser(), argv)


if __name__ == "__main__":
    sys.exit(main())

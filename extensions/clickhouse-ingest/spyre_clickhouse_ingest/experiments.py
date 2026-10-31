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

"""Write experiment bundles (JSONL, format in schema/README.md) into the 75- experiment ledger.

`write` loads bundles; `abab` turns a paired A/B driver's samples.jsonl into a bundle.
Every id is derived from content, so writing a bundle twice adds no rows.
"""

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import regex

from . import schema
from .identity import DerivedId

NIL = "00000000-0000-0000-0000-000000000000"
RECORDS = ("experiment", "arm", "env", "sample", "decision", "finding", "link")
# The A/B driver's per-pair outcome; anything but these is not a sample.
ABAB_SAMPLE_KINDS = ("warmup", "sample")
RPM_NAME = regex.compile(r"^(.+?)-(\d.*)$")


def _id(*parts: Any) -> str:
    return DerivedId.hash(*(str(p) for p in parts))


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def _ts(value: Any) -> datetime:
    """Epoch seconds or ISO-8601 (naive = UTC) as an aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (
        dt.replace(tzinfo=timezone.utc)
        if dt.tzinfo is None
        else dt.astimezone(timezone.utc)
    )


def _num(value: Any, default: float = math.nan) -> float:
    return default if value is None else float(value)


def _opt(value: Any) -> float | None:
    """NULL for a missing or NaN statistic: Nullable columns say "not computed"."""
    return None if value is None or value != value else float(value)


def _strs(m: Mapping | None) -> dict[str, str]:
    return {str(k): str(v) for k, v in (m or {}).items()}


class Bundle:
    """Bundle records -> rows per ledger table, validated by the table models."""

    def __init__(self, actor: str = "", actor_kind: str = "service"):
        self.actor, self.actor_kind = actor, actor_kind
        self.rows: dict[type[schema.Table], dict[str, dict]] = {
            t: {} for t in schema.EXPERIMENT_TABLES
        }

    def add(self, table: type[schema.Table], row: dict) -> None:
        table.row(row)
        self.rows[table][str(row[table.identity])] = row

    def load(self, records: Iterable[Mapping[str, Any]]) -> "Bundle":
        for n, rec in enumerate(records, 1):
            kind = rec.get("record")
            if kind not in RECORDS:
                raise schema.SchemaError(
                    f"record {n}: 'record' must be one of {RECORDS}, got {kind!r}"
                )
            getattr(self, "_" + kind)(rec)
        return self

    def _who(self, rec: Mapping) -> dict:
        actor = rec.get("actor") or self.actor
        if not actor:
            raise schema.SchemaError("no actor: set it on the record or pass --actor")
        return {"actor": actor, "actor_kind": rec.get("actor_kind") or self.actor_kind}

    @staticmethod
    def experiment_id(name: str) -> str:
        return _id("experiment", name)

    def _experiment(self, r: Mapping) -> None:
        eid = self.experiment_id(r["name"])
        row = {
            "ts": _ts(r["ts"]),
            "experiment_id": eid,
            "parent_id": self.experiment_id(r["parent"]) if r.get("parent") else NIL,
            "name": r["name"],
            "kind": r.get("kind", "other"),
            "state": r["state"],
            "subject_level": r.get("subject_level", ""),
            "subject_key": r.get("subject_key", ""),
            "workload": r.get("workload", ""),
            "hypothesis": r.get("hypothesis", ""),
            "spec": json.dumps(r["spec"], sort_keys=True)
            if r.get("spec") is not None
            else "",
            "spec_sha256": _sha(r["spec"]) if r.get("spec") is not None else "",
            "budget": {k: float(v) for k, v in (r.get("budget") or {}).items()},
            "spent": {k: float(v) for k, v in (r.get("spent") or {}).items()},
            "code_ref": r.get("code_ref", ""),
            "reason": r.get("reason", ""),
            "on_behalf_of": r.get("on_behalf_of", ""),
            "props": _strs(r.get("props")),
            **self._who(r),
        }
        row["event_id"] = _id(
            "experiment_event", eid, _sha({k: v for k, v in row.items() if k != "ts"})
        )
        self.add(schema.ExperimentEvents, row)

    def _arm(self, r: Mapping) -> None:
        eid = self.experiment_id(r["experiment"])
        knobs = _strs(r.get("knobs"))
        row = {
            "ts": _ts(r["ts"]),
            "experiment_id": eid,
            "arm": r["arm"],
            "role": r["role"],
            "state": r.get("state", "active"),
            "hypothesis": r.get("hypothesis", ""),
            "knobs": knobs,
            "knob_sha": _sha(knobs),
            "change_class": r.get("change_class", ""),
            "hazard_class": r.get("hazard_class", ""),
            "props": _strs(r.get("props")),
            **self._who(r),
        }
        row["event_id"] = _id(
            "arm_event", eid, _sha({k: v for k, v in row.items() if k != "ts"})
        )
        self.add(schema.ExperimentArms, row)

    @staticmethod
    def env_id(env_sha: str) -> str:
        return _id("env", env_sha) if env_sha else NIL

    def _env(self, r: Mapping) -> None:
        sha = r.get("env_sha") or _sha(
            {k: v for k, v in r.items() if k not in ("record", "ts")}
        )
        pb = r.get("profiler_build")
        self.add(
            schema.RunEnvReports,
            {
                "ts": _ts(r["ts"]),
                "env_id": self.env_id(sha),
                "env_sha": sha,
                "host": r.get("host", ""),
                "node": r.get("node", ""),
                "arch": DerivedId.arch(r.get("arch", "")),
                "cpu_model": r.get("cpu_model", ""),
                "ncpu": int(r.get("ncpu") or 0),
                "cpu_quota": str(r.get("cpu_quota", "")),
                "host_kernel": r.get("host_kernel", ""),
                "card_class": r.get("card_class", ""),
                "cards": [str(c) for c in r.get("cards") or []],
                "image": r.get("image", ""),
                "image_digest": r.get("image_digest", ""),
                "artifact_id": r.get("artifact_id") or NIL,
                "git_shas": _strs(r.get("git_shas")),
                "packages": _strs(r.get("packages")),
                "env": _strs(r.get("env")),
                "profiler_build": None
                if pb is None
                else str(pb).lower() in ("1", "true", "yes"),
                "load1": float(r.get("load1") or 0),
                "raw": r.get("raw", ""),
                "props": _strs(r.get("props")),
            },
        )

    def _sample(self, r: Mapping) -> None:
        eid = self.experiment_id(r["experiment"])
        metrics = r.get("metrics") or {}
        status = r.get("status", "ok")
        if status == "ok" and not metrics:
            raise schema.SchemaError(
                f"{r['experiment']}/{r['arm']}: an ok sample needs metrics"
            )
        units = r.get("units") or {}
        warmup = bool(r.get("warmup", False))
        key = (eid, r["arm"], int(r.get("trial", 0)), int(r.get("position", 0)), warmup)
        for metric, value in metrics.items() or [("", math.nan)]:
            self.add(
                schema.ExperimentSamples,
                {
                    "ts": _ts(r["ts"]),
                    "sample_id": _id("sample", *key, metric),
                    "experiment_id": eid,
                    "arm": r["arm"],
                    "trial": key[2],
                    "position": key[3],
                    "warmup": warmup,
                    "split": r.get("split", "search"),
                    "status": status,
                    "workload": r.get("workload", ""),
                    "stratum": r.get("stratum", ""),
                    "metric": metric,
                    "value": _num(value),
                    "unit": units.get(metric, ""),
                    "kind": r.get("kind", "measured"),
                    "fidelity": r.get("fidelity", "hardware"),
                    "wall_s": _num(r.get("wall_s"), 0.0),
                    "load1_before": _num(r.get("load1_before"), 0.0),
                    "load1_after": _num(r.get("load1_after"), 0.0),
                    "node": r.get("node", ""),
                    "env_id": self.env_id(r.get("env_sha", "")),
                    "run_id": r.get("run_id") or NIL,
                    "benchmark_id": r.get("benchmark_id") or NIL,
                    "props": _strs(r.get("props")),
                },
            )

    def _decision(self, r: Mapping) -> None:
        eid = self.experiment_id(r["experiment"])
        final = bool(r.get("final", False))
        self.add(
            schema.ExperimentDecisions,
            {
                "ts": _ts(r["ts"]),
                "decision_id": _id(
                    "decision", eid, r["arm"], r["metric"], r["n"], final, r["ts"]
                ),
                "experiment_id": eid,
                "arm": r["arm"],
                "baseline_arm": r["baseline_arm"],
                "metric": r["metric"],
                "final": final,
                "n": int(r["n"]),
                "test": r.get("test", ""),
                "effect_pct": _num(r.get("effect_pct")),
                "ci_lo_pct": _num(r.get("ci_lo_pct")),
                "ci_hi_pct": _num(r.get("ci_hi_pct")),
                "sd_log": _num(r.get("sd_log")),
                "e_value": _opt(r.get("e_value")),
                "p_value": _opt(r.get("p_value")),
                "floor_pct": _opt(r.get("floor_pct")),
                "decision": r["decision"],
                "stop_reason": r.get("stop_reason", ""),
                "props": _strs(r.get("props")),
            },
        )

    @classmethod
    def finding_id(cls, experiment: str, key: str) -> str:
        return _id("finding", cls.experiment_id(experiment), key)

    def _finding(self, r: Mapping) -> None:
        eid = self.experiment_id(r["experiment"])
        row = {
            "ts": _ts(r["ts"]),
            "finding_id": self.finding_id(r["experiment"], r["key"]),
            "experiment_id": eid,
            "kind": r["kind"],
            "arm": r.get("arm", ""),
            "baseline_arm": r.get("baseline_arm", ""),
            "subject_level": r.get("subject_level", ""),
            "subject_key": r.get("subject_key", ""),
            "workload": r.get("workload", ""),
            "stratum": r.get("stratum", ""),
            "metric": r["metric"],
            "unit": r.get("unit", "pct"),
            "estimate": float(r["estimate"]),
            "ci_lo": _num(r.get("ci_lo")),
            "ci_hi": _num(r.get("ci_hi")),
            "n": int(r.get("n") or 0),
            "floor_pct": _num(r.get("floor_pct")),
            "verdict": r.get("verdict", ""),
            "gain_evidence": r.get("gain_evidence", ""),
            "demoted_by": list(r.get("demoted_by") or []),
            "gates": _strs(r.get("gates")),
            "policy_version": r.get("policy_version", ""),
            "state": r.get("state", "published"),
            "summary": r.get("summary", ""),
            "props": _strs(r.get("props")),
            **self._who(r),
        }
        content = {k: v for k, v in row.items() if k != "ts"}
        row["event_id"] = _id("finding_event", row["finding_id"], _sha(content))
        self.add(schema.ExperimentFindings, row)

    def _link(self, r: Mapping) -> None:
        eid = self.experiment_id(r["experiment"])
        fid = (
            self.finding_id(r["experiment"], r["finding"]) if r.get("finding") else NIL
        )
        target_id, target = r.get("target_id") or NIL, r.get("target", "")
        self.add(
            schema.ExperimentLinks,
            {
                "ts": _ts(r["ts"]),
                "link_id": _id("link", eid, fid, r["kind"], target_id, target),
                "experiment_id": eid,
                "finding_id": fid,
                "link_kind": r["kind"],
                "role": r.get("role", "evidence"),
                "target_id": target_id,
                "target": target,
                "props": _strs(r.get("props")),
            },
        )

    def write(self, client, db: str) -> dict[str, int]:
        """Insert rows the database does not hold yet, env reports first."""
        order = (
            schema.RunEnvReports,
            *[t for t in schema.EXPERIMENT_TABLES if t is not schema.RunEnvReports],
        )
        return {t.name: t.insert_identities(client, self.rows[t], db=db) for t in order}

    def counts(self) -> dict[str, int]:
        return {t.name: len(rows) for t, rows in self.rows.items()}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── the paired A/B driver's samples.jsonl (kind env|warmup|sample|numerics|interim|result) ──


def _abab_env(report: Mapping, ts: Any) -> dict:
    env = report.get("env") or {}
    art = regex.search(
        r'"artifact_id":\s*"([0-9a-f-]{36})"', str(report.get("image_artifact", ""))
    )
    packages = {}
    for nevra in str(report.get("rpms", "")).split():
        m = RPM_NAME.match(nevra)
        if m:
            packages[m.group(1)] = m.group(2)
    kernel = str(report.get("kernel", ""))
    load = report.get("load") or [0]
    return {
        "record": "env",
        "ts": report.get("ts", ts),
        "env_sha": report.get("sha", ""),
        "host": report.get("host", ""),
        "node": report.get("node", ""),
        "arch": kernel.rsplit(".", 1)[-1] if "." in kernel else "",
        "cpu_model": report.get("cpu_model", ""),
        "ncpu": report.get("ncpu", 0),
        "cpu_quota": report.get("cpu_quota", ""),
        "host_kernel": kernel,
        "card_class": env.get("FLEX_DEVICE", "")
        if env.get("FLEX_DEVICE") in ("PF", "VF")
        else "",
        "cards": [
            c
            for v in (report.get("pci_env") or {}).values()
            for c in str(v).split(",")
            if c
        ],
        "artifact_id": art.group(1) if art else "",
        "packages": packages,
        "env": env,
        "profiler_build": report.get("profiler_build"),
        "load1": load[0],
        "raw": json.dumps(report, sort_keys=True, default=str),
    }


def abab_bundle(
    rows: list[Mapping],
    spec: Mapping,
    name: str,
    parent: str = "",
    status: Mapping | None = None,
    actor: str = "",
) -> list[dict]:
    """Bundle records for one driver run; `spec` is the driver's spec JSON."""
    arms = spec["arms"]
    a, b = arms[0]["name"], arms[1]["name"]
    aa = arms[0].get("env") == arms[1].get("env") and arms[0].get("args", "") == arms[
        1
    ].get("args", "")
    ts0 = next((r["ts"] for r in rows if "ts" in r), 0)
    workload = spec.get("workload", "")
    base = {"experiment": name, "actor": actor} if actor else {"experiment": name}
    out: list[dict] = []
    env_rec = next(
        (_abab_env(r["report"], r["ts"]) for r in rows if r.get("kind") == "env"), None
    )
    stratum, env_sha, profiler = "", "", None
    if env_rec:
        out.append(env_rec)
        env_sha, profiler = env_rec["env_sha"], env_rec["profiler_build"]
        stratum = "/".join(
            [env_rec["arch"], env_rec["card_class"], env_rec["cpu_model"]]
        )
    result = next((r for r in rows if r.get("kind") == "result"), None)
    state = "done" if result else (status or {}).get("state", "running")
    out.append(
        {
            "record": "experiment",
            "name": name,
            "parent": parent,
            "ts": ts0,
            "state": state,
            "kind": "noise" if aa else "ab",
            "workload": workload,
            "subject_level": "workload",
            "subject_key": workload,
            "spec": spec,
            "budget": {"card_min": status["budget_cap_card_min"]}
            if status and "budget_cap_card_min" in status
            else {},
            "spent": {"card_min": status["budget_used_card_min"]}
            if status and "budget_used_card_min" in status
            else {},
            "reason": result["stop"] if result else "",
            **({"actor": actor} if actor else {}),
        }
    )
    for i, arm in enumerate(arms):
        knobs = {f"env.{k}": v for k, v in (arm.get("env") or {}).items()}
        if arm.get("args"):
            knobs["args"] = arm["args"]
        out.append(
            {
                **base,
                "record": "arm",
                "ts": ts0,
                "arm": arm["name"],
                "role": "baseline" if i == 0 else "candidate",
                "knobs": knobs,
                "change_class": arm.get(
                    "change_class", "runtime_knob" if knobs else ""
                ),
            }
        )
    warmups: dict[str, int] = {}
    positions: dict[tuple, int] = {}
    gates: dict[str, str] = {}
    for r in rows:
        kind = r.get("kind")
        if kind in ABAB_SAMPLE_KINDS:
            warm = kind == "warmup"
            # A warm-up has no pair: its trial is the arm's warm-up count.
            trial = warmups.setdefault(r["arm"], 0) if warm else int(r.get("pair", 0))
            if warm:
                warmups[r["arm"]] += 1
            pos = positions.get((warm, trial), 0)
            positions[(warm, trial)] = pos + 1
            out.append(
                {
                    **base,
                    "record": "sample",
                    "ts": r["ts"],
                    "arm": r["arm"],
                    "trial": trial,
                    "position": pos,
                    "warmup": warm,
                    "status": r.get("status", "ok"),
                    "workload": workload,
                    "stratum": stratum,
                    "metrics": r.get("metrics") or {},
                    "wall_s": r.get("wall_s"),
                    "load1_before": r.get("load1_before"),
                    "load1_after": r.get("load1_after"),
                    "env_sha": env_sha,
                    "props": {
                        "run_dir": r.get("run_dir", ""),
                        "order": r.get("order", ""),
                    },
                }
            )
        elif kind == "numerics":
            gates["numerics"] = (
                "passed"
                if r.get("match") and gates.get("numerics") != "failed"
                else "failed"
            )
        elif kind == "interim":
            out.append(
                {
                    **base,
                    "record": "decision",
                    "ts": r["ts"],
                    "arm": b,
                    "baseline_arm": a,
                    "metric": spec.get("primary", ""),
                    "n": r["n"],
                    "test": "msprt",
                    "effect_pct": r.get("effect_pct"),
                    "ci_lo_pct": r.get("ci_lo_pct"),
                    "ci_hi_pct": r.get("ci_hi_pct"),
                    "sd_log": r.get("sd_log"),
                    "e_value": r.get("e_value"),
                    "p_value": r.get("t_p"),
                    "decision": r["decision"],
                }
            )
    if result:
        out += _abab_result(result, spec, base, a, b, aa, stratum, gates, profiler)
    return out


def _abab_result(res, spec, base, a, b, aa, stratum, gates, profiler) -> list[dict]:
    """Final decisions plus one finding per blocking metric: a noise floor for A/A, else an effect."""
    out = []
    ts = res.get("ts", 0)
    for metric, s in (res.get("metrics") or {}).items():
        decision = (
            s["decision"] if s["decision"] in schema.DECISION_VALUES else "stopped"
        )
        out.append(
            {
                **base,
                "record": "decision",
                "ts": ts,
                "arm": b,
                "baseline_arm": a,
                "metric": metric,
                "final": True,
                "n": s["n"],
                "test": "paired_log_t",
                "effect_pct": s.get("effect_pct"),
                "ci_lo_pct": s.get("ci_lo_pct"),
                "ci_hi_pct": s.get("ci_hi_pct"),
                "sd_log": s.get("sd_log"),
                "e_value": s.get("e_value"),
                "p_value": s.get("t_p"),
                "decision": decision,
                "stop_reason": res.get("stop", ""),
            }
        )
        common = {
            **base,
            "record": "finding",
            "ts": ts,
            "key": metric,
            "metric": metric,
            "workload": spec.get("workload", ""),
            "stratum": stratum,
            "subject_level": "workload",
            "subject_key": spec.get("workload", ""),
            "n": s["n"],
            "policy_version": "abab-v1",
        }
        if aa:
            sd = s.get("sd_log")
            if sd is None or sd != sd:
                continue
            out.append(
                {
                    **common,
                    "kind": "noise_floor",
                    "estimate": 100 * (math.exp(sd / math.sqrt(2)) - 1),
                    "summary": "A/A run sd from paired log differences",
                }
            )
            continue
        demoted = [
            r
            for r, bad in (
                ("profiler_build", str(profiler).lower() == "true"),
                ("no_floor", s.get("floor_sd_log") is None),
            )
            if bad
        ]
        confirmed = (
            decision == "effect"
            and s.get("confirmed")
            and gates.get("numerics") != "failed"
        )
        effect = _num(s.get("effect_pct"))
        verdict = (
            ("better" if effect > 0 else "worse")
            if decision == "effect"
            else ("equal" if decision == "null" else "inconclusive")
        )
        # G2 = confirmed same-card A/B; each demotion drops one level, never below observed (G1).
        level = max((2 if confirmed else 1) - len(demoted), 1)
        out.append(
            {
                **common,
                "kind": "effect",
                "arm": b,
                "baseline_arm": a,
                "estimate": effect,
                "ci_lo": s.get("ci_lo_pct"),
                "ci_hi": s.get("ci_hi_pct"),
                "verdict": verdict,
                "gain_evidence": f"G{level}",
                "demoted_by": demoted,
                "gates": gates,
            }
        )
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="", help="default: $CLICKHOUSE_DB_V2")
    sub = parser.add_subparsers(dest="cmd", required=True)
    wr = sub.add_parser("write", help="load bundle JSONL files")
    wr.add_argument("bundles", type=Path, nargs="+")
    wr.add_argument("--actor", default="", help="for records that name none")
    wr.add_argument("--actor-kind", default="service")
    wr.add_argument(
        "--dry-run", action="store_true", help="validate and count, write nothing"
    )
    ab = sub.add_parser(
        "abab", help="a paired A/B driver's samples.jsonl -> bundle JSONL on stdout"
    )
    ab.add_argument("samples", type=Path)
    ab.add_argument("--spec", type=Path, required=True)
    ab.add_argument("--name", required=True, help="experiment name")
    ab.add_argument("--parent", default="", help="campaign name")
    ab.add_argument(
        "--status", type=Path, help="the driver's status.json, for budget and spend"
    )
    ab.add_argument("--actor", default="")
    args = parser.parse_args(argv)

    if args.cmd == "abab":
        status = json.loads(args.status.read_text()) if args.status else None
        for rec in abab_bundle(
            read_jsonl(args.samples),
            json.loads(args.spec.read_text()),
            args.name,
            args.parent,
            status,
            args.actor,
        ):
            print(json.dumps(rec, default=str))
        return

    bundle = Bundle(args.actor, args.actor_kind)
    for path in args.bundles:
        bundle.load(read_jsonl(path))
    if args.dry_run:
        print(json.dumps(bundle.counts(), sort_keys=True))
        return

    from .client import ClickHouse, ClickHouseEnv

    db = args.database or ClickHouseEnv.target_database()
    if not db:
        sys.exit("[error] no database: pass --database or set CLICKHOUSE_DB_V2")
    done = bundle.write(ClickHouse.connect(database=db), db)
    print(f"[info] {db}: " + ", ".join(f"{k} {v}" for k, v in done.items()))


if __name__ == "__main__":
    main()

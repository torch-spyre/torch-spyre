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

"""`vs=`: what an on-demand run is compared with, resolved read-only against the store.

Pure over (plan, vs, store rows). The other side obeys the verdict trust rules: only a valid, non-external,
non-invalidated stored result counts. A PR run is capped at `suspected` and is never a baseline; a reference is
reachable only through `ref:` and is a comparison target, never a baseline. Grammar and rules: docs/README.md
("Comparing with vs=").
"""

import hashlib
import json
import math
from collections import namedtuple

import regex

from spyre_bench import engine

SCHEMA = "bench.compare/1"
KINDS = ("main", "tag", "build", "artifact", "ref")
RUNNABLE = ("tag", "build", "artifact")  # the other side may be measured here when it has no usable result
MAIN_PURPOSES = ("nightly", "weekly")  # purposes whose results `vs=main` may use
VERDICT_CAP = "suspected"
FORM = "main, tag:NAME, build:JOB#N, artifact:UUID or ref:KEY"

_NAME = regex.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BUILD = regex.compile(r"build:([A-Za-z0-9][A-Za-z0-9._/-]{0,79}#[0-9]{1,9})")
_UUID = regex.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_FIX = ("no_stored_result", "different_environment")  # reasons re-measuring the other side can repair


class Vs(namedtuple("Vs", "kind value")):
    """What to compare with; `main` has no value."""

    @property
    def text(self):
        return "main" if self.kind == "main" else f"{self.kind}:{self.value}"


def parse_vs(text):
    """-> Vs, or ValueError with the accepted forms. `nightly` is the old spelling of `main`."""
    s = str(text)
    if s in ("main", "nightly"):
        return Vs("main", "")
    m = _BUILD.fullmatch(s)
    if m and ".." not in s:
        return Vs("build", m.group(1))
    kind, sep, value = s.partition(":")
    ok = {"tag": _NAME, "artifact": _UUID, "ref": _NAME}.get(kind)
    if sep and ok and ok.fullmatch(value):
        return Vs(kind, value)
    raise ValueError(f"vs={s!r}: {FORM}")


class Refused(Exception):
    """The comparison target cannot be used at all."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def canon_arch(arch):
    """The platform spelling the stored side uses (amd64 and x86 fold to x86_64)."""
    return "x86_64" if str(arch) in ("amd64", "x86", "x86-64") else str(arch or "")


def short(artifact_id):
    return str(artifact_id or "")[:8]


def _invalidated(row, events):
    for ev in engine.active_invalidations(events):
        k = ev.get("scope_kind")
        if k in ("run", "run_benchmark") and ev.get("run_id") == row.get("run_id"):
            if k == "run" or ev.get("benchmark_id") == row.get("benchmark_id"):
                return True
    return False


def usable_targets(vs, candidates, *, product, arch, subject_artifact, vs_kinds):
    """Every candidate that may be compared with, newest first. Raises Refused when none is."""
    if vs.kind not in vs_kinds:
        raise Refused("vs_not_allowed", f"vs={vs.kind} is not available here")
    if not candidates:
        raise Refused("vs_not_found", f"nothing matches vs={vs.text}")
    seen, out = set(), []
    for c in candidates:
        if vs.kind != "ref" and (c.get("source") == "reference" or c.get("component") == "reference"):
            raise Refused("reference_only_via_ref", "a reference is a comparison target only through ref:<key>")
        if vs.kind == "ref":
            if c.get("ref_state") != "current":
                seen.add(f"reference is {c.get('ref_state') or 'not approved'}")
                continue
            if c.get("product") != product:
                seen.add("another product")
                continue
        elif c.get("component") != product:
            seen.add("another product")
            continue
        if c.get("artifact_id") == subject_artifact:
            if vs.kind == "main":
                continue  # the artifact being measured is not a nightly to compare with
            raise Refused("vs_is_subject", "vs names the artifact being measured")
        if not arch or canon_arch(c.get("arch")) != canon_arch(arch):
            seen.add("another or unknown architecture")
            continue
        out.append(c)
    if not out:
        raise Refused("vs_unusable", f"vs={vs.text} is unusable here: {'; '.join(sorted(seen))}")
    return out


def _row_issue(row, kind, *, env_id, labels, relax, events, purposes):
    """Why one stored result cannot be the other side, or ''."""
    if _invalidated(row, events):
        return "invalidated"
    if row.get("trust", "internal") == "external":
        return "trust_external"
    if not engine.validity_at_least(row.get("validity"), "valid"):
        return f"validity_{row.get('validity') or 'unknown'}"
    if kind == "ref" and not row.get("reference"):
        return "not_a_reference_value"
    if kind != "ref" and (row.get("reference") or row.get("purpose") == "reference"):
        return "reference_only_via_ref"
    if kind == "main" and row.get("purpose") not in purposes:
        return "not_a_main_result"
    if row.get("reference"):
        return ""  # a reference has no environment id of its own; the arch was matched on the target
    if relax:
        if not engine.labels_match(row.get("labels", {}), labels, relax):
            return "different_environment"
        return ""
    if not env_id:
        return "environment_unknown"
    if row.get("env_id") != env_id:
        return "different_environment"
    if engine.pin_label(row.get("labels")) != engine.pin_label(labels):
        return "different_pinning"
    return ""


def _other_side(rows, kind, **kw):
    """(row, reason): the newest usable stored result, else why the newest one was not."""
    why = "no_stored_result"
    for i, row in enumerate(sorted(rows, key=lambda r: str(r.get("ts")), reverse=True)):
        issue = _row_issue(row, kind, **kw)
        if not issue:
            return row, ""
        if i == 0:
            why = issue
    return None, why


def _hours(case):
    """Device-hours of one case, or None when its estimate is not a usable number."""
    est, acc = case.get("est_min"), case.get("accelerators")
    if not isinstance(est, (int, float)) or isinstance(est, bool) or not math.isfinite(est) or est < 0:
        return None
    return est * (2 if case.get("cache") == "both" else 1) * (acc if isinstance(acc, (int, float)) else 0) / 60


def _evaluate(cases, rows, events, vs, target, leg, relax, purposes):
    """The per-case verdicts of one target."""
    by, out = {}, []
    for r in rows:
        by.setdefault(r["benchmark_id"], []).append(r)
    for c in cases:
        if c["action"] not in ("run", "reuse"):
            out.append(_case(c, "cannot_compare", f"subject_{c['action']}:{c.get('reason') or ''}"))
            continue
        row, why = _other_side(
            by.get(c.get("benchmark_id"), []),
            vs.kind,
            env_id=leg.get("env_id"),
            labels=c.get("labels") or {},
            relax=tuple(relax),
            events=events,
            purposes=purposes,
        )
        if row:
            out.append(_case(c, "ready", "", other=row, target=target))
        elif vs.kind in RUNNABLE and why in _FIX:
            out.append(_case(c, "will_run_other", why, target=target))
        else:
            out.append(_case(c, "cannot_compare", why))
    return out


def plan_compare(plan, vs, store, *, product, host_kinds, relax=(), purposes=MAIN_PURPOSES, expect_artifact=""):
    """The compare document for one product's plan. Never raises on a store outage: it says so per case."""
    leg = plan.get("leg") or {}
    subject = ((plan.get("request") or {}).get("subject") or {}).get("artifact_id") or ""
    arch = leg.get("arch") or ""
    cases = plan["cases"]
    doc = {
        "schema": SCHEMA,
        "vs": {"text": vs.text, "kind": vs.kind, "value": vs.value},
        "product": product,
        "subject_artifact": subject,
        "status": "ok",
        "refusal": None,
        "target": None,
        "cap": {"verdict": VERDICT_CAP, "purpose_not_baseline": True},
        "cases": [],
        "other_requests": [],
    }

    def refuse(code, message):
        doc.update(status="refused", refusal={"code": code, "message": message})
        doc["cases"] = [_case(c, "cannot_compare", code) for c in cases]
        return _finish(doc, cases)

    if not subject:
        return refuse("no_subject", "the plan names no artifact, so there is nothing to compare")
    if store.name == "none":
        return refuse("no_store", "no result store is configured, so nothing can be compared")
    try:
        cands = store.resolve_vs(vs.kind, vs.value, product, purposes, canon_arch(arch))
        targets = usable_targets(vs, cands, product=product, arch=arch, subject_artifact=subject, vs_kinds=host_kinds)
        if expect_artifact and targets[0]["artifact_id"] != expect_artifact:
            raise Refused("target_moved", "the comparison target changed since the preview")
        ids = sorted({c["benchmark_id"] for c in cases if c.get("benchmark_id")})
        best = None
        for target in targets if vs.kind == "main" else targets[:1]:
            rows = store.target_results(vs.kind, vs.value, target, ids)
            events = store.invalidations([r["run_id"] for r in rows])
            out = _evaluate(cases, rows, events, vs, target, leg, relax, purposes)
            best = best or (target, out)
            if any(k["state"] == "ready" for k in out):
                best = (target, out)
                break
    except Refused as r:
        return refuse(r.code, r.message)
    except Exception as e:  # noqa: BLE001 - a store failure must read as "unavailable", never as "nothing to compare"
        if type(e).__name__ != "StoreError":
            raise
        doc["status"] = "unavailable"
        doc["refusal"] = {"code": "store_unavailable", "message": "the result store did not answer"}
        doc["cases"] = [_case(c, "cannot_compare", "store_unavailable") for c in cases]
        return _finish(doc, cases)
    target, doc["cases"] = best
    doc["target"] = {
        "artifact_id": target["artifact_id"],
        "source": target.get("source") or vs.kind,
        "ref_key": target.get("ref_key") or "",
        "arch": target.get("arch") or arch,
        "role": "reference_target" if vs.kind == "ref" else "comparison_target",
        "baseline_eligible": False if vs.kind == "ref" else None,
    }
    names = [k["name"] for k in doc["cases"] if k["state"] == "will_run_other"]
    if names:
        doc["other_requests"] = [
            {
                "purpose": plan.get("purpose") or "pr",
                "subject": {"artifact_id": target["artifact_id"]},
                "selection": {
                    "cases": names,
                    "type": (plan.get("request") or {}).get("selection", {}).get("type", "perf"),
                },
                "reuse": {"mode": "never", "override": True},
                **(
                    {"environment": plan["request"]["environment"]}
                    if (plan.get("request") or {}).get("environment")
                    else {}
                ),
            }
        ]
    return _finish(doc, cases)


def _case(c, state, why, other=None, target=None):
    out = {
        "name": c["name"],
        "benchmark_id": c.get("benchmark_id"),
        "state": state,
        "why": why,
        "subject": {"action": c["action"], "reason": c.get("reason") or "", "reused_from": c.get("reused_from")},
        "other": {"artifact_id": "", "run_id": "", "validity": "", "purpose": ""},
    }
    if other:
        out["other"] = {
            "artifact_id": (target or {}).get("artifact_id", ""),
            "run_id": "" if other.get("reference") else other["run_id"],
            "validity": other["validity"],
            "purpose": other.get("purpose") or "",
        }
    elif target:
        out["other"]["artifact_id"] = target["artifact_id"]
    return out


def _finish(doc, cases):
    by = {c["name"]: c for c in cases}
    count = lambda s: sum(1 for k in doc["cases"] if k["state"] == s)  # noqa: E731
    sub_h = [_hours(by[k["name"]]) for k in doc["cases"] if k["subject"]["action"] == "run" and k["name"] in by]
    oth_h = [_hours(by[k["name"]]) for k in doc["cases"] if k["state"] == "will_run_other" and k["name"] in by]
    known = None not in sub_h + oth_h
    sub, oth = sum(h or 0.0 for h in sub_h), sum(h or 0.0 for h in oth_h)
    doc["counts"] = {
        "cases": len(doc["cases"]),
        "ready": count("ready"),
        "will_run_other": count("will_run_other"),
        "cannot_compare": count("cannot_compare"),
    }
    doc["estimate"] = {"subject_device_hours": round(sub, 2), "other_device_hours": round(oth, 2), "known": known}
    doc["digest"] = hashlib.sha256(
        json.dumps(
            {k: doc[k] for k in ("vs", "status", "refusal", "target", "cases", "counts")}, sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    return doc


def request_props(doc):
    """The bench.* keys a ci_requests row carries so a reader can open the compare view with both sides."""
    t = doc.get("target") or {}
    return {
        "bench.vs": doc["vs"]["text"],
        "bench.vs.kind": doc["vs"]["kind"],
        "bench.vs.status": doc["status"],
        "bench.vs.artifact": t.get("artifact_id", ""),
        "bench.vs.role": t.get("role", ""),
        "bench.vs.subject_artifact": doc.get("subject_artifact", ""),
        "bench.vs.digest": doc["digest"],
    }


def compare_links(doc, subject_runs=None):
    """Perf Lab compare layer (X) addresses: /perf-lab/drill/compare/<run a>/<run b>/<benchmark id>.

    Only cases that have both sides as runs get one; `subject_runs` maps case name -> the run id of the subject
    (known once it ran, or from `reused_from`). Never a value, only ids.
    """
    out = []
    for k in doc["cases"]:
        a = (subject_runs or {}).get(k["name"]) or k["subject"].get("reused_from") or ""
        b = k["other"]["run_id"]
        if k["state"] == "ready" and a and b and k.get("benchmark_id"):
            out.append({"case": k["name"], "a": a, "b": b, "benchmark_id": k["benchmark_id"]})
    return out

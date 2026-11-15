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

"""Writers of the bench bundle kinds (offline.py checks them): `bench` Results into benchmarks,
benchmark_runs, environment_fingerprints, run_placements and the run's verdict; and the CI-only
`bench-definitions`, `bench-qualification` and `bench-invalidation` records.

Trust is the channel's, never the bundle's. A claim a lower-trust channel cannot back (purpose,
validity, clean.exclusive, environment labels) is stored as a `claim.*` prop beside the
downgraded value. Every repeat carries a content hash, so a retry skips what landed and a
different repeat under a known key is refused.
"""

import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import schema
from .bench_tables import (
    EnvironmentFingerprints,
    EnvironmentQualifications,
    Environments,
    NodeClasses,
    ResultInvalidations,
    RunPlacements,
)
from .identity import BenchmarkId, DerivedId
from .offline import (
    BundleError,
    bench_result,
    content_digest,
    _ts,
)
from .writer import insert_artifact_result, upload_row_props

LARGE_INVALIDATION = 50
LOW_TRUST_PURPOSES = {"lab": ("on-demand", "investigate"), "external": ("external",)}
DOWNGRADED_PURPOSE = {"lab": "on-demand", "external": "external"}
CLEAN_CLAIMS = ("enforced", "observed")


def _s(bag) -> dict:
    return {str(k): str(v) for k, v in (bag or {}).items() if v != ""}


def _utc(value):
    return _ts(value).astimezone(timezone.utc) if value else None


def placement_id(nodes, devices, links) -> str:
    return DerivedId.hash(
        "placement",
        ",".join(sorted(nodes)),
        ",".join(sorted(f"{d[0]}/{d[1]}" for d in devices)),
        ",".join(sorted(links)),
    )


def downgrade(trust: str, run_props: dict, clean: dict) -> tuple:
    """(run_props, clean) as stored for `trust`; the originals move to claim.* when changed."""
    run_props, clean = dict(run_props), dict(clean)
    if trust == "ci":
        return run_props, clean
    allowed = LOW_TRUST_PURPOSES[trust]
    purpose = run_props.get("run.purpose")
    if purpose and purpose not in allowed:
        run_props["claim.run.purpose"] = purpose
        run_props["run.purpose"] = DOWNGRADED_PURPOSE[trust]
    if run_props.get("run.validity") not in (None, "", "unknown"):
        run_props["claim.run.validity"] = run_props["run.validity"]
        run_props["run.validity"] = "unknown"
    if clean.get("exclusive") in CLEAN_CLAIMS:
        clean["claim.exclusive"] = clean["exclusive"]
        clean["exclusive"] = "unknown"
    return run_props, clean


def repeat_hash(parts: dict) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).hexdigest()


def build(root: Path, trust: str) -> dict:
    """Everything the bundle would write, computed before the first insert."""
    cases, repeats, envs, places = {}, [], {}, {}
    for path in sorted((root / "results").glob("*.json")):
        st = bench_result(path)["status"]
        c = st["case"]
        component, name = c["component"].strip(), c["name"].strip()
        tags, disc = list(c.get("tags") or []), _s(c.get("disc"))
        keys = list(c.get("disc_keys") or sorted(disc))
        bid = BenchmarkId.derive(component, name, tags, disc, keys)
        if not bid:
            raise BundleError(f"{path.name}: the benchmark identity is not derivable")
        if c.get("id_mode") == "ingest" and c.get("benchmark_id") != bid:
            raise BundleError(
                f"{path.name}: benchmark_id {c.get('benchmark_id')} differs from the ingest's {bid}"
            )
        props = {**_s(c.get("props")), **disc}
        if c.get("benchmark_id") and c["benchmark_id"] != bid:
            props["engine_id"] = c["benchmark_id"]
        tag_set = sorted({n for n in (DerivedId.norm(t) for t in tags) if n})
        cases[bid] = {
            "benchmark_id": bid,
            "component": component,
            "name": name,
            "tags": tag_set,
            "props": props,
        }
        env = st["environment"]
        env_id = env["env_id"]
        labels = _s(env.get("labels")) if trust == "ci" else {"env.label.origin": trust}
        envs.setdefault(env_id, {
            "env_id": env_id, "fingerprint_version": int(env.get("fingerprint_version", 1)), "labels": labels,
            "fingerprint": _s(env.get("fingerprint")), "members": [], "devices": "",
            "props": {"run.trust": trust},
        })  # fmt: skip
        run_props = _s(st.get("run_props"))
        backend = str(
            c.get("backend") or disc.get("backend") or props.get("backend") or ""
        )
        for r in st["repeats"]:
            n = r["run.repeat"]
            clean = _s(r.get("clean"))
            stored_props, stored_clean = downgrade(trust, run_props, clean)
            samples = {
                k: [float(x) for x in (v if isinstance(v, list) else [v])]
                for k, v in (r.get("samples") or {}).items()
            }
            pl = r.get("placement") or {}
            nodes = sorted({str(x) for x in pl.get("nodes") or []})
            devices = sorted({
                (str(d.get("node") or ""), str(d.get("serial") or ""), str(d.get("pci") or ""),
                 str(d.get("firmware") or ""), str(d.get("link") or ""))
                for d in pl.get("devices") or []
            })  # fmt: skip
            links = sorted({str(x) for x in pl.get("links") or []})
            repeats.append({
                "bid": bid, "component": component, "backend": backend, "file": path.name, "repeat": n,
                "env_id": env_id, "status": str(r.get("status", "ok")), "samples": samples,
                "iterations": int(r.get("iterations") or 0), "props": {**stored_props, **_s(r.get("props"))},
                "started_at": _utc(r.get("started_at")),
                "sha": repeat_hash({
                    "bid": bid, "repeat": n, "env": env_id, "status": r.get("status", "ok"), "samples": samples,
                    "iterations": r.get("iterations", 0), "run_props": run_props, "props": r.get("props"),
                    "placement": [nodes, devices, links], "clean": clean,
                }),
            })  # fmt: skip
            slot = places.setdefault(n, {"nodes": set(), "devices": set(), "links": set(), "env": env_id,
                                         "cleans": [], "started": None, "merged": False})  # fmt: skip
            slot["merged"] |= bool(slot["cleans"]) and (
                set(nodes) != slot["nodes"] or env_id != slot["env"]
            )
            slot["nodes"] |= set(nodes)
            slot["devices"] |= set(devices)
            slot["links"] |= set(links)
            slot["cleans"].append(stored_clean)
            if r.get("started_at") and (
                slot["started"] is None or _utc(r["started_at"]) < slot["started"]
            ):
                slot["started"] = _utc(r["started_at"])
    return {"cases": cases, "repeats": repeats, "envs": envs, "places": places}


def recorded(client, db: str, rid: str) -> tuple:
    """(`(benchmark_id, repeat)` -> row hash of the run's rows, repeats with a placement)."""
    rows = client.query(
        "SELECT toString(benchmark_id), props['run.repeat'], props['run.row_sha256'] "
        f"FROM {db}.benchmark_runs WHERE run_id = {{rid:UUID}}",
        parameters={"rid": rid},
    ).result_rows
    placed = client.query(
        f"SELECT DISTINCT repeat FROM {db}.run_placements WHERE run_id = {{rid:UUID}}",
        parameters={"rid": rid},
    ).result_rows
    return {(r[0], int(r[1] or 0)): r[2] for r in rows}, {int(r[0]) for r in placed}


def known(client, db: str, table: str, column: str, ids: list) -> set:
    if not ids:
        return set()
    rows = client.query(
        f"SELECT DISTINCT {column} FROM {db}.{table} WHERE {column} IN {{ids:Array(UUID)}}",
        parameters={"ids": ids},
    ).result_rows
    return {str(r[0]) for r in rows}


def ingest(client, db, root, meta, aid, rid, props, report, dry_run) -> dict:
    """Record a bench bundle; `report` carries the run, artifact and trust the caller derived."""
    trust = report.get("trust") or "external"
    report["trust"] = trust
    plan = build(root, trust)
    held, placed = recorded(client, db, rid)
    fresh, same = [], 0
    for r in plan["repeats"]:
        have = held.get((r["bid"], r["repeat"]))
        if have is None:
            fresh.append(r)
        elif have == r["sha"]:
            same += 1
        else:
            raise BundleError(
                f"repeat {r['repeat']} of benchmark {r['bid']} is already recorded under run {rid} with different content"
            )
    have_envs = known(
        client, db, "environment_fingerprints", "env_id", list(plan["envs"])
    )
    have_cases = known(client, db, "benchmarks", "benchmark_id", list(plan["cases"]))
    new_envs = [e for e in plan["envs"] if e not in have_envs]
    new_cases = [b for b in plan["cases"] if b not in have_cases]
    new_places = sorted(n for n in plan["places"] if n not in placed)
    report.update(cases=len(plan["cases"]), repeats=len(plan["repeats"]), new_repeats=len(fresh),
                  existing_repeats=same, new_benchmarks=len(new_cases), new_environments=len(new_envs))  # fmt: skip
    if trust != "ci":
        report["downgrades"] = (
            "purpose, validity, clean.exclusive and environment labels are stored as claims"
        )
    if dry_run:
        return {
            **report,
            "status": "would-ingest" if fresh or new_places else "duplicate",
        }
    content = content_digest(root)
    base = {"run.trust": trust, "run.content_sha256": content, "run.bundle_sha256": report["bundle_sha256"],
            **upload_row_props(props)}  # fmt: skip
    if props.get("uploader"):
        base["uploader"] = str(props["uploader"])
    EnvironmentFingerprints.insert(client, [plan["envs"][e] for e in new_envs], db=db)
    schema.Benchmarks.insert(client, [plan["cases"][b] for b in new_cases], db=db)
    rows = []
    for n in new_places:
        p = plan["places"][n]
        devs = sorted(p["devices"])
        nodes = sorted(p["nodes"])
        rows.append({
            "run_id": rid, "repeat": n, "started_at": p["started"], "env_id": p["env"],
            "placement_id": placement_id(nodes, devs, p["links"]), "nodes": nodes, "devices": devs,
            "links": sorted(p["links"]),
            "clean": p["cleans"][0] if all(c == p["cleans"][0] for c in p["cleans"]) else {"exclusive": "unknown"},
            "props": {**base, "placement.source": "probe" if trust == "ci" else "self_reported",
                      **({"placement.merged": "1"} if p["merged"] else {})},
        })  # fmt: skip
    RunPlacements.insert(client, rows, db=db)
    runs = [
        {"run_id": rid, "benchmark_id": r["bid"], "component": r["component"], "backend": r["backend"],
         "measurements": r["samples"], "iterations": r["iterations"],
         "props": {**r["props"], **base, "report_kind": "bench", "source_file": r["file"], "run.repeat": str(r["repeat"]),
                   "run.repeat_status": r["status"], "env.id": r["env_id"], "run.row_sha256": r["sha"]}}
        for r in fresh if r["samples"]
    ]  # fmt: skip
    schema.BenchmarkRuns.insert(client, runs, db=db)
    report["benchmark_runs"] = len(runs)
    bad = {
        c
        for c in plan["cases"]
        if not any(r["bid"] == c and r["status"] == "ok" for r in plan["repeats"])
    }
    state = "failed" if bad else "passed"
    prior = client.query(
        f"SELECT count() FROM {db}.artifact_results WHERE artifact_id = {{a:UUID}} AND run_id = {{r:UUID}} "
        "AND test_type = {t:String} AND state != 'running'",
        parameters={"a": aid, "r": rid, "t": meta["test_type"]},
    ).result_rows[0][0]
    if not (fresh or new_places) and prior:
        return {**report, "status": "duplicate"}
    insert_artifact_result(
        client, db, artifact_id=aid, run_id=rid, test_type=meta["test_type"], state=state, arch=meta["arch"],
        result_kind="performance", supersede=bool(prior),
        props={**base, "source": "bundle", "bundle_kind": "bench", "bundle_sha256": report["bundle_sha256"],
               "content_sha256": content, "cases": str(len(plan["cases"])), "repeats": str(len(plan["repeats"])),
               "run_url": meta.get("run_url") or props.get("bundle_url", ""), **({"failed_cases": str(len(bad))} if bad else {})},
    )  # fmt: skip
    return {**report, "status": "ingested", "content_sha256": content}


def require_channel(kind: str, trust: str, allowed: tuple) -> None:
    if trust not in allowed:
        raise BundleError(
            f"a {kind} bundle needs a channel of trust {' or '.join(allowed)}, not {trust or 'none'}"
        )


def records(root: Path, key: str) -> list:
    (path,) = sorted((root / "results").glob("*.json"))
    return json.loads(path.read_text())[key]


def ingest_admin(client, db, root, meta, args, trust, uploader, sha, dry_run) -> dict:
    kind = meta["kind"]
    out = {"kind": kind, "bundle_sha256": sha, "trust": trust}
    if kind == "bench-definitions":
        require_channel(kind, trust, ("ci",))
        return {**out, **definitions(client, db, root, dry_run)}
    if kind == "bench-qualification":
        require_channel(kind, trust, ("ci",))
        return {**out, **qualifications(client, db, root, uploader, sha, dry_run)}
    if kind == "bench-placement":
        require_channel(kind, trust, ("ci",))
        return {**out, **placement_updates(client, db, root, sha, dry_run)}
    require_channel(kind, trust, ("ci", "lab"))
    return {
        **out,
        **invalidations(
            client,
            db,
            root,
            uploader,
            sha,
            dry_run,
            getattr(args, "confirm_runs", None),
        ),
    }


def definitions(client, db, root, dry_run) -> dict:
    src = (
        json.loads(next((root / "results").glob("*.json")).read_text()).get("source")
        or {}
    )
    origin = {f"source.{k}": str(v) for k, v in src.items()}
    new = {"Environment": [], "NodeClass": []}
    for r in records(root, "records"):
        table = "environments" if r["kind"] == "Environment" else "node_classes"
        have = client.query(
            f"SELECT count() FROM {db}.{table} WHERE name = {{n:String}} AND spec_digest = {{d:String}}",
            parameters={"n": r["name"], "d": r["spec_digest"]},
        ).result_rows[0][0]
        if have:
            continue
        spec = r.get("spec") or {}
        if r["kind"] == "Environment":
            host = spec.get("host", "")
            new["Environment"].append({
                "name": r["name"], "spec_digest": r["spec_digest"], "labels": _s(spec.get("labels")),
                "devices": spec.get("devices") if isinstance(spec.get("devices"), str) else json.dumps(spec.get("devices") or "", sort_keys=True) if spec.get("devices") else "",
                "host": host if isinstance(host, str) else json.dumps(host, sort_keys=True),
                "policy": _s(spec.get("policy")), "props": origin,
            })  # fmt: skip
        else:
            new["NodeClass"].append({
                "name": r["name"], "spec_digest": r["spec_digest"], "host_profile": _s(spec.get("host_profile")),
                "qualification": _s(spec.get("qualification")), "props": origin,
            })  # fmt: skip
    n = sum(map(len, new.values()))
    if dry_run:
        return {"status": "would-ingest" if n else "duplicate", "new_records": n}
    Environments.insert(client, new["Environment"], db=db)
    NodeClasses.insert(client, new["NodeClass"], db=db)
    return {"status": "ingested" if n else "duplicate", "new_records": n}


def cv_pct(per_repeat: list) -> float:
    if len(per_repeat) < 2:
        return 0.0
    mean = statistics.fmean(per_repeat)
    return statistics.stdev(per_repeat) / abs(mean) * 100 if mean else float("inf")


def only_metric(client, db: str, rid: str) -> str:
    """The evidence run's one metric, when the record names none."""
    found = {
        k
        for (keys,) in client.query(
            f"SELECT mapKeys(measurements) FROM {db}.benchmark_runs WHERE run_id = {{r:UUID}}",
            parameters={"r": rid},
        ).result_rows
        for k in keys
    }
    if len(found) != 1:
        raise BundleError(
            f"evidence run {rid} holds {sorted(found) or 'no metric'}: name the canary metric"
        )
    return next(iter(found))


def evidence(client, db: str, rec: dict) -> tuple:
    """(cv_pct, repeats) recomputed from the evidence run's stored rows; refuses a run that is not CI's,
    did not run on the node, or does not hold the metric."""
    rid = rec["evidence_run_id"]
    metric = rec.get("metric") or only_metric(client, db, rid)
    rows = client.query(
        "SELECT toUInt16OrZero(props['run.repeat']), measurements[{m:String}], props['run.trust'] "
        f"FROM {db}.benchmark_runs WHERE run_id = {{r:UUID}} AND has(mapKeys(measurements), {{m:String}})",
        parameters={"r": rid, "m": metric},
    ).result_rows
    if not rows:
        raise BundleError(f"evidence run {rid} holds no {metric!r} rows")
    if {r[2] for r in rows} != {"ci"}:
        raise BundleError(f"evidence run {rid} is not a CI-trusted run")
    on_node = client.query(
        f"SELECT count() FROM {db}.run_placements WHERE run_id = {{r:UUID}} AND has(nodes, {{n:String}})",
        parameters={"r": rid, "n": rec["node"]},
    ).result_rows[0][0]
    if not on_node:
        raise BundleError(f"evidence run {rid} was not placed on {rec['node']}")
    per = {}
    for repeat, samples, _ in rows:
        per.setdefault(repeat, []).extend(samples)
    return cv_pct([statistics.median(v) for v in per.values()]), len(per)


def qualifications(client, db, root, uploader, sha, dry_run) -> dict:
    rows = []
    for rec in records(root, "records"):
        have = client.query(
            f"SELECT count() FROM {db}.environment_qualifications WHERE node = {{n:String}} "
            "AND env_id = {e:UUID} AND qualified_at = parseDateTime64BestEffort({q:String}, 3, 'UTC')",
            parameters={"n": rec["node"], "e": rec["env_id"], "q": rec["qualified_at"]},
        ).result_rows[0][0]
        if have:
            continue
        cv, repeats = evidence(client, db, rec)
        limit = float(rec.get("limit_pct") or 0)
        if rec["outcome"] == "pass" and (not limit or cv > limit):
            raise BundleError(
                f"{rec['node']}: a pass needs limit_pct and the recomputed cv {cv:.2f}% within it"
            )
        rows.append({
            "node": rec["node"], "env_id": rec["env_id"], "probe": str(rec.get("probe") or ""),
            "outcome": rec["outcome"], "cv_pct": cv, "limit_pct": limit, "repeats": repeats,
            "checks": _s(rec.get("checks")), "evidence_run_id": rec["evidence_run_id"],
            "qualified_at": _utc(rec["qualified_at"]), "expires_at": _utc(rec["expires_at"]),
            "props": {"run.trust": "ci", "bundle_sha256": sha, "recorded_by": uploader,
                      "claimed_cv_pct": str(rec.get("cv_pct", "")), "metric": rec.get("metric") or ""},
        })  # fmt: skip
    if not dry_run:
        EnvironmentQualifications.insert(client, rows, db=db)
    return {
        "status": ("would-ingest" if dry_run else "ingested") if rows else "duplicate",
        "new_records": len(rows),
    }


def matched_runs(client, db: str, scope_kind: str, scope: dict) -> int:
    """Runs a scope covers now (the preview count stored with the event)."""
    if scope_kind in ("run", "run_benchmark"):
        extra = " AND benchmark_id = {b:UUID}" if scope_kind == "run_benchmark" else ""
        n = client.query(
            f"SELECT count() FROM {db}.benchmark_runs WHERE run_id = {{r:UUID}}{extra}",
            parameters={
                "r": scope["run_id"],
                "b": scope.get("benchmark_id") or scope["run_id"],
            },
        ).result_rows[0][0]
        if not n:
            raise BundleError(f"no recorded rows for run {scope['run_id']}")
        return 1
    test = {"node": "has(nodes, {v:String})", "device": "arrayExists(d -> d.serial = {v:String}, devices)",
            "placement": "lower(toString(placement_id)) = lower({v:String})"}[scope_kind]  # fmt: skip
    value = scope.get(
        {"node": "node", "device": "device_serial", "placement": "placement_id"}[
            scope_kind
        ]
    )
    window = ""
    params = {"v": value}
    if scope.get("window_from"):
        window += " AND (started_at IS NULL OR started_at >= parseDateTime64BestEffort({wf:String}, 3, 'UTC'))"
        params["wf"] = scope["window_from"]
    if scope.get("window_to"):
        window += " AND (started_at IS NULL OR started_at <= parseDateTime64BestEffort({wt:String}, 3, 'UTC'))"
        params["wt"] = scope["window_to"]
    return int(client.query(
        f"SELECT uniqExact(run_id) FROM {db}.run_placements WHERE {test}{window}", parameters=params
    ).result_rows[0][0])  # fmt: skip


def invalidations(client, db, root, uploader, sha, dry_run, confirm) -> dict:
    events, rows, counts = records(root, "events"), [], []
    base = datetime.now(timezone.utc)
    for i, e in enumerate(events):
        scope = e.get("scope") or {}
        count = matched_runs(client, db, e["scope_kind"], scope)
        counts.append(count)
        latest = client.query(
            f"SELECT argMax(state, (ts, audit_timestamp, audit_uuid)) FROM {db}.result_invalidations "
            "WHERE invalidation_id = {i:UUID}",
            parameters={"i": e["invalidation_id"]},
        ).result_rows
        if latest and latest[0][0] == e["state"]:
            continue
        if e["state"] != "restored" and count > LARGE_INVALIDATION and confirm != count:
            raise BundleError(
                f"event {i} covers {count} runs (more than {LARGE_INVALIDATION}): preview with --dry-run, "
                f"then confirm with --confirm-runs {count}"
            )
        rows.append({
            "ts": base + timedelta(milliseconds=i), "invalidation_id": e["invalidation_id"],
            "scope_kind": e["scope_kind"], "run_id": scope.get("run_id"), "benchmark_id": scope.get("benchmark_id"),
            "repeat": scope.get("repeat"), "node": scope.get("node", ""), "device_serial": scope.get("device_serial", ""),
            "placement_id": scope.get("placement_id", ""), "window_from": _utc(scope.get("window_from")),
            "window_to": _utc(scope.get("window_to")), "match": {}, "match_expr": "", "matched_runs": count,
            "state": e["state"], "reason_code": e["reason_code"], "reason": e.get("reason", ""),
            "actor": uploader, "evidence_url": e.get("evidence_url", ""), "props": {"bundle_sha256": sha},
        })  # fmt: skip
    if not dry_run:
        ResultInvalidations.insert(client, rows, db=db)
    return {"status": ("would-ingest" if dry_run else "ingested") if rows else "duplicate",
            "new_records": len(rows), "matched_runs": counts}  # fmt: skip


def placement_updates(client, db, root, sha, dry_run) -> dict:
    """New run_placements rows over the run's newest (the effective view reads the newest per run and repeat):
    the clean decision, or the nodes and devices a probe found, after the run. Only a CI channel may write them."""
    rows = []
    for u in records(root, "updates"):
        rid = u["run_id"]
        have = client.query(
            "SELECT repeat, toString(env_id), nodes, devices, links, clean, started_at "
            f"FROM {db}.v_run_placements_effective WHERE run_id = {{r:UUID}}",
            parameters={"r": rid},
        ).result_rows
        if not have:
            raise BundleError(f"run {rid} has no placement rows to update")
        wanted = set(u["repeats"]) if u.get("repeats") is not None else None
        missing = sorted(wanted - {r[0] for r in have}) if wanted is not None else []
        if missing:
            raise BundleError(f"run {rid} has no placement for repeat(s) {missing}")
        for repeat, env_id, nodes, devices, links, clean, started in have:
            if wanted is not None and repeat not in wanted:
                continue
            old = (
                sorted(nodes),
                sorted(tuple(d) for d in devices),
                sorted(links),
                dict(clean),
            )
            new_devices = sorted(
                {
                    tuple(
                        str(d.get(k) or "")
                        for k in ("node", "serial", "pci", "firmware", "link")
                    )
                    for d in u["devices"]
                }
                if "devices" in u
                else old[1]
            )
            new = (
                sorted(set(u["nodes"])) if "nodes" in u else old[0],
                new_devices,
                sorted(set(u["links"])) if "links" in u else old[2],
                {**old[3], **_s(u.get("clean"))},
            )
            if new == old:
                continue
            rows.append(
                {
                    "run_id": rid, "repeat": repeat, "started_at": started, "env_id": env_id,
                    "placement_id": placement_id(new[0], new[1], new[2]), "nodes": new[0], "devices": new[1],
                    "links": new[2], "clean": new[3],
                    "props": {"run.trust": "ci", "placement.source": "ci_update", "bundle_sha256": sha},
                }
            )  # fmt: skip
    if not dry_run:
        RunPlacements.insert(client, rows, db=db)
    return {
        "status": ("would-ingest" if dry_run else "ingested") if rows else "duplicate",
        "new_records": len(rows),
    }

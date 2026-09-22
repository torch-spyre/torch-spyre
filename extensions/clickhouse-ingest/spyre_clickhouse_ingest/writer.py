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

"""The schema-v2 write path: dedup guard plus the test_cases/test_case_runs insert.

Rows are built as dicts and ordered by the schema model, so a field cannot be assigned
to the wrong column and the column order lives in exactly one place.
"""

import sys

from . import schema
from .identity import (
    _norm,
    benchmark_id_for,
    canonical_arch,
    capability_id_for,
    tags_for_case,
    case_id_for,
)


def cases_already_ingested(
    client, db: str, run_id: str, component: str, source_file: str = ""
) -> bool:
    """Has THIS source file's rows for this run already landed?

    test_case_runs is a plain MergeTree with no dedup key, so a double ingest of one leg
    DOUBLES its counts -- and v2 dropped the stored counters precisely because they are
    derived from these rows. This check is what keeps that correct.

    Scoped by source file, not just run_id: a sharded run is MANY xml files under ONE
    run_id (the pipeline passes --xml-dir with every shard in a single invocation), so a
    run-level check lets the first shard block all the others. Measured on a real
    Spyre-Next run: 9 of 10 cases silently dropped across 7 shards.

    `props['source_file']` carries the discriminator. props is a Map outside every key, so
    recording it costs no sort-order change.
    """
    table = schema.TEST_CASE_RUNS.qualified(db)
    if source_file:
        rows = client.query(
            f"SELECT count() FROM {table} "
            "WHERE component = {component:String} AND run_id = {run_id:UUID} "
            "AND props['source_file'] = {sf:String}",
            parameters={"component": component, "run_id": run_id, "sf": source_file},
        ).result_rows
    else:
        # No discriminator given: fall back to the run-level check rather than skip
        # dedup entirely, so a caller that cannot name the file is still protected.
        rows = client.query(
            f"SELECT count() FROM {table} "
            "WHERE component = {component:String} AND run_id = {run_id:UUID}",
            parameters={"component": component, "run_id": run_id},
        ).result_rows
    return bool(rows and rows[0][0] > 0)


def insert_test_results(
    client, db: str, component: str, run_id: str, cases: list, source_file: str = ""
) -> int:
    """Write test_cases (identity) + test_case_runs (outcome) for one leg.

    Rows are built as dicts and ordered by schema_model, so a field cannot be assigned to
    the wrong column and the column order lives in exactly one place.

    Dropped from v2 deliberately: filename, suite_name, runner_run_id, and every stored
    counter -- all derivable, and a stored counter invites drift.
    """
    if not cases:
        return 0
    ident_rows, run_rows = {}, []

    skipped_unidentifiable = 0
    for c in cases:
        tags = tags_for_case(c)
        classname, name = c.get("classname", ""), c.get("name", "")
        tcid = case_id_for(component, classname, name, tags)
        if not tcid:
            # Refused identity: writing the row anyway would collide it with every
            # other unidentifiable case rather than merely orphaning it.
            skipped_unidentifiable += 1
            continue
        # Keyed by id: identical identity rows within a leg are one fact.
        ident_rows[tcid] = {
            "test_case_id": tcid,
            "component": component,
            "classname": classname,
            "name": name,
            "tags": tags,
        }
        run_rows.append(
            {
                "run_id": run_id,
                "test_case_id": tcid,
                "component": component,
                "status": c.get("status", ""),
                "duration_s": float(c.get("duration_s", 0) or 0),
                "fail_message": (c.get("fail_message") or "")[:8192],
                # source_file names the xml this row came from, so a sharded run dedups
                # per file instead of the first shard blocking the rest.
                # ran_in names the run that ACTUALLY EXECUTED this case. For a case this
                # run ran it is this run_id; a reuse copy carries the original
                # executor's (see copy_reused_cases). Every "how much did we execute"
                # query must filter props['ran_in'] = run_id -- without it, reuse copies
                # inflate the count. Queries asking "what does this run report for the
                # tier" want the unfiltered total, which is the point of writing the
                # copies at all.
                "props": (
                    {
                        "ran_in": run_id,
                        **({"source_file": source_file} if source_file else {}),
                    }
                ),
            }
        )
    # Cross-run dedup, not just in-leg: test_cases is a plain MergeTree, so re-inserting
    # a known identity appends a duplicate instead of collapsing it.
    schema.insert_identities(client, schema.TEST_CASES, ident_rows, db=db)
    schema.insert(client, schema.TEST_CASE_RUNS, run_rows, db=db)
    if skipped_unidentifiable:
        print(
            f"  [warn] v2: {skipped_unidentifiable} case(s) skipped -- identity "
            "not derivable",
            file=sys.stderr,
        )
    return len(run_rows)


def benchmarks_already_ingested(
    client,
    db: str,
    run_id: str,
    component: str,
    report_kind: str = "",
    source_file: str = "",
) -> bool:
    """benchmark_runs has no dedup key, so a double ingest doubles the samples behind a
    mean -- which still looks plausible.

    report_kind is part of the scope because one invocation ingests SEVERAL files under one
    run_id: run_id_for honours a threaded --run-id verbatim, so a kernel-report and a
    benchmark-report XML from the same leg share it. Keyed on (component, run_id) alone,
    the first file written makes every later kind look already-ingested and its disjoint
    benchmarks are dropped silently. Empty matches rows written before this key existed.

    source_file narrows further, as cases_already_ingested's sibling check does: a sharded
    run can pass several same-kind XMLs (e.g. two kernel-report shards) under one run_id,
    and report_kind alone would let the first shard block the rest.
    """
    q = (
        f"SELECT count() FROM {schema.BENCHMARK_RUNS.qualified(db)} "
        "WHERE component = {component:String} AND run_id = {run_id:UUID}"
    )
    params = {"component": component, "run_id": run_id}
    if report_kind:
        q += " AND props['report_kind'] = {kind:String}"
        params["kind"] = report_kind
    if source_file:
        q += " AND props['source_file'] = {sf:String}"
        params["sf"] = source_file
    rows = client.query(q, parameters=params).result_rows
    return bool(rows and rows[0][0] > 0)


def insert_benchmarks(
    client,
    db: str,
    component: str,
    run_id: str,
    benchmarks: list,
    report_kind: str = "",
    source_file: str = "",
) -> int:
    """Write benchmarks (identity) + benchmark_runs (measurements) for one leg.

    Each entry is a dict: name, tags, props, backend, measurements, iterations, and
    `disc` plus `disc_keys` for the identity discriminators (see benchmark_id_for). One row
    per (benchmark, backend): a row per metric would multiply every trend point by the
    metric count.

    report_kind and source_file are stamped into each row's props so
    benchmarks_already_ingested can scope its dedup per source kind and per file --
    several files, some sharing a kind, share one run_id in an invocation.
    """
    if not benchmarks:
        return 0
    ident_rows, facts = {}, {}
    skipped_unidentifiable = 0
    for b in benchmarks:
        name, tags = b.get("name", ""), b.get("tags") or []
        disc = b.get("disc") or {}
        bid = benchmark_id_for(component, name, tags, disc, b.get("disc_keys") or ())
        if not bid:
            # Writing a refused identity would collide it with every other
            # unidentifiable one.
            skipped_unidentifiable += 1
            continue
        backend = b.get("backend", "")
        # Keyed by id: two files reporting one benchmark merge, richer props winning, so
        # the merge cannot drop a field the other side set.
        prev = ident_rows.get(bid)
        props = {k: str(v) for k, v in (b.get("props") or {}).items() if v != ""}
        # Union, for the same reason props merge: the id hashes NORMALIZED tags, so two
        # entries differing only in case or order share a bid, and taking the last
        # entry's list wholesale would drop tags the other side carried.
        # Normalized through the SAME helper the hash uses: unioning raw spellings put both
        # 'GPU' and 'gpu' on one identity, so has(tags,'gpu') and has(tags,'GPU') disagreed
        # about a single canonical row.
        tag_set = {n for n in (_norm(t) for t in tags) if n}
        if prev:
            merged = dict(prev["props"])
            merged.update(props)
            props = merged
            tag_set |= set(prev["tags"])
        # First-write-wins, like tags/props merge from a fixed side rather than the last
        # entry seen: bid is a content hash of (component, name, tags, disc), so every
        # entry sharing a bid already agrees on name in substance -- this only picks
        # which literal spelling (case, whitespace) survives, deterministically.
        if prev:
            name = prev["name"]
        ident_rows[bid] = {
            "benchmark_id": bid,
            "component": component,
            "name": name,
            "tags": sorted(tag_set),
            "props": props,
        }
        fact = facts.setdefault(
            (bid, backend),
            {
                "run_id": run_id,
                "benchmark_id": bid,
                "component": component,
                "backend": backend,
                "measurements": {},
                "iterations": 0,
                "props": {
                    **({"report_kind": report_kind} if report_kind else {}),
                    **({"source_file": source_file} if source_file else {}),
                },
            },
        )
        # Extended, not overwritten: two entries sharing (benchmark, backend) and a
        # metric key are two samples of that metric, and the column exists to keep both.
        for k, v in (b.get("measurements") or {}).items():
            fact["measurements"].setdefault(k, []).extend(v)
        # Summed, not maxed: measurements are extended across entries sharing this fact
        # key, so each entry's iterations is its own distinct contribution, not a
        # restatement of the same count. A single scalar still can't reflect per-metric
        # sample counts that diverge (recoverable as length(measurements[k]) whenever the
        # producer sends samples rather than a mean) -- this only fixes the multi-entry
        # undercount, not that ceiling.
        fact["iterations"] += int(b.get("iterations") or 0)
        # Merged on every entry, as the identity props are: a sparser first entry must
        # not drop a field a later one set for the same (benchmark, backend).
        fact["props"].update({k: str(v) for k, v in (b.get("run_props") or {}).items()})
        # Re-applied last: report_kind/source_file are the dedup scope, so a producer
        # prop of the same name must not be able to redefine either and let a re-ingest
        # through.
        if report_kind:
            fact["props"]["report_kind"] = report_kind
        if source_file:
            fact["props"]["source_file"] = source_file
    # The DDL's CHECK refuses an empty map, so one unmeasured benchmark would fail the
    # whole insert; dropped with a warning instead of losing a long perf leg to a parse
    # gap.
    run_rows = [f for f in facts.values() if f["measurements"]]
    dropped = len(facts) - len(run_rows)
    kept = {f["benchmark_id"] for f in run_rows}
    schema.insert_identities(
        client,
        schema.BENCHMARKS,
        {k: v for k, v in ident_rows.items() if k in kept},
        db=db,
    )
    schema.insert(client, schema.BENCHMARK_RUNS, run_rows, db=db)
    if skipped_unidentifiable:
        print(
            f"  [warn] v2: {skipped_unidentifiable} benchmark(s) skipped -- identity "
            "not derivable",
            file=sys.stderr,
        )
    if dropped:
        print(
            f"  [warn] v2: {dropped} benchmark(s) skipped -- no measurements parsed",
            file=sys.stderr,
        )
    return len(run_rows)


def capabilities_already_ingested(
    client,
    db: str,
    run_id: str,
    component: str,
    test_type: str = "",
    shard: str = "",
) -> bool:
    """capability_runs has no dedup key, so a double ingest doubles every verdict behind a
    coverage percentage -- which still looks plausible.

    Scoped by test_type as well as run: one run can carry several analyses (model_ops and
    model_support are separate scans), and keyed on (component, run_id) alone the first one
    written makes every later one look already-ingested and its rows are dropped silently.
    The same failure benchmark report_kind exists to prevent.

    `shard` narrows it once more, as cases_already_ingested's source_file does. An analysis can
    be SHARDED across parallel processes that share one run_id: hf-adapters' weekly scan fans
    out over up to 25 shards per tier, each with its own sink and its own client. Scoped by run
    alone, the first shard to flush makes every other shard look already-ingested and their
    verdicts are dropped with no error. Empty matches rows written before this key existed.
    """
    q = (
        f"SELECT count() FROM {schema.CAPABILITY_RUNS.qualified(db)} "
        "WHERE component = {component:String} AND run_id = {run_id:UUID}"
    )
    params = {"component": component, "run_id": run_id}
    if test_type:
        q += " AND test_type = {tt:String}"
        params["tt"] = test_type
    if shard:
        q += " AND props['shard'] = {shard:String}"
        params["shard"] = shard
    rows = client.query(q, parameters=params).result_rows
    return bool(rows and rows[0][0] > 0)


def insert_capabilities(
    client,
    db: str,
    component: str,
    run_id: str,
    test_type: str,
    results: list,
    arch: str = "",
    disc_keys=(),
    shard: str = "",
) -> int:
    """Write capabilities (identity) + capability_runs (verdict) for one analysis.

    Each entry is a dict: subject, name, status, backend, optional fail_reason, tags, and
    `disc` (the per-producer discriminator hashed into the identity -- input shapes/dtypes for
    model_ops, nothing for model_support).

    `backend` is NOT part of the identity: one capability measured on cpu and on spyre is one
    capability with two verdicts, so the same capability_id carries both rows. That is what
    makes "supported on spyre but only via cpu fallback" a self-join rather than a stored flag.

    `shard` names this writer's slice of a sharded analysis and is stamped into every row's
    props, so capabilities_already_ingested can scope its dedup per shard rather than letting
    the first of N parallel writers block the rest.
    """
    if not results:
        return 0
    ident_rows, run_rows = {}, []
    skipped_unidentifiable = 0
    for r in results:
        subject, name = r.get("subject", ""), r.get("name", "")
        disc = r.get("disc") or {}
        cid = capability_id_for(component, test_type, subject, name, disc, disc_keys)
        if not cid:
            # Refused identity: writing it anyway collides this row with every other
            # unidentifiable one rather than merely orphaning it.
            skipped_unidentifiable += 1
            continue
        tags = sorted({t for t in (r.get("tags") or []) if t})
        # Keyed by id: identical identity rows within one analysis are one fact. The
        # discriminator is hashed INTO cid, so it is recorded here rather than re-derived.
        ident_rows[cid] = {
            "capability_id": cid,
            "component": component,
            "test_type": test_type,
            "subject": subject,
            "name": name,
            "tags": tags,
            "props": {k: str(v) for k, v in disc.items() if v not in (None, "")},
        }
        run_rows.append(
            {
                "run_id": run_id,
                "capability_id": cid,
                "component": component,
                "test_type": test_type,
                "arch": canonical_arch(arch),
                "status": r.get("status", ""),
                "backend": _norm(r.get("backend")),
                "fail_reason": _norm(r.get("fail_reason")),
                # shard is applied LAST: it is the dedup scope, so a producer prop of the
                # same name must not be able to redefine it and let a re-ingest through.
                "props": {
                    **{k: str(v) for k, v in (r.get("props") or {}).items()},
                    **({"shard": shard} if shard else {}),
                },
            }
        )
    # Cross-run dedup: capabilities is a plain MergeTree, so re-inserting a known identity
    # appends a duplicate instead of collapsing it.
    schema.insert_identities(client, schema.CAPABILITIES, ident_rows, db=db)
    schema.insert(client, schema.CAPABILITY_RUNS, run_rows, db=db)
    if skipped_unidentifiable:
        print(
            f"  [warn] v2: {skipped_unidentifiable} capability result(s) skipped -- "
            "identity not derivable",
            file=sys.stderr,
        )
    return len(run_rows)

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

"""The v2 write path: one writer class per table pair, all sharing `RunWriter`."""

import math
from collections import Counter
import sys

from . import schema
from .mark_retried import PRIOR_MESSAGE, PRIOR_MESSAGE_MAX, PRIOR_STATUS
from .identity import (
    ArtifactIdentity,
    BenchmarkId,
    CapabilityId,
    CaseId,
    DerivedId,
)


CAPABILITY_PREFIX = "capability."
# A declaration without all three names no capability; it is skipped, never guessed.
CAPABILITY_REQUIRED = ("test_type", "subject", "name")


def capability_declaration(case: dict) -> tuple:
    """(declaration, problem) from a case's `capability.*` JUnit properties.

    The contract: `test_type`, `subject`, `name` (required) and `backend` are scalars;
    `sig.<k>` is hashed into capability_id; `tag` repeats; `prop.<k>` lands in the verdict's
    props. Any other key is reported in `unknown` and dropped. A case declaring no
    `capability.*` property gives (None, ""); one missing a required key, giving a scalar
    two values, or naming an unregistered test_type gives (None, <problem>).
    """
    decl = {"sig": {}, "tags": [], "props": {}, "unknown": [], "backend": ""}
    scalars: dict = {}
    seen = False
    for pname, pvalue in case.get("properties", []) or []:
        if not pname.startswith(CAPABILITY_PREFIX):
            continue
        seen = True
        key, value = pname[len(CAPABILITY_PREFIX) :], str(pvalue).strip()
        field, _, sub = key.partition(".")
        if field in CAPABILITY_REQUIRED + ("backend",) and not sub:
            if scalars.setdefault(field, value) != value:
                return None, f"conflicting {CAPABILITY_PREFIX}{field}"
        elif field == "sig" and sub:
            decl["sig"][sub] = value
        elif field == "prop" and sub:
            decl["props"][sub] = value
        elif key == "tag":
            if value:
                decl["tags"].append(value)
        else:
            decl["unknown"].append(f"unknown key {pname}")
    if not seen:
        return None, ""
    missing = [k for k in CAPABILITY_REQUIRED if not scalars.get(k)]
    if missing:
        return None, "no " + "/".join(CAPABILITY_PREFIX + k for k in missing)
    # Checked here, not at insert: there one bad case rejects the file's whole artifact_results batch.
    if scalars["test_type"] not in schema.CAPABILITY_TYPE_VALUES:
        return None, (
            f"unregistered {CAPABILITY_PREFIX}test_type '{scalars['test_type']}' (registered: "
            f"{', '.join(sorted(schema.CAPABILITY_TYPE_VALUES))}; a new one needs "
            "schema.CAPABILITY_TYPE_VALUES and a migration of artifact_results.chk_test_type)"
        )
    decl.update(scalars)
    return decl, ""


class RunWriter:
    """Base for a writer over an (identity, fact) table pair."""

    # No default: every concrete writer sets both, and a base-class None default made
    # mypy treat every use below as possibly-None instead of catching a real omission.
    identity_table: type[schema.Table]
    fact_table: type[schema.Table]

    @classmethod
    def _seen(cls, client, db: str, run_id: str, component: str, scopes=()) -> bool:
        """True when the fact table holds rows for this run within `scopes`."""
        where = "component = {component:String} AND run_id = {run_id:UUID}"
        params = {"component": component, "run_id": run_id}
        for column, key, value in scopes:
            if value:
                where += f" AND {column} = {{{key}:String}}"
                params[key] = value
        return cls.fact_table.count_rows(client, db, where, params) > 0

    @classmethod
    def _flush(cls, client, db: str, ident_rows: dict, fact_rows: list) -> int:
        """Write the unknown identities and every fact row; returns the row count."""
        cls.identity_table.insert_identities(client, ident_rows, db=db)
        cls.fact_table.insert(client, fact_rows, db=db)
        return len(fact_rows)

    @staticmethod
    def _warn(count: int, message: str) -> None:
        """Report skipped rows on stderr, so a parse gap is visible, not fatal."""
        if count:
            print(f"  [warn] v2: {count} {message}", file=sys.stderr)


class TestResultWriter(RunWriter):
    """test_cases (identity) + test_case_runs (outcome) for one JUnit leg."""

    identity_table = schema.TestCases
    fact_table = schema.TestCaseRuns

    # A re-run attempt reuses the run_id and every file name, so "this file has rows" alone
    # would refuse its results; only rows from this attempt or a later one count as landed.
    _ATTEMPT = "toUInt32OrZero(props['run_attempt'])"
    _FILE = (
        "component = {component:String} AND run_id = {run_id:UUID}"
        " AND props['source_file'] = {sf:String}"
    )
    # The counters sum rows, so a row that restates another row of the run is dropped: an exact
    # copy, an older attempt of the same file, a reused copy of a case the run executed, or a
    # skip where another file of the run executed the case. Differing outcomes otherwise stay:
    # one test_case_id can be two tests (names differing only in case, hf-adapters' base and
    # _adapter configs). Migration 009 applies the same rules to rows already written.

    @classmethod
    def already_ingested(
        cls,
        client,
        db: str,
        run_id: str,
        component: str,
        source_file: str = "",
        attempt: int = 0,
    ) -> bool:
        """Have this source file's rows for this run, from this attempt or later, landed?"""
        if not attempt:
            return cls._seen(
                client,
                db,
                run_id,
                component,
                (("props['source_file']", "sf", source_file),),
            )
        return (
            cls.fact_table.count_rows(
                client,
                db,
                f"{cls._FILE} AND {cls._ATTEMPT} >= {{attempt:UInt32}}",
                {
                    "component": component,
                    "run_id": run_id,
                    "sf": source_file,
                    "attempt": attempt,
                },
            )
            > 0
        )

    @classmethod
    def drop_older_attempts(
        cls,
        client,
        db: str,
        run_id: str,
        component: str,
        source_file: str,
        attempt: int,
    ) -> None:
        """Delete this file's outcomes and capability verdicts from attempts before
        `attempt`, so a re-run replaces them."""
        if not (attempt and source_file):
            return
        where = f"{cls._FILE} AND {cls._ATTEMPT} < {{attempt:UInt32}}"
        params = {
            "component": component,
            "run_id": run_id,
            "sf": source_file,
            "attempt": attempt,
        }
        if cls.fact_table.count_rows(client, db, where, params):
            client.command(
                f"DELETE FROM {cls.fact_table.qualified(db)} WHERE {where}",
                parameters=params,
            )
            cls._rebuild_counters(client, db, run_id, component)
        # The file's capability verdicts are scoped by shard = source_file.
        verdicts = where.replace("props['source_file']", "props['shard']")
        capability_runs = CapabilityWriter.fact_table
        if capability_runs.count_rows(client, db, verdicts, params):
            client.command(
                f"DELETE FROM {capability_runs.qualified(db)} WHERE {verdicts}",
                parameters=params,
            )

    # run_case_counters_mv's columns; `recovered` exists only once migration 010 has run, and the
    # writer is installed from main at run time, ahead of the schema.
    _COUNTERS = {
        "total_tests": "count()",
        "passed": "countIf(status = 'passed')",
        "failed": "countIf(status = 'failed')",
        "errors": "countIf(status = 'error')",
        "skipped": "countIf(status = 'skipped')",
        "xfail": "countIf(status = 'xfail')",
        "xpass": "countIf(status = 'xpass')",
        "recovered": "countIf(status = 'passed' AND (props['result.prior_status'] IN "
        "('failed', 'error') OR toUInt32OrZero(props['result.reruns']) > 0))",
    }

    @classmethod
    def _rebuild_counters(cls, client, db: str, run_id: str, component: str) -> None:
        """Recount this run's run_case_counters from test_case_runs.

        The counters MV fires on INSERT only, so a DELETE leaves the old attempt's counts
        summed in; the SELECT mirrors run_case_counters_mv.
        """
        counters = f"{db}.run_case_counters" if db else "run_case_counters"
        params = {"component": component, "run_id": run_id}
        where = "component = {component:String} AND run_id = {run_id:UUID}"
        has_recovered = client.query(
            "SELECT count() FROM system.columns WHERE table = 'run_case_counters' "
            "AND name = 'recovered' AND database = "
            + ("{db:String}" if db else "currentDatabase()"),
            parameters={"db": db},
        ).result_rows
        recovered = bool(has_recovered and has_recovered[0][0])
        cols = [c for c in cls._COUNTERS if c != "recovered" or recovered]
        client.command(f"DELETE FROM {counters} WHERE {where}", parameters=params)
        client.command(
            f"INSERT INTO {counters} (run_id, component, {', '.join(cols)}) "
            f"SELECT run_id, component, {', '.join(cls._COUNTERS[c] for c in cols)} "
            f"FROM {cls.fact_table.qualified(db)} WHERE {where} "
            "GROUP BY run_id, component",
            parameters=params,
        )

    @classmethod
    def insert(
        cls,
        client,
        db: str,
        component: str,
        run_id: str,
        cases: list,
        source_file: str = "",
        attempt: int = 0,
    ) -> int:
        """Write one leg's cases; returns the number of outcome rows written."""
        if not cases:
            return 0
        ident_rows, run_rows = {}, []
        verdicts: dict[tuple, list] = {}
        problems: Counter = Counter()
        skipped = ignored = 0
        for c in cases:
            tags, run_tags, results = CaseId.split_tags(CaseId.tags_for(c))
            measured, recorded, unrouted = cls._recorded(c)
            cls._capability(c, run_tags, attempt, verdicts, problems)
            ignored += unrouted
            classname, name = c.get("classname", ""), c.get("name", "")
            tcid = CaseId.derive(component, classname, name, tags)
            if not tcid:
                # Refused identity: writing it anyway collides this case with every
                # other unidentifiable one, rather than merely orphaning it.
                skipped += 1
                continue
            # Keyed by id: identical identity rows within a leg are one fact.
            ident_row: schema.TestCaseRow = {
                "test_case_id": tcid,
                "component": component,
                "classname": classname,
                "name": name,
                "tags": tags,
            }
            ident_rows[tcid] = ident_row
            run_row: schema.TestCaseRunRow = {
                "run_id": run_id,
                "test_case_id": tcid,
                "component": component,
                "status": c.get("status", ""),
                "duration_s": float(c.get("duration_s", 0) or 0),
                "fail_message": (c.get("fail_message") or "")[:8192],
                # ran_in names the run that ACTUALLY EXECUTED this case (a reuse
                # carries the original executor's), the filter for "how much did we
                # execute"; source_file is the shard discriminator the dedup reads.
                "props": {
                    **results,
                    **recorded,
                    "ran_in": run_id,
                    **({"source_file": source_file} if source_file else {}),
                    **({"run_attempt": str(attempt)} if attempt else {}),
                },
                "tags": run_tags,
                "measurements": measured,
            }
            run_rows.append(run_row)
        run_rows, superseded = cls._one_per_case(
            client, db, component, run_id, run_rows
        )
        written = cls._flush(client, db, ident_rows, run_rows)
        # Checked before any batch is written, so a type's second batch is not refused.
        landed = {
            t
            for t in {k[0] for k in verdicts}
            if CapabilityWriter.already_ingested(
                client, db, run_id, component, t, shard=source_file
            )
        }
        for (test_type, arch, disc_keys), results in verdicts.items():
            if test_type not in landed:
                CapabilityWriter.insert(
                    client,
                    db,
                    component,
                    run_id,
                    test_type,
                    results,
                    arch=arch,
                    disc_keys=disc_keys,
                    shard=source_file,
                )
        if superseded:
            # After the insert, so a failed insert never loses the outcome it would replace.
            client.command(
                f"DELETE FROM {cls.fact_table.qualified(db)} "
                "WHERE component = {component:String} AND run_id = {run_id:UUID} "
                "AND audit_uuid IN {uuids:Array(UUID)}",
                parameters={
                    "component": component,
                    "run_id": run_id,
                    "uuids": superseded,
                },
            )
            cls._rebuild_counters(client, db, run_id, component)
        cls._warn(skipped, "case(s) skipped -- identity not derivable")
        for problem, n in sorted(problems.items()):
            cls._warn(n, f"capability declaration(s) with {problem}")
        cls._warn(
            ignored,
            "property value(s) ignored -- not tag, metric.*, result.* or capability.*",
        )
        return written

    # A case's outcome as a capability verdict; a skipped case gave none.
    _VERDICT = {
        "passed": "passed",
        "xpass": "passed",
        "failed": "failed",
        # pytest reports a test-body exception as <failure>; <error> is a broken setup/teardown.
        "error": "undetermined",
        "xfail": "not_implemented",
    }

    @classmethod
    def _capability(
        cls, case: dict, run_tags, attempt: int, verdicts: dict, problems: Counter
    ) -> None:
        """Add the case's `capability.*` verdict, if it declares a valid one, to `verdicts`.

        Keyed by (test_type, arch, sig keys): one CapabilityWriter batch hashes one key set.
        """
        decl, problem = capability_declaration(case)
        problems.update(decl["unknown"] if decl else [])
        if problem:
            problems[problem] += 1
        status = cls._VERDICT.get(case.get("status", ""))
        if not (decl and status):
            return
        props = {"test_name": case.get("name", ""), **decl["props"]}
        if attempt:
            props["run_attempt"] = str(attempt)
        arch = next(
            (
                t.split("__", 1)[1]
                for t in run_tags
                if CaseId.namespace(t) == "platform"
            ),
            "",
        )
        key = (decl["test_type"], arch, tuple(sorted(decl["sig"])))
        verdicts.setdefault(key, []).append(
            {
                "subject": decl["subject"],
                "name": decl["name"],
                "status": status,
                "backend": decl["backend"],
                "disc": decl["sig"],
                "tags": decl["tags"],
                "props": {k: v for k, v in props.items() if v},
            }
        )

    @staticmethod
    def _restates(run_id: str, a: dict, b: dict) -> bool:
        """Does row `a` make row `b` redundant in the run's counts?"""

        def copied(r):
            return r["ran_in"] not in ("", str(run_id))

        def ran(r):
            return not copied(r) and r["status"] != "skipped"

        if ran(a) and copied(b):
            return True
        if a["source_file"] == b["source_file"] and copied(a) == copied(b):
            if a["attempt"] != b["attempt"]:
                return a["attempt"] > b["attempt"]
            return all(a[k] == b[k] for k in ("status", "duration_s", "fail_message"))
        return ran(a) and not ran(b) and not copied(b)

    @classmethod
    def _one_per_case(cls, client, db: str, component: str, run_id: str, rows: list):
        """(rows to insert, audit_uuids they make redundant), per the rules above."""

        def facts(r):
            return {
                "status": r["status"],
                "duration_s": round(float(r["duration_s"]), 3),
                "fail_message": r["fail_message"],
                "ran_in": r["props"].get("ran_in", ""),
                "attempt": int(r["props"].get("run_attempt") or 0),
                "source_file": r["props"].get("source_file", ""),
            }

        batch: dict = {}
        for r in rows:
            same = batch.setdefault(r["test_case_id"], [])
            if not any(cls._restates(run_id, facts(o), facts(r)) for o in same):
                same.append(r)
        ids = list(batch)
        held: dict = {}
        for i in range(0, len(ids), schema.IDENTITY_LOOKUP_CHUNK):
            found = client.query(
                "SELECT test_case_id, audit_uuid, status, duration_s, fail_message, "
                "props['ran_in'], props['run_attempt'], props['source_file'], "
                f"props['{PRIOR_STATUS}'], props['{PRIOR_MESSAGE}'] "
                f"FROM {cls.fact_table.qualified(db)} "
                "WHERE component = {component:String} AND run_id = {run_id:UUID} "
                "AND test_case_id IN {ids:Array(UUID)}",
                parameters={
                    "component": component,
                    "run_id": run_id,
                    "ids": ids[i : i + schema.IDENTITY_LOOKUP_CHUNK],
                },
            ).result_rows
            for tcid, uuid, status, dur, msg, ran_in, attempt, sf, *prior in found:
                held.setdefault(str(tcid), []).append(
                    {
                        "audit_uuid": str(uuid),
                        "status": status,
                        "duration_s": round(float(dur), 3),
                        "fail_message": msg,
                        "ran_in": ran_in,
                        "attempt": int(attempt or 0),
                        "source_file": sf,
                        "prior": tuple(prior),
                    }
                )
        keep, superseded = [], set()
        for tcid, new in batch.items():
            rows_held = held.get(str(tcid), [])
            for r in new:
                f = facts(r)
                if any(cls._restates(run_id, h, f) for h in rows_held):
                    continue
                keep.append(r)
                replaced = [h for h in rows_held if cls._restates(run_id, f, h)]
                superseded |= {h["audit_uuid"] for h in replaced}
                cls._keep_prior(run_id, r, f, replaced)
        return keep, sorted(superseded)

    @staticmethod
    def _keep_prior(run_id: str, row: dict, f: dict, replaced: list) -> None:
        """Stamp on `row` the newest failure of an older attempt it replaces, else it is lost."""
        if row["props"].get(PRIOR_STATUS):
            return
        older = [
            h
            for h in replaced
            if h["attempt"] < f["attempt"]
            and h["source_file"] == f["source_file"]
            and h["ran_in"] in ("", str(run_id))
        ]
        for h in sorted(older, key=lambda h: h["attempt"], reverse=True):
            # A rerun attempt also re-ingests the reports it did not re-run: an unchanged outcome
            # is the same execution, so only a prior mark it already carried moves forward.
            rerun = any(h[k] != f[k] for k in ("status", "duration_s", "fail_message"))
            if rerun and h["status"] in ("failed", "error"):
                prior = (h["status"], h["fail_message"][:PRIOR_MESSAGE_MAX])
            elif h["prior"] and h["prior"][0]:
                prior = h["prior"]
            else:
                continue
            row["props"][PRIOR_STATUS], row["props"][PRIOR_MESSAGE] = prior
            return

    @staticmethod
    def _recorded(case: dict) -> tuple:
        """(measurements, result props, count ignored) from the case's JUnit properties.

        `metric.<name>` must be a finite number and lands in measurements under `<name>`;
        `result.<name>` lands in props verbatim. Tag properties are read by CaseId.tags_for,
        `capability.*` by _capability.
        """
        measured, recorded, ignored = {}, {}, 0
        for pname, pvalue in case.get("properties", []) or []:
            if pname == "tag" or "__" in pname or pname.startswith("capability."):
                continue
            if pname.startswith("metric.") and len(pname) > len("metric."):
                try:
                    v = float(pvalue)
                except (TypeError, ValueError):
                    v = math.nan
                if math.isfinite(v):
                    measured[pname[len("metric.") :]] = v
                    continue
            elif pname.startswith("result."):
                recorded[pname] = str(pvalue)
                continue
            ignored += 1
        return measured, recorded, ignored


class BenchmarkWriter(RunWriter):
    """benchmarks (identity) + benchmark_runs (measurements) for one leg."""

    identity_table = schema.Benchmarks
    fact_table = schema.BenchmarkRuns

    @classmethod
    def already_ingested(
        cls,
        client,
        db: str,
        run_id: str,
        component: str,
        report_kind: str = "",
        source_file: str = "",
    ) -> bool:
        """Have this run's rows for this report kind and source file already landed?"""
        return cls._seen(
            client,
            db,
            run_id,
            component,
            (
                ("props['report_kind']", "kind", report_kind),
                ("props['source_file']", "sf", source_file),
            ),
        )

    @classmethod
    def insert(
        cls,
        client,
        db: str,
        component: str,
        run_id: str,
        benchmarks: list,
        report_kind: str = "",
        source_file: str = "",
    ) -> int:
        """Write one leg's benchmarks, one row per (benchmark, backend)."""
        if not benchmarks:
            return 0
        ident_rows: dict[str, schema.BenchmarkRow] = {}
        facts: dict[tuple[str, str], schema.BenchmarkRunRow] = {}
        skipped = 0
        for b in benchmarks:
            name, tags = b.get("name", ""), b.get("tags") or []
            disc = b.get("disc") or {}
            bid = BenchmarkId.derive(
                component, name, tags, disc, b.get("disc_keys") or ()
            )
            if not bid:
                skipped += 1
                continue
            backend = b.get("backend", "")
            name, ident = cls._merge_identity(ident_rows, bid, component, name, tags, b)
            ident_rows[bid] = ident
            default_fact: schema.BenchmarkRunRow = {
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
            }
            fact = facts.setdefault((bid, backend), default_fact)
            cls._merge_fact(fact, b, report_kind, source_file)
        # The DDL's CHECK refuses an empty map, so one unmeasured benchmark would fail
        # the whole insert; dropped with a warning rather than losing a long perf leg.
        run_rows = [f for f in facts.values() if f["measurements"]]
        dropped = len(facts) - len(run_rows)
        kept = {f["benchmark_id"] for f in run_rows}
        written = cls._flush(
            client,
            db,
            {k: v for k, v in ident_rows.items() if k in kept},
            run_rows,
        )
        cls._warn(skipped, "benchmark(s) skipped -- identity not derivable")
        cls._warn(dropped, "benchmark(s) skipped -- no measurements parsed")
        return written

    @staticmethod
    def _merge_identity(ident_rows, bid, component, name, tags, entry) -> tuple:
        """Fold this entry's tags/props into the identity row for `bid`."""
        prev = ident_rows.get(bid)
        props = {k: str(v) for k, v in (entry.get("props") or {}).items() if v != ""}
        # Normalised through the SAME helper the hash uses: one spelling per tag.
        tag_set = {n for n in (DerivedId.norm(t) for t in tags) if n}
        if prev:
            merged = dict(prev["props"])
            merged.update(props)
            props = merged
            tag_set |= set(prev["tags"])
            name = prev["name"]
        row: schema.BenchmarkRow = {
            "benchmark_id": bid,
            "component": component,
            "name": name,
            "tags": sorted(tag_set),
            "props": props,
        }
        return name, row

    @staticmethod
    def _merge_fact(
        fact: schema.BenchmarkRunRow, entry, report_kind: str, source_file: str
    ) -> None:
        """Extend samples, sum iterations, merge props into one fact row."""
        for k, v in (entry.get("measurements") or {}).items():
            fact["measurements"].setdefault(k, []).extend(v)
        fact["iterations"] += int(entry.get("iterations") or 0)
        fact["props"].update(
            {k: str(v) for k, v in (entry.get("run_props") or {}).items()}
        )
        if report_kind:
            fact["props"]["report_kind"] = report_kind
        if source_file:
            fact["props"]["source_file"] = source_file


class CapabilityWriter(RunWriter):
    """capabilities (identity) + capability_runs (verdict) for one analysis."""

    identity_table = schema.Capabilities
    fact_table = schema.CapabilityRuns

    @classmethod
    def already_ingested(
        cls,
        client,
        db: str,
        run_id: str,
        component: str,
        test_type: str = "",
        shard: str = "",
    ) -> bool:
        """Have this run's verdicts for this analysis and shard already landed?"""
        return cls._seen(
            client,
            db,
            run_id,
            component,
            (("test_type", "tt", test_type), ("props['shard']", "shard", shard)),
        )

    @classmethod
    def insert(
        cls,
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
        """Write one analysis's verdicts; `backend` is a column, not part of the id."""
        if not results:
            return 0
        ident_rows, run_rows = {}, []
        skipped = 0
        for r in results:
            subject, name = r.get("subject", ""), r.get("name", "")
            disc = r.get("disc") or {}
            cid = CapabilityId.derive(
                component, test_type, subject, name, disc, disc_keys
            )
            if not cid:
                skipped += 1
                continue
            tags = sorted({t for t in (r.get("tags") or []) if t})
            # The discriminator is hashed INTO cid, so it is recorded, not re-derived.
            ident_row: schema.CapabilityRow = {
                "capability_id": cid,
                "component": component,
                "test_type": test_type,
                "subject": subject,
                "name": name,
                "tags": tags,
                "props": {k: str(v) for k, v in disc.items() if v not in (None, "")},
            }
            ident_rows[cid] = ident_row
            run_row: schema.CapabilityRunRow = {
                "run_id": run_id,
                "capability_id": cid,
                "component": component,
                "test_type": test_type,
                "arch": DerivedId.arch(arch),
                "status": r.get("status", ""),
                "backend": DerivedId.norm(r.get("backend")),
                "fail_reason": DerivedId.norm(r.get("fail_reason")),
                # shard is applied LAST: it is the dedup scope, so a producer prop
                # of the same name must not redefine it and let a re-ingest through.
                "props": {
                    **{k: str(v) for k, v in (r.get("props") or {}).items()},
                    **({"shard": shard} if shard else {}),
                },
            }
            run_rows.append(run_row)
        written = cls._flush(client, db, ident_rows, run_rows)
        cls._warn(skipped, "capability result(s) skipped -- identity not derivable")
        return written


class ArtifactWriter:
    """Artifacts, how to fetch them, the tags naming them, and the verdicts on them."""

    artifact_table = schema.Artifacts
    ref_table = schema.ArtifactRefs
    tag_table = schema.ArtifactTags
    result_table = schema.ArtifactResults

    # kind -> (method, ref_kind) of the address artifact_refs records.
    REF_SHAPE = {
        "image": ("container-pull", "pullspec"),
        "rpm": ("dnf", "glob"),
        "wheel": ("pip", "url"),
        "generic": ("download", "url"),
    }

    @classmethod
    def artifact_recorded(cls, client, db: str, artifact_id: str) -> bool:
        """Does `artifacts` hold this identity? (Plain MergeTree -- no dedup key.)"""
        return (
            cls.artifact_table.count_rows(
                client,
                db,
                "artifact_id = {artifact_id:UUID}",
                {"artifact_id": artifact_id},
            )
            > 0
        )

    @classmethod
    def tag_recorded(cls, client, db: str, tag: str, artifact_id: str) -> bool:
        """Does this tag already point at this artifact? (Also a plain MergeTree.)"""
        return (
            cls.tag_table.count_rows(
                client,
                db,
                "tag = {tag:String} AND artifact_id = {artifact_id:UUID}",
                {"tag": tag, "artifact_id": artifact_id},
            )
            > 0
        )

    @classmethod
    def result_recorded(
        cls,
        client,
        db: str,
        artifact_id: str,
        run_id: str,
        result_kind: str,
        test_type: str,
        attempt: int = 0,
    ) -> bool:
        """Has this verdict, from this attempt or later, landed? One run reports N tiers.

        A `running` row is a pre-dispatch SEED (Jenkins writes it before the leg starts), not
        a recorded verdict -- it must not block the leg's own terminal insert at the same key.
        """
        where, params = cls._verdict_key(artifact_id, run_id, result_kind, test_type)
        if attempt:
            where += f" AND {TestResultWriter._ATTEMPT} >= {{attempt:UInt32}}"
            params["attempt"] = attempt
        return cls.result_table.count_rows(client, db, where, params) > 0

    @staticmethod
    def _verdict_key(artifact_id, run_id, result_kind, test_type) -> tuple[str, dict]:
        return (
            "artifact_id = {artifact_id:UUID} AND run_id = {run_id:UUID} "
            "AND result_kind = {result_kind:String} "
            "AND test_type = {test_type:String} AND state != 'running'",
            {
                "artifact_id": artifact_id,
                "run_id": run_id,
                "result_kind": result_kind,
                "test_type": test_type,
            },
        )

    @classmethod
    def insert_artifact(
        cls,
        client,
        db: str,
        identity: ArtifactIdentity,
        *,
        origin: str = "built",
        sources=(),
        identity_deps=(),
        context_deps=(),
        props=None,
        tags=(),
    ) -> str:
        """Record `identity` once, its fetch address, and each (tag, family, props) in `tags`.

        Returns the artifact_id, or '' when the identity is incomplete. Every step is
        existence-checked, so a re-run adds nothing.
        """
        aid = identity.artifact_id
        if not aid:
            print(
                f"  [warn] v2: artifact skipped -- identity incomplete: {identity}",
                file=sys.stderr,
            )
            return ""
        if not cls.artifact_recorded(client, db, aid):
            row: schema.ArtifactRow = {
                "artifact_id": aid,
                "component": DerivedId.norm(identity.component),
                "arch": identity.arch,
                "kind": identity.kind,
                "artifact_name": identity.artifact_name,
                "origin": origin,
                "identity_deps": [d for d in identity_deps if d],
                "context_deps": [d for d in context_deps if d],
                # Tuple order (repo, git_ref, git_sha) -- what the covered-tier join reads.
                "sources": [tuple(s) for s in sources if any(s)],
                "props": cls._props(
                    {
                        "id12": identity.id12,
                        "artifact_name": identity.artifact_name,
                        "ref": identity.ref,
                        **dict(identity.inputs),
                        **(props or {}),
                    }
                ),
            }
            cls.artifact_table.insert(client, [row], db=db)
        method, ref_kind = cls.ref_shape(identity.kind)
        if identity.ref:
            # ReplacingMergeTree on (artifact_id, method, ref): a repeat insert collapses.
            ref_row: schema.ArtifactRefRow = {
                "artifact_id": aid,
                "method": method,
                "ref_kind": ref_kind,
                "index_uri": cls._index_uri(identity.ref),
                "ref": identity.ref,
                "content_digest": identity.content_digest,
                "props": cls._props({"id12": identity.id12}),
            }
            cls.ref_table.insert(client, [ref_row], db=db)
        for tag, family, tag_props in tags:
            cls.insert_tag(client, db, identity, tag, family, props=tag_props)
        return aid

    @classmethod
    def ref_shape(cls, kind: str) -> tuple:
        """(method, ref_kind) for an artifact kind; an unknown kind is a plain download."""
        return cls.REF_SHAPE.get(kind, cls.REF_SHAPE["generic"])

    @classmethod
    def insert_tag(
        cls,
        client,
        db: str,
        identity: ArtifactIdentity,
        tag: str,
        family: str,
        *,
        ref: str | None = None,
        props=None,
    ) -> bool:
        """Point `tag` at `identity`, once. `ref` is the address the tag names, which may be a
        moving one (`:amd64`) rather than the artifact's own; it defaults to the latter."""
        aid = identity.artifact_id
        if not (tag and aid):
            return False
        if cls.tag_recorded(client, db, tag, aid):
            return True
        ref = identity.ref if ref is None else ref
        method, ref_kind = cls.ref_shape(identity.kind)
        tag_row: schema.ArtifactTagRow = {
            "tag": tag,
            "tag_family": family,
            "artifact_id": aid,
            "refs": [(method, ref_kind, cls._index_uri(ref), ref)] if ref else [],
            "published_refs": [],
            "props": cls._props({"id12": identity.id12, **(props or {})}),
        }
        cls.tag_table.insert(client, [tag_row], db=db)
        return True

    @classmethod
    def insert_result(
        cls,
        client,
        db: str,
        *,
        artifact_id: str,
        run_id: str,
        test_type: str,
        state: str,
        arch: str,
        result_kind: str = "",
        duration_s: float = 0.0,
        props=None,
        attempt: int = 0,
    ) -> bool:
        """One verdict of one leg on one artifact; refuses a partial key, skips a repeat.

        Given an attempt, the verdict replaces any from an earlier attempt of the same run.
        """
        aid, rid, a = (
            DerivedId.norm(artifact_id),
            DerivedId.norm(run_id),
            DerivedId.arch(arch),
        )
        kind = result_kind or (
            "performance"
            if test_type == "perf"
            else "capability"
            if test_type in schema.CAPABILITY_TYPE_VALUES
            else "functional"
        )
        if not (aid and rid and a):
            print(
                f"  [warn] v2: artifact result skipped -- artifact_id={aid or '<blank>'} "
                f"run_id={rid or '<blank>'} arch={a or '<blank>'}",
                file=sys.stderr,
            )
            return False
        if cls.result_recorded(client, db, aid, rid, kind, test_type, attempt):
            return True
        if attempt:
            where, params = cls._verdict_key(aid, rid, kind, test_type)
            client.command(
                f"DELETE FROM {cls.result_table.qualified(db)} WHERE {where} "
                f"AND {TestResultWriter._ATTEMPT} < {{attempt:UInt32}}",
                parameters={**params, "attempt": attempt},
            )
        result_row: schema.ArtifactResultRow = {
            "artifact_id": aid,
            "run_id": rid,
            "result_kind": kind,
            "test_type": test_type,
            "state": state,
            # Where it RAN; kept apart from artifacts.arch by design.
            "arch": a,
            "duration_s": float(duration_s or 0.0),
            "props": cls._props(
                {**(props or {}), "run_attempt": str(attempt) if attempt else ""}
            ),
        }
        cls.result_table.insert(client, [result_row], db=db)
        return True

    @classmethod
    def insert_gha_result(
        cls,
        client,
        db: str,
        *,
        artifact_id: str,
        component: str,
        arch: str,
        run_id: str,
        test_type: str,
        state: str,
        result_kind: str = "",
        duration_s: float = 0.0,
        base_artifact_id: str = "",
        installed: str = "",
        repo: str = "",
        git_ref: str = "",
        git_sha: str = "",
        run_url: str = "",
        attempt: int = 0,
    ) -> bool:
        """Record the artifact a GHA leg ran and its verdict; refuses a partial id."""
        aid, rid = DerivedId.norm(artifact_id), DerivedId.norm(run_id)
        comp, a = DerivedId.norm(component), DerivedId.arch(arch)
        if not (aid and rid and comp and a):
            print(
                f"  [warn] v2: artifact result skipped -- "
                f"artifact_id={aid or '<blank>'} "
                f"run_id={rid or '<blank>'} "
                f"component={comp or '<blank>'} arch={a or '<blank>'}",
                file=sys.stderr,
            )
            return False

        base = DerivedId.norm(base_artifact_id)
        # aid == base means the leg ran the image UNCHANGED, so the artifact is the
        # one the orchestrator already recorded, and our own row would be a duplicate.
        if aid != base and not cls.artifact_recorded(client, db, aid):
            identity = ArtifactIdentity.from_gha(comp, base, installed, a)
            # Keyed on the caller's id: the record it came from is the authority.
            cls.artifact_table.insert(
                client,
                [
                    {
                        "artifact_id": aid,
                        "component": comp,
                        "arch": a,
                        "kind": "image",
                        # The hashed name, not a display string.
                        "artifact_name": base,
                        # chk_origin admits no 'gha'; the 'base=' dep is what marks it derived.
                        "origin": "built",
                        "identity_deps": [f"{schema.DEP_BASE_PREFIX}{base}"]
                        if base
                        else [],
                        "context_deps": [],
                        "sources": [(repo, git_ref, git_sha)]
                        if (repo or git_ref or git_sha)
                        else [],
                        "props": cls._props(
                            {
                                # The digest GhaArtifactId put in the id12 slot.
                                "id12": identity.id12,
                                **dict(identity.inputs),
                                "run_url": run_url,
                                "source": "gha",
                            }
                        ),
                    }
                ],
                db=db,
            )
        return cls.insert_result(
            client,
            db,
            artifact_id=aid,
            run_id=rid,
            test_type=test_type,
            state=state,
            arch=a,
            result_kind=result_kind,
            duration_s=duration_s,
            props={"run_url": run_url, "source": "gha"},
            attempt=attempt,
        )

    @staticmethod
    def _props(values: dict) -> dict:
        """String-valued props with the empty ones dropped."""
        return {k: str(v) for k, v in values.items() if v not in ("", None)}

    @staticmethod
    def _index_uri(ref: str) -> str:
        """The registry/index host a ref names, '' when it names none."""
        head = (ref or "").split("://", 1)[-1]
        return head.split("/", 1)[0] if "/" in head else ""


# Function API, kept so installed consumers import one definition, not a copy.
cases_already_ingested = TestResultWriter.already_ingested
drop_older_case_attempts = TestResultWriter.drop_older_attempts
insert_test_results = TestResultWriter.insert
benchmarks_already_ingested = BenchmarkWriter.already_ingested
insert_benchmarks = BenchmarkWriter.insert
capabilities_already_ingested = CapabilityWriter.already_ingested
insert_capabilities = CapabilityWriter.insert
artifact_already_recorded = ArtifactWriter.artifact_recorded
artifact_result_already_recorded = ArtifactWriter.result_recorded
insert_gha_artifact_result = ArtifactWriter.insert_gha_result
insert_artifact = ArtifactWriter.insert_artifact
insert_artifact_result = ArtifactWriter.insert_result

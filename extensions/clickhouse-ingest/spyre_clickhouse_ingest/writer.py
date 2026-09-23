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

import sys

from . import schema
from .identity import BenchmarkId, CapabilityId, CaseId, DerivedId, GhaArtifactId


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

    @classmethod
    def already_ingested(
        cls, client, db: str, run_id: str, component: str, source_file: str = ""
    ) -> bool:
        """Have this source file's rows for this run already landed?"""
        return cls._seen(
            client,
            db,
            run_id,
            component,
            (("props['source_file']", "sf", source_file),),
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
    ) -> int:
        """Write one leg's cases; returns the number of outcome rows written."""
        if not cases:
            return 0
        ident_rows, run_rows = {}, []
        skipped = 0
        for c in cases:
            tags = CaseId.tags_for(c)
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
                    "ran_in": run_id,
                    **({"source_file": source_file} if source_file else {}),
                },
            }
            run_rows.append(run_row)
        written = cls._flush(client, db, ident_rows, run_rows)
        cls._warn(skipped, "case(s) skipped -- identity not derivable")
        return written


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
    """The GHA leg's own artifact row and its verdict, written as one call."""

    artifact_table = schema.Artifacts
    result_table = schema.ArtifactResults

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
    def result_recorded(
        cls,
        client,
        db: str,
        artifact_id: str,
        run_id: str,
        result_kind: str,
        test_type: str,
    ) -> bool:
        """Has this verdict landed? Scoped by the sort key: one run reports N tiers."""
        return (
            cls.result_table.count_rows(
                client,
                db,
                "artifact_id = {artifact_id:UUID} AND run_id = {run_id:UUID} "
                "AND result_kind = {result_kind:String} "
                "AND test_type = {test_type:String}",
                {
                    "artifact_id": artifact_id,
                    "run_id": run_id,
                    "result_kind": result_kind,
                    "test_type": test_type,
                },
            )
            > 0
        )

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
        result_kind: str = "functional",
        duration_s: float = 0.0,
        base_artifact_id: str = "",
        installed: str = "",
        repo: str = "",
        git_ref: str = "",
        git_sha: str = "",
        run_url: str = "",
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

        # aid == base means the leg ran the image UNCHANGED, so the artifact is the
        # one the orchestrator already recorded, and our own row would be a duplicate.
        if aid != DerivedId.norm(base_artifact_id) and not cls.artifact_recorded(
            client, db, aid
        ):
            cls.artifact_table.insert(
                client,
                [
                    cls._artifact_row(
                        aid,
                        comp,
                        a,
                        base_artifact_id,
                        installed,
                        repo,
                        git_ref,
                        git_sha,
                        run_url,
                    )
                ],
                db=db,
            )

        if cls.result_recorded(client, db, aid, rid, result_kind, test_type):
            return True

        result_row: schema.ArtifactResultRow = {
            "artifact_id": aid,
            "run_id": rid,
            "result_kind": result_kind,
            "test_type": test_type,
            "state": state,
            # Where it RAN; kept apart from artifacts.arch by design.
            "arch": a,
            "duration_s": float(duration_s or 0.0),
            "props": {
                k: v for k, v in {"run_url": run_url, "source": "gha"}.items() if v
            },
        }
        cls.result_table.insert(client, [result_row], db=db)
        return True

    @staticmethod
    def _artifact_row(
        aid, comp, arch, base_artifact_id, installed, repo, git_ref, git_sha, run_url
    ) -> "schema.ArtifactRow":
        """The artifacts row for a GHA leg; hash inputs stay readable beside the id."""
        base = DerivedId.norm(base_artifact_id)
        row: schema.ArtifactRow = {
            "artifact_id": aid,
            "component": comp,
            "arch": arch,
            "kind": "image",
            # The hashed name, not a display string.
            "artifact_name": base,
            # chk_origin admits no 'gha'; the 'base=' dep is what marks it derived.
            "origin": "built",
            "identity_deps": [f"{schema.DEP_BASE_PREFIX}{base}"] if base else [],
            "context_deps": [],
            # Tuple order (repo, git_ref, git_sha) -- what the covered-tier join reads.
            "sources": [(repo, git_ref, git_sha)]
            if (repo or git_ref or git_sha)
            else [],
            "props": {
                k: v
                for k, v in {
                    # The digest GhaArtifactId put in the id12 slot.
                    "id12": GhaArtifactId.installed_digest(installed),
                    "base_artifact_id": base,
                    "installed": (installed or "").strip(),
                    "run_url": run_url,
                    "source": "gha",
                }.items()
                if v
            },
        }
        return row


# Function API, kept so installed consumers import one definition, not a copy.
cases_already_ingested = TestResultWriter.already_ingested
insert_test_results = TestResultWriter.insert
benchmarks_already_ingested = BenchmarkWriter.already_ingested
insert_benchmarks = BenchmarkWriter.insert
capabilities_already_ingested = CapabilityWriter.already_ingested
insert_capabilities = CapabilityWriter.insert
artifact_already_recorded = ArtifactWriter.artifact_recorded
artifact_result_already_recorded = ArtifactWriter.result_recorded
insert_gha_artifact_result = ArtifactWriter.insert_gha_result

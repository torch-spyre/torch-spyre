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

"""bench / bench-definitions / bench-qualification / bench-invalidation bundles through the real offline
sealer and the real ingest, against schema 10-84 on chdb."""

import json
import types
import uuid

import pytest
import regex as re
from spyre_clickhouse_ingest import bench_bundle, bundle, client as client_mod, offline
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

FILES = (
    "10-functional-tests", "20-artifacts", "30-benchmarks", "49-result-uploads", "50-artifact-views",
    "60-benchmark-views", "81-bench-environments", "82-run-placements", "83-result-invalidations", "84-bench-views",
)  # fmt: skip
AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
ENV = "eeeeeeee-0000-5000-8000-000000000001"
RUN = "4b2b2c4e-1a7e-5c8e-9d2f-0a1b2c3d4e5f"
NODE = "n-01"
DEVICE = ("node", "serial", "pci", "firmware", "link")


class Chdb:
    """A chdb session behind the clickhouse_connect calls the writers make. chdb shares one engine across sessions
    and tests, so every table and view of the schema files is dropped before and after each test."""

    def __init__(self):
        self.s = session.Session()
        statements = [
            st
            for name in FILES
            for st in SchemaApplier.statements((SCHEMA_DIR / f"{name}.sql").read_text())
        ]
        self.objects = [
            (m.group(1), m.group(2))
            for st in statements
            if (
                m := re.match(
                    r"\s*CREATE\s+(TABLE|VIEW|MATERIALIZED VIEW)\s+IF NOT EXISTS\s+(\w+)",
                    st,
                )
            )
        ]
        self.name = "default"
        self.drop()
        for st in statements:
            self.s.query(st)
        self.s.query(
            "INSERT INTO artifacts (artifact_id, component, kind, artifact_name, arch, origin, props) VALUES "
            f"('{AID}', 'comp', 'image', 'img', 'x86_64', 'upstream', map())"
        )

    def drop(self):
        for kind, name in reversed(self.objects):
            self.s.query(
                f"DROP {'VIEW' if 'VIEW' in kind else 'TABLE'} IF EXISTS {name}"
            )

    @staticmethod
    def lit(v):
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(Chdb.lit(x) for x in v) + "]"
        if isinstance(v, (int, float)):
            return repr(v)
        return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"

    def sql(self, sql, parameters):
        return re.sub(
            r"\{(\w+):[\w()]+\}", lambda m: self.lit(parameters[m.group(1)]), sql
        )

    def query(self, sql, parameters=None, **kw):
        out = self.s.query(self.sql(sql, parameters or {}), "JSONCompact").data()
        out = out.decode() if isinstance(out, bytes) else out
        rows = json.loads(out)["data"] if out.strip() else []
        return types.SimpleNamespace(result_rows=[tuple(r) for r in rows])

    def command(self, sql, parameters=None, **kw):
        sql = self.sql(sql, parameters or {})
        if sql.lstrip().upper().startswith("EXISTS"):
            return int(self.s.query(sql, "TabSeparated").data().strip() or 0)
        self.s.query(sql)

    def insert(self, table, rows, column_names=None, database=None, **kw):
        def val(v):
            if hasattr(v, "isoformat"):
                return v.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if isinstance(v, list):
                return [dict(zip(DEVICE, x)) if isinstance(x, tuple) else x for x in v]
            return v

        body = "\n".join(
            json.dumps({c: val(v) for c, v in zip(column_names, r)}) for r in rows
        )
        self.s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")

    def set_client_setting(self, *a):
        pass

    def rows(self, sql):
        return self.query(sql).result_rows


@pytest.fixture
def db(monkeypatch):
    d = Chdb()
    monkeypatch.setattr(client_mod.ClickHouse, "connect", staticmethod(lambda **k: d))
    monkeypatch.setattr(
        client_mod.ClickHouseEnv, "target_database", staticmethod(lambda: d.name)
    )
    monkeypatch.setattr(client_mod, "tables_present", lambda *a, **k: True)
    found = types.SimpleNamespace(artifact_id=AID, arch="x86_64", component="comp")
    monkeypatch.setattr(
        bundle, "resolve_artifact", lambda *a, **k: (found, "artifact_id")
    )
    yield d
    d.drop()


def result(
    name="decode", samples=((10.0, 11.0), (12.0,), (10.5,)), placement=None, **over
):
    """An engine-shaped Result, run.trust included as the engine writes it."""
    reps = [
        {"run.repeat": i + 1, "status": "ok", "samples": {"latency_ms": list(s)}, "props": {"cache": "warm"},
         "placement": placement or {"nodes": [NODE], "devices": [{"node": NODE, "serial": f"SN{i}", "pci": "0000:1d:00.0"}]},
         "clean": {"exclusive": "observed"}, "started_at": f"2026-11-16T10:0{i}:00Z"}
        for i, s in enumerate(samples)
    ]  # fmt: skip
    status = {
        "case": {"component": "comp", "name": name, "tags": ["Nightly"], "disc": {"batch": "4"}, "props": {"model": "m"}},
        "subject": {"artifact_id": AID},
        "environment": {"env_id": ENV, "fingerprint_version": 1, "fingerprint": {"env.fp.kernel": "6.1"},
                        "labels": {"env.label.node.class": "std"}},
        "repeats": reps,
        "run_props": {"run.purpose": "nightly", "run.validity": "valid", "run.trust": "internal"},
    }  # fmt: skip
    status.update(over)
    return {
        "apiVersion": "bench/v1alpha1",
        "kind": "Result",
        "metadata": {"name": f"{RUN}.{name}"},
        "status": status,
    }


def run_dir(tmp_path, *results, name="run"):
    root = tmp_path / name
    for r in results:
        d = root / "cases" / r["metadata"]["name"]
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(r))
    return root


def seal(tmp_path, src, out="b.tgz", *extra):
    args = offline.parser().parse_args(["--offline", "--bench-dir", str(src), "--arch", "x86_64",
                                         "--out", str(tmp_path / out), *extra])  # fmt: skip
    return offline.write(args)


def ingest(path, trust="", dry_run=False, **over):
    args = types.SimpleNamespace(from_bundle=str(path), trusted_job_prefix=[], expect_key="", bundle_prop=[],
                                 dry_run=dry_run, uploader="alice", bundle_url="", channel_trust=trust,
                                 append_to_run=False, strict=False)  # fmt: skip
    args.__dict__.update(over)
    with offline.opened(path) as root:
        return bundle._ingest(root, args)


def test_sealing_drops_the_engines_trust_claim_and_keys_the_run_by_the_engine_run(
    tmp_path, capsys
):
    out = seal(tmp_path, run_dir(tmp_path, result()))
    assert "run.trust" in capsys.readouterr().err
    with offline.opened(out) as root:
        meta = offline.check(root)
        assert (meta["kind"], meta["artifact_id"], meta["run_key"]) == (
            "bench",
            AID,
            f"manual:spyre-bench:{RUN}",
        )
        assert "run.trust" not in json.dumps(
            [p.read_text() for p in (root / "results").iterdir()]
        )


def test_a_bundle_that_carries_a_channel_key_is_refused(tmp_path):
    out = seal(tmp_path, run_dir(tmp_path, result()))
    with offline.opened(out) as root:
        f = next((root / "results").iterdir())
        data = json.loads(f.read_text())
        data["status"]["repeats"][0]["props"]["run.channel"] = "ci"
        f.write_text(json.dumps(data))
        meta = json.loads((root / "bundle.json").read_text())
        meta["files"] = [
            {"path": p, "sha256": offline.sha256(root / p)}
            for p in offline.bundle_files(root)
        ]
        (root / "bundle.json").write_text(json.dumps(meta))
        with pytest.raises(offline.BundleError, match="run.channel"):
            offline.check(root)


@pytest.mark.parametrize("edit,problem", [
    (lambda r: r["status"]["repeats"].clear(), "at least one repeat"),
    (lambda r: r["status"]["repeats"][0]["samples"].update(latency_ms=[float("nan")]), "finite"),
    (lambda r: r["status"]["repeats"].append(dict(r["status"]["repeats"][0])), "appears twice"),
    (lambda r: r["status"]["subject"].update(artifact_id="x"), "artifact_id"),
    (lambda r: r["status"]["environment"].update(env_id="x"), "env_id"),
])  # fmt: skip
def test_a_bad_result_is_refused(tmp_path, edit, problem):
    r = result()
    edit(r)
    with pytest.raises(offline.BundleError, match=problem):
        seal(tmp_path, run_dir(tmp_path, r))


def test_two_results_for_one_case_are_refused(tmp_path):
    a, b = result(), result()
    b["metadata"]["name"] += "-2"
    with pytest.raises(offline.BundleError, match="same case"):
        seal(tmp_path, run_dir(tmp_path, a, b))


def test_ingest_writes_every_table_derives_trust_and_is_idempotent(tmp_path, db):
    out = seal(tmp_path, run_dir(tmp_path, result(), result("prefill")))
    dry = ingest(out, dry_run=True)
    assert (dry["status"], dry["repeats"], dry["trust"]) == (
        "would-ingest",
        6,
        "external",
    )
    assert db.rows("SELECT count() FROM benchmark_runs") == [(0,)]
    rep = ingest(out)
    assert (
        rep["status"],
        rep["benchmark_runs"],
        rep["new_benchmarks"],
        rep["new_environments"],
    ) == ("ingested", 6, 2, 1)
    assert db.rows("SELECT count() FROM benchmarks") == [(2,)]
    assert db.rows("SELECT count() FROM environment_fingerprints") == [(1,)]
    assert db.rows("SELECT count() FROM run_placements") == [
        (3,)
    ]  # one per repeat number across both cases
    props = db.rows(
        "SELECT props['run.trust'], props['run.repeat'], props['env.id'] FROM benchmark_runs ORDER BY props['run.repeat'] LIMIT 1"
    )
    assert props == [("external", "1", ENV)]
    verdict = db.rows(
        "SELECT state, result_kind, test_type, props['run.trust'], props['bundle_kind'] FROM artifact_results"
    )
    assert verdict == [("passed", "performance", "perf", "external", "bench")]
    assert ingest(out)["status"] == "duplicate"
    again = seal(
        tmp_path,
        run_dir(tmp_path, result(), result("prefill"), name="again"),
        "again.tgz",
    )
    assert (
        offline.report(again)["content_sha256"] == offline.report(out)["content_sha256"]
    )
    assert ingest(again)["status"] == "duplicate"
    assert db.rows("SELECT count() FROM benchmark_runs") == [(6,)]


def test_a_repeat_with_other_content_is_refused_and_new_repeats_are_added(tmp_path, db):
    ingest(seal(tmp_path, run_dir(tmp_path, result())))
    changed = seal(
        tmp_path,
        run_dir(tmp_path, result(samples=((99.0,), (12.0,), (10.5,))), name="r2"),
        "c.tgz",
    )
    with pytest.raises(offline.BundleError, match="different content"):
        ingest(changed)
    more = seal(
        tmp_path,
        run_dir(
            tmp_path,
            result(samples=((10.0, 11.0), (12.0,), (10.5,), (10.7,))),
            name="r3",
        ),
        "m.tgz",
    )
    rep = ingest(more)
    assert (rep["status"], rep["new_repeats"], rep["existing_repeats"]) == (
        "ingested",
        1,
        3,
    )
    assert db.rows("SELECT count() FROM benchmark_runs") == [(4,)]
    assert db.rows("SELECT count() FROM run_placements") == [(4,)]
    newest = "SELECT props['repeats'] FROM artifact_results ORDER BY ts DESC, audit_timestamp DESC LIMIT 1"
    assert db.rows(newest) == [
        ("4",)
    ]  # the verdict over the whole run supersedes the first


def test_a_lower_trust_channel_downgrades_claims(tmp_path, db):
    out = seal(tmp_path, run_dir(tmp_path, result()), "b.tgz", "--seal-trust", "lab")
    ingest(out, trust="lab")
    row = db.rows(
        "SELECT props['run.trust'], props['run.purpose'], props['claim.run.purpose'], props['run.validity'], props['claim.run.validity'] FROM benchmark_runs LIMIT 1"
    )
    assert row == [("lab", "on-demand", "nightly", "unknown", "valid")]
    assert db.rows(
        "SELECT clean['exclusive'], clean['claim.exclusive'], props['placement.source'] FROM run_placements LIMIT 1"
    ) == [("unknown", "observed", "self_reported")]
    assert db.rows("SELECT labels FROM environment_fingerprints") == [
        ({"env.label.origin": "lab"},)
    ]


def test_a_ci_channel_keeps_purpose_validity_labels_and_clean_claims(tmp_path, db):
    ingest(
        seal(tmp_path, run_dir(tmp_path, result()), "b.tgz", "--seal-trust", "none"),
        trust="ci",
    )
    row = db.rows(
        "SELECT props['run.trust'], props['run.purpose'], props['run.validity'], props['claim.run.purpose'] FROM benchmark_runs LIMIT 1"
    )
    assert row == [("ci", "nightly", "valid", "")]
    assert db.rows(
        "SELECT clean['exclusive'], props['placement.source'] FROM run_placements LIMIT 1"
    ) == [("observed", "probe")]
    assert db.rows("SELECT labels FROM environment_fingerprints") == [
        ({"env.label.node.class": "std"},)
    ]


def test_a_ci_bundle_files_its_rows_in_the_workflow_series(tmp_path, db):
    src = run_dir(tmp_path, result(), result("prefill"))
    ci = seal(
        tmp_path,
        src,
        "ci.tgz",
        "--seal-trust",
        "none",
        "--workflow",
        "jenkins-comp-bench-s390x",
    )
    ingest(ci, trust="ci")
    assert db.rows(
        "SELECT DISTINCT extract(props['source_file'], '^([^/]+)/') FROM benchmark_runs"
    ) == [("jenkins-comp-bench-s390x",)]


def test_a_lower_channel_cannot_name_a_series(tmp_path, db):
    src = run_dir(tmp_path, result())
    ingest(
        seal(tmp_path, src, "lab.tgz", "--workflow", "jenkins-comp-bench-s390x"),
        trust="lab",
    )
    assert db.rows(
        "SELECT DISTINCT extract(props['source_file'], '^([^/]+)/') FROM benchmark_runs"
    ) == [("",)]


def test_a_connected_results_run_ingests_a_bench_dir_as_the_channel_says(
    tmp_path, db, capsys
):
    from spyre_clickhouse_ingest.__main__ import main as cli

    src = run_dir(tmp_path, result())
    common = ["results", "--bench-dir", str(src), "--arch", "x86_64", "--trigger-type", "perf",
              "--workflow", "jenkins-comp-bench-s390x", "--run-url", "https://ci/run/1"]  # fmt: skip
    assert cli([*common, "--channel-trust", "ci"]) == 0
    assert db.rows(
        "SELECT DISTINCT props['run.trust'], extract(props['source_file'], '^([^/]+)/') FROM benchmark_runs"
    ) == [("ci", "jenkins-comp-bench-s390x")]
    assert cli(common) == 0
    assert "duplicate" in capsys.readouterr().out


def test_a_legacy_workflow_name_with_spaces_is_a_series_and_none_is_the_unprefixed_one(
    tmp_path, db
):
    src = run_dir(tmp_path, result())
    ingest(
        seal(
            tmp_path,
            src,
            "a.tgz",
            "--seal-trust",
            "none",
            "--workflow",
            "Perf Suite (nightly)",
        ),
        trust="ci",
    )
    assert db.rows(
        "SELECT DISTINCT extract(props['source_file'], '^([^/]+)/') FROM benchmark_runs"
    ) == [("Perf Suite (nightly)",)]
    src2 = run_dir(tmp_path, result("other"), name="run2")
    ingest(seal(tmp_path, src2, "b.tgz", "--seal-trust", "none"), trust="ci")
    assert db.rows(
        "SELECT DISTINCT extract(props['source_file'], '^([^/]+)/') FROM benchmark_runs WHERE position(props['source_file'], 'other') > 0"
    ) == [("",)]


@pytest.mark.parametrize("bad", [5, ["a"], "a/b", "a\nb", "x" * 129, "a\ud800"])
def test_a_malformed_workflow_is_a_rejection_for_ci_and_ignored_below(bad):
    with pytest.raises(offline.BundleError, match="workflow"):
        bench_bundle.series_prefix({"workflow": bad}, "ci")
    assert bench_bundle.series_prefix({"workflow": bad}, "lab") == ""


def test_a_series_name_with_a_slash_is_refused_for_a_ci_channel(tmp_path, db):
    src = run_dir(tmp_path, result())
    out = seal(tmp_path, src, "bad.tgz", "--seal-trust", "none", "--workflow", "a/b")
    with pytest.raises(offline.BundleError, match="series name"):
        ingest(out, trust="ci")


def test_the_sealed_bundles_own_label_can_only_lower_the_channel(tmp_path, db):
    ingest(
        seal(tmp_path, run_dir(tmp_path, result()), "l.tgz", "--seal-trust", "lab"),
        trust="ci",
    )
    assert db.rows("SELECT DISTINCT props['run.trust'] FROM benchmark_runs") == [
        ("lab",)
    ]


def test_a_retracted_bench_upload_leaves_the_readers(tmp_path, db):
    rep = ingest(seal(tmp_path, run_dir(tmp_path, result())))
    uid = "01a13151-2096-79bf-bd9c-311b8c8bc729"
    row = {"upload_id": uid, "state": "published", "actor": "alice", "artifact_id": AID, "run_id": rep["run_id"],
           "format": "bench", "result_kind": "performance", "test_type": "perf", "arch": "x86_64", "filename": "b.tgz",
           "size_bytes": 1, "content_sha256": "c", "bundle_url": "u", "ts": "2026-11-16 10:00:00.000"}  # fmt: skip
    db.s.query("INSERT INTO result_uploads FORMAT JSONEachRow\n" + json.dumps(row))
    assert db.rows("SELECT count() FROM v_invalidated_runs") == [(0,)]
    db.s.query(
        "INSERT INTO result_uploads FORMAT JSONEachRow\n"
        + json.dumps({**row, "state": "retracted", "ts": "2026-11-16 11:00:00.000"})
    )
    assert db.rows("SELECT run_id, reason_code FROM v_invalidated_runs") == [
        (rep["run_id"], "retracted_upload")
    ]


def admin_bundle(tmp_path, kind, payload, name="a.tgz"):
    f = tmp_path / f"{kind}.json"
    f.write_text(json.dumps(payload))
    args = offline.parser().parse_args(
        [
            "--offline",
            "--kind",
            kind,
            "--records-file",
            str(f),
            "--out",
            str(tmp_path / name),
        ]
    )
    return offline.write(args)


DEFS = {"source": {"repo": "r", "ref": "main"}, "records": [
    {"kind": "Environment", "name": "std", "spec_digest": "sha256:aa", "spec": {"labels": {"env.label.node.class": "std"}, "devices": {"n": 4}}},
    {"kind": "NodeClass", "name": "std", "spec_digest": "sha256:bb", "spec": {"host_profile": {"cpus": "192"}}}]}  # fmt: skip


def test_definitions_need_a_ci_channel_and_are_keyed_by_name_and_digest(tmp_path, db):
    out = admin_bundle(tmp_path, "bench-definitions", DEFS)
    assert "trust" not in json.loads(
        offline.opened(out).__enter__().joinpath("bundle.json").read_text()
    )
    for trust in ("", "lab", "external"):
        with pytest.raises(offline.BundleError, match="trust ci"):
            ingest(out, trust=trust)
    assert ingest(out, trust="ci", dry_run=True)["new_records"] == 2
    assert ingest(out, trust="ci")["status"] == "ingested"
    assert ingest(out, trust="ci")["status"] == "duplicate"
    assert db.rows("SELECT name, spec_digest FROM v_environments") == [
        ("std", "sha256:aa")
    ]


def test_a_catalog_kind_without_a_table_is_refused_at_seal(tmp_path):
    bad = {"records": [{"kind": "Benchmark", "name": "x", "spec_digest": "d"}]}
    with pytest.raises(offline.BundleError, match="only Environment, NodeClass"):
        admin_bundle(tmp_path, "bench-definitions", bad)


def qual(outcome="pass", limit=5.0, node=NODE, metric="latency_ms"):
    return {"records": [{"node": node, "env_id": ENV, "probe": "canary", "outcome": outcome, "metric": metric, "cv_pct": 0.1,
                         "limit_pct": limit, "evidence_run_id": rid(), "qualified_at": "2026-11-16T12:00:00Z",
                         "expires_at": "2026-11-23T12:00:00Z"}]}  # fmt: skip


RUN_ID = [""]  # the evidence run, set by the test that ingests it


def rid():
    return RUN_ID[0]


def test_a_qualification_recomputes_the_cv_from_the_evidence_run_and_refuses_an_unsupported_pass(
    tmp_path, db
):
    ev = ingest(
        seal(
            tmp_path,
            run_dir(tmp_path, result(samples=((10.0,), (10.1,), (9.9,)))),
            "b.tgz",
            "--seal-trust",
            "none",
        ),
        trust="ci",
    )
    RUN_ID[0] = ev["run_id"]
    ok = admin_bundle(tmp_path, "bench-qualification", qual(limit=5.0))
    assert ingest(ok, trust="ci")["status"] == "ingested"
    row = db.rows(
        "SELECT outcome, repeats, round(cv_pct, 2), props['claimed_cv_pct'], props['run.trust'] FROM environment_qualifications"
    )
    assert row == [("pass", 3, 1.0, "0.1", "ci")]
    tight = admin_bundle(
        tmp_path,
        "bench-qualification",
        {
            "records": [
                {
                    **qual(limit=0.5)["records"][0],
                    "qualified_at": "2026-11-16T13:00:00Z",
                }
            ]
        },
        "t.tgz",
    )
    with pytest.raises(offline.BundleError, match="within it"):
        ingest(tight, trust="ci")
    elsewhere = admin_bundle(
        tmp_path, "bench-qualification", qual(node="other"), "e.tgz"
    )
    with pytest.raises(offline.BundleError, match="not placed on other"):
        ingest(elsewhere, trust="ci")


def test_a_qualification_from_a_non_ci_run_is_refused(tmp_path, db):
    ev = ingest(seal(tmp_path, run_dir(tmp_path, result())), trust="lab")
    RUN_ID[0] = ev["run_id"]
    with pytest.raises(offline.BundleError, match="not a CI-trusted"):
        ingest(admin_bundle(tmp_path, "bench-qualification", qual()), trust="ci")


def event(state="invalidated", iid="018f0000-0000-7000-8000-000000000001", **scope):
    return {"events": [{"invalidation_id": iid, "scope_kind": "node", "scope": scope or {"node": NODE}, "state": state,
                        "reason_code": "faulty_hardware", "reason": "r"}]}  # fmt: skip


def test_admin_bundles_cannot_be_uploaded(tmp_path):
    out = admin_bundle(tmp_path, "bench-definitions", DEFS)
    with pytest.raises(offline.BundleError, match="ingested by its CI job"):
        offline.upload(
            out,
            types.SimpleNamespace(
                base_url="http://x",
                repo="r",
                token="t",
                root="{arch}",
                inbox="inbox",
                as_folder=False,
                wait=0,
            ),
        )


def test_a_qualification_without_a_metric_uses_the_evidence_runs_only_metric(
    tmp_path, db
):
    ev = ingest(
        seal(
            tmp_path,
            run_dir(tmp_path, result(samples=((10.0,), (10.1,), (9.9,)))),
            "b.tgz",
            "--seal-trust",
            "none",
        ),
        trust="ci",
    )
    RUN_ID[0] = ev["run_id"]
    record = {k: v for k, v in qual()["records"][0].items() if k != "metric"}
    assert (
        ingest(
            admin_bundle(tmp_path, "bench-qualification", {"records": [record]}),
            trust="ci",
        )["status"]
        == "ingested"
    )
    assert db.rows(
        "SELECT repeats, round(cv_pct, 2) FROM environment_qualifications"
    ) == [(3, 1.0)]


def placement(run_id, **update):
    return {"updates": [{"run_id": run_id, **update}]}


def test_a_ci_placement_update_sets_the_clean_decision_over_the_provisional_one(
    tmp_path, db
):
    ev = ingest(
        seal(tmp_path, run_dir(tmp_path, result()), "b.tgz", "--seal-trust", "none"),
        trust="ci",
    )
    clean = {"exclusive": "enforced", "co_tenants": "0/0", "lease": "L1", "node": NODE}
    out = admin_bundle(
        tmp_path, "bench-placement", placement(ev["run_id"], clean=clean)
    )
    for trust in ("", "lab", "external"):
        with pytest.raises(offline.BundleError, match="trust ci"):
            ingest(out, trust=trust)
    assert ingest(out, trust="ci", dry_run=True)["new_records"] == 3
    assert ingest(out, trust="ci")["status"] == "ingested"
    assert ingest(out, trust="ci")["status"] == "duplicate"
    row = "SELECT repeat, clean['exclusive'], clean['lease'], nodes FROM v_run_placements_effective ORDER BY repeat"
    assert db.rows(row) == [(n, "enforced", "L1", [NODE]) for n in (1, 2, 3)]
    assert db.rows(
        "SELECT count() FROM run_placements WHERE props['placement.source'] = 'ci_update'"
    ) == [(3,)]
    only = admin_bundle(
        tmp_path,
        "bench-placement",
        placement(ev["run_id"], repeats=[2], clean={"exclusive": "unknown"}),
        "o.tgz",
    )
    assert ingest(only, trust="ci")["new_records"] == 1
    assert db.rows(
        "SELECT repeat, clean['exclusive'] FROM v_run_placements_effective ORDER BY repeat"
    ) == [(1, "enforced"), (2, "unknown"), (3, "enforced")]


def test_a_placement_update_can_replace_devices_and_needs_an_existing_run(tmp_path, db):
    ev = ingest(
        seal(tmp_path, run_dir(tmp_path, result()), "b.tgz", "--seal-trust", "none"),
        trust="ci",
    )
    devs = [{"node": NODE, "serial": "SN-A", "pci": "0000:1d:00.0"}]
    out = admin_bundle(
        tmp_path, "bench-placement", placement(ev["run_id"], devices=devs, repeats=[1])
    )
    ingest(out, trust="ci")
    assert db.rows(
        "SELECT arrayMap(d -> d.serial, devices) FROM v_run_placements_effective WHERE repeat = 1"
    ) == [(["SN-A"],)]
    with pytest.raises(offline.BundleError, match="no placement rows"):
        ingest(
            admin_bundle(
                tmp_path,
                "bench-placement",
                placement(str(uuid.uuid4()), clean={"exclusive": "enforced"}),
                "n.tgz",
            ),
            trust="ci",
        )
    with pytest.raises(offline.BundleError, match="repeat\\(s\\) \\[9\\]"):
        ingest(
            admin_bundle(
                tmp_path,
                "bench-placement",
                placement(ev["run_id"], repeats=[9], clean={"a": "b"}),
                "m.tgz",
            ),
            trust="ci",
        )


@pytest.mark.parametrize("update,problem", [
    ({"run_id": "x", "clean": {"a": "b"}}, "run_id is a UUID"),
    ({"run_id": RUN, "repeats": []}, "non-empty list"),
    ({"run_id": RUN}, "nothing to update"),
    ({"run_id": RUN, "clean": {"run.trust": "ci"}}, "run.trust"),
])  # fmt: skip
def test_a_bad_placement_update_is_refused_at_seal(tmp_path, update, problem):
    with pytest.raises(offline.BundleError, match=problem):
        admin_bundle(tmp_path, "bench-placement", {"updates": [update]})


def test_per_repeat_props_cannot_skip_the_downgrade(tmp_path, db):
    r = result()
    r["status"]["repeats"][0]["props"] = {
        "run.purpose": "official",
        "run.validity": "valid",
    }
    ingest(
        seal(tmp_path, run_dir(tmp_path, r), "b.tgz", "--seal-trust", "lab"),
        trust="lab",
    )
    row = "SELECT props['run.purpose'], props['run.validity'], props['claim.run.purpose'] FROM benchmark_runs WHERE props['run.repeat'] = '1'"
    assert db.rows(row) == [("on-demand", "unknown", "official")]


def test_two_cases_that_share_one_id_are_refused_at_ingest(tmp_path, db):
    b = result()
    b["status"]["case"]["disc"] = {"batch": "4", "x": "Foo"}
    c = result()
    c["status"]["case"]["disc"] = {"batch": "4", "x": "foo"}
    c["status"]["case"]["disc_keys"] = ["batch"]
    b["status"]["case"]["disc_keys"] = ["batch"]
    b["metadata"]["name"] += "-b"
    out = seal(tmp_path, run_dir(tmp_path, b, c), "b.tgz")
    with pytest.raises(offline.BundleError, match="two cases share one id"):
        ingest(out)


def test_ci_describes_an_environment_a_lower_channel_registered_first(tmp_path, db):
    ingest(
        seal(tmp_path, run_dir(tmp_path, result()), "b.tgz", "--seal-trust", "none"),
        trust="external",
    )
    assert db.rows("SELECT labels FROM v_environment_fingerprints") == [
        ({"env.label.origin": "external"},)
    ]
    ci = result()
    ci["metadata"]["name"] = f"{RUN[:-1]}0.decode"
    rep = ingest(
        seal(
            tmp_path, run_dir(tmp_path, ci, name="ci"), "c.tgz", "--seal-trust", "none"
        ),
        trust="ci",
    )
    assert rep["new_environments"] == 1
    assert db.rows("SELECT labels FROM v_environment_fingerprints") == [
        ({"env.label.node.class": "std"},)
    ]


def test_a_lower_channel_cannot_add_repeats_to_a_higher_trust_run(tmp_path, db):
    ingest(
        seal(tmp_path, run_dir(tmp_path, result()), "b.tgz", "--seal-trust", "none"),
        trust="ci",
    )
    more = result(samples=((10.0, 11.0), (12.0,), (10.5,), (10.7,)))
    out = seal(
        tmp_path, run_dir(tmp_path, more, name="m"), "m.tgz", "--seal-trust", "none"
    )
    with pytest.raises(offline.BundleError, match="cannot add repeats"):
        ingest(out, trust="external")
    assert ingest(out, trust="ci")["new_repeats"] == 1


def test_a_lower_channel_with_no_purpose_is_not_baseline_eligible(tmp_path, db):
    r = result()
    r["status"]["run_props"] = {}
    ingest(seal(tmp_path, run_dir(tmp_path, r)), trust="external")
    row = (
        "SELECT props['run.purpose'], props['run.validity'] FROM benchmark_runs LIMIT 1"
    )
    assert db.rows(row) == [("external", "unknown")]


def test_a_lower_channel_cannot_outrank_ci_with_a_high_fingerprint_version(
    tmp_path, db
):
    low = result()
    low["status"]["environment"]["fingerprint_version"] = 65535
    low["status"]["environment"]["fingerprint"] = {"env.fp.kernel": "EVIL"}
    ingest(
        seal(tmp_path, run_dir(tmp_path, low), "l.tgz", "--seal-trust", "none"),
        trust="external",
    )
    ci = result()
    ci["metadata"]["name"] = f"{RUN[:-1]}0.decode"
    ingest(
        seal(
            tmp_path, run_dir(tmp_path, ci, name="ci"), "c.tgz", "--seal-trust", "none"
        ),
        trust="ci",
    )
    assert db.rows(
        "SELECT fingerprint['env.fp.kernel'] FROM v_environment_fingerprints"
    ) == [("6.1",)]


def test_an_admin_bundle_is_refused_when_it_arrives_through_an_inbox(tmp_path, db):
    out = admin_bundle(tmp_path, "bench-definitions", DEFS)
    with pytest.raises(offline.BundleError, match="is not its artifact_id"):
        ingest(out, trust="ci", expect_key="whatever")


def test_a_new_case_with_no_ok_repeat_updates_the_verdict(tmp_path, db):
    ingest(seal(tmp_path, run_dir(tmp_path, result())))
    both = result("prefill")
    both["status"]["repeats"] = [
        {
            "run.repeat": 1,
            "status": "failed",
            "samples": {},
            "placement": {"nodes": [NODE]},
        }
    ]
    out = seal(tmp_path, run_dir(tmp_path, result(), both, name="two"), "t.tgz")
    assert ingest(out)["status"] == "ingested"
    newest = "SELECT state FROM artifact_results ORDER BY ts DESC, audit_timestamp DESC LIMIT 1"
    assert db.rows(newest) == [("failed",)]


def test_a_failed_repeat_does_not_make_every_resend_new(tmp_path, db):
    r = result()
    r["status"]["repeats"].append(
        {
            "run.repeat": 9,
            "status": "failed",
            "samples": {},
            "placement": {"nodes": [NODE]},
        }
    )
    out = seal(tmp_path, run_dir(tmp_path, r))
    assert ingest(out)["status"] == "ingested"
    assert ingest(out)["status"] == "duplicate"
    assert ingest(out, dry_run=True)["status"] == "duplicate"


def test_a_pass_needs_enough_repeats_in_the_same_environment(tmp_path, db):
    ev = ingest(
        seal(
            tmp_path,
            run_dir(tmp_path, result(samples=((10.0,), (10.1,)))),
            "b.tgz",
            "--seal-trust",
            "none",
        ),
        trust="ci",
    )
    RUN_ID[0] = ev["run_id"]
    with pytest.raises(offline.BundleError, match="at least 3 repeats"):
        ingest(admin_bundle(tmp_path, "bench-qualification", qual()), trust="ci")
    other = {
        "records": [
            {**qual()["records"][0], "env_id": "eeeeeeee-0000-5000-8000-000000000002"}
        ]
    }
    with pytest.raises(offline.BundleError, match="another environment"):
        ingest(
            admin_bundle(tmp_path, "bench-qualification", other, "o.tgz"), trust="ci"
        )


@pytest.mark.parametrize("edit,problem", [
    (lambda r: r["status"]["repeats"][0].update(started_at="garbage"), "started_at"),
    (lambda r: r["status"]["repeats"][0].update(iterations="x"), "iterations"),
    (lambda r: r["status"]["repeats"][0]["placement"].update(links="ab"), "links"),
    (lambda r: r["status"]["case"].update(disc_keys="batch"), "disc_keys"),
])  # fmt: skip
def test_malformed_result_fields_are_rejected_not_retried(tmp_path, edit, problem):
    r = result()
    edit(r)
    with pytest.raises(offline.BundleError, match=problem) as err:
        seal(tmp_path, run_dir(tmp_path, r))
    assert err.value.code == offline.REJECTED


@pytest.mark.parametrize("event_edit,problem", [
    ({"scope_kind": "run", "scope": {"run_id": "x"}}, "run_id is a UUID"),
    ({"scope_kind": "run", "scope": {"run_id": RUN, "repeat": 1}}, "neither benchmark_id nor repeat"),
    ({"scope": {"node": "n", "repeat": 70000}}, "repeat is a count"),
])  # fmt: skip
def test_a_malformed_invalidation_is_rejected_at_seal(tmp_path, event_edit, problem):
    with pytest.raises(offline.BundleError, match=problem):
        admin_bundle(
            tmp_path,
            "bench-invalidation",
            {"events": [{**event()["events"][0], **event_edit}]},
        )

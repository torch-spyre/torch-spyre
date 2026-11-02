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

"""`kind: tables`: one results/<table>.json per table, recorded through the writers a CI run uses."""

import json

import pytest

from spyre_clickhouse_ingest import bundle, client, offline, schema, tables
from spyre_clickhouse_ingest.identity import BenchmarkId
from spyre_clickhouse_ingest.writer import DryRunClient

AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
RID = "4b2b2c4e-1a7e-5c8e-9d2f-0a1b2c3d4e5f"

CASES = {"rows": [
    {"classname": "tests.test_a", "name": "test_x", "status": "passed", "duration_s": 1.5,
     "tags": ["model__granite"], "measurements": {"latency_ms": 12.5}},
    {"classname": "tests.test_a", "name": "test_y", "status": "failed", "fail_message": "boom"},
]}  # fmt: skip
BENCH = {"rows": [
    {"name": "ragq/ds/q01", "backend": "spyre", "iterations": 1,
     "props": {"record_type": "rag_question", "question": "What is Spyre?", "reference_answer": "A card."},
     "identity": [], "measurements": {"ttft_s": 2.0, "total_s": [10.0, 12.0]},
     "run_props": {"generated_answer": "A card."}},
]}  # fmt: skip
CAP = {"rows": [
    {"test_type": "model_ops", "subject": "m-1", "name": "torch.mul", "status": "passed", "backend": "spyre",
     "arch": "s390x", "sig": {"input_dtypes": "bf16"}},
    {"test_type": "model_ops", "subject": "m-1", "name": "torch.add", "status": "not_implemented", "backend": "cpu"},
]}  # fmt: skip
VERDICT = {"rows": [{"test_type": "regression", "state": "passed", "duration_s": 3.0}]}


@pytest.fixture(autouse=True)
def _tables_exist(monkeypatch):
    monkeypatch.setattr(client, "tables_present", lambda *a, **k: True)


def _bundle(tmp_path, files, test_type="regression", **meta):
    root = tmp_path / "b"
    (root / "results").mkdir(parents=True)
    for table, data in files.items():
        (root / "results" / f"{table}.json").write_text(json.dumps(data))
    fields = {"kind": "tables", "artifact_id": AID, "test_type": test_type, "component": "spyre-inference",
              "arch": "s390x", **meta}  # fmt: skip
    (root / "bundle.json").write_text(json.dumps(fields))
    return root


def test_a_tables_bundle_validates(tmp_path):
    meta = offline.check(
        _bundle(tmp_path, {"test_case_runs": CASES, "artifact_results": VERDICT})
    )
    assert meta["kind"] == "tables"


@pytest.mark.parametrize(
    "files,test_type,problem",
    [
        ({"test_cases": CASES}, "regression", "name it after its table"),
        ({"test_case_runs": {"rows": []}}, "regression", "non-empty"),
        ({"test_case_runs": {"rows": [{"name": "t", "status": "bad"}]}}, "regression", "status 'bad'"),
        ({"test_case_runs": {"rows": [{"name": "t", "status": "passed", "note": 1}]}}, "regression", "unknown key"),
        ({"test_case_runs": {"rows": [{"name": "t", "status": "passed", "props": {"x": 1}}]}}, "regression", "result.*"),
        ({"test_case_runs": {"rows": [{"name": "t", "status": "passed", "measurements": {"m": float("inf")}}]}}, "regression", "finite"),
        ({"benchmark_runs": BENCH}, "regression", "perf or quality"),
        ({"capability_runs": CAP}, "regression", "capability test_type"),
        ({"test_case_runs": CASES}, "model_ops", "functional test_type"),
        ({"capability_runs": CAP}, "model_serving", "row has test_type 'model_ops'"),
        ({"artifact_results": {"rows": VERDICT["rows"] * 2}}, "regression", "one verdict"),
        ({"artifact_results": {"rows": [{"test_type": "unit", "state": "passed"}]}}, "regression", "the bundle is 'regression'"),
        ({"benchmark_runs": {"rows": [{**BENCH["rows"][0], "identity": ["missing"]}]}}, "quality", "does not have"),
    ],
)  # fmt: skip
def test_a_bad_bundle_is_refused_with_the_reason(tmp_path, files, test_type, problem):
    with pytest.raises(offline.BundleError, match=problem):
        offline.check(_bundle(tmp_path, files, test_type))


def test_cases_and_a_verdict_go_through_the_test_writer(tmp_path):
    root = _bundle(tmp_path, {"test_case_runs": CASES})
    meta = offline.check(root)
    c = DryRunClient()
    out = bundle.write_tables(
        c, "db", root, {**meta, "run_key": "k"}, AID, RID, {"source": "bundle"}
    )
    assert out["test_case_runs"] == 2
    runs = {
        r["classname"] + "." + r["name"] if "classname" in r else r["test_case_id"]: r
        for r in c.rows[schema.TEST_CASE_RUNS.name]
    }
    states = sorted(r["status"] for r in c.rows[schema.TEST_CASE_RUNS.name])
    assert states == ["failed", "passed"]
    ok = next(r for r in c.rows[schema.TEST_CASE_RUNS.name] if r["status"] == "passed")
    assert ok["measurements"] == {"latency_ms": 12.5} and ok["duration_s"] == 1.5
    assert len(runs) == 2
    (verdict,) = c.rows[schema.ARTIFACT_RESULTS.name]
    assert (verdict["state"], verdict["result_kind"], verdict["test_type"]) == (
        "failed",
        "functional",
        "regression",
    )


def test_a_files_own_verdict_wins_over_the_derived_one(tmp_path):
    root = _bundle(
        tmp_path,
        {
            "test_case_runs": CASES,
            "artifact_results": {
                "rows": [
                    {
                        "test_type": "regression",
                        "state": "passed",
                        "props": {"note": "retried"},
                    }
                ]
            },
        },
    )
    meta = offline.check(root)
    c = DryRunClient()
    bundle.write_tables(c, "db", root, {**meta, "run_key": "k"}, AID, RID, {})
    (verdict,) = c.rows[schema.ARTIFACT_RESULTS.name]
    assert verdict["state"] == "passed" and verdict["props"]["note"] == "retried"


def test_a_benchmark_row_can_key_itself_by_name_alone(tmp_path):
    corrected = {
        "rows": [
            {
                **BENCH["rows"][0],
                "props": {
                    **BENCH["rows"][0]["props"],
                    "reference_answer": "A corrected one.",
                },
            }
        ]
    }
    ids = []
    for data in (BENCH, corrected):
        c = DryRunClient()
        root = _bundle(tmp_path / str(len(ids)), {"benchmark_runs": data}, "quality")
        bundle.write_tables(
            c, "db", root, {**offline.check(root), "run_key": "k"}, AID, RID, {}
        )
        (b,) = c.rows[schema.BENCHMARKS.name]
        ids.append(b["benchmark_id"])
        (run,) = c.rows[schema.BENCHMARK_RUNS.name]
        assert run["measurements"] == {"ttft_s": [2.0], "total_s": [10.0, 12.0]}
        assert run["props"]["generated_answer"] == "A card."
    assert ids[0] == ids[1]


def test_a_quality_benchmark_bundle_records_a_performance_verdict(tmp_path):
    root = _bundle(tmp_path, {"benchmark_runs": BENCH}, "quality")
    c = DryRunClient()
    bundle.write_tables(
        c, "db", root, {**offline.check(root), "run_key": "k"}, AID, RID, {}
    )
    (verdict,) = c.rows[schema.ARTIFACT_RESULTS.name]
    assert (verdict["result_kind"], verdict["test_type"]) == ("performance", "quality")


def test_by_default_every_benchmark_prop_is_identity(tmp_path):
    plain = {"rows": [{k: v for k, v in BENCH["rows"][0].items() if k != "identity"}]}
    changed = {
        "rows": [
            {
                **plain["rows"][0],
                "props": {**plain["rows"][0]["props"], "reference_answer": "Other."},
            }
        ]
    }
    ids = []
    for data in (plain, changed):
        c = DryRunClient()
        root = _bundle(tmp_path / str(len(ids)), {"benchmark_runs": data}, "quality")
        bundle.write_tables(
            c, "db", root, {**offline.check(root), "run_key": "k"}, AID, RID, {}
        )
        ids.append(c.rows[schema.BENCHMARKS.name][0]["benchmark_id"])
    assert ids[0] != ids[1]


def test_the_name_only_identity_matches_the_bridge(tmp_path):
    root = _bundle(tmp_path, {"benchmark_runs": BENCH}, "quality")
    c = DryRunClient()
    bundle.write_tables(
        c, "db", root, {**offline.check(root), "run_key": "k"}, AID, RID, {}
    )
    bid = c.rows[schema.BENCHMARKS.name][0]["benchmark_id"]
    assert bid == BenchmarkId.derive("spyre-inference", "ragq/ds/q01", [], {}, ())


def test_capabilities_are_written_with_their_status_backend_and_arch(tmp_path):
    root = _bundle(tmp_path, {"capability_runs": CAP}, "model_ops")
    c = DryRunClient()
    bundle.write_tables(
        c, "db", root, {**offline.check(root), "run_key": "k"}, AID, RID, {}
    )
    runs = {r["status"]: r for r in c.rows[schema.CAPABILITY_RUNS.name]}
    assert set(runs) == {"passed", "not_implemented"}
    assert (runs["passed"]["backend"], runs["passed"]["arch"]) == ("spyre", "s390x")
    assert runs["not_implemented"]["backend"] == "cpu"
    (verdict,) = c.rows[schema.ARTIFACT_RESULTS.name]
    assert (verdict["result_kind"], verdict["test_type"]) == ("capability", "model_ops")


def test_every_uploadable_table_has_a_generated_schema_and_the_rest_are_reference_only():
    sch = tables.json_schemas()
    assert {f"{t}.upload" for t in tables.UPLOADABLE} <= set(sch)
    stored = {k.removesuffix(".stored") for k in sch if k.endswith(".stored")}
    declared = {
        cls.name
        for cls in vars(schema).values()
        if isinstance(cls, type) and issubclass(cls, schema.Table) and cls.name
    }
    assert stored == declared
    assert declared >= set(tables.UPLOADABLE) | set(tables.REFERENCE_ONLY)
    assert all(
        sch[f"{t}.stored"]["x-uploadable"] == (t in tables.UPLOADABLE) for t in declared
    )


def test_every_table_is_classified_as_uploadable_reference_only_or_identity():
    """A new table in schema.py must be placed on purpose: uploadable, reference-only, or an identity table."""
    declared = {
        c.name
        for c in vars(schema).values()
        if isinstance(c, type) and issubclass(c, schema.Table) and c.name
    }
    placed = (
        set(tables.UPLOADABLE)
        | set(tables.REFERENCE_ONLY)
        | {"test_cases", "benchmarks", "capabilities"}
    )
    assert declared - placed == set(), "classify in tables.py and docs/table-formats.md"
    assert placed - declared == set(), "names a table schema.py does not declare"


def test_the_generated_enums_come_from_the_table_classes():
    sch = tables.json_schemas()
    status = sch["test_case_runs.upload"]["properties"]["rows"]["items"]["properties"][
        "status"
    ]["enum"]
    assert status == sorted(schema.STATUS_VALUES)
    stored = sch["artifact_results.stored"]["properties"]
    assert stored["test_type"]["enum"] == sorted(schema.TEST_TYPE_VALUES)


def test_the_committed_schemas_match_the_table_classes(tmp_path):
    """docs/schemas/ is generated; regenerate with `python -m spyre_clickhouse_ingest.tables --out docs/schemas`."""
    from pathlib import Path

    tables.main(["--out", str(tmp_path)])
    committed = Path(__file__).parent.parent / "docs" / "schemas"
    fresh = {p.name: p.read_text() for p in tmp_path.iterdir()}
    assert fresh == {p.name: p.read_text() for p in committed.iterdir()}

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

"""Schema 81-84 on chdb: invalidation scopes resolve to the right runs, node intervals, qualification currency, and
the flattened bench views drop invalidated repeats. Each event in a fixture needs its own ts: same-ts events tie in argMax."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

FILES = (
    "10-functional-tests",
    "20-artifacts",
    "30-benchmarks",
    "49-result-uploads",
    "50-artifact-views",
    "60-benchmark-views",
    "81-bench-environments",
    "82-run-placements",
    "83-result-invalidations",
    "84-bench-views",
)
R1, R2, R3, RFF = (
    f"aaaaaaaa-0000-5000-8000-0000000000{n}" for n in ("01", "02", "03", "ff")
)
RB = "bbbbbbbb-0000-5000-8000-000000000001"
BENCH = "cccccccc-0000-5000-8000-000000000001"
E1, E2 = "eeeeeeee-0000-5000-8000-000000000001", "eeeeeeee-0000-5000-8000-000000000002"
INV = [f"018f0000-0000-7000-8000-00000000000{n}" for n in range(1, 6)]


class Db:
    def __init__(self):
        self.s = session.Session()
        for name in FILES:
            for stmt in SchemaApplier.statements(
                (SCHEMA_DIR / f"{name}.sql").read_text()
            ):
                self.s.query(stmt)

    def rows(self, sql):
        raw = self.s.query(sql, "JSONEachRow").data()
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return [json.loads(line) for line in raw.splitlines()]

    def insert(self, table, *rows):
        body = "\n".join(json.dumps(r) for r in rows)
        self.s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")

    def event(self, n, ts, **fields):
        self.insert(
            "result_invalidations",
            {"invalidation_id": INV[n], "ts": ts, "state": "invalidated", "actor": "amy@example.com",
             "reason_code": "bad_card", **fields},
        )  # fmt: skip

    def placement(
        self,
        run,
        repeat,
        ts,
        nodes,
        serials,
        env=E1,
        pid=None,
        started_at="ts",
        **extra,
    ):
        devices = [
            {"node": n, "serial": sn, "pci": "0000:00:00.0", "firmware": "1", "link": ""}
            for n, sn in zip(nodes, serials)
        ]  # fmt: skip
        self.insert(
            "run_placements",
            {"run_id": run, "repeat": repeat, "ts": ts, "env_id": env,
             "placement_id": pid or f"dddddddd-0000-5000-8000-0000000000{len(serials)}{repeat}",
             "nodes": nodes, "devices": devices, "started_at": ts if started_at == "ts" else started_at,
             **extra},
        )  # fmt: skip

    def hit(self):
        return sorted(
            (r["run_id"][-2:], r["state"])
            for r in self.rows(
                "SELECT run_id, state FROM v_invalidated_runs WHERE run_id IS NOT NULL"
            )
        )


@pytest.fixture
def db():
    d = Db()
    yield d
    d.s.close()


@pytest.fixture
def placed(db):
    db.placement(R1, 0, "2026-11-01 10:00:00", ["n1"], ["SN1"])
    db.placement(R2, 0, "2026-11-01 11:00:00", ["n2"], ["SN2"])
    db.placement(R3, 0, "2026-11-01 12:00:00", ["n1", "n2"], ["SN1", "SN2"])
    return db


def test_a_device_scope_hits_every_run_that_used_the_card(placed):
    placed.event(0, "2026-11-02 00:00:00", scope_kind="device", device_serial="SN1")
    assert placed.hit() == [("01", "invalidated"), ("03", "invalidated")]


def test_a_node_scope_keeps_its_state_and_a_window_bounds_it(placed):
    placed.event(
        0, "2026-11-02 00:00:00", scope_kind="node", node="n2", state="suspect"
    )
    assert placed.hit() == [("02", "suspect"), ("03", "suspect")]
    placed.event(
        1,
        "2026-11-02 00:00:01",
        scope_kind="node",
        node="n1",
        window_to="2026-11-01 10:30:00",
    )
    assert ("03", "invalidated") not in placed.hit()
    assert ("01", "invalidated") in placed.hit()


def test_a_run_scope_hits_only_that_run(placed):
    placed.event(0, "2026-11-02 00:00:00", scope_kind="run", run_id=RFF)
    assert placed.hit() == [("ff", "invalidated")]


def test_a_restored_event_removes_the_runs(placed):
    placed.event(0, "2026-11-02 00:00:00", scope_kind="device", device_serial="SN1")
    placed.event(
        0,
        "2026-11-02 00:00:01",
        scope_kind="device",
        device_serial="SN1",
        state="restored",
    )
    assert placed.hit() == []
    assert placed.rows("SELECT state FROM v_invalidation_state") == [
        {"state": "restored"}
    ]
    placed.event(1, "2026-11-03 00:00:00", scope_kind="run", run_id=RFF)
    placed.event(
        1, "2026-11-03 00:00:01", scope_kind="run", run_id=RFF, state="restored"
    )
    assert placed.hit() == []


def test_environment_nodes_close_each_interval_at_the_next(db):
    db.insert(
        "environment_nodes",
        {"node": "n1", "env_id": E1, "valid_from": "2026-10-01 00:00:00"},
        {"node": "n1", "env_id": E2, "valid_from": "2026-11-01 00:00:00"},
        {"node": "n2", "env_id": E1, "valid_from": "2026-10-15 00:00:00"},
    )
    got = db.rows(
        "SELECT node, env_id, valid_to FROM v_environment_nodes ORDER BY node, valid_from"
    )
    assert [(r["node"], r["env_id"][-1], r["valid_to"]) for r in got] == [
        ("n1", "1", "2026-11-01 00:00:00.000"),
        ("n1", "2", None),
        ("n2", "1", None),
    ]


def test_the_latest_qualification_decides_whether_a_node_is_current(db):
    def q(node, env, outcome, at, expires):
        return {"node": node, "env_id": env, "probe": "canary", "outcome": outcome, "cv_pct": 1.0,
                "limit_pct": 3.0, "repeats": 5, "evidence_run_id": R1,
                "qualified_at": at, "expires_at": expires}  # fmt: skip

    db.insert(
        "environment_qualifications",
        q("n1", E1, "pass", "2026-11-01 00:00:00", "2099-01-01 00:00:00"),
        q("n1", E1, "fail", "2026-11-02 00:00:00", "2099-01-01 00:00:00"),
        q("n2", E1, "pass", "2026-11-01 00:00:00", "2099-01-01 00:00:00"),
        q("n3", E1, "pass", "2026-01-01 00:00:00", "2026-02-01 00:00:00"),
    )
    got = {
        r["node"]: r["is_current"]
        for r in db.rows("SELECT node, is_current FROM v_environment_qualified")
    }
    assert got == {"n1": 0, "n2": 1, "n3": 0}


@pytest.fixture
def bench(db):
    db.insert("benchmarks", {"benchmark_id": BENCH, "component": "torch-spyre", "name": "matmul",
                             "props": {"shape": "1024"}})  # fmt: skip
    db.insert("environment_fingerprints", {"env_id": E1, "fingerprint_version": 1,
                                           "fingerprint": {"card": "spyre"}, "labels": {"pool": "perf"}})  # fmt: skip
    for repeat, value in ((0, 10.0), (1, 11.0)):
        db.insert(
            "benchmark_runs",
            {"run_id": RB, "benchmark_id": BENCH, "component": "torch-spyre", "backend": "spyre",
             "measurements": {"latency_ms": [value]},
             "props": {"run.repeat": str(repeat), "env.id": E1, "run.tier": "nightly"}},
        )  # fmt: skip
        db.placement(RB, repeat, "2026-11-01 10:00:00", ["n1"], [f"SN{repeat + 8}"])
    return db


def test_attrs_flatten_what_was_recorded_and_omit_the_rest(bench):
    (row,) = bench.rows("SELECT attrs FROM v_bench_attrs WHERE repeat = 0")
    attrs = row["attrs"]
    assert attrs["case.shape"] == "1024"
    assert attrs["run.tier"] == "nightly"
    assert attrs["env.fp.card"] == "spyre"
    assert attrs["env.label.pool"] == "perf"
    assert "env.fp.gpu" not in attrs
    assert "arch" not in attrs and "subject.id" not in attrs
    assert all(v != "" for v in attrs.values())


def test_results_aggregate_repeats_and_drop_an_invalidated_one(bench):
    sql = "SELECT n_repeats, value, cv_pct FROM v_bench_results WHERE metric = 'latency_ms'"
    (both,) = bench.rows(sql)
    assert both["n_repeats"] == 2
    assert both["cv_pct"] == pytest.approx(6.73, abs=0.01)

    bench.event(0, "2026-11-02 00:00:00", scope_kind="device", device_serial="SN9")
    (one,) = bench.rows(sql)
    assert (one["n_repeats"], one["value"], one["cv_pct"]) == (1, 10.0, None)

    bench.event(
        0,
        "2026-11-02 00:00:01",
        scope_kind="device",
        device_serial="SN9",
        state="restored",
    )
    assert bench.rows(sql)[0]["n_repeats"] == 2


def test_a_suspect_run_stays_in_the_results_but_is_flagged(bench):
    bench.event(0, "2026-11-02 00:00:00", scope_kind="node", node="n1", state="suspect")
    (row,) = bench.rows(
        "SELECT n_repeats, is_suspect FROM v_bench_results WHERE metric = 'latency_ms'"
    )
    assert (row["n_repeats"], row["is_suspect"]) == (2, 1)


def test_a_later_placement_row_supersedes_the_earlier_one(db):
    db.placement(R1, 0, "2026-11-01 10:00:00", ["n1"], ["SN1"])
    db.placement(R1, 0, "2026-11-01 10:05:00", ["n1"], ["SN7"])
    assert db.rows("SELECT serial FROM v_run_devices WHERE repeat = 0") == [
        {"serial": "SN7"}
    ]
    db.event(0, "2026-11-02 00:00:00", scope_kind="device", device_serial="SN1")
    assert db.hit() == []
    db.event(1, "2026-11-02 00:00:01", scope_kind="device", device_serial="SN7")
    assert db.hit() == [("01", "invalidated")]


def test_two_placement_rows_for_one_repeat_count_once_per_invalidation(db):
    db.placement(R1, 0, "2026-11-01 10:00:00", ["n1"], ["SN1"])
    db.placement(R1, 0, "2026-11-01 10:05:00", ["n1"], ["SN1"])
    db.event(0, "2026-11-02 00:00:00", scope_kind="node", node="n1")
    db.event(1, "2026-11-02 00:00:01", scope_kind="device", device_serial="SN1")
    assert db.hit() == [("01", "invalidated"), ("01", "invalidated")]
    assert len(db.rows("SELECT 1 FROM v_run_devices")) == 1


def test_a_placement_scope_matches_case_insensitively(placed):
    pid = "DDDDDDDD-0000-5000-8000-0000000000AA"
    placed.placement(R1, 1, "2026-11-01 10:10:00", ["n1"], ["SN1"], pid=pid.lower())
    placed.event(0, "2026-11-02 00:00:00", scope_kind="placement", placement_id=pid)
    assert placed.hit() == [("01", "invalidated")]


def test_windows_are_inclusive_and_apply_to_started_at(placed):
    placed.event(
        0,
        "2026-11-02 00:00:00",
        scope_kind="node",
        node="n1",
        window_from="2026-11-01 12:00:00",
    )
    assert placed.hit() == [("03", "invalidated")]
    placed.event(
        1,
        "2026-11-02 00:00:01",
        scope_kind="node",
        node="n1",
        window_to="2026-11-01 10:00:00",
    )
    assert placed.hit() == [("01", "invalidated"), ("03", "invalidated")]
    # started_at, not insert time, decides; a later corrected row does not move the repeat out of the window
    placed.placement(
        RFF, 0, "2026-11-09 00:00:00", ["n9"], ["SN9"], started_at="2026-11-01 10:00:00"
    )
    placed.placement(RFF, 0, "2026-11-10 00:00:00", ["n9"], ["SN9"], started_at=None)
    placed.event(
        2,
        "2026-11-02 00:00:02",
        scope_kind="node",
        node="n9",
        window_to="2026-11-01 10:30:00",
    )
    assert ("ff", "invalidated") in placed.hit()


def test_scope_is_fixed_by_the_creation_event(placed):
    placed.event(
        0, "2026-11-02 00:00:00", scope_kind="node", node="n1", state="suspect"
    )
    placed.event(
        0,
        "2026-11-02 00:00:01",
        scope_kind="node",
        node="n1",
        window_to="2026-11-01 10:30:00",
    )
    assert placed.hit() == [("01", "invalidated"), ("03", "invalidated")]


def test_same_ts_events_resolve_by_insert_order_whichever_row_is_stored_first(placed):
    def ev(n, serial, state, audit):
        placed.event(n, "2026-11-02 00:00:00", scope_kind="device", device_serial=serial, state=state,
                     audit_timestamp=audit)  # fmt: skip

    ev(0, "SN1", "restored", "2026-11-02 00:00:00.002")
    ev(0, "SN1", "invalidated", "2026-11-02 00:00:00.001")
    ev(1, "SN2", "invalidated", "2026-11-02 00:00:00.001")
    ev(1, "SN2", "restored", "2026-11-02 00:00:00.002")
    assert placed.hit() == []


def test_empty_scope_keys_are_refused(db):
    for bad in (
        {"scope_kind": "device"},
        {"scope_kind": "node"},
        {"scope_kind": "placement"},
        {"scope_kind": "run"},
        {"scope_kind": "run_benchmark", "run_id": RB},
        {"scope_kind": "run", "run_id": RB, "benchmark_id": BENCH},
        {"scope_kind": "run", "run_id": RB, "repeat": 1},
    ):
        with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
            db.event(0, "2026-11-02 00:00:00", **bad)


def test_retracted_uploads_are_included_but_appended_rounds_are_not(db):
    def upload(n, ts, state, run, **props):
        return {"upload_id": f"018f0000-0000-7000-8000-00000000010{n}", "ts": ts, "state": state, "actor": "amy@example.com",
                "artifact_id": E1, "run_id": run, "format": "junit", "result_kind": "functional", "test_type": "smoke",
                "arch": "x86_64", "filename": "a.xml", "size_bytes": 1, "content_sha256": "x", "bundle_url": "u",
                "props": props}  # fmt: skip

    db.insert(
        "result_uploads",
        upload(1, "2026-11-01 00:00:00", "published", R1),
        upload(1, "2026-11-02 00:00:00", "retracted", R1),
        upload(2, "2026-11-01 00:00:00", "published", R2),
        upload(3, "2026-11-01 00:00:00", "published", R3, append_to_run="1"),
        upload(3, "2026-11-02 00:00:00", "retracted", R3, append_to_run="1"),
    )
    assert db.hit() == [("01", "invalidated")]


def test_environment_dimension_views_keep_the_latest_row(db):
    db.insert(
        "environment_fingerprints",
        {
            "env_id": E1,
            "ts": "2026-11-01 00:00:00",
            "fingerprint_version": 1,
            "fingerprint": {"card": "old"},
        },
        {
            "env_id": E1,
            "ts": "2026-11-02 00:00:00",
            "fingerprint_version": 2,
            "fingerprint": {"card": "v2"},
        },
        {
            "env_id": E1,
            "ts": "2026-11-03 00:00:00",
            "fingerprint_version": 1,
            "fingerprint": {"card": "late-v1"},
        },
    )
    db.insert(
        "environments",
        {"name": "e", "spec_digest": "d", "ts": "2026-11-01 00:00:00", "host": "a"},
        {"name": "e", "spec_digest": "d", "ts": "2026-11-02 00:00:00", "host": "b"},
    )
    assert db.rows(
        "SELECT fingerprint_version AS v, fingerprint FROM v_environment_fingerprints"
    ) == [{"v": 2, "fingerprint": {"card": "v2"}}]
    assert db.rows("SELECT host FROM v_environments") == [{"host": "b"}]


def test_a_repeated_node_observation_makes_no_zero_length_interval(db):
    db.insert(
        "environment_nodes",
        {
            "node": "n1",
            "env_id": E1,
            "valid_from": "2026-10-01 00:00:00",
            "ts": "2026-10-01 00:00:00",
        },
        {
            "node": "n1",
            "env_id": E2,
            "valid_from": "2026-10-01 00:00:00",
            "ts": "2026-10-01 00:00:05",
        },
    )
    (row,) = db.rows("SELECT env_id, valid_to FROM v_environment_nodes")
    assert row["env_id"][-1] == "2" and row["valid_to"] is None


def test_equal_qualified_at_resolves_by_insert_order(db):
    def q(node, outcome, audit):
        return {"node": node, "env_id": E1, "probe": "canary", "outcome": outcome, "cv_pct": 1.0, "limit_pct": 3.0,
                "repeats": 5, "evidence_run_id": R1, "qualified_at": "2026-11-01 00:00:00",
                "expires_at": "2099-01-01 00:00:00", "audit_timestamp": audit}  # fmt: skip

    db.insert(
        "environment_qualifications",
        q("n1", "fail", "2026-11-01 00:00:00.002"),
        q("n1", "pass", "2026-11-01 00:00:00.001"),
        q("n2", "pass", "2026-11-01 00:00:00.001"),
        q("n2", "fail", "2026-11-01 00:00:00.002"),
    )
    assert (
        db.rows("SELECT outcome FROM v_environment_qualified")
        == [{"outcome": "fail"}] * 2
    )


def _bench_run(db, run, benchmark, repeat, value, env=True, name="matmul"):
    db.insert(
        "benchmark_runs",
        {"run_id": run, "benchmark_id": benchmark, "component": "torch-spyre", "backend": "spyre",
         "measurements": {"latency_ms": [value]},
         "props": {"run.repeat": str(repeat), **({"env.id": E1} if env else {})}},
    )  # fmt: skip


BENCH2 = "cccccccc-0000-5000-8000-000000000002"


def test_a_run_benchmark_scope_with_a_repeat_drops_only_that_repeat(bench):
    bench.insert(
        "benchmarks",
        {"benchmark_id": BENCH2, "component": "torch-spyre", "name": "conv"},
    )
    _bench_run(bench, RB, BENCH2, 0, 5.0)
    bench.event(
        0,
        "2026-11-02 00:00:00",
        scope_kind="run_benchmark",
        run_id=RB,
        benchmark_id=BENCH,
        repeat=1,
    )
    got = bench.rows(
        "SELECT benchmark_id, n_repeats FROM v_bench_results ORDER BY n_repeats"
    )
    assert [(r["benchmark_id"][-1], r["n_repeats"]) for r in got] == [
        ("1", 1),
        ("2", 1),
    ]
    bench.event(
        1,
        "2026-11-02 00:00:01",
        scope_kind="run_benchmark",
        run_id=RB,
        benchmark_id=BENCH,
    )
    assert [
        r["benchmark_id"][-1]
        for r in bench.rows("SELECT benchmark_id FROM v_bench_results")
    ] == ["2"]


def test_a_duplicate_repeat_row_counts_once(bench):
    _bench_run(bench, RB, BENCH, 0, 10.0)
    (row,) = bench.rows("SELECT n_repeats, value FROM v_bench_results")
    assert (row["n_repeats"], row["value"]) == (2, 10.5)


def test_a_run_without_env_id_has_no_environment_attrs(bench):
    other = "bbbbbbbb-0000-5000-8000-000000000002"
    _bench_run(bench, other, BENCH, 0, 9.0, env=False)
    (row,) = bench.rows(
        f"SELECT env_id, attrs FROM v_bench_attrs WHERE run_id = '{other}'"
    )
    assert row["env_id"] is None
    assert not [k for k in row["attrs"] if k.startswith("env.")]


def test_suspicion_is_flagged_per_repeat_and_per_benchmark(bench):
    bench.insert(
        "benchmarks",
        {"benchmark_id": BENCH2, "component": "torch-spyre", "name": "conv"},
    )
    _bench_run(bench, RB, BENCH2, 0, 5.0)
    bench.event(
        0,
        "2026-11-02 00:00:00",
        scope_kind="device",
        device_serial="SN9",
        state="suspect",
    )
    got = bench.rows(
        "SELECT benchmark_id, repeat, is_suspect FROM v_bench_attrs ORDER BY benchmark_id, repeat"
    )
    assert [(r["benchmark_id"][-1], r["repeat"], r["is_suspect"]) for r in got] == [
        ("1", 0, 0),
        ("1", 1, 1),
        ("2", 0, 0),
    ]
    bench.event(
        1,
        "2026-11-02 00:00:01",
        scope_kind="run_benchmark",
        run_id=RB,
        benchmark_id=BENCH2,
        state="suspect",
    )
    got = bench.rows(
        "SELECT benchmark_id, repeat, is_suspect FROM v_bench_attrs ORDER BY benchmark_id, repeat"
    )
    assert [r["is_suspect"] for r in got] == [0, 1, 1]


def test_a_later_row_corrects_started_at_and_an_unknown_one_matches_any_window(db):
    db.placement(
        R1, 0, "2026-11-03 00:00:00", ["n1"], ["SN1"], started_at="2026-11-01 09:00:00"
    )
    db.placement(
        R1, 0, "2026-11-03 01:00:00", ["n1"], ["SN1"], started_at="2026-11-02 09:00:00"
    )
    db.placement(R2, 0, "2026-11-03 00:00:00", ["n2"], ["SN2"], started_at=None)
    db.placement(
        R3, 0, "2026-11-03 00:00:00", ["n3"], ["SN3"], started_at="2026-11-01 09:00:00"
    )
    for i, node in enumerate(("n1", "n2", "n3")):
        db.event(
            i,
            f"2026-11-04 00:00:0{i}",
            scope_kind="node",
            node=node,
            window_from="2026-11-02 00:00:00",
        )
    # a later row that omits started_at does not erase the recorded one
    db.placement(
        RFF, 0, "2026-11-03 00:00:00", ["n4"], ["SN4"], started_at="2026-11-01 09:00:00"
    )
    db.placement(RFF, 0, "2026-11-03 01:00:00", ["n4"], ["SN4"], started_at=None)
    db.event(
        3,
        "2026-11-04 00:00:03",
        scope_kind="node",
        node="n4",
        window_from="2026-11-02 00:00:00",
    )
    # R1's corrected start is inside the window; R2's is unknown, so the window cannot rule it out; R3 started before it
    assert db.hit() == [("01", "invalidated"), ("02", "invalidated")]


def test_an_env_id_of_nil_uuid_is_no_environment(bench):
    other = "bbbbbbbb-0000-5000-8000-000000000002"
    bench.insert(
        "benchmark_runs",
        {"run_id": other, "benchmark_id": BENCH, "component": "torch-spyre", "backend": "spyre",
         "measurements": {"latency_ms": [9.0]}, "props": {"env.id": "00000000-0000-0000-0000-000000000000"}},
    )  # fmt: skip
    (row,) = bench.rows(
        f"SELECT env_id, attrs FROM v_bench_attrs WHERE run_id = '{other}'"
    )
    assert row["env_id"] is None and "env.id" not in row["attrs"]


def test_a_result_takes_the_env_id_of_the_lowest_repeat_that_recorded_one(db):
    db.insert(
        "benchmarks",
        {"benchmark_id": BENCH, "component": "torch-spyre", "name": "matmul"},
    )
    _bench_run(db, RB, BENCH, 0, 10.0, env=False)
    db.insert(
        "benchmark_runs",
        {"run_id": RB, "benchmark_id": BENCH, "component": "torch-spyre", "backend": "spyre",
         "measurements": {"latency_ms": [12.0]}, "props": {"run.repeat": "2", "env.id": E2}},
    )  # fmt: skip
    _bench_run(db, RB, BENCH, 1, 11.0)
    (row,) = db.rows("SELECT env_id FROM v_bench_results")
    assert row["env_id"] == E1


def test_two_rows_in_one_second_give_a_value_and_attrs_of_the_same_row(db):
    db.insert(
        "benchmarks",
        {"benchmark_id": BENCH, "component": "torch-spyre", "name": "matmul"},
    )
    for value, tag in ((10.0, "a"), (20.0, "b")):
        db.insert(
            "benchmark_runs",
            {"run_id": RB, "benchmark_id": BENCH, "component": "torch-spyre", "backend": "spyre", "ts": "2026-11-01 10:00:00",
             "measurements": {"latency_ms": [value]}, "props": {"run.tag": tag}},
        )  # fmt: skip
    (row,) = db.rows("SELECT value, attrs FROM v_bench_results")
    assert (row["value"], row["attrs"]["run.tag"]) in {(10.0, "a"), (20.0, "b")}


def test_attrs_keys_do_not_collide_and_empty_values_do_not_erase(db):
    db.insert(
        "benchmarks",
        {
            "benchmark_id": BENCH,
            "component": "torch-spyre",
            "name": "matmul",
            "props": {"shape": "1024"},
        },
    )
    db.insert(
        "benchmark_runs",
        {"run_id": RB, "benchmark_id": BENCH, "component": "torch-spyre", "backend": "spyre",
         "measurements": {"latency_ms": [1.0]},
         "props": {"x": "legacy", "run.x": "native", "case.shape": "", "unit.y": "u"}},
    )  # fmt: skip
    (row,) = db.rows("SELECT mapKeys(attrs) AS keys, attrs FROM v_bench_attrs")
    assert len(row["keys"]) == len(set(row["keys"]))
    assert row["attrs"]["run.x"] == "native"
    assert row["attrs"]["case.shape"] == "1024"
    assert row["attrs"]["run.unit.y"] == "u"

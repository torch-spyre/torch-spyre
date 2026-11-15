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

"""The invalidation preview views (schema 89) on real chdb.

v_bench_scope_hits (a scope with no event yet), v_bench_invalidation_impact (an event, any state) and
v_invalidated_runs (the active events the verdicts read) must agree on what a scope covers."""

import pytest
from bench_chdb import LAYOUT, PID, REPEATS, Db, device, place  # noqa: F401
from bench_verdict_fixture import BENCHMARKS, Fixture, bench_id, run_uuid

pytest.importorskip("chdb.session")

SCOPE_ARGS = {
    "scope_kind": "", "run_id": "", "benchmark_id": "", "repeat": -1, "node": "", "device_serial": "",
    "placement_id": "", "window_from": "", "window_to": "",
}  # fmt: skip


@pytest.fixture
def inj():
    d = Db()
    f = Fixture(d)
    f.history(22)
    for n in range(22):
        place(d, n, lambda n, rep: f"2026-10-{n + 1:02d} 02:{rep:02d}:00")
    yield f
    d.s.close()


def hits(db, **kw):
    args = {**SCOPE_ARGS, **kw}
    call = ", ".join(
        f"{k} = {v!r}" if isinstance(v, str) else f"{k} = {v}" for k, v in args.items()
    )
    return db.rows(
        f"SELECT run_id, benchmark_id, repeat FROM v_bench_scope_hits({call})"
    )


def grain(rows):
    return {(r["run_id"], r["benchmark_id"], r["repeat"]) for r in rows}


def impact(db, inv):
    return grain(
        db.rows(
            f"SELECT run_id, benchmark_id, repeat FROM v_bench_invalidation_impact WHERE invalidation_id = '{inv}'"
        )
    )


def active(db, inv):
    """What v_invalidated_runs reads, expanded to every benchmark of the named (run, repeat)."""
    rows = db.rows(
        "SELECT h.run_id AS run_id, e.benchmark_id AS benchmark_id, e.repeat AS repeat FROM v_invalidated_runs AS h "
        "INNER JOIN v_bench_repeat_scope AS e ON e.run_id = h.run_id "
        f"WHERE h.invalidation_id = '{inv}' AND (h.benchmark_id IS NULL OR h.benchmark_id = e.benchmark_id) "
        "AND (h.repeat IS NULL OR h.repeat = e.repeat)"
    )
    return grain(rows)


def event(db, inv, ts, **kw):
    db.insert(
        "result_invalidations",
        {"invalidation_id": inv, "ts": ts, "state": "invalidated", "reason_code": "faulty_hardware", "actor": "amy@example.com", **kw},
    )  # fmt: skip


SCOPES = {
    "run": {"scope_kind": "run", "run_id": run_uuid(3)},
    "run_benchmark": {"scope_kind": "run_benchmark", "run_id": run_uuid(3), "benchmark_id": bench_id("op-alpha")},
    "run_benchmark_repeat": {
        "scope_kind": "run_benchmark", "run_id": run_uuid(3), "benchmark_id": bench_id("op-alpha"), "repeat": 1,
    },
    "node": {"scope_kind": "node", "node": "n1"},
    "node_window": {
        "scope_kind": "node", "node": "n1", "window_from": "2026-10-05 00:00:00", "window_to": "2026-10-12 23:59:59",
    },
    "device": {"scope_kind": "device", "device_serial": "SN9"},
    "placement": {"scope_kind": "placement", "placement_id": PID[4].upper()},
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(SCOPES))
def test_scope_hits_impact_and_active_runs_agree(inj, name):
    fields = SCOPES[name]
    inv = "018f0000-0000-7000-8000-000000000001"
    mine = grain(hits(inj.db, **fields))
    event(inj.db, inv, "2026-11-01 00:00:00", **fields)
    assert mine, name
    assert mine == impact(inj.db, inv) == active(inj.db, inv)


def test_a_node_window_covers_exactly_the_runs_started_in_it(inj):
    got = hits(inj.db, **SCOPES["node_window"])
    runs = {r["run_id"] for r in got}
    assert runs == {run_uuid(n) for n in (4, 5, 6, 8, 10)}
    assert len(got) == len(runs) * REPEATS * len(BENCHMARKS)


def test_window_edges_are_inclusive_and_repeats_are_cut_individually(inj):
    got = hits(
        inj.db,
        scope_kind="device",
        device_serial="SN9",
        window_to="2026-10-05 02:01:00",
    )
    assert {(r["run_id"], r["repeat"]) for r in got} == {
        (run_uuid(4), 0),
        (run_uuid(4), 1),
    }
    got = hits(
        inj.db,
        scope_kind="device",
        device_serial="SN9",
        window_from="2026-10-06 02:00:00",
    )
    assert {(r["run_id"], r["repeat"]) for r in got} == {
        (run_uuid(5), r) for r in range(REPEATS)
    }


def test_a_repeat_with_an_unknown_start_matches_any_window(inj):
    run = inj.run()
    inj.db.insert(
        "run_placements",
        {"run_id": run, "repeat": 0, "env_id": run_uuid(900), "placement_id": PID[7], "started_at": None,
         "nodes": ["n7"], "devices": [{"node": "n7", "serial": "SN7", "pci": "", "firmware": "", "link": ""}]},
    )  # fmt: skip
    got = hits(inj.db, scope_kind="node", node="n7", window_from="2030-01-01 00:00:00")
    assert {(r["run_id"], r["repeat"]) for r in got} == {(run, 0)}
    inv = "018f0000-0000-7000-8000-000000000005"
    event(
        inj.db,
        inv,
        "2026-11-01 00:00:00",
        scope_kind="node",
        node="n7",
        window_from="2030-01-01 00:00:00",
    )
    assert grain(got) == impact(inj.db, inv) == active(inj.db, inv)


def test_a_scope_that_matches_nothing_has_no_rows_and_a_summary_of_zero(inj):
    assert hits(inj.db, scope_kind="device", device_serial="nope") == []
    inv = "018f0000-0000-7000-8000-000000000002"
    event(inj.db, inv, "2026-11-01 00:00:00", scope_kind="device", device_serial="nope")
    (row,) = inj.db.rows(
        f"SELECT runs, repeats, benchmarks FROM v_bench_invalidation_summary WHERE invalidation_id = '{inv}'"
    )
    assert row == {"runs": 0, "repeats": 0, "benchmarks": 0}


def test_a_restored_event_keeps_its_impact_but_leaves_the_active_view(inj):
    inv = "018f0000-0000-7000-8000-000000000003"
    fields = {"scope_kind": "device", "device_serial": "SN9"}
    event(inj.db, inv, "2026-11-01 00:00:00", **fields)
    event(inj.db, inv, "2026-11-01 00:00:01", state="restored", **fields)
    assert impact(inj.db, inv) and not active(inj.db, inv)
    (row,) = inj.db.rows(
        f"SELECT state, runs, repeats FROM v_bench_invalidation_summary WHERE invalidation_id = '{inv}'"
    )
    assert row == {
        "state": "restored",
        "runs": 2,
        "repeats": 2 * REPEATS,
    }


def test_scope_hits_ignore_a_malformed_value_without_erroring(inj):
    assert hits(inj.db, scope_kind="run", run_id="not-a-uuid") == []
    assert hits(inj.db, scope_kind="query") == []


def test_invalidating_a_card_window_restores_the_later_baseline(inj):
    # Ten consecutive nights on one bad card are slow; the next night, on another card, is judged against them.
    poisoned = []
    for n in range(10):
        run = inj.run(slow={"op-alpha": 0.5})
        for rep in range(REPEATS):
            inj.db.insert(
                "run_placements",
                {"run_id": run, "repeat": rep, "env_id": run_uuid(900), "placement_id": PID[0], "started_at": f"2026-11-{n + 1:02d} 02:00:00",
                 "nodes": ["n3"], "devices": [{"node": "n3", "serial": "SNBAD", "pci": "", "firmware": "", "link": ""}]},
            )  # fmt: skip
        poisoned.append(run)
    nxt = inj.run()
    _, before, _ = inj.baseline(nxt)
    assert before > 11
    preview = hits(inj.db, scope_kind="device", device_serial="SNBAD")
    assert {r["run_id"] for r in preview} == set(poisoned)
    inv = "018f0000-0000-7000-8000-000000000004"
    event(
        inj.db, inv, "2026-12-01 00:00:00", scope_kind="device", device_serial="SNBAD"
    )
    ids, after, _ = inj.baseline(nxt)
    assert 9.5 < after < 10.5 and not set(ids) & set(poisoned)
    event(
        inj.db,
        inv,
        "2026-12-01 00:00:01",
        state="restored",
        scope_kind="device",
        device_serial="SNBAD",
    )
    _, restored, _ = inj.baseline(nxt)
    assert restored == before

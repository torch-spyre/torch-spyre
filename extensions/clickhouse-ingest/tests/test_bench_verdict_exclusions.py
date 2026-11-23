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

"""v_benchmark_metric_verdicts and the gates read invalidations and run.purpose, on real chdb.

An invalidated run is out of the verdict output and every baseline; a `suspect` run is shown but
out of every baseline; a run whose run.purpose is not a baseline purpose is judged against the
baseline, never joins it, and cannot be `regressed`. Thresholds are the view's own and unchanged.
Existing-style rows reading as before is pinned by test_bench_reader_exclusion.py's golden file."""

import json

import pytest
from bench_verdict_fixture import Fixture, run_uuid
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

FILES = (
    "10-functional-tests",
    "20-artifacts",
    "30-benchmarks",
    "49-result-uploads",
    "50-artifact-views",
    "60-benchmark-views",
    "62-benchmark-verdicts",
    "81-bench-environments",
    "82-run-placements",
    "83-result-invalidations",
    "85-benchmark-verdict-views",
)
SLOW = {"op-alpha": 0.25}
KEY = ("op-alpha", "kernel_mean_ms")
OTHERS = [
    ("op-beta", "kernel_mean_ms"),
    ("op-gamma", "total_duration_ms"),
    ("op-delta", "output_throughput"),
]
PR = {"run.purpose": "pr-check"}


class Db:
    def __init__(self, nulls=False):
        self.s = session.Session()
        if nulls:
            self.s.query("SET join_use_nulls = 1")
        for name in FILES:
            path = SCHEMA_DIR / f"{name}.sql"
            for o in SchemaApplier.objects(path, path.read_text()):
                # The refreshable MV only fills the history table; the views are under test.
                if o.kind != "refresh":
                    self.s.query(o.sql)

    def rows(self, sql):
        raw = self.s.query(sql, "JSONEachRow").data()
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return [json.loads(line) for line in raw.splitlines()]

    def insert(self, table, *rows):
        body = "\n".join(json.dumps(r) for r in rows)
        self.s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")


@pytest.fixture
def inj():
    d = Db()
    f = Fixture(d)
    f.history(22)
    yield f
    d.s.close()


def status(inj, run, key=KEY):
    return inj.verdicts(run)[key][:2]


def flagged_only_alpha(inj, run, expected):
    v = inj.verdicts(run)
    assert v[KEY][0] == expected
    assert {v[k][0] for k in OTHERS} == {"unchanged"}


# --- behaviour the view keeps -------------------------------------------------------------


def test_history_is_judged_against_a_full_window_of_20(inj):
    assert {v[2] for v in inj.verdicts(run_uuid(21)).values()} == {20}


def test_one_slowed_run_is_suspected_and_two_in_a_row_regressed(inj):
    first = inj.run(slow=SLOW)
    assert status(inj, first) == ("suspected", "single_run")
    second = inj.run(slow=SLOW)
    flagged_only_alpha(inj, second, "regressed")


# --- invalidation -------------------------------------------------------------------------


def test_an_invalidated_run_has_no_verdict_rows(inj):
    inj.run(slow=SLOW)
    bad = inj.run(slow=SLOW)
    assert status(inj, bad)[0] == "regressed"
    inj.invalidate(bad)
    assert inj.verdicts(bad) == {}
    assert inj.db.rows(
        f"SELECT count() n FROM v_benchmark_gate WHERE run_id = toUUID('{bad}')"
    ) == [{"n": 0}]


def test_persistence_skips_an_invalidated_previous_run(inj):
    first = inj.run(slow=SLOW)
    second = inj.run(slow=SLOW)
    inj.invalidate(first)
    assert status(inj, second) == ("suspected", "single_run")


def test_invalidating_poisoned_runs_restores_the_baseline_median(inj):
    poisoned = [inj.run(slow={"op-alpha": 0.5}) for _ in range(10)]
    nxt = inj.run()
    _, before, _ = inj.baseline(nxt)
    assert before > 11
    for i, p in enumerate(poisoned):
        inj.invalidate(p, n=100 + i)
    ids, after, _ = inj.baseline(nxt)
    assert 9.5 < after < 10.5
    assert len(ids) == 20 and not set(ids) & set(poisoned)
    assert set(ids) == {run_uuid(n) for n in range(2, 22)}
    flagged_only_alpha(inj, nxt, "unchanged")


def test_invalidating_one_benchmark_of_a_run_leaves_the_others(inj):
    run = inj.run(slow=SLOW)
    inj.invalidate(run, benchmark="op-alpha")
    v = inj.verdicts(run)
    assert KEY not in v and {v[k][0] for k in OTHERS} == {"unchanged"}


def test_invalidating_one_repeat_keeps_the_remaining_repeats(inj):
    run = inj.run(slow=SLOW)
    inj.invalidate(run, benchmark="op-alpha", repeat=1)
    assert inj.baseline(run)[2] == 2 * 5
    assert status(inj, run)[0] == "suspected"


def test_a_restored_invalidation_counts_again(inj):
    run = inj.run(slow=SLOW)
    inj.invalidate(run)
    assert inj.verdicts(run) == {}
    inj.event(run, "restored", 1, "2026-12-02 00:00:00")
    assert status(inj, run) == ("suspected", "single_run")


# --- suspect ------------------------------------------------------------------------------


def test_a_suspect_run_is_shown_capped_and_out_of_later_baselines(inj):
    slowed = inj.run(slow=SLOW)
    inj.suspect(slowed)
    assert status(inj, slowed) == ("suspected", "suspect_data")
    nxt = inj.run()
    ids, med, _ = inj.baseline(nxt)
    assert slowed not in ids and len(ids) == 20 and 9.5 < med < 10.5


def test_a_suspect_run_cannot_be_regressed(inj):
    inj.run(slow=SLOW)
    second = inj.run(slow=SLOW)
    inj.suspect(second)
    assert status(inj, second) == ("suspected", "suspect_data")


# --- purpose ------------------------------------------------------------------------------


def test_a_pr_run_sharing_the_nightly_source_job_is_excluded_and_capped(inj):
    pr = inj.run(slow={"op-alpha": 1.0}, props=PR)
    nightly = inj.run()
    assert pr not in inj.baseline(nightly)[0]
    flagged_only_alpha(inj, nightly, "unchanged")
    assert status(inj, pr) == ("suspected", "purpose_not_baseline")


def test_a_pr_run_is_never_regressed_even_after_a_slowed_nightly(inj):
    inj.run(slow=SLOW)
    pr = inj.run(slow=SLOW, props=PR)
    assert status(inj, pr) == ("suspected", "purpose_not_baseline")
    gate = inj.db.rows(
        f"SELECT gate_status FROM v_benchmark_gate WHERE run_id = toUUID('{pr}') AND name = 'op-alpha'"
    )
    assert gate == [{"gate_status": "warn"}]


def test_a_pr_run_does_not_make_the_next_nightly_regressed(inj):
    inj.run(slow=SLOW, props=PR)
    nightly = inj.run(slow=SLOW)
    assert status(inj, nightly) == ("suspected", "single_run")


def test_a_pr_run_between_nightlies_neither_joins_later_baselines_nor_feeds_persistence(
    inj,
):
    pr = [inj.run(slow={"op-alpha": 1.0}, props=PR) for _ in range(8)]
    inj.run()
    last = inj.run(slow=SLOW)
    ids, med, _ = inj.baseline(last)
    assert not set(pr) & set(ids) and len(ids) == 20 and 9.5 < med < 10.5
    assert inj.baseline_is_its_runs(last)
    assert status(inj, last) == ("suspected", "single_run")


def test_a_suspect_run_between_nightlies_is_out_of_later_baselines(inj):
    sus = [inj.run(slow={"op-alpha": 1.0}) for _ in range(8)]
    for i, r in enumerate(sus):
        inj.suspect(r, n=50 + i)
    inj.run()
    last = inj.run()
    ids, med, _ = inj.baseline(last)
    assert not set(sus) & set(ids) and 9.5 < med < 10.5
    assert inj.baseline_is_its_runs(last)


def test_pr_runs_do_not_take_window_slots(inj):
    for _ in range(5):
        inj.run(slow=SLOW, props=PR)
    nxt = inj.run()
    ids, _, _ = inj.baseline(nxt)
    assert set(ids) == {run_uuid(n) for n in range(2, 22)}


@pytest.mark.parametrize("purpose", ["pr-check", "release", "investigate", "on-demand"])
def test_other_purposes_are_not_baseline_eligible(inj, purpose):
    run = inj.run(slow=SLOW, props={"run.purpose": purpose})
    assert status(inj, run) == ("suspected", "purpose_not_baseline")
    assert run not in inj.baseline(inj.run())[0]


@pytest.mark.parametrize("purpose", ["nightly", "weekly", "merge", "NIGHTLY", None])
def test_baseline_purposes_and_an_absent_prop_join_the_baseline(inj, purpose):
    run = inj.run(props={"run.purpose": purpose})
    assert run in inj.baseline(inj.run())[0]


def test_old_style_rows_without_validity_or_purpose_props_read_as_before(inj):
    old = Fixture(inj.db, seed=11)
    old.next_run = 100
    runs = [old.run(props={"run.purpose": None}) for _ in range(12)]
    last = old.run(slow=SLOW, props={"run.purpose": None})
    ids, _, _ = inj.baseline(last)
    assert runs[-1] in ids and status(inj, last)[0] in ("suspected", "regressed")


# --- settings -----------------------------------------------------------------------------


def _scenario(f):
    """Score and floor are meaningless below the minimum baseline and differ there by setting."""
    f.history(22)
    bad = f.run(slow=SLOW)
    f.run(slow=SLOW, props=PR)
    f.suspect(f.run(slow=SLOW), n=2)
    f.invalidate(bad)
    f.run(slow=SLOW)
    return f.db.rows(
        "SELECT toString(run_id) r, name, metric, status, reason, n_baseline, n_current, baseline_value, "
        "current_value, worse_pct, if(n_baseline < 10, NULL, score) score, if(n_baseline < 10, NULL, floor_pct) floor_pct, arrayMap(x -> toString(x), baseline_run_ids) ids "
        "FROM v_benchmark_metric_verdicts ORDER BY r, name"
    )


def test_the_verdicts_are_the_same_under_join_use_nulls():
    out = []
    for nulls in (False, True):
        # One chdb session at a time: a new one closes the previous.
        d = Db(nulls=nulls)
        try:
            out.append(_scenario(Fixture(d)))
        finally:
            d.s.close()
    assert out[0] and out[0] == out[1]
    assert {r["reason"] for r in out[0]} >= {"purpose_not_baseline", "suspect_data"}


def test_the_perf_regression_gate_reads_the_exclusions(inj):
    art = "ffffffff-0000-5000-8000-000000000001"
    run = inj.run(slow=SLOW, props=PR)
    inj.db.insert(
        "artifact_results",
        {"run_id": run, "artifact_id": art, "result_kind": "performance", "arch": ARCH, "test_type": "perf", "state": "passed"},
    )  # fmt: skip
    assert inj.db.rows(
        f"SELECT gate_status FROM v_perf_regression_gate WHERE artifact_id = toUUID('{art}')"
    ) == [{"gate_status": "warn"}]
    inj.invalidate(run)
    assert (
        inj.db.rows(
            f"SELECT gate_status FROM v_perf_regression_gate WHERE artifact_id = toUUID('{art}')"
        )
        == []
    )


ARCH = "x86_64"

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

"""Injected slowdowns against the real rolling-baseline view and the verdict reader, on chdb.

Observed behaviour of v_benchmark_metric_verdicts that these tests pin: one slowed run is
`suspected`; `regressed` needs the series' previous run over the floor too. The view drops
invalidated samples and keeps suspect or non-baseline-purpose runs out of baselines; the reader
adds the repeat and validity caps.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from bench_injection import (
    SCHEMA_FILES,
    ClickHouseSandbox,
    Injection,
    bench_id,
    run_uuid,
)
from spyre_clickhouse_ingest import bench_verdicts as bv
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

SLOW = {"op-alpha": 0.25}
KEY = ("op-alpha", "kernel_mean_ms")
OTHERS = [
    ("op-beta", "kernel_mean_ms"),
    ("op-gamma", "total_duration_ms"),
    ("op-delta", "output_throughput"),
]


class Db:
    def __init__(self, files=SCHEMA_FILES):
        self.s = session.Session()
        for name in files:
            for stmt in SchemaApplier.statements(
                (SCHEMA_DIR / f"{name}.sql").read_text()
            ):
                # The refreshable MV and its history table are not needed to read the view.
                if name == "62-benchmark-verdicts" and not stmt.startswith(
                    "CREATE VIEW"
                ):
                    continue
                self.s.query(stmt)

    def rows(self, sql):
        try:
            raw = self.s.query(sql, "JSONEachRow").data()
        except RuntimeError as exc:
            # What bv.Source does for a clickhouse_connect client.
            if bv.is_missing(exc):
                raise bv.ViewMissing(str(exc).splitlines()[0]) from exc
            raise
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return [json.loads(line) for line in raw.splitlines()]

    def insert(self, table, *rows):
        body = "\n".join(json.dumps(r) for r in rows)
        self.s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")


@pytest.fixture
def inj():
    d = Db()
    i = Injection(d)
    i.history(22)
    yield i
    d.s.close()


def statuses(inj, run):
    return {k: v[0] for k, v in inj.verdicts(run).items()}


def only_alpha_flagged(inj, run, status):
    st = statuses(inj, run)
    assert st[KEY] == status
    assert {st[k] for k in OTHERS} == {"unchanged"}


def test_history_is_judged_against_a_baseline(inj):
    assert {v[2] for v in inj.verdicts(run_uuid(21)).values()} == {20}


def test_one_slowed_run_is_suspected_never_regressed(inj):
    run = inj.run(slow=SLOW)
    only_alpha_flagged(inj, run, "suspected")
    assert inj.verdicts(run)[KEY][1] == "single_run"


def test_two_consecutive_slowed_runs_are_regressed_for_exactly_that_metric(inj):
    inj.run(slow=SLOW)
    run = inj.run(slow=SLOW)
    only_alpha_flagged(inj, run, "regressed")
    report = bv.read_verdicts(inj.db, run_id=run)
    assert [(r["name"], r["metric"]) for r in bv.actionable(report)] == [KEY]
    assert report["summary"]["counts"]["regressed"] == 1
    assert report["summary"]["counts"]["pass"] == 3


def test_a_slowed_higher_is_better_metric_is_regressed(inj):
    inj.run(slow={"op-delta": 0.3})
    run = inj.run(slow={"op-delta": 0.3})
    st = statuses(inj, run)
    assert st[("op-delta", "output_throughput")] == "regressed"
    assert {v for k, v in st.items() if k[0] != "op-delta"} == {"unchanged"}


def test_recovery_returns_to_pass_and_improvement_is_reported(inj):
    inj.run(slow=SLOW)
    inj.run(slow=SLOW)
    only_alpha_flagged(inj, inj.run(), "unchanged")
    assert statuses(inj, inj.run(slow={"op-alpha": -0.3}))[KEY] == "improved"


def test_reader_fields_for_a_regressed_run(inj):
    inj.run(slow=SLOW)
    run = inj.run(slow=SLOW)
    r = next(
        x
        for x in bv.read_verdicts(inj.db, run_id=run)["results"]
        if (x["name"], x["metric"]) == KEY
    )
    assert r["verdict"] == "regressed" and r["view_verdict"] == "regressed"
    assert r["tier"] == "blocking" and r["kind"] == "perf"
    assert r["noise"]["n_repeats"] == 3 and 0 < r["noise"]["cv_pct"] < 5
    assert r["baseline"]["n_runs"] == 20 and 9.5 < r["baseline"]["median"] < 10.5
    assert r["baseline"]["low"] < r["baseline"]["median"] < r["baseline"]["high"]
    assert r["delta_pct"] > 20 and r["worse_pct"] == r["delta_pct"]
    assert (
        r["trust"] == "internal"
        and r["validity"] == "valid"
        and r["purpose"] == "nightly"
    )
    assert r["series_key"].endswith("/synth-backend/x86_64/synth-nightly")
    assert not r["invalidated"]


def test_artifact_id_resolves_to_its_runs(inj):
    run = inj.run(slow=SLOW)
    art = "ffffffff-0000-5000-8000-000000000001"
    inj.db.insert(
        "artifact_results",
        {"run_id": run, "artifact_id": art, "result_kind": "performance", "arch": "x86_64", "test_type": "perf", "state": "passed"},
    )  # fmt: skip
    report = bv.read_verdicts(inj.db, artifact_id=art)
    assert report["runs"] == [run] and len(report["results"]) == 4


def test_an_invalidated_run_has_no_verdict_rows_and_is_listed_as_excluded(inj):
    inj.run(slow=SLOW)
    run = inj.run(slow=SLOW)
    inj.invalidate(run, "2026-12-01 00:00:00")
    assert inj.verdicts(run) == {}
    report = bv.read_verdicts(inj.db, run_id=run)
    assert report["results"] == [] and report["excluded_runs"] == [run]
    assert any("invalidated" in n for n in report["notes"])
    assert report["summary"]["actionable"] == 0


def test_invalidating_the_slowed_runs_removes_them_from_the_baseline_and_the_next_run_passes(
    inj,
):
    first = inj.run(slow=SLOW)
    second = inj.run(slow=SLOW)
    inj.invalidate(first, "2026-12-01 00:00:00", n=1)
    inj.invalidate(second, "2026-12-01 00:00:01", n=2)
    nxt = inj.run()
    only_alpha_flagged(inj, nxt, "unchanged")
    assert not {first, second} & set(inj.baseline_ids(nxt))
    assert bv.actionable(bv.read_verdicts(inj.db, run_id=nxt)) == []


def test_invalidating_one_repeat_keeps_the_run_with_its_other_repeats(inj):
    run = inj.run(slow=SLOW)
    inj.invalidate_repeat(run, 0, "2026-12-01 00:00:00")
    report = bv.read_verdicts(inj.db, run_id=run)
    r = next(x for x in report["results"] if (x["name"], x["metric"]) == KEY)
    assert r["noise"]["n_repeats"] == 2 and r["invalidated_repeats"] == [0]
    assert not r["invalidated"] and report["excluded_runs"] == []


def test_suspect_run_is_capped_with_suspect_data_and_kept_out_of_the_baseline(inj):
    first = inj.run(slow=SLOW)
    inj.invalidate(first, "2026-12-01 00:00:00", state="suspect")
    second = inj.run(slow=SLOW)
    inj.invalidate(second, "2026-12-01 00:00:01", n=2, state="suspect")
    report = bv.read_verdicts(inj.db, run_id=second)
    r = next(x for x in report["results"] if (x["name"], x["metric"]) == KEY)
    assert (r["verdict"], r["reason"]) == ("suspected", "suspect_data")
    assert report["summary"]["reasons"] == {"suspect_data": 1}
    assert "suspect_data" in bv.to_markdown(report)
    assert not {first, second} & set(inj.baseline_ids(inj.run()))


def test_few_repeats_cap_regressed_to_suspected(inj):
    inj.run(slow=SLOW)
    run = inj.run(slow=SLOW, repeats=1)
    assert statuses(inj, run)[KEY] == "regressed"
    r = next(
        x
        for x in bv.read_verdicts(inj.db, run_id=run)["results"]
        if (x["name"], x["metric"]) == KEY
    )
    assert (r["verdict"], r["reason"], r["view_verdict"]) == (
        "suspected",
        "few_repeats",
        "regressed",
    )
    assert (
        bv.read_verdicts(inj.db, run_id=run, min_repeats=1)["summary"]["counts"][
            "regressed"
        ]
        == 1
    )


@pytest.mark.parametrize("state", ["suspect", "external"])
def test_a_run_below_valid_cannot_be_regressed(inj, state):
    inj.run(slow=SLOW)
    run = inj.run(slow=SLOW, props={"run.validity": state})
    r = next(
        x
        for x in bv.read_verdicts(inj.db, run_id=run)["results"]
        if (x["name"], x["metric"]) == KEY
    )
    assert (r["verdict"], r["reason"]) == ("suspected", f"validity_{state}")


PR = {"run.purpose": "pr-check"}


def test_pr_runs_with_their_own_source_job_stay_out_of_the_nightly_baseline(inj):
    pr = [
        inj.run(slow={"op-alpha": 1.0}, job="synth-pr", props=PR, day=30)
        for _ in range(3)
    ]
    nightly = inj.run()
    only_alpha_flagged(inj, nightly, "unchanged")
    assert not set(pr) & set(inj.baseline_ids(nightly))
    assert {v[0] for v in inj.verdicts(pr[-1]).values()} == {"no_baseline"}


def test_a_pr_run_sharing_the_nightly_source_job_is_capped_and_kept_out_of_the_baseline(
    inj,
):
    first = inj.run(slow={"op-alpha": 1.0}, props=PR)
    pr = inj.run(slow={"op-alpha": 1.0}, props=PR)
    report = bv.read_verdicts(inj.db, run_id=pr)
    r = next(x for x in report["results"] if (x["name"], x["metric"]) == KEY)
    assert (r["verdict"], r["reason"]) == ("suspected", "purpose_not_baseline")
    assert "purpose_not_baseline" in bv.to_markdown(report)
    nightly = inj.run()
    only_alpha_flagged(inj, nightly, "unchanged")
    assert not {first, pr} & set(inj.baseline_ids(nightly))
    assert inj.verdicts(nightly)[KEY][2] == 20


def test_missing_optional_views_degrade_to_notes_and_keep_the_verdicts():
    d = Db(files=tuple(f for f in SCHEMA_FILES if f[:2] != "84"))
    try:
        i = Injection(d)
        i.history(22)
        i.run(slow=SLOW)
        run = i.run(slow=SLOW)
        report = bv.read_verdicts(d, run_id=run)
        assert report["summary"]["counts"]["regressed"] == 1
        assert any("v_bench_results" in n for n in report["notes"])
        assert all(r["noise"] is None for r in report["results"])
    finally:
        d.s.close()


def test_missing_verdict_view_gives_a_clear_message():
    d = Db(
        files=(
            "10-functional-tests",
            "20-artifacts",
            "30-benchmarks",
            "49-result-uploads",
            "50-artifact-views",
            "60-benchmark-views",
        )
    )
    try:
        report = bv.read_verdicts(d, run_id=run_uuid(0))
        assert report["results"] == [] and any(
            "v_benchmark_metric_verdicts" in n for n in report["notes"]
        )
    finally:
        d.s.close()


def test_renderers(inj):
    inj.run(slow=SLOW)
    report = bv.read_verdicts(inj.db, run_id=inj.run(slow=SLOW))
    assert json.loads(bv.to_json(report))["schema"] == bv.SCHEMA
    md = bv.to_markdown(report)
    assert (
        md.index("regressed") < md.index("pass")
        and "op-alpha" in md
        and "| verdict |" in md
    )


def test_a_bad_id_never_reaches_sql(inj):
    with pytest.raises(ValueError):
        bv.read_verdicts(inj.db, run_id="x'); DROP TABLE benchmark_runs; --")
    with pytest.raises(ValueError):
        bv.read_verdicts(inj.db)


def test_a_benchmark_wide_invalidation_is_reported_as_excluded(inj):
    run = inj.run(slow=SLOW)
    inj._event(
        run,
        "2026-12-01 00:00:00",
        5,
        "bad_card",
        "invalidated",
        "run_benchmark",
        bench_id("op-alpha"),
    )
    report = bv.read_verdicts(inj.db, run_id=run)
    assert KEY not in {(r["name"], r["metric"]) for r in report["results"]}
    assert report["excluded_benchmarks"] == [f"{run}/{bench_id('op-alpha')}"]
    assert len(report["results"]) == 3


def test_an_unknown_run_id_says_so(inj):
    report = bv.read_verdicts(inj.db, run_id=run_uuid(999))
    assert report["results"] == [] and any(
        "no verdict rows" in n for n in report["notes"]
    )


def test_arch_filter_and_validation(inj):
    run = inj.run()
    assert len(bv.read_verdicts(inj.db, run_id=run, arch="x86_64")["results"]) == 4
    assert bv.read_verdicts(inj.db, run_id=run, arch="other")["results"] == []
    with pytest.raises(ValueError):
        bv.read_verdicts(inj.db, run_id=run, arch="x'\\")


def test_reports_are_strict_json(inj):
    assert bv._num(float("inf")) is None and bv._num(float("nan")) is None


class FakeClient:
    def __init__(self):
        self.inserts = []

    def insert(self, table, data, column_names):
        self.inserts.append((table, data, column_names))


def test_sandbox_inserts_pass_utc_datetimes_not_strings(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    client = FakeClient()
    inj = Injection(ClickHouseSandbox(client, "sandbox_x__f1"))
    inj.run(slow=SLOW)
    inj.invalidate(run_uuid(0), "2026-12-01 00:00:00")
    for table, data, cols in client.inserts:
        for name in ("ts", "window_from", "window_to"):
            if name in cols:
                for row in data:
                    v = row[cols.index(name)]
                    assert v is None or (
                        isinstance(v, datetime) and v.utcoffset() == timedelta(0)
                    ), (table, name, v)
    ts = [
        d[0][c.index("ts")]
        for t, d, c in client.inserts
        if t.endswith(".benchmark_runs")
    ]
    assert ts[0] == datetime(2026, 10, 1, 2, 0, tzinfo=timezone.utc)


def test_sandbox_refuses_other_databases():
    with pytest.raises(SystemExit):
        ClickHouseSandbox(FakeClient(), "spyre_v2")


def test_an_unknown_validity_value_counts_as_suspect():
    from spyre_clickhouse_ingest.bench_verdicts import _worst_validity

    assert _worst_validity(["valid", "weird"]) == "weird"
    assert _worst_validity(["shared_env", "weird", "valid"]) == "weird"
    assert _worst_validity(["invalid", "weird"]) == "invalid"

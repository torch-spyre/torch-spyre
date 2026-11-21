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

"""v_benchmark_anchor and v_benchmark_anchored_verdicts on real chdb.

A persistent shift is absorbed by the rolling baseline and goes quiet there; the anchored verdict
keeps saying `regressed` until a new anchor is accepted (a new reference version, tag moved)."""

import uuid
from datetime import timedelta

import pytest
from bench_anchor_fixture import (
    KEY,
    NIGHTS,
    REF_KEY,
    Db,
    accept,
    history,
    ref_id,
    ref_run,
    stamp,
)
from bench_verdict_fixture import (
    ARCH,
    BACKEND,
    BENCHMARKS,
    COMPONENT,
    NS,
    START,
    Fixture,
    bench_id,
    run_uuid,
)


def anchored(f, run, key=KEY):
    out = f.db.rows(
        "SELECT status, reason, since_ts, n_exceeding_runs, days_exceeding, anchor_value, current_value, floor_pct, version "
        "FROM v_benchmark_anchored_verdicts "
        f"WHERE run_id = toUUID('{run}') AND name = '{key[0]}' AND metric = '{key[1]}'"
    )
    return out[0] if out else None


def rolling(f, run, key=KEY):
    return f.verdicts(run)[key][:2]


@pytest.fixture
def fx():
    d = Db()
    f = Fixture(d)
    yield f
    d.s.close()


# --- the absorption scenario --------------------------------------------------------------


def test_rolling_absorbs_a_persistent_shift_but_the_anchor_does_not(fx):
    runs = history(fx)
    accept(fx, 40, 1)
    last = runs[-1]
    assert rolling(fx, last)[0] == "unchanged"
    a = anchored(fx, last)
    assert (a["status"], a["reason"]) == ("regressed", "anchor_regressed")
    assert a["since_ts"].startswith(stamp(45)[:10])
    assert a["n_exceeding_runs"] == NIGHTS - 45
    assert a["days_exceeding"] == NIGHTS - 1 - 45
    assert 1.10 < a["current_value"] / a["anchor_value"] < 1.14


def test_rolling_goes_quiet_between_night_55_and_75(fx):
    runs = history(fx)
    accept(fx, 40, 1)
    states = [rolling(fx, r)[0] for r in runs]
    assert states[45] == "suspected" and states[46] == "regressed"
    quiet = next(i for i in range(47, NIGHTS) if states[i] in ("unchanged", "improved"))
    assert 50 <= quiet <= 75
    assert all(
        anchored(fx, runs[i])["status"] == "regressed" for i in range(46, NIGHTS)
    )


def test_first_exceeding_night_is_suspected_single_run_then_regressed(fx):
    runs = history(fx)
    accept(fx, 40, 1)
    first = anchored(fx, runs[45])
    assert (first["status"], first["reason"]) == ("suspected", "single_run")
    assert first["n_exceeding_runs"] == 1 and first["days_exceeding"] == 0
    second = anchored(fx, runs[46])
    assert (second["status"], second["reason"]) == ("regressed", "anchor_regressed")
    assert second["n_exceeding_runs"] == 2


def test_accepting_a_new_anchor_flips_the_verdict(fx):
    runs = history(fx)
    accept(fx, 40, 1)
    assert anchored(fx, runs[-1])["status"] == "regressed"
    accept(fx, 100, 2)
    a = anchored(fx, runs[-1])
    assert (a["status"], a["reason"], a["version"]) == (
        "unchanged",
        "anchor_unchanged",
        "2",
    )
    assert a["n_exceeding_runs"] == 0 and a["since_ts"] is None
    assert anchored(fx, runs[100])["status"] == "no_anchor"
    assert anchored(fx, runs[60])["status"] == "no_anchor"


def test_a_value_carried_into_a_new_version_keeps_its_own_time_and_streak(fx):
    runs = history(fx)
    accept(fx, 40, 1)
    accept(fx, 100, 2, carry={"op-beta": 40})
    beta = ("op-beta", "kernel_mean_ms")
    b = anchored(fx, runs[-1], beta)
    assert (b["status"], b["version"], b["n_exceeding_runs"]) == (
        "regressed",
        "2",
        NIGHTS - 45,
    )
    a = anchored(fx, runs[-1])
    assert (a["status"], a["version"]) == ("unchanged", "2")


def test_a_further_shift_after_the_new_anchor_is_regressed_against_it(fx):
    history(fx, nights=100)
    accept(fx, 60, 1)
    more = [fx.run(slow={n: 0.30 for n, *_ in BENCHMARKS}) for _ in range(3)]
    assert [anchored(fx, r)["status"] for r in more] == [
        "suspected",
        "regressed",
        "regressed",
    ]
    accept(fx, 101, 2)
    assert [anchored(fx, r)["status"] for r in more] == [
        "no_anchor",
        "no_anchor",
        "unchanged",
    ]


# --- scope and lifecycle of the anchor ----------------------------------------------------


def test_no_anchor_means_no_anchor_for_every_run(fx):
    runs = history(fx, nights=30)
    assert {anchored(fx, r)["status"] for r in runs} == {"no_anchor"}
    assert {anchored(fx, r)["reason"] for r in runs} == {"no_anchor"}


def test_runs_up_to_the_anchor_time_are_not_compared(fx):
    runs = history(fx, nights=60)
    accept(fx, 50, 1)
    assert anchored(fx, runs[50])["status"] == "no_anchor"
    assert anchored(fx, runs[51])["status"] != "no_anchor"


def test_a_reference_that_is_not_flagged_as_an_anchor_is_ignored(fx):
    runs = history(fx, nights=60)
    accept(fx, 40, 1, anchor="")
    assert anchored(fx, runs[-1])["status"] == "no_anchor"
    assert fx.db.rows("SELECT count() n FROM v_benchmark_anchor") == [{"n": 0}]


def test_a_retired_or_untagged_or_unpublished_version_is_no_anchor(fx):
    runs = history(fx, nights=60)
    accept(fx, 40, 1, tag="ref-retired")
    assert anchored(fx, runs[-1])["status"] == "no_anchor"
    accept(fx, 41, 2, state="pending", current=True)
    assert anchored(fx, runs[-1])["status"] == "no_anchor"


def test_a_superseded_version_stops_anchoring(fx):
    runs = history(fx, nights=60)
    accept(fx, 30, 1)
    accept(fx, 50, 2)
    assert {
        r["version"] for r in fx.db.rows("SELECT version FROM v_benchmark_anchor")
    } == {"2"}
    assert anchored(fx, runs[40])["status"] == "no_anchor"


def test_rollback_to_the_older_version_re_anchors_on_it(fx):
    runs = history(fx, nights=100)
    accept(fx, 40, 1)
    accept(fx, 90, 2)
    assert anchored(fx, runs[-1])["status"] == "unchanged"
    fx.db.insert("artifact_tags", {"ts": "2027-03-01 00:00:00", "tag": "ref-current", "tag_family": "reference", "artifact_id": ref_id(1),
                                   "refs": [], "published_refs": [], "props": {"ref.key": REF_KEY}})  # fmt: skip
    assert anchored(fx, runs[-1])["status"] == "regressed"


def test_the_anchor_only_covers_the_benchmarks_in_its_value_rows(fx):
    runs = history(fx, nights=60)
    accept(fx, 40, 1, only=["op-alpha"])
    assert anchored(fx, runs[-1])["status"] == "regressed"
    assert (
        anchored(fx, runs[-1], ("op-beta", "kernel_mean_ms"))["status"] == "no_anchor"
    )


def test_a_job_specific_anchor_beats_the_job_less_one_and_other_jobs_use_the_job_less(
    fx,
):
    f = fx
    f.register_benchmarks()
    for n in range(40):
        f.run()
        f.run(job="synth-weekly")
    # job-less anchor at the nightly level; a weekly-specific anchor taken after a +50 % level change
    accept(f, 30, 1)
    for _ in range(3):
        f.run(slow={n: 0.5 for n, *_ in BENCHMARKS}, job="synth-weekly")
    last_weekly = f.run(slow={n: 0.5 for n, *_ in BENCHMARKS}, job="synth-weekly")
    last_nightly = f.run()
    assert anchored(f, last_weekly)["status"] == "regressed"
    assert anchored(f, last_nightly)["status"] == "unchanged"
    accept(f, 82, 2, source_job="synth-weekly", key=REF_KEY + ".weekly")
    cur = {
        r["version"]
        for r in f.db.rows(
            "SELECT version FROM v_benchmark_anchor WHERE source_job = 'synth-weekly'"
        )
    }
    assert cur == {"2"}
    assert anchored(f, last_weekly)["status"] == "unchanged"
    assert anchored(f, last_nightly)["version"] == "1"


# --- noise, direction, exclusions ---------------------------------------------------------


def test_a_small_shift_inside_the_floor_is_unchanged(fx):
    runs = history(fx, nights=60, slow={n: 0.01 for n, *_ in BENCHMARKS})
    accept(fx, 40, 1)
    assert {anchored(fx, r)["status"] for r in runs[50:]} == {"unchanged"}


def test_a_shift_the_other_way_is_improved_and_throughput_is_direction_aware(fx):
    runs = history(
        fx, shift_from=45, nights=60, slow={n: -0.15 for n, *_ in BENCHMARKS}
    )
    accept(fx, 40, 1)
    assert anchored(fx, runs[-1])["status"] == "improved"
    assert anchored(fx, runs[-1])["reason"] == "anchor_improved"
    assert (
        anchored(fx, runs[-1], ("op-delta", "output_throughput"))["status"]
        == "improved"
    )


def test_the_floor_widens_with_a_noisy_anchor(fx):
    fx.register_benchmarks()
    for n in range(40):
        fx.run()
    # an anchor of widely scattered samples: floor = 2 robust sigmas of its own MAD, above the 3 % policy floor
    noisy = str(uuid.uuid5(NS, "noisy"))
    fx.db.insert("benchmark_runs", *[
        {"ts": stamp(40), "run_id": noisy, "benchmark_id": bench_id("op-alpha"), "component": COMPONENT,
         "backend": BACKEND, "measurements": {"kernel_mean_ms": [8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 10.0]}, "iterations": 7,
         "props": {"source_file": "synth-nightly/r.json", "arch": ARCH, "run_utc": stamp(40), "run.purpose": "nightly"}}
    ])  # fmt: skip
    fx.next_run = 41
    accept(fx, 40, 1, only=["op-alpha"], src_run=noisy)
    a = anchored(fx, fx.run(slow={"op-alpha": 0.10}))
    assert a["floor_pct"] > 15 and a["status"] == "unchanged"


def test_a_pr_purpose_run_is_judged_but_capped_and_does_not_break_the_streak(fx):
    f = fx
    f.register_benchmarks()
    for n in range(45):
        f.run()
    accept(f, 40, 1)
    slow = {n: 0.12 for n, *_ in BENCHMARKS}
    first = f.run(slow=slow)
    pr = f.run(props={"run.purpose": "pr-check"})
    pr_slow = f.run(slow=slow, props={"run.purpose": "pr-check"})
    third = f.run(slow=slow)
    assert anchored(f, first)["status"] == "suspected"
    assert (anchored(f, pr)["status"], anchored(f, pr)["reason"]) == (
        "unchanged",
        "anchor_unchanged",
    )
    assert (anchored(f, pr_slow)["status"], anchored(f, pr_slow)["reason"]) == (
        "suspected",
        "purpose_not_baseline",
    )
    t = anchored(f, third)
    assert (t["status"], t["n_exceeding_runs"]) == ("regressed", 2)


def test_a_suspect_run_is_capped_and_an_invalidated_run_is_dropped(fx):
    f = fx
    f.register_benchmarks()
    for n in range(45):
        f.run()
    accept(f, 40, 1)
    slow = {n: 0.12 for n, *_ in BENCHMARKS}
    a, b, c = f.run(slow=slow), f.run(slow=slow), f.run(slow=slow)
    f.suspect(b)
    f.invalidate(a, n=2)
    assert anchored(f, a) is None
    sb = anchored(f, b)
    assert (sb["status"], sb["reason"]) == ("suspected", "suspect_data")
    sc = anchored(f, c)
    assert (sc["status"], sc["reason"]) == ("suspected", "single_run")
    assert sc["n_exceeding_runs"] == 1


def test_a_recovery_night_resets_the_streak(fx):
    f = fx
    f.register_benchmarks()
    for n in range(45):
        f.run()
    accept(f, 40, 1)
    slow = {n: 0.12 for n, *_ in BENCHMARKS}
    f.run(slow=slow)
    f.run(slow=slow)
    ok = f.run()
    again = f.run(slow=slow)
    again2 = f.run(slow=slow)
    assert anchored(f, ok)["status"] == "unchanged"
    assert (
        anchored(f, again)["status"] == "suspected"
        and anchored(f, again)["n_exceeding_runs"] == 1
    )
    a2 = anchored(f, again2)
    assert a2["status"] == "regressed" and a2["n_exceeding_runs"] == 2
    assert a2["since_ts"][:10] == (START + timedelta(days=48)).strftime("%Y-%m-%d")


# --- the existing views are untouched -----------------------------------------------------


def test_series_runs_agree_with_the_rolling_verdict_rows(fx):
    runs = history(fx, nights=70)
    fx.suspect(runs[20], n=1)
    fx.invalidate(runs[21], n=2)
    fx.run(props={"run.purpose": "pr-check"})
    a = fx.db.rows(
        "SELECT run_id, benchmark_id, metric, round(arrayReduce('median', cur), 4) AS v FROM v_benchmark_series_runs ORDER BY run_id, benchmark_id, metric"
    )
    b = fx.db.rows(
        "SELECT run_id, benchmark_id, metric, current_value AS v FROM v_benchmark_metric_verdicts ORDER BY run_id, benchmark_id, metric"
    )
    assert a == b and len(a) == 4 * 70


def test_the_anchored_view_is_the_same_with_join_use_nulls():
    outs = []
    for nulls in (False, True):
        d = Db(nulls=nulls)
        f = Fixture(d)
        history(f, nights=70)
        accept(
            f, 40, 1, only=["op-alpha", "op-delta"]
        )  # op-beta and op-gamma have no anchor
        outs.append(
            d.rows(
                "SELECT * FROM v_benchmark_anchored_verdicts ORDER BY run_id, name, metric"
            )
        )
        d.s.close()
    assert outs[0] == outs[1] and len(outs[0]) == 4 * 70
    free = [r for r in outs[0] if r["name"] == "op-beta"]
    assert {r["status"] for r in free} == {"no_anchor"}
    assert {
        (
            r["anchor_value"],
            r["delta_pct"],
            r["worse_pct"],
            r["floor_pct"],
            r["version"],
        )
        for r in free
    } == {(0, 0, 0, 0, "")}


def test_a_streak_starting_on_an_ineligible_run_has_no_start_until_an_eligible_one(fx):
    f = fx
    f.register_benchmarks()
    for n in range(45):
        f.run()
    accept(f, 40, 1)
    slow = {n: 0.12 for n, *_ in BENCHMARKS}
    pr = f.run(slow=slow, props={"run.purpose": "pr-check"})
    first = f.run(slow=slow)
    p, e = anchored(f, pr), anchored(f, first)
    assert (p["since_ts"], p["n_exceeding_runs"], p["days_exceeding"]) == (None, 0, 0)
    assert (e["since_ts"][:10], e["n_exceeding_runs"], e["days_exceeding"]) == (
        stamp(46)[:10],
        1,
        0,
    )


def test_the_floor_is_computed_from_the_unrounded_anchor_on_tiny_values(fx):
    f = fx
    f.register_benchmarks()
    noisy = [
        {"ts": stamp(n), "run_id": str(uuid.uuid5(NS, f"tiny:{n}")), "benchmark_id": bench_id("op-alpha"), "component": COMPONENT,
         "backend": BACKEND, "measurements": {"kernel_mean_ms": [3.00e-5 * (1 + 0.005 * k) for k in range(5)]}, "iterations": 5,
         "props": {"source_file": "synth-nightly/r.json", "arch": ARCH, "run_utc": stamp(n), "run.purpose": "nightly", "run.repeat": str(r)}}
        for n in range(30)
        for r in range(3)
    ]  # fmt: skip
    f.db.insert("benchmark_runs", *noisy)
    accept(f, 10, 1, only=["op-alpha"], src_run=str(uuid.uuid5(NS, "tiny:10")))
    a = anchored(f, str(uuid.uuid5(NS, "tiny:29")))
    assert (a["status"], a["anchor_value"]) == (
        "unchanged",
        0.0,
    )  # the displayed value is rounded; the maths is not
    row = f.db.rows("SELECT anchor_value FROM v_benchmark_anchor")[0]
    assert 2.9e-5 < row["anchor_value"] < 3.2e-5


def test_an_anchor_of_fewer_than_three_samples_is_no_anchor(fx):
    runs = history(fx, nights=60)
    accept(fx, 40, 1)
    thin = fx.db.rows("SELECT n_samples FROM v_benchmark_anchor LIMIT 1")[0][
        "n_samples"
    ]
    assert thin >= 3
    fx.db.insert("benchmark_runs", {"ts": stamp(41), "run_id": "bbbbbbbb-0000-5000-8000-0000000000aa", "benchmark_id": bench_id("op-alpha"),
                                    "component": COMPONENT, "backend": BACKEND, "measurements": {"kernel_mean_ms": [10.0, 10.1]}, "iterations": 2,
                                    "props": {"source_file": "synth-nightly/r.json", "arch": ARCH, "run_utc": stamp(41)}})  # fmt: skip
    accept(
        fx,
        41,
        2,
        key=REF_KEY + ".thin",
        only=["op-alpha"],
        src_run="bbbbbbbb-0000-5000-8000-0000000000aa",
        source_job="synth-nightly",
    )
    # the job-specific anchor has 2 samples: unusable, so the job-less anchor of version 1 still applies
    a = anchored(fx, runs[-1])
    assert (a["status"], a["version"]) == ("regressed", "1")


def test_a_job_specific_anchor_applies_only_after_its_own_time(fx):
    f = fx
    f.register_benchmarks()
    for n in range(40):
        f.run()
        f.run(job="synth-weekly")
    accept(f, 30, 1)
    early = f.run(job="synth-weekly")
    for _ in range(3):
        f.run(slow={n: 0.5 for n, *_ in BENCHMARKS}, job="synth-weekly")
    accept(f, 83, 2, source_job="synth-weekly", key=REF_KEY + ".weekly")
    assert anchored(f, early)["version"] == "1"
    assert anchored(f, run_uuid(79))["version"] == "1"
    assert (
        anchored(f, run_uuid(83))["version"] == "1"
    )  # the run the specific anchor was taken from is not compared with it
    late = f.run(slow={n: 0.5 for n, *_ in BENCHMARKS}, job="synth-weekly")
    assert (anchored(f, late)["version"], anchored(f, late)["status"]) == (
        "2",
        "unchanged",
    )


def test_a_measured_row_sharing_the_reference_run_id_is_not_part_of_the_anchor(fx):
    runs = history(fx, nights=60)
    accept(fx, 40, 1)
    before = fx.db.rows(
        "SELECT anchor_value, n_samples FROM v_benchmark_anchor WHERE metric = 'kernel_mean_ms' AND benchmark_id = toUUID('%s')"
        % bench_id("op-alpha")
    )
    fx.db.insert("benchmark_runs", {"ts": stamp(40), "run_id": ref_run(1), "benchmark_id": bench_id("op-alpha"), "component": COMPONENT,
                                    "backend": BACKEND, "measurements": {"kernel_mean_ms": [900.0, 901.0, 902.0]}, "iterations": 3,
                                    "props": {"source_file": "synth-nightly/r.json", "arch": ARCH}})  # fmt: skip
    after = fx.db.rows(
        "SELECT anchor_value, n_samples FROM v_benchmark_anchor WHERE metric = 'kernel_mean_ms' AND benchmark_id = toUUID('%s')"
        % bench_id("op-alpha")
    )
    assert after == before
    assert anchored(fx, runs[-1])["status"] == "regressed"


def test_a_job_less_anchor_of_fewer_than_three_samples_is_no_anchor(fx):
    runs = history(fx, nights=60)
    fx.db.insert("benchmark_runs", {"ts": stamp(41), "run_id": "bbbbbbbb-0000-5000-8000-0000000000bb", "benchmark_id": bench_id("op-alpha"),
                                    "component": COMPONENT, "backend": BACKEND, "measurements": {"kernel_mean_ms": [10.0, 10.1]}, "iterations": 2,
                                    "props": {"source_file": "synth-nightly/r.json", "arch": ARCH, "run_utc": stamp(41)}})  # fmt: skip
    accept(fx, 41, 1, only=["op-alpha"], src_run="bbbbbbbb-0000-5000-8000-0000000000bb")
    assert anchored(fx, runs[-1])["status"] == "no_anchor"


def test_an_anchor_whose_median_is_zero_is_no_anchor_not_infinite(fx):
    runs = history(fx, nights=60)
    fx.db.insert("benchmark_runs", {"ts": stamp(41), "run_id": "bbbbbbbb-0000-5000-8000-0000000000cc", "benchmark_id": bench_id("op-alpha"),
                                    "component": COMPONENT, "backend": BACKEND, "measurements": {"kernel_mean_ms": [-2.0, -1.0, 1.0, 2.0]}, "iterations": 4,
                                    "props": {"source_file": "synth-nightly/r.json", "arch": ARCH, "run_utc": stamp(41)}})  # fmt: skip
    accept(fx, 41, 1, only=["op-alpha"], src_run="bbbbbbbb-0000-5000-8000-0000000000cc")
    a = anchored(fx, runs[-1])
    assert (a["status"], a["anchor_value"]) == ("no_anchor", 0)


def test_rows_of_one_series_in_one_version_are_pooled_with_the_latest_time(fx):
    f = fx
    runs = history(f, nights=60)
    f.db.insert("benchmark_runs", {"ts": stamp(41), "run_id": "bbbbbbbb-0000-5000-8000-0000000000aa", "benchmark_id": bench_id("op-alpha"),
                                   "component": COMPONENT, "backend": BACKEND, "measurements": {"kernel_mean_ms": [10.0, 10.1]}, "iterations": 2,
                                   "props": {"source_file": "synth-nightly/r.json", "arch": ARCH, "run_utc": stamp(41)}})  # fmt: skip
    accept(f, 41, 1, only=["op-alpha"], src_run="bbbbbbbb-0000-5000-8000-0000000000aa")
    f.db.insert("benchmark_runs", {"ts": stamp(42), "run_id": ref_run(1), "benchmark_id": bench_id("op-alpha"), "component": COMPONENT,
                                   "backend": BACKEND, "measurements": {"kernel_mean_ms": [10.2, 10.3]}, "iterations": 2,
                                   "props": {"run.ref_only": "1", "run.record_type": "reference", "anchor.as_of": stamp(42)}})  # fmt: skip
    a = f.db.rows(
        "SELECT n_samples, toString(anchor_from) AS anchor_from FROM v_benchmark_anchor"
    )[0]
    assert (a["n_samples"], a["anchor_from"]) == (4, stamp(42))
    assert anchored(f, runs[-1])["status"] == "regressed"
    assert anchored(f, runs[42])["status"] == "no_anchor"


def test_a_reference_registered_with_an_arch_alias_still_anchors(fx):
    runs = history(fx, nights=60)
    accept(fx, 40, 1)
    fx.db.insert("artifacts", {"ts": "2027-01-01 00:00:00", "artifact_id": ref_id(1), "component": "reference", "arch": "amd64", "kind": "generic",
                               "artifact_name": REF_KEY, "origin": "built", "props": {"ref.key": REF_KEY}})  # fmt: skip
    assert anchored(fx, runs[-1])["status"] == "regressed"


def test_two_current_anchors_for_one_series_resolve_to_the_newest_registered(fx):
    runs = history(fx, nights=60)
    accept(fx, 30, 1, key=REF_KEY + ".a")
    accept(fx, 40, 2, key=REF_KEY + ".b")
    assert {
        r["version"] for r in fx.db.rows("SELECT version FROM v_benchmark_anchor")
    } == {"2"}
    assert anchored(fx, runs[-1])["version"] == "2"
    assert (
        anchored(fx, runs[35])["status"] == "no_anchor"
    )  # v1 (from night 30) is not in force once v2 is the current one


def test_a_job_specific_anchor_whose_median_is_zero_is_no_anchor_too(fx):
    runs = history(fx, nights=60)
    fx.db.insert("benchmark_runs", {"ts": stamp(41), "run_id": "bbbbbbbb-0000-5000-8000-0000000000dd", "benchmark_id": bench_id("op-alpha"),
                                    "component": COMPONENT, "backend": BACKEND, "measurements": {"kernel_mean_ms": [-2.0, -1.0, 1.0, 2.0]}, "iterations": 4,
                                    "props": {"source_file": "synth-nightly/r.json", "arch": ARCH, "run_utc": stamp(41)}})  # fmt: skip
    accept(
        fx,
        41,
        1,
        only=["op-alpha"],
        src_run="bbbbbbbb-0000-5000-8000-0000000000dd",
        source_job="synth-nightly",
    )
    a = anchored(fx, runs[-1])
    assert (a["status"], a["anchor_value"]) == ("no_anchor", 0)


def test_a_noiseless_anchor_still_has_the_policy_floor(fx):
    f = fx
    f.register_benchmarks()

    def row(n, level):
        return {"ts": stamp(n), "run_id": str(uuid.uuid5(NS, f"flat:{n}")), "benchmark_id": bench_id("op-alpha"), "component": COMPONENT,
                "backend": BACKEND, "measurements": {"kernel_mean_ms": [level] * 5}, "iterations": 5,
                "props": {"source_file": "synth-nightly/r.json", "arch": ARCH, "run_utc": stamp(n), "run.purpose": "nightly"}}  # fmt: skip

    f.db.insert(
        "benchmark_runs", *[row(n, 10.0 if n < 20 else 10.2) for n in range(30)]
    )
    accept(f, 5, 1, only=["op-alpha"], src_run=str(uuid.uuid5(NS, "flat:5")))
    a = anchored(f, str(uuid.uuid5(NS, "flat:29")))
    assert (a["status"], a["floor_pct"]) == (
        "unchanged",
        3.0,
    )  # +2 % is inside the 3 % policy floor; MAD is 0


def test_an_ineligible_run_over_the_floor_does_not_make_the_next_eligible_one_regressed(
    fx,
):
    f = fx
    f.register_benchmarks()
    for n in range(45):
        f.run()
    accept(f, 40, 1)
    slow = {n: 0.12 for n, *_ in BENCHMARKS}
    f.run()  # eligible and fine
    f.run(
        slow=slow, props={"run.purpose": "pr-check"}
    )  # over the floor, but ineligible
    nxt = f.run(slow=slow)
    a = anchored(f, nxt)
    assert (a["status"], a["reason"], a["n_exceeding_runs"]) == (
        "suspected",
        "single_run",
        1,
    )


def test_an_ineligible_over_floor_run_sharing_a_position_does_not_make_a_later_one_regressed(
    fx,
):
    f = fx
    f.register_benchmarks()
    for n in range(45):
        f.run()
    accept(f, 40, 1)
    slow = {n: 0.12 for n, *_ in BENCHMARKS}
    f.run()  # eligible and fine
    f.run(
        slow=slow, props={"run.purpose": "pr-check"}
    )  # over the floor, but ineligible
    f.run()  # eligible and fine: shares its position with the ineligible run
    nxt = f.run(slow=slow)
    a = anchored(f, nxt)
    assert (a["status"], a["reason"], a["n_exceeding_runs"]) == (
        "suspected",
        "single_run",
        1,
    )

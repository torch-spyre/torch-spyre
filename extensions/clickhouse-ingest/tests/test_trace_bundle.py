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

"""trace_bundle: a neutral trace description becomes a benchmarks bundle the existing writer ingests, one rank per file."""

import copy
import json
import statistics

import pytest
import trace_fixture as tf
from spyre_clickhouse_ingest import bundle, offline, schema, trace_bundle
from spyre_clickhouse_ingest.writer import BenchmarkWriter, DryRunClient

AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"


def small():
    return tf.trace(steps=6, layers=2)


def test_one_file_per_rank_and_a_bundle_the_offline_check_accepts(tmp_path):
    names = trace_bundle.write_bundle(small(), tmp_path, AID, "x86_64")
    assert names == [f"rank{r}-rep0.json" for r in range(4)]
    meta = offline.check(tmp_path)
    assert (meta["kind"], meta["test_type"], meta["component"]) == (
        "benchmarks",
        "perf",
        "synth-trace",
    )


def test_rows_hold_aligned_arrays_roles_rules_summary_and_headline():
    files = trace_bundle.emit(small())
    step, *spans = files["rank2-rep0.json"]["benchmarks"]
    m, rp = step["measurements"], step["run_props"]
    assert step["props"] == {
        "model": "synth-model",
        "world_size": 4,
        "record_type": "step_trace",
    }
    assert (
        len(m["step_kind"]) == len(m["sig_total"]) == len(m["cold"]) == 6
        and m["step_kind"][0] == 0.0
    )
    assert (
        rp["run.rank"],
        rp["run.world_size"],
        rp["role.total"],
        rp["agg.sig_total"],
    ) == (2, 4, "sig_total", "max")
    assert "agg.sig_wait" not in rp
    assert m["summary.itl.sig_total"] == [statistics.mean(m["sig_total"][1:])]
    warm = [
        max(tf.step_values(r, s)["sig_total"] for r in range(4)) for s in range(2, 6)
    ]
    assert (
        m["headline.itl.sig_total"] == [pytest.approx(statistics.median(warm))]
        and "headline.ttft.sig_total" not in m
    )
    assert len(spans) == 6 and {s["props"]["record_type"] for s in spans} == {"span"}
    assert all(set(s["measurements"]) == set(trace_bundle.SPAN_KEYS) for s in spans)
    assert not any(
        k.startswith(("role.", "agg.")) for s in spans for k in s["run_props"]
    )


def test_every_rank_row_and_span_pattern_is_its_own_row_after_the_writer_folds_by_benchmark():
    c = DryRunClient()
    for name, body in trace_bundle.emit(small()).items():
        BenchmarkWriter.insert(
            c,
            "d",
            "synth-trace",
            "r",
            bundle.generic_entries(body["benchmarks"]),
            report_kind="benchmarks",
            source_file=name,
        )
    runs = c.rows[schema.BENCHMARK_RUNS.name]
    assert (
        len(runs) == 4 * 7
        and len({b["benchmark_id"] for b in c.rows[schema.BENCHMARKS.name]}) == 7
    )
    assert sorted({r["props"]["run.rank"] for r in runs}) == ["0", "1", "2", "3"]
    spans = [r for r in runs if "dur_ns" in r["measurements"]]
    assert {len(r["measurements"]["dur_ns"]) for r in spans} == {6 * 2}


def test_the_case_id_of_a_span_benchmark_is_the_step_benchmarks_id():
    files = trace_bundle.emit(small())
    step, span = files["rank0-rep0.json"]["benchmarks"][:2]
    entry = bundle.generic_entries([step])[0]
    from spyre_clickhouse_ingest.identity import BenchmarkId

    assert span["props"]["case_id"] == BenchmarkId.derive(
        "synth-trace", step["name"], [], entry["disc"], entry["disc_keys"]
    )


@pytest.mark.parametrize(
    "mutate,problem",
    [
        (lambda t: t.pop("roles"), "roles.total"),
        (lambda t: t["roles"].update(bogus="x"), "roles are"),
        (lambda t: t.update(ranks=[]), "at least one rank"),
        (lambda t: t["ranks"].append(copy.deepcopy(t["ranks"][0])), "appears twice"),
        (
            lambda t: t["ranks"][0].update(steps=t["ranks"][0]["steps"][:-1], spans=[]),
            "different step counts",
        ),
        (lambda t: t["ranks"][0]["steps"][0].update(kind="x"), "ttft or itl"),
        (
            lambda t: t["ranks"][0]["steps"][2]["values"].pop("sig_total"),
            "total signal",
        ),
        (lambda t: t.update(rank_rules={"sig_total": "p99"}), "must be one of"),
        (lambda t: t["ranks"][0]["spans"][0].pop("dur_ns"), "needs lane, name"),
        (
            lambda t: t["ranks"][0]["steps"][1]["values"].update(
                sig_total=float("nan")
            ),
            "finite",
        ),
        (lambda t: t["ranks"][0]["spans"][0].update(dur_ns=float("inf")), "finite"),
        (lambda t: t["ranks"][0].update(rank=-1), "rank must be an integer"),
        (lambda t: t["ranks"][0].update(rank=1.0), "rank must be an integer"),
        (lambda t: t["ranks"][0].update(prompt="../x"), "prompt must be"),
        (lambda t: t["ranks"][0].update(prompt="a b"), "prompt must be"),
        (lambda t: t["ranks"][0].update(prompt="a\nb"), "prompt must be"),
        (lambda t: t.update(rank_rules={"nope": "max"}), "no step has"),
        (lambda t: t["ranks"][0]["steps"][2].update(index=True), "unique non-negative"),
        (lambda t: t["ranks"][0]["spans"][0].update(step=99), "step and seq"),
        (
            lambda t: t["ranks"][1]["steps"][2].update(index=7),
            "different .kind, index. steps",
        ),
        (lambda t: t["case"]["props"].pop("world_size"), "world_size is required"),
        (
            lambda t: t["ranks"][0]["steps"][3]["values"].pop("sig_wait"),
            "lacks the wait signal",
        ),
        (
            lambda t: [s.update(index=1) for s in t["ranks"][0]["steps"][1:3]],
            "unique non-negative",
        ),
        (lambda t: t["ranks"][0]["steps"][2].update(index=-1), "unique non-negative"),
        (lambda t: t["ranks"][0]["steps"][2].update(cold=2), "cold is 0 or 1"),
        (
            lambda t: [s["values"].update(cold=1.0) for s in t["ranks"][0]["steps"]],
            "reserved",
        ),
        (
            lambda t: [
                s["values"].update(**{"summary.x": 1.0}) for s in t["ranks"][0]["steps"]
            ],
            "reserved",
        ),
        (lambda t: t["ranks"][0]["spans"][0].update(depth=300), "depth within"),
        (lambda t: t["ranks"][0]["spans"][0].update(seq=-1), "seq must be"),
        (lambda t: t["ranks"][0]["spans"][0].update(dur_ns=1e19), "dur_ns within"),
        (lambda t: t["ranks"][0]["spans"][0].update(dur_ns=-1), "dur_ns within"),
        (lambda t: t["ranks"][0]["spans"][0].update(layer=40000), "layer within"),
        (lambda t: t["ranks"][0]["spans"][0].update(depth=0.5), "integral"),
    ],
)
def test_a_malformed_trace_is_refused(mutate, problem):
    t = small()
    mutate(t)
    with pytest.raises(trace_bundle.TraceError, match=problem):
        trace_bundle.emit(t)


def test_a_signal_missing_in_one_step_is_refused():
    t = small()
    del t["ranks"][1]["steps"][3]["values"]["sig_aux"]
    with pytest.raises(trace_bundle.TraceError, match="missing in some step"):
        trace_bundle.emit(t)


def test_prompts_and_repeats_get_their_own_files_and_the_cli_reports_a_bad_trace(
    tmp_path, capsys
):
    t = small()
    t["ranks"][1].update(prompt="p00", repeat=1)
    assert "rank1-pp00-rep1.json" in trace_bundle.emit(t)
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps({"component": "x"}))
    assert (
        trace_bundle.main(
            [
                str(bad),
                "--out",
                str(tmp_path / "o"),
                "--artifact-id",
                AID,
                "--arch",
                "x86_64",
            ]
        )
        == 2
    )
    assert "roles.total" in capsys.readouterr().err


def test_prompts_may_have_different_step_counts_but_ranks_of_one_prompt_may_not():
    t = small()
    t["ranks"] = [dict(rk, prompt=p) for rk in t["ranks"] for p in ("p0", "p1")]
    for rk in t["ranks"]:
        if rk["prompt"] == "p1":
            rk["steps"] = rk["steps"][:4]
            rk["spans"] = []
    assert len(trace_bundle.emit(t)) == 8
    t["ranks"][0]["steps"] = t["ranks"][0]["steps"][:5]
    t["ranks"][0]["spans"] = []
    with pytest.raises(trace_bundle.TraceError, match="different step counts"):
        trace_bundle.emit(t)


def test_headline_pairs_steps_across_ranks_by_kind_and_index_not_position():
    t = tf.trace(steps=4, layers=1, with_spans=False)
    t["rank_rules"] = {"sig_total": "mean"}
    for rk in t["ranks"]:
        rk["steps"] = rk["steps"][:1] + [dict(s, cold=0) for s in rk["steps"][1:3]]
        rk["steps"][1].update(
            index=1, values=dict(rk["steps"][1]["values"], sig_total=1.0)
        )
        rk["steps"][2].update(
            index=2, values=dict(rk["steps"][2]["values"], sig_total=100.0)
        )
    t["ranks"][1]["steps"][1:3] = [t["ranks"][1]["steps"][2], t["ranks"][1]["steps"][1]]
    t["ranks"] = t["ranks"][:2]
    t["case"]["props"]["world_size"] = 2
    for rk in t["ranks"][:1]:
        pass
    body = trace_bundle.emit(t)["rank0-rep0.json"]["benchmarks"][0]["measurements"]
    assert body["headline.itl.sig_total"] == [50.5]


def test_write_bundle_refuses_a_directory_that_already_holds_results(tmp_path):
    trace_bundle.write_bundle(small(), tmp_path, AID, "x86_64")
    with pytest.raises(trace_bundle.TraceError, match="not empty"):
        trace_bundle.write_bundle(small(), tmp_path, AID, "x86_64")


def test_the_cli_reports_unreadable_and_oversized_input(tmp_path, capsys, monkeypatch):
    bad = tmp_path / "t.json"
    bad.write_text(
        json.dumps(
            {
                "component": "x",
                "case": {"name": "c"},
                "roles": {"total": "t"},
                "ranks": [{"rank": 0, "steps": [1]}],
            }
        )
    )
    assert (
        trace_bundle.main(
            [
                str(bad),
                "--out",
                str(tmp_path / "o"),
                "--artifact-id",
                AID,
                "--arch",
                "x86_64",
            ]
        )
        == 2
    )
    monkeypatch.setattr(trace_bundle, "MAX_TRACE_BYTES", 3)
    assert (
        trace_bundle.main(
            [
                str(bad),
                "--out",
                str(tmp_path / "o2"),
                "--artifact-id",
                AID,
                "--arch",
                "x86_64",
            ]
        )
        == 2
    )
    assert "larger than" in capsys.readouterr().err

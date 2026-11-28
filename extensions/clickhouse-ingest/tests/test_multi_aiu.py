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

"""spyre-perf-suite's Multi-AIU report XMLs: detected, parsed per card, and written to spyre_v2."""

import xml.etree.ElementTree as etree

import json

import pytest

from spyre_clickhouse_ingest import multi_aiu, results, schema
from spyre_clickhouse_ingest.identity import BenchmarkId

RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"


def _case(name, ms, **tags):
    props = "".join(f'<property name="tag" value="{k}__{v}"/>' for k, v in tags.items())
    return (
        f'<testcase classname="spyre_perf_suite.benchmark" name="{name}" time="{ms}">'
        f"<properties>{props}</properties></testcase>"
    )


def _xml(suites_name, suite_name, cases, props=""):
    return (
        f'<?xml version="1.0" ?><testsuites name="{suites_name}">'
        f'<testsuite name="{suite_name}" tests="{len(cases)}" time="1.0"'
        ' timestamp="2026-11-10T11:12:59+00:00">'
        f"<properties>{props}</properties>{''.join(cases)}</testsuite></testsuites>"
    )


RANK_PROPS = (
    '<property name="card_rank" value="{r}"/>'
    '<property name="source_report" value="/host/path/report.txt"/>'
    '<property name="pid" value="41000"/><property name="workload_id" value="3"/>'
)


def _latency(rank):
    return _xml(
        "spyre-perf-suite",
        f"multi_aiu_decode_trace_rank_{rank}",
        [
            _case(
                f"summary_mean_rank_{rank}_ttft_torch_spyre_ms",
                1362.5 + rank,
                card_rank=rank, scope="summary_mean", metric="ttft_torch_spyre_ms",
            ),
            _case(
                f"step_rank_{rank}_itl_0001_n_flex",
                7,
                card_rank=rank, step="itl_0001", metric="n_flex",
            ),
        ],
        RANK_PROPS.format(r=rank),
    )  # fmt: skip


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture
def reports(tmp_path):
    return {
        "report_rank_0.xml": _write(tmp_path, "report_rank_0.xml", _latency(0)),
        "report_rank_1.xml": _write(tmp_path, "report_rank_1.xml", _latency(1)),
        "spyre_kernel_report_rank_0.xml": _write(
            tmp_path,
            "spyre_kernel_report_rank_0.xml",
            _xml(
                "spyre-kernel-suite",
                "spyre_kernel_rank_0",
                [
                    _case(
                        "kernel_rank_0_ttft_sdsc_softmax_fp16",
                        2.044,
                        card_rank=0,
                        step="ttft",
                        kernel="sdsc_softmax_fp16",
                    )
                ],
                RANK_PROPS.format(r=0),
            ),
        ),  # fmt: skip
        "cpu_kernel_report_rank_0.xml": _write(
            tmp_path,
            "cpu_kernel_report_rank_0.xml",
            _xml(
                "cpu-kernel-suite",
                "cpu_kernel_rank_0",
                [
                    _case(
                        "cpu_kernel_rank_0_ttft_aten__mm",
                        1.419,
                        card_rank=0,
                        step="ttft",
                        kernel="aten::mm",
                    )
                ],
                RANK_PROPS.format(r=0),
            ),
        ),  # fmt: skip
        "multi_aiu_summary.xml": _write(
            tmp_path,
            "multi_aiu_summary.xml",
            _xml(
                "spyre-multi-aiu-summary",
                "multi_aiu_4card_summary",
                [
                    _case(
                        "multi_card_mean_ttft_torch_spyre_ms_rank_0",
                        1362.5,
                        card_rank=0,
                        metric="ttft_torch_spyre_ms",
                    )
                ],
                '<property name="total_cards" value="4"/>',
            ),
        ),  # fmt: skip
    }


def _root(path):
    return etree.parse(str(path)).getroot()


def test_the_four_report_files_are_detected(reports):
    assert all(multi_aiu.is_multi_aiu_xml(_root(p)) for p in reports.values())


@pytest.mark.parametrize(
    "root",
    [
        # CI's own report.xml shares the suite name but has no card_rank
        '<testsuites name="spyre-perf-suite"><testsuite name="s">'
        '<testcase classname="spyre_perf_suite.benchmark" name="perf_add_wall_clock_ms_x" time="1"/>'
        "</testsuite></testsuites>",
        '<testsuites name="spyre-perf-suite"/>',
        '<testsuites><testsuite name="s"><testcase classname="T" name="t"/></testsuite></testsuites>',
    ],
)
def test_other_xml_is_not_taken(root):
    assert not multi_aiu.is_multi_aiu_xml(etree.fromstring(root))


def test_a_latency_report_is_one_benchmark_per_testcase_per_card(reports):
    mean, flex = multi_aiu.parse(reports["report_rank_1.xml"])
    assert (mean["name"], mean["backend"]) == (
        "multi_aiu/latency/summary_mean_ttft_torch_spyre_ms",
        "spyre",
    )
    assert mean["tags"] == [
        "card_rank__1", "metric__ttft_torch_spyre_ms", "scope__summary_mean",
    ]  # fmt: skip
    assert mean["measurements"] == {"duration_ms": [1363.5]}
    # a span count the generator carries in `time` is not a duration
    assert flex["measurements"] == {"n_flex": [7.0]}
    # host paths stay out of the run row; the trace facts go in
    assert mean["run_props"] == {
        "pid": "41000",
        "workload_id": "3",
        "num_cards": "1",
        "run_utc": "2026-11-10 11:12:59",
    }


def test_cpu_kernels_are_a_cpu_backend_and_spyre_kernels_a_spyre_one(reports):
    (cpu,) = multi_aiu.parse(reports["cpu_kernel_report_rank_0.xml"])
    (spyre,) = multi_aiu.parse(reports["spyre_kernel_report_rank_0.xml"])
    assert (cpu["backend"], cpu["props"]["kernel"]) == ("cpu", "aten::mm")
    assert (spyre["backend"], spyre["props"]["run_mode"]) == ("spyre", "spyre_kernel")


def test_each_card_is_its_own_benchmark(reports):
    def ids(name):
        return {
            e["name"]: BenchmarkId.derive("torch-spyre", e["name"], e["tags"], e["disc"], e["disc_keys"])
            for e in multi_aiu.parse(reports[name])
        }  # fmt: skip

    r0, r1 = ids("report_rank_0.xml"), ids("report_rank_1.xml")
    assert set(r0) == set(r1)
    assert all(r0[n] != r1[n] for n in r0)
    # and a re-ingest of the same card lands on the same ids
    assert ids("report_rank_0.xml") == r0


def test_the_run_size_comes_from_the_summary_and_is_part_of_every_benchmark(
    reports, tmp_path
):
    files = sorted(tmp_path.glob("*.xml"))
    assert multi_aiu.num_cards_of(files) == 4  # the summary's total_cards
    # a report read alone knows only its own card
    assert multi_aiu.num_cards_of([reports["report_rank_1.xml"]]) == 1

    four = multi_aiu.parse(reports["report_rank_1.xml"], 4)
    one = multi_aiu.parse(reports["report_rank_1.xml"], 1)
    assert {e["props"]["config_name"] for e in four} == {"4-card"}
    assert {e["props"]["num_cards"] for e in four} == {"4"}
    assert four[0]["run_props"]["num_cards"] == "4"

    def ident(e):
        return BenchmarkId.derive(
            "torch-spyre", e["name"], e["tags"], e["disc"], e["disc_keys"]
        )

    # card 1 of a 4-card run and of a 1-card run are not the same series
    assert all(ident(a) != ident(b) for a, b in zip(four, one))


def test_a_declared_card_count_beats_the_ranks_that_happen_to_be_present(tmp_path):
    # A strict producer writes `num_cards` on every rank file (from its `# cards: N, rank: R`
    # header), so a 4-card run with only rank 1 on disk is still a 4-card run, not a 1-card one.
    rank_1 = _write(
        tmp_path,
        "report_rank_1.xml",
        _xml(
            "spyre-perf-suite",
            "multi_aiu_decode_trace_rank_1",
            [_case("summary_mean_rank_1_ttft_torch_spyre_ms", 1363.5, card_rank=1)],
            RANK_PROPS.format(r=1) + '<property name="num_cards" value="4"/>',
        ),
    )
    assert multi_aiu.num_cards_of([rank_1]) == 4
    (entry,) = multi_aiu.parse(rank_1)
    assert (entry["props"]["config_name"], entry["props"]["num_cards"]) == (
        "4-card",
        "4",
    )


def test_the_model_is_read_from_the_xml_and_is_part_of_every_benchmark(tmp_path):
    rank_0 = _write(
        tmp_path,
        "report_rank_0.xml",
        _xml(
            "spyre-perf-suite",
            "multi_aiu_decode_trace_rank_0",
            [_case("summary_mean_rank_0_ttft_torch_spyre_ms", 1362.5, card_rank=0)],
            RANK_PROPS.format(r=0) + '<property name="model" value="org/m-8b"/>',
        ),
    )
    assert multi_aiu.model_of([rank_0]) == "org/m-8b"
    (entry,) = multi_aiu.parse(rank_0, 4)
    assert "model__org/m-8b" in entry["tags"]
    assert (entry["props"]["model"], entry["run_props"]["model"]) == ("org/m-8b",) * 2
    # a caller's model is only a fallback for a file that names none ...
    (other,) = multi_aiu.parse(rank_0, 4, "org/other")
    assert other["props"]["model"] == "org/other"

    def ident(e):
        return BenchmarkId.derive(
            "torch-spyre", e["name"], e["tags"], e["disc"], e["disc_keys"]
        )

    # ... and two models are two series
    assert ident(entry) != ident(other)


def test_no_model_anywhere_leaves_it_out(reports):
    assert multi_aiu.model_of(list(reports.values())) == ""
    entries = multi_aiu.parse(reports["report_rank_0.xml"])
    assert all("model" not in e["props"] for e in entries)
    assert not any(t.startswith("model__") for e in entries for t in e["tags"])


def test_a_report_with_no_timestamp_or_cards_still_parses(tmp_path):
    bare = _write(
        tmp_path,
        "bare.xml",
        '<testsuites name="spyre-kernel-suite"><testsuite name="k"><properties/>'
        '<testcase classname="c" name="kernel_x" time="1.5"/></testsuite></testsuites>',
    )
    (entry,) = multi_aiu.parse(bare)
    assert entry["props"]["config_name"] == ""
    assert entry["run_props"] == {}


class RecordingClient:
    """A v2 database with every table and column present; records what is inserted."""

    def __init__(self):
        self.rows = {}

    def command(self, sql, parameters=None):
        return 1

    def query(self, sql, parameters=None):
        columns = [(c,) for t in schema.Table.__subclasses__() for c in t.columns]
        if "system.columns" in sql:
            rows = columns
        elif "count()" in sql:
            seen = [
                r
                for r in self.rows.get("benchmark_runs", [])
                if r["props"].get("source_file") == (parameters or {}).get("sf")
            ]
            rows = [(len(seen),)]
        else:
            rows = []

        class R:
            result_rows = rows

        return R()

    def insert(self, table, rows, column_names=None, database=None, **_):
        self.rows.setdefault(table, []).extend(dict(zip(column_names, r)) for r in rows)


def _ingest(monkeypatch, where, *extra):
    ch = RecordingClient()
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setattr(results, "get_client", lambda: ch)
    argv = ["--schema", "v2", "--strict", "--run-id", RUN, "--trigger-type", "perf",
            "--arch", "x86_64", *extra]  # fmt: skip
    results.main(argv + ["--xml-dir" if where.is_dir() else "--xml-file", str(where)])
    return ch


def test_results_ingests_multi_aiu_xml_instead_of_parsing_zero_records(
    monkeypatch, reports, capsys
):
    # --trigger-type perf exits non-zero when a perf XML yields no records; it must not here.
    ch = _ingest(monkeypatch, reports["report_rank_0.xml"])
    out = capsys.readouterr().out
    assert "Detected: multi-AIU report XML" in out
    runs = ch.rows["benchmark_runs"]
    assert len(runs) == 2
    assert {r["props"]["report_kind"] for r in runs} == {multi_aiu.REPORT_KIND}
    assert {r["props"]["source_file"] for r in runs} == {"report_rank_0.xml"}
    assert {"duration_ms", "n_flex"} == {k for r in runs for k in r["measurements"]}
    assert len(ch.rows["benchmarks"]) == 2


def test_every_report_file_of_one_run_lands_under_one_run_id(
    monkeypatch, reports, tmp_path
):
    ch = _ingest(monkeypatch, tmp_path)
    runs = ch.rows["benchmark_runs"]
    assert len(runs) == 7  # 2 + 2 latency, 1 + 1 kernel, 1 summary
    assert len({r["run_id"] for r in runs}) == 1
    assert {r["backend"] for r in runs} == {"spyre", "cpu"}


def test_results_model_flag_names_the_model_when_the_xml_does_not(
    monkeypatch, reports, tmp_path
):
    ch = _ingest(monkeypatch, tmp_path, "--model", "org/m-8b")
    runs = ch.rows["benchmark_runs"]
    assert {r["props"]["model"] for r in runs} == {"org/m-8b"}


def test_a_model_the_xml_declares_beats_the_model_flag(monkeypatch, tmp_path):
    _write(
        tmp_path,
        "report_rank_0.xml",
        _xml(
            "spyre-perf-suite",
            "multi_aiu_decode_trace_rank_0",
            [_case("summary_mean_rank_0_ttft_torch_spyre_ms", 1362.5, card_rank=0)],
            RANK_PROPS.format(r=0) + '<property name="model" value="org/declared"/>',
        ),
    )
    ch = _ingest(monkeypatch, tmp_path, "--model", "org/flag")
    assert {r["props"]["model"] for r in ch.rows["benchmark_runs"]} == {"org/declared"}


def test_the_workload_description_is_kept_as_run_props_not_identity(
    monkeypatch, reports, tmp_path
):
    wl = tmp_path / "RUN.json"
    wl.write_text(json.dumps({
        "model": "org/m-8b", "num_cards": 4, "decode_steps": 8, "prompt": "The capital of France is",
        "versions": {"torch": "2.14.1", "torch-spyre": "0.6.0"}, "empty": "", "dtype": "float16",
        "spyre_perf_suite": {"commit": "a5cbc21", "uncommitted_files": 1},
    }))  # fmt: skip
    plain = _ingest(monkeypatch, tmp_path)
    described = _ingest(monkeypatch, tmp_path, "--workload-json", str(wl))
    props = described.rows["benchmark_runs"][0]["props"]
    assert props["workload.decode_steps"] == "8"
    assert props["workload.prompt"] == "The capital of France is"
    assert props["workload.versions.torch-spyre"] == "0.6.0"
    assert props["workload.spyre_perf_suite.commit"] == "a5cbc21"
    assert "workload.empty" not in props
    # the same benchmarks: a description never changes what a benchmark is
    ids = lambda ch: sorted(r["benchmark_id"] for r in ch.rows["benchmarks"])  # noqa: E731
    assert ids(plain) == ids(described)


def test_a_dry_run_writes_nothing(monkeypatch, reports, capsys):
    ch = _ingest(monkeypatch, reports["report_rank_0.xml"], "--dry-run")
    assert ch.rows == {}
    assert "row(s) -> spyre_v2.benchmark_runs" in capsys.readouterr().out


def _layers(rank, flex, rt, hw):
    """A latency report with the three device layers set to the given values."""
    cases = [
        _case(
            f"summary_mean_rank_{rank}_ttft_torch_spyre_ms",
            7636.95,
            card_rank=rank,
            scope="summary_mean",
            metric="ttft_torch_spyre_ms",
        ),  # fmt: skip
    ]
    for layer, ms in (("flex_rt", flex), ("aiu_rt", rt), ("aiu_hw", hw)):
        cases.append(
            _case(
                f"summary_mean_rank_{rank}_ttft_{layer}_ms",
                ms,
                card_rank=rank,
                scope="summary_mean",
                metric=f"ttft_{layer}_ms",
            )  # fmt: skip
        )
        cases.append(
            _case(
                f"step_rank_{rank}_ttft_{layer}_ms",
                ms,
                card_rank=rank,
                step="ttft",
                metric=f"{layer}_ms",
            )  # fmt: skip
        )
    return _xml(
        "spyre-perf-suite",
        f"multi_aiu_decode_trace_rank_{rank}",
        cases,
        RANK_PROPS.format(r=rank),
    )


@pytest.mark.parametrize(
    "layers, lacks",
    [
        ((0, 0, 0), True),  # no flexRoundTrip / aiuRoundtrip spans in the trace
        ((12.5, 10.25, 8.0), False),
        ((0, 0, 8.0), False),  # one layer measured is enough to trust the others' zeros
    ],
)
def test_all_zero_device_layers_mean_the_layers_were_not_measured(
    tmp_path, layers, lacks
):
    path = _write(tmp_path, "report_rank_0.xml", _layers(0, *layers))
    assert multi_aiu.lacks_device_spans(multi_aiu.parse(path)) is lacks


def test_a_report_with_no_device_layer_says_nothing_about_them(reports):
    # kernel reports and the summary have no flex_rt / aiu_rt / aiu_hw to be zero
    for name in ("spyre_kernel_report_rank_0.xml", "cpu_kernel_report_rank_0.xml"):
        assert not multi_aiu.lacks_device_spans(multi_aiu.parse(reports[name]))
    assert not multi_aiu.lacks_device_spans([])


def test_n_flex_is_not_a_device_layer(tmp_path):
    # n_flex counts spans and is 0 whenever there are none; it must not decide on its own
    path = _write(
        tmp_path,
        "report_rank_0.xml",
        _xml(
            "spyre-perf-suite",
            "multi_aiu_decode_trace_rank_0",
            [_case("step_rank_0_itl_0001_n_flex", 0, card_rank=0, step="itl_0001", metric="n_flex")],
            RANK_PROPS.format(r=0),
        ),
    )  # fmt: skip
    assert not multi_aiu.lacks_device_spans(multi_aiu.parse(path))


class _Verdicts:
    """results.insert_artifact_result and ensure, recording the verdicts written."""

    def __init__(self, monkeypatch):
        from types import SimpleNamespace

        self.written = []
        monkeypatch.setattr(
            results,
            "ensure",
            lambda *a, **kw: SimpleNamespace(
                dry_run=False,
                identity=SimpleNamespace(
                    artifact_id="81051064-eb94-5a50-a4bd-8445065e2d5d"
                ),
            ),
        )
        monkeypatch.setattr(
            results,
            "insert_artifact_result",
            lambda *a, **kw: self.written.append(kw) or True,
        )


def _ingest_with_artifact(monkeypatch, where):
    verdicts = _Verdicts(monkeypatch)
    _ingest(
        monkeypatch, where,
        "--artifact", "id:81051064-eb94-5a50-a4bd-8445065e2d5d",
        "--jenkins-run-key", "local#1", "--registry", "off",
    )  # fmt: skip
    return verdicts.written


def test_a_run_without_device_spans_is_not_recorded_as_passed(monkeypatch, tmp_path):
    _write(tmp_path, "report_rank_0.xml", _layers(0, 0, 0, 0))
    (verdict,) = _ingest_with_artifact(monkeypatch, tmp_path)
    assert (verdict["test_type"], verdict["state"]) == ("perf", "error")
    assert verdict["props"]["failure_reason"] == "infra_env"
    assert verdict["props"]["failure_subreason"] == "no_device_spans"
    assert "report_rank_0.xml" in verdict["props"]["failure_detail"]
    assert "flexRoundTrip" in verdict["props"]["failure_detail"]


def test_a_run_with_device_spans_is_recorded_as_passed(monkeypatch, tmp_path):
    _write(tmp_path, "report_rank_0.xml", _layers(0, 12.5, 10.25, 8.0))
    (verdict,) = _ingest_with_artifact(monkeypatch, tmp_path)
    assert (verdict["state"], "failure_reason" in verdict["props"]) == ("passed", False)


def test_one_card_missing_its_spans_fails_the_whole_run(monkeypatch, tmp_path):
    # the verdict is the run's, so a card with spans does not hide one without
    _write(tmp_path, "report_rank_0.xml", _layers(0, 12.5, 10.25, 8.0))
    _write(tmp_path, "report_rank_1.xml", _layers(1, 0, 0, 0))
    (verdict,) = _ingest_with_artifact(monkeypatch, tmp_path)
    assert verdict["state"] == "error"
    assert "report_rank_1.xml" in verdict["props"]["failure_detail"]


def test_the_leg_state_reads_an_incomplete_run_as_error():
    assert results._leg_state(0, 8) == "passed"
    assert results._leg_state(0, 8, "why") == "error"
    assert results._leg_state(0, 0) == "error"
    assert results._leg_state(2, 8) == "failed"

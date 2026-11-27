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

"""Model records are told apart from op records by the explicit model tag; granite_ is a fallback."""

import pytest

from spyre_clickhouse_ingest import results, sendnn
from spyre_clickhouse_ingest.identity import BenchmarkId


def _tc(name, tags, time="1.0", classname="benchmark"):
    props = "".join(f'<property name="tag" value="{t}"/>' for t in tags)
    return (
        f'<testcase classname="{classname}" name="{name}" time="{time}">'
        f"<properties>{props}</properties></testcase>"
    )


def _write(tmp_path, cases):
    p = tmp_path / "r.xml"
    p.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="s" '
        'timestamp="2026-10-08T00:00:00+00:00">'
        + "".join(cases)
        + "</testsuite></testsuites>"
    )
    return p


def _bench(op, extra, mode="decode_only"):
    tags = [f"op__{op}", f"mode__{mode}", *extra]
    return [
        _tc(f"perf_{op}_wall_clock_ms_x", tags, "2.0"),
        _tc(f"perf_{op}_spyre_ms_x", tags, "1.0"),
    ]


_EXPLICIT = [
    "model__gpt-neo-125m",
    "config__full_model",
    "batch_size__3",
    "prompt_length__96",
]


def test_benchmark_explicit_model_tags(tmp_path):
    op = "gpt-neo-125m_decode-mode_full_model"
    _, rows = results.parse_benchmark_xml(_write(tmp_path, _bench(op, _EXPLICIT)))
    assert len(rows) == 1
    row = rows[0]
    assert row["record_type"] == "model"
    assert row["operation_name"] == "gpt-neo-125m"
    assert row["config_name"] == "full_model"
    assert row["run_mode"] == "decode_only"
    assert (row["batch_size"], row["prompt_length"]) == (3, 96)
    assert row["input_shapes"] is None


def test_benchmark_model_tag_without_shape_leaves_nulls(tmp_path):
    op = "gpt-neo-125m_decode-mode_full_model"
    extra = ["model__gpt-neo-125m", "config__full_model"]
    _, rows = results.parse_benchmark_xml(_write(tmp_path, _bench(op, extra)))
    assert (rows[0]["batch_size"], rows[0]["prompt_length"]) == (None, None)


@pytest.mark.parametrize("extra", [["config__full_model"], ["config__null"]])
def test_benchmark_legacy_granite_fallback(tmp_path, extra):
    op = "granite_bs4_pl128_generate_full_model"
    _, rows = results.parse_benchmark_xml(_write(tmp_path, _bench(op, extra)))
    row = rows[0]
    assert row["record_type"] == "model"
    # no model recorded: the FMS run's own label, never the bare family "granite"
    assert row["operation_name"] == results.LEGACY_GRANITE_MODEL == "granite-8b-fms"
    assert (row["batch_size"], row["prompt_length"]) == (4, 128)


def test_benchmark_granite_with_model_tag_is_named_by_its_model(tmp_path):
    op = "granite_bs4_pl128_generate_full_model"
    extra = ["model__granite-test-1b", "batch_size__4", "prompt_length__128"]
    _, rows = results.parse_benchmark_xml(_write(tmp_path, _bench(op, extra)))
    assert rows[0]["operation_name"] == "granite-test-1b"
    assert (rows[0]["batch_size"], rows[0]["prompt_length"]) == (4, 128)


def test_two_granite_models_in_one_report_are_two_benchmarks(tmp_path):
    """hf-adapters' weekly list has granite-3.3-2b-instruct and granite-4.0-1b-base: one
    "granite" name gave them one benchmark id, and one model's numbers hid the other's."""
    cases = []
    for model in ("granite-3.3-2b-instruct", "granite-4.0-1b-base"):
        op = f"granite_{model}_bs1_pl512_prefill_only_full_model"
        cases += _bench(op, [f"model__{model}", "config__full_model"], "prefill_only")
    _, rows = results.parse_benchmark_xml(_write(tmp_path, cases))
    names = {r["operation_name"] for r in rows}
    assert names == {"granite-3.3-2b-instruct", "granite-4.0-1b-base"}
    entries = results._bench_entries(rows)
    ids = {
        BenchmarkId.derive(
            "torch-spyre", e["name"], e["tags"], e["disc"], e["disc_keys"]
        )
        for e in entries
    }
    assert len(ids) == 2


@pytest.mark.parametrize(
    "op, name",
    [
        # hf-adapters names carry the model even without a model tag
        (
            "granite_granite-3.3-2b-instruct_bs1_pl512_prefill_only_full_model",
            "granite-3.3-2b-instruct",
        ),
        (
            "granite_granite-4.0-1b-base_bs4_pl128_decode_only_one_block",
            "granite-4.0-1b-base",
        ),
        # the FMS run's names do not
        ("granite_bs1_pl512_prefill_only_full_model", "granite-8b-fms"),
    ],
)
def test_no_model_record_is_ever_named_just_granite(tmp_path, op, name):
    _, rows = results.parse_benchmark_xml(
        _write(tmp_path, _bench(op, ["config__full_model"]))
    )
    assert rows[0]["operation_name"] == name != "granite"


def test_sendnn_names_the_fms_granite_run_as_results_does():
    assert sendnn.LEGACY_GRANITE_MODEL == results.LEGACY_GRANITE_MODEL
    (rec,) = [r for r in [sendnn._section("Granite Config: bs1_pl512_full_model")] if r]
    assert rec["operation_name"] == results.LEGACY_GRANITE_MODEL


def test_benchmark_config_tag_alone_is_not_a_model(tmp_path):
    _, rows = results.parse_benchmark_xml(
        _write(tmp_path, _bench("gpt-neo-125m_x", ["config__full_model"]))
    )
    assert rows[0]["record_type"] == "op"


def test_benchmark_op_record_unchanged(tmp_path):
    cases = _bench("matmul", ["config__null"], "null")
    _, rows = results.parse_benchmark_xml(_write(tmp_path, cases))
    assert rows[0]["record_type"] == "op"
    assert rows[0]["operation_name"] == "matmul"
    assert rows[0]["run_mode"] == "op_benchmark"


def _kernel(op, extra):
    tags = [
        f"op__{op}",
        "kernel__matmul",
        "mode__decode_only",
        "metric__kernel_ms",
        *extra,
    ]
    return _tc(f"perf_{op}_kernel", tags, classname="kernel_benchmark")


def test_kernel_explicit_model_tags(tmp_path):
    op = "gpt-neo-125m_decode-mode_full_model"
    _, rows = results.parse_kernel_xml(_write(tmp_path, [_kernel(op, _EXPLICIT)]))
    r = rows[0]
    assert (r["record_type"], r["operation_name"]) == ("model", "gpt-neo-125m")
    assert (r["config_name"], r["run_mode"]) == ("full_model", "decode_only")
    assert (r["batch_size"], r["prompt_length"]) == (3, 96)


@pytest.mark.parametrize(
    "op, extra, rtype, name",
    [
        (
            "granite_bs4_pl128_generate_full_model",
            ["config__full_model"],
            "model",
            "granite-8b-fms",
        ),
        ("matmul", ["config__null"], "op", "matmul"),
    ],
)
def test_kernel_legacy_and_op_records(tmp_path, op, extra, rtype, name):
    _, rows = results.parse_kernel_xml(_write(tmp_path, [_kernel(op, extra)]))
    assert rows[0]["record_type"] == rtype
    assert rows[0]["operation_name"] == name


def test_bench_entries_date_the_run_by_the_report_timestamp(tmp_path):
    """run_utc is the suite timestamp, so a late ingest still lands in the week it was measured."""
    _, rows = results.parse_benchmark_xml(
        _write(tmp_path, _bench("gpt-neo-125m_decode-mode_full_model", _EXPLICIT))
    )
    (entry,) = results._bench_entries(rows)
    assert entry["run_props"] == {"run_utc": "2026-10-08 00:00:00"}


def test_bench_entries_without_a_timestamp_carry_no_run_utc():
    (entry,) = results._bench_entries(
        [{"operation_name": "x", "total_duration_ms": 1.0}]
    )
    assert "run_props" not in entry


def test_how_a_model_ran_is_a_run_prop_never_identity(tmp_path):
    """num_cards and the PVC storage a runner tags per model land in the row's run props; the
    same model on another card or with a bigger cache stays the same benchmark."""
    op = "gpt2_decode-mode_full_model"
    base = ["model__gpt2", "config__full_model"]
    ran = ["num_cards__1", "spyre_devices__2", "pvc_hf_gib__0.52", "pvc_total_gib__1.9"]
    _, plain = results.parse_benchmark_xml(_write(tmp_path, _bench(op, base)))
    _, tagged = results.parse_benchmark_xml(_write(tmp_path, _bench(op, base + ran)))
    (e_plain,) = results._bench_entries(plain)
    (e_tagged,) = results._bench_entries(tagged)
    assert e_tagged["run_props"]["num_cards"] == "1"
    assert e_tagged["run_props"]["pvc_total_gib"] == "1.9"
    assert e_tagged["run_props"]["spyre_devices"] == "2"
    assert "num_cards" not in e_plain.get("run_props", {})
    assert "num_cards" not in e_tagged["props"]

    def ident(e):
        return BenchmarkId.derive(
            "torch-spyre", e["name"], e["tags"], e["disc"], e["disc_keys"]
        )

    assert ident(e_plain) == ident(e_tagged)

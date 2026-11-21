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

from spyre_clickhouse_ingest import results


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


def test_measurement_and_scope_suffixes_stay_one_row(tmp_path, capsys):
    op = "gemma-4-26B-A4B-it_full-mode_full_model_torch-spyre"
    tags = [
        f"op__{op}",
        "mode__generate",
        "model__gemma-4-26B-A4B-it",
        "config__full_model",
        "batch_size__4",
        "prompt_length__128",
        "measurement__generate-v1",
    ]
    kernel_tags = tags + [
        "kernel_scope__request",
        "regression_status__insufficient_data",
    ]
    cases = [
        _tc(f"perf_{op}_wall_clock_ms_measurement_generate-v1_{op}", tags, "1889.869"),
        _tc(f"perf_{op}_spyre_ms_measurement_generate-v1_{op}", tags, "1812.747"),
        _tc(
            f"perf_{op}_kernel_ms_measurement_generate-v1_scope_request_{op}",
            kernel_tags,
            "1791.818",
        ),
        _tc(
            f"perf_{op}_memory_transfer_ms_measurement_generate-v1_scope_request_{op}",
            kernel_tags,
            "20.929",
        ),
    ]
    _, rows = results.parse_benchmark_xml(_write(tmp_path, cases))
    assert len(rows) == 1
    row = rows[0]
    assert row["record_type"] == "model"
    assert row["total_duration_ms"] == pytest.approx(1889.869)
    assert row["spyre_ms"] == pytest.approx(1812.747)
    assert row["kernel_mean_ms"] == pytest.approx(1791.818)
    assert row["memory_transfer_mean_ms"] == pytest.approx(20.929)
    assert row["regression_status"] == "insufficient_data"
    assert row["measurement"] == "generate-v1"
    assert row["kernel_scope"] == "request"
    entries = results._bench_entries(rows)
    assert len(entries) == 2
    wall, scoped = entries
    assert set(wall["measurements"]) == {"total_duration_ms", "spyre_ms"}
    assert set(scoped["measurements"]) == {
        "kernel_mean_ms",
        "memory_transfer_mean_ms",
    }
    assert wall["props"]["measurement"] == "generate-v1"
    assert "kernel_scope" not in wall["props"]
    assert "kernel_scope" not in wall["disc_keys"]
    assert "measurement" in wall["disc_keys"]
    assert scoped["props"]["kernel_scope"] == "request"
    assert scoped["props"]["measurement"] == "generate-v1"
    assert "kernel_scope" in scoped["disc_keys"]
    assert "measurement" in scoped["disc_keys"]
    err = capsys.readouterr().err
    assert "duplicate" not in err
    assert "disagrees" not in err


def test_measurement_value_with_underscore_stays_one_row(tmp_path):
    op = "gemma-4-26B-A4B-it_full-mode_full_model_torch-spyre"
    tags = [f"op__{op}", "model__gemma-4-26B-A4B-it", "measurement__generate_v2"]
    cases = [
        _tc(f"perf_{op}_wall_clock_ms_measurement_generate_v2_{op}", tags, "10"),
        _tc(f"perf_{op}_spyre_ms_measurement_generate_v2_{op}", tags, "9"),
        _tc(
            f"perf_{op}_kernel_ms_measurement_generate_v2_scope_request_{op}",
            tags + ["kernel_scope__request"],
            "8",
        ),
    ]
    _, rows = results.parse_benchmark_xml(_write(tmp_path, cases))
    assert len(rows) == 1
    assert rows[0]["total_duration_ms"] == pytest.approx(10)
    assert rows[0]["spyre_ms"] == pytest.approx(9)
    assert rows[0]["kernel_mean_ms"] == pytest.approx(8)
    assert rows[0]["measurement"] == "generate_v2"
    assert results._group_shapes("bs1_pl512") == "bs1_pl512"
    assert results._group_shapes("measurement_generate_v2_bs1_pl512") == "bs1_pl512"
    assert results._group_shapes("measurement_generate_v2") == "measurement_generate_v2"
    assert results._group_shapes("measurement_generate-v1") == "measurement_generate-v1"


def test_duplicate_metric_after_scope_collapse_keeps_last(tmp_path, capsys):
    op = "gemma-4-26B-A4B-it_full-mode_full_model_torch-spyre"
    tags = [f"op__{op}", "model__gemma-4-26B-A4B-it"]
    region = f"perf_{op}_kernel_ms_measurement_generate-v1_scope_region_{op}"
    request = f"perf_{op}_kernel_ms_measurement_generate-v1_scope_request_{op}"
    cases = [
        _tc(region, tags + ["kernel_scope__region"], "10"),
        _tc(request, tags + ["kernel_scope__request"], "20"),
    ]
    _, rows = results.parse_benchmark_xml(_write(tmp_path, cases))
    assert len(rows) == 1
    assert rows[0]["kernel_mean_ms"] == pytest.approx(20)
    assert rows[0]["kernel_scope"] == "request"
    err = capsys.readouterr().err
    assert "duplicate kernel" in err
    assert "replacing" in err
    assert region in err
    assert request in err


def test_two_measurement_values_keep_last(tmp_path, capsys):
    op = "gemma"
    first = f"perf_{op}_wall_clock_ms_measurement_generate-v1_{op}"
    second = f"perf_{op}_wall_clock_ms_measurement_generate_v2_{op}"
    cases = [
        _tc(first, ["model__gemma", "measurement__generate-v1"], "11"),
        _tc(second, ["model__gemma", "measurement__generate_v2"], "22"),
    ]
    _, rows = results.parse_benchmark_xml(_write(tmp_path, cases))
    assert len(rows) == 1
    assert rows[0]["total_duration_ms"] == pytest.approx(22)
    assert rows[0]["measurement"] == "generate_v2"
    err = capsys.readouterr().err
    assert "duplicate wall_clock" in err
    assert first in err and second in err


def test_shape_prefixed_with_scope_stays_its_own_row(tmp_path, capsys):
    real = "perf_matmul_wall_clock_ms_64x64"
    prefixed = "perf_matmul_wall_clock_ms_scope_region_64x64"
    _, rows = results.parse_benchmark_xml(
        _write(
            tmp_path,
            [_tc(real, ["op__matmul"], "9"), _tc(prefixed, ["op__matmul"], "1")],
        )
    )
    assert sorted(r["input_shapes"] for r in rows) == ["64x64", "scope_region_64x64"]
    assert "duplicate" not in capsys.readouterr().err
    assert results._group_shapes("scope_region_64x64") == "scope_region_64x64"
    assert results._group_shapes("measurement_generate-v1_scope_request_op") == "op"
    assert results._group_shapes("scope_request_measurement_generate_v2_op") == "op"


def test_mixed_scope_drops_kernel_and_memory(tmp_path, capsys):
    op = "gemma"
    wall = f"perf_{op}_wall_clock_ms_measurement_generate-v1_{op}"
    kernel = f"perf_{op}_kernel_ms_measurement_generate-v1_scope_region_{op}"
    memory = f"perf_{op}_memory_transfer_ms_measurement_generate-v1_scope_request_{op}"
    cases = [
        _tc(wall, ["model__gemma", "measurement__generate-v1"], "11"),
        _tc(kernel, ["model__gemma", "kernel_scope__region"], "10"),
        _tc(memory, ["model__gemma", "kernel_scope__request"], "99"),
    ]
    _, rows = results.parse_benchmark_xml(_write(tmp_path, cases))
    assert len(rows) == 1
    assert rows[0]["total_duration_ms"] == pytest.approx(11)
    assert rows[0]["kernel_mean_ms"] is None
    assert rows[0]["memory_transfer_mean_ms"] is None
    assert rows[0]["kernel_scope"] is None
    err = capsys.readouterr().err
    assert "kernel_scope disagrees" in err
    assert "dropping both metrics" in err


def test_memory_only_scope_falls_back(tmp_path):
    op = "gemma"
    mem = f"perf_{op}_memory_transfer_ms_measurement_generate-v1_scope_request_{op}"
    _, rows = results.parse_benchmark_xml(
        _write(
            tmp_path,
            [_tc(mem, ["model__gemma", "kernel_scope__request"], "5")],
        )
    )
    assert rows[0]["memory_transfer_mean_ms"] == pytest.approx(5)
    assert rows[0]["kernel_scope"] == "request"


def test_v1_insert_ignores_measurement_and_kernel_scope():
    from datetime import UTC, datetime

    class Client:
        def __init__(self):
            self.column_names = None
            self.rows = None

        def query(self, sql, parameters=None):
            class R:
                # All optional columns present so the insert list stays full.
                result_rows = [(c,) for c in results._PERF_BENCHMARK_OPTIONAL_COLUMNS]

            return R()

        def insert(self, table, rows, column_names=None):
            self.column_names = column_names
            self.rows = rows

    row = {
        "benchmark_id": 1,
        "record_type": "model",
        "operation_name": "gemma",
        "config_name": "full_model",
        "input_shapes": None,
        "batch_size": 4,
        "prompt_length": 128,
        "run_mode": "generate",
        "total_duration_ms": 1.0,
        "cpu_ms": None,
        "spyre_ms": None,
        "kernel_mean_ms": None,
        "memory_transfer_mean_ms": None,
        "compile_ms": None,
        "runtime_ms": None,
        "mem_size_mb": None,
        "pt_util_percent": None,
        "num_runs": None,
        "custom_op_file": None,
        "regression_status": None,
        "measurement": "generate-v1",
        "kernel_scope": "request",
        "created_at": datetime(2026, 11, 11, tzinfo=UTC),
    }
    client = Client()
    results.insert_perf_benchmarks(client, 42, [row])
    assert client.column_names == list(results._PERF_BENCHMARK_COLUMNS)
    assert "measurement" not in client.column_names
    assert "kernel_scope" not in client.column_names
    assert len(client.rows) == 1


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
    assert row["operation_name"] == "granite"
    assert (row["batch_size"], row["prompt_length"]) == (4, 128)


def test_benchmark_granite_with_model_tag_stays_granite(tmp_path):
    op = "granite_bs4_pl128_generate_full_model"
    extra = ["model__granite-test-1b", "batch_size__4", "prompt_length__128"]
    _, rows = results.parse_benchmark_xml(_write(tmp_path, _bench(op, extra)))
    assert rows[0]["operation_name"] == "granite"
    assert (rows[0]["batch_size"], rows[0]["prompt_length"]) == (4, 128)


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
            "granite",
        ),
        ("matmul", ["config__null"], "op", "matmul"),
    ],
)
def test_kernel_legacy_and_op_records(tmp_path, op, extra, rtype, name):
    _, rows = results.parse_kernel_xml(_write(tmp_path, [_kernel(op, extra)]))
    assert rows[0]["record_type"] == rtype
    assert rows[0]["operation_name"] == name

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

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

"""Model records are told apart from op records by tag, not only the granite_ prefix."""

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


def _bench(op, config, mode="decode_only"):
    tags = [f"op__{op}", f"config__{config}", f"mode__{mode}"]
    return [
        _tc(f"perf_{op}_wall_clock_ms_x", tags, "2.0"),
        _tc(f"perf_{op}_spyre_ms_x", tags, "1.0"),
    ]


@pytest.mark.parametrize(
    "op, config, name",
    [
        (
            "gpt-neo-125m_decode-mode_full_model",
            "full_model",
            "gpt-neo-125m_decode-mode_full_model",
        ),
        (
            "gpt-neo-125m_prefill-mode_one_block",
            "one_block",
            "gpt-neo-125m_prefill-mode_one_block",
        ),
        ("granite_bs4_pl128_generate_full_model", "full_model", "granite"),
        ("granite_bs4_pl128_generate_full_model", "null", "granite"),
    ],
)
def test_benchmark_model_records(tmp_path, op, config, name):
    _, rows = results.parse_benchmark_xml(_write(tmp_path, _bench(op, config)))
    assert len(rows) == 1
    row = rows[0]
    assert row["record_type"] == "model"
    assert row["operation_name"] == name
    assert row["input_shapes"] is None
    assert row["run_mode"] == "decode_only"


def test_benchmark_granite_keeps_batch_and_prompt(tmp_path):
    op = "granite_bs4_pl128_generate_full_model"
    _, rows = results.parse_benchmark_xml(_write(tmp_path, _bench(op, "full_model")))
    assert (rows[0]["batch_size"], rows[0]["prompt_length"]) == (4, 128)


def test_benchmark_op_record_unchanged(tmp_path):
    cases = _bench("matmul", "null", "null")
    _, rows = results.parse_benchmark_xml(_write(tmp_path, cases))
    assert rows[0]["record_type"] == "op"
    assert rows[0]["operation_name"] == "matmul"
    assert rows[0]["run_mode"] == "op_benchmark"


def _kernel(op, config):
    tags = [
        f"op__{op}",
        "kernel__matmul",
        f"config__{config}",
        "mode__decode_only",
        "metric__kernel_ms",
    ]
    return _tc(f"perf_{op}_kernel", tags, classname="kernel_benchmark")


@pytest.mark.parametrize(
    "op, config, rtype, name",
    [
        (
            "gpt-neo-125m_decode-mode_full_model",
            "full_model",
            "model",
            "gpt-neo-125m_decode-mode_full_model",
        ),
        ("granite_bs4_pl128_generate_full_model", "full_model", "model", "granite"),
        ("granite_bs4_pl128_generate_full_model", "null", "model", "granite"),
        ("matmul", "null", "op", "matmul"),
    ],
)
def test_kernel_records(tmp_path, op, config, rtype, name):
    _, rows = results.parse_kernel_xml(_write(tmp_path, [_kernel(op, config)]))
    assert rows[0]["record_type"] == rtype
    assert rows[0]["operation_name"] == name

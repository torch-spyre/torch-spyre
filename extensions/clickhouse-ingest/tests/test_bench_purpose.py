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

"""`results --purpose` records run.purpose on benchmark_runs.props, and the verdict view reads it.

The ingest half runs the report.xml parser and the writer against a recording client; the view half (real chdb)
feeds the props the ingest wrote into the rolling-baseline view, so a rename of the prop or a change of the
eligible list on either side fails here."""

from argparse import Namespace

import pytest
from spyre_clickhouse_ingest import insert_benchmarks, results
from spyre_clickhouse_ingest.schema import BENCHMARK_RUNS

RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"
XML = (
    '<?xml version="1.0"?><testsuites><testsuite name="s" timestamp="2026-01-01T00:00:00Z">'
    '<testcase name="perf_op-alpha_wall_clock_ms_1_8" time="1.0"/></testsuite></testsuites>'
)


class Recorder:
    def __init__(self):
        self.inserts = []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows))

    def query(self, sql, parameters=None):
        class R:
            result_rows = []

        return R()


def ingested_props(tmp_path, purpose):
    """The props benchmark_runs gets for a report.xml ingested with --purpose=`purpose`."""
    xml = tmp_path / "report.xml"
    xml.write_text(XML)
    _, records = results.parse_benchmark_xml(xml, "w", "", "x86_64")
    args = Namespace(purpose=results._purpose_value(purpose))
    client = Recorder()
    entries = results._bench_entries(records, results._purpose_props(args))
    assert insert_benchmarks(client, "db", "torch-spyre", RUN, entries) == 1
    (rows,) = [r for t, r in client.inserts if t == BENCHMARK_RUNS.name]
    return rows[0][BENCHMARK_RUNS.columns.index("props")]


@pytest.mark.parametrize(
    "given, stored",
    [("nightly", "nightly"), (" Weekly ", "weekly"), ("pr-check", "pr-check")],
)
def test_the_purpose_reaches_benchmark_runs_props(tmp_path, given, stored):
    assert ingested_props(tmp_path, given)["run.purpose"] == stored


def test_no_purpose_writes_no_prop(tmp_path):
    assert "run.purpose" not in ingested_props(tmp_path, "")


def test_the_purpose_is_not_part_of_the_benchmark_identity(tmp_path):
    xml = tmp_path / "report.xml"
    xml.write_text(XML)
    _, records = results.parse_benchmark_xml(xml, "w", "", "x86_64")
    plain = results._bench_entries(records)
    tagged = results._bench_entries(records, {"run.purpose": "nightly"})
    assert [(e["name"], e["tags"], e["disc"]) for e in plain] == [
        (e["name"], e["tags"], e["disc"]) for e in tagged
    ]
    assert all("run_props" not in e for e in plain)


@pytest.mark.parametrize("bad", ["a b", "-x", "x" * 33, "nïghtly", "1st", "a/b"])
def test_a_malformed_purpose_is_refused(bad):
    with pytest.raises(SystemExit):
        results._purpose_value(bad)


@pytest.mark.parametrize(
    "flags",
    [["--sendnn-report", "x.txt"], ["--vllm-results-dir", "d"], ["--kind", "vllm"]],
)
def test_a_purpose_is_refused_for_inputs_that_cannot_record_it(flags):
    with pytest.raises(SystemExit) as err:
        results.main(["--purpose", "nightly", *flags])
    assert "--purpose" in str(err.value)

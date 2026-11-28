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

"""`results --model-list-json` records the model list a run was measured against as run props."""

import json
from argparse import Namespace

import pytest
from spyre_clickhouse_ingest import insert_benchmarks, results
from spyre_clickhouse_ingest.identity import BenchmarkId
from spyre_clickhouse_ingest.schema import BENCHMARK_RUNS

RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"
XML = (
    '<?xml version="1.0"?><testsuites><testsuite name="s" timestamp="2026-01-01T00:00:00Z">'
    '<testcase name="perf_op-alpha_wall_clock_ms_1_8" time="1.0"/></testsuite></testsuites>'
)
LIST = {
    "name": "weekly",
    "models": ["gpt2", "EleutherAI/gpt-neo-125m", " "],
    "commit": "0123abc",
    "source": "https://github.com/torch-spyre/hf-adapters/blob/0123abc/tests/model_lists/perf.yaml",
    "committed_at": "2026-11-20T10:00:00Z",
}


class Recorder:
    def __init__(self):
        self.inserts = []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows))

    def query(self, sql, parameters=None):
        class R:
            result_rows = []

        return R()


def _props(tmp_path, data):
    xml = tmp_path / "report.xml"
    xml.write_text(XML)
    path = tmp_path / "MODEL_LIST.json"
    path.write_text(json.dumps(data))
    _, records = results.parse_benchmark_xml(xml, "w", "", "x86_64")
    args = Namespace(purpose="", model_list_json=str(path))
    client = Recorder()
    entries = results._bench_entries(records, results._run_level_props(args))
    assert insert_benchmarks(client, "db", "torch-spyre", RUN, entries) == 1
    (rows,) = [r for t, r in client.inserts if t == BENCHMARK_RUNS.name]
    return rows[0][BENCHMARK_RUNS.columns.index("props")], entries


def test_the_list_reaches_benchmark_runs_props(tmp_path):
    props, _ = _props(tmp_path, LIST)
    assert props["model_list.name"] == "weekly"
    assert (
        props["model_list.models"] == "gpt2, EleutherAI/gpt-neo-125m"
    )  # blanks dropped
    assert props["model_list.commit"] == "0123abc"
    assert props["model_list.source"].endswith("/0123abc/tests/model_lists/perf.yaml")
    assert props["model_list.committed_at"] == "2026-11-20T10:00:00Z"


def test_the_list_is_not_part_of_the_benchmark_identity(tmp_path):
    xml = tmp_path / "report.xml"
    xml.write_text(XML)
    path = tmp_path / "MODEL_LIST.json"
    path.write_text(json.dumps(LIST))
    _, records = results.parse_benchmark_xml(xml, "w", "", "x86_64")
    listed = results._model_list_props(Namespace(model_list_json=str(path)))

    def ids(run_props):
        return [
            BenchmarkId.derive(
                "torch-spyre", e["name"], e["tags"], e["disc"], e["disc_keys"]
            )
            for e in results._bench_entries(records, run_props)
        ]

    assert ids(listed) == ids(None)


def test_no_flag_writes_no_list():
    assert results._model_list_props(Namespace(model_list_json="")) == {}


@pytest.mark.parametrize(
    "bad", [{"name": "weekly", "models": []}, {"models": ["gpt2"]}]
)
def test_a_list_without_a_name_or_models_is_refused(tmp_path, bad):
    path = tmp_path / "MODEL_LIST.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        results._model_list_props(Namespace(model_list_json=str(path)))

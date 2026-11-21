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

"""series and overview on a database that predates schema 88: the anchor is `unavailable` and a note, never an error."""

import pytest
from spyre_clickhouse_ingest import perf_api as api
from spyre_clickhouse_ingest import perf_fixture as fx
from test_perf_api import MATMUL, NOW, ChdbDb, bid

pytest.importorskip("chdb.session")


class OldDb(ChdbDb):
    def skip(self, stmt):
        return (
            super().skip(stmt)
            or "v_benchmark_anchor" in stmt
            or "v_benchmark_series_runs" in stmt
        )


@pytest.fixture(scope="module")
def db():
    d = OldDb().prepare()
    yield d
    d.close()


def test_series_degrades_to_unavailable(db):
    r = api.series(db, bid(db, MATMUL), NOW, metric="total_duration_ms", arch="x86_64")
    assert r["data"]["anchor"] == {"state": "unavailable"}
    assert r["meta"]["notes"] == ["schema 88 (anchored verdicts) is not applied"]
    assert r["data"]["points"] and all(
        p["anchored"] is None for p in r["data"]["points"]
    )
    assert r["data"]["summary"]["last_anchored_status"] is None


def test_overview_degrades_to_zero_counts_and_a_note(db):
    d = api.overview(db, NOW)
    assert d["meta"]["notes"] == ["schema 88 (anchored verdicts) is not applied"]
    assert d["data"]["persistent_regressions"] == []
    tile = {(t["arch"], t["component"]): t for t in d["data"]["tiles"]}[
        ("x86_64", fx.TS)
    ]
    assert tile["anchored"] == {
        "regressed": 0,
        "suspected": 0,
        "improved": 0,
        "unchanged": 0,
        "no_anchor": 0,
    }

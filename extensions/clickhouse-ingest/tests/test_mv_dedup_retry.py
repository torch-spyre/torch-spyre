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

"""A retry under the same token reaches a view target that missed the first attempt.

Single-node chdb with `non_replicated_deduplication_window` on both tables stands in for a
Replicated pair; Keeper-backed behaviour is not exercised here.
"""

import pytest

from spyre_clickhouse_ingest import resilience as rz

SETTING = "deduplicate_blocks_in_dependent_materialized_views"
WINDOW = "SETTINGS non_replicated_deduplication_window=100"


def count(s, table):
    return int(s.query(f"SELECT count() FROM {table}").bytes().decode().strip())


@pytest.fixture
def session():
    chdb_session = pytest.importorskip("chdb.session")
    s = chdb_session.Session()
    s.query("CREATE DATABASE IF NOT EXISTS mvd")
    yield s
    s.close()


def test_the_setting_resilient_insert_passes_recovers_the_view_exactly_once(session):
    s = session
    sent = rz._settings(rz.Policy(), "tok-1", True, False)
    assert sent[SETTING] == 1
    clause = f"insert_deduplication_token='tok-1', {SETTING}={sent[SETTING]}"
    s.query(f"CREATE TABLE mvd.src (id UInt32) ENGINE=MergeTree ORDER BY id {WINDOW}")
    s.query(f"CREATE TABLE mvd.dst (id UInt32) ENGINE=MergeTree ORDER BY id {WINDOW}")
    # First attempt: the source block lands but the view target does not (view not attached yet).
    s.query(f"INSERT INTO mvd.src SETTINGS {clause} VALUES (1),(2)")
    s.query("CREATE MATERIALIZED VIEW mvd.mv TO mvd.dst AS SELECT id FROM mvd.src")
    for _ in range(2):  # the retry, then a second retry
        s.query(f"INSERT INTO mvd.src SETTINGS {clause} VALUES (1),(2)")
    assert (count(s, "mvd.src"), count(s, "mvd.dst")) == (2, 2)

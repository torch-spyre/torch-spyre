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

"""pipeline_run_links' vocabularies and column order are the DDL's, not their own."""

import regex as re
from spyre_clickhouse_ingest import schema
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR

SQL = (SCHEMA_DIR / "56-pipeline-run-lineage.sql").read_text()


def _check(name: str) -> set:
    m = re.search(rf"CONSTRAINT\s+{name}\s+CHECK\s+\w+\s+IN\s*\(([^)]*)\)", SQL)
    assert m, name
    return set(re.findall(r"'([^']+)'", m.group(1)))


def test_vocabularies_match_the_ddl_checks():
    assert _check("chk_relation") == schema.RUN_LINK_RELATION_VALUES
    assert _check("chk_evidence") == schema.RUN_LINK_EVIDENCE_VALUES
    assert _check("chk_confidence") == schema.RUN_LINK_CONFIDENCE_VALUES


def test_columns_follow_the_ddl_order():
    body = SQL.split("CREATE TABLE IF NOT EXISTS pipeline_run_links", 1)[1].split(
        "ENGINE", 1
    )[0]
    ddl = re.findall(r"^    (\w+)\s+[A-Z]", body, flags=re.M)
    ddl = [c for c in ddl if c not in ("audit_uuid", "audit_timestamp")]
    assert list(schema.PipelineRunLinks.columns) == ddl

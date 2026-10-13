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

"""result_uploads' vocabularies are artifact_results' and the bundle schema's, not its own."""

import regex as re
from spyre_clickhouse_ingest import offline
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR


def _check(file: str, name: str) -> list:
    sql = (SCHEMA_DIR / file).read_text()
    m = re.search(rf"CONSTRAINT\s+{name}\s+CHECK\s+\w+\s+IN\s*\(([^)]*)\)", sql)
    assert m, f"{name} not in {file}"
    return re.findall(r"'([^']+)'", m.group(1))


def test_result_kind_and_test_type_match_artifact_results():
    for name in ("chk_result_kind", "chk_test_type"):
        assert _check("49-result-uploads.sql", name) == _check("20-artifacts.sql", name)


def test_format_matches_the_bundle_kinds():
    kinds = offline.schema()["properties"]["kind"]["enum"]
    assert _check("49-result-uploads.sql", "chk_format") == kinds

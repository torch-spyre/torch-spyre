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

"""Shared schema-v2 ClickHouse schema, identity and write path for the Spyre CI ingests.

Kept out of the `torch_spyre` package on purpose: depending on it would pull torch/numpy/ortools,
and ortools has no ppc64le/s390x wheel, so the ingest could not run on p/z.
"""

from . import gha_logs, hw_parse, hw_schema, schema
from .client import client_summary, get_client, v2_database, v2_tables_present
from .hw_diagnostics import (
    RunContext,
    build_row,
    filter_suite_records,
    insert_rows,
    load_records,
)
from .hw_schema import (
    HW_COLUMN_NAMES,
    already_ingested,
    ensure_extra_columns,
)
from .identity import (
    V2_COMPONENT_DEFAULT,
    V2_NAMESPACE,
    V2_SEP,
    v2_canonical_arch,
    v2_component,
    v2_run_id,
    v2_run_id_for,
    v2_tags_for_case,
    v2_test_case_id,
    v2_artifact_id,
    v2_gha_artifact_id,
)
from .junit import (
    extract_properties,
    promote_xpass,
    v2_source_and_external_run_id,
)
from .v2_writer import insert_v2, v2_already_ingested

__all__ = [
    "HW_COLUMN_NAMES",
    "RunContext",
    "V2_COMPONENT_DEFAULT",
    "V2_NAMESPACE",
    "V2_SEP",
    "already_ingested",
    "build_row",
    "client_summary",
    "ensure_extra_columns",
    "extract_properties",
    "filter_suite_records",
    "get_client",
    "gha_logs",
    "hw_parse",
    "hw_schema",
    "insert_rows",
    "insert_v2",
    "load_records",
    "promote_xpass",
    "schema",
    "v2_already_ingested",
    "v2_canonical_arch",
    "v2_component",
    "v2_database",
    "v2_run_id",
    "v2_run_id_for",
    "v2_source_and_external_run_id",
    "v2_tables_present",
    "v2_tags_for_case",
    "v2_test_case_id",
    "v2_artifact_id",
    "v2_gha_artifact_id",
]

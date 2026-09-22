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
from .client import client_summary, get_client, target_database, tables_present
from .hw_diagnostics import (
    RunContext,
    build_row,
    filter_suite_records,
    insert_rows,
    load_records,
)
from .hw_schema import HW_COLUMN_NAMES, already_ingested
from .identity import (
    COMPONENT_DEFAULT,
    ID_NAMESPACE,
    ID_SEP,
    artifact_id_for,
    benchmark_id_for,
    canonical_arch,
    capability_id_for,
    component_of,
    base_artifact_id,
    gha_artifact_id,
    run_id_of,
    run_id_for,
    tags_for_case,
    case_id_for,
)
from .junit import (
    extract_properties,
    promote_xpass,
    source_and_external_run_id,
)
from .writer import (
    insert_benchmarks,
    insert_capabilities,
    insert_test_results,
    cases_already_ingested,
    benchmarks_already_ingested,
    capabilities_already_ingested,
)

__all__ = [
    "HW_COLUMN_NAMES",
    "RunContext",
    "COMPONENT_DEFAULT",
    "ID_NAMESPACE",
    "ID_SEP",
    "already_ingested",
    "build_row",
    "client_summary",
    "extract_properties",
    "filter_suite_records",
    "get_client",
    "gha_logs",
    "hw_parse",
    "hw_schema",
    "insert_benchmarks",
    "insert_capabilities",
    "insert_rows",
    "insert_test_results",
    "load_records",
    "promote_xpass",
    "schema",
    "cases_already_ingested",
    "artifact_id_for",
    "benchmark_id_for",
    "benchmarks_already_ingested",
    "canonical_arch",
    "capabilities_already_ingested",
    "capability_id_for",
    "component_of",
    "target_database",
    "base_artifact_id",
    "gha_artifact_id",
    "run_id_of",
    "run_id_for",
    "source_and_external_run_id",
    "tables_present",
    "tags_for_case",
    "case_id_for",
]

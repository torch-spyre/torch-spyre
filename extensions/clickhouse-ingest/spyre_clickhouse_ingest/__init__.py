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

"""Shared schema-v2 ClickHouse schema, identity and write path for the Spyre CI ingests."""

from . import gha_logs, hw_parse, hw_schema, schema
from .client import (
    ClickHouse,
    client_summary,
    get_client,
    tables_present,
    target_database,
)
from .hw_diagnostics import (
    RunContext,
    build_row,
    filter_suite_records,
    insert_rows,
    load_records,
)
from .hw_schema import HW_COLUMN_NAMES, HwFailureDiagnostics, already_ingested
from .identity import (
    COMPONENT_DEFAULT,
    ID_NAMESPACE,
    ID_SEP,
    ArtifactId,
    BenchmarkId,
    CapabilityId,
    CaseId,
    Component,
    DerivedId,
    GhaArtifactId,
    RunId,
    artifact_id_for,
    base_artifact_id,
    benchmark_id_for,
    canonical_arch,
    capability_id_for,
    case_id_for,
    component_of,
    gha_artifact_id,
    installed_digest,
    run_id_for,
    run_id_of,
    tags_for_case,
)
from .junit import (
    JUnitXml,
    RunCoordinates,
    extract_properties,
    promote_xpass,
    source_and_external_run_id,
)
from .writer import (
    ArtifactWriter,
    BenchmarkWriter,
    CapabilityWriter,
    TestResultWriter,
    artifact_already_recorded,
    artifact_result_already_recorded,
    benchmarks_already_ingested,
    capabilities_already_ingested,
    cases_already_ingested,
    insert_benchmarks,
    insert_capabilities,
    insert_gha_artifact_result,
    insert_test_results,
)

__all__ = [
    "COMPONENT_DEFAULT",
    "HW_COLUMN_NAMES",
    "ID_NAMESPACE",
    "ID_SEP",
    "ArtifactId",
    "ArtifactWriter",
    "BenchmarkId",
    "BenchmarkWriter",
    "CapabilityId",
    "CapabilityWriter",
    "CaseId",
    "ClickHouse",
    "Component",
    "DerivedId",
    "GhaArtifactId",
    "HwFailureDiagnostics",
    "JUnitXml",
    "RunContext",
    "RunCoordinates",
    "RunId",
    "TestResultWriter",
    "already_ingested",
    "artifact_already_recorded",
    "artifact_id_for",
    "artifact_result_already_recorded",
    "base_artifact_id",
    "benchmark_id_for",
    "benchmarks_already_ingested",
    "build_row",
    "canonical_arch",
    "capabilities_already_ingested",
    "capability_id_for",
    "case_id_for",
    "cases_already_ingested",
    "client_summary",
    "component_of",
    "extract_properties",
    "filter_suite_records",
    "get_client",
    "gha_artifact_id",
    "gha_logs",
    "hw_parse",
    "hw_schema",
    "insert_benchmarks",
    "insert_capabilities",
    "insert_gha_artifact_result",
    "insert_rows",
    "insert_test_results",
    "installed_digest",
    "load_records",
    "promote_xpass",
    "run_id_for",
    "run_id_of",
    "schema",
    "source_and_external_run_id",
    "tables_present",
    "tags_for_case",
    "target_database",
]

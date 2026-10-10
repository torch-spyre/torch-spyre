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

Names resolve on first use, so the offline paths (offline.py) run with the standard library
alone: an air-gapped host needs neither clickhouse-connect nor regex to write a bundle.
"""

import importlib

_SUBMODULES = ("gha_logs", "hw_parse", "hw_schema", "schema")
_EXPORTS = {
    "client": (
        "ClickHouse",
        "client_summary",
        "get_client",
        "tables_present",
        "target_database",
    ),
    "hw_diagnostics": (
        "RunContext",
        "build_row",
        "filter_suite_records",
        "insert_rows",
        "load_records",
    ),
    "hw_schema": (
        "HW_COLUMN_NAMES",
        "HwFailureDiagnostics",
        "already_ingested",
    ),
    "identity": (
        "COMPONENT_DEFAULT",
        "ID_NAMESPACE",
        "ID_SEP",
        "LEGACY_TAG_ALIASES",
        "RESULT_TAG_NAMESPACES",
        "RUN_CONTEXT_TAG_NAMESPACES",
        "ArtifactId",
        "ArtifactIdentity",
        "BenchmarkId",
        "CapabilityId",
        "CaseId",
        "Component",
        "DerivedId",
        "GhaArtifactId",
        "RunId",
        "artifact_id_for",
        "artifact_identity",
        "base_artifact_id",
        "benchmark_id_for",
        "canonical_arch",
        "capability_id_for",
        "case_id_for",
        "component_of",
        "gha_artifact_id",
        "installed_digest",
        "run_id_for",
        "run_id_of",
        "split_case_tags",
        "tags_for_case",
    ),
    "options": ("ci_tags",),
    "resolver": (
        "Resolution",
        "ensure",
        "ensure_artifact",
        "resolve",
        "resolve_artifact",
    ),
    "junit": (
        "JUnitXml",
        "RunCoordinates",
        "extract_properties",
        "promote_xpass",
        "source_and_external_run_id",
    ),
    "writer": (
        "CAPABILITY_PREFIX",
        "CAPABILITY_REQUIRED",
        "ArtifactWriter",
        "BenchmarkWriter",
        "CapabilityWriter",
        "TestResultWriter",
        "artifact_already_recorded",
        "artifact_result_already_recorded",
        "benchmarks_already_ingested",
        "capabilities_already_ingested",
        "capability_declaration",
        "cases_already_ingested",
        "drop_older_case_attempts",
        "insert_artifact",
        "insert_artifact_result",
        "insert_artifact_result_reason",
        "insert_benchmarks",
        "insert_capabilities",
        "insert_gha_artifact_result",
        "insert_test_results",
    ),
}
_HOME = {name: module for module, names in _EXPORTS.items() for name in names}


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name not in _HOME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_HOME[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "CAPABILITY_PREFIX",
    "CAPABILITY_REQUIRED",
    "COMPONENT_DEFAULT",
    "HW_COLUMN_NAMES",
    "ID_NAMESPACE",
    "ID_SEP",
    "LEGACY_TAG_ALIASES",
    "RESULT_TAG_NAMESPACES",
    "RUN_CONTEXT_TAG_NAMESPACES",
    "ArtifactId",
    "ArtifactIdentity",
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
    "artifact_identity",
    "artifact_result_already_recorded",
    "base_artifact_id",
    "benchmark_id_for",
    "benchmarks_already_ingested",
    "build_row",
    "canonical_arch",
    "capabilities_already_ingested",
    "capability_declaration",
    "capability_id_for",
    "case_id_for",
    "cases_already_ingested",
    "client_summary",
    "component_of",
    "drop_older_case_attempts",
    "ensure",
    "ensure_artifact",
    "extract_properties",
    "filter_suite_records",
    "get_client",
    "gha_artifact_id",
    "gha_logs",
    "hw_parse",
    "hw_schema",
    "insert_artifact",
    "insert_artifact_result",
    "insert_artifact_result_reason",
    "ci_tags",
    "insert_benchmarks",
    "insert_capabilities",
    "insert_gha_artifact_result",
    "insert_rows",
    "insert_test_results",
    "installed_digest",
    "load_records",
    "promote_xpass",
    "Resolution",
    "resolve",
    "resolve_artifact",
    "run_id_for",
    "run_id_of",
    "schema",
    "source_and_external_run_id",
    "split_case_tags",
    "tables_present",
    "tags_for_case",
    "target_database",
]

"""Shared schema-v2 ClickHouse schema, identity and write path for the Spyre CI ingests.

Kept out of the `torch_spyre` package on purpose: depending on it would pull torch/numpy/ortools,
and ortools has no ppc64le/s390x wheel, so the ingest could not run on p/z.
"""

from . import schema
from .client import get_client, v2_database, v2_tables_present
from .identity import (
    V2_COMPONENT_DEFAULT,
    V2_NAMESPACE,
    V2_SEP,
    v2_canonical_arch,
    v2_component,
    v2_run_id,
    v2_tags_for_case,
    v2_test_case_id,
)
from .junit import (
    extract_properties,
    promote_xpass,
    v2_source_and_external_run_id,
)
from .v2_writer import insert_v2, v2_already_ingested

__all__ = [
    "V2_COMPONENT_DEFAULT",
    "V2_NAMESPACE",
    "V2_SEP",
    "extract_properties",
    "get_client",
    "insert_v2",
    "promote_xpass",
    "schema",
    "v2_already_ingested",
    "v2_canonical_arch",
    "v2_component",
    "v2_database",
    "v2_run_id",
    "v2_source_and_external_run_id",
    "v2_tables_present",
    "v2_tags_for_case",
    "v2_test_case_id",
]

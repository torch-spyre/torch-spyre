"""Shared schema-v2 ClickHouse schema, identity and write path for the Spyre CI ingests.

Extracted from the three product repos' ingest scripts, which carried byte-identical copies of
this code kept in sync by hand and policed by an AST drift checker. The drift checker existed
only because the copies did.

Why a library and not a class hierarchy: the per-repo variation is a single value, the component
name, which is already a runtime argument (`--component`). Everything else was identical.

Why not part of the `torch_spyre` package: installing that to obtain a schema module would pull
torch/numpy/ortools, and ortools has no ppc64le/s390x wheel -- the ingest would break on p/z.
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

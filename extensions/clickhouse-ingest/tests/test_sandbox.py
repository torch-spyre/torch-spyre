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

"""Pins the sandbox's sampling rules: runs pick the rows, dimensions precede their facts,
and a filter on a column a source lacks drops that source rather than the filter."""

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier
from spyre_clickhouse_ingest.sandbox import Sandbox, SeedFilter

COLS = {
    "test_case_runs": {"run_id", "component", "ts"},
    "artifact_results": {"run_id", "arch", "ts"},
}


@pytest.mark.parametrize("bad", ["", "A", "1x", "x-y", "x;drop", "a" * 42])
def test_name_is_validated(bad):
    with pytest.raises(SystemExit):
        Sandbox.database(bad)


def test_names():
    assert Sandbox.database("ana") == "sandbox_ana"
    assert Sandbox.user("ana") == "sandbox_ana_admin"


def test_component_filter_drops_sources_without_component():
    sql = " ".join(Sandbox.run_selects(SeedFilter(components=("hf-adapters",)), COLS))
    assert "table='test_case_runs'" in sql and "table='artifact_results'" not in sql
    assert "component IN ('hf-adapters')" in sql and "LIMIT 200 BY component" in sql


def test_unfiltered_sample_uses_every_present_source():
    selects = Sandbox.run_selects(SeedFilter(days=3, runs_per_component=7), COLS)
    assert len(selects) == 2
    assert all("INTERVAL 3 DAY" in s and "LIMIT 7" in s for s in selects)


def test_tag_filter_restricts_through_artifact_tags():
    sql = Sandbox.run_selects(SeedFilter(tags=("torch-spyre@main",)), COLS)[0]
    assert "table='artifact_tags'" in sql and "'torch-spyre@main'" in sql


def test_values_are_quoted():
    assert Sandbox.quoted(["a'b", "c\\"]) == "'a\\'b', 'c\\\\'"


def test_render_reads_prod_and_cuts_to_the_runs():
    table, template, key = Sandbox.SEED[0]
    sql = Sandbox.render(table, template, key, "sandbox_x", SeedFilter())
    assert "remote(prod_v2, table='test_case_runs')" in sql
    assert "SELECT run_id FROM sandbox_x._seed_runs" in sql
    assert "{" not in sql


def test_each_dimension_precedes_its_fact():
    order = [t for t, _, _ in Sandbox.SEED]
    for dim, fact in (
        ("test_cases", "test_case_runs"),
        ("benchmarks", "benchmark_runs"),
        ("capabilities", "capability_runs"),
        ("artifacts", "artifact_results"),
        ("artifacts", "artifact_tags"),
        ("pipeline_runs", "pipeline_run_legs"),
    ):
        assert order.index(dim) < order.index(fact)


def test_seed_covers_every_base_table_in_schema():
    """A new v2 table needs a SEED rule, or sandboxes silently come up without its rows."""
    declared = {
        o.name
        for path, text in SchemaApplier.selected_files(SCHEMA_DIR)
        for o in SchemaApplier.objects(path, text)
        if o.kind == "table"
    }
    filled_by_mv = {
        "run_case_counters",
        "oss_ci_benchmark_v3",
        "oss_ci_benchmark_metadata",
    }
    unseeded = declared - {t for t, _, _ in Sandbox.SEED} - filled_by_mv
    assert {t for t in unseeded if not t.startswith("otel_")} == set()


@pytest.mark.parametrize("table, template, key", Sandbox.SEED)
def test_a_reseed_skips_keys_the_sandbox_holds(table, template, key):
    sql = Sandbox.render(table, template, key, "sandbox_x", SeedFilter())
    assert sql.endswith(
        f"AND {key} GLOBAL NOT IN (SELECT {key.strip('()')} FROM sandbox_x.{table})"
    )

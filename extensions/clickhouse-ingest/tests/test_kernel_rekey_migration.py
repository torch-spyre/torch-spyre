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

"""Runs migrations/012 on chdb: a pass stopped anywhere, then rerun, must end with every
kernel's rows under its new id."""

import uuid

import pytest
from spyre_clickhouse_ingest import benchmark_id_for
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

MIGRATION = SCHEMA_DIR / "migrations" / "012_benchmark_id_without_kernel_hash.sql"
STATEMENTS = SchemaApplier.statements(MIGRATION.read_text())
KEYS = (
    "record_type",
    "config_name",
    "input_shapes",
    "run_mode",
    "kernel_name",
    "is_total",
)
RUN = "0192a000-0000-7000-8000-000000000001"
STEM = "spyre_kernel_v1_fused_add#2"
# raw name -> mean ms; the slower kernel ranks @1.
KERNELS = {
    "spyre_kernel_v1_fused_add_aaaaaaaaaaaaaaaa#2": 0.003,
    "spyre_kernel_v1_fused_add_bbbbbbbbbbbbbbbb#2": 0.27,
}


class Chdb:
    def __init__(self):
        self.s = session.Session()

    def command(self, sql):
        out = str(self.s.query(sql, "TSV")).rstrip("\n")
        return int(out) if out.isdigit() else out


def _id(kernel):
    disc = {"record_type": "op", "kernel_name": kernel}
    return benchmark_id_for("torch-spyre", "pointwise_add", [], disc, KEYS)


@pytest.fixture
def db():
    c = Chdb()
    for stmt in SchemaApplier.statements(
        (SCHEMA_DIR / "30-benchmarks.sql").read_text()
    ):
        c.command(stmt)
    for raw, ms in KERNELS.items():
        bid = _id(raw)
        c.command(
            "INSERT INTO benchmarks (benchmark_id, component, name, props, audit_timestamp) "
            f"VALUES ('{bid}', 'torch-spyre', 'pointwise_add', "
            f"{{'record_type': 'op', 'kernel_name': '{raw}'}}, now64(3) - INTERVAL 1 DAY)"
        )
        c.command(
            "INSERT INTO benchmark_runs (run_id, benchmark_id, component, backend, "
            f"measurements, props, audit_uuid) VALUES ('{RUN}', '{bid}', 'torch-spyre', "
            f"'spyre', {{'duration_ms': [{ms}]}}, {{'source_file': 'f.json'}}, "
            f"'{uuid.uuid4()}')"
        )
    yield c
    c.s.close()


def _rows(c, sql):
    return {tuple(line.split("\t")) for line in c.command(sql).splitlines()}


@pytest.mark.parametrize("stop", range(len(STATEMENTS)))
def test_a_rerun_after_a_partial_pass_moves_every_kernel(db, stop):
    for stmt in STATEMENTS[:stop]:
        if "benchmark_metric_verdicts" not in stmt:
            db.command(stmt)
    SchemaApplier.rerun(db, MIGRATION)
    SchemaApplier.rerun(db, MIGRATION)
    slow, fast = sorted(KERNELS, key=KERNELS.get, reverse=True)
    assert _rows(
        db, "SELECT benchmark_id, props['kernel_name'] FROM benchmark_runs"
    ) == {(_id(f"{STEM}@1"), slow), (_id(f"{STEM}@2"), fast)}
    assert _rows(db, "SELECT benchmark_id, props['kernel_key'] FROM benchmarks") == {
        (_id(f"{STEM}@1"), f"{STEM}@1"),
        (_id(f"{STEM}@2"), f"{STEM}@2"),
    }

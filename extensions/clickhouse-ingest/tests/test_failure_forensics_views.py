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

"""Static checks on schema/57; the views' behaviour is proven on a sandbox (verify_schema)."""

import json
import re
from pathlib import Path

from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

FILE = SCHEMA_DIR / "57-failure-forensics-views.sql"
VIEWS = {
    "v_result_run_context",
    "v_failure_fingerprint",
    "v_case_run_history",
    "v_failure_recurrence",
    "v_failure_first_seen",
    "v_failure_groups",
}


def test_declares_only_the_forensics_views_idempotently():
    stmts = SchemaApplier.statements(FILE.read_text())
    names = {re.search(r"VIEW IF NOT EXISTS (\w+)", s).group(1) for s in stmts}
    assert names == VIEWS and len(stmts) == len(VIEWS)


def test_readme_lists_the_file_and_every_view():
    row = next(
        ln
        for ln in (SCHEMA_DIR / "README.md").read_text().splitlines()
        if "57-failure-forensics-views.sql" in ln
    )
    assert all(v in row for v in VIEWS)


def test_fingerprint_columns_are_exposed_to_consumers():
    text = FILE.read_text()
    assert all(
        f"AS {c}" in text for c in ("fp_coarse", "fp_fine", "fp_text", "runner_nodes")
    )


def test_event_record_tables_are_create_if_missing():
    text = (SCHEMA_DIR / "58-ci-event-records.sql").read_text()
    stmts = SchemaApplier.statements(text)
    names = {re.search(r"TABLE IF NOT EXISTS (\w+)", s).group(1) for s in stmts}
    assert names == {"poller_decisions", "ci_write_failures", "merge_queue_events"}


CASES = json.loads(
    (Path(__file__).parent / "data" / "failure_fingerprint_cases.json").read_text()
)["cases"]


def by(name):
    return next(c for c in CASES if c["case"] == name)


def test_fixture_shape():
    keys = {
        "case",
        "test",
        "status",
        "message",
        "exception_class",
        "fp_coarse",
        "fp_fine",
        "fp_text",
    }
    assert all(set(c) == keys for c in CASES)
    assert len({c["case"] for c in CASES}) == len(CASES)
    assert all(c["fp_coarse"].isdigit() and c["fp_fine"].isdigit() for c in CASES)


def test_fixture_wrapper_cause_beats_wrapper_line():
    for name in ("wrapper_indented_error", "wrapper_mlir_error"):
        assert "SubprocException" not in by(name)["fp_text"]
    assert by("truncated_no_cause")["exception_class"] == ""


def test_fixture_numbers_paths_collapse_but_tests_and_causes_stay_apart():
    a, b, o = by("numbers_a"), by("numbers_b"), by("same_cause_other_test")
    assert a["fp_coarse"] == b["fp_coarse"] == o["fp_coarse"]
    assert len({a["fp_fine"], b["fp_fine"], o["fp_fine"]}) == 3
    assert by("tensor_likes_a")["fp_coarse"] == by("tensor_likes_b")["fp_coarse"]
    coarse = [
        by(n)["fp_coarse"]
        for n in (
            "wrapper_indented_error",
            "wrapper_mlir_error",
            "numbers_a",
            "tensor_likes_a",
            "unrelated_a",
            "unrelated_b",
        )
    ]
    assert len(set(coarse)) == len(coarse)

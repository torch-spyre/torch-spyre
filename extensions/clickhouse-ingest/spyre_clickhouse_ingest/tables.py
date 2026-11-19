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

"""Per-table JSON for `kind: tables` bundles: one `results/<table>.json` per results table.

A file holds `{"rows": [...]}`. Each row carries the identity fields its table hashes, never the
ids: `test_case_runs` rows name the test (`classname`, `name`, `tags`), `benchmark_runs` rows the
benchmark (`name`, `props`), `capability_runs` rows the capability (`subject`, `name`, `sig`), and
the writers derive `test_case_id` / `benchmark_id` / `capability_id`, dedup and enforce the DDL's
CHECKs. `artifact_results` rows are verdicts; `run_id` and `artifact_id` come from the bundle.

`json_schemas()` is the machine-readable form, generated from the same tables so it cannot drift.
"""

import math
from pathlib import Path

from . import schema

# Uploadable results tables -> (identity table the writer fills too, what a row is).
UPLOADABLE = {
    "test_case_runs": ("test_cases", "one test's outcome"),
    "benchmark_runs": ("benchmarks", "one benchmark's measurements on one backend"),
    "capability_runs": ("capabilities", "one capability's verdict on one backend"),
    "artifact_results": (None, "one leg's verdict on the bundle's artifact"),
}
# Reference-only: documented and schema'd, never uploaded (CI-internal or on their own path).
REFERENCE_ONLY = (
    "artifacts", "artifact_refs", "artifact_tags", "artifact_result_reasons", "pipeline_runs",
    "ci_run_timings", "artifact_subscriptions", "dispatch_requests", "artifact_dispatches",
    "dispatch_targets", "pipeline_run_links", "rbac_roles", "rbac_bindings",
)  # fmt: skip
FILE_SUFFIX = ".json"
# What a bundle's tests may call themselves; the DDL admits more for verdicts than a run uploads.
OUTCOMES = sorted(schema.STATUS_VALUES)
CAPABILITY_STATUSES = sorted(schema.CAPABILITY_STATUS_VALUES)


class RowError(ValueError):
    """A row the table cannot take; the message names the file's row and the problem."""


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _text_map(row: dict, key: str, where: str) -> None:
    m = row.get(key, {})
    if not isinstance(m, dict) or not all(
        isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in m.values()
    ):
        raise RowError(f"{where}: {key} maps names to strings or numbers")


def _numbers(row: dict, key: str, where: str, lists: bool) -> None:
    m = row.get(key)
    if not isinstance(m, dict) or not m:
        raise RowError(f"{where}: needs {key}")
    for k, v in m.items():
        values = v if (lists and isinstance(v, list)) else [v]
        if not values or not all(_finite(x) for x in values):
            raise RowError(f"{where}: {key}[{k!r}] must be finite numbers")


def _strings(row: dict, key: str, where: str) -> None:
    v = row.get(key, [])
    if not isinstance(v, list) or not all(isinstance(t, str) for t in v):
        raise RowError(f"{where}: {key} is a list of strings")


def _need(row: dict, key: str, where: str) -> str:
    v = row.get(key)
    if not isinstance(v, str) or not v.strip():
        raise RowError(f"{where}: needs {key}")
    return v


def _unknown(row: dict, allowed: set, where: str) -> None:
    extra = set(row) - allowed
    if extra:
        raise RowError(f"{where}: unknown key(s) {sorted(extra)}")


def check_test_case_run(row: dict, where: str) -> None:
    _unknown(
        row,
        {
            "classname",
            "name",
            "status",
            "duration_s",
            "fail_message",
            "tags",
            "measurements",
            "props",
        },
        where,
    )
    _need(row, "name", where)
    if row.get("status") not in schema.STATUS_VALUES:
        raise RowError(
            f"{where}: status {row.get('status')!r} is not one of {OUTCOMES}"
        )
    if "duration_s" in row and not (
        _finite(row["duration_s"]) and row["duration_s"] >= 0
    ):
        raise RowError(f"{where}: duration_s is a number of seconds")
    for key in ("classname", "fail_message"):
        if not isinstance(row.get(key, ""), str):
            raise RowError(f"{where}: {key} is a string")
    _strings(row, "tags", where)
    _text_map(row, "props", where)
    stray = sorted(k for k in row.get("props", {}) if not k.startswith("result."))
    if stray:
        raise RowError(f"{where}: props are result.* values; not recorded: {stray}")
    if "measurements" in row:
        _numbers(row, "measurements", where, lists=False)


def check_benchmark_run(row: dict, where: str) -> None:
    _unknown(
        row,
        {
            "name",
            "backend",
            "measurements",
            "iterations",
            "tags",
            "props",
            "run_props",
            "identity",
        },
        where,
    )
    _need(row, "name", where)
    _numbers(row, "measurements", where, lists=True)
    if not isinstance(row.get("backend", ""), str):
        raise RowError(f"{where}: backend is a string")
    n = row.get("iterations", 0)
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise RowError(f"{where}: iterations is a count")
    _strings(row, "tags", where)
    for key in ("props", "run_props"):
        _text_map(row, key, where)
    if "identity" in row:
        ident = row["identity"]
        if not isinstance(ident, list) or not all(isinstance(k, str) for k in ident):
            raise RowError(f"{where}: identity is a list of prop names")
        absent = sorted(set(ident) - set(row.get("props", {})))
        if absent:
            raise RowError(
                f"{where}: identity names prop(s) the row does not have: {absent}"
            )


def check_capability_run(row: dict, where: str) -> None:
    _unknown(
        row,
        {
            "test_type",
            "subject",
            "name",
            "status",
            "backend",
            "fail_reason",
            "arch",
            "tags",
            "sig",
            "props",
        },
        where,
    )
    for key in ("subject", "name"):
        _need(row, key, where)
    if row.get("test_type") not in schema.CAPABILITY_TYPE_VALUES:
        raise RowError(
            f"{where}: test_type {row.get('test_type')!r} is not one of {sorted(schema.CAPABILITY_TYPE_VALUES)}"
        )
    if row.get("status") not in schema.CAPABILITY_STATUS_VALUES:
        raise RowError(
            f"{where}: status {row.get('status')!r} is not one of {CAPABILITY_STATUSES}"
        )
    for key in ("backend", "fail_reason", "arch"):
        if not isinstance(row.get(key, ""), str):
            raise RowError(f"{where}: {key} is a string")
    _strings(row, "tags", where)
    for key in ("sig", "props"):
        _text_map(row, key, where)


def check_artifact_result(row: dict, where: str) -> None:
    _unknown(
        row, {"result_kind", "test_type", "state", "arch", "duration_s", "props"}, where
    )
    if row.get("test_type") not in schema.TEST_TYPE_VALUES:
        raise RowError(
            f"{where}: test_type {row.get('test_type')!r} is not one of {sorted(schema.TEST_TYPE_VALUES)}"
        )
    if row.get("state") not in schema.STATE_VALUES - {"running"}:
        raise RowError(
            f"{where}: state {row.get('state')!r} is not passed, failed or error"
        )
    if (
        row.get("result_kind", "")
        and row["result_kind"] not in schema.RESULT_KIND_VALUES
    ):
        raise RowError(
            f"{where}: result_kind {row['result_kind']!r} is not one of {sorted(schema.RESULT_KIND_VALUES)}"
        )
    if "duration_s" in row and not (
        _finite(row["duration_s"]) and row["duration_s"] >= 0
    ):
        raise RowError(f"{where}: duration_s is a number of seconds")
    _text_map(row, "props", where)


CHECKS = {
    "test_case_runs": check_test_case_run,
    "benchmark_runs": check_benchmark_run,
    "capability_runs": check_capability_run,
    "artifact_results": check_artifact_result,
}


def table_of(filename: str) -> str:
    """The table a results file names, or '' when it names none this kind takes."""
    stem = (
        Path(filename).name[: -len(FILE_SUFFIX)]
        if filename.endswith(FILE_SUFFIX)
        else ""
    )
    return stem if stem in UPLOADABLE else ""


def validate_rows(table: str, rows, source: str = "") -> list:
    """`rows` checked for `table`; raises RowError on the first bad one and returns the list."""
    if table not in UPLOADABLE:
        raise RowError(f"{table!r} is not an uploadable table ({sorted(UPLOADABLE)})")
    if not isinstance(rows, list) or not rows:
        raise RowError(f'{source or table}: needs a non-empty "rows" list')
    for i, row in enumerate(rows):
        where = f"{source or table} rows[{i}]"
        if not isinstance(row, dict):
            raise RowError(f"{where}: a row is an object")
        CHECKS[table](row, where)
    return rows


def load_rows(path: Path) -> tuple:
    """(table, rows) of one results file, checked."""
    import json

    table = table_of(path.name)
    if not table:
        raise RowError(
            f"{path.name}: name it after its table: {sorted(t + FILE_SUFFIX for t in UPLOADABLE)}"
        )
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise RowError(f"{path.name}: not JSON: {err}") from None
    rows = data.get("rows") if isinstance(data, dict) else None
    return table, validate_rows(table, rows, path.name)


# ── JSON Schemas, generated from the table classes ──

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"
_STR = {"type": "string"}
_NUM = {"type": "number"}
_STRMAP = {"type": "object", "additionalProperties": {"type": ["string", "number"]}}
_TAGS = {"type": "array", "items": _STR}


def _row_schema(table: str) -> dict:
    """The upload row for an uploadable table."""
    if table == "test_case_runs":
        return {"type": "object", "required": ["name", "status"], "additionalProperties": False, "properties": {
            "classname": {**_STR, "description": "The test's class or module; part of its identity."},
            "name": {**_STR, "minLength": 1, "description": "The test's name; part of its identity."},
            "status": {"enum": OUTCOMES},
            "duration_s": {"type": "number", "minimum": 0},
            "fail_message": {**_STR, "description": "Truncated to 8192 characters."},
            "tags": {**_TAGS, "description": "namespace__value tags. Identity tags name the test; platform__/testtype__/cadence__ describe this run."},
            "measurements": {"type": "object", "additionalProperties": _NUM, "description": "Numbers this run measured."},
            "props": {**_STRMAP, "description": "result.* values recorded verbatim."}}}  # fmt: skip
    if table == "benchmark_runs":
        return {"type": "object", "required": ["name", "measurements"], "additionalProperties": False, "properties": {
            "name": {**_STR, "minLength": 1, "description": "The benchmark's name; part of its identity."},
            "backend": {**_STR, "description": "spyre, cpu, ...: the axis two runs of one benchmark compare across. Not identity."},
            "measurements": {"type": "object", "minProperties": 1, "additionalProperties": {"oneOf": [_NUM, {"type": "array", "minItems": 1, "items": _NUM}]},
                             "description": "metric -> a number or a list of samples."},
            "iterations": {"type": "integer", "minimum": 0, "description": "The n behind a pre-averaged number."},
            "tags": _TAGS,
            "props": {**_STRMAP, "description": "Describe the benchmark. Every prop is identity unless `identity` lists a subset."},
            "identity": {"type": "array", "items": _STR, "description": "Which props are identity. Omitted: all of them. [] : the name alone."},
            "run_props": {**_STRMAP, "description": "Describe this run of the benchmark (never identity)."}}}  # fmt: skip
    if table == "capability_runs":
        return {"type": "object", "required": ["test_type", "subject", "name", "status"], "additionalProperties": False, "properties": {
            "test_type": {"enum": sorted(schema.CAPABILITY_TYPE_VALUES)},
            "subject": {**_STR, "minLength": 1, "description": "What the capability belongs to (a model, a config)."},
            "name": {**_STR, "minLength": 1},
            "status": {"enum": CAPABILITY_STATUSES},
            "backend": {**_STR, "description": "spyre or cpu; a column, not identity."},
            "fail_reason": _STR, "arch": _STR, "tags": _TAGS,
            "sig": {**_STRMAP, "description": "Keys that, with subject and name, make the capability unique. Hashed."},
            "props": _STRMAP}}  # fmt: skip
    return {"type": "object", "required": ["test_type", "state"], "additionalProperties": False, "properties": {
        "test_type": {"enum": sorted(schema.TEST_TYPE_VALUES)},
        "state": {"enum": ["passed", "failed", "error"]},
        "result_kind": {"enum": sorted(schema.RESULT_KIND_VALUES), "description": "Derived from test_type when omitted."},
        "arch": _STR, "duration_s": {"type": "number", "minimum": 0}, "props": _STRMAP}}  # fmt: skip


def _column_type(name: str, table) -> dict:
    """A reference schema for one stored column, from the class's enums and required list."""
    node: dict = {}
    for col, allowed in table.enums:
        if col == name:
            node["enum"] = sorted(allowed)
    if name in table.required:
        node["minLength"] = 1
    return node


def json_schemas() -> dict:
    """{name: JSON Schema}: an upload file for each uploadable table, a reference for every table."""
    out = {}
    for table in UPLOADABLE:
        out[f"{table}.upload"] = {
            "$schema": _SCHEMA, "title": f"{table} upload file: results/{table}.json",
            "description": f"{UPLOADABLE[table][1]}. Identity fields ride in the row; ids are derived.",
            "type": "object", "required": ["rows"], "additionalProperties": False,
            "properties": {"rows": {"type": "array", "minItems": 1, "items": _row_schema(table)}}}  # fmt: skip
    for cls in vars(schema).values():
        if (
            isinstance(cls, type)
            and issubclass(cls, schema.Table)
            and cls is not schema.Table
            and cls.name
        ):
            out[f"{cls.name}.stored"] = {
                "$schema": _SCHEMA, "title": f"{cls.name}: the stored row",
                "description": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
                "x-uploadable": cls.name in UPLOADABLE,
                "type": "object", "additionalProperties": False, "required": list(cls.columns),
                "properties": {c: _column_type(c, cls) for c in cls.columns}}  # fmt: skip
    return out


def main(argv=None) -> int:
    """`python -m spyre_clickhouse_ingest.tables --out DIR`: write every generated JSON Schema."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument(
        "--out",
        required=True,
        help="the directory to write <name>.schema.json files into",
    )
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, node in sorted(json_schemas().items()):
        (out / f"{name}.schema.json").write_text(
            json.dumps(node, indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

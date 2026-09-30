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

# v1 model-ops: delete once the dashboard reads v2 capabilities
"""The JUnit capability verdicts name the same capabilities as the v1 log parser."""

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github" / "scripts"
for p in (ROOT / "extensions" / "clickhouse-ingest", ROOT / "tests" / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))

import model_ops_capability as cap  # noqa: E402
from spyre_clickhouse_ingest import insert_test_results  # noqa: E402
from spyre_clickhouse_ingest.identity import CapabilityId  # noqa: E402


def _parser():
    spec = importlib.util.spec_from_file_location(
        "parse_model_ops_logs", SCRIPTS / "parse_model_ops_logs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor(shape, dtype="torch.bfloat16", stride=None):
    return {"tensor": {"shape": shape, "dtype": dtype, "stride": stride}}


# Content-based test names (<op>__<8 digits>), as the tests now print them.
LOG = """\
2026-09-30T03:00:00.0000000Z collected 3 items
test_model_ops_v2.py::T::test_model_ops_db_torch_mul__11111111_spyre_bfloat16 XPASS [TAGS = model__m-1 torch.mul.1]
  [INPUT SHAPES]
  arg[0]: Tensor(shape=[1, 2], dtype=torch.bfloat16, stride=[2, 1])
  arg[1]: value=12.0

test_model_ops_v2.py::T::test_model_ops_db_torch_add__22222222_spyre_bfloat16 XFAIL [TAGS = model__m-1 torch.add.1]
  [INPUT SHAPES]
  arg[0]: Tensor(shape=[4], dtype=torch.float16, stride=[1])

test_model_ops_v2.py::T::test_model_ops_db_torch_arange__33333333_spyre_float16 XPASS [TAGS = model__m-1 torch.arange.1]
  [INPUT SHAPES]
  arg[0]: value=24

  FallbackWarning: aten.arange.default is falling back to cpu
===== 2 xpassed, 1 xfailed in 1.00s =====
"""

# The v1 classifications, read as v2 (status, backend).
_V1_VERDICT = {
    "spyre_enabled": ("passed", "spyre"),
    "not_implemented": ("not_implemented", "spyre"),
    "cpu_fallback": ("passed", "cpu"),
}


def _v1_verdicts(parsed: dict) -> set:
    """(capability_id, backend, status) for each variant the parser classified."""
    out = set()

    def _add(v: dict, classification: str):
        status, backend = _V1_VERDICT[classification]
        disc = {
            "input_shapes": json.dumps(v.get("input_shapes", [])),
            "input_dtypes": json.dumps(v.get("input_dtypes", [])),
        }
        cid = CapabilityId.derive(
            "torch-spyre", "model_ops", "m-1", v["operation"], disc, sorted(disc)
        )
        out.add((cid, backend, status))

    ops = parsed["operations"]
    for classification in ("spyre_enabled", "not_implemented"):
        for group in ops.get(classification, []):
            for v in group.get("variants", []):
                _add(v, classification)
    for group in ops.get("spyre_failed", []):
        for v in group.get("xpass_variants", []):
            _add(v, "spyre_enabled")
        for v in group.get("xfail_variants", []):
            _add(v, "not_implemented")
    for entry in ops.get("cpu_fallback", []):
        for v in entry.get("variants", []):
            _add(v, "cpu_fallback")
    return out


class _Client:
    def __init__(self):
        self.inserts = []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, [dict(zip(column_names, r)) for r in rows]))

    def query(self, sql, parameters=None):
        class R:
            result_rows = [(0,)]

        return R()


def _case(name, op, args, status, backend="spyre", fallback=""):
    props = {
        **cap.capability_properties(op, "m-1", f"{op}.1", args),
        "capability.backend": backend,
        "capability.prop.fallback_ops": fallback,
    }
    return {
        "classname": "T",
        "name": name,
        "status": status,
        "properties": [("tag", "platform__x86_64"), *props.items()],
    }


def test_content_based_names_parse_into_the_same_v1_variants():
    parsed = _parser().parse_log_file(LOG, "1", "M 1 Spyre", "m-1")
    ops = parsed["operations"]

    def names(kind):
        return {v["test_name"] for g in ops[kind] for v in g.get("variants", [])}

    assert names("spyre_enabled") == {
        "test_model_ops_db_torch_mul__11111111_spyre_bfloat16"
    }
    assert names("not_implemented") == {
        "test_model_ops_db_torch_add__22222222_spyre_bfloat16"
    }
    assert names("cpu_fallback") == {
        "test_model_ops_db_torch_arange__33333333_spyre_float16"
    }


def test_junit_verdicts_match_the_log_parser():
    parsed = _parser().parse_log_file(LOG, "1", "M 1 Spyre", "m-1")
    cases = [
        _case(
            "test_model_ops_db_torch_mul__11111111_spyre_bfloat16",
            "torch.mul",
            [_tensor([1, 2], stride=[2, 1]), {"value": 12.0}],
            "xpass",
        ),
        _case(
            "test_model_ops_db_torch_add__22222222_spyre_bfloat16",
            "torch.add",
            [_tensor([4], "torch.float16", [1])],
            "xfail",
        ),
        _case(
            "test_model_ops_db_torch_arange__33333333_spyre_float16",
            "torch.arange",
            [{"value": 24}],
            "xpass",
            backend="cpu",
            fallback="aten.arange.default",
        ),
    ]
    client = _Client()
    insert_test_results(client, "db", "torch-spyre", "r", cases, "m-1.xml")
    new = {
        (r["capability_id"], r["backend"], r["status"])
        for t, rows in client.inserts
        if t == "capability_runs"
        for r in rows
    }
    assert new == _v1_verdicts(parsed) and len(new) == 3

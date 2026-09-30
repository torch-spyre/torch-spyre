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

"""The capability.* properties model-ops tests record, and parity with the log parser."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github" / "scripts"
for p in (
    SCRIPTS,
    ROOT / "extensions" / "clickhouse-ingest",
    ROOT / "tests" / "models",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cap():
    import model_ops_capability

    return model_ops_capability


def _tensor(shape, dtype="torch.bfloat16", stride=None):
    return {"tensor": {"shape": shape, "dtype": dtype, "stride": stride}}


def test_signature_follows_the_parser_rules(cap):
    sig = cap.signature
    got = sig(
        "torch.mul", [_tensor([1, 41, 4096], stride=[167936, 4096, 1]), {"value": 12.0}]
    )
    assert got["input_shapes"] == ["[1,41,4096]"]
    assert got["input_strides"] == ["[167936,4096,1]"]
    assert got["input_dtypes"] == ["torch.bfloat16"]
    assert got["arg_values"] == ["12.0"]
    # The parser never reads a tensor printed without a stride, or a 0-d one.
    assert (
        sig("torch.add", [_tensor([4]), _tensor([], stride=[])])["input_shapes"] == []
    )
    assert sig("torch.Tensor.view", [{"value": [1, 12, -1, 128]}])["target_shape"] == (
        "[1, 12, -1, 128]"
    )
    assert sig("torch.zeros", [{"value": [1, 8]}])["input_shapes"] == ["[1,8]"]
    # A bare dimension-like value stands in for the missing tensor shapes.
    assert sig("torch.arange", [{"value": 24}])["input_shapes"] == ["[24]"]
    assert sig("torch.Tensor.__getitem__", [{"py": "(None, slice(None))"}])[
        "arg_values"
    ] == ["(None, slice(None))"]
    listed = sig(
        "torch.cat", [{"tensor_list": [_tensor([2], stride=[1])["tensor"]] * 2}]
    )
    assert listed["input_shapes"] == ["[2]", "[2]"]


def test_normalize_op_folds_aliases(cap):
    assert cap.normalize_op("torch.embedding") == "torch.nn.functional.embedding"
    assert cap.normalize_op("torch.index_copy") == "torch.index_copy_"
    assert cap.normalize_op("aten.cumsum.default") == "torch.cumsum"
    assert (
        cap.normalize_op("torch.nn_functional_linear") == "torch.nn.functional.linear"
    )


def test_capability_properties(cap):
    props = cap.capability_properties(
        "torch.mul", "m-1", "torch.mul.1", [_tensor([1, 2], stride=[2, 1])]
    )
    assert props == {
        "capability.test_type": "model_ops",
        "capability.subject": "m-1",
        "capability.name": "torch.mul",
        "capability.sig.input_shapes": '["[1,2]"]',
        "capability.sig.input_dtypes": '["torch.bfloat16"]',
        "capability.tag": "torch.mul.1",
        "capability.input_strides": '["[2,1]"]',
    }
    # The parser only ever counted torch.* operations.
    assert cap.capability_properties("operator.getitem", "m-1", "", []) == {}


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


class _Client:
    def __init__(self):
        self.inserts = []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, [dict(zip(column_names, r)) for r in rows]))

    def query(self, sql, parameters=None):
        class R:
            result_rows = [(0,)]

        return R()


def test_parity_with_the_log_parser(cap):
    from spyre_clickhouse_ingest import insert_test_results
    from spyre_clickhouse_ingest.identity import CapabilityId

    parser, ingest = _load("parse_model_ops_logs"), _load("ingest_model_ops")
    parsed = parser.parse_log_file(LOG, "1", "M 1 Spyre", "m-1")

    def _case(name, op, args, status, backend="spyre", fallback=""):
        props = {
            **cap.capability_properties(op, "m-1", f"{op}.1", args),
            "capability.backend": backend,
            "capability.fallback_ops": fallback,
        }
        return {
            "classname": "T",
            "name": name,
            "status": status,
            "properties": [("tag", "platform__x86_64"), *props.items()],
        }

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
    old = {
        (
            CapabilityId.derive(
                "torch-spyre",
                "model_ops",
                e["subject"],
                e["name"],
                e["disc"],
                sorted(e["disc"]),
            ),
            e["backend"],
            e["status"],
        )
        for e in ingest._v2_capability_results([parsed])
    }
    assert new == old and len(new) == 3

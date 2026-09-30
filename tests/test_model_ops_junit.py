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

"""Tests for .github/scripts/model_ops_junit.py, including parity with the log parser."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"
_CHLIB = Path(__file__).resolve().parents[1] / "extensions" / "clickhouse-ingest"
for p in (SCRIPTS, _CHLIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(name):
    sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def junit():
    return _load("model_ops_junit")


def _tensor(shape, dtype="torch.bfloat16", stride=None):
    return {"tensor": {"shape": shape, "dtype": dtype, "stride": stride}}


def _case(name, args, op, status="", backend="spyre", fallback="", model="m-1"):
    props = [("tag", f"model__{model}"), ("tag", "platform__x86_64")]
    if op:
        props += [
            ("result.op", op),
            ("result.variant", f"{op}.1"),
            ("result.args", json.dumps(args)),
            ("result.backend", backend),
            ("result.fallback_ops", fallback),
        ]
    body = "".join(
        f"<property name={quoteattr(k)} value={quoteattr(v)} />" for k, v in props
    )
    child = {
        "xfail": '<skipped type="pytest.xfail" message="expected failure" />',
        "skip": '<skipped type="pytest.skip" message="Filtered by --model" />',
        "fail": '<failure message="boom" />',
    }.get(status, "")
    return (
        f'<testcase classname="tests.models.test_model_ops_v2.T" name="{name}">'
        f"<properties>{body}</properties>{child}</testcase>"
    )


def _xml(tmp_path, *cases):
    path = tmp_path / "suite.xml"
    path.write_text(f"<testsuites><testsuite>{''.join(cases)}</testsuite></testsuites>")
    return path


def test_signature_follows_the_parser_rules(junit):
    sig = junit.signature
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


def test_normalize_op_folds_aliases(junit):
    assert junit.normalize_op("torch.embedding") == "torch.nn.functional.embedding"
    assert junit.normalize_op("torch.index_copy") == "torch.index_copy_"
    assert junit.normalize_op("aten.cumsum.default") == "torch.cumsum"
    assert (
        junit.normalize_op("torch.nn_functional_linear") == "torch.nn.functional.linear"
    )


def test_capability_results_from_recorded_properties(junit, tmp_path):
    t = [_tensor([1, 2], stride=[2, 1])]
    path = _xml(
        tmp_path,
        _case("a", t, "torch.mul", backend="cpu", fallback="aten.mul.Tensor"),
        _case("b", t, "torch.add", status="xfail", backend="cpu"),
        _case("c", t, "torch.sub", status="fail"),
        _case("d", t, "torch.sub", status="skip"),
        _case("dedupe", t, "", status="xfail"),
    )
    got = {r["props"]["test_name"]: r for r in junit.capability_results(path)}
    assert sorted(got) == ["a", "b", "c"]
    assert (got["a"]["status"], got["a"]["backend"]) == ("passed", "cpu")
    assert got["a"]["props"]["fallback_ops"] == "aten.mul.Tensor"
    # Only a pass says where the op ran.
    assert (got["b"]["status"], got["b"]["backend"]) == ("not_implemented", "spyre")
    assert got["c"]["status"] == "failed"
    assert got["a"]["subject"] == "m-1"
    assert got["a"]["tags"] == ["torch.mul.1"]
    assert got["a"]["disc"] == {
        "input_shapes": '["[1,2]"]',
        "input_dtypes": '["torch.bfloat16"]',
    }


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


def test_parity_with_the_log_parser(junit, tmp_path):
    parser, ingest = _load("parse_model_ops_logs"), _load("ingest_model_ops")
    parsed = parser.parse_log_file(LOG, "1", "M 1 Spyre", "m-1-spyre")
    path = _xml(
        tmp_path,
        _case(
            "test_model_ops_db_torch_mul__11111111_spyre_bfloat16",
            [_tensor([1, 2], stride=[2, 1]), {"value": 12.0}],
            "torch.mul",
        ),
        _case(
            "test_model_ops_db_torch_add__22222222_spyre_bfloat16",
            [_tensor([4], "torch.float16", [1])],
            "torch.add",
            status="xfail",
        ),
        _case(
            "test_model_ops_db_torch_arange__33333333_spyre_float16",
            [{"value": 24}],
            "torch.arange",
            backend="cpu",
            fallback="aten.arange.default",
        ),
    )
    old = junit._verdicts(ingest._v2_capability_results([parsed]))
    assert junit._verdicts(junit.capability_results(path)) == old
    assert sum(old.values()) == 3

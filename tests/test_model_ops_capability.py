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

"""The capability.* properties model-ops tests record."""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "extensions" / "clickhouse-ingest", ROOT / "tests" / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))


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
        "capability.backend": "spyre",
        "capability.prop.input_strides": '["[2,1]"]',
    }
    # Only torch.* operations are capabilities.
    assert cap.capability_properties("operator.getitem", "m-1", "", []) == {}


def test_properties_satisfy_the_ingest_contract(cap):
    from spyre_clickhouse_ingest import capability_declaration

    props = cap.capability_properties(
        "torch.mul", "m-1", "torch.mul.1", [_tensor([1, 2], stride=[2, 1])]
    )
    props["capability.prop.fallback_ops"] = ""
    decl, problem = capability_declaration({"properties": list(props.items())})
    assert problem == "" and decl["unknown"] == []
    assert sorted(decl["sig"]) == ["input_dtypes", "input_shapes"]

# Copyright 2025 The Torch-Spyre Authors.
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
"""Unit tests for the model-ops YAML -> SampleInput conversion helpers."""

import pytest
import torch

from .runner import make_SampleInput

_CPU = torch.device("cpu")


def _case(op, *value_entries):
    """Build a minimal case dict: one f32 input tensor plus literal args."""
    inputs = [{"tensor": {"shape": [2, 3], "dtype": "torch.float32", "init": "rand"}}]
    inputs.extend({"value": v} for v in value_entries)
    return {"op": op, "inputs": inputs}


def _sample(case, test_device=_CPU):
    return make_SampleInput(case, 0, torch.float16, test_device)


@pytest.mark.parametrize(
    "spec,expected",
    [("torch.float32", torch.float32), ("torch.bfloat16", torch.bfloat16)],
)
def test_positional_dtype_string_becomes_dtype(spec, expected):
    """A ``value:`` dtype scalar must reach the op as a real torch.dtype.

    ``ast.literal_eval`` cannot parse ``torch.float32`` (it is an attribute
    expression, not a literal), so an unconverted string used to reach
    ``Tensor.to`` positionally and fail as ``Invalid device string``.
    """
    sample = _sample(_case("torch.to", spec))
    assert sample.args[0] is expected


def test_positional_dtype_survives_the_call():
    """The converted arg must actually drive the cast."""
    sample = _sample(_case("torch.to", "torch.bfloat16"))
    assert sample.input.to(*sample.args).dtype is torch.bfloat16


def test_device_and_dtype_positional_args_keep_their_roles():
    """``.to(device, dtype)``: the device is remapped, the dtype is resolved."""
    sample = _sample(_case("torch.to", "cuda:0", "torch.bfloat16"))
    assert sample.args == (_CPU, torch.bfloat16)


def test_cuda_device_string_is_remapped_to_test_device():
    sample = _sample(_case("torch.to", "cuda:0"))
    assert sample.args[0] == _CPU


def test_non_dtype_strings_are_left_alone():
    """Only ``torch.``-prefixed scalars are dtypes; other strings pass through."""
    sample = _sample(_case("torch._C._log_api_usage_once", "python.nn_module"))
    assert sample.args[0] == "python.nn_module"


def test_literal_tuple_strings_still_parse():
    """Guard the pre-existing ast.literal_eval path against regressions."""
    sample = _sample(_case("torch.reshape", "(1, 1, -1, 128)"))
    assert sample.args[0] == (1, 1, -1, 128)


def test_unknown_torch_dtype_fails_loudly():
    """A typo'd dtype must raise a clear error, not 'Invalid device string'."""
    with pytest.raises(ValueError, match="Unknown torch dtype"):
        _sample(_case("torch.to", "torch.bfloat32"))

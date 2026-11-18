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

"""Tests that torch.compile(dynamic=True) is refused on Spyre, and only that."""

import pytest
import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch._dynamo.exc import BackendCompilerFailed

from utils_inductor import DEVICE, expects_raise

REFUSAL = r"does not support: torch\.compile\(dynamic=True\)"
SHAPES = [(128, 64), (256, 128), (512, 64), (64, 256)]


@pytest.fixture(autouse=True)
def _reset_dynamo():
    dynamo.reset()
    yield
    dynamo.reset()


def _gelu(a):
    return F.gelu(a)


def _rand(shape):
    return torch.randn(*shape, dtype=torch.float16)


@expects_raise
def test_dynamic_true_refused_at_every_shape():
    compiled = torch.compile(_gelu, dynamic=True)
    # A refused compile caches nothing, so every call, including a repeated
    # shape, has to be refused again.
    for shape in SHAPES + SHAPES[:1]:
        with pytest.raises(BackendCompilerFailed, match=REFUSAL):
            compiled(_rand(shape).to(DEVICE))


@expects_raise
def test_refusal_names_the_supported_interface():
    with pytest.raises(BackendCompilerFailed) as excinfo:
        torch.compile(_gelu, dynamic=True)(_rand(SHAPES[0]).to(DEVICE))
    message = str(excinfo.value)
    assert 'x.to("spyre", dynamic=' in message
    assert "dynamic=None" in message


@expects_raise
def test_dynamic_true_refused_for_two_inputs():
    compiled = torch.compile(torch.add, dynamic=True)
    for shape in SHAPES[:3]:
        x = _rand(shape).to(DEVICE)
        with pytest.raises(BackendCompilerFailed, match=REFUSAL):
            compiled(x, x)


@expects_raise
def test_dynamic_true_refused_for_module_with_fullgraph():
    compiled = torch.compile(
        torch.nn.GELU(), backend="inductor", fullgraph=True, dynamic=True
    )
    with pytest.raises(BackendCompilerFailed, match=REFUSAL):
        compiled(_rand(SHAPES[0]).to(DEVICE))


@expects_raise
def test_assume_static_by_default_false_refused():
    # torch.compile(dynamic=True) only clears this switch, so setting it
    # directly is the same request and is refused the same way.
    with dynamo.config.patch(assume_static_by_default=False):
        with pytest.raises(BackendCompilerFailed, match=REFUSAL):
            torch.compile(_gelu)(_rand(SHAPES[0]).to(DEVICE))
    assert dynamo.config.assume_static_by_default is True


def test_default_dynamic_compiles_across_shapes():
    compiled = torch.compile(_gelu)
    for shape in SHAPES:
        x = _rand(shape)
        out = compiled(x.to(DEVICE)).cpu()
        torch.testing.assert_close(out.float(), F.gelu(x.float()), atol=1e-2, rtol=1e-2)


def test_dynamic_false_compiles_across_shapes():
    compiled = torch.compile(torch.add, dynamic=False)
    for shape in SHAPES:
        x = _rand(shape)
        x_dev = x.to(DEVICE)
        out = compiled(x_dev, x_dev).cpu()
        torch.testing.assert_close(out.float(), (x + x).float(), atol=1e-2, rtol=1e-2)


def test_mark_dynamic_is_not_refused():
    compiled = torch.compile(_gelu)
    for shape in SHAPES[:3]:
        x = _rand(shape)
        x_dev = x.to(DEVICE)
        dynamo.mark_dynamic(x_dev, 0)
        out = compiled(x_dev).cpu()
        torch.testing.assert_close(out.float(), F.gelu(x.float()), atol=1e-2, rtol=1e-2)


def test_cpu_compile_with_dynamic_true_is_unaffected():
    # Only Spyre graphs are refused.
    compiled = torch.compile(_gelu, dynamic=True)
    for shape in SHAPES:
        x = torch.randn(*shape)
        torch.testing.assert_close(compiled(x), F.gelu(x))

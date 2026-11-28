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

"""CPU tests: a function with a ``spyre_hint`` compiles under ``aot_compile``, which
calls the backend after the Dynamo trace has ended."""

import pytest
import torch
from torch._dynamo.backends.registry import lookup_backend

from torch_spyre._inductor.propagate_hints import spyre_hint


def fn(x):
    with spyre_hint(work_div={"M": 2}):
        y = x + 1
    with spyre_hint(named_dims=["M"]):
        return y * 2


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_plain_compile(backend):
    torch._dynamo.reset()
    x = torch.arange(4.0)
    assert torch.equal(torch.compile(fn, backend=backend, fullgraph=True)(x), fn(x))


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_aot_compile(backend):
    torch._dynamo.reset()
    x = torch.arange(4.0)
    compiled = torch.compile(fn, backend=backend, fullgraph=True).aot_compile(
        ((x,), {})
    )
    assert torch.equal(compiled(x), fn(x))


def _hint_keys(compile_):
    seen = []

    def backend(gm, example_inputs):
        seen.append(
            sorted(
                {
                    k
                    for node in gm.graph.nodes
                    for k in (node.meta.get("custom") or {})
                    if k.startswith("_hint_")
                }
            )
        )
        return lookup_backend("aot_eager")(gm, example_inputs)

    torch._dynamo.reset()
    compile_(torch.compile(fn, backend=backend, fullgraph=True), torch.arange(4.0))
    return seen


def test_aot_compile_keeps_the_hints():
    plain = _hint_keys(lambda f, x: f(x))
    aot = _hint_keys(lambda f, x: f.aot_compile(((x,), {}))(x))
    assert plain == aot == [["_hint_0", "_hint_1"]]

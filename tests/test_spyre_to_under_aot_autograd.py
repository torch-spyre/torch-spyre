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

"""Tensor.to traced by AOTAutograd outside Dynamo (as aot_compile runs it).

aot_compile calls the backend after the Dynamo trace has ended, so AOTAutograd
traces with is_compiling() False and with FunctionalTensor inputs. A dtype
change on a Spyre tensor must still trace as the plain compile traces it
(its plain ops, not as the eager to_dtype_d2d op.
"""

import torch
from functorch.compile import aot_function, make_boxed_func
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_spyre  # noqa: F401


def _fn(x):
    return (x.to(torch.float32) * 2).to(torch.float16)


def _traced_targets(fn, device):
    graphs = []

    def record(gm, example_inputs):
        graphs.append(
            [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        )
        return make_boxed_func(gm.forward)

    with FakeTensorMode():
        x = torch.empty(4, 64, dtype=torch.float16, device=device)
        aot_function(fn, fw_compiler=record)(x)
    return graphs


def test_dtype_change_traces_as_convert_element_type():
    (targets,) = _traced_targets(_fn, "spyre")
    assert not any("to_dtype_d2d" in t for t in targets), targets
    # the same ops as for a CPU tensor, which Tensor.to does not reroute
    assert targets == _traced_targets(_fn, "cpu")[0]

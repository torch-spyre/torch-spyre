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

from contextlib import contextmanager

import torch
from torch._dynamo.backends.common import AotAutograd
from torch._inductor.virtualized import V


@contextmanager
def spyre_data_types():
    saved = torch._prims_common._computation_dtype_map
    torch._prims_common._computation_dtype_map = {
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.complex32: torch.complex32,
    }
    try:
        yield
    finally:
        torch._prims_common._computation_dtype_map = saved


class SpyreAotAutograd(AotAutograd):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs, **kwargs):
        if any(
            isinstance(t, torch.Tensor) and t.device.type == "spyre"
            for t in example_inputs
        ):
            with (
                spyre_data_types(),
                V.set_real_inputs(example_inputs),
            ):
                return super().__call__(gm, example_inputs, **kwargs)
        else:
            return super().__call__(gm, example_inputs, **kwargs)

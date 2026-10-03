# Copyright 2026 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import pytest
import torch

from torch_spyre._inductor.wsr import for_each_tile


@pytest.mark.parametrize("axis", [0, -1])
def test_pytree_map_outputs_reuse_graph(axis):
    graphs = []

    def capture(gm, _):
        graphs.append(gm)
        return gm.forward

    def kernel(x):
        def body(_, tiles):
            (tile,) = tiles
            return None, (tile + 1, {"sums": tile.sum(dim=1 if axis == 0 else 0)})

        return for_each_tile(body, (x,), dims=(axis,), tile_size=32, out_dim=axis)[1]

    torch._dynamo.reset()
    with (
        patch("torch.accelerator.is_available", return_value=False),
        torch.inference_mode(),
    ):
        compiled = torch.compile(kernel, backend=capture, fullgraph=True, dynamic=False)
        for step in range(2):
            x = torch.arange(96 * 64, dtype=torch.float32).reshape(96, 64) + step
            shifted, stats = compiled(x)
            torch.testing.assert_close(shifted, x + 1)
            torch.testing.assert_close(stats["sums"], x.sum(dim=1 if axis == 0 else 0))
    assert len(graphs) == 1

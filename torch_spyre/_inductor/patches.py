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
from torch._inductor.graph import GraphLowering

from .lowering import enable_spyre_lowerings
from .decompositions import (
    spyre_decompositions,
    spyre_decompositions_to_exclude,
    enable_spyre_decompositions,
)
from torch_spyre.fallbacks import fallback_ops
from torch._inductor.decomposition import decompositions


def _should_run_on_spyre(
    graph_inputs: torch.Tensor = [], graph: torch.fx.graph.Graph = None
):
    # Check if example inputs exists and whether one of them is on the spyre device
    if any(
        isinstance(t, torch.Tensor) and t.device.type == "spyre" for t in graph_inputs
    ):
        return True

    # Check the example_values of the last "real" node of the graph whether it resides on the spyre device
    if (
        graph is not None
        and graph.output_node().prev.meta.get("example_value", None) is not None
        and graph.output_node().prev.meta.get("example_value", None).device.type
        == "spyre"
    ):
        return True

    # Check the kwargs of the last "real" node of the graph whether it resides on the spyre device
    if (
        graph is not None
        and "device" in graph.output_node().prev.kwargs
        and (
            (
                isinstance(graph.output_node().prev.kwargs["device"], str)
                and graph.output_node().prev.kwargs["device"] == "spyre"
            )
            or (
                isinstance(graph.output_node().prev.kwargs["device"], torch.device)
                and graph.output_node().prev.kwargs["device"].type == "spyre"
            )
        )
    ):
        return True

    # If the spyre device could not be detected until now, fallback to the CPU device
    return False


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
        if _should_run_on_spyre(example_inputs, gm.graph):
            # Merge Spyre-specific decompositions with any existing decompositions
            # Note: the decompositions additionally need to be merged in this way,
            # which is not required for the lowerings.
            # The reason is that PyTorch maintains a separate
            # CURRENT_DECOMPOSITION_TABLE in torch.fx.experimental.proxy_tensor
            # During FX tracing.
            # AotAutograd reads decompositions from self.kwargs["decompositions"] and
            # thus using the kwargs of the compile process will ensure that the
            # spyre-specific decompositions are loaded correctly
            existing_decomps = self.kwargs.get("decompositions", {})
            if callable(existing_decomps):
                existing_decomps = existing_decomps()

            # Remove the selected decompositions from Inductor's registry for Spyre.
            torch._decomp.remove_decompositions(
                existing_decomps, spyre_decompositions_to_exclude
            )
            torch._decomp.remove_decompositions(
                decompositions, spyre_decompositions_to_exclude
            )

            # Remove decompositions for fallback ops defined in fallbacks.py
            torch._decomp.remove_decompositions(existing_decomps, fallback_ops)
            torch._decomp.remove_decompositions(decompositions, fallback_ops)

            # Spyre decompositions take precedence over existing ones
            merged_decomps = {**existing_decomps, **spyre_decompositions}
            self.kwargs["decompositions"] = merged_decomps

            with (
                spyre_data_types(),
                enable_spyre_lowerings(),
                enable_spyre_decompositions(),
                V.set_real_inputs(example_inputs),
            ):
                return super().__call__(gm, example_inputs, **kwargs)
        else:
            return super().__call__(gm, example_inputs, **kwargs)


def spyre_compile_to_module(graph: GraphLowering, original_compile_to_module):
    if _should_run_on_spyre(graph.example_inputs, graph.graph):
        # with spyre_data_types(), enable_spyre_lowerings():
        with (
            spyre_data_types(),
            enable_spyre_lowerings(),
            enable_spyre_decompositions(),
        ):
            return original_compile_to_module(graph)
    else:
        return original_compile_to_module(graph)

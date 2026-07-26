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
from functools import wraps

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import InputType
from torch._inductor.virtualized import V
from typing import Callable, Optional


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


@contextmanager
def enable_spyre_context(
    example_inputs: list[InputType],
    decomps: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
):
    """
    Context manager that sets up the complete Spyre compilation environment.

    This CM configures PyTorch Inductor to compile graphs for the Spyre device by:
      - Enabling Spyre-specific data type handling
      - Activating Spyre lowerings and decompositions
      - Configuring Inductor settings optimized for Spyre
      - Setting up custom pre/post compilation passes
      - Disabling incompatible optimizations (e.g., reduction splitting, permute fusion)

    Args:
        example_inputs: List of example inputs to the graph being compiled. Used to
            set real inputs in the virtualized context for shape inference and
            optimization decisions.
        decomps: Decomposition table to be populated with Spyre-specific
            decompositions. Maps operator overloads to their decomposition implementations.
            This is typically a clone of PyTorch Inductor's global decomposition registry.
    """

    if decomps is None:
        decomps = torch._inductor.decomposition.decompositions

    from torch_spyre._inductor.lowering import enable_spyre_lowerings  # your CM

    # Ensure decorators run (custom ops/decomp/lowerings modules)
    import torch_spyre._inductor.customops  # noqa: F401
    from torch_spyre._inductor.decompositions import (
        enable_spyre_decompositions,
    )

    import torch_spyre._inductor.lowering  # noqa: F401
    from torch_spyre._inductor.choices import SpyreHeuristics
    from torch_spyre._inductor.passes import (
        CustomPreGradPasses,
        CustomPrePasses,
        CustomPostPasses,
        CustomPreFusionPasses,
        CustomPostFusionPasses,
        CustomPreSchedulingPasses,
    )

    # *) Inductor config tweaks (saved/restored)
    new_config = {
        "split_reductions": False,
        "benchmark_harness": False,
        "pre_grad_custom_pass": CustomPreGradPasses(),
        "post_grad_custom_pre_pass": CustomPrePasses(),
        "post_grad_custom_post_pass": CustomPostPasses(),
        "_pre_fusion_custom_pass": CustomPreFusionPasses(),
        "_post_fusion_custom_pass": CustomPostFusionPasses(),
        # Adding this configuration in so as to avoid the optimization of turning small matmuls into non-matmuls
        # found here: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py#L1580
        "unroll_reductions_threshold": 1,
        # Disable fusing of mm + permute/transpose for now.
        "permute_fusion": False,
        "allow_buffer_reuse": False,  # For now, as buffer reuse does not consider stride_map.
    }

    from torch._inductor.ir import Loops

    # Force all operations to be realized when LoopLevel IR is initially constructed
    old_loop = Loops.has_large_inner_fn
    Loops.has_large_inner_fn = lambda self, threshold=None: True

    from torch._inductor.fx_passes import joint_graph

    origin_pass = list(joint_graph.pass_patterns)
    # disable mul_softmax_pattern and div_softmax_pattern for now
    joint_graph.pass_patterns.pop()

    # Inject the pre_scheduling_passes before the Scheduler is constructed,
    # allowing the passes to modify the graph IR (buffers, inputs, constants).
    old_update_scheduler = GraphLowering._update_scheduler

    _pre_scheduling_pass = CustomPreSchedulingPasses()

    def _spyre_update_scheduler(self: GraphLowering) -> None:
        _pre_scheduling_pass(self)
        old_update_scheduler(self)

    GraphLowering._update_scheduler = _spyre_update_scheduler  # type: ignore[method-assign]

    with (
        spyre_data_types(),
        enable_spyre_lowerings(),
        enable_spyre_decompositions(decomps=decomps) as spyre_context_decompositions,
        V.set_real_inputs(example_inputs),
        V.set_choices_handler(SpyreHeuristics()),
        torch._inductor.config.patch(new_config),
    ):
        try:
            yield spyre_context_decompositions
        finally:
            joint_graph.pass_patterns[:] = origin_pass
            Loops.has_large_inner_fn = old_loop
            GraphLowering._update_scheduler = old_update_scheduler  # type: ignore[method-assign]


OBSERVER_HOOKS_KEY = "__spyre_hooks_meta"


def patch_inductor_fusions():
    import torch._inductor.fx_passes.post_grad

    # disable addmm fusion. The fusion will be undone by the decomposition that is
    # registered in torch-spyre, but the hints are lost in the process
    addmm_fusion_found = False
    for entries in torch._inductor.fx_passes.post_grad.pass_patterns[
        2
    ].patterns.values():
        for entry in entries:
            if (
                entry.extra_check
                == torch._inductor.fx_passes.post_grad.is_valid_addmm_fusion
            ):
                entry.extra_check = lambda x: False
                addmm_fusion_found = True

    assert addmm_fusion_found, (
        "Couldn't find addmm fusion. This patch needs to be reviewed."
    )

    # Install observer patch
    from torch.fx.passes.graph_transform_observer import GraphTransformObserver

    _original = GraphTransformObserver.apply_graph_pass

    @wraps(GraphTransformObserver.apply_graph_pass)
    def apply_graph_pass(self, pass_fn):
        meta = self.gm.meta.get(OBSERVER_HOOKS_KEY, {})
        self.gm.meta[OBSERVER_HOOKS_KEY] = meta
        meta["pass"] = self.passname
        meta["subsystem"] = self.subsystem
        try:
            return _original(self, pass_fn)
        finally:
            meta.pop("pass", None)
            meta.pop("subsystem", None)

    GraphTransformObserver.apply_graph_pass = apply_graph_pass

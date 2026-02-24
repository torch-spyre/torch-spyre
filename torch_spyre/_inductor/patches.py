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


@contextmanager
def enable_spyre_context(example_inputs):
    from torch_spyre._inductor.lowering import enable_spyre_lowerings  # your CM

    # Ensure decorators run (custom ops/decomp/lowerings modules)
    import torch_spyre._inductor.customops  # noqa: F401
    import torch_spyre._inductor.decompositions  # noqa: F401
    import torch_spyre._inductor.lowering  # noqa: F401
    from torch_spyre._inductor.choices import SpyreHeuristics
    from torch_spyre._inductor.passes import (
        CustomPrePasses,
        CustomPostPasses,
        scheduler_passes,
        _maybe_run_scheduler_pass,
    )

    # *) Inductor config tweaks (saved/restored)
    import torch._inductor.config as inductor_config

    saved_config = {
        "split_reductions": inductor_config.split_reductions,
        "benchmark_harness": inductor_config.benchmark_harness,
        "post_grad_custom_pre_pass": inductor_config.post_grad_custom_pre_pass,
        "post_grad_custom_post_pass": inductor_config.post_grad_custom_post_pass,
        "_pre_fusion_custom_pass": inductor_config._pre_fusion_custom_pass,
        "unroll_reductions_threshold": inductor_config.unroll_reductions_threshold,
        "permute_fusion": inductor_config.permute_fusion,
    }
    inductor_config.split_reductions = False
    inductor_config.benchmark_harness = False
    inductor_config.post_grad_custom_pre_pass = CustomPrePasses()
    inductor_config.post_grad_custom_post_pass = CustomPostPasses()
    inductor_config._pre_fusion_custom_pass = lambda nodes: _maybe_run_scheduler_pass(
        scheduler_passes, nodes
    )
    # Adding this configuration in so as to avoid the optimization of turning small matmuls into non-matmuls
    # found here: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py#L1580
    inductor_config.unroll_reductions_threshold = 1
    # Disable fusing of mm + permute/transpose for now.
    inductor_config.permute_fusion = False

    from torch._inductor.ir import Loops

    # Force all operations to be realized when LoopLevel IR is initially constructed
    old_loop = Loops.has_large_inner_fn
    Loops.has_large_inner_fn = lambda self, threshold=None: True

    from torch._inductor.fx_passes import joint_graph

    origin_pass = list(joint_graph.pass_patterns)
    # disable mul_softmax_pattern and div_softmax_pattern for now
    joint_graph.pass_patterns.pop()

    with (
        spyre_data_types(),
        enable_spyre_lowerings(),
        V.set_real_inputs(example_inputs),
        V.set_choices_handler(SpyreHeuristics()),
    ):
        try:
            yield
        finally:
            joint_graph.pass_patterns[:] = origin_pass
            Loops.has_large_inner_fn = old_loop
            # restore configs
            for k, v in saved_config.items():
                setattr(inductor_config, k, v)

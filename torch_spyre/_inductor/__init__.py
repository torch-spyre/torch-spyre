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

from .constants import DEVICE_NAME

import threading
from functools import wraps

_autoload_lock = threading.Lock()


class Unsupported(RuntimeError):
    def __init__(self, thing) -> None:
        super().__init__(f"Spyre backend does not support: {thing}")


def enable_spyre_compile_fx_wrapper():
    import torch._inductor.compile_fx as cfx
    import torch.fx as fx
    import torch

    if getattr(cfx, "_spyre_wrapped", False):
        return
    with _autoload_lock:
        if getattr(cfx, "_spyre_wrapped", False):
            return
        _orig = cfx.compile_fx

        # Iterate over producer nodes (supports nested containers of nodes)
        def iter_nodes(x):
            if isinstance(x, fx.Node):
                yield x
            elif isinstance(x, (tuple, list)):
                for e in x:
                    yield from iter_nodes(e)
            elif isinstance(x, dict):
                for e in x.values():
                    yield from iter_nodes(e)

        def iter_tensors(v):
            if isinstance(v, torch.Tensor):
                yield v  # FakeTensor is a Tensor subclass, so this works
            elif isinstance(v, (tuple, list)):
                for e in v:
                    yield from iter_tensors(e)
            elif isinstance(v, dict):
                for e in v.values():
                    yield from iter_tensors(e)

        def _uses_spyre(gm, example_inputs, device_name=DEVICE_NAME) -> bool:
            # Inputs
            if any(
                isinstance(x, torch.Tensor)
                and getattr(x.device, "type", None) == device_name
                for x in (example_inputs or ())
            ):
                return True
            # Output
            out_node = gm.graph.output_node()
            out_puts = out_node.args[0] if out_node.args else []
            for n in iter_nodes(out_puts):
                meta = getattr(n, "meta", {}) or {}
                mv = meta.get("val", None) or meta.get("example_value", None)
                if mv is None:
                    continue

                for t in iter_tensors(mv):
                    if getattr(getattr(t, "device", None), "type", None) == device_name:
                        return True

            # Graph nodes (covers tensorless factories)
            for n in gm.graph.nodes:
                dev = n.kwargs.get("device")
                if dev is None:
                    continue
                if isinstance(dev, torch.device) and dev.type == device_name:
                    return True
                if isinstance(dev, str) and dev.split(":")[0] == device_name:
                    return True
            return False

        @wraps(_orig)
        def _wrapper(gm, example_inputs, *args, **kwargs):
            if _uses_spyre(gm, example_inputs):
                import torch

                torch.spyre._impl._lazy_init()
            return _orig(gm, example_inputs, *args, **kwargs)

        cfx.compile_fx = _wrapper
        cfx._spyre_wrapped = True


def _light_autoload():
    enable_spyre_compile_fx_wrapper()


def _autoload():
    if getattr(_autoload, "_ran", False):
        return

    with _autoload_lock:
        if getattr(_autoload, "_ran", False):
            return
        from torch._dynamo.device_interface import register_interface_for_device

        from torch_spyre.utils.device_interface import SpyreInterface

        register_interface_for_device(DEVICE_NAME, SpyreInterface)

        from torch._inductor.codegen.common import (
            register_backend_for_device,
            register_device_op_overrides,
        )

        # Register in-tree CPU and CUDA device
        from torch._inductor.codegen import cpu_device_op_overrides  # noqa: F401  # usort: skip
        from torch._inductor.codegen.cuda import device_op_overrides  # noqa: F401  # usort: skip

        from torch_spyre.utils.device_op_overrides import SpyreDeviceOpOverrides

        register_device_op_overrides(
            device=DEVICE_NAME, device_op_overrides=SpyreDeviceOpOverrides()
        )

        from .dsc import SuperDSCScheduling
        from .wrapper import SpyrePythonWrapperCodegen

        register_backend_for_device(
            DEVICE_NAME, SuperDSCScheduling, SpyrePythonWrapperCodegen
        )

        # Set all the appropriate state on PyTorch
        import torch

        # Define Spyre-specific custom ops, decompositions, and lowerings
        import torch_spyre._inductor.customops  # noqa: F401  # usort: skip
        import torch_spyre._inductor.decompositions  # noqa: F401  # usort: skip
        import torch_spyre._inductor.lowering  # noqa: F401  # usort: skip
        from .patches import SpyreAotAutograd, spyre_compile_to_module

        # Monkey patching these methods let us install Spyre-specific overrides
        # and contexts that are not supported by existing extension points.
        # We need to hook both, because a user may directly compile a module for spyre without going through AotAutograd.
        torch._dynamo.backends.common.aot_autograd = lambda **kwargs: SpyreAotAutograd(
            **kwargs
        )
        orig_compile_to_module = torch._inductor.graph.GraphLowering.compile_to_module
        torch._inductor.graph.GraphLowering.compile_to_module = (
            lambda graph: spyre_compile_to_module(graph, orig_compile_to_module)
        )
        import torch._inductor.compile_fx  # noqa: F401  # usort: skip

        # This overwrites the copy of `aot_autograd` imported by compile_fx.py to use our monkey patch.
        torch._inductor.compile_fx.aot_autograd = (
            torch._dynamo.backends.common.aot_autograd
        )

        # Customize inductor heuristics
        from .choices import SpyreHeuristics

        torch._inductor.virtualized.V.set_choices_handler(SpyreHeuristics())

        # Customize inductor configuration
        from .passes import (
            CustomPrePasses,
            CustomPostPasses,
            scheduler_passes,
            _maybe_run_scheduler_pass,
        )

        torch._inductor.config.split_reductions = False
        torch._inductor.config.benchmark_harness = False
        torch._inductor.config.post_grad_custom_pre_pass = CustomPrePasses()
        torch._inductor.config.post_grad_custom_post_pass = CustomPostPasses()
        torch._inductor.config._pre_fusion_custom_pass = (
            lambda nodes: _maybe_run_scheduler_pass(scheduler_passes, nodes)
        )
        # Adding this configuration in so as to avoid the optimization of turning small matmuls into non-matmuls
        # found here: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py#L1580
        torch._inductor.config.unroll_reductions_threshold = 1

        from torch._inductor.ir import Loops

        # Force all operations to be realized when LoopLevel IR is initially constructed
        Loops.has_large_inner_fn = lambda self, threshold=None: True

        from torch._inductor.fx_passes import joint_graph

        # disable mul_softmax_pattern and div_softmax_pattern for now
        joint_graph.pass_patterns.pop()

        # Disable fusing of mm + permute/transpose for now.
        torch._inductor.config.permute_fusion = False
        _autoload._ran = True

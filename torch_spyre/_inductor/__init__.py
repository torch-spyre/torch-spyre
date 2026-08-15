# Copyright 2025-2026 The Torch-Spyre Authors.
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
from .patches import enable_spyre_context, patch_inductor_fusions
from . import config

import threading
from functools import wraps
from typing import Any

from .propagate_hints import spyre_hint, get_op_hints  # noqa: F401
from torch_spyre.profiler._ffdc import CATEGORY_COMPILE_FRONTEND, try_collect

_autoload_lock = threading.Lock()


def _spyre_inner_compile(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around ``compile_fx_inner`` that pins a picklable ``get_decomp_fn``.

    Background: passing ``decompositions=<dict>`` to ``compile_fx`` causes it
    to wrap the dict in a local ``def get_decomp_fn`` closure (compile_fx.py).
    That closure is unpicklable, so the FX graph cache silently bypasses
    itself with ``BypassFxGraphCache("Failed to pickle cache key")``.

    Two-stage decomposition design (these are not contradictory):

    * Outer stage — ``enable_spyre_compile_fx_wrapper``'s ``_wrapper`` DOES pass
      ``decompositions=get_spyre_decomp_table()`` to ``compile_fx``. That dict
      only feeds AOTAutograd's joint-graph decomposition; it is consumed before
      the FX graph cache key is built, so its unpicklable ``get_decomp_fn``
      closure is never part of the cache key.

    * Inner stage — this wrapper (installed as ``inner_compile``) never receives
      ``decompositions=``. Instead it clobbers ``get_decomp_fn`` at call time
      with the module-level ``get_spyre_decomp_table`` — a picklable,
      name-resolvable callable — so the post-AOT inner compile decomposes with
      the same table while keeping the FX graph cache key picklable.

    NOTE: We are working on improving this in upstream PyTorch
    """
    from torch._inductor.compile_fx import compile_fx_inner
    from torch_spyre._inductor.decompositions import get_spyre_decomp_table

    kwargs["get_decomp_fn"] = get_spyre_decomp_table
    return compile_fx_inner(*args, **kwargs)


def enable_spyre_compile_fx_wrapper():
    import torch._inductor.compile_fx as cfx
    import torch.fx as fx
    import torch

    if getattr(cfx, "_spyre_wrapped", False):
        return
    with _autoload_lock:
        if getattr(cfx, "_spyre_wrapped", False):
            return

        patch_inductor_fusions()

        _orig = cfx.compile_fx
        from torch_spyre._inductor.logging_utils import get_inductor_logger

        logger = get_inductor_logger("compile_fx_wrapper")

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

                if any(
                    getattr(getattr(t, "device", None), "type", None) == device_name
                    for t in iter_tensors(mv)
                ):
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
            uses_spyre = _uses_spyre(gm, example_inputs)

            try:
                if uses_spyre:
                    torch.spyre._impl._lazy_init()
                    # AOTAutograd uses the dict passed via ``decompositions=``
                    # to decompose the joint graph; Spyre-specific
                    # decompositions must be applied at this stage so ops like
                    # aten.logical_not / aten.ceil / aten.sign are reduced to
                    # primitives the Spyre OpFuncs handler implements.
                    from torch_spyre._inductor.decompositions import (
                        get_spyre_decomp_table,
                    )

                    kwargs.setdefault("decompositions", get_spyre_decomp_table())
                    # Route inner compilation through _spyre_inner_compile,
                    # which re-binds ``get_decomp_fn`` to a picklable
                    # module-level callable so the FX graph cache key stays
                    # serializable.
                    kwargs.setdefault("inner_compile", _spyre_inner_compile)
                    with enable_spyre_context(example_inputs):
                        return _orig(gm, example_inputs, *args, **kwargs)

                # Non-Spyre graphs: no FFDC — avoids capturing unrelated CPU
                # compiles.
                return _orig(gm, example_inputs, *args, **kwargs)
            except Exception as exc:
                if uses_spyre:
                    try_collect(
                        exc, logger=logger, failure_category=CATEGORY_COMPILE_FRONTEND
                    )
                raise

        cfx.compile_fx = _wrapper
        cfx._spyre_wrapped = True


def _light_autoload():
    from . import decompositions  # noqa: F401
    from . import distributed as _distributed_init  # noqa: F401  registers spyre::broadcast_async/wait_work

    enable_spyre_compile_fx_wrapper()


def _autoload():
    if getattr(_autoload, "_ran", False):
        return

    with _autoload_lock:
        if getattr(_autoload, "_ran", False):
            return
        from torch._dynamo.device_interface import register_interface_for_device

        from torch_spyre.device.interface import SpyreInterface

        register_interface_for_device(DEVICE_NAME, SpyreInterface)

        from torch._inductor.codegen.common import (
            register_backend_for_device,
            register_device_op_overrides,
        )

        # Register in-tree CPU and CUDA device
        from torch._inductor.codegen import cpu_device_op_overrides  # noqa: F401  # usort: skip
        from torch._inductor.codegen.cuda import device_op_overrides  # noqa: F401  # usort: skip

        from torch_spyre.device.op_overrides import SpyreDeviceOpOverrides

        register_device_op_overrides(
            device=DEVICE_NAME, device_op_overrides=SpyreDeviceOpOverrides()
        )

        from .scheduler import SuperDSCScheduling
        from .wrapper import SpyrePythonWrapperCodegen

        register_backend_for_device(
            DEVICE_NAME,
            SuperDSCScheduling,
            SpyrePythonWrapperCodegen,
            device_custom_config=config,
        )

        _autoload._ran = True

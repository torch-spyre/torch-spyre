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

import os
import threading

import torch
from functools import wraps
from typing import Any

from .propagate_hints import spyre_hint, get_op_hints  # noqa: F401
from torch_spyre.profiler._ffdc import CATEGORY_COMPILE_FRONTEND, try_collect

_autoload_lock = threading.Lock()

# Set for the duration of a Spyre ``compile_fx`` call (the ``_wrapper`` Spyre
# branch below). Read by the invoke_subgraph decomposition monkey-patch
# (torch_spyre/_monkey_patch.py) to decide whether a nested_compile_region /
# invoke_subgraph HOP body that inherits its decompositions (config is None)
# should be re-traced with the Spyre decomp table. The HOP body is re-traced
# INSIDE the compile_fx call (AOTAutograd / proxy tracing), so this flag is
# active exactly when the extract runs. Thread-local because compiles can run
# concurrently and this must not leak the Spyre table into an unrelated
# CPU-only compile on another thread.
_compile_state = threading.local()


def in_spyre_compile() -> bool:
    """True while a Spyre ``compile_fx`` is on the stack for this thread."""
    return getattr(_compile_state, "in_spyre_compile", False)


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


def _declared_dims_of(
    gm, example_inputs, reserved_dims_of=None
) -> dict[str, tuple[int, int, int]]:
    """{symbol name: (min, max, granularity)} for every graph input dimension
    declared with ``tensor.to("spyre", dynamic=...)``.

    A Dynamo placeholder carries the real input (``meta["grapharg"].example``),
    which is where ``get_reserved_dims`` reads the reservation, and the fake
    input (``meta["example_value"]``), whose symbolic size names the ShapeEnv
    symbol this compile will see. Both come from one GraphArg, so they pair
    without positional zipping. A graph that did not come through Dynamo has no
    GraphArg and falls back to the real inputs by position when the counts
    match. A declared dim whose fake size is a plain int was specialised and is
    not symbolic in this compile, so it is not recorded.

    Two things are refused rather than guessed. Two inputs declaring the same
    symbol with different contracts: the registry holds one contract per
    symbol, and picking either would build the other input's geometry wrong.
    And a symbol shared by a declared dimension and an undeclared one (Dynamo's
    duck shaping gives two dynamic dimensions of equal size one symbol): the
    undeclared tensor's buffer was never reserved at the maximum the registry
    would build for, so declare it too or keep their sizes apart.

    ``reserved_dims_of`` is the reader of a tensor's reservation, injectable for
    a test without a device. The fake tensors are read through ``dim()`` and
    ``size(d)`` only.

    Raises:
        Unsupported: on a conflicting or a shared declaration, as above.
    """
    from torch_spyre._inductor.errors import Unsupported

    if reserved_dims_of is None:
        from torch_spyre._C import get_reserved_dims as reserved_dims_of

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    by_position = (
        list(example_inputs)
        if example_inputs is not None and len(example_inputs) == len(placeholders)
        else None
    )
    declared: dict[str, tuple[int, int, int]] = {}
    undeclared: set[str] = set()
    for i, node in enumerate(placeholders):
        grapharg = node.meta.get("grapharg")
        real = getattr(grapharg, "example", None)
        if real is None and by_position is not None:
            real = by_position[i]
        fake = node.meta.get("example_value")
        if fake is None:
            fake = getattr(grapharg, "fake_tensor", None)
        if fake is None or not hasattr(fake, "size"):
            continue
        reserved = (
            reserved_dims_of(real) if isinstance(real, torch.Tensor) else None
        ) or {}
        rank = fake.dim()
        reserved = {d % rank: spec for d, spec in reserved.items()}
        for dim in range(rank):
            size = fake.size(dim)
            if not isinstance(size, torch.SymInt):
                continue
            expr = size.node.expr
            if not getattr(expr, "is_Symbol", False):
                continue
            name = str(expr)
            spec = reserved.get(dim)
            if spec is None:
                undeclared.add(name)
                continue
            contract = (spec["min"], spec["max"], spec["granularity"])
            previous = declared.get(name)
            if previous is not None and previous != contract:
                raise Unsupported(
                    f"graph input {node.name} declares dimension {dim} ({name}) "
                    f"as (min, max, granularity) {contract}, but an earlier input "
                    f"declared the same symbol as {previous}. One symbol carries "
                    f"one contract: give both inputs the same declaration."
                )
            declared[name] = contract
    shared = sorted(declared.keys() & undeclared)
    if shared:
        raise Unsupported(
            f'symbol(s) {shared} are declared with tensor.to("spyre", '
            f"dynamic=...) on one graph input and undeclared on another that "
            f"shares the symbol (two dynamic dimensions of equal size, or a "
            f"torch._check tie). Geometry built at the declared maximum would "
            f"address past the undeclared input's buffer, which was never "
            f"reserved for it. Declare that input too, or keep its size distinct."
        )
    return declared


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
        from torch_spyre._inductor import timing_recorder
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
                    # Mark this thread as inside a Spyre compile so the
                    # invoke_subgraph HOP re-trace (which happens within this
                    # _orig call) threads the Spyre decomp table into its
                    # subgraph bodies. Save/restore to stay correct under
                    # nested compile_fx calls.
                    _prev_in_spyre = getattr(_compile_state, "in_spyre_compile", False)
                    _compile_state.in_spyre_compile = True
                    try:
                        # The outermost torch-spyre region of one compile; it
                        # runs inside Dynamo's mirrored entire_frame_compile. A
                        # frontend total is this minus the per-kernel backend calls.
                        with timing_recorder.stage("stage:compile_fx:spyre_compile"):
                            # Read before lowering: concretize_expr is called
                            # during lowering, and a declared dimension's
                            # geometry is built from its declared maximum.
                            declared = _declared_dims_of(gm, example_inputs)
                            with enable_spyre_context(
                                example_inputs, declared_dims=declared
                            ):
                                return _orig(gm, example_inputs, *args, **kwargs)
                    finally:
                        _compile_state.in_spyre_compile = _prev_in_spyre

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
    from . import distributed as _distributed_init  # noqa: F401  registers spyre::*_run/wait_work fakes
    from . import torch_phases

    enable_spyre_compile_fx_wrapper()
    # Before any compile rather than inside enable_spyre_context, which opens too
    # late to see Dynamo tracing.
    torch_phases.install()


# Name of the ``scan`` higher-order op as PyTorch's cache checks see it.
_SCAN_HOP_NAME = "torch.ops.higher_order.scan"
# Salt for the cache key of graphs that contain ``scan``. Change it to drop
# every cached graph that has a ``scan`` in it.
_SCAN_CACHE_SALT = "torch-spyre-scan-v1"


def _allow_scan_graph_caching() -> None:
    """Let the FX graph and AOTAutograd caches keep graphs that use ``scan``.

    On by default; turn off with ``TORCH_SPYRE_CACHE_SCAN=0``. ``for_each_tile`` (used by the
    SDPA K/V loop) is the ``scan`` op, and PyTorch skips the caches for any
    graph with a higher-order op it does not know is cacheable. So every
    SDPA with more than one K/V block is compiled again in every process.
    The ``scan`` body is a subgraph of the cached graph and is part of the
    cache key like the rest of the graph; the salt is added to the key too.

    What "unsafe" leaves to us: PyTorch does not check that ``scan`` itself
    is safe to cache. Tensors used inside the body are hashed by value (or
    are graph inputs), except non-inlined frozen parameters, which are hashed
    by metadata only, as for any other graph. A change to ``scan`` in PyTorch
    is not in the key: bump ``_SCAN_CACHE_SALT`` to drop old entries.
    """
    if os.environ.get("TORCH_SPYRE_CACHE_SCAN", "1") != "1":
        return
    import torch._inductor.config as inductor_config

    marked = dict(inductor_config.unsafe_marked_cacheable_functions)
    marked.setdefault(_SCAN_HOP_NAME, _SCAN_CACHE_SALT)
    inductor_config.unsafe_marked_cacheable_functions = marked


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

        _allow_scan_graph_caching()

        _autoload._ran = True

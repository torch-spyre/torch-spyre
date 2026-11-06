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

"""Stage 2: follow one torch-level call down to the ATen ops Inductor lowers.

Each unique call is traced three times with ``make_fx`` on fake tensors that
sit on the Spyre device (no device or runtime is touched):

1. ``pre_dispatch=True`` gives the ATen op the call maps to (``aten.conv2d``).
2. With Spyre's decomposition table and functionalization, as AOTAutograd
   does for a Spyre compile, gives the leaf ops that reach Inductor lowering.
   Each decomposition is wrapped to record the graph nodes it created, which
   gives the exact decomposition tree of every leaf.
3. Without decompositions gives the ops eager dispatch reaches.

The per-call trace leaves out graph-level FX passes (pattern rewrites,
``mm_to_bmm``), which see more than one op at a time; ``probe`` covers them.
"""

from __future__ import annotations

import collections
import contextlib
import functools
from typing import Any, Callable, Iterator, Optional

import torch
import torch.fx
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import dispatch_functionalize
from torch.fx.experimental.proxy_tensor import get_proxy_mode, make_fx
from torch.utils import _pytree as pytree

from .capture import CapturedCall, TensorMeta
from .model import DecompStep, LeafOp, SignatureAnalysis, Support, Verdict
from .registry import SpyreRegistry, is_structural, op_name


_DATA_CONSTRUCTORS = (torch.tensor, torch.as_tensor, torch.asarray)


class _Span:
    __slots__ = ("op", "source", "nodes")

    def __init__(self, op: Any, source: str) -> None:
        self.op = op
        self.source = source
        self.nodes: list[str] = []


class _DecompRecorder:
    """Records which graph nodes each decomposition emits.

    Wraps every entry of Spyre's decomposition table. While ``composites()``
    is active it also wraps ``OpOverload.decompose``, through which
    functionalization expands CompositeImplicitAutograd ops (``upsample_*``)
    without consulting the table.
    """

    def __init__(self, registry: SpyreRegistry) -> None:
        self.spans: list[_Span] = []
        self.table = {
            op: self._wrap(op, fn, registry.decomposition_source(op))
            for op, fn in registry.decomp_table.items()
        }

    def _record(
        self, op: Any, source: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        mode = get_proxy_mode()
        if mode is None:
            return fn(*args, **kwargs)
        graph = mode.tracer.graph
        start = len(graph.nodes)
        span = _Span(op, source)
        self.spans.append(span)
        try:
            return fn(*args, **kwargs)
        finally:
            span.nodes = [n.name for n in list(graph.nodes)[start:]]

    def _wrap(self, op: Any, fn: Callable[..., Any], source: str) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self._record(op, source, fn, *args, **kwargs)

        return wrapped

    @contextlib.contextmanager
    def composites(self) -> Iterator[None]:
        original = OpOverload.decompose
        recorder = self

        def decompose(op: OpOverload, *args: Any, **kwargs: Any) -> Any:
            return recorder._record(op, "composite", original, op, *args, **kwargs)

        OpOverload.decompose = decompose  # type: ignore[method-assign]
        try:
            yield
        finally:
            OpOverload.decompose = original  # type: ignore[method-assign]


def short_error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    first = next((line for line in text if line.strip()), "")
    return f"{type(exc).__name__}: {first[:300]}"


class Tracer:
    """Turns captured calls into :class:`SignatureAnalysis` records."""

    def __init__(self, registry: SpyreRegistry, device: str = "spyre") -> None:
        self.registry = registry
        self.device = device
        self.recorder = _DecompRecorder(registry)

    def _materialize(
        self, call: CapturedCall
    ) -> tuple[Callable[..., Any], list[torch.Tensor]]:
        flat, spec = pytree.tree_flatten((call.args, call.kwargs))
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        if call.target in _DATA_CONSTRUCTORS and not any(
            isinstance(v, TensorMeta) for v in flat
        ):
            return self._constant(call, mode)
        positions = []
        fakes = []
        with mode:
            for i, value in enumerate(flat):
                if isinstance(value, TensorMeta):
                    fakes.append(
                        torch.empty_strided(
                            value.shape,
                            value.stride,
                            dtype=value.dtype,
                            device=self.device,
                        )
                    )
                    positions.append(i)
                elif isinstance(value, str) and value.split(":")[0] in ("cpu", "cuda"):
                    # A factory's device= argument; the op runs on Spyre.
                    flat[i] = self.device

        def fn(*tensors: torch.Tensor) -> Any:
            values = list(flat)
            for pos, tensor in zip(positions, tensors):
                values[pos] = tensor
            args, kwargs = pytree.tree_unflatten(values, spec)
            if call.kind == "call_method":
                return getattr(args[0], call.target)(*args[1:], **kwargs)
            return call.target(*args, **kwargs)

        return fn, fakes

    def _constant(
        self, call: CapturedCall, mode: FakeTensorMode
    ) -> tuple[Callable[..., Any], list[torch.Tensor]]:
        """``torch.tensor(data, device=...)``, traced as AOTAutograd sees it.

        These constructors copy Python data to the device below the dispatch
        modes, so tracing them directly would allocate on the real device.
        A compile records the result as a graph constant on the device and
        applies ``lift_fresh_copy`` to it; that is what is traced here, with a
        fake constant of the same shape and dtype.
        """
        kwargs = dict(call.kwargs)
        device = self.device if kwargs.pop("device", None) is not None else "cpu"
        cpu = call.target(*call.args, **kwargs)
        with mode:
            constant = torch.empty_strided(
                cpu.shape, cpu.stride(), dtype=cpu.dtype, device=device
            )

        def fn(tensor: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.lift_fresh_copy.default(tensor)

        return fn, [constant]

    def analyze(self, call: CapturedCall, count: int = 1) -> SignatureAnalysis:
        from .capture import describe_args

        result = SignatureAnalysis(
            inputs=describe_args(call.args, call.kwargs), count=count
        )
        if call.target is None:
            result.error = f"cannot resolve op name {call.name!r}"
            return result
        try:
            fn, fakes = self._materialize(call)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            result.error = "could not build inputs: " + short_error(exc)
            return result

        top_gm = self._try(result, "pre-dispatch trace", fn, fakes, pre_dispatch=True)
        top_ops = _call_targets(top_gm) if top_gm is not None else []
        result.top_level = sorted({op_name(op) for op in top_ops})

        self.recorder.spans.clear()
        with self.recorder.composites():
            compile_gm = self._try(
                result, "trace", fn, fakes, table=self.recorder.table, functional=True
            )
        if compile_gm is not None:
            result.leaves = self._leaves(compile_gm, list(self.recorder.spans))
        elif top_ops:
            # Without the decomposed graph, judge the ops the call maps to.
            result.leaves = [
                LeafOp(op_name(op), self.registry.compile_verdict(op)) for op in top_ops
            ]
        if call.target in _DATA_CONSTRUCTORS and "device" in call.kwargs:
            # Eagerly this is a host-to-device copy, not lift_fresh_copy.
            copy = self.registry.eager_verdict(torch.ops.aten._copy_from.default)
            result.eager = Verdict(
                copy.status, f"host-to-device copy: {copy.mechanism}"
            )
        else:
            result.eager = self._eager(fn, fakes, top_ops, result)
        return result

    def _try(
        self,
        result: SignatureAnalysis,
        what: str,
        fn: Callable[..., Any],
        fakes: list[torch.Tensor],
        *,
        table: Optional[dict[Any, Any]] = None,
        functional: bool = False,
        pre_dispatch: bool = False,
    ) -> Optional[torch.fx.GraphModule]:
        target = fn
        if functional:
            # As AOTAutograd does. The inputs are returned too, so an in-place
            # update survives dead-code elimination as its functional op.
            def with_inputs(*tensors: torch.Tensor) -> Any:
                return fn(*tensors), tensors

            target = dispatch_functionalize(with_inputs)
        try:
            gm = make_fx(
                target,
                decomposition_table=table or {},
                tracing_mode="fake",
                pre_dispatch=pre_dispatch,
                _allow_non_fake_inputs=True,
            )(*fakes)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            message = f"{what} failed: {short_error(exc)}"
            result.error = message if result.error is None else result.error
            return None
        if functional:
            gm.graph.eliminate_dead_code()
        return gm

    def _leaves(self, gm: torch.fx.GraphModule, spans: list[_Span]) -> list[LeafOp]:
        owners: dict[str, list[_Span]] = collections.defaultdict(list)
        for span in spans:
            for name in span.nodes:
                owners[name].append(span)
        leaves: list[LeafOp] = []
        seen = set()
        for node, extra in _leaf_nodes(gm):
            path = [
                DecompStep(op_name(span.op), span.source) for span in owners[node.name]
            ]
            if path and path[-1].op == op_name(node.target):
                # The decomposition re-emitted its own op (cat with empty
                # inputs filtered out); not a step worth showing.
                path.pop()
            path += extra
            dtypes = _input_dtypes(node)
            key = (op_name(node.target), tuple((s.op, s.source) for s in path))
            if key in seen:
                continue
            seen.add(key)
            leaves.append(
                LeafOp(
                    op_name(node.target),
                    self.registry.compile_verdict(node.target, dtypes),
                    path,
                    sorted({str(d).replace("torch.", "") for d in dtypes}),
                )
            )
        return leaves

    def _eager(
        self,
        fn: Callable[..., Any],
        fakes: list[torch.Tensor],
        top_ops: list[Any],
        result: SignatureAnalysis,
    ) -> Verdict:
        settled = (Support.SPYRE, Support.VIEW, Support.CPU_FALLBACK)
        verdicts = {op_name(op): self.registry.eager_verdict(op) for op in top_ops}
        if not verdicts or any(v.status not in settled for v in verdicts.values()):
            # A composite at the top: judge the ops it dispatches to instead.
            probe = SignatureAnalysis(inputs="", count=0)
            post_gm = self._try(probe, "eager trace", fn, fakes)
            if post_gm is not None:
                verdicts = {
                    op_name(op): self.registry.eager_verdict(op)
                    for op in _call_targets(post_gm)
                }
        if not verdicts:
            if top_ops or result.error:
                return Verdict(Support.UNKNOWN, "no ATen ops traced")
            return Verdict(Support.VIEW, "no ATen ops (returns its input)")
        status = Support.worst(v.status for v in verdicts.values())
        worst = [
            f"{name}: {v.mechanism}"
            for name, v in verdicts.items()
            if v.status is status
        ]
        return Verdict(status, "; ".join(worst[:3]))


def _call_targets(gm: Optional[torch.fx.GraphModule]) -> list[Any]:
    if gm is None:
        return []
    return [
        node.target
        for node in gm.graph.nodes
        if node.op == "call_function" and not is_structural(node.target)
    ]


def _leaf_nodes(gm: torch.fx.GraphModule, extra: Optional[list[DecompStep]] = None):
    """Leaf call nodes, descending into higher-order op bodies."""
    extra = extra or []
    for node in gm.graph.nodes:
        if node.op != "call_function" or is_structural(node.target):
            continue
        yield node, extra
        if isinstance(node.target, HigherOrderOperator):
            step = DecompStep(op_name(node.target), "hop")
            for arg in pytree.tree_leaves((node.args, node.kwargs)):
                if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                    sub = getattr(gm, str(arg.target), None)
                    if isinstance(sub, torch.fx.GraphModule):
                        yield from _leaf_nodes(sub, extra + [step])


def _input_dtypes(node: torch.fx.Node) -> list[torch.dtype]:
    dtypes = []
    for arg in pytree.tree_leaves((node.args, node.kwargs)):
        if isinstance(arg, torch.fx.Node):
            val = arg.meta.get("val")
            if isinstance(val, torch.Tensor):
                dtypes.append(val.dtype)
    return dtypes

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

"""Stage 1: capture the torch-level ops a model calls.

The model runs on the CPU under ``torch.compile`` with a backend that records
each Dynamo graph and then runs it eagerly, so no Spyre device is needed. Each
``call_function`` / ``call_method`` node that produces a tensor becomes a
:class:`CapturedCall`, with tensor arguments replaced by :class:`TensorMeta`.

Op names follow hf-adapters' ``utils/model_ops`` collector (``torch.conv2d``,
``torch.nn.functional.silu``, ``torch.Tensor.view``, ``torch.getitem``), so a
report from a live capture lines up with one built from a model_ops YAML.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import logging
import operator
from typing import Any, Callable, Optional

import regex as re
import torch
import torch.fx
from torch.utils import _pytree as pytree


@dataclasses.dataclass(frozen=True)
class TensorMeta:
    """Shape, stride and dtype of a tensor argument; no data."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype

    def __str__(self) -> str:
        dtype = _DTYPE_SHORT.get(self.dtype, str(self.dtype).replace("torch.", ""))
        return f"{dtype}[{','.join(str(d) for d in self.shape)}]"

    def is_contiguous(self) -> bool:
        expected = 1
        for size, stride in reversed(list(zip(self.shape, self.stride))):
            if size != 1 and stride != expected:
                return False
            expected *= size
        return True


_DTYPE_SHORT = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.bool: "bool",
}


@dataclasses.dataclass
class CapturedCall:
    """One torch-level op call, with tensors replaced by their metadata."""

    name: str
    target: Any  # a callable, or a method name for call_method
    kind: str  # "call_function" or "call_method"
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    module: Optional[str] = None
    location: Optional[str] = None

    def signature_key(self) -> str:
        return f"{self.name}({describe_args(self.args, self.kwargs)})"


@dataclasses.dataclass
class CaptureResult:
    calls: list[CapturedCall]
    graphs: int
    graph_breaks: dict[str, int]
    skipped: collections.Counter


# ``_operator`` functions spelled differently from "torch." + name; see
# hf-adapters utils/model_ops/utils/torchop_yaml.py (_OPERATOR_OP_NAMES).
_OPERATOR_OP_NAMES = {
    "and_": "torch.bitwise_and",
    "or_": "torch.bitwise_or",
    "iand": "torch.Tensor.bitwise_and_",
    "ior": "torch.Tensor.bitwise_or_",
}
_TENSOR_METHODS = {"expand", "copy_", "contiguous", "view"}


def _method_name(method: str) -> str:
    if method in _TENSOR_METHODS:
        return "torch.Tensor." + method
    return "torch." + method


def torch_op_name(node: torch.fx.Node) -> str:
    """The model_ops-compatible name of a Dynamo graph node."""
    if node.op == "call_method":
        assert isinstance(node.target, str)
        return _method_name(node.target)
    target = node.target
    if getattr(target, "__module__", "") == "_operator":
        name = target.__name__  # type: ignore[union-attr]
        if name in _OPERATOR_OP_NAMES:
            return _OPERATOR_OP_NAMES[name]
        if name in torch.fx.graph.magic_methods:
            return "torch." + name
        if name in torch.fx.graph.inplace_methods:
            return "torch.setitem" if name == "setitem" else "torch.Tensor." + name[1:]
    qualified = torch.fx.node._get_qualified_name(target)  # type: ignore[arg-type]
    if qualified.startswith("torch_spyre._monkey_patch.spyre_"):
        # torch-spyre replaces Tensor.to with spyre_to; name it as the method
        # call it stands for, so x.to(...) reads the same as in model_ops.
        return _method_name(qualified.rsplit(".spyre_", 1)[1])
    if qualified.startswith("torch._C._nn."):
        qualified = "torch.nn.functional." + qualified[len("torch._C._nn.") :]
    elif qualified.startswith("torch._C._VariableFunctionsClass."):
        qualified = "torch." + qualified.rsplit(".", 1)[1]
    return qualified


def describe_args(args: Any, kwargs: Optional[dict[str, Any]] = None) -> str:
    """A compact, stable text form of call arguments."""

    def one(value: Any) -> str:
        if isinstance(value, TensorMeta):
            text = str(value)
            return text if value.is_contiguous() else text + "(strided)"
        if isinstance(value, (list, tuple)):
            inner = ", ".join(one(v) for v in value)
            return f"[{inner}]" if isinstance(value, list) else f"({inner})"
        if isinstance(value, torch.dtype):
            return str(value).replace("torch.", "")
        if isinstance(value, float):
            return f"{value:g}"
        return repr(value)

    parts = [one(a) for a in args]
    parts += [f"{k}={one(v)}" for k, v in (kwargs or {}).items()]
    return ", ".join(parts)


def _meta_of(value: Any) -> Any:
    """Replace tensors in a node's example value with TensorMeta."""

    def convert(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return TensorMeta(
                tuple(int(s) for s in x.shape),
                tuple(int(s) for s in x.stride()),
                x.dtype,
            )
        if isinstance(x, torch.SymInt):
            return int(x)
        if isinstance(x, torch.SymFloat):
            return float(x)
        if isinstance(x, torch.SymBool):
            return bool(x)
        if isinstance(x, torch.Size):
            return tuple(int(s) for s in x)
        if isinstance(x, torch.device):
            return str(x)
        return x

    return pytree.tree_map(convert, value)


def _arg_value(arg: Any, gm: torch.fx.GraphModule) -> Any:
    """The example value of a node argument, as TensorMeta or plain data."""
    if isinstance(arg, torch.fx.Node):
        if arg.op == "get_attr":
            return _meta_of(_get_attr(gm, str(arg.target)))
        return _meta_of(arg.meta.get("example_value", arg.meta.get("val")))
    if isinstance(arg, (list, tuple)):
        values = [_arg_value(a, gm) for a in arg]
        return values if isinstance(arg, list) else tuple(values)
    if isinstance(arg, dict):
        return {k: _arg_value(v, gm) for k, v in arg.items()}
    if isinstance(arg, slice):
        return slice(
            _arg_value(arg.start, gm),
            _arg_value(arg.stop, gm),
            _arg_value(arg.step, gm),
        )
    return _meta_of(arg)


def _get_attr(gm: torch.fx.GraphModule, target: str) -> Any:
    obj: Any = gm
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _has_tensor(value: Any) -> bool:
    return any(isinstance(x, torch.Tensor) for x in pytree.tree_leaves(value))


def _is_tuple_getitem(node: torch.fx.Node) -> bool:
    """``out[0]`` on a tuple result (chunk, topk, ...) is bookkeeping, not an op."""
    if node.target is not operator.getitem or not node.args:
        return False
    src = node.args[0]
    if not isinstance(src, torch.fx.Node):
        return False
    return isinstance(src.meta.get("example_value"), (tuple, list))


_MODULE_ROOT = re.compile(r"^L\['[^']+'\]\.?")


def _module_path(
    node: torch.fx.Node, module_names: Optional[dict[str, str]] = None
) -> Optional[str]:
    """The submodule a node came from, e.g. ``model.9.m``.

    ``nn_module_stack`` is keyed by ``id(module)``; with the root model's
    ``named_modules()`` that gives the full name. Without it, the path is
    relative to the ``self`` of the compiled frame.
    """
    stack = node.meta.get("nn_module_stack") or {}
    if not stack:
        return None
    key, (path, _) = list(stack.items())[-1]
    if module_names and str(key) in module_names:
        return module_names[str(key)] or None
    path = _MODULE_ROOT.sub("", str(path))
    path = path.replace(".forward.__self__", "")
    path = re.sub(r"\[(\d+)\]", r".\1", path)
    path = re.sub(r"\['([^']+)'\]", r".\1", path)
    path = re.sub(r"\._modules\.|\._modules\['([^']+)'\]", ".", path)
    return path.strip(".") or None


_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')


def _location(node: torch.fx.Node) -> Optional[str]:
    trace = node.meta.get("stack_trace")
    if not trace:
        return None
    frames = _FRAME.findall(trace)
    if not frames:
        return None
    path, line, func = frames[-1]
    for marker in ("site-packages/", "dist-packages/"):
        if marker in path:
            path = path.split(marker, 1)[1]
            break
    return f"{path}:{line} in {func}"


def calls_from_graph(
    gm: torch.fx.GraphModule, module_names: Optional[dict[str, str]] = None
) -> tuple[list[CapturedCall], collections.Counter]:
    """The tensor-producing torch-level calls of one Dynamo graph."""
    calls = []
    skipped: collections.Counter = collections.Counter()
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        name = torch_op_name(node)
        if _is_tuple_getitem(node):
            continue
        if not _has_tensor(node.meta.get("example_value", node.meta.get("val"))):
            skipped[name] += 1
            continue
        calls.append(
            CapturedCall(
                name=name,
                target=node.target,
                kind=node.op,
                args=tuple(_arg_value(a, gm) for a in node.args),
                kwargs={k: _arg_value(v, gm) for k, v in node.kwargs.items()},
                module=_module_path(node, module_names),
                location=_location(node),
            )
        )
    return calls, skipped


def capture(
    run: Callable[[Callable[..., Any]], Any],
    fn: Callable[..., Any],
    *,
    root: Optional[torch.nn.Module] = None,
    recompile_limit: int = 64,
) -> CaptureResult:
    """Run ``run(compiled_fn)`` and collect every graph Dynamo produces.

    ``run`` gets the compiled ``fn`` and drives it (one forward call, or a
    whole ``generate`` loop). Graphs run eagerly on whatever device the inputs
    are on. ``root`` names submodules in the report (``model.9.m``).
    """
    module_names = (
        {str(id(m)): name for name, m in root.named_modules()}
        if root is not None
        else None
    )
    graphs: list[torch.fx.GraphModule] = []

    def backend(gm: torch.fx.GraphModule, example_inputs: Any) -> Callable[..., Any]:
        graphs.append(gm)
        return gm.forward

    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    config = torch._dynamo.config
    breaks = _GraphBreakLog()
    with contextlib.ExitStack() as stack:
        stack.enter_context(config.patch(recompile_limit=recompile_limit))
        stack.enter_context(torch.no_grad())
        stack.enter_context(breaks)
        compiled = torch.compile(fn, backend=backend, dynamic=False)
        run(compiled)
    graph_breaks: dict[str, int] = collections.Counter(breaks.found)
    if not graph_breaks:
        for reason, count in counters["graph_break"].items():
            graph_breaks[_short_reason(str(reason))] += int(count)
    torch._dynamo.reset()

    calls: list[CapturedCall] = []
    skipped: collections.Counter = collections.Counter()
    for gm in graphs:
        graph_calls, graph_skipped = calls_from_graph(gm, module_names)
        calls.extend(graph_calls)
        skipped.update(graph_skipped)
    return CaptureResult(calls, len(graphs), dict(graph_breaks), skipped)


_GENERIC_REASONS = (
    "Encountered graph break when attempting to trace",
    "Failed to handle graph break gracefully",
)


def _short_reason(reason: str) -> str:
    """The informative part of a Dynamo graph-break reason."""
    lines = [
        line.strip()
        for line in reason.splitlines()
        if line.strip() and not line.strip().startswith(_GENERIC_REASONS)
    ]
    text = " ".join(lines) or " ".join(reason.split())
    for marker in ("Hint:", "Developer debug context:", "User code traceback:"):
        text = text.split(marker, 1)[0]
    return text.strip()[:240]


_BREAK = re.compile(
    r"in user code at (\S+):(\d+)\s*\nGraph Break Reason: (.*)", re.DOTALL
)


class _GraphBreakLog(logging.Handler):
    """Collects where Dynamo broke the graph, from its graph_breaks logs.

    The artifact loggers are enabled only for the duration and restored
    afterwards; nothing is printed.
    """

    LOGGERS = (
        "torch._dynamo.symbolic_convert.__graph_breaks",
        "torch._dynamo.convert_frame.__graph_breaks",
    )

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.found: collections.Counter = collections.Counter()
        self._saved: list[tuple[logging.Logger, int, bool]] = []

    def emit(self, record: logging.LogRecord) -> None:
        match = _BREAK.search(record.getMessage())
        if not match:
            return
        path, line, reason = match.groups()
        for marker in ("site-packages/", "dist-packages/"):
            if marker in path:
                path = path.split(marker, 1)[1]
                break
        self.found[f"{_short_reason(reason)} (at {path}:{line})"] += 1

    def __enter__(self) -> "_GraphBreakLog":
        for name in self.LOGGERS:
            logger = logging.getLogger(name)
            self._saved.append((logger, logger.level, logger.propagate))
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            logger.addHandler(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        for logger, level, propagate in self._saved:
            logger.removeHandler(self)
            logger.setLevel(level)
            logger.propagate = propagate
        self._saved.clear()

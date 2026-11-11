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

"""JSON encoding of captured call arguments, and resolving op names.

A replayable call is ``{"name", "kind", "target", "args", "kwargs"}`` where
``target`` is a qualified function name (``call_function``) or a method name
(``call_method``).
"""

from __future__ import annotations

import importlib
import operator
from typing import Any, Callable

import torch

from .capture import TensorMeta


def encode(value: Any) -> Any:
    if isinstance(value, TensorMeta):
        return {
            "__tensor__": {
                "shape": list(value.shape),
                "stride": list(value.stride),
                "dtype": str(value.dtype).replace("torch.", ""),
            }
        }
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, slice):
        return {
            "__slice__": [encode(value.start), encode(value.stop), encode(value.step)]
        }
    if value is Ellipsis:
        return {"__ellipsis__": True}
    if isinstance(value, torch.dtype):
        return {"__dtype__": str(value).replace("torch.", "")}
    if isinstance(value, torch.memory_format):
        return {"__memory_format__": str(value).replace("torch.", "")}
    if isinstance(value, torch.layout):
        return {"__layout__": str(value).replace("torch.", "")}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"__repr__": repr(value)}


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__tensor__" in value:
        spec = value["__tensor__"]
        shape = tuple(int(d) for d in spec["shape"])
        stride = spec.get("stride")
        if stride is None:
            stride = contiguous_stride(shape)
        return TensorMeta(shape, tuple(int(s) for s in stride), dtype_of(spec["dtype"]))
    if "__tuple__" in value:
        return tuple(decode(v) for v in value["__tuple__"])
    if "__slice__" in value:
        return slice(*(decode(v) for v in value["__slice__"]))
    if "__ellipsis__" in value:
        return Ellipsis
    if "__dtype__" in value:
        return dtype_of(value["__dtype__"])
    if "__memory_format__" in value:
        return getattr(torch, value["__memory_format__"])
    if "__layout__" in value:
        return getattr(torch, value["__layout__"])
    if "__repr__" in value:
        raise ValueError(f"cannot rebuild argument {value['__repr__']}")
    return {k: decode(v) for k, v in value.items()}


def dtype_of(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name).replace("torch.", ""), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown dtype {name!r}")
    return dtype


def contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    acc = 1
    for size in reversed(shape):
        stride.append(acc)
        acc *= max(size, 1)
    return tuple(reversed(stride))


def resolve_qualified(name: str) -> Any:
    """Import ``a.b.c`` as module ``a.b`` attribute ``c`` (or deeper)."""
    if name.startswith("_operator."):
        return getattr(operator, name.split(".", 1)[1])
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for part in parts[i:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise LookupError(f"cannot resolve {name!r}")


def resolve_op_name(name: str) -> tuple[str, Any]:
    """Map a model_ops-style op name to ``(kind, target)``.

    ``torch.Tensor.view`` and names that are only Tensor methods
    (``torch.float``) become ``call_method``; ``torch.getitem`` and other
    operator spellings map to the ``operator`` module.
    """
    if name.startswith("torch.Tensor."):
        return "call_method", name[len("torch.Tensor.") :]
    rest = name[len("torch.") :] if name.startswith("torch.") else name
    if name in ("torch.getitem", "torch.setitem"):
        return "call_function", getattr(operator, rest)
    try:
        target = resolve_qualified(name)
    except LookupError:
        target = None
    if callable(target) and not isinstance(target, torch.dtype):
        return "call_function", target
    if hasattr(operator, rest) and rest in torch.fx.graph.magic_methods:
        return "call_function", getattr(operator, rest)
    if hasattr(torch.Tensor, rest):
        return "call_method", rest
    raise LookupError(f"cannot resolve op {name!r}")


def call_target(kind: str, target: str) -> Callable[..., Any] | str:
    if kind == "call_method":
        return target
    return resolve_qualified(target)

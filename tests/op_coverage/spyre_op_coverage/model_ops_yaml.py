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

"""Read the op YAML that hf-adapters' ``utils/model_ops`` collector writes.

model_ops captures an LLM's torch-level ops on a CUDA host and writes one
test case per unique signature (``tests/configs/model_ops_tests/*.yaml``).
Reading that file gives the same calls a live capture would, so a model that
needs a GPU or a large download to run can still get a Spyre report.

Both layouts are read: ``test_suite_config.files[].tests[].edits.ops.include``
(current) and ``cases`` (``USE_OLDFORMAT``).
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, Optional

import regex as re
import torch

from .capture import CapturedCall, TensorMeta
from .serialize import contiguous_stride, dtype_of, resolve_op_name


def _tensor(spec: dict[str, Any], cast_float: Optional[torch.dtype]) -> TensorMeta:
    shape = tuple(int(d) for d in spec.get("shape") or ())
    stride = spec.get("stride")
    stride = (
        tuple(int(s) for s in stride)
        if stride is not None
        else contiguous_stride(shape)
    )
    dtype = dtype_of(spec.get("dtype", "float32"))
    if cast_float is not None and dtype.is_floating_point:
        dtype = cast_float
    return TensorMeta(shape, stride, dtype)


_SAFE_CALLS = {"slice": slice}


def _eval_py(text: str) -> Any:
    """Evaluate a model_ops ``py:`` literal: tuples, slices, Ellipsis, None."""

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            items = [ev(e) for e in node.elts]
            return tuple(items) if isinstance(node, ast.Tuple) else items
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _SAFE_CALLS
        ):
            return _SAFE_CALLS[node.func.id](*(ev(a) for a in node.args))
        if isinstance(node, ast.Name) and node.id in ("None", "Ellipsis"):
            return None if node.id == "None" else Ellipsis
        raise ValueError(f"unsupported literal: {ast.dump(node)}")

    return ev(ast.parse(text.strip(), mode="eval"))


_LITERAL = re.compile(r"^\s*[\(\[].*[\)\]]\s*$")


def _value(value: Any) -> Any:
    if isinstance(value, str):
        if value == "None":
            return None
        if value.startswith("torch.") and isinstance(
            getattr(torch, value[6:], None), torch.dtype
        ):
            return dtype_of(value)
        if _LITERAL.match(value):
            try:
                return _eval_py(value)
            except (ValueError, SyntaxError):
                return value
    return value


def _arg(item: Any, cast_float: Optional[torch.dtype]) -> Any:
    if not isinstance(item, dict):
        return _value(item)
    if "tensor" in item:
        return _tensor(item["tensor"], cast_float)
    if "tensor_list" in item:
        return [_tensor(t, cast_float) for t in item["tensor_list"]]
    if "py" in item:
        return _eval_py(str(item["py"]))
    if "value" in item:
        return _value(item["value"])
    return {k: _arg(v, cast_float) for k, v in item.items()}


def _entries(
    config: dict[str, Any],
) -> list[tuple[str, list[Any], dict[str, Any], str]]:
    out = []
    suite = config.get("test_suite_config")
    if suite:
        for file in suite.get("files", []):
            for test in file.get("tests", []):
                include = ((test.get("edits") or {}).get("ops") or {}).get(
                    "include"
                ) or []
                for case in include:
                    inputs = case.get("sample_inputs_func") or {}
                    out.append(
                        (
                            case["name"],
                            list(inputs.get("args") or []),
                            dict(inputs.get("kwargs") or {}),
                            str(case.get("description") or ""),
                        )
                    )
    for case in config.get("cases") or []:
        out.append(
            (
                case.get("op") or case["name"],
                list(case.get("inputs") or []),
                dict(case.get("kwmap") or {}),
                str(case.get("description") or ""),
            )
        )
    return out


def model_name(config: dict[str, Any], path: pathlib.Path) -> str:
    if config.get("model"):
        return str(config["model"])
    for file in (config.get("test_suite_config") or {}).get("files", []):
        for test in file.get("tests", []):
            for tag in test.get("tags") or []:
                if str(tag).startswith("model__"):
                    return str(tag)[len("model__") :]
    return path.stem


def _location(description: str) -> Optional[str]:
    match = re.search(r"(\S+\.py)(?:#L|:)(\d+)", description)
    if not match:
        return None
    path = match.group(1)
    for marker in ("/src/", "site-packages/"):
        if marker in path:
            path = path.split(marker, 1)[1]
    return f"{path}:{match.group(2)}"


def load(
    path: pathlib.Path, cast_float: Optional[torch.dtype] = None
) -> tuple[str, list[CapturedCall], list[str]]:
    """``(model name, calls, warnings)`` from a model_ops YAML file."""
    import yaml

    config = yaml.safe_load(path.read_text()) or {}
    calls = []
    unresolved: dict[str, int] = {}
    for name, args, kwargs, description in _entries(config):
        try:
            kind, target = resolve_op_name(name)
        except LookupError:
            unresolved[name] = unresolved.get(name, 0) + 1
            kind, target = "unresolved", None
        calls.append(
            CapturedCall(
                name=name,
                target=target,
                kind=kind,
                args=tuple(_arg(a, cast_float) for a in args),
                kwargs={k: _arg(v, cast_float) for k, v in kwargs.items()},
                location=_location(description),
            )
        )
    warnings = [
        f"model_ops op `{name}` ({count}x) does not name a torch callable"
        for name, count in sorted(unresolved.items())
    ]
    return model_name(config, path), calls, warnings

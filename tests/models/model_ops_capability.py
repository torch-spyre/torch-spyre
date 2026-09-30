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

"""What a model-ops test declares as its capability verdict (JUnit `capability.*` properties).

The op naming and shape rules are the log parser's, so capability ids carry over from it.
"""

import json
from typing import Any, Dict, List

import regex as re

_SIZE_OPS = {
    "torch.zeros",
    "torch.ones",
    "torch.empty",
    "torch.full",
    "torch.rand",
    "torch.randn",
    "torch.sym_sum",
}
# The parser only reads a shape/stride of digits, commas, spaces and minus signs.
_DIMS = re.compile(r"^\[[\d,\s\-]+\]$")
_TORCH_DTYPE = re.compile(r"^torch\.\w+$")


def normalize_op(op: str) -> str:
    """One name for aliases of one op."""
    if not op:
        return op
    op = op.replace("nn_", "nn.").replace("functional_", "functional.")
    if op.startswith("aten."):
        op = op.replace("aten.", "torch.").split(".default")[0].split(".out")[0]
    if "embedding" in op.lower():
        return "torch.nn.functional.embedding"
    if "index_copy" in op.lower() or "index.copy" in op.lower():
        return "torch.index_copy_"
    return op


def _compact(dims) -> str:
    return re.sub(r"\s+", "", str(dims))


def signature(op: str, args: list) -> dict:
    """The parser's input_shapes/strides/dtypes, arg_values and target_shape for one test.

    Mirrors what it reads from the [INPUT SHAPES] block, including what it skips: a tensor
    without a stride, a 0-d tensor, or a dtype not spelled torch.<name>.
    """
    shapes, strides, dtypes, values = [], [], [], []
    target_shape = ""

    def _tensor(t: dict):
        shape, stride = _compact(t.get("shape")), _compact(t.get("stride"))
        dtype = str(t.get("dtype"))
        if _DIMS.match(shape) and _DIMS.match(stride) and _TORCH_DTYPE.match(dtype):
            shapes.append(shape)
            strides.append(stride)
            dtypes.append(dtype)

    for arg in args:
        if "tensor" in arg:
            _tensor(arg["tensor"])
        elif "tensor_list" in arg:
            for t in arg["tensor_list"]:
                _tensor(t)
        elif "value" in arg:
            raw = repr(arg["value"]).strip().strip("'\"")
            inner = raw[1:-1].strip() if raw[:1] + raw[-1:] in ("()", "[]") else ""
            if inner and ("view" in op or "reshape" in op):
                target_shape = raw
            elif inner and op in _SIZE_OPS and set(inner) <= set("0123456789,- "):
                shapes.append(
                    "["
                    + ",".join(d.strip() for d in inner.split(",") if d.strip())
                    + "]"
                )
            else:
                values.append(raw)
        elif "py" in arg:
            values.append(str(arg["py"]).strip().strip("'\""))

    if not shapes and values:
        dims = [v for v in values if re.match(r"^[\d,\s]+$", v)]
        if dims:
            shapes = [f"[{v.replace(' ', '')}]" for v in dims]
            values = [v for v in values if v not in dims]

    return {
        "input_shapes": shapes,
        "input_strides": strides,
        "input_dtypes": dtypes,
        "arg_values": values,
        "target_shape": target_shape,
    }


def capability_properties(
    op: str, subject: str, variant: str, args: List[Dict[str, Any]]
) -> Dict[str, str]:
    """The `capability.*` properties for one test, or {} for an op the parser never counted."""
    name = normalize_op(op)
    if not name.startswith("torch."):
        return {}
    sig = signature(name, args)
    props = {
        "capability.test_type": "model_ops",
        "capability.subject": subject,
        "capability.name": name,
        "capability.sig.input_shapes": json.dumps(sig["input_shapes"]),
        "capability.sig.input_dtypes": json.dumps(sig["input_dtypes"]),
        "capability.tag": variant,
    }
    if sig["input_strides"]:
        props["capability.prop.input_strides"] = json.dumps(sig["input_strides"])
    return {k: v for k, v in props.items() if v}

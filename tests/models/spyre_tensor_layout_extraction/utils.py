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

import logging
import yaml
from typing import Any
from collections import defaultdict

log = logging.getLogger(__name__)

# ── YAML helpers ──────────────────────────────────────────────────────────


class FlowList(list):
    """Emits inline bracket sequences: [1, 2, 3] instead of block-style."""


def _flow_seq_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowList, _flow_seq_representer)


# ── Dtype conversion ──────────────────────────────────────────────────────

_DEVICE_DTYPE_TO_TORCH = {
    "DataFormats.IEEE_FP32": "torch.float32",
    "DataFormats.IEEE_FP16": "torch.float16",
    "DataFormats.SEN169_FP16": "torch.float16",
    "DataFormats.BF16": "torch.bfloat16",
    "DataFormats.IEEE_INT32": "torch.int32",
    "DataFormats.IEEE_INT64": "torch.int64",
}

_DEVICE_DTYPE_TO_SIMPLE = {
    "DataFormats.IEEE_FP32": "float32",
    "DataFormats.IEEE_FP16": "float16",
    "DataFormats.SEN169_FP16": "float16",
    "DataFormats.BF16": "bfloat16",
    "DataFormats.IEEE_INT32": "int32",
    "DataFormats.IEEE_INT64": "int64",
}


def _to_torch_dtype(s: str) -> str:
    if s in _DEVICE_DTYPE_TO_TORCH:
        return _DEVICE_DTYPE_TO_TORCH[s]
    if s.startswith("torch."):
        return s
    if s.startswith("DataFormats."):
        log.warning(
            "_to_torch_dtype: no mapping for %r — emitting raw string. "
            "Add it to _DEVICE_DTYPE_TO_TORCH to get a valid torch dtype.",
            s,
        )
        return s
    return s


def _contiguous_stride(shape):
    if not shape:
        return []
    stride = [1]
    for dim in reversed(shape[1:]):
        stride.insert(0, stride[0] * dim)
    return stride


def _init_for_dtype(dtype_str: str):
    d = dtype_str.lower()
    if "bool" in d:
        return "randint", {"high": 2}
    if "uint8" in d:
        return "randint", {"high": 255}
    if "int8" in d:
        return "randint", {"high": 127}
    if "int16" in d:
        return "randint", {"high": 32767}
    if "int" in d or "long" in d:
        return "randint", {"high": 2**31 - 1}
    return "rand", {}


# ── Tensor spec ───────────────────────────────────────────────────────────


def _tensor_spec(tensor_info: dict, include_spyre_layout: bool) -> dict:
    """Build one tensor entry (value under the 'tensor:' key).

    Compiled ops supply: size, stride, dtype, device_size, stride_map, device_dtype.
    Primitive ops supply: shape, stride, dtype.
    """
    shape = tensor_info.get("size", tensor_info.get("shape", []))
    raw_stride = tensor_info.get("stride")
    if raw_stride is None:
        stride = _contiguous_stride(shape)
    else:
        stride = raw_stride
    dtype = tensor_info.get("dtype", "torch.float32")

    init_method, init_args = _init_for_dtype(dtype)
    spec = {
        "shape": FlowList(shape),
        "stride": FlowList(stride),
        "storage_offset": 0,
        "dtype": _to_torch_dtype(dtype),
        "device": "cuda:0",
        "init": init_method,
    }
    if init_args:
        spec["init_args"] = init_args

    if include_spyre_layout:
        device_size = tensor_info.get("device_size")
        if device_size is not None:
            raw_device_dtype = tensor_info.get("device_dtype")
            if not raw_device_dtype:
                log.warning(
                    "_tensor_spec: device_size is set but device_dtype is missing "
                    "for tensor with shape %s — device_layout will be incomplete. "
                    "Check that the extraction sets device_dtype alongside device_size.",
                    list(device_size),
                )
            spec["device_layout"] = {
                "device_size": FlowList(device_size),
                "stride_map": FlowList(tensor_info.get("stride_map") or []),
                "device_dtype": _to_torch_dtype(raw_device_dtype)
                if raw_device_dtype
                else None,
            }
    return spec


def _build_tensor_info(
    pre_tensor: dict, ir_input: "dict | None", is_compiled: bool
) -> dict:
    t_info: dict = {
        "size": pre_tensor["shape"],
        "stride": pre_tensor.get("stride", []),
        "dtype": pre_tensor.get("dtype", "torch.float32"),
    }
    if is_compiled and ir_input and ir_input.get("device_size") is not None:
        t_info["device_size"] = ir_input["device_size"]
        t_info["stride_map"] = ir_input.get("stride_map", [])
        t_info["device_dtype"] = ir_input.get("device_dtype", "")
    return t_info


def _is_plain(v) -> bool:
    """True for values that can be emitted as a plain YAML scalar."""
    return v is None or isinstance(v, (bool, int, float, str))


# ── Op entry ─────────────────────────────────────────────────────────────


def _op_entry(op_data: dict, op_counter: dict) -> dict:
    """Build one element of the `ops.include` list."""
    is_compiled = not op_data.get("primitive", False)
    op_name = op_data.get("op", "unknown")
    op_counter[op_name] += 1
    idx = op_counter[op_name]

    pre_node_args = op_data.get("pre_node_args", [])
    pre_input_tensors = op_data.get("pre_input_tensors", [])
    pre_kwarg_tensors = op_data.get("pre_kwarg_tensors", {})
    ir_inputs = op_data.get("inputs", []) if is_compiled else []

    ir_by_name: dict[str, dict] = (
        {ir["name"]: ir for ir in ir_inputs if ir.get("name")} if is_compiled else {}
    )
    ir_by_pre_name: dict[str, dict] = (
        {
            ir["pre_arg_name"]: ir
            for ir in ir_inputs
            if ir.get("pre_arg_name") is not None
        }
        if is_compiled
        else {}
    )

    tensor_idx = 0
    args: list[dict[str, Any]] = []

    for i, arg_val in enumerate(pre_node_args):
        if isinstance(arg_val, list):
            if all(v is None for v in arg_val):
                tensor_specs = []
                for _ in arg_val:
                    pt = (
                        pre_input_tensors[tensor_idx]
                        if tensor_idx < len(pre_input_tensors)
                        else None
                    )
                    if pt is not None:
                        ir = (
                            ir_by_name.get(pt.get("ir_name") or "")
                            or ir_by_pre_name.get(pt.get("fx_name") or "")
                            or (
                                ir_inputs[tensor_idx]
                                if tensor_idx < len(ir_inputs)
                                else None
                            )
                        )
                        tensor_specs.append(
                            _tensor_spec(
                                _build_tensor_info(pt, ir, is_compiled),
                                include_spyre_layout=is_compiled,
                            )
                        )
                    tensor_idx += 1
                if tensor_specs:
                    args.append({"tensor_list": tensor_specs})
            else:
                if all(_is_plain(v) for v in arg_val):
                    args.append({"value": str(arg_val)})

        else:
            pre_tensor = (
                pre_input_tensors[tensor_idx]
                if tensor_idx < len(pre_input_tensors)
                else None
            )

            if pre_tensor is not None:
                # Single tensor arg.
                ir = (
                    ir_by_name.get(pre_tensor.get("ir_name") or "")
                    or ir_by_pre_name.get(pre_tensor.get("fx_name") or "")
                    or (ir_inputs[tensor_idx] if tensor_idx < len(ir_inputs) else None)
                )
                args.append(
                    {
                        "tensor": _tensor_spec(
                            _build_tensor_info(pre_tensor, ir, is_compiled),
                            include_spyre_layout=is_compiled,
                        )
                    }
                )
                tensor_idx += 1

            elif arg_val is not None:
                if isinstance(arg_val, (bool, int, float, str)):
                    args.append({"value": arg_val})
                elif isinstance(arg_val, (tuple, list)) and all(
                    _is_plain(v) for v in arg_val
                ):
                    args.append({"value": str(tuple(arg_val))})

            else:
                args.append({"value": None})

    op_kwargs = {}
    for kw_name, kw_val in pre_kwarg_tensors.items():
        if isinstance(kw_val, dict):
            continue
        if kw_val is not None:
            op_kwargs[kw_name] = kw_val

    dtype_str = next(
        (t["dtype"] for t in pre_input_tensors if t),
        op_data.get("dtype", "torch.float32"),
    ).lower()
    if "bfloat16" in dtype_str or "bf16" in dtype_str:
        dtype_tag = "bf16operation"
    elif "float16" in dtype_str or "fp16" in dtype_str or "half" in dtype_str:
        dtype_tag = "fp16operation"
    else:
        dtype_tag = "fp32operation"

    op_tag = f"{op_name}.{idx}_spyre" if is_compiled else f"{op_name}.{idx}"

    entry: dict = {"name": op_name}
    if args:
        entry["sample_inputs_func"] = {"args": args}
    entry["description"] = (
        op_data.get("description")
        or f"Operation: {op_name}, Node: {op_data.get('pre_node', 'unknown')}"
    )
    entry["tags"] = [dtype_tag, "constant", op_tag]
    if op_kwargs:
        entry["kwargs"] = op_kwargs
    return entry


# ── Global test-suite config ──────────────────────────────────────────────

_SUPPORTED_DTYPES = [
    {"name": n, "precision": {"atol": 0.005, "rtol": 0.005}}
    for n in [
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "complex32",
        "complex64",
        "complex128",
        "bool",
        "half",
    ]
]


# ── Public entry point ────────────────────────────────────────────────────


def write_yaml(all_layouts: list, output_path: str, model_name: str) -> None:
    """Serialise layout data to a YAML test-config file."""
    op_counter: dict = defaultdict(int)
    yaml_doc = {
        "test_suite_config": {
            "global": {
                "supported_dtypes": _SUPPORTED_DTYPES,
                "input_config": {"seed": 123},
            },
            "files": [
                {
                    "path": "${TORCH_DEVICE_ROOT}/tests/models/test_model_ops_v2.py",
                    "unlisted_test_mode": "skip",
                    "tests": [
                        {
                            "names": ["TestSpyreModelOps::test_model_ops_db"],
                            "mode": "xfail",
                            "tags": [f"model__{model_name}"],
                            "edits": {
                                "ops": {
                                    "include": [
                                        _op_entry(op, op_counter) for op in all_layouts
                                    ],
                                }
                            },
                        }
                    ],
                }
            ],
        }
    }
    with open(output_path, "w") as f:
        yaml.dump(yaml_doc, f, default_flow_style=False, sort_keys=False, width=120)

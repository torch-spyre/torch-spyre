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
import ast
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from torch.testing._internal.opinfo.core import (  # noqa: F401
    SampleInput,
)

from op_registry import OP_REGISTRY

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def parse_py_value(expr: str):
    """
    Safely parse a restricted Python literal expression used in YAML.
    Supports: tuples, None, Ellipsis, slice(None/ints), ints, floats, lists.
    Disallows function calls (except slice) and attribute access.
    """
    ALLOWED_NODES = {
        ast.Expression,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Name,
        ast.Call,
        ast.Load,
        ast.UnaryOp,
        ast.USub,
        ast.UAdd,
    }

    allowed_names = {
        "None": None,
        "Ellipsis": Ellipsis,
        "slice": slice,
        "inf": float("inf"),
        "-inf": float("-inf"),
        "nan": float("nan"),
    }
    node = ast.parse(expr, mode="eval")

    for n in ast.walk(node):
        if type(n) not in ALLOWED_NODES:
            raise ValueError(f"Node type {type(n).__name__} not allowed in py: {expr}")
        if isinstance(n, ast.Call):
            if not (isinstance(n.func, ast.Name) and n.func.id == "slice"):
                raise ValueError(f"Only slice(...) calls are allowed in py: {expr}")
        if isinstance(n, ast.Name) and n.id not in allowed_names:
            raise ValueError(f"Name {n.id} not allowed in py: {expr}")

    return eval(compile(node, "<py>", "eval"), {"__builtins__": {}}, allowed_names)


# ---------- tensor construction (deterministic) ----------
def make_tensor_from_conf(
    tconf: Dict[str, Any], *, dtype: torch.dtype, seed: Optional[int]
) -> torch.Tensor:
    shape = list(tconf["shape"])
    init = tconf.get("init", "rand")
    init_args = dict(tconf.get("init_args", {}))

    with torch.random.fork_rng(devices=[]):
        assert init == "rand" or init == "randint", f"Unknown init: {init}"
        if seed is not None:
            torch.manual_seed(int(seed))
        if init == "rand":
            low = int(init_args.get("low", 0))
            high = int(init_args.get("high", 1))
            if low > high:
                raise ValueError(
                    "Invalid value (high for randint): must be larger than low"
                )
        elif init == "randint":
            low = int(init_args.get("low", 0))
            high = int(init_args.get("high", -1))
            if high < 0:
                raise ValueError(
                    "Invalid value (high for randint): must be provided (via init_args) and must be positive"
                )
        else:
            raise ValueError(f"Unknown init: {init}")
        t = torch.testing.make_tensor(
            tuple(shape), dtype=dtype, device="cpu", high=high, low=low
        )

    return t


def confirm_device(x: Any, expected_device: torch.device) -> bool:
    """
    Recursively verify that all tensors in x are on the expected device.

    This handles cases where operations return nested structures (tuples/lists)
    containing multiple tensors, ensuring all outputs are on the correct device
    before comparison. Non-tensor values (e.g. scalar) are considered valid on
    any device.
    """
    if torch.is_tensor(x):
        return str(expected_device) in str(x.device)
    if isinstance(x, (tuple, list)):
        return all(confirm_device(item, expected_device) for item in x)
    return True


def to_device(x: Any, device: torch.device) -> Any:
    """
    Recursively move all tensors in x to the specified device.

    This handles cases where operation inputs or outputs are nested structures
    (tuples/lists) containing multiple tensors. All tensors are moved to the
    target device while preserving the structure. Non-tensor values (e.g. scalar)
    pass through unchanged.
    """
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(to_device(y, device) for y in x)
    return x


def _normalize_out(out: Any) -> Any:
    """
    Normalize operation output to a consistent format for comparison.

    Some torch operations return lists while others return tuples, even when
    semantically equivalent. This function converts all list outputs to tuples
    recursively, ensuring consistent comparison between reference (CPU) and
    test (device) outputs regardless of container type differences.
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        return tuple(_normalize_out(x) for x in out)
    return out


def _assert_same(
    testCase,
    ref_out: Any,
    test_out: Any,
    *,
    rtol: float,
    atol: float,
    case_name: str,
    description,
) -> None:
    ref_out = _normalize_out(ref_out)
    test_out = _normalize_out(test_out)

    if torch.is_tensor(ref_out):
        try:
            testCase.assertEqual(test_out, ref_out, atol=atol, rtol=rtol)
        except AssertionError as e:
            raise AssertionError(
                f"{case_name} FAILED since output is not close to an expected result\n"
                f"{e}\n"
                f"shape={tuple(ref_out.shape)} dtype={ref_out.dtype}\n"
                f"location in a model: {description}\n"
            ) from e
        return

    if isinstance(ref_out, tuple):
        assert isinstance(test_out, tuple) and len(test_out) == len(ref_out)
        for r, d in zip(ref_out, test_out):
            _assert_same(
                testCase,
                r,
                d,
                rtol=rtol,
                atol=atol,
                case_name=case_name,
                description=description,
            )
        return

    assert test_out == ref_out


# ---------- optional torch.compile path ----------
class _OpModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _maybe_compile_call(
    fn, sample, device: torch.device, compile_backend: Optional[str]
):
    if compile_backend is None or device.type == "cpu":
        return fn(sample.input, *sample.args, **sample.kwargs)
    mod = _OpModule(fn).to(device)
    torch._dynamo.reset_code_caches()  # kernel caching workaround
    compiled = torch.compile(mod, backend=compile_backend)
    return compiled(sample.input, *sample.args, **sample.kwargs)


def parse_dtype(spec) -> torch.dtype:
    # already a torch dtype?
    if isinstance(spec, torch.dtype):
        return spec

    if not isinstance(spec, str):
        raise TypeError(f"dtype must be str or torch.dtype, got {type(spec)}")

    s = spec.strip()

    # allow "torch.float16" (or "Torch.Float16" if you choose lower())
    if s.startswith("torch."):
        attr = s.split(".", 1)[1]
        dt = getattr(torch, attr, None)
        if isinstance(dt, torch.dtype):
            return dt
        raise ValueError(f"Unknown torch dtype: {spec}")

    # allow your aliases (optionally case-insensitive)
    key = s.lower()
    if key in DTYPE_MAP:
        return DTYPE_MAP[key]

    # optionally: allow bare torch attribute names beyond your whitelist
    # (e.g., "float64") if you want:
    dt = getattr(torch, s, None) or getattr(torch, key, None)
    if isinstance(dt, torch.dtype):
        return dt

    raise ValueError(
        f"Unsupported dtype: {spec!r}. Supported: {sorted(DTYPE_MAP)} and torch.<dtype>"
    )


def make_SampleInput(
    case: Dict[str, Any], seed, dtype: torch.dtype, test_device: torch.device
) -> SampleInput:
    dtype_str = str(dtype)
    cpu_args = []
    for i, inp in enumerate(case.get("inputs", [])):
        # derive per-input seed so tensors differ deterministically
        inp_seed = None if seed is None else int(seed) + i * 1000

        if "tensor" in inp:
            tensor_conf = inp["tensor"]
            tensor_dtype = parse_dtype(tensor_conf.get("dtype", dtype_str))
            cpu_args.append(
                make_tensor_from_conf(tensor_conf, dtype=tensor_dtype, seed=inp_seed)
            )
        elif "tensor_list" in inp:
            lst = [
                make_tensor_from_conf(
                    t,
                    dtype=parse_dtype(t.get("dtype", dtype_str)),
                    seed=(None if seed is None else int(seed) + i * 1000 + j),
                )
                for j, t in enumerate(inp["tensor_list"])
            ]
            cpu_args.append(lst)
        elif "value" in inp:
            val = inp["value"]
            if isinstance(val, str):
                try:
                    val = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    pass
                # if test target is tensor.to("cuda:0"), replace "cuda:0" with test_device
                op_name = case.get("op")
                if test_device is not None and op_name == "torch.to":
                    if "cuda" in val:
                        val = test_device
            cpu_args.append(val)  # python scalar or list, etc.
        elif "py" in inp:
            cpu_args.append(parse_py_value(inp["py"]))
        else:
            raise ValueError(f"Unknown input entry: {inp}")

    attrs: dict[Any, Any] = dict(case.get("attrs", {}))
    for key, value in case.get("kwmap", {}).items():
        if key == "dtype":
            value = parse_dtype(value)
        else:
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
            # if test target has (device="cuda:0"), replace "cuda:0" with test_device
            if test_device is not None and key == "device":
                if "cuda" in value:
                    value = test_device
        attrs[key] = value

    args = tuple(cpu_args[1:]) if len(cpu_args) > 1 else None
    sample_input = SampleInput(cpu_args[0], args=args, kwargs=attrs)

    return sample_input


# ---------- main entry ----------
def run_test(
    op,
    testCase,
    cpu_sample: SampleInput,
    test_sample: SampleInput,
    device: torch.device,
    compile_backend: str,
    rtol: float,
    atol: float,
    description: str = "",
) -> None:
    op_name = op.aten_name
    adapter = OP_REGISTRY[op_name]

    case_name = testCase._testMethodName

    if adapter.pre:
        cpu_sample = adapter.pre(cpu_sample)
        test_sample = adapter.pre(test_sample)

    # Run
    with torch.no_grad():
        ref_out = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)
        test_out = _maybe_compile_call(
            op,
            test_sample,
            device,
            compile_backend,
        )

        if adapter.is_inplace:
            # compare mutated arg0
            ref_out = cpu_sample.input
            test_out = test_sample.input

    ref_out_cpu = to_device(ref_out, torch.device("cpu"))

    # Check if the operation explicitly specifies CPU device
    is_cpu_operation = False
    if op_name == "torch.to":
        # Check if device argument for torch.to is "cpu"
        if str(test_sample.kwargs.get("device", "")) == "cpu" or (
            test_sample.args
            and len(test_sample.args) > 0
            and str(test_sample.args[0]) == "cpu"
        ):
            is_cpu_operation = True
    else:
        # Check if device kwarg is "cpu"
        if str(test_sample.kwargs.get("device", "")) == "cpu":
            is_cpu_operation = True

    # Check if the output tensor is on expected device
    if is_cpu_operation:
        assert confirm_device(test_out, torch.device("cpu")), (
            "result must be on cpu for explicit cpu operations"
        )
    else:
        assert confirm_device(test_out, device), "this result must be on spyre"

    test_out_cpu = to_device(test_out, torch.device("cpu"))
    _assert_same(
        testCase,
        ref_out_cpu,
        test_out_cpu,
        rtol=rtol,
        atol=atol,
        case_name=case_name,
        description=description,
    )

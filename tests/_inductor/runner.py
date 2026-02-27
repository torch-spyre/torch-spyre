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
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest
import regex as re
import torch
import torch.nn as nn

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


@dataclass(frozen=True)
class RunConfig:
    test_device: torch.device
    ref_device: torch.device = torch.device("cpu")
    compile_backend: Optional[str] = None  # if set, run Spyre through torch.compile


# ---------- safe skip/xfail condition evaluation ----------
_ALLOWED_AST = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _safe_eval_bool(expr: str, ctx: dict) -> bool:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise ValueError(
                f"Disallowed syntax in expr: {expr} (node={type(node).__name__})"
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed")
    return bool(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, ctx))


def _maybe_skip_or_xfail(
    case: Dict[str, Any], defaults: Dict[str, Any], ctx: dict
) -> None:
    skip_if = case.get("skip_if", defaults.get("skip_if", None))
    if skip_if:
        if isinstance(skip_if, str):
            if _safe_eval_bool(skip_if, ctx):
                pytest.skip("skip_if matched")
        elif isinstance(skip_if, list):
            for item in skip_if:
                expr = item["expr"]
                reason = item.get("reason", "skip_if matched")
                if _safe_eval_bool(expr, ctx):
                    pytest.skip(reason)

    xfail_reason = case.get("xfail_reason", defaults.get("xfail_reason", None))
    if xfail_reason:
        xfail_if = case.get("xfail_if", defaults.get("xfail_if", None))
        if xfail_if is None or _safe_eval_bool(str(xfail_if), ctx):
            pytest.xfail(str(xfail_reason))


def parse_py_value(expr: str):
    """
    Safely parse a restricted Python literal expression used in YAML.
    Supports: tuples, None, Ellipsis, slice(None/ints), ints, floats, lists.
    Disallows function calls and attribute access.
    """
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
        if isinstance(n, ast.Call):
            # only allow calling slice(...)
            if not (isinstance(n.func, ast.Name) and n.func.id == "slice"):
                raise ValueError(f"Only slice(...) calls are allowed in py: {expr}")
        if isinstance(n, ast.Attribute):
            raise ValueError(f"Attributes not allowed in py: {expr}")
        if isinstance(n, ast.Name) and n.id not in allowed_names:
            raise ValueError(f"Name {n.id} not allowed in py: {expr}")

    return eval(compile(node, "<py>", "eval"), {"__builtins__": {}}, allowed_names)


# ---------- tensor construction (deterministic) ----------
def _fork_seed(seed: Optional[int]):
    if seed is None:
        return torch.random.fork_rng(devices=[])
    # fork_rng keeps global RNG clean
    ctx = torch.random.fork_rng(devices=[])
    ctx.__enter__()
    torch.manual_seed(int(seed))
    return ctx


def _numel(shape) -> int:
    if len(shape) == 0:
        return 1
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _make_base(shape, dtype, seed):
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(int(seed))
        return torch.randn(tuple(shape), device="cpu", dtype=dtype)


def make_tensor_from_conf(
    tconf: Dict[str, Any], *, dtype: torch.dtype, seed: Optional[int]
) -> torch.Tensor:
    shape = list(tconf["shape"])
    init = tconf.get("init", "rand")
    init_args = dict(tconf.get("init_args", {}))
    preset = tconf.get("preset", None)
    preset_args = dict(tconf.get("preset_args", {}))
    base_shape = tconf.get("base_shape", None)

    # presets create noncontig/aliasing views
    if preset is None:
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(int(seed))
            if init == "rand" and dtype is torch.bool:
                threshold = 0.5  # 50% chance of True
                t = torch.rand(tuple(shape), device="cpu") < threshold
            elif init == "rand":
                t = torch.rand(tuple(shape), dtype=dtype, device="cpu")
            elif init == "randn":
                t = torch.randn(tuple(shape), dtype=dtype, device="cpu")
            elif init == "zeros":
                t = torch.zeros(tuple(shape), dtype=dtype, device="cpu")
            elif init == "ones":
                t = torch.ones(tuple(shape), dtype=dtype, device="cpu")
            elif init == "arange":
                t = torch.arange(_numel(shape), dtype=dtype, device="cpu").reshape(
                    shape
                )
            elif init == "randint":
                low = int(init_args.get("low", 0))
                high = int(init_args.get("high", -1))
                if high < 0:
                    raise ValueError(
                        "Invalid value (high for randint): must be provided (via init_args) and must be positive"
                    )
                t = torch.randint(size=tuple(shape), low=low, high=high, device="cpu")
            elif init == "uniform":
                low = float(init_args.get("low", 0.0))
                high = float(init_args.get("high", 1.0))
                t = torch.empty(tuple(shape), dtype=dtype, device="cpu").uniform_(
                    low, high
                )
            elif init == "data":
                data = tconf["data"]
                t = torch.tensor(data, dtype=dtype, device="cpu").reshape(shape)
            else:
                raise ValueError(f"Unknown init: {init}")
    else:
        # build a base then view it
        if preset == "noncontig_slice":
            dim = int(preset_args.get("dim", 1 if len(shape) > 1 else 0))
            step = int(preset_args.get("step", 2))
            if base_shape is None:
                base_shape = shape[:]
                base_shape[dim] = shape[dim] * step
            base = _make_base(base_shape, dtype, seed)
            slc = [slice(None)] * base.ndim
            slc[dim] = slice(0, base.shape[dim], step)
            t = base[tuple(slc)]
        elif preset == "transpose_view":
            dim0 = int(preset_args.get("dim0", 0))
            dim1 = int(preset_args.get("dim1", 1))
            if base_shape is None:
                base_shape = shape[:]
                base_shape[dim0], base_shape[dim1] = shape[dim1], shape[dim0]
            base = _make_base(base_shape, dtype, seed)
            t = base.transpose(dim0, dim1)
        elif preset == "expand_view0":
            if base_shape is None:
                raise ValueError("expand_view0 requires base_shape")
            base = _make_base(base_shape, dtype, seed)
            t = base.expand(*shape)
        else:
            raise ValueError(f"Unknown preset: {preset}")

        if list(t.shape) != shape:
            raise ValueError(
                f"Preset {preset} produced shape {list(t.shape)} != expected {shape}"
            )

    if tconf.get("contiguous", True):
        t = t.contiguous()
    return t


def confirm_device(x: Any, expected_device: str) -> bool:
    if torch.is_tensor(x):
        return expected_device in str(x.device)
    if isinstance(x, (tuple, list)):
        return all(confirm_device(item, expected_device) for item in x)
    return True


def to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(to_device(y, device) for y in x)
    return x


def _normalize_out(out: Any) -> Any:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        return tuple(_normalize_out(x) for x in out)
    return out


def _assert_same(
    ref_out: Any, test_out: Any, *, rtol: float, atol: float, case_name: str
) -> None:
    ref_out = _normalize_out(ref_out)
    test_out = _normalize_out(test_out)

    if torch.is_tensor(ref_out):
        try:
            torch.testing.assert_close(test_out, ref_out, rtol=rtol, atol=atol)
        except AssertionError as e:
            raise AssertionError(
                f"{case_name} FAILED since output is not close to an expected result\n"
                f"{e}\n"
                f"shape={tuple(ref_out.shape)} dtype={ref_out.dtype}\n"
            ) from e
        return

    if isinstance(ref_out, tuple):
        assert isinstance(test_out, tuple) and len(test_out) == len(ref_out)
        for r, d in zip(ref_out, test_out):
            _assert_same(r, d, rtol=rtol, atol=atol, case_name=case_name)
        return

    assert test_out == ref_out


# ---------- expects_error ----------
_ERR_NAME_TO_TYPE = {
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "AssertionError": AssertionError,
}


def _exc_type(name: Optional[str]):
    if not name:
        return Exception
    return _ERR_NAME_TO_TYPE.get(str(name), Exception)


def _run_and_capture(fn, args, attrs):
    try:
        out = fn(*args, **attrs)
        return False, None, out
    except BaseException as e:
        return True, e, None


def _assert_raised(label: str, raised: bool, exc: BaseException, spec: dict):
    assert raised, f"{label} expected to raise but did not"
    want_t = _exc_type(spec.get("type", "RuntimeError"))
    assert isinstance(exc, want_t), f"{label} raised {type(exc)} expected {want_t}"
    m = spec.get("match", None)
    if m:
        assert re.search(m, str(exc)), f"{label} message did not match /{m}/. msg={exc}"


# ---------- optional torch.compile path ----------
class _OpModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _maybe_compile_call(
    fn, args, attrs, device: torch.device, compile_backend: Optional[str]
):
    if compile_backend is None or device.type == "cpu":
        return fn(*args, **attrs)
    mod = _OpModule(fn).to(device)
    torch._dynamo.reset_code_caches()  # kernel caching workaround
    compiled = torch.compile(mod, backend=compile_backend)
    return compiled(*args, **attrs)


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


# ---------- main entry ----------
def run_case(case: Dict[str, Any], defaults: Dict[str, Any], cfg: RunConfig) -> None:
    op_name = case["op"]
    adapter = OP_REGISTRY[op_name]

    case_name = case.get("name", op_name)

    dtype_str = case.get("dtype", defaults.get("dtype", "fp16"))
    dtype = parse_dtype(dtype_str)
    dtype_str = str(dtype)
    seed = case.get("seed", defaults.get("seed", None))

    rtol = float(case.get("rtol", defaults.get("rtol", 5e-3)))
    atol = float(case.get("atol", defaults.get("atol", 5e-3)))
    attrs = dict(case.get("attrs", {}))

    ctx = {
        "cfg": cfg,
        "op_name": op_name,
        "dtype_str": dtype_str,
        "env": dict(os.environ),
        "torch": torch,
    }
    _maybe_skip_or_xfail(case, defaults, ctx)

    # Build CPU args ONCE, then copy to Spyre (identical values)
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
            cpu_args.append(val)  # python scalar or list, etc.
        elif "py" in inp:
            cpu_args.append(parse_py_value(inp["py"]))
        else:
            raise ValueError(f"Unknown input entry: {inp}")

    for key, value in case.get("kwmap", {}).items():
        if key == "dtype":
            value = parse_dtype(value)
        else:
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
        attrs[key] = value

    test_args = []
    for a in cpu_args:
        # also move tensors inside lists (cat)
        if isinstance(a, list):
            test_args.append([to_device(x, cfg.test_device) for x in a])
        else:
            test_args.append(to_device(a, cfg.test_device))

    if adapter.pre:
        cpu_args, attrs = adapter.pre(cpu_args, attrs)
        test_args, _ = adapter.pre(test_args, attrs)

    # expects_error: both must raise
    expects_error = case.get("expects_error", None)
    if expects_error:
        if "ref" in expects_error or "spyre" in expects_error:
            ref_spec = expects_error.get("ref", {})
            test_spec = expects_error.get("spyre", {})
        else:
            ref_spec = test_spec = expects_error

        ref_raised, ref_exc, _ = _run_and_capture(adapter.fn, cpu_args, attrs)
        test_raised, test_exc, _ = _run_and_capture(adapter.fn, test_args, attrs)
        _assert_raised("CPU", ref_raised, ref_exc, ref_spec)
        _assert_raised("Spyre", test_raised, test_exc, test_spec)
        return

    # Run
    with torch.no_grad():
        ref_out = adapter.fn(*cpu_args, **attrs)
        test_out = _maybe_compile_call(
            adapter.fn, test_args, attrs, cfg.test_device, cfg.compile_backend
        )

        if adapter.is_inplace:
            # compare mutated arg0
            ref_out = cpu_args[0]
            test_out = test_args[0]

    ref_out_cpu = to_device(ref_out, torch.device("cpu"))
    assert confirm_device(test_out, "spyre"), "this result must be on spyre"
    test_out_cpu = to_device(test_out, torch.device("cpu"))
    _assert_same(ref_out_cpu, test_out_cpu, rtol=rtol, atol=atol, case_name=case_name)

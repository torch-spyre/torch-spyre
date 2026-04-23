"""
Pydantic models for the OOT PyTorch test framework YAML config.

Used by spyre_test_parsing.py to validate and parse the YAML config.
"""

import ast
import regex as re
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import torch
from pydantic import BaseModel, field_validator, model_validator  # type: ignore

from spyre_test_constants import (
    DTYPE_STR_MAP,
    MODE_MANDATORY_SUCCESS,
    MODE_SKIP,
    MODE_XFAIL,
    MODE_XFAIL_STRICT,
    REL_PATH_TOKENS,
)
from spyre_test_matching import parse_dtype


# ---------------------------------------------------------------------------
# Valid dtype strings (used in validators)
# ---------------------------------------------------------------------------

_VALID_DTYPE_STRINGS = {
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
}
# -------------------------------------------
# Valid tensor generation strategies
# -------------------------------------------
_VALID_INIT_STRATEGIES = {
    "rand",
    "randn",
    "zeros",
    "ones",
    "randint",
    "arange",
    "eye",
    "full",
    "file",
}

_VALID_TEST_MODES = {MODE_MANDATORY_SUCCESS, MODE_XFAIL, MODE_XFAIL_STRICT, MODE_SKIP}

_VALID_UNLISTED_MODES = {"skip", "xfail", "xfail_strict", "mandatory_success"}

# ---------------------------------------------------------------------------
# Python literal evaluator — shared by InputArgPy and InputsEdits
# ---------------------------------------------------------------------------

_PY_ALLOWED_NODES = {
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
_PY_ALLOWED_NAMES: Dict[str, Any] = {
    "None": None,
    "Ellipsis": Ellipsis,
    "slice": slice,
    "inf": float("inf"),
    "nan": float("nan"),
}

_TOKEN_RE = re.compile(r"\$\{([^}]+)\}")


def _eval_py_literal(expr: str) -> Any:
    """Safely evaluate a restricted Python literal (slice, tuple, Ellipsis, etc.)."""
    node = ast.parse(expr, mode="eval")
    for n in ast.walk(node):
        if type(n) not in _PY_ALLOWED_NODES:
            raise ValueError(
                f"Node type {type(n).__name__!r} not allowed in py: {expr!r}"
            )
        if isinstance(n, ast.Call):
            if not (isinstance(n.func, ast.Name) and n.func.id == "slice"):
                raise ValueError(f"Only slice(...) calls are allowed in py: {expr!r}")
        if isinstance(n, ast.Name) and n.id not in _PY_ALLOWED_NAMES:
            raise ValueError(f"Name {n.id!r} not allowed in py: {expr!r}")
    return eval(compile(node, "<py>", "eval"), {"__builtins__": {}}, _PY_ALLOWED_NAMES)


# ---------------------------------------------------------------------------
# Dtype resolution using DTYPE_STR_MAP
# ---------------------------------------------------------------------------


def _resolve_dtype_str(spec: str) -> torch.dtype:
    """Resolve a dtype string using DTYPE_STR_MAP. Accepts 'float16' or 'torch.float16'."""
    bare = spec.removeprefix("torch.")
    if bare in DTYPE_STR_MAP:
        return DTYPE_STR_MAP[bare]
    try:
        return parse_dtype(bare)
    except ValueError:
        pass
    raise ValueError(
        f"Unsupported dtype: {spec!r}. "
        f"Supported aliases: {sorted(DTYPE_STR_MAP)} and torch.<dtype>"
    )


def _resolve_tensor_path(raw_path: str) -> str:
    """Expand ``${TOKEN}`` placeholders in a tensor init_args.path and return
    an absolute path.

    Resolution order:
    1. Replace every ``${TOKEN}`` using the env-var declared in REL_PATH_TOKENS.
    2. If the result is already absolute, return it.
    3. Otherwise resolve relative to the process working directory.

    Raises:
        ValueError:      Unknown token or its env-var is unset.
        FileNotFoundError: Resolved path does not exist on disk.
    """
    token_map: dict[str, str] = {
        tok.strip("${}") if tok.startswith("${") else tok: env_var
        for tok, env_var in REL_PATH_TOKENS
    }

    def _replace(m: re.Match) -> str:
        name = m.group(1)
        if name not in token_map:
            raise ValueError(
                f"Unknown path token '${{{name}}}' in init_args.path={raw_path!r}. "
                f"Known tokens: {sorted(token_map)}"
            )
        value = os.environ.get(token_map[name])
        if value is None:
            raise ValueError(
                f"Environment variable '{token_map[name]}' (for token '${{{name}}}') "
                f"is not set. Export it before running tests."
            )
        return value

    expanded = _TOKEN_RE.sub(_replace, raw_path)
    resolved = str(Path(expanded).resolve())

    if not Path(resolved).exists():
        raise FileNotFoundError(
            f"Tensor file not found: {resolved!r}  (from init_args.path={raw_path!r})"
        )
    return resolved


# ---------------------------
# edits.inputs models
# ---------------------------


class InputInitArgs(BaseModel):
    """Optional extra arguments for tensor initialization strategies."""

    low: int = 0  # randint: lower bound
    high: Optional[int] = None  # randint: upper bound (required)
    fill_value: Optional[float] = None  # full: fill value (required)
    path: Optional[str] = None  # file: path to .pt / .npy / .safetensors
    key: Optional[str] = None  # file: key within file (dict/.safetensors)


class InputTensorSpec(BaseModel):
    """Specification for constructing a single input tensor."""

    shape: List[int]
    dtype: str
    device: str = "spyre"
    init: str = "rand"
    init_args: InputInitArgs = InputInitArgs()
    stride: Optional[List[int]] = None
    storage_offset: int = 0

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        # Accept both short names ("float16") and torch-prefixed ("torch.float16")
        bare = v.removeprefix("torch.")
        if bare not in _VALID_DTYPE_STRINGS:
            raise ValueError(
                f"Unknown dtype {v!r}. Valid values: {sorted(_VALID_DTYPE_STRINGS)}"
            )
        return v

    @field_validator("init")
    @classmethod
    def validate_init(cls, v: str) -> str:
        if v not in _VALID_INIT_STRATEGIES:
            raise ValueError(
                f"Unknown init strategy {v!r}. "
                f"Valid values: {sorted(_VALID_INIT_STRATEGIES)}"
            )
        return v

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, v: List[int]) -> List[int]:
        for dim in v:
            if not isinstance(dim, int) or dim < 0:
                raise ValueError(
                    f"Each shape dimension must be a non-negative int, got {dim!r}"
                )
        return v

    @field_validator("storage_offset")
    @classmethod
    def validate_storage_offset(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"storage_offset must be non-negative, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "InputTensorSpec":
        if self.init == "randint" and self.init_args.high is None:
            raise ValueError("init_args.high is required when init: randint")
        if self.init == "full" and self.init_args.fill_value is None:
            raise ValueError("init_args.fill_value is required when init: full")
        if self.init == "file" and self.init_args.path is None:
            raise ValueError("init_args.path is required when init: file")
        if self.init == "arange" and len(self.shape) != 1:
            raise ValueError(f"arange requires a 1-D shape, got {self.shape}")
        if self.init == "eye" and (
            len(self.shape) != 2 or self.shape[0] != self.shape[1]
        ):
            raise ValueError(f"eye requires a square 2-D shape, got {self.shape}")
        if self.stride is not None and len(self.stride) != len(self.shape):
            raise ValueError(
                f"stride length {len(self.stride)} must match shape length {len(self.shape)}"
            )
        return self

    def resolved_dtype(self) -> torch.dtype:
        return _resolve_dtype_str(self.dtype)

    def build(self, *, seed: Optional[int]) -> torch.Tensor:
        """Build and return a CPU tensor according to this spec."""
        shape = list(self.shape)
        dtype = self.resolved_dtype()
        init = self.init
        ia = self.init_args

        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(int(seed))

            if init == "rand":
                t = torch.rand(shape, dtype=dtype)
            elif init == "randn":
                t = torch.randn(shape, dtype=dtype)
            elif init == "zeros":
                t = torch.zeros(shape, dtype=dtype)
            elif init == "ones":
                t = torch.ones(shape, dtype=dtype)
            elif init == "randint":
                t = torch.randint(ia.low, ia.high, shape, dtype=dtype)
            elif init == "arange":
                t = torch.arange(shape[0], dtype=dtype)
            elif init == "eye":
                t = torch.eye(shape[0], dtype=dtype)
            elif init == "full":
                t = torch.full(shape, ia.fill_value, dtype=dtype)
            elif init == "file":
                t = self._load_from_file()
            else:
                raise ValueError(f"Unknown init strategy: {init!r}")

        if self.stride is not None or self.storage_offset != 0:
            stride = self.stride if self.stride is not None else list(t.stride())
            offset = self.storage_offset
            needed = offset + (
                sum((s - 1) * st for s, st in zip(shape, stride)) + 1 if shape else 1
            )
            backing = torch.empty(needed, dtype=dtype)
            t = torch.as_strided(backing, shape, stride, offset)
            with torch.no_grad():
                if init == "rand":
                    t.copy_(torch.rand(shape, dtype=dtype))
                elif init == "randn":
                    t.copy_(torch.randn(shape, dtype=dtype))
                elif init == "randint":
                    t.copy_(torch.randint(ia.low, ia.high, shape, dtype=dtype))

        return t

    def _load_from_file(self) -> torch.Tensor:
        """Load a tensor from disk (.pt, .npy, .safetensors)."""
        ia = self.init_args
        assert ia.path is not None
        path = _resolve_tensor_path(ia.path)

        if path.endswith(".npy"):
            import numpy as np

            t = torch.from_numpy(np.load(path))
        elif path.endswith(".safetensors"):
            from safetensors.torch import load_file

            tensors = load_file(path)
            if ia.key is None:
                if len(tensors) != 1:
                    raise ValueError(
                        f"safetensors {path!r} contains multiple tensors; specify init_args.key"
                    )
                t = next(iter(tensors.values()))
            else:
                t = tensors[ia.key]
        else:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                if ia.key is None:
                    raise ValueError(
                        f".pt file {path!r} is a dict; specify init_args.key"
                    )
                t = obj[ia.key]
            else:
                t = obj

        if list(t.shape) != list(self.shape):
            raise ValueError(
                f"Loaded tensor shape {list(t.shape)} != spec shape {self.shape} from {path!r}"
            )
        if t.dtype != self.resolved_dtype():
            raise ValueError(
                f"Loaded tensor dtype {t.dtype} != spec dtype {self.dtype!r} from {path!r}"
            )
        return t


class InputArgTensor(BaseModel):
    """A single tensor positional argument."""

    tensor: InputTensorSpec


class InputArgTensorList(BaseModel):
    """A list of tensors as one positional argument (e.g. torch.cat)."""

    tensor_list: List[InputTensorSpec]


class InputArgValue(BaseModel):
    """A plain Python scalar / None positional argument."""

    value: Any  # number, None, bool


class InputArgPy(BaseModel):
    """A Python literal expression (slice, tuple, Ellipsis)."""

    py: str  # evaluated with ast.literal_eval at runtime

    @field_validator("py")
    @classmethod
    def validate_py(cls, v: str) -> str:
        try:
            _eval_py_literal(v)
        except Exception as e:
            raise ValueError(f"Invalid py expression {v!r}: {e}") from e
        return v


# Union type for a single element of edits.inputs.args
InputArg = Union[InputArgTensor, InputArgTensorList, InputArgValue, InputArgPy]


def _parse_input_arg(raw: Any) -> InputArg:
    """Parse one element of edits.inputs.args into the correct InputArg variant."""
    if not isinstance(raw, dict):
        raise ValueError(f"Each args element must be a dict, got {type(raw)}")
    keys = set(raw.keys())
    if "tensor" in keys:
        return InputArgTensor(tensor=InputTensorSpec(**raw["tensor"]))
    if "tensor_list" in keys:
        return InputArgTensorList(
            tensor_list=[InputTensorSpec(**t) for t in raw["tensor_list"]]
        )
    if "value" in keys:
        return InputArgValue(value=raw["value"])
    if "py" in keys:
        return InputArgPy(py=raw["py"])
    raise ValueError(
        f"Each args element must contain exactly one of: "
        f"tensor, tensor_list, value, py. Got keys: {keys}"
    )


class InputsEdits(BaseModel):
    """
    Per-test input specification (edits.inputs).

    args:  ordered list of positional arguments
    kwargs: keyword arguments passed to the op / module forward
    """

    args: List[InputArg] = []
    kwargs: Dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def parse_args(cls, values: Any) -> Any:
        if isinstance(values, dict) and "args" in values:
            raw_args = values["args"] or []
            values["args"] = [_parse_input_arg(item) for item in raw_args]
        return values

    def has_inputs(self) -> bool:
        return bool(self.args) or bool(self.kwargs)

    def build_cpu_args(
        self,
        *,
        seed: Optional[int],
        op_name: str = "",
        test_device: Optional[torch.device] = None,
    ) -> List[Any]:
        """Build all positional args on CPU. Delegates to InputTensorSpec.build()."""
        cpu_args: List[Any] = []
        for i, arg in enumerate(self.args):
            inp_seed = None if seed is None else seed + i * 1000

            if isinstance(arg, InputArgTensor):
                cpu_args.append(arg.tensor.build(seed=inp_seed))

            elif isinstance(arg, InputArgTensorList):
                lst = [
                    spec.build(seed=(None if seed is None else seed + i * 1000 + j * 7))
                    for j, spec in enumerate(arg.tensor_list)
                ]
                cpu_args.append(lst)

            elif isinstance(arg, InputArgValue):
                val = arg.value
                if (
                    test_device is not None
                    and op_name == "torch.to"
                    and isinstance(val, str)
                    and "cuda" in val
                ):
                    val = test_device
                cpu_args.append(val)

            elif isinstance(arg, InputArgPy):
                cpu_args.append(_eval_py_literal(arg.py))

            else:
                raise ValueError(f"Unknown InputArg type: {type(arg)}")

        return cpu_args

    def resolved_kwargs(
        self,
        *,
        test_device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Return kwargs with dtype strings resolved to torch.dtype objects.

        Resolution order for each string value:
        1. dtype alias ("float16" / "torch.float16") -> torch.dtype via DTYPE_STR_MAP
        2. device key with "cuda:*" value            -> test_device
        3. ast.literal_eval fallback                 -> Python literal (tuple, int, etc.)
        4. pass through as-is

        None, bool, and numeric values pass through unchanged.
        """
        import ast as _ast

        out: Dict[str, Any] = {}
        for k, v in self.kwargs.items():
            if isinstance(v, str):
                # 1. dtype resolution
                bare = v.removeprefix("torch.")
                if bare in DTYPE_STR_MAP:
                    out[k] = DTYPE_STR_MAP[bare]
                    continue
                # 2. device replacement
                if k == "device" and test_device is not None and "cuda" in v:
                    out[k] = test_device
                    continue
                # 3. ast.literal_eval for tuples, ints, etc. expressed as strings
                try:
                    out[k] = _ast.literal_eval(v)
                    continue
                except (ValueError, SyntaxError):
                    pass
            out[k] = v
        return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Precision(BaseModel):
    """Precision sub-model for tolerance overrides."""

    atol: Optional[float] = None
    rtol: Optional[float] = None


class NamedItem(BaseModel):
    """A named item in an include/exclude list."""

    name: str
    description: Optional[str] = None


class ModulesNamedItem(BaseModel):
    """A named item in an include list in a module"""

    name: str
    description: Optional[str] = None
    sample_inputs_func: InputsEdits = InputsEdits()


class OpsNamedItem(BaseModel):
    """A named item in an include list in an op"""

    name: str
    description: Optional[str] = None
    tags: List[str] = []  # optional per-op tags
    sample_inputs_func: InputsEdits = InputsEdits()

    def build_sample_input(
        self,
        *,
        seed: Optional[int],
        test_device: Optional[torch.device],
        SampleInput,
    ) -> Any:
        """Build a SampleInput from the config inputs.

        SampleInput is passed in as an argument to avoid importing
        torch.testing internals into this models file.
        """
        cpu_args = self.sample_inputs_func.build_cpu_args(
            seed=seed,
            op_name=self.name,
            test_device=test_device,
        )
        resolved_kw = self.sample_inputs_func.resolved_kwargs(test_device=test_device)
        inp = cpu_args[0] if cpu_args else None
        rest = tuple(cpu_args[1:]) if len(cpu_args) > 1 else ()
        return SampleInput(inp, args=rest, kwargs=resolved_kw)


class DtypeNamedItem(BaseModel):
    """A dtype item with optional precision override."""

    name: str
    description: Optional[str] = None
    precision: Optional[Precision] = None


class OpsEdits(BaseModel):
    """Per-test op list overrides."""

    include: List[OpsNamedItem] = []  # inject ops into @ops.op_list
    exclude: List[NamedItem] = []  # remove ops from @ops.op_list

    def included_op_names(self) -> Set[str]:
        return {item.name for item in self.include}

    def excluded_op_names(self) -> Set[str]:
        return {item.name for item in self.exclude}


class ModulesEdits(BaseModel):
    """Per-test module list overrides."""

    include: List[
        ModulesNamedItem
    ] = []  # inject modules into @modules.module_info_list
    exclude: List[NamedItem] = []  # remove modules from @modules.module_info_list

    def included_module_names(self) -> Set[str]:
        return {item.name for item in self.include}

    def excluded_module_names(self) -> Set[str]:
        return {item.name for item in self.exclude}


class DtypesEdits(BaseModel):
    """Per-test dtype overrides."""

    include: List[DtypeNamedItem] = []  # inject dtypes into @ops.allowed_dtypes
    exclude: List[NamedItem] = []  # remove dtype variants for this test

    @field_validator("include", "exclude", mode="before")
    @classmethod
    def validate_dtype_names(cls, v: list) -> list:
        for item in v or []:
            name = item.get("name") if isinstance(item, dict) else item
            if name not in _VALID_DTYPE_STRINGS:
                raise ValueError(
                    f"Unknown dtype {name!r}. "
                    f"Valid values: {sorted(_VALID_DTYPE_STRINGS)}"
                )
        return v

    def included_dtype_names(self) -> Set[str]:
        return {item.name for item in self.include}

    def excluded_dtype_names(self) -> Set[str]:
        return {item.name for item in self.exclude}

    def resolved_include(self) -> Set[torch.dtype]:
        return {parse_dtype(item.name) for item in self.include}

    def resolved_exclude(self) -> Set[torch.dtype]:
        return {parse_dtype(item.name) for item in self.exclude}

    def resolved_include_precision(self) -> Dict[torch.dtype, Precision]:
        """Return {dtype -> Precision} for included dtypes that have precision overrides."""
        return {
            parse_dtype(item.name): item.precision
            for item in self.include
            if item.precision is not None
        }


class TestEdits(BaseModel):
    ops: OpsEdits = OpsEdits()
    dtypes: DtypesEdits = DtypesEdits()
    modules: ModulesEdits = ModulesEdits()


class TestEntry(BaseModel):
    """A single test entry in the per-file tests: names, mode, tags and edits"""

    __test__ = False  # prevent pytest from collecting this as a test class

    names: List[str]
    mode: str = MODE_MANDATORY_SUCCESS
    tags: List[str] = []
    edits: TestEdits = TestEdits()

    @field_validator("names", mode="before")
    @classmethod
    def validate_name(cls, v) -> List[str]:
        if isinstance(v, str):
            v = [v]
        for item in v:
            parts = item.split("::")
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"Invalid test id {item!r}, expected 'ClassName::method_name'"
                )
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in _VALID_TEST_MODES:
            raise ValueError(
                f"Invalid mode {v!r}. Valid values: {sorted(_VALID_TEST_MODES)}"
            )
        return v

    def name_pairs(self) -> List[tuple]:
        """Return [(class_name, method_name), ...] for all entries in names."""
        return [tuple(n.split("::")) for n in self.names]

    def method_names(self) -> List[str]:
        """Return just the method_name part of each entry."""
        return [n.split("::")[1] for n in self.names]

    def class_names(self) -> List[str]:
        """Return just the class_name part of each entry."""
        return [n.split("::")[0] for n in self.names]


class FileEntry(BaseModel):
    """Per file model containing path, unlisted_test_mode and a list of tests."""

    path: str
    unlisted_test_mode: str = MODE_XFAIL
    tests: List[TestEntry] = []

    @field_validator("unlisted_test_mode")
    @classmethod
    def validate_unlisted_mode(cls, v: str) -> str:
        if v not in _VALID_UNLISTED_MODES:
            raise ValueError(
                f"Invalid unlisted_test_mode {v!r}. "
                f"Valid values: {sorted(_VALID_UNLISTED_MODES)}"
            )
        return v

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        known_tokens = {token for token, _ in REL_PATH_TOKENS}
        has_token = any(token in v for token in known_tokens)
        if not has_token and not Path(v).is_absolute():
            warnings.warn(
                f"path {v!r} contains no known token "
                f"({sorted(known_tokens)}) and is not absolute. "
                "Make sure the path is resolvable at runtime.",
                stacklevel=2,
            )
        return v

    def get_test_entry(self, class_name: str, method_name: str) -> Optional[TestEntry]:
        """Look up a TestEntry by class and method name, or None if not listed."""
        target = f"{class_name}::{method_name}"
        for entry in self.tests:
            if target in entry.names:
                return entry
        return None


class SupportedOpDtypeConfig(BaseModel):
    """Model for supported_ops.dtype: name, precision."""

    name: str
    precision: Optional[Precision] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if v not in _VALID_DTYPE_STRINGS:
            raise ValueError(f"Unknown dtype {v!r}.")
        return v

    def resolved_dtype(self) -> torch.dtype:
        return parse_dtype(self.name)


class SupportedOpConfig(BaseModel):
    """Model for storing supported ops config: name, force_xfail, list of dtypes."""

    name: str
    force_xfail: bool = False
    dtypes: List[SupportedOpDtypeConfig] = []

    def resolved_dtype_names(self) -> Optional[Set[str]]:
        if not self.dtypes:
            return None
        return {d.name for d in self.dtypes}

    def resolved_dtypes(self) -> Optional[Set[torch.dtype]]:
        if not self.dtypes:
            return None
        return {d.resolved_dtype() for d in self.dtypes}

    def get_precision(self, dtype_name: str) -> Optional[Precision]:
        """Return Precision for a specific dtype, or None if not set."""
        for d in self.dtypes:
            if d.name == dtype_name and d.precision is not None:
                return d.precision
        return None


class SupportedModuleConfig(BaseModel):
    """Model for storing supported modules config: name, force_xfail."""

    name: str
    force_xfail: bool = False
    dtypes: List[SupportedOpDtypeConfig] = []

    def get_name(self) -> str:
        return self.name

    def resolved_dtypes(self) -> Optional[Set[torch.dtype]]:
        if not self.dtypes:
            return None
        return {d.resolved_dtype() for d in self.dtypes}


class InputConfig(BaseModel):
    """Global configuration for test input generation."""

    seed: Optional[int] = None


class GlobalConfig(BaseModel):
    """Model for global configs: supported_dtypes, supported_ops."""

    supported_dtypes: List[DtypeNamedItem] = []
    supported_ops: Optional[List[SupportedOpConfig]] = None
    supported_modules: Optional[List[SupportedModuleConfig]] = None
    input_config: InputConfig = InputConfig()

    @field_validator("supported_dtypes", mode="before")
    @classmethod
    def validate_supported_dtypes(cls, v: list) -> list:
        for item in v or []:
            name = item.get("name") if isinstance(item, dict) else item
            if name not in _VALID_DTYPE_STRINGS:
                raise ValueError(f"Unknown dtype {name!r} in global.supported_dtypes.")
        return v

    @model_validator(mode="before")
    @classmethod
    def normalize_supported_ops(cls, values: object) -> object:
        """Accept both plain string list and structured dict list for supported_ops.

        Format 1 (plain): supported_ops: [add, mul, sub]
        Format 2 (structured): supported_ops: [{name: add, dtypes: [float16]}, ...]

        Plain strings are normalised to dicts so SupportedOpConfig can parse them.
        """
        if isinstance(values, dict):
            if "supported_ops" in values:
                ops = values["supported_ops"]
                if ops is not None:
                    values["supported_ops"] = [
                        {"name": op} if isinstance(op, str) else op for op in ops
                    ]
            if "supported_modules" in values:
                mods = values["supported_modules"]
                if mods is not None:
                    values["supported_modules"] = [
                        {"name": m} if isinstance(m, str) else m for m in mods
                    ]
        return values

    def resolved_supported_dtypes(self) -> Optional[Set[torch.dtype]]:
        """Return supported_dtypes as a set, or None if not specified (no filtering)."""
        if not self.supported_dtypes:
            return None
        return {parse_dtype(item.name) for item in self.supported_dtypes}

    def resolved_supported_dtypes_precision(
        self,
    ) -> Dict[torch.dtype, Precision]:
        """Return {dtype -> Precision} for dtypes that have precision overrides."""
        return {
            parse_dtype(item.name): item.precision
            for item in self.supported_dtypes
            if item.precision is not None
        }

    def resolved_supported_ops(self) -> Optional[Set[str]]:
        if self.supported_ops is None:
            return None
        return {op.name for op in self.supported_ops}

    def resolved_supported_modules(self) -> Optional[Set[str]]:
        if self.supported_modules is None:
            return None
        return {m.name for m in self.supported_modules}

    def resolved_supported_ops_config(self) -> Optional[Dict[str, SupportedOpConfig]]:
        if self.supported_ops is None:
            return None
        return {op.name: op for op in self.supported_ops}

    def resolved_supported_modules_config(
        self,
    ) -> Optional[Dict[str, SupportedModuleConfig]]:
        if self.supported_modules is None:
            return None
        return {m.name: m for m in self.supported_modules}


class TestsBlock(BaseModel):
    """Holds the inner YAML keys: files and global."""

    files: List[FileEntry]
    global_config: GlobalConfig = GlobalConfig()

    @model_validator(mode="before")
    @classmethod
    def rename_global(cls, values: object) -> object:
        # "global" is a Python keyword so rename it to "global_config"
        # before Pydantic processes the fields.
        if isinstance(values, dict) and "global" in values:
            values["global_config"] = values.pop("global")
        return values


class OOTTestConfig(BaseModel):
    test_suite_config: TestsBlock

    @property
    def files(self) -> List[FileEntry]:
        return self.test_suite_config.files

    @property
    def global_config(self) -> GlobalConfig:
        return self.test_suite_config.global_config

    @property
    def seed(self) -> Optional[int]:
        return self.test_suite_config.global_config.input_config.seed

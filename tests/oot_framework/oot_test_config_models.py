"""
# Copyright Author: Anubhav Jana (Anubhav.Jana97@ibm.com)

Pydantic models for the OOT PyTorch test framework YAML config.

Used by oot_test_parsing.py to validate and parse the YAML config.
"""

import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator  # type: ignore

from .oot_test_constants import (
    REL_PATH_TOKENS,
    DTYPE_STR_MAP,
    MODE_MANDATORY_SUCCESS,
    MODE_XFAIL,
    _VALID_DTYPE_STRINGS,
    _VALID_INIT_STRATEGIES,
    _VALID_TEST_MODES,
    _VALID_UNLISTED_MODES,
)
from .oot_test_matching import parse_dtype
from .oot_test_utilities import (
    _eval_py_literal,
    _resolve_dtype_str,
    _resolve_tensor_path,
)

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get("TORCH_SPYRE_DEBUG") else logging.INFO)


def _resolve_device_dtype(device_dtype_str: str):
    """Resolve a yaml device_dtype string (a torch dtype alias) to a DataFormats member."""
    from torch_spyre._C import DataFormats, get_device_dtype

    torch_dtype = _resolve_dtype_str(device_dtype_str)
    device_dtype = get_device_dtype(torch_dtype)
    if device_dtype == DataFormats.INVALID:
        raise ValueError(
            f"dtype {torch_dtype} (from device_dtype {device_dtype_str!r}) has "
            f"no Spyre device representation."
        )
    return device_dtype


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


class SpyreTensorLayoutSpec(BaseModel):
    """Specifies a SpyreTensorLayout to use when transferring a tensor to Spyre.

    Uses explicit device layout specification:
    - device_size: explicit device size specification (required)
    - stride_map: explicit stride map specification (required)
    - device_dtype: device data format (e.g., DataFormats.IEEE_FP32) (optional)
    """

    device_size: List[int]
    stride_map: List[int]
    device_dtype: str

    @model_validator(mode="after")
    def validate_layout_format(self) -> "SpyreTensorLayoutSpec":
        """Validate device_size and stride_map have matching lengths."""
        if len(self.device_size) != len(self.stride_map):
            raise ValueError(
                f"device_size length ({len(self.device_size)}) must match "
                f"stride_map length ({len(self.stride_map)})"
            )
        return self


class InputTensorSpec(BaseModel):
    """Specification for constructing a single input tensor."""

    model_config = ConfigDict(extra="forbid")

    shape: List[int]
    dtype: str
    device: str = "privateuse1"
    init: str = "rand"
    init_args: InputInitArgs = InputInitArgs()
    stride: Optional[List[int]] = None
    storage_offset: int = 0
    device_layout: Optional["SpyreTensorLayoutSpec"] = None

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
        if self.init == "xavier" and len(self.shape) < 2:
            raise ValueError(f"xavier requires 2-D or larger shape, got {self.shape}")
        if self.stride is not None and len(self.stride) != len(self.shape):
            raise ValueError(
                f"stride length {len(self.stride)} must match shape length {len(self.shape)}"
            )
        return self

    def resolved_dtype(self) -> torch.dtype:
        return _resolve_dtype_str(self.dtype)

    def _effective_dtype(self, dtype_override: Optional[torch.dtype]) -> torch.dtype:
        """Resolve the dtype to build this tensor with.

        The YAML-declared dtype is honored as-is for non-floating specs (e.g.
        int64 position_ids), which must not change with the dtype variant
        under test. Floating-point specs follow `dtype_override` when given,
        so the same YAML spec can be exercised at float16/float32/bfloat16
        without diverging from the module's own parameter dtype (which the
        upstream @modules dtype sweep casts separately via module.to(dtype)).
        """
        resolved = self.resolved_dtype()
        if dtype_override is not None and resolved.is_floating_point:
            return dtype_override
        return resolved

    def to_spyre(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """Transfer a CPU tensor to Spyre with explicit SpyreTensorLayout.

        Uses explicit device_size and stride_map to create the layout.
        Automatically validates the created layout matches the specification.
        """
        from torch_spyre._C import SpyreTensorLayout, get_spyre_tensor_layout

        layout_spec = self.device_layout
        assert layout_spec is not None, (
            "to_spyre() should only be called when device_layout is set"
        )

        shape = list(cpu_tensor.shape)
        stride = list(cpu_tensor.stride())
        dtype = cpu_tensor.dtype

        device_size = layout_spec.device_size
        stride_map = layout_spec.stride_map
        device_dtype = _resolve_device_dtype(layout_spec.device_dtype)

        logger.debug(
            "Transferring tensor shape=%s stride=%s dtype=%s to Spyre with "
            "device_size=%s stride_map=%s device_dtype=%s",
            shape,
            stride,
            dtype,
            device_size,
            stride_map,
            layout_spec.device_dtype,
        )

        # Build the SpyreTensorLayout from explicit device_size + stride_map
        stl = SpyreTensorLayout(
            device_size=device_size,
            stride_map=stride_map,
            device_dtype=device_dtype,
        )
        logger.debug("Layout created: %s", stl)

        # Step 1: move to device; Step 2: apply custom layout
        spyre_tensor = cpu_tensor.to("spyre", device_layout=stl)

        # Validate the applied layout
        actual_layout = get_spyre_tensor_layout(spyre_tensor)
        logger.debug(
            "Applied layout: device_size=%s stride_map=%s device_dtype=%s "
            "(spec: device_size=%s stride_map=%s device_dtype=%s)",
            list(actual_layout.device_size),
            list(actual_layout.stride_map),
            actual_layout.device_dtype,
            device_size,
            stride_map,
            device_dtype,
        )
        # The size/stride checks below don't cover dtype,
        # so a wrong device_dtype from the YAML spec would previously slip
        # through unnoticed.
        assert actual_layout.device_dtype == device_dtype, (
            f"device_dtype mismatch for tensor shape={shape}:\n"
            f"  expected: {device_dtype}\n"
            f"  actual:   {actual_layout.device_dtype}"
        )

        # H2D and D2H use the same stored layout, making the round-trip
        # self-inverting. Validate layout invariants instead.
        n_logical = math.prod(shape) if shape else 1
        n_device = math.prod(device_size) if device_size else 1
        assert n_device >= n_logical, (
            f"device_size {device_size} holds {n_device} elements < the "
            f"tensor's {n_logical} (shape={shape}); a valid device layout "
            f"only ever pads up, never loses elements."
        )

        from torch_spyre._C import get_device_dtype

        expected_dd = get_device_dtype(dtype)
        assert device_dtype == expected_dd, (
            f"device_dtype {device_dtype} is not the natural device dtype "
            f"{expected_dd} for tensor dtype {dtype}."
        )

        roundtrip = spyre_tensor.cpu()
        assert torch.equal(roundtrip, cpu_tensor), (
            f"Data mismatch after applying device_layout for tensor shape={shape}:\n"
            f"  device_size: {device_size}\n"
            f"  stride_map:  {stride_map}\n"
            f"This usually means device_size/stride_map is not a valid device "
            f"layout for this tensor."
        )

        return spyre_tensor

    def build(
        self, *, seed: Optional[int], dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Build and return a CPU tensor according to this spec.

        Uses PyTorch's upstream make_tensor utility for consistency with
        upstream test patterns. `dtype`, if given, overrides the YAML's
        declared dtype for floating-point specs only (see _effective_dtype).
        """
        try:
            from torch.testing._internal.common_utils import make_tensor
        except ImportError:
            # Fallback to direct torch functions if make_tensor not available
            return self._build_fallback(seed=seed, dtype=dtype)

        shape = list(self.shape)
        dtype = self._effective_dtype(dtype)
        init = self.init
        ia = self.init_args

        # Special cases that don't use make_tensor
        if init == "file":
            return self._load_from_file()
        elif init == "arange":
            return torch.arange(shape[0], dtype=dtype)
        elif init == "eye":
            return torch.eye(shape[0], dtype=dtype)
        elif init == "xavier":
            return torch.nn.init.xavier_uniform_(torch.empty(shape, dtype=dtype))
        elif init == "full":
            return torch.full(shape, ia.fill_value, dtype=dtype)
        elif init == "zeros":
            return torch.zeros(shape, dtype=dtype)
        elif init == "ones":
            return torch.ones(shape, dtype=dtype)

        # Use make_tensor for random tensors (rand, randn, randint)
        # make_tensor signature: make_tensor(*shape, dtype, device, low, high, requires_grad, noncontiguous, exclude_zero, memory_format)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(int(seed))

            if init == "rand":
                # rand uses uniform [0, 1), map to make_tensor with low=0, high=1
                t = make_tensor(*shape, dtype=dtype, device="cpu", low=0.0, high=1.0)
            elif init == "randn":
                # randn means a standard normal distribution (mean 0, std 1).
                t = torch.randn(*shape, dtype=dtype)
            elif init == "randint":
                # randint needs explicit low/high
                t = make_tensor(
                    *shape, dtype=dtype, device="cpu", low=ia.low, high=ia.high
                )
            else:
                raise ValueError(f"Unknown init strategy: {init!r}")

        # Handle custom stride/storage_offset
        # if self.stride is not None or self.storage_offset != 0:
        #     stride = self.stride if self.stride is not None else list(t.stride())
        #     offset = self.storage_offset
        #     needed = offset + (
        #         sum((s - 1) * st for s, st in zip(shape, stride)) + 1 if shape else 1
        #     )
        #     backing = torch.empty(needed, dtype=dtype)
        #     t = torch.as_strided(backing, shape, stride, offset)
        if self.stride is not None or self.storage_offset != 0:
            stride = self.stride if self.stride is not None else list(t.stride())
            offset = self.storage_offset
            needed = offset + (
                sum((s - 1) * st for s, st in zip(shape, stride)) + 1 if shape else 1
            )
            backing = torch.empty(needed, dtype=dtype)
            with torch.no_grad():
                if init == "rand":
                    backing.copy_(  # fill flat backing, no aliasing
                        make_tensor(
                            needed, dtype=dtype, device="cpu", low=0.0, high=1.0
                        )
                    )
                elif init == "randn":
                    # See note above: make_tensor is uniform, not normal.
                    backing.copy_(torch.randn(needed, dtype=dtype))
                elif init == "randint":
                    backing.copy_(
                        make_tensor(
                            needed, dtype=dtype, device="cpu", low=ia.low, high=ia.high
                        )
                    )
            t = torch.as_strided(backing, shape, stride, offset)  # view created after

        return t

    def _build_fallback(
        self, *, seed: Optional[int], dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Fallback tensor builder when make_tensor is not available."""
        shape = list(self.shape)
        dtype = self._effective_dtype(dtype)
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
            with torch.no_grad():
                if init == "rand":
                    backing.copy_(torch.rand(needed, dtype=dtype))
                elif init == "randn":
                    backing.copy_(torch.randn(needed, dtype=dtype))
                elif init == "randint":
                    backing.copy_(torch.randint(ia.low, ia.high, [needed], dtype=dtype))

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
            from safetensors.torch import load_file  # type: ignore

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


class InputArgConfig(BaseModel):
    """A HuggingFace-style config object positional/keyword argument.

    Resolved at runtime to a ``PretrainedConfig`` (see :func:`_build_hf_config`).
    Two mutually exclusive reconstruction strategies, chosen by which field is
    set:

    - ``model_id`` (preferred): load the full, faithful config with
      ``AutoConfig.from_pretrained(model_id)`` — carries every config field the
      model actually had. ``config_overrides`` then setattr's a few resolved
      values on top (e.g. ``_attn_implementation``, which ``from_pretrained`` may
      leave as ``None``).
    - ``config_kwargs``: rebuild by importing ``config_path`` and calling
      ``config_cls(**config_kwargs)`` — only the captured dimensions are set;
      everything else falls back to library defaults (historical behaviour).

    Exactly one of ``model_id`` / ``config_path`` must be set. ``model_id`` takes
    precedence when both are present; the ``model_id`` path needs no
    ``config_path`` at all (``AutoConfig`` resolves the class), so a ``model_id``
    spec may omit ``config_path`` entirely.
    """

    config_path: Optional[str] = None  # e.g. "transformers.models...GraniteConfig"
    config_kwargs: Dict[str, Any] = {}
    model_id: Optional[str] = None  # HF path/dir for AutoConfig.from_pretrained
    config_overrides: Dict[str, Any] = {}  # applied via setattr after from_pretrained

    @model_validator(mode="after")
    def _require_source(self) -> "InputArgConfig":
        if not self.model_id and not self.config_path:
            raise ValueError(
                "config arg needs either 'model_id' (load full config via "
                "AutoConfig.from_pretrained) or 'config_path' (rebuild from "
                "config_kwargs); neither was set."
            )
        return self


class InputArgModule(BaseModel):
    """An ``nn.Module`` positional/keyword argument, built from its class + config.

    For wrappers whose ``__init__`` takes a *live module* rather than a config.
    An out-of-tree adapter may wrap an already-constructed upstream module to
    adopt its submodules -- e.g. Spyre's ``StandardGQAAttention(attn)`` reuses an
    HF attention's q/k/v/o projections -- and such a wrapper cannot be built from
    a config spec alone: the inner module has to be constructed first, and a live
    ``nn.Module`` is not expressible in YAML. ``py`` is no escape hatch either
    (``_eval_py_literal`` permits only literals and ``slice(...)``).

    The inner module is built by importing ``module_path`` and calling it with a
    config resolved exactly like :class:`InputArgConfig` (``model_id`` preferred,
    ``config_path`` + ``config_kwargs`` as the narrower fallback), plus
    ``module_kwargs`` (e.g. ``layer_idx``). A module needing no config at all may
    set neither and pass only ``module_kwargs``.

    Weights are whatever the inner module's own ``__init__`` produces (fresh
    init): this spec carries *shape*, not values. Put the dimensions the module
    actually runs with in the config -- for a device that pads a dimension for
    alignment, that means the padded value, not the checkpoint's.
    """

    module_path: str  # e.g. "transformers.models.granite...GraniteAttention"
    config_path: Optional[str] = None
    config_kwargs: Dict[str, Any] = {}
    model_id: Optional[str] = None  # HF path/dir for AutoConfig.from_pretrained
    config_overrides: Dict[str, Any] = {}  # applied via setattr after from_pretrained
    module_kwargs: Dict[str, Any] = {}  # e.g. {"layer_idx": 0}

    def config_arg(self) -> Optional["InputArgConfig"]:
        """Return the config spec to build the inner module with, if any.

        Reuses :class:`InputArgConfig` rather than duplicating its two
        reconstruction strategies, so ``model_id`` / ``config_path`` behave
        identically here and in a bare config arg.
        """
        if not self.model_id and not self.config_path:
            return None
        return InputArgConfig(
            config_path=self.config_path,
            config_kwargs=self.config_kwargs,
            model_id=self.model_id,
            config_overrides=self.config_overrides,
        )


class InputArgCache(BaseModel):
    """A pre-populated KV cache argument (e.g. a decode-step ``past_key_values``).

    A DecoderLayer's decode path runs
    ``if past_key_values is not None: past_key_values.update(...)`` to append the
    new token's K/V to the cached past, then attends over the whole span. To
    reproduce that under a module test we must hand the layer a real ``Cache``
    already primed with ``past_len`` tokens — a bare tensor list won't do, since
    the layer calls ``.update()`` on it.

    Built at runtime (see ``InputsEdits.resolved_kwargs``): construct
    ``cache_path`` from ``config_path``/``config_kwargs`` + ``max_cache_len``,
    then prime layer ``layer_idx`` with the ``key``/``value`` tensors via
    ``update()`` so ``get_seq_length()`` reflects the past.
    """

    cache_path: str  # e.g. "transformers.cache_utils.StaticCache"
    layer_idx: int
    key: InputTensorSpec
    value: InputTensorSpec
    max_cache_len: Optional[int] = None
    config_path: Optional[str] = None
    config_kwargs: Dict[str, Any] = {}


# Union type for a single element of edits.inputs.args
InputArg = Union[
    InputArgTensor,
    InputArgTensorList,
    InputArgModule,
    InputArgConfig,
    InputArgCache,
    InputArgValue,
    InputArgPy,
]


def _parse_input_arg(raw: Any) -> InputArg:
    """Parse one element of edits.inputs.args into the correct InputArg variant.

    Handles both:
    - Fresh dict parsing (first YAML load)
    - Already-parsed InputArg objects (from YAML anchor reuse like *id001)
    """
    # Handle already-parsed InputArg objects (from YAML anchors/aliases)
    if isinstance(
        raw,
        (
            InputArgTensor,
            InputArgTensorList,
            InputArgModule,
            InputArgConfig,
            InputArgCache,
            InputArgValue,
            InputArgPy,
        ),
    ):
        return raw

    if not isinstance(raw, dict):
        raise ValueError(f"Each args element must be a dict, got {type(raw)}")
    keys = set(raw.keys())
    if "tensor" in keys:
        return InputArgTensor(tensor=InputTensorSpec(**raw["tensor"]))
    if "tensor_list" in keys:
        return InputArgTensorList(
            tensor_list=[InputTensorSpec(**t) for t in raw["tensor_list"]]
        )
    if "cache" in keys:
        c = raw["cache"]
        return InputArgCache(
            cache_path=c["cache_path"],
            layer_idx=c["layer_idx"],
            key=InputTensorSpec(**c["key"]),
            value=InputTensorSpec(**c["value"]),
            max_cache_len=c.get("max_cache_len"),
            config_path=c.get("config_path"),
            config_kwargs=c.get("config_kwargs", {}) or {},
        )
    # Checked BEFORE the config keys: a module spec carries "module_path" AND
    # (usually) "model_id"/"config_path", so the config branch would otherwise
    # swallow it and build the bare config instead of the module.
    if "module_path" in keys:
        return InputArgModule(
            module_path=raw["module_path"],
            config_path=raw.get("config_path"),
            config_kwargs=raw.get("config_kwargs", {}) or {},
            model_id=raw.get("model_id"),
            config_overrides=raw.get("config_overrides", {}) or {},
            module_kwargs=raw.get("module_kwargs", {}) or {},
        )
    # A config arg is identified by either key: "model_id" (load full config via
    # AutoConfig.from_pretrained — no config_path required) or "config_path"
    # (rebuild from config_kwargs).
    if "config_path" in keys or "model_id" in keys:
        return InputArgConfig(
            config_path=raw.get("config_path"),
            config_kwargs=raw.get("config_kwargs", {}) or {},
            model_id=raw.get("model_id"),
            config_overrides=raw.get("config_overrides", {}) or {},
        )
    if "value" in keys:
        return InputArgValue(value=raw["value"])
    if "py" in keys:
        return InputArgPy(py=raw["py"])
    raise ValueError(
        f"Each args element must contain exactly one of: "
        f"tensor, tensor_list, module_path, config_path, model_id, value, py. "
        f"Got keys: {keys}"
    )


def _build_hf_config(arg: "InputArgConfig") -> Any:
    """Resolve an :class:`InputArgConfig` to a ``PretrainedConfig`` instance.

    Shared by both the positional (``build_cpu_args``) and keyword
    (``resolved_kwargs``) resolution paths so the two strategies stay in one
    place.

    - ``model_id`` set: load the full config via
      ``AutoConfig.from_pretrained(model_id)``, then ``setattr`` each
      ``config_overrides`` entry on top (the resolved ``_attn_implementation``
      etc.). This yields every field the real model had, not just the handful of
      captured dimensions.
    - otherwise: import ``config_path`` and call ``config_cls(**config_kwargs)``.
    """
    import importlib

    if arg.model_id:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(arg.model_id)
        for key, value in arg.config_overrides.items():
            setattr(config, key, value)
        return config

    assert arg.config_path is not None
    module_path, _, cls_name = arg.config_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Invalid config_path {arg.config_path!r}: expected "
            f"'package.module.ClassName'"
        )
    config_cls = getattr(importlib.import_module(module_path), cls_name)
    return config_cls(**arg.config_kwargs)


def _build_inner_module(arg: "InputArgModule") -> Any:
    """Construct the inner ``nn.Module`` described by an :class:`InputArgModule`.

    Shared by the positional (``build_cpu_args``) and keyword
    (``resolved_kwargs``) resolution paths, mirroring
    :func:`_build_hf_config`.

    Built on CPU like every other arg; the caller relocates the assembled wrapper
    to the test device afterwards. The module is left in whatever mode its
    ``__init__`` chose -- the test harness sets train/eval on the outer wrapper.
    """
    import importlib

    mod_path, _, cls_name = arg.module_path.rpartition(".")
    if not mod_path:
        raise ValueError(
            f"Invalid module_path {arg.module_path!r}: expected "
            f"'package.module.ClassName'"
        )
    inner_cls = getattr(importlib.import_module(mod_path), cls_name)

    config_arg = arg.config_arg()
    ctor_args = [] if config_arg is None else [_build_hf_config(config_arg)]
    return inner_cls(*ctor_args, **arg.module_kwargs)


def _dtypes_from_input_arg(arg: "InputArg") -> Set[torch.dtype]:
    """Return the dtype(s) baked into a single positional arg, if any."""
    if isinstance(arg, InputArgTensor):
        return {arg.tensor.resolved_dtype()}
    if isinstance(arg, InputArgTensorList):
        return {spec.resolved_dtype() for spec in arg.tensor_list}
    return set()


def _dtypes_from_kwarg_value(v: Any) -> Set[torch.dtype]:
    """Return the dtype(s) baked into a raw (unparsed) kwarg value, if any.

    Kwarg values are stored as raw dicts until ``resolved_kwargs()`` builds
    them, so a tensor/tensor_list spec is recognized the same way
    ``resolved_kwargs()`` recognizes it: by its dict keys.
    """
    if isinstance(v, dict):
        if "tensor" in v:
            return {InputTensorSpec(**v["tensor"]).resolved_dtype()}
        if "tensor_list" in v:
            return {InputTensorSpec(**t).resolved_dtype() for t in v["tensor_list"]}
    return set()


def _dtypes_from_inputs_edits(edits: Optional["InputsEdits"]) -> Set[torch.dtype]:
    """Collect every dtype baked into an InputsEdits' args/kwargs tensor specs."""
    if edits is None:
        return set()
    dtypes: Set[torch.dtype] = set()
    for arg in edits.args:
        dtypes |= _dtypes_from_input_arg(arg)
    for v in edits.kwargs.values():
        dtypes |= _dtypes_from_kwarg_value(v)
    return dtypes


def _move_to_test_device(obj: Any, test_device: Optional[torch.device]) -> Any:
    """Move built tensors (or lists of tensors) to the target test device.

    Tensor specs are always built on CPU for reproducible seeded random data
    (see ``InputTensorSpec.build``). The module under test, however, is moved to
    ``test_device`` by the upstream ``test_forward`` harness via ``m.to(device)``,
    so its parameters/buffers live on the device. Forward inputs must therefore
    be placed on the same device or ``F.linear`` (and Spyre decompositions) raise
    a device-mismatch error. Upstream torch builds sample inputs directly on the
    device; we build on CPU then relocate here.

    ``test_device`` is None only for CPU-target runs, where the tensors already
    live on the correct device and no move is needed.
    """
    if test_device is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(test_device)
    if isinstance(obj, list):
        return [_move_to_test_device(item, test_device) for item in obj]
    return obj


def _build_cache(
    arg: "InputArgCache",
    *,
    seed: Optional[int],
    test_device: Optional[torch.device] = None,
) -> Any:
    """Reconstruct a KV cache primed with past tokens from an ``InputArgCache``.

    Builds the concrete cache class named by ``cache_path`` (e.g. StaticCache)
    from ``config_path``/``config_kwargs`` + ``max_cache_len``, then primes layer
    ``layer_idx`` by a single ``update()`` of the captured past K/V. After this
    the cache reports ``get_seq_length() == past_len``, so a DecoderLayer's
    decode forward runs the real ``past_key_values.update(...)`` branch and
    attends over past + new token — the behaviour a bare tensor list can't
    reproduce (the layer calls ``.update()`` on the cache object).

    The captured ``key``/``value`` specs hold the populated slice
    ``[B, num_kv_heads, past_len, head_dim]`` (not the full fixed allocation).

    ``test_device`` places the cache's backing buffers on the target device.
    A lazily-initialized layer cache (e.g. ``StaticLayer``) pins its ``device``
    from the FIRST ``update()`` call's ``key_states``, so the priming K/V below
    are moved to ``test_device`` BEFORE ``update()``. Without this the whole
    cache stays on CPU, and the module-under-test's own (on-device) decode
    ``update(key_states, ...)`` mixes an on-device tensor with a CPU cache
    buffer -> a device-mismatch error inside the compiled region. ``None``
    (CPU-target runs) leaves the priming tensors on CPU.
    """
    import importlib

    # Resolve the cache class (e.g. transformers.cache_utils.StaticCache).
    cache_module, _, cache_cls_name = arg.cache_path.rpartition(".")
    if not cache_module:
        raise ValueError(
            f"Invalid cache_path {arg.cache_path!r}: expected "
            f"'package.module.ClassName'"
        )
    cache_cls = getattr(importlib.import_module(cache_module), cache_cls_name)

    # Build the model config the cache needs (StaticCache.__init__ requires one).
    if not arg.config_path:
        raise ValueError(
            f"cache spec for {arg.cache_path!r} is missing config_path; "
            f"regenerate the module config with a config-emitting generator."
        )
    cfg_module, _, cfg_cls_name = arg.config_path.rpartition(".")
    config_cls = getattr(importlib.import_module(cfg_module), cfg_cls_name)
    config = config_cls(**arg.config_kwargs)

    # Build the past K/V tensors (populated slice) on CPU, then relocate to the
    # test device so the cache pins its backing buffers there (see docstring).
    key = _move_to_test_device(arg.key.build(seed=seed), test_device)
    value = _move_to_test_device(
        arg.value.build(seed=(None if seed is None else seed + 1)), test_device
    )

    # max_cache_len must cover the past; fall back to the captured past length.
    past_len = key.shape[-2]
    max_cache_len = arg.max_cache_len or past_len

    cache = cache_cls(config=config, max_cache_len=max_cache_len)
    # Prime the target layer's slot; update() appends and returns the full span.
    # The device-placed key/value make a lazily-initialized layer pin its
    # buffers on test_device, matching the on-device decode update() later.
    cache.update(key, value, arg.layer_idx)
    return cache


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
        dtype: Optional[torch.dtype] = None,
    ) -> List[Any]:
        """Build all positional args on CPU. Delegates to InputTensorSpec.build().

        `dtype`, if given, is applied to floating-point tensor specs only (see
        InputTensorSpec._effective_dtype), so the dtype variant under test is
        reflected in the built inputs rather than always using the YAML's
        literal dtype.
        """
        cpu_args: List[Any] = []
        for i, arg in enumerate(self.args):
            inp_seed = None if seed is None else seed + i * 1000

            if isinstance(arg, InputArgTensor):
                t = arg.tensor.build(seed=inp_seed, dtype=dtype)
                cpu_args.append(_move_to_test_device(t, test_device))

            elif isinstance(arg, InputArgTensorList):
                lst = [
                    spec.build(
                        seed=(None if seed is None else seed + i * 1000 + j * 7),
                        dtype=dtype,
                    )
                    for j, spec in enumerate(arg.tensor_list)
                ]
                cpu_args.append(_move_to_test_device(lst, test_device))

            elif isinstance(arg, InputArgModule):
                cpu_args.append(_build_inner_module(arg))

            elif isinstance(arg, InputArgConfig):
                cpu_args.append(_build_hf_config(arg))

            elif isinstance(arg, InputArgValue):
                val = arg.value
                # Reject the legacy bare "<config:PATH>" marker: it carries no
                # config_kwargs and cannot be resolved to a correctly-shaped
                # config. Regenerate the YAML with the config-emitting generator.
                if isinstance(val, str) and val.startswith("<config:"):
                    raise ValueError(
                        f"Unresolved config marker {val!r}. Regenerate this module "
                        f"config so the constructor arg uses 'config_path' + "
                        f"'config_kwargs' instead of a bare '<config:...>' value."
                    )
                if (
                    test_device is not None
                    and op_name == "torch.to"
                    and isinstance(val, str)
                    and "cuda" in val
                ):
                    val = test_device
                # Handle tuples/lists from YAML (e.g., view/reshape shapes)
                # If value is a string that looks like a tuple/list, convert it
                elif isinstance(val, str) and (
                    val.startswith("(") or val.startswith("[")
                ):
                    import ast

                    try:
                        val = ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        # If conversion fails, keep as string
                        pass
                # Tuples and lists are already valid Python values
                elif isinstance(val, (tuple, list)):
                    pass
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
        seed: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Return kwargs with tensor specs built and dtype strings resolved.

        `dtype`, if given, is applied to floating-point tensor/tensor_list
        specs only (see InputTensorSpec._effective_dtype) so kwarg tensors
        (e.g. hidden_states) follow the dtype variant under test the same
        way positional args do, while non-floating kwargs (e.g. int64
        position_ids) are unaffected.

        A kwarg value may itself be a tensor spec — a dict carrying one of
        ``tensor`` / ``tensor_list`` / ``config_path`` / ``model_id`` / ``py`` — just
        like a positional arg. Those are built into real tensors/objects here via
        the same ``_parse_input_arg`` path used for positional args. Modules such
        as attention/rotary layers receive ``hidden_states`` / ``position_ids`` /
        ``position_embeddings`` as kwargs, so without this they would arrive as
        raw dicts (``'dict' object has no attribute 'shape'``).
        ``tensor`` / ``tensor_list`` / ``config_path`` / ``model_id`` / ``cache``
        / ``py`` — just like a positional arg. Those are built into real
        tensors/objects here via the same ``_parse_input_arg`` path used for
        positional args. Modules such as attention/rotary layers receive
        ``hidden_states`` / ``position_ids`` / ``position_embeddings`` as kwargs,
        so without this they would arrive as raw dicts (``'dict' object has no
        attribute 'shape'``). A DecoderLayer additionally receives
        ``past_key_values`` as a primed ``Cache`` (the ``cache`` spec) so its
        decode path runs ``past_key_values.update(...)`` over real past tokens.

        For plain (non-spec) string values the resolution order is:
        1. dtype alias ("float16" / "torch.float16") -> torch.dtype via DTYPE_STR_MAP
        2. device key with "cuda:*" value            -> test_device
        3. ast.literal_eval fallback                 -> Python literal (tuple, int, etc.)
        4. pass through as-is

        None, bool, and numeric values pass through unchanged.

        A bare ``device_layout`` dict with no ``tensor`` wrapper isn't a shape
        _parse_input_arg understands (device_layout only exists nested inside
        an InputTensorSpec), so it's rejected loudly rather than silently
        passed through as an unbuilt raw dict.
        """
        import ast as _ast

        # Tensor-spec dicts carry exactly one of these keys; anything else is a
        # plain scalar/dtype/device value handled by the string branch below.
        _SPEC_KEYS = {
            "tensor",
            "tensor_list",
            "module_path",
            "config_path",
            "model_id",
            "cache",
            "py",
        }

        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(self.kwargs.items()):
            # Build tensor/tensor_list/config/cache/py specs into real objects,
            # mirroring build_cpu_args() for positional args. Use a per-key seed
            # offset so distinct kwargs don't share identical random data.
            if isinstance(v, dict) and (set(v.keys()) & _SPEC_KEYS):
                arg = _parse_input_arg(v)
                inp_seed = None if seed is None else seed + 500000 + i * 131
                if isinstance(arg, InputArgTensor):
                    t = arg.tensor.build(seed=inp_seed, dtype=dtype)
                    out[k] = _move_to_test_device(t, test_device)
                elif isinstance(arg, InputArgTensorList):
                    lst = [
                        spec.build(
                            seed=(None if inp_seed is None else inp_seed + j * 7),
                            dtype=dtype,
                        )
                        for j, spec in enumerate(arg.tensor_list)
                    ]
                    out[k] = _move_to_test_device(lst, test_device)
                elif isinstance(arg, InputArgModule):
                    # Left on CPU: _move_to_test_device only relocates tensors
                    # (an nn.Module passes through untouched), and the wrapper
                    # that adopts this module is itself moved to the test device
                    # by the harness, which carries the adopted submodules with
                    # it.
                    out[k] = _build_inner_module(arg)
                elif isinstance(arg, InputArgConfig):
                    out[k] = _build_hf_config(arg)
                elif isinstance(arg, InputArgCache):
                    out[k] = _build_cache(arg, seed=inp_seed, test_device=test_device)
                elif isinstance(arg, InputArgPy):
                    out[k] = _eval_py_literal(arg.py)
                continue

            if isinstance(v, dict) and "device_layout" in v:
                raise NotImplementedError(
                    f"kwarg {k!r} looks like a device_layout spec ({v!r}) with no "
                    f"'tensor' wrapper. device_layout only exists inside an "
                    f"InputTensorSpec — wrap this as "
                    f"{{'tensor': {{..., 'device_layout': {v!r}}}}} instead."
                )

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
    """A named item in an include list in a module.

    Supports two input specifications:
    - constructor_inputs: Args/kwargs for module.__init__()
    - forward_inputs: Args/kwargs for module.forward() (single or list for multiple invocations)
    """

    name: str
    module_path: Optional[str] = None  # Full import path (e.g., "torch.nn.Linear")
    description: Optional[str] = None

    # When true, a device-side instance of this module has its parameters and
    # buffers REALLOCATED with the device layout the production path uses,
    # instead of a plain ``.to(device)``.
    #
    # Distinct from ``InputTensorSpec.device_layout``, which lays out a single
    # *input* tensor from an explicit device_size/stride_map: this flag concerns
    # the module's own *parameters*, whose layout the production code derives per
    # tensor (on Spyre: row-major [1, 0] dim_order on 2-D matmul weights,
    # embeddings excluded) rather than spelling out in YAML. Those rules live in
    # the device backend's own code -- on Spyre,
    # ``hf_adapters.hf_common.apply_spyre_layout_to_module`` -- so what the test
    # allocates cannot drift from what production allocates.
    #
    # Consumed by the out-of-tree custom module tests; ignored on devices with no
    # layout concept (e.g. CPU).
    apply_device_layout: bool = False

    sample_inputs_func: InputsEdits = InputsEdits()  # Legacy: forward inputs only
    constructor_inputs: Optional[InputsEdits] = None  # New: explicit constructor inputs
    forward_inputs: Optional[Union[InputsEdits, List[InputsEdits]]] = (
        None  # New: explicit forward inputs (single or list)
    )

    @model_validator(mode="before")
    @classmethod
    def parse_forward_inputs(cls, values: Any) -> Any:
        """Parse forward_inputs to handle both dict and list formats."""
        if isinstance(values, dict) and "forward_inputs" in values:
            forward_inputs = values["forward_inputs"]
            # If it's a list of dicts, parse each one as InputsEdits
            if isinstance(forward_inputs, list):
                parsed_list = []
                for item in forward_inputs:
                    if isinstance(item, dict):
                        # Parse each dict as InputsEdits
                        parsed_list.append(InputsEdits.model_validate(item))
                    else:
                        parsed_list.append(item)
                values["forward_inputs"] = parsed_list
        return values

    def resolved_input_dtypes(self) -> Set[torch.dtype]:
        """Return the floating-point dtype(s) baked into this module's tensor specs.

        edits.modules.include is additive: this is the sole source of
        ModuleInfo.dtypes for a YAML-registered module (see
        _register_custom_modules_from_edits), which in turn is the sole
        determinant of which dtype variants @modules ever generates for it.
        global.supported_dtypes only filters that set further (it can skip a
        generated variant, never add one) -- so a dtype absent here is never
        generated at all, no matter what global.supported_dtypes says.

        Non-floating dtypes (e.g. int64 position_ids) are excluded: they are
        never recast per dtype variant (see InputTensorSpec._effective_dtype)
        and have no bearing on which dtype variants should be generated.
        """
        dtypes: Set[torch.dtype] = set()
        dtypes |= _dtypes_from_inputs_edits(self.constructor_inputs)

        forward_spec = self.forward_inputs or self.sample_inputs_func
        if isinstance(forward_spec, list):
            for spec in forward_spec:
                dtypes |= _dtypes_from_inputs_edits(spec)
        else:
            dtypes |= _dtypes_from_inputs_edits(forward_spec)

        return {d for d in dtypes if d.is_floating_point}

    def build_module_input(
        self,
        *,
        seed: Optional[int],
        test_device: Optional[torch.device],
        FunctionInput,
        ModuleInput,
        dtype: Optional[torch.dtype] = None,
    ) -> Any:
        """Build a ModuleInput from the config inputs.

        Follows PyTorch's upstream module_inputs_func signature:
        module_inputs_func(module_info, device, dtype, requires_grad, training, **kwargs) -> list[ModuleInput]

        Returns a ModuleInput with:
        - constructor_input: FunctionInput with args/kwargs for module.__init__()
        - forward_input: FunctionInput with args/kwargs for module.forward()

        FunctionInput and ModuleInput are passed in as arguments to avoid importing
        torch.testing internals into this models file. `dtype`, if given, is
        applied to floating-point tensor specs only (see
        InputTensorSpec._effective_dtype) so constructor/forward tensors match
        the dtype variant the caller is currently exercising, the same way
        module.to(dtype) recasts the module's own floating parameters.
        """
        # Build constructor inputs
        constructor_spec = self.constructor_inputs or InputsEdits()
        constructor_args = constructor_spec.build_cpu_args(
            seed=seed,
            op_name=self.name,
            test_device=test_device,
            dtype=dtype,
        )
        constructor_kwargs = constructor_spec.resolved_kwargs(
            test_device=test_device, dtype=dtype
        )
        constructor_input = FunctionInput(*constructor_args, **constructor_kwargs)

        # Build forward inputs (prefer forward_inputs, fallback to sample_inputs_func for backward compat)
        forward_spec = self.forward_inputs or self.sample_inputs_func

        # Handle list format (multiple invocations) - return first one for backward compat
        # The full list handling is done in create_module_inputs_func_from_yaml
        if isinstance(forward_spec, list):
            if forward_spec:
                forward_spec = forward_spec[0]  # Use first invocation
            else:
                forward_spec = InputsEdits()  # Empty if list is empty

        forward_args = forward_spec.build_cpu_args(
            seed=(None if seed is None else seed + 10000),  # Different seed for forward
            op_name=self.name,
            test_device=test_device,
            dtype=dtype,
        )
        forward_kwargs = forward_spec.resolved_kwargs(
            test_device=test_device, dtype=dtype
        )
        forward_input = FunctionInput(*forward_args, **forward_kwargs)

        return ModuleInput(
            constructor_input=constructor_input,
            forward_input=forward_input,
        )


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
    force_xfail: bool = False


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


class FunctionItem(BaseModel):
    """A single function entry for function modification."""

    name: str  # Method name (e.g., "assertEqual")
    description: Optional[str] = None  # Optional description


class FunctionsEdits(BaseModel):
    """Per-test function modification configuration.

    Container for all function-level modifications. cpu_move is a list of
    function names that will have their tensor arguments moved to CPU.
    Extensible for future functionality.
    """

    cpu_move: List[FunctionItem] = []

    def resolved_cpu_move_functions(self) -> List[str]:
        """Return list of function names to patch with CPU move."""
        return [item.name for item in self.cpu_move]


class TestEdits(BaseModel):
    ops: OpsEdits = OpsEdits()
    dtypes: DtypesEdits = DtypesEdits()
    modules: ModulesEdits = ModulesEdits()
    functions: FunctionsEdits = FunctionsEdits()


class TestEntry(BaseModel):
    """A single test entry in the per-file tests: names, mode, tags and edits"""

    __test__ = False  # prevent pytest from collecting this as a test class

    names: List[str]
    mode: str = MODE_MANDATORY_SUCCESS
    tags: List[str] = []
    labels: List[str] = []
    no_grad: bool = False
    edits: TestEdits = TestEdits()

    @field_validator("names", mode="before")
    @classmethod
    def validate_name(cls, v) -> List[str]:
        if isinstance(v, str):
            v = [v]
        for item in v:
            parts = item.split("::")
            if len(parts) == 1:
                # Plain method name (no class) -- valid for module-level test functions
                if not parts[0]:
                    raise ValueError(
                        f"Invalid test id {item!r}: test name cannot be empty"
                    )
            elif len(parts) == 2:
                # ClassName::method_name format
                if not all(parts):
                    raise ValueError(
                        f"Invalid test id {item!r}, expected 'ClassName::method_name' or plain 'method_name'"
                    )
            else:
                raise ValueError(
                    f"Invalid test id {item!r}, expected 'ClassName::method_name' or plain 'method_name'"
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
        """Return [(class_name_or_None, method_name), ...] for all entries in names."""
        result: List[tuple] = []
        for n in self.names:
            parts = n.split("::")
            if len(parts) == 1:
                result.append((None, parts[0]))
            else:
                result.append((parts[0], parts[1]))
        return result

    def method_names(self) -> List[str]:
        """Return just the method_name part of each entry."""
        return [n.split("::")[-1] for n in self.names]

    def class_names(self) -> List[Optional[str]]:
        """Return just the class_name part of each entry, or None for plain method names."""
        result: List[Optional[str]] = []
        for n in self.names:
            parts = n.split("::")
            result.append(parts[0] if len(parts) == 2 else None)
        return result


class FileEntry(BaseModel):
    """Per file model containing path, unlisted_test_mode and a list of tests."""

    path: str
    unlisted_test_mode: str = MODE_XFAIL
    # Not a real YAML field -- never set by a hand-written config. Populated
    # by merge_yaml_configs() with the origin config's test_suite_config.labels
    # so per-file labels survive a multi-config merge (which drops the
    # top-level labels field, since it can't represent per-file provenance
    # once files from different configs are combined). See
    # OOTTestBase._load_test_suite_config(), which prefers this over
    # test_suite_config.labels when non-empty.
    labels: List[str] = []
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
        qualified = f"{class_name}::{method_name}"
        for entry in self.tests:
            if qualified in entry.names or method_name in entry.names:
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
    """Model for storing supported modules config: name, force_xfail, dtypes.

    Supports inline input specification via constructor_inputs and forward_inputs.
    """

    name: str
    force_xfail: bool = False
    dtypes: List[SupportedOpDtypeConfig] = []
    constructor_inputs: Optional[InputsEdits] = None  # Inline constructor inputs
    forward_inputs: Optional[Union[InputsEdits, List[InputsEdits]]] = (
        None  # Inline forward inputs (single or list)
    )

    def get_name(self) -> str:
        return self.name

    def resolved_dtypes(self) -> Optional[Set[torch.dtype]]:
        if not self.dtypes:
            return None
        return {d.resolved_dtype() for d in self.dtypes}

    def has_inline_inputs(self) -> bool:
        """Check if this config has inline input specifications."""
        has_constructor = (
            self.constructor_inputs is not None and self.constructor_inputs.has_inputs()
        )
        has_forward = False
        if self.forward_inputs is not None:
            if isinstance(self.forward_inputs, list):
                has_forward = any(inp.has_inputs() for inp in self.forward_inputs)
            else:
                has_forward = self.forward_inputs.has_inputs()
        return has_constructor or has_forward


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

    def resolved_supported_dtypes_force_xfail(self) -> Set[torch.dtype]:
        """Return the set of dtypes that have force_xfail: true."""
        return {
            parse_dtype(item.name) for item in self.supported_dtypes if item.force_xfail
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
    """Holds the inner YAML keys: files, global, and suite-level metadata."""

    files: List[FileEntry]
    global_config: GlobalConfig = GlobalConfig()
    labels: List[str] = []

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

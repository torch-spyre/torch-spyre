"""
Pydantic models for the OOT PyTorch test framework YAML config.

Used by spyre_test_parsing.py to validate and parse the YAML config.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from pydantic import BaseModel, field_validator, model_validator  # type: ignore

from spyre_test_constants import (
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


_VALID_TEST_MODES = {MODE_MANDATORY_SUCCESS, MODE_XFAIL, MODE_XFAIL_STRICT, MODE_SKIP}

_VALID_UNLISTED_MODES = {"skip", "xfail", "xfail_strict", "mandatory_success"}


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


class DtypeNamedItem(BaseModel):
    """A dtype item with optional precision override."""

    name: str
    description: Optional[str] = None
    precision: Optional[Precision] = None


class OpsEdits(BaseModel):
    """Per-test op list overrides."""

    include: List[NamedItem] = []  # inject ops into @ops.op_list
    exclude: List[NamedItem] = []  # remove ops from @ops.op_list

    def included_op_names(self) -> Set[str]:
        return {item.name for item in self.include}

    def excluded_op_names(self) -> Set[str]:
        return {item.name for item in self.exclude}


class ModulesEdits(BaseModel):
    """Per-test module list overrides."""

    include: List[NamedItem] = []  # inject modules into @modules.module_info_list
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
    """Per file model containing path, unlisted_test_mode and a list of tests"""

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
        for entry in self.tests:
            if entry.class_name == class_name and entry.method_name == method_name:
                return entry
        return None


class SupportedOpDtypeConfig(BaseModel):
    """Model for supported_ops.dtype: name, precision"""

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
    """Model for storing supported ops config: name, force_xfail, list of dtypes"""

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
    """Model for storing supported modules config: name, force_xfail"""

    name: str
    force_xfail: bool = False

    def get_name(self) -> str:
        return self.name


class GlobalConfig(BaseModel):
    """Model for global configs: supported_dtypes, supported_ops"""

    supported_dtypes: List[DtypeNamedItem] = []
    supported_ops: Optional[List[SupportedOpConfig]] = None
    supported_modules: Optional[List[SupportedModuleConfig]] = None

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

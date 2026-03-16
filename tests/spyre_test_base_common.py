"""
Shared class and methods for all Spyre PyTorch test overrides.

Each per-suite file (e.g. spyre_test_binaryfuncs.py) imports
SpyreTestBase from here and declares WHITELISTED_TESTS and/or BLACKLISTED_TESTS
as class attributes.  A single environment variable selects which dict
is active at runtime.

# New ENV VAR introduced
SPYRE_PYTORCH_TEST_FILTER_TYPE=whitelist --> use WHITELISTED_TESTS  (default when it exists)
SPYRE_PYTORCH_TEST_FILTER_TYPE=blacklist --> use BLACKLISTED_TESTS (default when only that exists)

If a suite file defines BOTH dicts, set SPYRE_PYTORCH_TEST_FILTER_TYPE explicitly to
choose which one governs the run.  When only one dict is defined the
mode is inferred automatically and SPYRE_PYTORCH_TEST_FILTER_TYPE need not be set.

Usage as we already had apart from a new environment variable that got added
    export PYTORCH_TESTING_DEVICE_ONLY_FOR="privateuse1"

    # Clone pytorch
    $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/checkout-pytorch-src.sh

    export TORCH_TEST_DEVICES="$DTI_PROJECT_ROOT/torch-spyre/tests/spyre_test_base_common.py"
    export SPYRE_PYTORCH_TEST_FILTER_TYPE=whitelist  # or blacklist
    export SPYRE_PYTORCH_TEST_CONFIG=tests/test_binary_ufuncs.yaml

    cd $DTI_PROJECT_ROOT/pytorch/test/
    python3 -m pytest test_binary_ufuncs.py -v (Example upstream test)
"""

import os
import re
import unittest
from functools import wraps
from typing import Dict, Optional, Set
import yaml
from pathlib import Path

import torch
# from torch.testing._internal.common_device_type import ops as _ops_parametrizer
# common_device_type.py is the one running our suite file via runpy, so it's not fully initialized yet when we try to import from it.
# The fix is to do a lazy import inside the function, not at module level:


# ------------
# Constants
# -----------

DEFAULT_FLOATING_PRECISION: float = 1e-3

# Default set of unsupported dtypes on spyre (Per-suite subclasses may extend this set)
DEFAULT_UNSUPPORTED_DTYPES: Set[torch.dtype] = {
    torch.complex32,
    torch.complex64,
    torch.complex128,
}

# Valid values for SPYRE_PYTORCH_TEST_FILTER_TYPE
_MODE_WHITELIST = "whitelist"
_MODE_BLACKLIST = "blacklist"

# ----------------------------
# Dtype helper data structures
# ----------------------------

_DTYPE_STR_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "bool": torch.bool,
}

# Ordered longest-first so "complex128" matches before "complex12"
_DTYPE_NAMES_ORDERED = sorted(_DTYPE_STR_MAP.keys(), key=len, reverse=True)


def extract_dtype_from_name(method_name: str) -> Optional[str]:
    """Return the dtype suffix embedded in *method_name*, or None."""
    for dtype in _DTYPE_NAMES_ORDERED:
        if f"_{dtype}_" in method_name or method_name.endswith(f"_{dtype}"):
            return dtype
    return None


def parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in _DTYPE_STR_MAP:
        raise ValueError(f"Unknown dtype string: {dtype_str!r}")
    return _DTYPE_STR_MAP[dtype_str]


# -------------------
# Match-set helpers
# -------------------


class MatchSet:
    """Holds exact names and regex patterns for fast membership tests."""

    def __init__(self):
        self.exact: Set[str] = set()
        self.regex: Set[str] = set()

    @classmethod
    def from_iterable(cls, items):
        ms = cls()
        for m in items:
            if re.match(r"\w+$", m):
                ms.exact.add(m)
            else:
                ms.regex.add(m)
        return ms

    def matches(self, name: str) -> bool:
        if name in self.exact:
            return True
        return any(re.match(pattern, name) for pattern in self.regex)


def _build_match_sets(d: Dict[str, set]) -> Dict[str, MatchSet]:
    return {k: MatchSet.from_iterable(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# PrivateUse1TestBase filter
# ---------------------------------------------------------------------------
# TODO: figure out why this filter is needed - expected to use default PrivateUse1TestBase
def remove_builtin_privateuse1_test_base():
    """
    Remove built-in PrivateUse1TestBase from device_type_test_bases.

    This ensures only SpyreTestBase handles the privateuse1 device type,
    preventing nondeterministic overwrites when list(set(...)) randomizes order.

    Side effect: Modifies the global device_type_test_bases list in-place.

    TODO: investigate whether this filter will still be needed once the upstream
          PrivateUse1TestBase correctly defers to registered custom backends.
    """
    device_type_test_bases[:] = [  # type: ignore[name-defined] # noqa: F821
        b
        for b in device_type_test_bases  # type: ignore[name-defined] # noqa: F821
        if b is not PrivateUse1TestBase  # type: ignore[name-defined] # noqa: F821
    ]


# Call the filter function to apply the side effect
remove_builtin_privateuse1_test_base()


class _SpyreDtypePatcher:
    """Patches @ops allowed_dtypes on a bound test method before instantiation.

    Needed because upstream @ops(..., allowed_dtypes=(...)) restricts which dtype
    variants are generated -- dtypes absent here are never instantiated, so they
    cannot be whitelisted. We inject extra dtypes before
    super().instantiate_test() calls _parametrize_test.
    """

    def __init__(self, test, extra_dtypes: set):
        from torch.testing._internal.common_device_type import ops as _ops_cls

        # @ops instance lives at test.__func__.parametrize_fn.__self__
        underlying_fn = test.__func__ if hasattr(test, "__func__") else test
        p = getattr(underlying_fn, "parametrize_fn", None)
        self._ops_instance = (
            p.__self__
            if p is not None
            and hasattr(p, "__self__")
            and isinstance(p.__self__, _ops_cls)
            else None
        )
        self._extra_dtypes = extra_dtypes

    def patch(self) -> None:
        if (
            self._ops_instance is not None
            and self._ops_instance.allowed_dtypes is not None
        ):
            self._ops_instance.allowed_dtypes |= self._extra_dtypes


# PrivateUse1TestBase injected via globals()
class SpyreTestBase(PrivateUse1TestBase):  # type: ignore[name-defined] # noqa: F821
    """
    Base class for Spyre device-type tests.

    You will need to inherit this class + PrivateUse1TestBase in each per-suite
    file.  Declare WHITELISTED_TESTS, BLACKLISTED_TESTS, or both as class
    attributes (which will be controlled by SPYRE_PYTORCH_TEST_FILTER_TYPE env variable).
    """

    device_type: str = "privateuse1"
    precision: float = DEFAULT_FLOATING_PRECISION

    # Override in per-suite subclasses.
    WHITELISTED_TESTS: Dict[str, set] = {}
    BLACKLISTED_TESTS: Dict[str, set] = {}
    PRECISION_OVERRIDES: Dict[str, float] = {}
    # Maps base test name --> set of torch.dtype to inject into @ops `allowed_dtypes``.
    # This is to add the capability when upstream @ops(..., allowed_dtypes=...) omits dtypes Spyre supports.
    EXTRA_ALLOWED_DTYPES: Dict[str, set] = {}

    # Extend in per-suite subclasses for backend-specific dtype gaps.
    unsupported_dtypes: Set[torch.dtype] = DEFAULT_UNSUPPORTED_DTYPES

    @classmethod
    def _load_test_suite_config(cls) -> None:
        """
        Load YAML config pointed to by SPYRE_PYTORCH_TEST_CONFIG
        and populate class attributes dynamically.
        """
        path = os.environ.get("SPYRE_PYTORCH_TEST_CONFIG")
        if not path:
            return

        # Avoid reloading multiple times
        if getattr(cls, "_yaml_loaded", False):
            return

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Spyre config file not found: {config_path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # --------------------------
        # VALIDATE CONFIG KEYS
        # --------------------------
        # Catch typos early
        valid_keys = {
            "_WHITELISTED",
            "_BLACKLISTED",
            "_PRECISION_OVERRIDES",
            "_UNSUPPORTED_DTYPES",
        }
        invalid_keys = set(data.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid keys in {config_path}: {invalid_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        # --------------------------
        # WHITELIST / BLACKLIST
        # --------------------------
        cls.WHITELISTED_TESTS = {
            k: set(v) for k, v in data.get("_WHITELISTED", {}).items()
        }

        cls.BLACKLISTED_TESTS = {
            k: set(v) for k, v in data.get("_BLACKLISTED", {}).items()
        }

        # --------------------------
        # PRECISION OVERRIDES
        # --------------------------
        cls.PRECISION_OVERRIDES = data.get("_PRECISION_OVERRIDES", {})

        # EXTRA ALLOWED DTYPES
        extra: Dict[str, set] = {}
        for _, tests in data.get("_WHITELISTED", {}).items():
            for test_name, test_cfg in (tests or {}).items():
                if test_cfg and "extra_allowed_dtypes" in test_cfg:
                    extra[test_name] = {
                        parse_dtype(dt) for dt in test_cfg["extra_allowed_dtypes"]
                    }
        cls.EXTRA_ALLOWED_DTYPES = extra

        # --------------------------
        # UNSUPPORTED DTYPES
        # --------------------------
        unsupported = data.get("_UNSUPPORTED_DTYPES")
        if unsupported:
            cls.unsupported_dtypes = {parse_dtype(dt) for dt in unsupported}

        cls._yaml_loaded = True

    @classmethod
    def _resolve_mode(cls) -> str:
        """
        Return the active mode: 'whitelist' or 'blacklist'.
        Priority:
          1. SPYRE_PYTORCH_TEST_FILTER_TYPE env var
          2. Inferred from which dicts are populated on the class
        """
        env = os.environ.get("SPYRE_PYTORCH_TEST_FILTER_TYPE", "").strip().lower()
        if env in (_MODE_WHITELIST, _MODE_BLACKLIST):
            return env
        if env:
            raise ValueError(
                f"SPYRE_PYTORCH_TEST_FILTER_TYPE={env!r} is invalid. "
                f"Use 'whitelist' or 'blacklist'."
            )

        # Prefer whitelist if WHITELISTED_TESTS is populated (priority)
        if cls.WHITELISTED_TESTS:
            return _MODE_WHITELIST

        # Prefer blacklist if BLACKLISTED_TESTS is populated
        if cls.BLACKLISTED_TESTS:
            return _MODE_BLACKLIST

        # Nothing is defined ->  blacklist mode (run everything by default)
        return _MODE_BLACKLIST

    # ----------------------------
    # Compiled match-set cache
    # ----------------------------
    @classmethod
    def _get_active_match_sets(cls) -> Dict[str, MatchSet]:
        """Return compiled MatchSets for whichever dict is active."""
        mode = cls._resolve_mode()
        cache_attr = f"_cached_msets_{mode}"
        if cache_attr not in cls.__dict__ or cls.__dict__[cache_attr] is None:
            source = (
                cls.WHITELISTED_TESTS
                if mode == _MODE_WHITELIST
                else cls.BLACKLISTED_TESTS
            )
            setattr(cls, cache_attr, _build_match_sets(source))
        return cls.__dict__[cache_attr]

    # Decide whether an instantiated test method should run
    @classmethod
    def _should_run(
        cls,
        method_name: str,
        base_test_name: str,
        generic_cls_name: str,
    ) -> tuple[bool, Optional[str]]:
        """

        Whitelist mode
        -> Test is in WHITELISTED_TESTS for this class then RUN
        -> Otherwise SKIP

        Blacklist mode
        -> Test is in BLACKLISTED_TESTS for this class then SKIP
        -> Otherwise RUN with dtype filter applied

        Dtype filtering (blacklist mode only)
          Tests with unsupported dtype are skipped even if
          not explicitly listed in BLACKLISTED_TESTS.
          In whitelist mode, we assume that the
          user is aware of the supported dtype.
        """
        mode = cls._resolve_mode()
        match_sets = cls._get_active_match_sets()
        mset = match_sets.get(generic_cls_name)

        def _name_matches(ms: Optional[MatchSet]) -> bool:
            if ms is None:
                return False
            return ms.matches(method_name) or ms.matches(base_test_name)

        if mode == _MODE_WHITELIST:
            if _name_matches(mset):
                return True, None
            return False, "Not in WHITELISTED_TESTS"

        else:  # blacklist
            if _name_matches(mset):
                return False, "DISABLED FOR SPYRE"

            # Dtype filter
            dtype_str = extract_dtype_from_name(method_name)
            if dtype_str:
                try:
                    dtype = parse_dtype(dtype_str)
                    if dtype in cls.unsupported_dtypes:
                        return False, f"Unsupported dtype: {dtype_str}"
                except ValueError:
                    pass

            return True, None

    # ---------------------------
    # instantiate_test override
    # ---------------------------
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        # Load test-suite config
        cls._load_test_suite_config()

        # Per-test precision override
        cls.precision = cls.PRECISION_OVERRIDES.get(name, DEFAULT_FLOATING_PRECISION)
        extra_dtypes = cls.EXTRA_ALLOWED_DTYPES.get(name)

        if extra_dtypes:
            # test is a bound method; @ops instance is at test.__func__.parametrize_fn.__self__
            # We patch allowed_dtypes directly on it before super() calls _parametrize_test,
            # so extra dtype variants are generated in the normal flow.
            # Safe to mutate since `test` is already a deepcopy from upstream.
            _SpyreDtypePatcher(test, extra_dtypes).patch()

        # Let the parent class generate all variant methods first
        existing_methods = set(cls.__dict__.keys())
        super().instantiate_test(name, test, generic_cls=generic_cls)
        new_methods = set(cls.__dict__.keys()) - existing_methods

        for method_name in new_methods:
            enabled, reason = cls._should_run(
                method_name=method_name,
                base_test_name=name,
                generic_cls_name=generic_cls.__name__,
            )

            if not enabled:
                skip_reason = reason or "Skipped for Spyre"

                @wraps(test)
                def _skip(self, _reason=skip_reason):
                    raise unittest.SkipTest(_reason)

                setattr(cls, method_name, _skip)


TEST_CLASS = SpyreTestBase

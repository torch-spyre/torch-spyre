"""
Constants for the Spyre PyTorch test framework.
All string literals, default values, and dtype maps are here.
"""

from typing import Dict, Set

import torch


DEFAULT_FLOATING_PRECISION: float = 1e-3

MODE_MANDATORY_SUCCESS = "mandatory_success"
MODE_XFAIL = "xfail"
MODE_XFAIL_STRICT = "xfail_strict"
MODE_SKIP = "skip"

MODE_DEFAULT_TEST = "mandatory_success"

UNLISTED_MODE_SKIP = "skip"
UNLISTED_MODE_XFAIL = "xfail"
UNLISTED_MODE_XFAIL_STRICT = "xfail_strict"
UNLISTED_MODE_MANDATORY_SUCCESS = "mandatory_success"
UNLISTED_TEST_MODE_DEFAULT = UNLISTED_MODE_XFAIL

# ADD: all valid modes in one set for validation
_VALID_TEST_MODES = {MODE_MANDATORY_SUCCESS, MODE_XFAIL, MODE_XFAIL_STRICT, MODE_SKIP}
_VALID_UNLISTED_MODES = {
    UNLISTED_MODE_SKIP,
    UNLISTED_MODE_XFAIL,
    UNLISTED_MODE_XFAIL_STRICT,
    UNLISTED_MODE_MANDATORY_SUCCESS,
}

# --------------------
# Dtype defaults
# --------------------

DEFAULT_UNSUPPORTED_DTYPES: Set[torch.dtype] = {
    torch.complex32,
    torch.complex64,
    torch.complex128,
}

# ---------------------------------------------------------------------------
# Dtype string -> torch.dtype map
# Ordered longest-first so "complex128" is matched before "complex12", etc.
# ---------------------------------------------------------------------------

DTYPE_STR_MAP: Dict[str, torch.dtype] = {
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
    "half": torch.half,
}

DTYPE_NAMES_ORDERED = sorted(DTYPE_STR_MAP.keys(), key=len, reverse=True)

# ------------------------------
# Environment variables
# ------------------------------

ENV_TEST_CONFIG = "PYTORCH_TEST_CONFIG"
ENV_TORCH_ROOT = "TORCH_ROOT"
ENV_TORCH_DEVICE_ROOT = "TORCH_DEVICE_ROOT"

# -------------------------------------
# rel_path tokens -> env var names
# -------------------------------------

REL_PATH_TOKENS = (
    ("${TORCH_ROOT}", ENV_TORCH_ROOT),
    ("${TORCH_DEVICE_ROOT}", ENV_TORCH_DEVICE_ROOT),
)

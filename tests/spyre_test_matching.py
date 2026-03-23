"""
Dtype helpers and MatchSet for the Spyre PyTorch test framework.

Responsibilities:
  - parse_dtype / extract_dtype_from_name
  - MatchSet: exact + regex membership tests
  - build_match_sets: compile a {class_name -> MatchSet} dict
"""

from typing import Dict, Optional, Set

import regex as re
import torch

from spyre_test_constants import DTYPE_STR_MAP, DTYPE_NAMES_ORDERED


# ----------------
# Dtype helpers
# ----------------


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Map a dtype string (e.g. 'float16') to a torch.dtype.  Raises ValueError if unknown."""
    if dtype_str not in DTYPE_STR_MAP:
        raise ValueError(f"Unknown dtype string: {dtype_str!r}")
    return DTYPE_STR_MAP[dtype_str]


def extract_dtype_from_name(method_name: str) -> Optional[str]:
    """Return the dtype suffix embedded in *method_name*, or None.

    Matches '_float16' as a suffix or '_float16_' as an infix.
    Longest dtype names are tried first to avoid partial matches.
    """
    for dtype in DTYPE_NAMES_ORDERED:
        if f"_{dtype}_" in method_name or method_name.endswith(f"_{dtype}"):
            return dtype
    return None


# ----------
# MatchSet
# ----------


class MatchSet:
    """Holds exact names and compiled regex patterns for fast membership tests.

    Plain identifiers (e.g. 'test_add') go into the exact set for O(1) lookup.
    Anything containing non-word characters is treated as a regex pattern.
    """

    def __init__(self):
        self.exact: Set[str] = set()
        self.patterns: Set[str] = set()

    @classmethod
    def from_iterable(cls, items) -> "MatchSet":
        ms = cls()
        for item in items:
            if re.match(r"\w+$", item):
                ms.exact.add(item)
            else:
                ms.patterns.add(item)
        return ms

    def matches(self, name: str) -> bool:
        if name in self.exact:
            return True
        return any(re.match(pattern, name) for pattern in self.patterns)


def build_match_sets(d: Dict[str, set]) -> Dict[str, "MatchSet"]:
    """Compile a {class_name -> set-of-names} dict into {class_name -> MatchSet}."""
    return {k: MatchSet.from_iterable(v) for k, v in d.items()}

"""
Core class data structure used across all modules.
"""

from __future__ import annotations

import regex as re
from pathlib import Path


class PatternEntry:
    """
    One pattern entry read from a YAML config file.

    Attributes
    ----------
    config_file : Path
        The config YAML file this pattern came from.
    test_file_basename : str
        Basename of the test .py file this pattern targets
        (e.g. "test_inductor_ops.py").
    raw : str
        The raw pattern string as written in the YAML, including any
        "ClassName::" prefix (e.g. "TestOps::test_foo.*").
    pattern : str
        The regex part only, with any "ClassName::" prefix stripped.
        This is what gets passed to re.fullmatch against base names.
    """

    __slots__ = ("config_file", "test_file_basename", "raw", "pattern")

    def __init__(self, config_file: Path, test_file_basename: str, raw: str):
        self.config_file = config_file
        self.test_file_basename = test_file_basename
        self.raw = raw
        # Strip optional "ClassName::" prefix -- matching is done on base name only
        self.pattern = raw.split("::", 1)[-1].strip() if "::" in raw else raw.strip()

    def matches(self, base_name: str) -> bool:
        """Return True if this pattern fullmatches the given base method name."""
        try:
            return bool(re.fullmatch(self.pattern, base_name))
        except re.error:
            return False

    def __repr__(self) -> str:
        return f"{self.raw!r}  ({self.config_file.name})"

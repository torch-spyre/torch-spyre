"""
Loads OOT YAML config files and extracts PatternEntry objects from them.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required:  pip install pyyaml")

from .models import PatternEntry


def load_all_patterns(config_files: list[Path]) -> dict[str, list[PatternEntry]]:
    """
    Load all YAML config files and return their patterns grouped by the
    basename of the test .py file they target.

    Parameters
    ----------
    config_files : list[Path]
        Paths to OOT YAML config files.

    Returns
    -------
    dict[str, list[PatternEntry]]
        Keys are test-file basenames (e.g. "test_inductor_ops.py").
        Values are all PatternEntry objects targeting that file, across
        all configs.
    """
    result: dict[str, list[PatternEntry]] = defaultdict(list)
    for cf in config_files:
        try:
            cfg = yaml.safe_load(cf.read_text()) or {}
        except Exception as e:
            _warn(f"Cannot load {cf}: {e}")
            continue
        for file_entry in cfg.get("test_suite_config", {}).get("files", []):
            raw_path = file_entry.get("path", "")
            basename = Path(raw_path).name  # last component only
            for block in file_entry.get("tests", []):
                for raw_name in block.get("names", []):
                    result[basename].append(PatternEntry(cf, basename, raw_name))
    return result


def _warn(msg: str) -> None:
    import sys

    print(f"[WARN] {msg}", file=sys.stderr)

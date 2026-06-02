"""
Parses a Python test file using the AST to determine which test methods
are collectable by the OOT framework and which are helper-only.

The OOT framework collects:
  (a) PARAMS primaries -- the 1st element of each PARAMS tuple key, e.g.
      ("test_addmm", "test_addmm_cpu") → "test_addmm" gets collected.
  (b) Standalone def test_* methods  not mentioned in PARAMS at all.

Helper-only methods (only the 2nd element of a PARAMS tuple and NOT a
primary) are never directly collected and are excluded from MISS checks.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def parse_test_file(test_file: Path) -> tuple[list[str], set[str]]:
    """
    Parse a Python test file and classify its test methods.

    Parameters
    ----------
    test_file : Path
        Path to the Python test file (e.g. test_inductor_ops.py).

    Returns
    -------
    collectable_names : list[str]
        Sorted list of method names the OOT framework will actually collect:
        PARAMS primaries + standalone test methods.
    helper_only : set[str]
        Method names that are ONLY PARAMS helpers (2nd tuple element) and
        are never directly collected by the framework.
    """
    try:
        tree = ast.parse(test_file.read_text())
    except SyntaxError as e:
        sys.exit(f"Cannot parse {test_file}: {e}")

    all_methods = _collect_all_test_methods(tree)
    primaries, helpers = _collect_params_roles(tree)

    # Helper-only: appears as a helper but never as a primary
    helper_only = helpers - primaries

    # Collectable = PARAMS primaries + standalones (not in PARAMS at all)
    in_params = primaries | helpers
    standalones = all_methods - in_params
    collectable = sorted(primaries | standalones)

    return collectable, helper_only


# -------------------------
# Internal helpers
# -------------------------


def _collect_all_test_methods(tree: ast.AST) -> set[str]:
    """Return every def test_* method name defined inside any class."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    names.add(item.name)
    return names


def _collect_params_roles(tree: ast.AST) -> tuple[set[str], set[str]]:
    """
    Walk the PARAMS class-level dict and return (primaries, helpers).

    PARAMS is expected to be a dict with tuple keys:
        ("test_primary", "test_helper"): { ... }

    The first element of each key is the primary (gets collected + expanded).
    The second element is the helper (implements the actual assertion logic).
    """
    primaries: set[str] = set()
    helpers: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.Assign):
                continue
            for target in item.targets:
                if not (isinstance(target, ast.Name) and target.id == "PARAMS"):
                    continue
                if not isinstance(item.value, ast.Dict):
                    continue
                for key in item.value.keys:
                    if isinstance(key, ast.Tuple) and len(key.elts) >= 2:
                        if isinstance(key.elts[0], ast.Constant):
                            primaries.add(key.elts[0].value)
                        if isinstance(key.elts[1], ast.Constant):
                            helpers.add(key.elts[1].value)

    return primaries, helpers

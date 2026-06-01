"""
ANSI colour helpers and heuristic base-name derivation used for output
formatting and no-test-file fallback mode.
"""

from __future__ import annotations

import regex as re
import sys

from .models import PatternEntry


# -----------------------------------------------------------------------------
# ANSI colour helpers for better readability
# -----------------------------------------------------------------------------


def _c(text: str, code: str) -> str:
    return f"{code}{text}\033[0m" if sys.stdout.isatty() else text


def red(t: str) -> str:
    return _c(t, "\033[31m")


def yellow(t: str) -> str:
    return _c(t, "\033[33m")


def green(t: str) -> str:
    return _c(t, "\033[32m")


def bold(t: str) -> str:
    return _c(t, "\033[1m")


def cyan(t: str) -> str:
    return _c(t, "\033[36m")


# ---------------------
# Warning helper
# ---------------------


def warn(msg: str) -> None:
    """Print a warning to stderr."""
    print(f"[WARN] {msg}", file=sys.stderr)


# ----------------------------------------------------------------------
# Heuristic base-name derivation (used when no --test-file is given)
# ---------------------------------------------------------------------


def heuristic_base_names(patterns: list[PatternEntry]) -> list[str]:
    """
    Derive candidate base names from the patterns themselves when no
    reference test file is available.

    Handles two common pattern forms:
      - Alternation:  test_(addmm|mm|bmm).*  ->  [test_addmm, test_mm, test_bmm]
      - Literal prefix:  test_clone.*        ->  [test_clone]

    """
    candidates: set[str] = set()
    for pe in patterns:
        p = pe.pattern
        # Expand alternation groups: test_(a|b|c).* -> test_a, test_b, test_c
        m = re.match(r"^(test_\w*)\(([^)]+)\)", p)
        if m:
            prefix = m.group(1)
            for alt in m.group(2).split("|"):
                candidates.add(prefix + alt)
        else:
            # Strip metacharacters to recover the longest literal prefix
            literal = re.split(r"[.*(+?\[{|]", p)[0]
            if literal.startswith("test_"):
                candidates.add(literal)
    return sorted(candidates)

"""
The three independent checks run against the loaded config patterns and
the parsed test file:

  check_duplicates    -- same base name covered by multiple configs
  check_missing       -- collectable base name not covered by any config
  check_dead_patterns -- config pattern that matches nothing in the test file
"""

from __future__ import annotations

from .models import PatternEntry
from .display import red, yellow, green, cyan, bold


# -----------------------
# CHECK 1: Duplicates
# -----------------------


def check_duplicates(
    base_names: list[str],
    patterns: list[PatternEntry],
    label: str,
) -> int:
    """
    For each base name, find every pattern that matches it.
    If more than one config file covers the same name -> duplicate.

    Parameters
    ----------
    base_names : list[str]
        All names to check (collectable + helpers for thorough detection).
    patterns : list[PatternEntry]
        All patterns across all configs for this test file.
    label : str
        Display label (usually the test file basename).

    Returns
    -------
    int
        Number of duplicated base names found.
    """
    # Map each name to all patterns that match it
    coverage: dict[str, list[PatternEntry]] = {}
    for name in base_names:
        hits = [pe for pe in patterns if pe.matches(name)]
        if hits:
            coverage[name] = hits

    # Keep only names matched by patterns from MORE THAN ONE config file
    dups = {
        name: hits
        for name, hits in coverage.items()
        if len({pe.config_file for pe in hits}) > 1
    }

    if not dups:
        print(green(f"  [{label}] No duplicates.\n"))
        return 0

    print(red(f"  [{label}] {len(dups)} duplicate(s):\n"))
    for name in sorted(dups):
        print(f"    {red('DUP')}  {bold(name)}")
        for pe in dups[name]:
            print(f"          pattern : {cyan(pe.raw)}")
            print(f"          config  : {pe.config_file.name}")
        print()
    return len(dups)


# -----------------------
# CHECK 2: Missing
# -----------------------


def check_missing(
    collectable_names: list[str],
    helper_only: set[str],
    patterns: list[PatternEntry],
    label: str,
) -> int:
    """
    Find collectable base names not covered by any config pattern.

    A name is considered covered if:
      (a) Some pattern fullmatches it directly (e.g. TestOps::test_mean), OR
      (b) Some pattern targets a param-key variant of it, i.e. the pattern's
          literal text starts with base_name + "_"
          (e.g. "test_mean_3d_dim0" covers the PARAMS primary "test_mean").

    Rule (b) exists because developers sometimes write patterns against
    param-key-suffixed names rather than the bare base name. The framework
    collects and expands the base; the config selects which param variants
    to run but coverage-wise the base is considered covered as long as
    at least one of its param variants is targeted.

    Helper-only methods are excluded (never directly collected by the OOT
    framework).

    Returns
    -------
    int
        Number of uncovered collectable base names.
    """
    uncovered = [n for n in collectable_names if not _is_covered(n, patterns)]

    # filter out helpers that somehow slipped into collectable_names
    true_misses = [n for n in uncovered if n not in helper_only]
    false_positives = [n for n in uncovered if n in helper_only]

    if false_positives:
        import sys

        print(
            f"[WARN] {len(false_positives)} helper method(s) skipped from "
            f"MISS check (never directly collected by OOT framework)",
            file=sys.stderr,
        )

    if not true_misses:
        print(
            green(
                f"  [{label}] All {len(collectable_names)} collectable names covered.\n"
            )
        )
        return 0

    print(
        yellow(
            f"  [{label}] {len(true_misses)} collectable name(s) not "
            f"covered by any config:\n"
        )
    )
    for name in true_misses:
        print(f"    {yellow('MISS')}  {name}")
    print()
    return len(true_misses)


def _is_covered(base_name: str, patterns: list[PatternEntry]) -> bool:
    """
    Return True if at least one pattern covers this base name.
    """
    # Rule 1: direct fullmatch
    if any(pe.matches(base_name) for pe in patterns):
        return True
    # Rule 2: pattern targets a param-key variant (prefix match)
    if any(pe.pattern.startswith(base_name + "_") for pe in patterns):
        return True
    return False


# -----------------------
# CHECK 3: Dead patterns
# -----------------------


def check_dead_patterns(
    collectable_names: list[str],
    patterns: list[PatternEntry],
    label: str,
) -> int:
    """
    Find config patterns that match zero collectable base names (stale/typo/renamed).

    A pattern is considered live if:
      (a) It fullmatches at least one collectable name, OR
      (b) Its literal text starts with a collectable base name + "_",
          meaning it intentionally targets a param-key variant of that base.
          (e.g. "test_mean_3d_dim0" is live because "test_mean" is collectable)

    Returns
    -------
    int
        Number of dead patterns found. These are warnings, not hard failures.
    """
    dead = [pe for pe in patterns if not _is_live(pe, collectable_names)]

    if not dead:
        print(green(f"  [{label}] All patterns match at least one collectable name.\n"))
        return 0

    print(
        yellow(
            f"  [{label}] {len(dead)} pattern(s) match no collectable "
            f"name (dead/typo):\n"
        )
    )
    for pe in dead:
        print(f"    {yellow('DEAD')}  {cyan(pe.raw)}")
        print(f"          config: {pe.config_file.name}")
    print()
    return len(dead)


def _is_live(pe: PatternEntry, collectable_names: list[str]) -> bool:
    """
    Return True if this pattern is live (targets at least one collectable name).
    """
    # Rule (a): fullmatch
    if any(pe.matches(n) for n in collectable_names):
        return True
    # Rule (b): param-key variant of a known collectable base
    if any(pe.pattern.startswith(n + "_") for n in collectable_names):
        return True
    return False

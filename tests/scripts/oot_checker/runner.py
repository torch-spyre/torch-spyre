"""
runner.py
---------
Orchestrates the three checks across all loaded config files.
Orchestrates together loader, parser, checks, display, and discovery modules.
"""

from __future__ import annotations

from pathlib import Path

from .checks import check_duplicates, check_missing, check_dead_patterns
from .discovery import find_test_files
from .display import bold, green, red, yellow, heuristic_base_names, warn
from .loader import load_all_patterns
from .parser import parse_test_file


def run(
    config_files: list[Path],
    test_file: Path | None,
    test_root: Path,
    fail_on_problems: bool,
) -> int:
    """
    Run all checks and print a summary.

    Parameters
    ----------
    config_files : list[Path]
        YAML config files to check.
    test_file : Path | None
        Explicit reference Python test file. When given, only this file is
        used for CHECK 2 + 3. When None, test files are auto-discovered
        under test_root using the basenames declared in the configs.
    test_root : Path
        Root directory to search for test files when test_file is None.
        Defaults to the current working directory.
    fail_on_problems : bool
        Exit code 1 when duplicates or missing tests are found.
        Dead patterns are warnings and do not affect exit code.

    Returns
    -------
    int
        0 if clean (or only warnings), 1 if problems and fail_on_problems.
    """
    all_patterns = load_all_patterns(config_files)

    if not any(all_patterns.values()):
        print("[WARN] No test patterns found in any config file.")
        return 0

    # Build the reference map: basename → (collectable_names, helper_only)
    ref_collectable: dict[str, list[str]] = {}
    ref_helper_only: dict[str, set[str]] = {}

    if test_file:
        # Explicit file supplied -- use only that one
        _load_reference(test_file, ref_collectable, ref_helper_only)
    else:
        # Auto-discover: search test_root for every basename the configs mention
        basenames = list(all_patterns.keys())
        discovered = find_test_files(basenames, test_root)

        if not discovered:
            warn(
                f"No reference test files found under '{test_root}' for "
                f"basenames: {basenames}. "
                f"CHECK 2 (missing) and CHECK 3 (dead patterns) will be skipped. "
                f"Pass --test-root <repo-root> or --test-file <path> to enable them."
            )
        else:
            for basename, path in sorted(discovered.items()):
                _load_reference(path, ref_collectable, ref_helper_only)
            # Report any basenames we could not find
            missing_files = set(basenames) - set(discovered)
            if missing_files:
                for bf in sorted(missing_files):
                    warn(
                        f"Could not find '{bf}' under '{test_root}' — "
                        f"CHECK 2 + 3 skipped for this file."
                    )

    total_hard_problems = 0
    total_warnings = 0

    # Determine which basenames to process:
    #   - explicit --test-file  -> only the basename of that file
    #   - auto-discovery        -> every basename that was successfully resolved
    #   - neither               -> all basenames (CHECK 1 heuristic only)
    if test_file:
        active_basenames = {test_file.name}
    elif ref_collectable:
        active_basenames = set(ref_collectable.keys())
    else:
        active_basenames = set(all_patterns.keys())

    for basename, patterns in sorted(all_patterns.items()):
        if basename not in active_basenames:
            continue
        _print_section_header(basename, patterns)

        if basename in ref_collectable:
            collectable = ref_collectable[basename]
            helper_only = ref_helper_only[basename]
            source_label = f"{len(collectable)} collectable names from {basename}"
        else:
            collectable = heuristic_base_names(patterns)
            helper_only = set()
            source_label = (
                f"{len(collectable)} heuristic names (auto-discovery found no match)"
            )

        print(f"  Base names: {source_label}\n")

        # CHECK 1: Duplicates (runs always)
        print(bold("  CHECK 1: Duplicates"))
        all_names_for_dup = (
            sorted(set(collectable) | ref_helper_only.get(basename, set()))
            if basename in ref_collectable
            else collectable
        )
        total_hard_problems += check_duplicates(all_names_for_dup, patterns, basename)

        # CHECK 2 + 3: only when we have real base names from a parsed file
        if basename in ref_collectable:
            print(bold("  CHECK 2: Missing tests"))
            total_hard_problems += check_missing(
                collectable, helper_only, patterns, basename
            )

            print(bold("  CHECK 3: Dead patterns (warnings only)"))
            total_warnings += check_dead_patterns(collectable, patterns, basename)

    _print_summary(total_hard_problems, total_warnings)

    if fail_on_problems and total_hard_problems > 0:
        return 1
    return 0


# --------------------
# Internal helpers
# --------------------


def _load_reference(
    path: Path,
    ref_collectable: dict[str, list[str]],
    ref_helper_only: dict[str, set[str]],
) -> None:
    """Parse one test file and populate the reference dicts."""
    if not path.exists():
        warn(f"Reference file not found: {path} — skipping.")
        return
    collectable, helper_only = parse_test_file(path)
    ref_collectable[path.name] = collectable
    ref_helper_only[path.name] = helper_only
    print(f"Reference: {path}")
    print(f"  {len(collectable)} collectable name(s)  (PARAMS primaries + standalones)")
    print(f"  {len(helper_only)} helper-only method(s) (excluded from MISSING check)\n")


def _print_section_header(basename: str, patterns) -> None:
    n_configs = len({pe.config_file for pe in patterns})
    print(bold("━" * 68))
    print(
        bold(
            f"Test file: {basename}  "
            f"({len(patterns)} pattern(s) across {n_configs} config(s))"
        )
    )
    print(bold("━" * 68))


def _print_summary(hard: int, warnings: int) -> None:
    print(bold("━" * 68))
    parts = []
    parts.append(
        green("0 duplicates/missing")
        if hard == 0
        else red(f"{hard} duplicate(s)/missing test(s)")
    )
    parts.append(
        green("0 dead patterns")
        if warnings == 0
        else yellow(f"{warnings} dead pattern warning(s)")
    )
    print(bold("RESULT: ") + "  |  ".join(parts))
    print(bold("━" * 68))

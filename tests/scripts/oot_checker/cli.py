"""
cli.py
------
Command-line interface for the OOT config checker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import run


_DESCRIPTION = "Check OOT configs for duplicate and missing test coverage."

_EPILOG = """
examples
--------
  # Full auto-discovery (no --test-file needed finds test files automatically):
  python scripts/check_oot_configs.py \\
      --config-dir configs/torch_spyre_tests/inductor/

  # Explicit test file (faster, unambiguous):
  python scripts/check_oot_configs.py \\
      --config-dir configs/torch_spyre_tests/inductor/ \\
      --test-file  inductor/test_inductor_ops.py

  # CI gate — exit 1 on any duplicate or missing test:
  python scripts/check_oot_configs.py \\
      --config-dir configs/torch_spyre_tests/inductor/ \\
      --fail-on-problems
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="check_oot_configs",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )

    # -- Config source (one of --config-dir or --configs required) --
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--config-dir",
        type=Path,
        metavar="DIR",
        help="Directory to recursively scan for *.yaml / *.yml config files.",
    )
    src.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Explicit list of YAML config files to check.",
    )

    # ------- Test file resolution -------
    tf = p.add_mutually_exclusive_group()
    tf.add_argument(
        "--test-file",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Explicit reference Python test file "
            "(e.g. tests/inductor/test_inductor_ops.py). "
            "When given, only this file is used for CHECK 2 + CHECK 3. "
            "Mutually exclusive with --test-root."
        ),
    )
    tf.add_argument(
        "--test-root",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Root directory to search for test files when --test-file is not "
            "given (default: current working directory). The tool extracts the "
            "test file basename from each config's 'path:' field and searches "
            "recursively under this directory. "
            "Mutually exclusive with --test-file."
        ),
    )

    # ------- Behaviour flags -------
    p.add_argument(
        "--fail-on-problems",
        action="store_true",
        default=False,
        help=(
            "Exit with code 1 if any duplicates or missing tests are found. "
            "Dead patterns are warnings and do not affect the exit code."
        ),
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve config file paths
    if args.config_dir:
        paths = sorted(args.config_dir.rglob("*.yaml")) + sorted(
            args.config_dir.rglob("*.yml")
        )
        if not paths:
            sys.exit(f"No YAML files found under {args.config_dir}")
        print(f"Scanning {len(paths)} config file(s) under {args.config_dir}\n")
    else:
        paths = list(args.configs)
        for cp in paths:
            if not cp.exists():
                sys.exit(f"File not found: {cp}")
        print(f"Checking {len(paths)} explicit config file(s)\n")

    # Resolve test-root: explicit flag or cwd
    test_root = args.test_root or Path.cwd()

    sys.exit(
        run(
            config_files=paths,
            test_file=args.test_file,
            test_root=test_root,
            fail_on_problems=args.fail_on_problems,
        )
    )


if __name__ == "__main__":
    main()

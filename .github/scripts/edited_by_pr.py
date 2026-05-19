#!/usr/bin/env python3
"""EditedByPR heuristic for torch-spyre.

This script reads a list of changed files and prints the predicted
modified and related tests. It is intentionally lightweight so it can
run inside GitHub Actions without extra dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EditedByPR test prioritization for a PR.")
    parser.add_argument(
        "--changed-files",
        type=Path,
        help="Path to a newline-separated file containing changed files.")
    return parser.parse_args()


def read_changed_files(changed_files: Path | None) -> list[str]:
    if changed_files is None:
        data = sys.stdin.read()
    else:
        data = changed_files.read_text()
    return [line.strip() for line in data.splitlines() if line.strip()]


def normalize_test_name(path: str) -> str:
    p = Path(path)
    if p.suffix == ".py":
        p = p.with_suffix("")
    if p.parts and p.parts[0] == "tests":
        p = Path(*p.parts[1:])
    return " ".join(p.parts)


def find_test_files(repo_root: Path) -> list[Path]:
    return sorted(repo_root.glob("tests/**/*.py"))


def main() -> int:
    args = parse_args()
    changed_files = read_changed_files(args.changed_files)
    repo_root = Path(__file__).resolve().parents[2]
    test_files = find_test_files(repo_root)

    direct_tests: set[str] = set()
    for changed in changed_files:
        if changed.startswith("tests/") and changed.endswith(".py"):
            direct_tests.add(normalize_test_name(changed))

    additional_tests: set[str] = set()
    for changed in changed_files:
        if changed.startswith("tests/configs/") and changed.endswith(".yaml"):
            partial = Path(changed).stem
            if partial.endswith("_config"):
                partial = partial[: -len("_config")]
            for test_path in test_files:
                test_name = normalize_test_name(str(test_path.relative_to(repo_root)))
                if partial in Path(test_path).stem:
                    additional_tests.add(test_name)
        if changed.startswith("torch_spyre/inductor/"):
            for test_path in test_files:
                if len(test_path.parts) >= 2 and test_path.parts[1] == "inductor":
                    additional_tests.add(normalize_test_name(str(test_path.relative_to(repo_root))))
        if changed.startswith("torch_spyre/tensor/"):
            for test_path in test_files:
                if len(test_path.parts) >= 2 and test_path.parts[1] == "tensor":
                    additional_tests.add(normalize_test_name(str(test_path.relative_to(repo_root))))
        if changed.startswith("torch_spyre/ops/"):
            for test_path in test_files:
                if Path(test_path).stem == "test_modules":
                    additional_tests.add(normalize_test_name(str(test_path.relative_to(repo_root))))

    indirect_tests = sorted(additional_tests - direct_tests)
    direct_tests_sorted = sorted(direct_tests)

    print("EditedByPR heuristic results")
    print("============================")
    print("\nChanged files:")
    if changed_files:
        for changed in changed_files:
            print(f" - {changed}")
    else:
        print(" (none)")

    print("\nPriority decisions:")
    if direct_tests_sorted:
        print("  Directly edited tests (score=1):")
        for name in direct_tests_sorted:
            print(f"   - {name}")
    else:
        print("  No direct test files were edited.")

    if indirect_tests:
        print("\n  Indirectly related tests via additional mappings (score=1):")
        for name in indirect_tests:
            print(f"   - {name}")
    else:
        print("\n  No indirect mappings matched changed files.")

    print("\n  All other tests are implicitly score=0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Adopted from https://github.com/tdoublep/vllm/blob/full_cg_mamba/tools/enforce_regex_import.py
from __future__ import annotations

import subprocess
from pathlib import Path

import regex as re

FORBIDDEN_PATTERNS = re.compile(r"^\s*(?:import\s+re(?:$|\s|,)|from\s+re\s+import)")
ALLOWED_PATTERNS = [
    re.compile(r"^\s*import\s+regex\s+as\s+re\s*$"),
    re.compile(r"^\s*import\s+regex\s*$"),
]


def get_staged_python_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        return [f for f in files if f.endswith(".py")]
    except subprocess.CalledProcessError:
        return []


def is_forbidden_import(line: str) -> bool:
    line = line.strip()
    return bool(
        FORBIDDEN_PATTERNS.match(line)
        and not any(pattern.match(line) for pattern in ALLOWED_PATTERNS)
    )


def check_file(filepath: str) -> list[tuple[int, str]]:
    violations = []
    try:
        with open(filepath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if is_forbidden_import(line):
                    violations.append((line_num, line.strip()))
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def main() -> int:
    files = get_staged_python_files()
    if not files:
        return 0

    total_violations = 0

    # Files that legitimately use stdlib `re`. Keep this list short; only
    # add a path here when the file must be portable to environments where
    # the third-party `regex` module isn't available.
    skip_files = {
        "setup.py",
        # Vendored verbatim from spyre-runtimes/scripts/get_torch_minor.py;
        # used by the runner-image build pipeline AND by torch-spyre CI to
        # derive the pytorch release branch from pyproject.toml. Must run
        # without third-party deps to stay portable across both call sites.
        "tools/get_torch_minor.py",
    }

    for filepath in files:
        if not Path(filepath).exists():
            continue

        if filepath in skip_files:
            continue

        violations = check_file(filepath)
        if violations:
            print(f"\n❌ {filepath}:")
            for line_num, line in violations:
                print(f"  Line {line_num}: {line}")
                total_violations += 1

    if total_violations > 0:
        print(f"\n💡 Found {total_violations} violation(s).")
        print("❌ Please replace 'import re' with 'import regex as re'")
        print("   Also replace 'from re import ...' with 'from regex import ...'")  # noqa: E501
        print("✅ Allowed imports:")
        print("   - import regex as re")
        print("   - import regex")  # noqa: E501
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

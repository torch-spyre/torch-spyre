"""
Reads the upstream `hw_classification` class attribute added by pytorch/pytorch
PR 190173 (commit 8e9663d1, 14 Oct 2026) and reports where an OOT config
disagrees with what upstream declares about a test class.

Upstream values and what they mean for Spyre (PrivateUse1):

  GENERIC      runs once, no device parametrization -- never yields a Spyre variant
  ACCELERATOR  parametrized via instantiate_device_type_tests -> Spyre variant exists
  CPU/CUDA/MPS/XPU
               device-specific class -- no Spyre variant is ever generated

The attribute is advisory upstream: its linter emits LintSeverity.ADVICE only,
and 920 of the test files are allowlisted out of the check entirely. A missing
attribute therefore means "upstream has not tagged this yet", NOT "this class
is wrong". Every finding here is a warning; nothing in this module feeds the
--fail-on-problems exit code.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .display import red, yellow, green, cyan

HW_ATTR = "hw_classification"

# A class carrying one of these is device-specific. CPU is runnable: such a
# class is NOT device-parametrized, it just runs on CPU, which is what the
# `edits.functions.cpu_move` configs rely on (verified: TestAutocastCPU is
# tagged CPU, is absent from every instantiate_device_type_tests call in
# test_autocast.py, and its 5 configured tests pass today as mandatory_success).
# Only a foreign accelerator can never run here.
FOREIGN_ACCELERATOR = frozenset({"CUDA", "MPS", "XPU"})

# Modes that assert the test actually runs on Spyre. skip/xfail* do not.
RUNS_ON_SPYRE = frozenset({"mandatory_success"})


def read_hw_classifications(test_file: Path) -> dict[str, str | None]:
    """
    Return {class_name: classification_or_None} for every test class in a file.

    A test class is any class defining a `test*` method -- the same detection
    rule the upstream linter uses, since `TestCase` inheritance cannot be
    resolved reliably from the AST alone.
    """
    try:
        tree = ast.parse(test_file.read_text())
    except (SyntaxError, OSError):
        # Unparseable or unreadable: the caller degrades to "unknown", never fails.
        return {}

    result: dict[str, str | None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            isinstance(item, ast.FunctionDef) and item.name.startswith("test")
            for item in node.body
        ):
            continue
        result[node.name] = _declared_classification(node)
    return result


def _declared_classification(node: ast.ClassDef) -> str | None:
    """Return the declared classification member name, or None if absent."""
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if isinstance(target, ast.Name) and target.id == HW_ATTR:
                # Accept both `HardwareClassification.GENERIC` and a bare string.
                value = item.value
                if isinstance(value, ast.Attribute):
                    return value.attr
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                return None
    return None


class ConfigClassEntry:
    """
    One (class_name, method_name, mode) triple from a config.

    PatternEntry deliberately strips the "ClassName::" prefix and the loader
    drops `mode`, because CHECK 1-3 match on base method name only. This check
    needs both, so it reads the YAML itself rather than widening PatternEntry
    and perturbing the existing checks.
    """

    __slots__ = ("config_file", "test_file_basename", "class_name", "method", "mode")

    def __init__(
        self,
        config_file: Path,
        test_file_basename: str,
        class_name: str | None,
        method: str,
        mode: str,
    ):
        self.config_file = config_file
        self.test_file_basename = test_file_basename
        self.class_name = class_name
        self.method = method
        self.mode = mode

    def __repr__(self) -> str:
        cls = self.class_name or "<module-level>"
        return f"{cls}::{self.method} [{self.mode}] ({self.config_file.name})"


def load_class_entries(config_files: list[Path]) -> dict[str, list[ConfigClassEntry]]:
    """Group ConfigClassEntry objects by target test-file basename."""
    import yaml
    from collections import defaultdict

    result: dict[str, list[ConfigClassEntry]] = defaultdict(list)
    for cf in config_files:
        try:
            cfg = yaml.safe_load(cf.read_text()) or {}
        except Exception:
            continue
        for file_entry in cfg.get("test_suite_config", {}).get("files", []):
            basename = Path(file_entry.get("path", "")).name
            for block in file_entry.get("tests", []):
                mode = block.get("mode", "mandatory_success")
                for raw in block.get("names", []):
                    parts = str(raw).split("::")
                    class_name = parts[0] if len(parts) == 2 else None
                    method = parts[-1]
                    result[basename].append(
                        ConfigClassEntry(cf, basename, class_name, method, mode)
                    )
    return result


# ---------------------------------------------
# CHECK 4: hw_classification agreement (warnings)
# ---------------------------------------------


def check_hw_classification(
    entries: list[ConfigClassEntry],
    declared: dict[str, str | None],
    basename: str,
) -> int:
    """
    Cross-check config expectations against upstream's declared classification.

    Reports three kinds of disagreement:

      UNRUNNABLE  config expects a pass (mode: mandatory_success) on a class
                  tagged for a foreign accelerator (CUDA/MPS/XPU). That
                  hardware is absent here, so the entry can never pass.
                  CPU is deliberately NOT flagged: a CPU class is not
                  device-parametrized and runs on CPU normally.
      UNTAGGED    the class is configured but carries no hw_classification.
                  Upstream has not migrated the file; informational only.
      UNCOVERED   the class is ACCELERATOR (a Spyre variant exists) but no
                  config entry mentions it -- a coverage gap.

    Returns the number of warnings printed. Never contributes to the
    --fail-on-problems exit code: the upstream attribute is advisory and
    heavily allowlisted, so gating on it would redden CI on upstream's
    schedule rather than on a real regression here.
    """
    if not declared:
        print(f"  [hw] No test classes parsed from {basename} — check skipped.\n")
        return 0

    configured = {e.class_name for e in entries if e.class_name}

    unrunnable: list[tuple[ConfigClassEntry, str]] = []
    untagged: set[str] = set()

    for e in entries:
        if not e.class_name or e.class_name not in declared:
            continue
        hw = declared[e.class_name]
        if hw is None:
            untagged.add(e.class_name)
        elif hw in FOREIGN_ACCELERATOR and e.mode in RUNS_ON_SPYRE:
            unrunnable.append((e, hw))

    uncovered = sorted(
        name
        for name, hw in declared.items()
        if hw == "ACCELERATOR" and name not in configured
    )

    total = len(unrunnable) + len(untagged) + len(uncovered)
    if total == 0:
        print(
            green(
                f"  [hw] {len(declared)} class(es) in {basename} agree with "
                f"their hw_classification.\n"
            )
        )
        return 0

    if unrunnable:
        print(
            red(
                f"  [hw] {len(unrunnable)} entr(ies) expect a pass on a "
                f"foreign-accelerator class:\n"
            )
        )
        for e, hw in unrunnable:
            print(f"    {red('UNRUNNABLE')}  {e.class_name}::{e.method}")
            print(
                f"                hw_classification = {hw}; that hardware is not "
                f"present on a Spyre runner"
            )
            print(f"                {e.config_file}  (mode: {e.mode})")
        print()

    if untagged:
        print(
            yellow(
                f"  [hw] {len(untagged)} configured class(es) carry no "
                f"hw_classification upstream:\n"
            )
        )
        for name in sorted(untagged):
            print(f"    {yellow('UNTAGGED')}  {name}")
        print(
            "          upstream has not migrated this file "
            "(the attribute is advisory + allowlisted)\n"
        )

    if uncovered:
        print(
            cyan(
                f"  [hw] {len(uncovered)} ACCELERATOR class(es) have no config entry:\n"
            )
        )
        for name in uncovered:
            print(f"    {cyan('UNCOVERED')}  {name}")
        print("          a Spyre variant is generated but nothing configures it\n")

    return total

"""
Unit tests for CHECK 4 (hw_classification agreement).

Both polarities are covered for every rule: a case that must fire and a case
that must stay silent. The CPU-vs-foreign-accelerator distinction is the one
that matters most -- an earlier draft of this check flagged every
`TestAutocastCPU` entry as unrunnable, which was wrong (see test_cpu_*).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from oot_checker.hw_classification import (  # noqa: E402
    ConfigClassEntry,
    check_hw_classification,
    load_class_entries,
    read_hw_classifications,
)


def _entry(cls, method="test_x", mode="mandatory_success"):
    return ConfigClassEntry(Path("cfg.yaml"), "t.py", cls, method, mode)


def _write(tmp_path: Path, body: str) -> Path:
    f = tmp_path / "t.py"
    f.write_text(textwrap.dedent(body))
    return f


# ----------------------------
# read_hw_classifications
# ----------------------------


def test_reads_attribute_form(tmp_path):
    f = _write(
        tmp_path,
        """
        class TestA:
            hw_classification = HardwareClassification.ACCELERATOR
            def test_a(self, device): pass
        """,
    )
    assert read_hw_classifications(f) == {"TestA": "ACCELERATOR"}


def test_reads_bare_string_form(tmp_path):
    f = _write(
        tmp_path,
        """
        class TestA:
            hw_classification = "GENERIC"
            def test_a(self): pass
        """,
    )
    assert read_hw_classifications(f) == {"TestA": "GENERIC"}


def test_untagged_class_is_none(tmp_path):
    f = _write(
        tmp_path,
        """
        class TestA:
            def test_a(self): pass
        """,
    )
    assert read_hw_classifications(f) == {"TestA": None}


def test_non_test_class_is_ignored(tmp_path):
    """A class with no test* method is not a test class (upstream's own rule)."""
    f = _write(
        tmp_path,
        """
        class Helper:
            hw_classification = HardwareClassification.CPU
            def helper(self): pass
        """,
    )
    assert read_hw_classifications(f) == {}


def test_unparseable_file_returns_empty(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("class Broken(:\n")
    assert read_hw_classifications(f) == {}


def test_missing_file_returns_empty(tmp_path):
    assert read_hw_classifications(tmp_path / "nope.py") == {}


# ----------------------------
# UNRUNNABLE: foreign accelerator only
# ----------------------------


@pytest.mark.parametrize("hw", ["CUDA", "MPS", "XPU"])
def test_foreign_accelerator_mandatory_success_fires(capsys, hw):
    n = check_hw_classification([_entry("TestF")], {"TestF": hw}, "t.py")
    assert n == 1
    assert "UNRUNNABLE" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["skip", "xfail", "xfail_strict"])
def test_foreign_accelerator_non_running_mode_silent(capsys, mode):
    """skip/xfail do not assert a pass, so there is nothing to flag."""
    n = check_hw_classification([_entry("TestF", mode=mode)], {"TestF": "CUDA"}, "t.py")
    assert n == 0
    assert "UNRUNNABLE" not in capsys.readouterr().out


def test_cpu_mandatory_success_is_not_flagged(capsys):
    """
    The TestAutocastCPU case: a CPU class is NOT device-parametrized, it runs
    on CPU, and our configs rely on that via edits.functions.cpu_move. Flagging
    it was the bug in the first draft of this check.
    """
    n = check_hw_classification(
        [_entry("TestAutocastCPU")], {"TestAutocastCPU": "CPU"}, "t.py"
    )
    assert n == 0
    assert "UNRUNNABLE" not in capsys.readouterr().out


# ----------------------------
# UNTAGGED
# ----------------------------


def test_untagged_configured_class_warns(capsys):
    n = check_hw_classification([_entry("TestTorch")], {"TestTorch": None}, "t.py")
    assert n == 1
    assert "UNTAGGED" in capsys.readouterr().out


def test_tagged_configured_class_not_untagged(capsys):
    n = check_hw_classification([_entry("TestA")], {"TestA": "ACCELERATOR"}, "t.py")
    assert n == 0


# ----------------------------
# UNCOVERED
# ----------------------------


def test_accelerator_without_config_is_uncovered(capsys):
    n = check_hw_classification([], {"TestAccel": "ACCELERATOR"}, "t.py")
    assert n == 1
    assert "UNCOVERED" in capsys.readouterr().out


def test_generic_without_config_is_not_uncovered(capsys):
    """GENERIC yields no Spyre variant, so absent config is not a gap."""
    n = check_hw_classification([], {"TestGen": "GENERIC"}, "t.py")
    assert n == 0


def test_covered_accelerator_is_silent(capsys):
    n = check_hw_classification(
        [_entry("TestAccel")], {"TestAccel": "ACCELERATOR"}, "t.py"
    )
    assert n == 0


def test_module_level_entry_does_not_crash(capsys):
    """A plain `method_name` entry has class_name None and must be skipped."""
    n = check_hw_classification(
        [_entry(None, method="test_plain")], {"TestA": "GENERIC"}, "t.py"
    )
    assert n == 0


def test_empty_declared_skips_check(capsys):
    n = check_hw_classification([_entry("TestA")], {}, "t.py")
    assert n == 0
    assert "check skipped" in capsys.readouterr().out


# ----------------------------
# load_class_entries
# ----------------------------


def test_load_class_entries_keeps_class_and_mode(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            test_suite_config:
              files:
                - path: ${TORCH_ROOT}/test/test_foo.py
                  tests:
                    - names:
                        - TestA::test_one
                        - test_module_level
                      mode: skip
                    - names:
                        - TestB::test_two
            """
        )
    )
    got = load_class_entries([cfg])
    assert set(got) == {"test_foo.py"}
    byname = {(e.class_name, e.method): e.mode for e in got["test_foo.py"]}
    assert byname[("TestA", "test_one")] == "skip"
    assert byname[(None, "test_module_level")] == "skip"
    # mode defaults to mandatory_success when the block omits it
    assert byname[("TestB", "test_two")] == "mandatory_success"

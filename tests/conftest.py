# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import yaml
import pytest


import shared_config


def _get_case_marks(case: dict) -> set[str]:
    """
    Support either:
      marks: paddedtensor
      marks: [paddedtensor, fpoperation]
    """
    marks = set()
    m = case.get("marks")
    if isinstance(m, str) and m.strip():
        marks.add(m.strip())

    ms = case.get("marks")
    if isinstance(ms, (list, tuple)):
        for x in ms:
            if isinstance(x, str) and x.strip():
                marks.add(x.strip())

    return marks


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """

    # avoid circular imports when using xdist
    import torch  # noqa: F401

    os.environ.setdefault("DTLOG_LEVEL", "error")
    os.environ.setdefault("DT_DEEPRT_VERBOSE", "-1")

    cfg = session.config
    root = cfg.rootpath

    selected = set(cfg.getoption("--model") or [])

    if cfg.getoption("--list-models"):
        models = sorted({m for (m, _, __, ___, _case) in _iter_yaml_cases(root)})
        for m in models:
            print(m)
        pytest.exit("listed models", returncode=0)

    if cfg.getoption("--list-cases"):
        for model, name, op, p, _case in _iter_yaml_cases(root):
            if selected and model not in selected:
                continue
            print(f"{model}::{name}::{op}  ({p})")
        pytest.exit("listed cases", returncode=0)

    opt = cfg.getoption("--list-cases-by-mark")
    if opt is not None:
        if opt == "__USE_PYTEST_M__":
            # This is the *effective* -m expression after addopts + CLI parsing.
            expr = (cfg.option.markexpr or "").strip()
            # If no -m anywhere, treat as "select all"
            if not expr:
                expr = "True"
        else:
            expr = opt.strip()
        from _pytest.mark.expression import Expression

        compiled = Expression.compile(expr)

        show_excluded = cfg.getoption("--show-excluded")
        chosen = []
        excluded = []

        def case_selected(case: dict) -> bool:
            marks = _get_case_marks(case)  # set[str]
            return compiled.evaluate(lambda m: m in marks)

        for model, name, op, p, case in _iter_yaml_cases(root):
            if selected and model not in selected:
                continue

            rec = f"{model}::{name}::{op}  ({p})"
            if case_selected(case):
                chosen.append(rec)
            else:
                excluded.append(rec)

        if show_excluded:
            for r in excluded:
                print(r)
            pytest.exit(f"listed excluded cases by mark (NOT {expr})", returncode=0)
        else:
            for r in chosen:
                print(r)
            pytest.exit(f"listed selected cases by mark ({expr})", returncode=0)


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="append",
        default=[],
        help="Run only these models (repeatable). Example: --model granite3-speech",
    )
    parser.addoption(
        "--dedupe",
        dest="dedupe",
        action="store_true",
        default=True,  # default ON
        help="Skip duplicate op+input signatures across models (runtime).",
    )
    parser.addoption(
        "--no-dedupe",
        action="store_false",
        dest="dedupe",
        help="Disable deduplication.",
    )

    # NEW: inventory modes
    parser.addoption(
        "--list-models",
        action="store_true",
        default=False,
        help="List models found in tests/_inductor/models/*.yaml and exit.",
    )
    parser.addoption(
        "--list-cases",
        action="store_true",
        default=False,
        help="List cases found in tests/_inductor/models/*.yaml and exit. Use --model to filter.",
    )
    parser.addoption(
        "--compile-backend",
        action="store",
        default=os.environ.get("TEST_COMPILE_BACKEND", "inductor"),
        help="If set, run test via torch.compile(..., backend=...).",
    )
    group = parser.getgroup("yaml-cases")
    group.addoption(
        "--list-cases-by-mark",
        action="store",
        const="__USE_PYTEST_M__",
        default=None,
        nargs="?",
        metavar="EXPR",
        help=(
            "List YAML test cases whose mark(s) match a pytest -m style expression. "
            "Examples: paddedtensor | 'paddedtensor and not fpoperation'"
            "If EXPR is omitted, uses the effective pytest -m expression (including pytest.ini addopts)."
        ),
    )
    group.addoption(
        "--show-excluded",
        action="store_true",
        default=False,
        help="With --list-cases-by-mark, list cases excluded by the mark expression (i.e., NOT matching).",
    )
    group.addoption(
        "--show-skipped",
        action="store_true",
        default=False,
        help="List cases skipped by model filtering or duplications",
    )
    parser.addoption(
        "--test-names",
        action="append",
        default=[],
        help="Run only tests matching these test names",
    )


def _models_dir(rootpath: Path) -> Path:
    return rootpath / "tests" / "_inductor" / "models"


def load_yaml_or_fail(path: Path) -> dict:
    text = path.read_text()
    try:
        data = yaml.safe_load(text)
        if data is None:
            raise pytest.UsageError(f"{path}: YAML is empty")
        return data
    except yaml.YAMLError as e:
        # Build a nice error message with file + location + snippet
        msg = [f"Invalid YAML in {path}"]

        mark = getattr(e, "problem_mark", None)
        if mark is not None:
            # PyYAML lines are 0-based internally; show 1-based to humans
            line = mark.line + 1
            col = mark.column + 1
            msg.append(f"Location: line {line}, column {col}")

            lines = text.splitlines()
            start = max(0, mark.line - 2)
            end = min(len(lines), mark.line + 3)

            msg.append("Context:")
            for i in range(start, end):
                prefix = ">>" if i == mark.line else "  "
                msg.append(f"{prefix} {i + 1:4d}: {lines[i]}")
                if i == mark.line:
                    msg.append(f"     {' ' * (col - 1)}^")

        # Include the underlying YAML error message too
        msg.append(f"YAML error: {e}")

        # Fail pytest configuration cleanly (instead of INTERNALERROR)
        raise pytest.UsageError("\n".join(msg)) from e


def _iter_yaml_cases(rootpath: Path):
    """
    Yields tuples: (model, case_name, op_name, yaml_path, case)
    Supports either:
      - per-case 'op'
      - or top-level 'op' applied to cases that don't specify 'op'
    """
    for p in sorted(_models_dir(rootpath).glob("*.yaml")):
        if p.name.endswith("template.yaml"):  # skip template.yaml file
            continue
        spec = load_yaml_or_fail(p)
        model = spec.get("model", p.stem)
        top_op = spec.get("op", None)
        for case in spec.get("cases", []):
            op = case.get("op", top_op)
            name = case.get("name", op or "<unnamed>")
            yield model, name, op, p, case


@pytest.fixture(scope="session")
def selected_models(pytestconfig):
    return set(pytestconfig.getoption("--model") or [])


@pytest.fixture(scope="session")
def dedupe_enabled(pytestconfig):
    return bool(pytestconfig.getoption("dedupe"))


@pytest.fixture(scope="session")
def test_device_str(pytestconfig):
    return "spyre"


@pytest.fixture(scope="session")
def seen_case_keys():
    # track which case has been run to avoid rerun again
    return set()


@pytest.fixture(scope="session")
def compile_backend(pytestconfig):
    s = str(pytestconfig.getoption("--compile-backend") or "").strip()
    return s or None


def pytest_configure(config):
    shared_config._PYTEST_CONFIG = config
    # auto-register model_<name> markers based on YAML files
    mdir = config.rootpath / "tests" / "_inductor" / "models"
    for p in mdir.glob("*.yaml"):
        spec = load_yaml_or_fail(p)
        model = spec.get("model", p.stem)
        mark = "model_" + "".join(
            ch if ch.isalnum() or ch == "_" else "_" for ch in model
        )
        config.addinivalue_line(
            "markers", f"{mark}: auto-generated mark for model {model}"
        )


def pytest_collection_modifyitems(config, items):
    selected_models = config.getoption("--model") or []
    if not selected_models:
        return  # normal behavior

    # Keep only model-yaml runner tests
    keep = []
    deselect = []

    for item in items:
        # item.nodeid includes the file path, e.g. "tests/_inductor/test_model_ops.py::test_model_ops[...]"
        if "tests/_inductor/test_model_ops.py::" in item.nodeid:
            keep.append(item)
        else:
            deselect.append(item)

    if deselect:
        config.hook.pytest_deselected(items=deselect)
        items[:] = keep


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--show-skipped"):
        return
    skipped = terminalreporter.stats.get("skipped", [])
    if not skipped:
        return

    terminalreporter.section("Skipped tests (full list)")
    for rep in skipped:
        # terminalreporter.write_line(rep)
        terminalreporter.write_line(rep.nodeid)

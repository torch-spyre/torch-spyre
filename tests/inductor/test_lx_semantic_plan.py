# Copyright 2026 The Torch-Spyre Authors.
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

"""Host checks for strict semantic-plan capture, not native plan qualification."""

from copy import deepcopy
from dataclasses import dataclass

import pytest
import sympy

from tools.lx_planning.lx_semantic_plan import compare_plans, freeze, validate_contract


def _contract():
    return {
        "fixture": "existing-prefix",
        "input_sha256": "0" * 64,
        "hints_sha256": "1" * 64,
        "matched_upstream": "same-upstream",
        "settings": {
            "lx_extended_ownership": False,
            "sencores": 32,
            "num_cpus": 1,
            "layout_solver": "cpsat",
            "co_optimizing_lx_planning": True,
            "lx_planner_relayout": True,
            "lx_same_tile_group_relayout": False,
            "ktir_emitter": False,
        },
    }


def test_mapping_and_sequence_order_remain_significant():
    assert freeze({"b": 2, "a": 1}) != freeze({"a": 1, "b": 2})
    assert freeze([1, 2]) != freeze([2, 1])
    assert freeze({1, 2}) == freeze({2, 1})


def test_unknown_semantic_fields_cannot_silently_disappear():
    with pytest.raises(TypeError, match="unsupported semantic field"):
        freeze(object())


def test_all_record_fields_and_symbol_assumptions_are_retained():
    @dataclass
    class Example:
        owners: object
        stride: tuple
        offset: int

    value = Example(sympy.Symbol("x", positive=True), (128, 1), 0)
    assert freeze(value) != freeze(Example(sympy.Symbol("x"), (128, 1), 0))
    assert freeze(value) != freeze(Example(value.owners, (64, 1), 0))
    assert freeze(value) != freeze(Example(value.owners, (128, 1), 128))


def _plan():
    return {
        "contract": _contract(),
        "compiler_head": "head",
        "kernels": [
            {
                "graph": {
                    "inputs": ["input"],
                    "operations": [
                        {
                            "placement": "lx",
                            "physical_stride": [128, 1],
                            "owners": [0, 1],
                            "dependencies": ["input"],
                        }
                    ],
                    "outputs": ["out"],
                    "removed_buffers": [],
                    "relayout_registry": [["input", "copy"]],
                },
                "op_specs": {"sequence": [{"op": "identity", "coordinates": [0, 1]}]},
                "pool_size": 1024,
                "arguments": ["input", "copy"],
            }
        ],
    }


@pytest.mark.parametrize(
    "field", ["placement", "physical_stride", "owners", "dependencies"]
)
def test_plan_comparison_keeps_storage_and_dependencies(field):
    left = _plan()
    right = deepcopy(left)
    right["compiler_head"] = "another-head"
    compare_plans(left, right)
    right["kernels"][0]["graph"]["operations"][0][field] = "changed"
    with pytest.raises(AssertionError, match="semantic plans differ"):
        compare_plans(left, right)


@pytest.mark.parametrize("field", ["op_specs", "pool_size", "arguments"])
def test_plan_comparison_keeps_final_program_fields(field):
    left = _plan()
    right = deepcopy(left)
    right["kernels"][0][field] = (
        {"sequence": ["changed"]} if field == "op_specs" else "changed"
    )
    with pytest.raises(AssertionError, match="semantic plans differ"):
        compare_plans(left, right)


def test_copy_registry_and_missing_sections_cannot_disappear():
    left = _plan()
    right = deepcopy(left)
    right["kernels"][0]["graph"]["relayout_registry"] = []
    with pytest.raises(AssertionError, match="semantic plans differ"):
        compare_plans(left, right)
    del right["kernels"][0]["graph"]["relayout_registry"]
    with pytest.raises(ValueError, match="incomplete final graph"):
        compare_plans(left, right)


def test_unmatched_controls_and_empty_capture_cannot_pass():
    contract = _contract()
    validate_contract(contract)
    left = _plan()
    other = deepcopy(left)
    other["contract"]["settings"]["sencores"] = 16
    with pytest.raises(AssertionError, match="unmatched"):
        compare_plans(left, other)
    other = deepcopy(left)
    other["kernels"] = []
    with pytest.raises(ValueError, match="final kernels"):
        compare_plans(left, other)
    contract["settings"]["lx_extended_ownership"] = True
    with pytest.raises(ValueError, match="flag OFF"):
        validate_contract(contract)


def test_real_dependency_sets_keep_indices_and_ranges():
    from torch._inductor.dependencies import MemoryDep, ReadWrites
    from torch.utils._ordered_set import OrderedSet

    i = sympy.Symbol("i", integer=True, nonnegative=True)
    dep = MemoryDep("input", 64 * i, (i,), (8,))
    reads = ReadWrites(
        reads=OrderedSet([dep]), writes=OrderedSet(), index_exprs=OrderedSet()
    )
    changed = ReadWrites(
        reads=OrderedSet([MemoryDep("input", 128 * i, (i,), (8,))]),
        writes=OrderedSet(),
        index_exprs=OrderedSet(),
    )
    assert freeze(reads) != freeze(changed)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_driver_writes_only_after_the_entire_fixture_passes(
    monkeypatch, tmp_path, exit_code
):
    import importlib.util
    import json
    from pathlib import Path
    import sys

    import tools.lx_planning.lx_semantic_plan as recorder

    path = Path(recorder.__file__).with_name("run_off_plan_capture.py")
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("off_plan_driver_test", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(_contract()))
    output = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--contract",
            str(contract),
            "--output",
            str(output),
            _contract()["fixture"],
        ],
    )
    monkeypatch.setattr(
        driver.subprocess,
        "check_output",
        lambda command, **kwargs: "" if "status" in command else "head\n",
    )

    def run_fixture(args, plugins):
        plugins[0].record = _plan()
        return exit_code

    monkeypatch.setattr(driver.pytest, "main", run_fixture)
    if exit_code:
        with pytest.raises(SystemExit, match="capture did not finish"):
            driver.main()
        assert not output.exists()
    else:
        driver.main()
        assert json.loads(output.read_text()) == _plan()


def test_reordered_iteration_axes_cannot_pass():
    @dataclass
    class Operation:
        iteration_space: dict

    row, feature = sympy.symbols("row feature")
    left, right = _plan(), _plan()
    left["kernels"][0]["op_specs"] = freeze(
        [Operation({row: (512, 8), feature: (64, 4)})]
    )
    right["kernels"][0]["op_specs"] = freeze(
        [Operation({feature: (64, 4), row: (512, 8)})]
    )
    with pytest.raises(AssertionError, match="semantic plans differ"):
        compare_plans(left, right)


@pytest.mark.parametrize("baseline", ["_PRE_FLAG_BASELINE", "_PRE_FLAG_MAIN_BASELINE"])
def test_only_the_verified_pre_flag_baseline_may_omit_its_switch(
    monkeypatch, tmp_path, baseline
):
    import hashlib
    from types import SimpleNamespace

    import tools.lx_planning.lx_semantic_plan as recorder

    source = tmp_path / "config.py"
    source.write_text("verified pre-flag source\n")
    monkeypatch.setattr(
        recorder,
        "_PRE_FLAG_CONFIG_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    config = SimpleNamespace(__file__=str(source), num_cpus=1)
    head = getattr(recorder, baseline)
    assert recorder._live_setting(config, "lx_extended_ownership", head) is False
    with pytest.raises(AttributeError, match="missing compiler setting"):
        recorder._live_setting(config, "lx_extended_ownership", "another-source")
    with pytest.raises(AttributeError, match="missing compiler setting"):
        recorder._live_setting(config, "sencores", head)
    source.write_text("different source\n")
    with pytest.raises(ValueError, match="does not match"):
        recorder._live_setting(config, "lx_extended_ownership", head)


def test_later_pre_flag_baseline_requires_actual_unchanged_main_checkout(
    monkeypatch, tmp_path
):
    import hashlib
    import subprocess
    from types import SimpleNamespace

    import tools.lx_planning.lx_semantic_plan as recorder

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Recorder test",
                "-c",
                "user.email=recorder@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            text=True,
        ).strip()

    git("init", "-b", "main")
    source = tmp_path / "config.py"
    source.write_text("verified pre-flag source\n")
    git("add", "config.py")
    git("commit", "-m", "pre-flag main")
    base = git("rev-parse", "HEAD")
    monkeypatch.setattr(recorder, "_PRE_FLAG_MAIN_BASELINE", base)
    monkeypatch.setattr(
        recorder,
        "_PRE_FLAG_CONFIG_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    git("commit", "--allow-empty", "-m", "later main")
    head = git("rev-parse", "HEAD")
    config = SimpleNamespace(__file__=str(source))
    assert recorder._live_setting(config, "lx_extended_ownership", head) is False
    with pytest.raises(AttributeError, match="missing compiler setting"):
        recorder._live_setting(config, "lx_extended_ownership", "0" * 40)
    source.write_text("different source\n")
    with pytest.raises(ValueError, match="does not match"):
        recorder._live_setting(config, "lx_extended_ownership", head)
    source.write_text("verified pre-flag source\n")
    git("checkout", "-b", "feature")
    git("commit", "--allow-empty", "-m", "not on main")
    with pytest.raises(AttributeError, match="missing compiler setting"):
        recorder._live_setting(
            config, "lx_extended_ownership", git("rev-parse", "HEAD")
        )
    git("checkout", "--orphan", "unrelated")
    git("commit", "-m", "unrelated history")
    git("branch", "-f", "main", "HEAD")
    with pytest.raises(AttributeError, match="missing compiler setting"):
        recorder._live_setting(
            config, "lx_extended_ownership", git("rev-parse", "HEAD")
        )

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

"""CPU contract checks; geometry does not grant LX storage permission."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import sympy

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor.core_mapping import PerCoreView, work_division_from_view
from torch_spyre._inductor.lx_read_contracts import (
    LocalReadEmission,
    LocalReadGeometry,
    has_partial_read,
    local_read_certificate,
    normalize_local_read,
)
from torch_spyre._inductor.op_spec import TensorArg, TensorWorkDivision


def _local_read():
    row, feature = sympy.symbols("row feature", integer=True, nonnegative=True)
    core = sympy.Symbol("core_id")
    view = PerCoreView(((1, 32),), ((1, core),), num_cores=32)
    geometry = LocalReadGeometry((4, 512, 64), (2, 0, 0), (2, 512, 64), 2)
    certificate = local_read_certificate(geometry, view, view)
    assert certificate is not None
    coordinates = [sympy.floor(feature / 64) + 2, row, sympy.Mod(feature, 64)]
    space = {row: 512, feature: 128}
    division = work_division_from_view(
        view, geometry.parent_shape, coordinates, space, allow_extended=True
    )
    argument = TensorArg(
        True,
        -1,
        DataFormats.SEN169_FP16,
        list(geometry.parent_shape),
        coordinates,
        {"lx": 0},
        element_arrangement=ElementArrangement.STANDARD,
        work_division=division,
    )
    contract = LocalReadEmission(
        certificate,
        view,
        DataFormats.SEN169_FP16,
        ElementArrangement.STANDARD,
        (("lx", 0),),
        "consumer",
        (("lx", 32768),),
    )
    return geometry, contract, argument, space


def test_facts_rebase_only_the_final_argument():
    geometry, contract, argument, space = _local_read()
    result = normalize_local_read(
        contract, "parent", argument, space, allow_extended=True
    )
    assert result.device_size == [2, 512, 64] and result.allocation == {"lx": 4096}
    assert result.work_division.same_ownership(argument.work_division)
    assert argument.device_size == [4, 512, 64] and argument.allocation == {"lx": 0}
    assert contract.certificate.source_reservation_bytes == 8192
    with pytest.raises(FrozenInstanceError):
        geometry.origin = (0, 0, 0)


@pytest.mark.parametrize(
    "case",
    [
        "shape",
        "coordinates",
        "allocation",
        "dtype",
        "arrangement",
        "advance",
        "overlap",
        "owners",
        "unaligned",
        "negative",
    ],
)
def test_stale_or_unsafe_final_facts_are_rejected(case):
    _, contract, argument, space = _local_read()
    message = "differs from its certificate"
    if case == "shape":
        argument = replace(argument, device_size=[4, 511, 64])
    elif case == "coordinates":
        argument = replace(
            argument,
            device_coordinates=[
                argument.device_coordinates[0] - 2,
                *argument.device_coordinates[1:],
            ],
        )
    elif case == "allocation":
        argument = replace(argument, allocation={"lx": 128})
    elif case == "dtype":
        argument = replace(argument, device_dtype=None)
    elif case == "arrangement":
        argument = replace(argument, element_arrangement=None)
    elif case == "advance":
        argument = replace(argument, device_tile_advance_expr=sympy.Integer(1))
    elif case == "overlap":
        contract = replace(contract, consumer_allocation=(("lx", 4096),))
        message = "overlaps its live parent"
    elif case == "owners":
        row = next(iter(argument.work_division.work_slices))
        changed = TensorWorkDivision(
            {row: 32}, {row: 31 - sympy.Symbol("core_id")}, num_cores=32
        )
        argument = replace(argument, work_division=changed)
        message = "rebasing changed"
    else:
        base = 1 if case == "unaligned" else -128
        argument = replace(argument, allocation={"lx": base})
        contract = replace(contract, source_allocation=(("lx", base),))
        message = "concrete stick address"
    with pytest.raises(ValueError, match=message):
        normalize_local_read(contract, "parent", argument, space, allow_extended=True)


def test_missing_facts_and_unknown_coverage_decline():
    geometry, contract, argument, space = _local_read()
    assert (
        local_read_certificate(None, contract.source_view, contract.source_view) is None
    )
    assert local_read_certificate(geometry, None, contract.source_view) is None
    with pytest.raises(ValueError, match="lacks a local certificate"):
        normalize_local_read(None, "parent", argument, space, allow_extended=True)
    assert not has_partial_read((4, 8), [SimpleNamespace(get_numel=lambda: 32)])
    assert has_partial_read((4, 8), [SimpleNamespace(get_numel=lambda: 16)])
    assert has_partial_read((4, 8), [object()])


def test_work_division_and_emission_use_the_fact_interface():
    import ast
    from pathlib import Path

    from torch_spyre._inductor import lx_read_contracts as contracts_module

    root = Path(contracts_module.__file__).resolve().parent
    forbidden = {
        "normalize_local_lx_subrange",
        "local_lx_subrange_certificate",
        "is_partial_lx_read",
        "work_division_from_view",
    }
    for filename in ("work_division.py", "spyre_kernel.py"):
        tree = ast.parse((root / filename).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "scratchpad.lx_relayout"
            ):
                assert not forbidden.intersection(alias.name for alias in node.names)
    tree = ast.parse((root / "lx_read_contracts.py").read_text())
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and ("scratchpad" in node.module or node.module.endswith("pass_utils"))
        for node in ast.walk(tree)
    )


def test_residency_edge_declines_partial_reads_without_planner_facts():
    from torch_spyre._inductor.work_division import ResidencyEdge

    geometry, contract, _, _ = _local_read()
    edge = SimpleNamespace(local_read_geometry=None, _has_partial_read=lambda: True)
    check = ResidencyEdge._local_reads_compatible
    assert not check(edge, contract.source_view, contract.source_view)
    edge.local_read_geometry = geometry
    assert check(edge, contract.source_view, contract.source_view)
    assert not check(edge, None, contract.source_view)
    edge.local_read_geometry = None
    edge._has_partial_read = lambda: False
    assert check(edge, contract.source_view, contract.source_view)


def test_rollout_off_declines_partial_facts_before_certification(monkeypatch):
    from torch_spyre._inductor import config
    from torch_spyre._inductor import work_division

    def unexpected_certificate(*args):
        raise AssertionError("OFF queried an optional local-read certificate")

    monkeypatch.setattr(work_division, "local_read_certificate", unexpected_certificate)
    edge = SimpleNamespace(local_read_geometry=object(), _has_partial_read=lambda: True)
    with config.patch(lx_extended_ownership=False):
        assert not work_division.ResidencyEdge._local_reads_compatible(edge, None, None)
        edge._has_partial_read = lambda: False
        assert work_division.ResidencyEdge._local_reads_compatible(edge, None, None)

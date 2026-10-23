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


"""The solve's CP-SAT parameters and the relayout commit path, driven without
a graph.

- the solve presolves every model and sizes its parallel search portfolio from
  the CPU count (``config.get_cpu_count``), restricted only in deterministic
  mode;
- the commit path regroups fired edges by (source, destination view), and the
  plan carries the candidate's views, measured spans and core counts.

The solver's relayout decision itself (lazy relayouts over view ids) is covered
by ``test_lazy_strategy.py`` and the end-to-end suite.
"""

import pytest
import sympy

pytest.importorskip("ortools")

from torch_spyre._inductor import config
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad import buffer_graph
from torch_spyre._inductor.scratchpad.allocator import SolvedPlan
from torch_spyre._inductor.scratchpad.lazy_strategy import LazyStrategy
from torch_spyre._inductor.scratchpad.lx_relayout import (
    ChosenRelayout,
    FiredRelayoutGroup,
    RelayoutCandidate,
)
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision
from buffer_spec import CoreDivisionBuffer, build_graph

_SPILL_NS = 20000.0  # what spilling P charges in the crafted objective
_CORE = sympy.Symbol("core_id")


def _view(slot: int, num_cores: int = 4) -> PerCoreView:
    """A 4-way per-core view of device dim 1 on ``num_cores`` cores; ``slot``
    rotates the ownership so distinct slots are distinct (relayout-compatible)
    views. More cores than owners is a broadcast destination."""
    return PerCoreView(((1, 4),), ((1, sympy.Mod(_CORE + slot, 4)),), num_cores)


def _candidate(
    consumer, i, cost_ns, group=0, j=0, destination_span=16, destination_cores=4
) -> RelayoutCandidate:
    """A priced candidate for P -> consumer under source division ``i`` /
    consumer division ``j``, landing on destination view ``group``. The commit
    path only carries the views, so their exact geometry is immaterial here;
    they must merely differ from the source and be distinct per group."""
    return RelayoutCandidate(
        parent="P",
        consumer=consumer,
        source_division=i,
        consumer_division=j,
        group=group,
        source_view=_view(0),
        destination_view=_view(group + 1, destination_cores),
        cost_ns=cost_ns,
        # P is 64 bytes over 4 cores; a permutation of an outer split keeps
        # the equal share as its per-core span on both sides.
        source_footprint_bytes=16,
        destination_footprint_bytes=destination_span,
    )


def _buffers() -> list[CoreDivisionBuffer]:
    """P (uses [0, 1]) feeds C (uses [1, 2]); one 4-way division each, which
    C reads through a match pair."""
    division = CoreDivision(splits={1: 4})
    return [
        CoreDivisionBuffer("P", 64, [0, 1], core_divisions=[division]),
        CoreDivisionBuffer(
            "C",
            64,
            [1, 2],
            core_divisions=[division],
            parents=["P"],
            cd_parent_matches={"P": [(0, 0)]},
        ),
    ]


def _objective(buffers) -> sympy.Expr:
    (p,) = (b for b in buffers if b.name == "P")
    return (1 - p.sym_is_lx) * _SPILL_NS


def _solve(buffers, expr, size=64) -> SolvedPlan:
    """CP-SAT's plan for ``buffers`` over ``size`` bytes of LX, in byte units."""
    graph = build_graph(buffers, 1, size, buffer_graph.empty_graph(size, 1).settings)
    nodes = graph.solve(cost_expr=expr, strategy=LazyStrategy())
    return SolvedPlan(nodes, graph.alignment)


@pytest.mark.parametrize("priced", [False, True])
def test_solve_presolves_by_default(monkeypatch, priced):
    from ortools.sat.python import cp_model

    buffers = _buffers()
    original = cp_model.CpSolver.Solve
    parameters = []

    def solve(solver, model, *args, **kwargs):
        parameters.append(solver.parameters.cp_model_presolve)
        assert solver.parameters.max_time_in_seconds == config.cpsat_time_limit_seconds
        return original(solver, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "Solve", solve)
    plan = _solve(buffers, _objective(buffers) if priced else None)
    assert parameters and all(parameters)
    assert plan.address(plan.by_name["P"]) is not None, (
        "P fits and C reads it under a match"
    )


@pytest.mark.parametrize(
    "deterministic,num_cpus,expected_workers",
    # Two counts, so the case cannot pass by detection happening to match one.
    [(False, 4, 4), (False, 16, 16), (True, 16, 1)],
)
def test_solve_uses_available_parallel_search_workers(
    monkeypatch, deterministic, num_cpus, expected_workers
):
    """Only deterministic mode restricts CP-SAT's parallel search portfolio."""
    from ortools.sat.python import cp_model

    buffers = _buffers()
    original = cp_model.CpSolver.Solve
    workers = []

    monkeypatch.setattr(config, "num_cpus", num_cpus)
    monkeypatch.setattr(
        buffer_graph.torch,
        "are_deterministic_algorithms_enabled",
        lambda: deterministic,
    )

    def solve(solver, model, *args, **kwargs):
        workers.append(solver.parameters.num_search_workers)
        assert solver.parameters.share_level_zero_bounds
        return original(solver, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "Solve", solve)
    _solve(buffers, _objective(buffers))
    assert workers == [expected_workers]


def test_cpu_count_setting_overrides_detection(monkeypatch):
    """A set ``num_cpus`` is returned as is, before any detection runs."""
    monkeypatch.setattr(config, "num_cpus", 3)
    assert config.get_cpu_count() == 3


def test_cpu_count_detects_when_the_setting_is_unset(monkeypatch):
    """None means "detect"; a set 0 is not a CPU count either, so it detects the
    same count, and a patched count lasts only as long as its patch."""
    monkeypatch.setattr(config, "num_cpus", None)
    detected = config.get_cpu_count()
    assert detected >= 1
    monkeypatch.setattr(config, "num_cpus", 0)
    assert config.get_cpu_count() == detected
    monkeypatch.setattr(config, "num_cpus", None)
    with config.patch(num_cpus=detected + 1):
        assert config.get_cpu_count() == detected + 1
    assert config.num_cpus is None
    assert config.get_cpu_count() == detected


def test_num_cpus_stays_out_of_the_inductor_cache_key():
    """The worker count describes the machine, not the graph: setting it must not
    change the FX graph cache key, which ``save_config_portable`` of this config
    module feeds (it is the backend's ``device_custom_config``). A setting that
    does shape the compile still changes the key, so the config is still in it."""

    def key():
        return config.save_config_portable(
            ignore_private_configs=False, readonly_values=True
        )

    base = key()
    assert "num_cpus" not in base
    with config.patch(num_cpus=3):
        assert key() == base
    with config.patch(cpsat_time_limit_seconds=config.cpsat_time_limit_seconds + 1):
        assert key() != base


def test_cpu_count_is_read_per_call(monkeypatch):
    """No cached count: a patched ``num_cpus`` takes effect on the next call."""
    monkeypatch.setattr(config, "num_cpus", 5)
    assert config.get_cpu_count() == 5
    monkeypatch.setattr(config, "num_cpus", 7)
    assert config.get_cpu_count() == 7


@pytest.mark.parametrize(
    "cpu_max,expected",
    [
        ("400000 100000\n", 4),  # a container quota below the host count
        ("50000 100000\n", 1),  # a fractional quota still gets one worker
    ],
)
def test_cpu_count_detects_the_cgroup_quota(monkeypatch, cpu_max, expected):
    """Without a setting, the cgroup v2 quota wins over the host's CPU count."""
    import builtins
    import io

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/sys/fs/cgroup/cpu.max":
            return io.StringIO(cpu_max)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(config, "num_cpus", None)
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(config.os, "cpu_count", lambda: 32)
    assert config.get_cpu_count() == expected


def test_cpu_count_without_a_quota_is_at_least_one(monkeypatch):
    """An unlimited cgroup (``max``) falls through to the host count, never 0."""
    import builtins
    import io

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/sys/fs/cgroup/cpu.max":
            return io.StringIO("max 100000\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(config, "num_cpus", None)
    monkeypatch.setattr(builtins, "open", fake_open)
    assert config.get_cpu_count() >= 1


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("8", 8),
        (" 8 ", 8),
        ("0", None),
        ("-2", None),
        ("abc", None),
    ],
)
def test_num_cpus_env_accepts_only_positive_integers(monkeypatch, value, expected):
    """SPYRE_NUM_CPUS backs ``num_cpus``; anything but a positive integer is None,
    "detect", as before the setting moved into the config."""
    if value is None:
        monkeypatch.delenv("SPYRE_NUM_CPUS", raising=False)
    else:
        monkeypatch.setenv("SPYRE_NUM_CPUS", value)
    assert config._positive_int_env("SPYRE_NUM_CPUS") == expected


# ---------------------------------------------------------------------------
# The commit path's view
# ---------------------------------------------------------------------------


def test_a_broadcast_plan_keeps_the_source_core_count():
    """A source on 4 cores feeding a matmul on 8 (#3440 broadcast): the plan
    keeps the source's core count and the destination view keeps its own."""
    (group,) = FiredRelayoutGroup.from_chosen(
        [ChosenRelayout(_candidate("C", 0, 5000.0, destination_cores=8), 16)]
    )
    plan = group.plan(source_address=0)
    assert plan.num_cores == 4, "the plan's core count is the source's (#3440)"
    assert plan.destination_view.num_cores == 8
    with pytest.raises(ValueError, match="not a multiple"):
        _candidate("C", 0, 5000.0, destination_cores=6)
    with pytest.raises(ValueError, match="no physical core count"):
        RelayoutCandidate("P", "C", 0, 0, 0, _view(0), _view(1, None), 1.0, 16, 16)


def test_plan_carries_the_measured_spans():
    """The plan hands the allocator the spans the enumeration measured, so the
    committed side sizes source and destination exactly as the solver did."""
    fired = ChosenRelayout(_candidate("C1", 0, 5000.0, destination_span=48), 16)
    (group,) = FiredRelayoutGroup.from_chosen([fired])
    plan = group.plan(source_address=0)
    assert (plan.source_footprint_bytes, plan.destination_footprint_bytes) == (16, 48)
    with pytest.raises(AssertionError, match="disagree"):
        FiredRelayoutGroup.from_chosen(
            [
                fired,
                ChosenRelayout(_candidate("C2", 0, 5000.0, destination_span=32), 16),
            ]
        )


def test_fired_groups_regroup_edges_by_source_and_view():
    """The commit path's view of a solve: fired edges regroup into one group
    per (parent, view), members sorted by consumer, and the plan carries the
    candidate's views and the copy's address. Members that disagree on
    placement are a solver invariant violation, not a plan."""
    c1 = ChosenRelayout(_candidate("C1", 0, 5000.0), 16)
    c2 = ChosenRelayout(_candidate("C2", 0, 5000.0), 16)
    other = ChosenRelayout(_candidate("C3", 0, 7000.0, group=1), 48)
    groups = FiredRelayoutGroup.from_chosen([other, c2, c1])
    assert [(g.group, g.consumer_names) for g in groups] == [
        (0, ("C1", "C2")),
        (1, ("C3",)),
    ]
    plan = groups[0].plan(source_address=0)
    assert (plan.source_name, plan.consumer_names) == ("P", ("C1", "C2"))
    assert (plan.source_view, plan.destination_view) == (_view(0), _view(1))
    assert (plan.num_cores, plan.source_address, plan.destination_address) == (
        4,
        0,
        16,
    )
    assert c1.scaled(128).destination_address == 16 * 128
    with pytest.raises(AssertionError, match="disagree"):
        FiredRelayoutGroup.from_chosen(
            [c1, ChosenRelayout(_candidate("C2", 0, 5000.0), 64)]
        )
    with pytest.raises(ValueError, match="equal views"):
        RelayoutCandidate("P", "C", 0, 0, 0, _view(0), _view(0), 1.0, 16, 16)
    # The core counts are the views': a view without one cannot be a candidate.
    bare = PerCoreView(((1, 4),), ((1, sympy.Mod(_CORE + 1, 4)),))
    with pytest.raises(ValueError, match="no physical core count"):
        RelayoutCandidate("P", "C", 0, 0, 0, _view(0), bare, 1.0, 16, 16)

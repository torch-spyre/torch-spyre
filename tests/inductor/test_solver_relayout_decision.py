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

"""The relayout decision, driven without a graph.

A relayout group's destination is a ``RelayoutCopyBuffer``: an ordinary buffer
whose residency IS the decision to shuffle, placed by the same no-overlap as
everything else and priced by a plain term of the shared sympy objective
(``cost_term``, one ``RelayoutCharge`` node per copy). Handcrafted buffers
drive the pieces directly - no graph, no compile:

- the price term is solver-agnostic: evaluated by ``lambdify`` it charges the
  source's chosen division exactly when the copy is resident;
- under ``CpSatLayoutSolver`` a relayout fires when its fitted cost beats the
  spill it avoids, and only then; the copy occupies real LX (capacity can
  veto); no relayout is decided under the fallback objective; the gate's old
  behavior (no match, no table -> spilled) survives;
- one copy serves every consumer of a group (one charge), distinct views are
  distinct copies, the charge follows the chosen source division, and a copy
  that cannot be held across its consumers spills the source rather than
  re-shuffling;
- the commit path regroups fired edges by (source, destination view).

The producer P (uses [0, 1]) feeds consumer C (uses [1, 2]). P and C each
carry one 4-way division; ``cd_parent_matches`` is EMPTY on the edge, so
residency for P is possible only through the relayout copy.
"""

import logging
import random
from dataclasses import replace

import pytest
import sympy

pytest.importorskip("ortools")

from torch_spyre._inductor import config
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad import ilp_solver_ortools, sa_cooptimizer, utils
from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver
from torch_spyre._inductor.scratchpad.sa_cooptimizer import SaCoOptimizingSolver
from torch_spyre._inductor.scratchpad.sa_cooptimizer import logger as sa_logger
from torch_spyre._inductor.scratchpad.lx_relayout import (
    ChosenRelayout,
    FiredRelayoutGroup,
    RelayoutCandidate,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    RelayoutCopyBuffer,
    relayout_copy_name,
)
from test_sa_cooptimizer import _live_state  # noqa: E402

_SPILL_NS = 20000.0  # what spilling P charges in the crafted objective
_CORE = sympy.Symbol("core_id")
_PER_CORE = 16  # P is 64 bytes sliced 4 ways


@pytest.mark.parametrize(
    "cap,costs,expected",
    [(1, {0: 10000, 1: 1000}, [1]), (1, None, [0]), (0, {0: 10000, 1: 1000}, [0, 1])],
)
def test_relayout_shortlist_prices_the_consumer(monkeypatch, cap, costs, expected):
    """Saving 500 ns on a copy must not hide a 9000 ns faster consumer."""
    monkeypatch.setattr(config, "lx_solver_relayout_groups_per_edge", cap)
    candidates = [
        _candidate("C", 0, 500, group=0, j=0),
        _candidate("C", 0, 1000, group=1, j=1),
    ]
    divisions = [CoreDivision(splits={sympy.Symbol("d0"): 4})] * 2
    kept = CoOptimizingAllocator._cap_relayout_groups(
        "P", "C", candidates, divisions, costs
    )
    assert [c.group for c in kept] == expected


@pytest.mark.parametrize(
    "cap,pinned,expected",
    [
        (2, (), [0, 2]),
        (2, (1,), [1, 2]),
        (1, (1,), [1]),
        (1, (0, 1), [0, 1]),
        (1, (2,), [2]),
        (0, (1,), [0, 1, 2]),
    ],
)
def test_relayout_full_replica_preserves_sibling_pins(
    monkeypatch, cap, pinned, expected
):
    monkeypatch.setattr(config, "lx_solver_relayout_groups_per_edge", cap)
    candidates = [
        _candidate("C", 0, 100, group=0, j=0),
        _candidate("C", 0, 200, group=1, j=1),
        replace(
            _candidate("C", 0, 1000, group=2, j=2),
            destination_view=PerCoreView((), (), num_cores=4),
            destination_footprint_bytes=64,
        ),
    ]
    divisions = [CoreDivision(splits={sympy.Symbol("d0"): 4})] * 3
    kept = CoOptimizingAllocator._cap_relayout_groups(
        "P",
        "C",
        candidates,
        divisions,
        pinned_consumer_divisions=frozenset(pinned),
    )
    assert [c.group for c in kept] == expected
    assert all(any(c is original for original in candidates) for c in kept)


def _view(slot: int, num_cores: int = 4) -> PerCoreView:
    """A 4-way per-core view of device dim 1 on ``num_cores`` cores; ``slot``
    rotates the ownership so distinct slots are distinct (relayout-compatible)
    views. More cores than owners is a broadcast destination."""
    return PerCoreView(((1, 4),), ((1, sympy.Mod(_CORE + slot, 4)),), num_cores)


def _candidate(
    consumer,
    i,
    cost_ns,
    group=0,
    j=0,
    destination_span=16,
    destination_cores=4,
    parent="P",
) -> RelayoutCandidate:
    """The priced candidate the allocator would enumerate for P -> consumer under
    source division ``i`` / consumer division ``j``, landing on destination
    view ``group``. The solver only carries the views, so their exact geometry
    is immaterial here; they must merely differ from the source and be
    distinct per group."""
    return RelayoutCandidate(
        parent=parent,
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


def _producer(uses, divisions=1) -> CoreDivisionBuffer:
    """P with ``divisions`` 4-way divisions (index 0 splits dim 1, index 1 dim 0)."""
    return CoreDivisionBuffer(
        "P",
        64,
        uses,
        core_divisions=[CoreDivision(splits={1 - d: 4}) for d in range(divisions)],
    )


def _consumer(name, start, end, candidates, *, matches=()) -> CoreDivisionBuffer:
    return CoreDivisionBuffer(
        name,
        64,
        [start, end],
        core_divisions=[CoreDivision(splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": list(matches)},
        cd_parent_relayouts={"P": list(candidates)} if candidates else {},
    )


def _with_copies(*buffers) -> list[CoreDivisionBuffer]:
    """The buffer set the allocator hands the solver: the graph buffers plus
    one copy per relayout group, built by the allocator's own builder."""
    return [*buffers, *CoOptimizingAllocator._relayout_copy_buffers(list(buffers))]


def _objective(buffers, spill_ns=None) -> sympy.Expr:
    """Spilling a source costs its ``spill_ns`` entry (P: _SPILL_NS); every
    copy adds its own price term, exactly as the allocator composes it."""
    spill_ns = {"P": _SPILL_NS, **(spill_ns or {})}
    expr = sympy.Integer(0)
    for b in buffers:
        if isinstance(b, RelayoutCopyBuffer):
            expr = expr + b.cost_term()
        elif b.name in spill_ns:
            expr = expr + (1 - b.sym_is_lx) * spill_ns[b.name]
    return expr


_ENGINES = {"cpsat": CpSatLayoutSolver, "annealer": SaCoOptimizingSolver}


@pytest.fixture(params=sorted(_ENGINES))
def engine(request):
    """Each engine that decides relayouts, by name."""
    return request.param


def _solve(engine, buffers, size=64, expr="default", **spill_ns):
    if expr == "default":
        expr = _objective(buffers, spill_ns)
    solver = _ENGINES[engine](buffers, size, alignment=1)
    return {b.name: b for b in solver.plan_layout_and_core_divisions(expr)}


def _copy(result, group=0) -> RelayoutCopyBuffer:
    return result[relayout_copy_name("P", group)]


def _disjoint(a_addr, b_addr, footprint=_PER_CORE) -> bool:
    return not (a_addr < b_addr + footprint and b_addr < a_addr + footprint)


@pytest.mark.parametrize("priced", [False, True])
def test_relayout_solve_presolves_by_default(monkeypatch, priced):
    from ortools.sat.python import cp_model

    p = _producer([0, 1])
    c = _consumer("C", 1, 2, [_candidate("C", 0, 5000.0)])
    buffers = _with_copies(p, c)
    original = cp_model.CpSolver.Solve
    parameters = []

    def solve(solver, model, *args, **kwargs):
        parameters.append(solver.parameters.cp_model_presolve)
        assert solver.parameters.max_time_in_seconds == config.cpsat_time_limit_seconds
        return original(solver, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "Solve", solve)
    result = _solve("cpsat", buffers, expr=_objective(buffers) if priced else None)
    assert parameters and all(parameters)
    assert (_copy(result).address is not None) == priced


@pytest.mark.parametrize(
    "deterministic,num_cpus,expected_workers",
    # Two counts, so the case cannot pass by detection happening to match one.
    [(False, 4, 4), (False, 16, 16), (True, 16, 1)],
)
def test_relayout_solve_uses_available_parallel_search_workers(
    monkeypatch, deterministic, num_cpus, expected_workers
):
    """Only deterministic mode restricts CP-SAT's parallel search portfolio."""
    from ortools.sat.python import cp_model

    p = _producer([0, 1])
    c = _consumer("C", 1, 2, [_candidate("C", 0, 5000.0)])
    buffers = _with_copies(p, c)
    original = cp_model.CpSolver.Solve
    workers = []

    monkeypatch.setattr(config, "num_cpus", num_cpus)
    monkeypatch.setattr(
        ilp_solver_ortools.torch,
        "are_deterministic_algorithms_enabled",
        lambda: deterministic,
    )

    def solve(solver, model, *args, **kwargs):
        workers.append(solver.parameters.num_search_workers)
        assert solver.parameters.share_level_zero_bounds
        return original(solver, model, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "Solve", solve)
    _solve("cpsat", buffers, expr=_objective(buffers))
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


def _inductor_cache_key_config():
    return config.save_config_portable(
        ignore_private_configs=False, readonly_values=True
    )


# Settings that describe the machine, or where logs and kernels go, not the
# graph. Each maps to a function giving a value that differs from the current
# one, so the test also changes something when the environment sets it.
_KEPT_OUT_OF_THE_CACHE_KEY = {
    "num_cpus": lambda value: (value or 1) + 2,
    "backend_compile_timeout_seconds": lambda value: value + 1,
    "timing": lambda value: not value,
    "timing_out": lambda value: value + ".other",
    "dump_cost_expr_file": lambda value: value + ".other",
    "spyre_kernel_cache": lambda value: not value,
}


@pytest.mark.parametrize("name", sorted(_KEPT_OUT_OF_THE_CACHE_KEY))
def test_setting_stays_out_of_the_inductor_cache_key(name):
    """These settings do not change the compiled code: setting one must not
    change the FX graph cache key, which ``save_config_portable`` of this config
    module feeds (it is the backend's ``device_custom_config``). A setting that
    does shape the compile still changes the key, so the config is still in it."""
    base = _inductor_cache_key_config()
    assert name not in base
    new_value = _KEPT_OUT_OF_THE_CACHE_KEY[name](getattr(config, name))
    with config.patch({name: new_value}):
        assert _inductor_cache_key_config() == base
    with config.patch(cpsat_time_limit_seconds=config.cpsat_time_limit_seconds + 1):
        assert _inductor_cache_key_config() != base


def test_each_cache_key_ignore_prefix_matches_only_known_settings():
    """The ignore list matches by prefix, so a new setting whose name starts with
    one of them would leave the cache key without notice. Pin what each prefix
    matches today: such a setting fails here first."""
    names = config.get_config_copy()
    matched = {
        prefix: sorted(name for name in names if name.startswith(prefix))
        for prefix in config._cache_config_ignore_prefix
    }
    assert matched == {
        "num_cpus": ["num_cpus"],
        "backend_compile_timeout_seconds": ["backend_compile_timeout_seconds"],
        "timing": ["timing", "timing_out"],
        "dump_cost_expr_file": ["dump_cost_expr_file"],
        "spyre_kernel_cache": ["spyre_kernel_cache"],
    }


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
# The copy buffer and its generic price term
# ---------------------------------------------------------------------------


def test_copy_buffer_is_the_destination_the_allocator_would_build():
    p = _producer([0, 3])
    c1 = _consumer("C1", 1, 2, [_candidate("C1", 0, 5000.0)])
    c2 = _consumer("C2", 2, 3, [_candidate("C2", 0, 5000.0)])
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c1, c2])
    assert copy.name == relayout_copy_name("P", 0)
    assert copy.name.startswith("__spyre_lx_relayout__:"), (
        "must share the synthetic prefix so nothing tries to push or commit it"
    )
    assert copy.relayout_parent == "P" and copy.group == 0
    assert copy.consumers == ("C1", "C2")
    # Live from the first consumer's tick to the last's; the per-core footprint
    # is the destination view's measured span, sliced num_cores ways.
    assert (copy.start_time, copy.end_time) == (1, 3)
    assert copy.size == 64 and copy.num_cores == 4 and copy.min_footprint == 16
    assert copy.parents == [] and copy.cd_parent_matches == {}
    assert copy.cost_by_source_division == {0: 5000.0}


def test_copy_is_sized_by_the_destination_span_not_the_source_share():
    """A view that splits an inner device dim spans more LX per core than its
    equal share (#3440 reserves the span for the committed destination). The
    copy must reserve the same, or the solver packs a neighbour into bytes the
    shuffle will write."""
    p = _producer([0, 3])
    c1 = _consumer("C1", 1, 2, [_candidate("C1", 0, 5000.0, destination_span=48)])
    c2 = _consumer("C2", 2, 3, [_candidate("C2", 0, 5000.0, destination_span=48)])
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c1, c2])
    assert copy.size == 48 * 4 and copy.min_footprint == 48, (
        "per-core footprint must be the destination span, not size / num_cores"
    )
    # Members of one group land on one view, so they were measured alike; a
    # disagreement is an enumeration error, never averaged or maxed away.
    c3 = _consumer("C3", 2, 3, [_candidate("C3", 0, 5000.0, destination_span=32)])
    with pytest.raises(AssertionError, match="mixes destination spans"):
        CoOptimizingAllocator._relayout_copy_buffers([p, c1, c3])


def test_a_broadcast_copy_lives_on_the_destination_cores():
    """A source on 4 cores feeding a matmul on 8 (#3440 broadcast): the copy is
    the destination, so it is sliced the destination's way, one span per
    destination core, while the plan keeps the source's core count."""
    p = _producer([0, 3])
    c = _consumer(
        "C",
        1,
        2,
        [_candidate("C", 0, 5000.0, destination_span=48, destination_cores=8)],
    )
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c])
    assert copy.num_cores == 8 and copy.size == 48 * 8 and copy.min_footprint == 48
    (group,) = FiredRelayoutGroup.from_chosen(
        [ChosenRelayout(_candidate("C", 0, 5000.0, destination_cores=8), 16)]
    )
    plan = group.plan(source_address=0)
    assert plan.num_cores == 4, "the plan's core count is the source's (#3440)"
    assert plan.destination_view.num_cores == 8
    with pytest.raises(ValueError, match="destination on 6 cores is incompatible.*4"):
        _candidate("C", 0, 5000.0, destination_cores=6)
    with pytest.raises(ValueError, match="no physical core count"):
        RelayoutCandidate("P", "C", 0, 0, 0, _view(0), _view(1, None), 1.0, 16, 16)
    with pytest.raises(AssertionError, match="mixes destination core counts"):
        CoOptimizingAllocator._relayout_copy_buffers(
            [p, c, _consumer("D", 1, 2, [_candidate("D", 0, 5000.0)])]
        )


def test_one_destination_may_be_fed_from_sources_on_different_core_counts():
    """A producer's division menu spans core counts; under broadcast admission
    several of its divisions can land on the same 32-core matmul view. The copy
    is that view, so it is built once, and each source division keeps its own
    price in the table."""
    p = _producer([0, 3], divisions=2)
    c = _consumer(
        "C",
        1,
        2,
        [
            _candidate("C", 0, 5000.0, destination_cores=8),  # source on 4 cores
            RelayoutCandidate(
                parent="P",
                consumer="C",
                source_division=1,
                consumer_division=0,
                group=0,
                source_view=_view(0, 2),  # a source division on 2 cores
                destination_view=_view(1, 8),
                cost_ns=3000.0,
                source_footprint_bytes=32,
                destination_footprint_bytes=16,
            ),
        ],
    )
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c])
    assert copy.num_cores == 8 and copy.size == 16 * 8
    assert copy.cost_by_source_division == {0: 5000.0, 1: 3000.0}


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


def test_price_term_charges_the_chosen_source_division_while_resident():
    """Evaluated the way the annealer evaluates the objective: ``lambdify`` over
    (is_lx, division) values. The term is the solver-agnostic contract, so pin
    it here independently of any engine."""
    p = _producer([0, 2], divisions=2)
    c = _consumer("C", 1, 2, [_candidate("C", 0, 5000.0), _candidate("C", 1, 3000.0)])
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c])
    term = copy.cost_term()
    assert term.free_symbols == {copy.sym_is_lx, p.sym_division}
    f = sympy.lambdify([copy.sym_is_lx, p.sym_division], term, modules="math")
    assert (f(1, 0), f(1, 1)) == (5000.0, 3000.0)
    assert f(0, 0) == f(0, 1) == 0.0, "no charge while the copy is not resident"


def test_price_disagreement_within_a_group_is_an_error():
    p = _producer([0, 2])
    c1 = _consumer("C1", 1, 2, [_candidate("C1", 0, 5000.0)])
    c2 = _consumer("C2", 1, 2, [_candidate("C2", 0, 7000.0)])
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c1, c2])
    with pytest.raises(AssertionError, match="disagree on the price"):
        copy.cost_by_source_division


def test_group_without_its_source_gets_no_copy():
    c = _consumer("C", 1, 2, [_candidate("C", 0, 5000.0)])
    assert CoOptimizingAllocator._relayout_copy_buffers([c]) == []


# ---------------------------------------------------------------------------
# The decision, under every engine that makes it
# ---------------------------------------------------------------------------


def _pc(relayout_cost_ns):
    return _with_copies(
        _producer([0, 1]), _consumer("C", 1, 2, [_candidate("C", 0, relayout_cost_ns)])
    )


def test_relayout_fires_when_cheaper_than_the_spill(engine):
    r = _solve(engine, _pc(5000.0))
    p, c, copy = r["P"], r["C"], _copy(r)
    assert p.address is not None, "P must reside: relayout beats the spill"
    assert copy.address is not None, "the copy IS the decision"
    chosen = c.chosen_relayouts["P"]
    assert chosen.candidate == _candidate("C", 0, 5000.0)
    assert chosen.destination_address == copy.address
    # The copy is real LX space, disjoint from P's per-core footprint (both
    # are alive at C's tick): [addr, addr+16) each, 64-byte capacity.
    assert 0 <= copy.address <= 64 - _PER_CORE
    assert _disjoint(copy.address, p.address), "copy overlaps its source"


def test_relayout_declines_when_dearer_than_the_spill(engine):
    r = _solve(engine, _pc(50000.0))
    assert r["P"].address is None, "spilling is cheaper: P must not reside"
    assert _copy(r).address is None and r["C"].chosen_relayouts == {}


def test_capacity_vetoes_a_profitable_relayout(engine):
    # 16-byte LX: P's per-core footprint fills it, so the copy cannot coexist
    # with P at C's tick. The copy stays out no matter how profitable, and P
    # without a serving copy cannot reside (gate: match or served).
    r = _solve(engine, _pc(5000.0), size=16)
    assert _copy(r).address is None and r["C"].chosen_relayouts == {}
    assert r["P"].address is None


def test_fallback_objective_never_decides_a_relayout(engine):
    # Under the HBM-bytes fallback (cost_expr=None) a shuffle is unpriced and
    # would look free; every copy must be pinned out, restoring the old gate:
    # no match pair -> P spilled.
    r = _solve(engine, _pc(5000.0), expr=None)
    assert r["P"].address is None
    assert _copy(r).address is None and r["C"].chosen_relayouts == {}


def test_copy_never_resides_without_its_source(engine):
    # P is barred from LX by the allocator (residency_reason). Its copy has a
    # profitable price and room to spare, but a copy needs its source
    # resident: it must stay out.
    p = _producer([0, 1])
    p.residency_reason = "barred for the test"
    bufs = _with_copies(p, _consumer("C", 1, 2, [_candidate("C", 0, 1.0)]))
    r = _solve(engine, bufs)
    assert r["P"].address is None
    assert _copy(r).address is None and r["C"].chosen_relayouts == {}


def test_gate_without_a_relayout_table_is_unchanged(engine):
    # No match pairs and no relayout table: in_buffer is forced off exactly
    # as before (regression guard for the constrain_residency rewrite). No
    # copy exists, so no price term either.
    bufs = _with_copies(_producer([0, 1]), _consumer("C", 1, 2, []))
    assert not any(isinstance(b, RelayoutCopyBuffer) for b in bufs)
    r = _solve(engine, bufs)
    assert r["P"].address is None and r["C"].chosen_relayouts == {}


def _fanout(relayout_cost_ns, same_view: bool):
    """P feeds C1 (uses [1, 2]) and C2 (uses [2, 3]); both edges have an empty
    match table and one priced relayout candidate. With ``same_view`` both
    candidates land on the same destination view of P (group 0); otherwise
    they are distinct views (groups 0 and 1)."""
    return _with_copies(
        _producer([0, 3]),
        _consumer("C1", 1, 2, [_candidate("C1", 0, relayout_cost_ns, 0)]),
        _consumer(
            "C2", 2, 3, [_candidate("C2", 0, relayout_cost_ns, 0 if same_view else 1)]
        ),
    )


def test_two_consumers_on_one_view_share_one_copy_and_one_charge(engine):
    """Fan-out to two consumers wanting the SAME destination view: one copy
    spanning both consumers' ticks, one charge. Spilling costs 20000; the
    shuffle 12000. Charged per edge it would be 24000 and the solver would
    spill; charged per copy it fires."""
    bufs = _fanout(12000.0, same_view=True)
    assert sum(isinstance(b, RelayoutCopyBuffer) for b in bufs) == 1
    r = _solve(engine, bufs)
    copy = _copy(r)
    assert r["P"].address is not None, "one shared shuffle beats the spill"
    assert (copy.start_time, copy.end_time) == (1, 3) and copy.address is not None
    a1, a2 = r["C1"].chosen_relayouts["P"], r["C2"].chosen_relayouts["P"]
    assert a1.candidate.group == a2.candidate.group == 0
    assert a1.destination_address == a2.destination_address == copy.address
    assert _disjoint(copy.address, r["P"].address)


def test_two_consumers_on_different_views_get_two_copies(engine):
    """Distinct destination views cannot share: two copies, two charges. At
    12000 each the pair (24000) loses to the 20000 spill, so P is spilled;
    at 8000 each (16000) both fire, each copy alive at its own tick."""
    r = _solve(engine, _fanout(12000.0, same_view=False))
    assert r["P"].address is None
    assert _copy(r, 0).address is None and _copy(r, 1).address is None
    assert r["C1"].chosen_relayouts == {} == r["C2"].chosen_relayouts

    r = _solve(engine, _fanout(8000.0, same_view=False))
    assert r["P"].address is not None
    assert _copy(r, 0).address is not None and _copy(r, 1).address is not None
    assert (
        r["C1"].chosen_relayouts["P"].candidate.group == 0
        and r["C2"].chosen_relayouts["P"].candidate.group == 1
    )
    assert r["C1"].chosen_relayouts["P"].destination_address == _copy(r, 0).address
    assert r["C2"].chosen_relayouts["P"].destination_address == _copy(r, 1).address


def _distant(relayout_cost_ns, blocker_size):
    """P (uses [0, 10]) feeds C1 at tick 1 and C2 at tick 8, same destination
    view, so the copy lives [1, 9). B (uses [3, 7], no parents) needs
    ``blocker_size`` bytes of LX in the gap. With a 64-byte LX, P's slice (16)
    plus the held copy (16) leave 32: a 40-byte B fits only if the copy is
    not held."""
    return _with_copies(
        _producer([0, 10]),
        _consumer("C1", 1, 2, [_candidate("C1", 0, relayout_cost_ns)]),
        _consumer("C2", 8, 9, [_candidate("C2", 0, relayout_cost_ns)]),
        CoreDivisionBuffer(
            "B",
            blocker_size,
            [3, 7],
            core_divisions=[CoreDivision(splits={1: 1})],
        ),
    )


def test_one_copy_per_group_spans_the_gap_or_the_source_spills(engine):
    """The model holds ONE copy across a group's consumers and never
    re-shuffles a released view. Holding it through B spills B; the solver
    weighs that against spilling P. Dear B (30000): P spills (20000) and B
    stays. Cheap B (1000): the copy is held, both consumers read it, B
    spills."""
    r = _solve(engine, _distant(5000.0, blocker_size=40), B=30000.0)
    assert r["P"].address is None and r["B"].address is not None
    assert _copy(r).address is None
    assert r["C1"].chosen_relayouts == {} == r["C2"].chosen_relayouts

    r = _solve(engine, _distant(5000.0, blocker_size=40), B=1000.0)
    copy = _copy(r)
    assert r["P"].address is not None and r["B"].address is None
    assert (copy.start_time, copy.end_time) == (1, 9) and copy.address is not None
    a1, a2 = r["C1"].chosen_relayouts["P"], r["C2"].chosen_relayouts["P"]
    assert a1.destination_address == a2.destination_address == copy.address


def test_copy_spans_a_consumer_outside_the_group(engine):
    """C2 (tick 4) reads P through a free slicing MATCH and offers no relayout
    candidate, so it is not a consumer of the copy at all; C1 (tick 1) and C3
    (tick 7) must relayout. One copy spans over C2 in time and serves both,
    while C2 keeps reading the source directly."""
    bufs = _with_copies(
        _producer([0, 9]),
        _consumer("C1", 1, 2, [_candidate("C1", 0, 5000.0)]),
        _consumer("C2", 4, 5, [], matches=[(0, 0)]),
        _consumer("C3", 7, 8, [_candidate("C3", 0, 5000.0)]),
    )
    r = _solve(engine, bufs)
    copy = _copy(r)
    assert copy.consumers == ("C1", "C3")
    assert r["P"].address is not None and copy.address is not None
    assert r["C2"].chosen_relayouts == {}
    assert (
        r["C1"].chosen_relayouts["P"].destination_address
        == r["C3"].chosen_relayouts["P"].destination_address
        == copy.address
    )


def test_charge_follows_the_chosen_source_division(engine):
    """P offers two divisions whose shuffles into the same destination view
    price differently (5000 vs 3000 ns). The solver picks the cheaper source
    division, and the RelayoutCharge term charges that price: with the
    spill at 4000 only the cheaper division makes the relayout worth it."""
    bufs = _with_copies(
        _producer([0, 3], divisions=2),
        _consumer("C", 1, 2, [_candidate("C", 0, 5000.0), _candidate("C", 1, 3000.0)]),
    )
    r = _solve(engine, bufs, P=4000.0)
    assert r["P"].address is not None and r["P"].chosen_division == 1
    chosen = r["C"].chosen_relayouts["P"]
    assert chosen.candidate == _candidate("C", 1, 3000.0), "cheaper division wins"
    assert chosen.destination_address == _copy(r).address


def test_a_consumer_reads_the_copy_only_under_a_priced_pair(engine):
    """P has two divisions but only division 0 is priced into C's view, and
    that shuffle (50000) is dearer than the spill. Division 1 is unpriced, so
    the served literal cannot pin it: P must spill rather than read the copy
    under a pair the candidates never listed."""
    bufs = _with_copies(
        _producer([0, 3], divisions=2),
        _consumer("C", 1, 2, [_candidate("C", 0, 50000.0)]),
    )
    r = _solve(engine, bufs)
    assert r["P"].address is None and r["C"].chosen_relayouts == {}
    assert _copy(r).address is None


def _chain(relayout_cost_ns):
    """P (uses [0, 1]) -> M (uses [1, 2]) -> C (uses [2, 3]), no slicing match
    on either edge, so P and M reach LX only by relaying out. P -> M is priced
    only into M's division 1; M -> C lands on one view of M, priced from either
    of M's divisions. Both links fire only with M at division 1: M is the
    consumer of one copy and the source of the next."""
    m = CoreDivisionBuffer(
        "M",
        64,
        [1, 2],
        core_divisions=[CoreDivision(splits={1 - d: 4}) for d in range(2)],
        parents=["P"],
        cd_parent_matches={"P": []},
        cd_parent_relayouts={"P": [_candidate("M", 0, relayout_cost_ns, j=1)]},
    )
    c = CoreDivisionBuffer(
        "C",
        64,
        [2, 3],
        core_divisions=[CoreDivision(splits={1: 4})],
        parents=["M"],
        cd_parent_matches={"M": []},
        cd_parent_relayouts={
            "M": [
                _candidate("C", 0, relayout_cost_ns, parent="M"),
                _candidate("C", 1, relayout_cost_ns, parent="M"),
            ]
        },
    )
    return _with_copies(_producer([0, 1]), m, c)


def test_a_chain_of_relayouts_fires_both_links(engine):
    """Spilling P or M costs 20000 each, a shuffle 1000: both links fire, with
    M at the one division both are priced for."""
    r = _solve(engine, _chain(1000.0), M=20000.0)
    assert r["P"].address is not None and r["M"].address is not None
    assert r["M"].chosen_division == 1
    assert r["M"].chosen_relayouts["P"].candidate.consumer_division == 1
    assert r["C"].chosen_relayouts["M"].candidate.source_division == 1
    assert _copy(r).address is not None
    assert r[relayout_copy_name("M", 0)].address is not None


# ---------------------------------------------------------------------------
# The annealer's half of the coupling
# ---------------------------------------------------------------------------


def _primed_annealer(buffers, size=64, expr="default"):
    """An annealer primed to its seed state (every copy switched off), so a
    test can drive the relayout machinery directly."""
    if expr == "default":
        expr = _objective(buffers)
    solver = SaCoOptimizingSolver(buffers, size, alignment=1)
    solver._score_fn = solver._build_score_fn(expr)
    solver._rng = random.Random(0)
    solver._precompute_topology()
    solver.chosen = solver._seed_configs()
    solver.packer = solver._build_seed_packer()
    solver._flippable_ops = solver._flippable()
    return solver


def _index(solver, name) -> int:
    return [b.name for b in solver._bufs].index(name)


def _assert_eligible_count(solver):
    assert solver._n_eligible == sum(
        solver._eligible(i) for i in range(len(solver._bufs))
    ), "the incremental eligible count drifted from a recount"
    assert list(solver._eligible_mask) == [
        solver._eligible(i) for i in range(len(solver._bufs))
    ], "the eligibility mask drifted from a recount"
    assert solver._eligible_copies == {
        c for c in solver._relayout_source if solver._eligible(c)
    }, "the tracked eligible copies drifted from a recount"


def test_annealer_seeds_with_every_copy_off():
    """Off, a copy is ineligible, and so is a source only it could serve: the
    seed is the plan without relayouts, which best-seen never does worse than."""
    solver = _primed_annealer(_pc(5000.0))
    copy = _index(solver, relayout_copy_name("P", 0))
    assert solver._relayout_on == set()
    assert not solver._eligible(copy) and not solver._eligible(_index(solver, "P"))
    solver._toggle_relayout(copy)
    assert solver._eligible(copy) and solver._eligible(_index(solver, "P"))
    _assert_eligible_count(solver)


def test_annealer_drops_a_placed_source_whose_copy_did_not_fit():
    """The placement half of the coupling. With 16 bytes of LX, P's per-core
    slice fills it at C's tick, so the packer, placing P first, cannot fit the
    copy C must read P through. Resident P would be read by C in the wrong
    layout, so the resident set drops it, and the score and the write-back
    both see it spilled."""
    solver = _primed_annealer(_pc(5000.0), size=16)
    p, copy = _index(solver, "P"), _index(solver, relayout_copy_name("P", 0))
    solver._toggle_relayout(copy)
    solver.packer.rotate(list(solver.packer.permutation).index(p), 0)
    placed = solver.packer.addresses
    assert placed[p] is not None and placed[copy] is None
    resident = solver._resident_addresses()
    assert resident[p] is None and resident[copy] is None
    solver._write_back()
    assert solver._bufs[p].address is None
    assert solver._bufs[_index(solver, "C")].chosen_relayouts == {}


def test_annealer_logs_what_the_coupling_dropped(caplog):
    """A buffer the packer placed but the plan drops held LX the packer gave
    no one else: the write-back says so at debug level, the signal to look for
    before teaching the packer the coupling."""
    solver = _primed_annealer(_pc(5000.0), size=16)
    p, copy = _index(solver, "P"), _index(solver, relayout_copy_name("P", 0))
    solver._toggle_relayout(copy)
    solver.packer.rotate(list(solver.packer.permutation).index(p), 0)
    with caplog.at_level(logging.DEBUG, logger=sa_logger.name):
        solver._write_back()
    assert "relayout coupling dropped 1 placed buffer(s), 16 B of LX: P" in caplog.text


def test_annealer_drops_a_placed_copy_whose_source_did_not_fit():
    """The other direction: packed first, the copy fills the 16 bytes at C's
    tick, which P is live through, so P is left out. A copy placed without its
    source is a shuffle from nowhere."""
    solver = _primed_annealer(_pc(5000.0), size=16)
    p, copy = _index(solver, "P"), _index(solver, relayout_copy_name("P", 0))
    solver._toggle_relayout(copy)
    solver.packer.rotate(list(solver.packer.permutation).index(copy), 0)
    solver.packer.rotate(
        list(solver.packer.permutation).index(p), len(solver._bufs) - 1
    )
    placed = solver.packer.addresses
    assert placed[copy] is not None and placed[p] is None
    assert solver._resident_addresses()[copy] is None


def test_annealer_snapshot_carries_the_copies_switched_on():
    """Nothing else can rebuild which copies are on, so a rejected move must
    restore it with the rest of the state."""
    solver = _primed_annealer(_pc(5000.0))
    copy = _index(solver, relayout_copy_name("P", 0))
    snap = solver._snapshot()
    solver._toggle_relayout(copy)
    assert copy in solver._relayout_on
    solver._adopt(snap)
    assert solver._relayout_on == set()
    _assert_eligible_count(solver)


def test_annealer_offers_no_relayout_under_the_fallback_objective():
    solver = _primed_annealer(_pc(5000.0), expr=None)
    assert "relayout" not in solver._applicable_moves()
    copy = _index(solver, relayout_copy_name("P", 0))
    solver._toggle_relayout(copy)
    assert not solver._eligible(copy) and not solver._eligible(_index(solver, "P"))


def test_annealer_relayout_move_buys_the_group_with_its_divisions():
    """Switching a copy on alone pays only if the divisions already sit on one
    of its candidates' pairs. The move sets them too: P is seeded at division 0
    and only division 1 is priced into C's view, so switching the copy on moves
    P to division 1 and makes both eligible in one step."""
    bufs = _with_copies(
        _producer([0, 3], divisions=2),
        _consumer("C", 1, 2, [_candidate("C", 1, 3000.0)]),
    )
    solver = _primed_annealer(bufs)
    p, copy = _index(solver, "P"), _index(solver, relayout_copy_name("P", 0))
    assert solver.chosen[p].menu_index == 0
    solver._relayout()
    assert copy in solver._relayout_on
    assert solver.chosen[p].menu_index == 1
    assert solver._eligible(copy) and solver._eligible(p)
    _assert_eligible_count(solver)
    solver._relayout()  # the only copy, drawn again: switched back off
    assert solver._relayout_on == set() and not solver._eligible(p)
    _assert_eligible_count(solver)


def test_annealer_relayout_move_keeps_a_link_another_copy_is_priced_for():
    """With P -> M switched on (M at division 1), switching on M's copy must
    not move M off division 1, although that copy is priced from M's division
    0 too: that would unprice the first link, so a chain could only ever trade
    one link for the other. Every draw must keep it."""
    solver = _primed_annealer(_chain(1000.0))
    m = _index(solver, "M")
    first = _index(solver, relayout_copy_name("P", 0))
    second = _index(solver, relayout_copy_name("M", 0))
    solver._toggle_relayout(first)
    solver._apply_assignment(solver._relayout_assignment(first))
    assert solver.chosen[m].menu_index == 1
    for seed in range(20):
        solver._rng = random.Random(seed)
        assignment = solver._relayout_assignment(second)
        assert assignment[m].menu_index == 1, "the anchored division moved"


def _paying_chain_annealer():
    """The chain, with spilling M priced as spilling P is: each link's
    shuffle (1000) is then worth its source's residency (20000)."""
    bufs = _chain(1000.0)
    return _primed_annealer(bufs, expr=_objective(bufs, {"M": _SPILL_NS}))


def test_annealer_relayout_move_buys_the_seams_it_opens():
    """Switching on P -> M moves M to division 1, which no longer matches C:
    the move switches on the copy that pair names in the same step, and keeps
    it because it pays, so P, M and both copies are eligible at once."""
    solver = _paying_chain_annealer()
    first = _index(solver, relayout_copy_name("P", 0))
    second = _index(solver, relayout_copy_name("M", 0))
    solver._rng.choice = lambda options: first if second in options else options[0]
    solver._relayout()
    assert {first, second} <= solver._relayout_on
    for name in ("P", "M"):
        assert solver._eligible(_index(solver, name)), name
    assert solver._eligible(first) and solver._eligible(second)
    _assert_eligible_count(solver)


class _CountingPacker:
    """The packer, counting its re-places."""

    def __init__(self, packer):
        self._packer = packer
        self.updates = 0

    def update(self, *args):
        self.updates += 1
        return self._packer.update(*args)

    def __getattr__(self, name):
        return getattr(self._packer, name)


def test_annealer_relayout_move_is_one_re_place():
    """The divisions, the drawn copy and the seam it opens all land in one
    packer update: a re-place is O(n), and one per copy would cost a recolor
    whose boundary opens k seams k + 1 of them. (Judging the seams afterwards
    costs one more, however many there are.)"""
    solver = _paying_chain_annealer()
    first = _index(solver, relayout_copy_name("P", 0))
    second = _index(solver, relayout_copy_name("M", 0))
    solver.packer = _CountingPacker(solver.packer)
    seams = solver._apply_assignment(
        solver._relayout_assignment(first), buy_seams=True, switch_on=(first,)
    )
    assert seams == {second} and {first, second} <= solver._relayout_on
    assert solver.packer.updates == 1
    _assert_eligible_count(solver)


@pytest.mark.parametrize("expr", ["default", None])
def test_annealer_buys_seams_only_under_the_cost_objective(expr):
    """The memory-only fallback has no price for a shuffle, so a division
    change there leaves every copy off rather than switching on copies that can
    never be admitted. Under the cost objective the divisions that price
    P -> M open both edges' seams, so both copies are bought."""
    solver = _primed_annealer(_chain(1000.0), expr=expr)
    first = _index(solver, relayout_copy_name("P", 0))
    second = _index(solver, relayout_copy_name("M", 0))
    solver._apply_assignment(solver._relayout_assignment(first), buy_seams=True)
    assert solver._relayout_on == (set() if expr is None else {first, second})
    _assert_eligible_count(solver)


def _sibling(relayout_cost_ns):
    """P feeds C1 and C2. C1 has two divisions and both match P; C2 matches
    no division of P and reads it only through a copy priced
    ``relayout_cost_ns``. Moving C1 changes no edge into C2."""
    c1 = CoreDivisionBuffer(
        "C1",
        64,
        [1, 2],
        core_divisions=[CoreDivision(splits={1: 4}), CoreDivision(splits={1: 2})],
        parents=["P"],
        cd_parent_matches={"P": [(0, 0), (0, 1)]},
    )
    c2 = _consumer("C2", 2, 3, [_candidate("C2", 0, relayout_cost_ns)])
    return _with_copies(_producer([0, 3]), c1, c2)


def test_annealer_sibling_move_keeps_a_switched_off_copy_off():
    """A flip of C1 changes P -> C1 only. The P -> C2 copy, off because it costs
    more than spilling P, must stay off: bought again, it would make P eligible
    and charge the flip a shuffle on an edge it did not change."""
    solver = _primed_annealer(_sibling(50000.0))
    p, c1 = _index(solver, "P"), _index(solver, "C1")
    copy = _index(solver, relayout_copy_name("P", 0))
    assert solver.chosen[c1].menu_index == 0 and copy not in solver._relayout_on
    seams = solver._atomic_flip(c1, solver._menus[c1].configs[1], buy_seams=True)
    assert seams == set() and copy not in solver._relayout_on
    assert not solver._eligible(p)
    _assert_eligible_count(solver)


def _seam(relayout_cost_ns):
    """P feeds C, which matches P at its division 0 and reads it through a copy
    priced ``relayout_cost_ns`` at its division 1."""
    c = CoreDivisionBuffer(
        "C",
        64,
        [1, 2],
        core_divisions=[CoreDivision(splits={1: 4}), CoreDivision(splits={0: 4})],
        parents=["P"],
        cd_parent_matches={"P": [(0, 0)]},
        cd_parent_relayouts={"P": [_candidate("C", 0, relayout_cost_ns, j=1)]},
    )
    return _with_copies(_producer([0, 3]), c)


@pytest.mark.parametrize("relayout_cost_ns, kept", [(5000.0, True), (50000.0, False)])
def test_annealer_keeps_a_seam_only_where_it_pays(relayout_cost_ns, kept):
    """Flipping C to division 1 opens the P -> C seam, and the flip buys its
    copy. The move is then judged at the better of its two plans: the shuffle
    where it costs less than spilling P (20000), the spill where it costs more."""
    solver = _primed_annealer(_seam(relayout_cost_ns))
    p, c = _index(solver, "P"), _index(solver, "C")
    copy = _index(solver, relayout_copy_name("P", 0))
    seams = solver._atomic_flip(c, solver._menus[c].configs[1], buy_seams=True)
    assert seams == {copy}
    with_seam = solver._score()
    solver._settle(seams)
    assert (copy in solver._relayout_on) is kept
    assert solver._eligible(p) is kept
    _assert_eligible_count(solver)
    spill = utils.to_fixed_us(_SPILL_NS / 1000)
    assert solver._score() == min(with_seam, spill)


@pytest.mark.parametrize("relayout_weight", [None, 0.0])
@pytest.mark.parametrize("seed", range(5))
def test_annealer_moves_a_sibling_of_a_spilled_source(
    monkeypatch, seed, relayout_weight
):
    """The best plan spills P (20000 against a 50000 shuffle) and moves C1 to
    its division 1, which saves 5000. Moving C1 must not buy the P -> C2 copy
    back: with the relayout move off, nothing else could switch it off again,
    and the search would stop at the seed. LX holds everything, so the packer
    cannot make the call for it by leaving the copy out."""
    monkeypatch.setattr(sa_cooptimizer, "_SEED", seed)
    if relayout_weight is not None:
        monkeypatch.setitem(sa_cooptimizer._MOVE_WEIGHTS, "relayout", relayout_weight)
    bufs = _sibling(50000.0)
    c1 = next(b for b in bufs if b.name == "C1")
    expr = _objective(bufs) + 5000 * (1 - c1.sym_division)
    r = _solve("annealer", bufs, size=1024, expr=expr)
    assert r["C1"].chosen_division == 1
    assert r["P"].address is None and _copy(r).address is None
    assert r["C2"].chosen_relayouts == {}


def test_annealer_never_draws_a_copy_whose_source_is_pinned_out():
    """A copy whose source can never be resident can never fire: the relayout
    move must not draw it, so it does not move divisions for nothing."""
    bufs = _pc(5000.0)
    next(b for b in bufs if b.name == "P").residency_reason = "pinned to HBM"
    solver = _primed_annealer(bufs)
    assert solver._relayout_source and solver._relayout_copy_list == []
    assert "relayout" not in solver._applicable_moves()


def test_annealer_relayout_move_leaves_a_consumer_the_copy_cannot_serve():
    """One copy (group 0) with candidates for C, which reads P, and for D,
    which is in the solve but does not list P as a parent. The relayout move
    sets the divisions that make the copy serve: P's and C's, never D's, since
    no copy reads for D."""
    c = _consumer("C", 1, 2, [_candidate("C", 0, 5000.0)])
    d = CoreDivisionBuffer(
        "D",
        64,
        [2, 3],
        core_divisions=[CoreDivision(splits={1 - k: 4}) for k in range(2)],
        cd_parent_relayouts={"P": [_candidate("D", 0, 5000.0, j=1)]},
    )
    s = _primed_annealer(_with_copies(_producer([0, 3]), c, d))
    copy = _index(s, relayout_copy_name("P", 0))
    assert s._relayout_consumers[copy] == [_index(s, "C")]
    for _ in range(20):
        assert set(s._relayout_assignment(copy)) == {_index(s, "P"), _index(s, "C")}


def _unread_copy():
    """P and C with one copy whose only candidate names the edge P -> C, which
    the solve does not carry: C does not list P as a parent."""
    reader = CoreDivisionBuffer(
        "C",
        64,
        [1, 2],
        core_divisions=[CoreDivision(splits={1: 4})],
        cd_parent_relayouts={"P": [_candidate("C", 0, 5000.0)]},
    )
    return _with_copies(_producer([0, 1]), reader)


def test_annealer_never_draws_a_copy_no_edge_in_the_solve_reads():
    """A copy whose candidates name no edge the solve carries can never serve.
    The relayout move must not draw it (it would have no division to set), and
    the eligibility ripple must not look up a relation the edge does not have."""
    bufs = _unread_copy()
    solver = _primed_annealer(bufs)
    copy = _index(solver, relayout_copy_name("P", 0))
    assert solver._relayout_consumers[copy] == []
    assert solver._relayout_copy_list == []
    assert "relayout" not in solver._applicable_moves()
    r = _solve("annealer", bufs)
    assert r[relayout_copy_name("P", 0)].address is None


def _crowded(consumers=3):
    """P feeds C0..C2 one tick each, each through a copy of its own (group k,
    of the three views a 4-core source can move to) and none matching."""
    return _with_copies(
        _producer([0, consumers + 2]),
        *(
            _consumer(f"C{k}", k + 1, k + 2, [_candidate(f"C{k}", 0, 5000.0, k)])
            for k in range(consumers)
        ),
    )


def _copy_indices(solver) -> set[int]:
    return {i for i, b in enumerate(solver._bufs) if isinstance(b, RelayoutCopyBuffer)}


def test_annealer_seed_pi_puts_every_copy_after_the_buffers():
    """Every copy is off in the seed, so no address depends on it. Its slot in
    pi must not shape the buffers' order either: the seed orders the buffers
    as the solve without copies does, and the copies after them."""
    bufs = _crowded()
    solver = _primed_annealer(bufs, size=1024)
    copies = _copy_indices(solver)
    perm = list(solver.packer.permutation)
    head = len(perm) - len(copies)
    assert set(perm[head:]) == copies
    without = _primed_annealer(
        [b for b in bufs if not isinstance(b, RelayoutCopyBuffer)], size=1024
    )
    assert [solver._bufs[i].name for i in perm[:head]] == [
        without._bufs[i].name for i in without.packer.permutation
    ]


def test_annealer_layout_moves_draw_no_copy_that_is_off():
    """A copy that is off is transparent to the packer, so a layout move on it
    moves nothing. Neither the rotate draws nor the reinsertion source may land
    on one; a copy joins the draws once it is eligible."""
    solver = _primed_annealer(_crowded(), size=1024)
    copies = _copy_indices(solver)
    perm = solver.packer.permutation
    allocated = [solver.packer.is_fully_allocated(perm[k]) for k in range(len(perm))]
    for _ in range(300):
        assert perm[solver._draw_layout_position()] not in copies
        assert perm[solver._choose_reinsertion_source(allocated)] not in copies

    solver._switch_relayouts(copies, on=True)
    assert solver._eligible_copies == copies
    drawn = {perm[solver._draw_layout_position()] for _ in range(300)}
    assert drawn == set(range(len(solver._bufs)))


def test_annealer_sweep_bound_hides_no_layout_past_the_last_drawn_occupant():
    """Past the last occupant a layout move draws, only copies that are off
    remain, so reinserting anywhere there lands the layout the bound does."""
    solver = _primed_annealer(_crowded(), size=1024)
    copies = sorted(_copy_indices(solver))
    # Scatter the copies through pi, then switch half of them off again.
    solver._switch_relayouts(set(copies), on=True)
    for _ in range(40):
        solver.packer.rotate(
            solver._rng.randrange(len(solver._bufs)),
            solver._rng.randrange(len(solver._bufs)),
        )
    solver._switch_relayouts(set(copies[::2]), on=False)
    n = len(solver._bufs)
    for i in solver._layout_positions():
        perm = solver.packer.permutation
        allocated = [solver.packer.is_fully_allocated(perm[k]) for k in range(n)]
        upper = solver._sweep_upper_bound(i, allocated)
        snap = solver._snapshot()
        solver.packer.rotate(i, upper)
        bound = list(solver.packer.addresses)
        solver._adopt(snap)
        for j in range(upper + 1, n):
            snap = solver._snapshot()
            solver.packer.rotate(i, j)
            assert list(solver.packer.addresses) == bound, (i, j, upper)
            solver._adopt(snap)


def test_a_solve_replaces_chosen_relayouts(engine):
    """Every consumer's ``chosen_relayouts`` holds only this solve's reads: an
    entry left on the buffer from before is dropped, not merged."""
    bufs = _pc(50000.0)
    c = next(b for b in bufs if b.name == "C")
    c.chosen_relayouts = {
        "Q": ChosenRelayout(_candidate("C", 0, 5000.0, parent="Q"), 16)
    }
    r = _solve(engine, bufs)
    assert r["C"].chosen_relayouts == {}


@pytest.mark.parametrize(
    "bufs, spill",
    [
        (lambda: _chain(1000.0), {"M": 20000.0}),
        (lambda: _fanout(8000.0, same_view=False), {}),
        (lambda: _distant(5000.0, blocker_size=40), {"B": 1000.0}),
    ],
)
def test_annealer_state_stays_consistent_through_a_full_anneal(bufs, spill):
    """After a whole anneal (moves, rejections restored from snapshots, the best
    state adopted), the incremental eligible count and the tracked eligible
    copies must still equal a recount."""
    buffers = bufs()
    solver = SaCoOptimizingSolver(buffers, 64, alignment=1)
    solver.plan_layout_and_core_divisions(_objective(buffers, spill))
    _assert_eligible_count(solver)


def test_annealer_switches_off_a_copy_a_division_change_strands():
    """A switched-on copy no consumer reads through after a division change is
    switched off with that change: it is ineligible anyway, and carrying it on
    would drag it through every later ripple. Returning the divisions does not
    revive it; a relayout move would buy it again."""
    solver = _primed_annealer(_chain(1000.0))
    m = _index(solver, "M")
    first = _index(solver, relayout_copy_name("P", 0))
    solver._toggle_relayout(first)
    solver._apply_assignment(solver._relayout_assignment(first))
    assert first in solver._relayout_on and solver.chosen[m].menu_index == 1
    solver._atomic_flip(m, solver._menus[m].configs[0])
    assert first not in solver._relayout_on
    _assert_eligible_count(solver)
    solver._atomic_flip(m, solver._menus[m].configs[1])
    assert first not in solver._relayout_on
    _assert_eligible_count(solver)


def _two_way(name, uses, parents=(), matches=None, relayouts=None):
    """A buffer with two 4-way divisions (0 splits dim 1, 1 dim 0)."""
    return CoreDivisionBuffer(
        name,
        64,
        list(uses),
        core_divisions=[CoreDivision(splits={1 - d: 4}) for d in range(2)],
        parents=list(parents),
        cd_parent_matches=matches or {},
        cd_parent_relayouts=relayouts or {},
    )


def _mixed():
    """A chain and a fan-out over one source, with cheap and dear copies:
    P -> M -> C relays out twice (P -> M cheap, M -> C dear), P -> C1 matches
    at (0, 0) and relays out cheaply from P's division 1, and P -> C2 matches
    nowhere and relays out dearly. Every buffer has two divisions, so every
    move applies."""
    return _with_copies(
        _two_way("P", [0, 4]),
        _two_way(
            "M",
            [1, 2],
            ["P"],
            {"P": []},
            {"P": [_candidate("M", 0, 1000.0, 0, j=1)]},
        ),
        _two_way(
            "C",
            [2, 3],
            ["M"],
            {"M": []},
            {
                "M": [
                    _candidate("C", i, 50000.0, parent="M", j=j)
                    for i in (0, 1)
                    for j in (0, 1)
                ]
            },
        ),
        _two_way(
            "C1",
            [1, 2],
            ["P"],
            {"P": [(0, 0)]},
            {"P": [_candidate("C1", 1, 1000.0, 1)]},
        ),
        _two_way(
            "C2",
            [3, 4],
            ["P"],
            {"P": []},
            {
                "P": [
                    _candidate("C2", i, 50000.0, 2, j=j) for i in (0, 1) for j in (0, 1)
                ]
            },
        ),
    )


def _resident_oracle(s) -> set[int]:
    """The coupling read directly off the graph, without the tracked eligible
    copies: placed, minus each source a mismatched consumer cannot read
    through a placed copy, minus each copy whose source is then not
    resident."""
    resident = set(s.packer.allocated_indices())
    for source in list(resident):
        for consumer, relation in s._children[source]:
            if relation.compatible(s.chosen[source], s.chosen[consumer]):
                continue
            read = s._relayout_read(source, consumer)
            if read is None or read[0] not in resident:
                resident.discard(source)
                break
    return {
        i
        for i in resident
        if i not in s._relayout_source or s._relayout_source[i] in resident
    }


def _score_oracle(s, expr, resident: set[int]) -> int:
    """``expr`` evaluated as ``solved_bindings`` binds a solved plan, from the
    live state: the division index looked up in the buffer's own menu."""
    bindings = {}
    for i, b in enumerate(s._bufs):
        bindings[b.sym_is_lx] = int(i in resident)
        division = s.chosen[i].division
        bindings[b.sym_division] = (
            b.core_divisions.index(division) if division in b.core_divisions else -1
        )
        for key, sym in b.sym_core_divs.items():
            bindings[sym] = division.splits.get(key, 1)
        for key, sym in b.sym_tile_counts.items():
            bindings[sym] = division.tiling.count_for(key)
    return utils.to_fixed_us(max(0.0, float(expr.xreplace(bindings))) / 1000.0)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("size", [40, 1024])
def test_annealer_tracking_survives_random_moves_and_restores(seed, size):
    """Random moves, each kept or restored at random, checked after every step
    against references that share no state with the tracking: the eligibility
    recount, the coupling read off the graph, and the objective evaluated
    directly. A drift then fails at the move that caused it, not at the end of
    a long anneal."""
    bufs = _mixed()
    expr = _objective(bufs, {"M": 20000.0, "C1": 3000.0, "C2": 3000.0})
    s = _primed_annealer(bufs, size=size, expr=expr)
    s._rng = random.Random(seed)
    s._flippable_ops = s._flippable()
    coin = random.Random(1000 + seed)
    cur = s._score()
    # Best-seen, as _anneal starts it: the reorder step updates it.
    s._best_score, s._best_snap = cur, s._snapshot()
    for step in range(300):
        name = s._choose_move()
        if name == "reorder":
            cur = s._step(name, coin.choice([1e-9, 1e9]), cur)
        else:
            snap = s._snapshot()
            s._execute_move(name)
            if coin.random() < 0.5:
                s._adopt(snap)
            cur = s._score()
        tag = f"seed={seed} size={size} step={step} move={name}"
        _assert_eligible_count(s)
        resident = _resident_oracle(s)
        assert s._resident_indices() == resident, tag
        companions = s._companions(resident)
        assert s._score() == _score_oracle(s, expr, resident) + companions, tag
    s._check_tracked_eligibility()


def test_annealer_write_back_raises_on_drifted_eligibility():
    """A drift the moves left behind fails the solve loudly, naming the
    buffer, before anything is committed; ``python -O`` must not strip it."""
    s = _primed_annealer(_mixed(), size=1024)
    p = _index(s, "P")
    s._eligible_mask[p] = not s._eligible_mask[p]
    before = [(b.chosen_division, b.address) for b in s._bufs]
    with pytest.raises(RuntimeError, match="drifted from a recount: \\['P'\\]"):
        s._write_back()
    assert [(b.chosen_division, b.address) for b in s._bufs] == before


def test_annealer_restore_rewinds_the_relayout_state():
    """A restore puts back the copies switched on and eligible with the rest
    of the joint state: nothing else can rebuild them."""
    s = _primed_annealer(_mixed(), size=1024)
    m = _index(s, "M")
    before = _live_state(s)
    snap = s._snapshot()
    s._atomic_flip(m, s._menus[m].configs[1])
    s._switch_relayouts(
        {_index(s, relayout_copy_name("P", g)) for g in (0, 2)}, on=True
    )
    assert s._eligible_copies and _live_state(s) != before
    s._adopt(snap)
    assert _live_state(s) == before


def test_annealer_anchors_only_for_copies_that_are_eligible():
    """In ``_mixed``, P -> M's copy can be on while P is ineligible: P -> C2
    mismatches and C2's copy is off. That copy delivers nothing, so it must
    not anchor P and M; once C2's copy is on too, P is eligible and both
    copies anchor their ends."""
    s = _primed_annealer(_mixed(), size=1024)
    p, m, c2 = (_index(s, n) for n in ("P", "M", "C2"))
    to_m = _index(s, relayout_copy_name("P", 0))
    to_c2 = _index(s, relayout_copy_name("P", 2))
    s._atomic_flip(m, s._menus[m].configs[1])  # M onto the pair P -> M prices
    s._switch_relayouts({to_m}, on=True)
    assert to_m in s._relayout_on and to_m not in s._eligible_copies
    assert s._relayout_anchored(exclude=-1) == set()

    s._switch_relayouts({to_c2}, on=True)
    assert {to_m, to_c2} <= s._eligible_copies
    assert s._relayout_anchored(exclude=-1) == {p, m, c2}
    assert s._relayout_anchored(exclude=to_c2) == {p, m}


def test_annealer_scores_relayout_charges_as_the_expression_does():
    """The annealer sums the copies' RelayoutCharge terms itself instead of
    compiling them. Over every division of M and every resident set, its score
    must equal the whole objective evaluated directly."""
    import itertools

    bufs = _chain(1000.0)
    expr = _objective(bufs, {"M": 20000.0})
    solver = _primed_annealer(bufs, expr=expr)
    assert solver._score_fn is not None
    m = _index(solver, "M")
    names = [b.name for b in bufs]
    for division in (0, 1):
        chosen = list(solver.chosen)
        chosen[m] = solver._menus[m].configs[division]
        for flags in itertools.product((0, 1), repeat=len(bufs)):
            resident = frozenset(n for n, flag in zip(names, flags) if flag)
            bindings = {b.sym_is_lx: int(b.name in resident) for b in bufs}
            bindings.update(
                {b.sym_division: chosen[i].menu_index for i, b in enumerate(bufs)}
            )
            for i, b in enumerate(bufs):
                for key, sym in b.sym_core_divs.items():
                    bindings[sym] = chosen[i].splits.get(key, 1)
            direct = float(expr.xreplace(bindings))
            assert solver._score_fn(chosen, resident) == utils.to_fixed_us(
                max(0.0, direct) / 1000.0
            ), (division, sorted(resident))


# ---------------------------------------------------------------------------
# The commit path's view
# ---------------------------------------------------------------------------


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

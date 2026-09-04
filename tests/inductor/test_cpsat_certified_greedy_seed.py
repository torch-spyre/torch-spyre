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

"""Unit tests for the certified greedy fast path inside
``CpSatLayoutSolver.plan_layout()``.

Placement-only ``plan_layout`` optimizes only level 1 of the CP-SAT
lex-solve:

    minimize sum(spill_cost(b) * (1 - in_buffer(b)))

Every ``spill_cost(b) >= 0``, and ``_add_core_division`` pins every
buffer returned by ``record_exclusions()`` to ``in_buffer = 0``. So
the objective is bounded below by

    L = sum(spill_cost(b) for b in buffers if b.name in record_exclusions())

and any plan reaching ``L`` is globally optimal. The fast path runs
greedy on a solver-local copy, evaluates that objective in CP-SAT's
alignment-unit domain, checks representability against CP-SAT's
domain (``_capacity_units``, alignment-aligned offsets, top-of-buffer
within ``_capacity_units * alignment``), and either commits greedy's
placement or falls through to the full CP-SAT solve.

These tests do not measure wall clock. They check:
"""

import copy
import unittest
from unittest.mock import patch

# torch first so torch_spyre's PrivateUse1 backend autoload sees a
# fully-initialised module.
import torch  # noqa: F401

try:
    from ortools.sat.python import cp_model  # noqa: F401

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer


LARGE_SIZE = 1 << 20  # 1 MiB, always enough for the tiny fixtures below
ALIGNMENT = 1


def _mk(name, size, uses, **kwargs):
    return LifetimeBoundBuffer(name, size, uses, **kwargs)


def _placed(buffers):
    return {b.name for b in buffers if b.address is not None}


def _greedy(buffers, size=LARGE_SIZE, alignment=ALIGNMENT):
    return GreedyLayoutSolver(
        copy.deepcopy(list(buffers)), size, alignment
    ).plan_layout()


def _cpsat_only(buffers, size=LARGE_SIZE, alignment=ALIGNMENT):
    """Standalone CP-SAT: bypasses the seed by driving
    ``_plan_layout_generic`` directly, so the objective is what CP-SAT
    alone would have produced on the same input.
    """
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
        CpSatLayoutSolver,
    )

    solver = CpSatLayoutSolver(copy.deepcopy(list(buffers)), size, alignment)
    return list(solver._plan_layout_generic())


def _hybrid(buffers, size=LARGE_SIZE, alignment=ALIGNMENT):
    """The full public path: seed first, CP-SAT fallback.

    Deep-copies so the caller's buffers stay unaddressed and can also be
    passed to :func:`_cpsat_only` (which asserts unplanned inputs)."""
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
        CpSatLayoutSolver,
    )

    return CpSatLayoutSolver(
        copy.deepcopy(list(buffers)), size, alignment
    ).plan_layout()


def _obj_units(buffers, alignment):
    """Evaluate the placement-only CP-SAT objective on a plan, in
    alignment units (the units CP-SAT actually optimises)."""
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
        _hbm_spill_cost,
    )
    from torch_spyre._inductor.scratchpad.plan_solver import ceil_div
    from dataclasses import replace as _replace

    return sum(
        _hbm_spill_cost(_replace(b, size=ceil_div(b.size, alignment)))
        for b in buffers
        if b.address is None
    )


@unittest.skipUnless(_HAS_ORTOOLS, "certified greedy seed is a CP-SAT feature")
class TestCertifiedGreedySeed(unittest.TestCase):
    """The certified greedy fast path inside ``CpSatLayoutSolver.plan_layout``."""

    # -- 1. greedy at exact CP-SAT lower bound skips Solve() --
    def test_lower_bound_greedy_plan_skips_cpsat_solve(self):
        """Three buffers, plenty of capacity, no forced exclusion: greedy
        places all three (objective = 0 = lower bound). ``CpSolver.Solve``
        must never run."""
        buffers = [
            _mk("a", 10, [0, 3]),
            _mk("b", 20, [0, 3]),
            _mk("c", 30, [0, 3]),
        ]
        with patch.object(
            cp_model.CpSolver,
            "Solve",
            side_effect=AssertionError(
                "CP-SAT Solve should not run when greedy certifies"
            ),
        ):
            result = _hybrid(buffers, LARGE_SIZE, ALIGNMENT)
        self.assertEqual(_placed(result), {"a", "b", "c"})
        self.assertEqual(_obj_units(result, ALIGNMENT), 0)

    # -- 2. nonzero greedy objective invokes CP-SAT --
    def test_nonzero_objective_falls_through_to_cpsat(self):
        """The classic ``test_largest_buffer_evicted_when_full`` shape:
        capacity 50 forces exactly one spill; greedy spills the largest
        (obj 60), CP-SAT spills the smallest (obj 20). Seed rejects and
        the returned placement is CP-SAT's optimum."""
        buffers = [
            _mk("a", 10, [0, 3]),
            _mk("b", 20, [0, 3]),
            _mk("c", 30, [0, 3]),
        ]
        capacity = 50
        greedy_only = _greedy(buffers, capacity, ALIGNMENT)
        self.assertEqual(_obj_units(greedy_only, ALIGNMENT), 60)
        hybrid = _hybrid(buffers, capacity, ALIGNMENT)
        cpsat_only = _cpsat_only(buffers, capacity, ALIGNMENT)
        # The invariant that matters: hybrid == standalone CP-SAT.
        self.assertEqual(
            _obj_units(hybrid, ALIGNMENT),
            _obj_units(cpsat_only, ALIGNMENT),
        )
        self.assertEqual(_obj_units(hybrid, ALIGNMENT), 20)

    # -- 3. forced exclusion via residency_reason contributes to L --
    def test_residency_reason_contributes_to_lower_bound(self):
        """A buffer with ``residency_reason`` is in ``record_exclusions``
        and its full spill_cost contributes to L. Greedy places the
        others; hybrid certifies."""
        buffers = [
            _mk("barred", 40, [0, 1], residency_reason="not eligible"),
            _mk("free", 40, [0, 1]),
        ]
        result = _hybrid(buffers, LARGE_SIZE, ALIGNMENT)
        self.assertEqual(_placed(result), {"free"})
        # Both hybrid and standalone cpsat return the same objective.
        cpsat_only = _cpsat_only(buffers, LARGE_SIZE, ALIGNMENT)
        self.assertEqual(
            _obj_units(result, ALIGNMENT),
            _obj_units(cpsat_only, ALIGNMENT),
        )
        # Nonzero: barred spill_cost has to appear somewhere.
        self.assertGreater(_obj_units(result, ALIGNMENT), 0)

    # -- 4. forced exclusion via min_footprint > limit contributes to L --
    def test_min_footprint_over_limit_contributes_to_lower_bound(self):
        """A buffer whose ``min_footprint > limit`` is pinned non-resident
        by ``record_exclusions`` even without ``residency_reason``.
        ``min_footprint`` defaults to ``size``, so a buffer bigger than
        the capacity is the natural fixture."""
        big = _mk("big", 200, [0, 3])
        small = _mk("small", 10, [0, 3])
        capacity = 100  # big.min_footprint = 200 > 100
        result = _hybrid([big, small], capacity, ALIGNMENT)
        cpsat_only = _cpsat_only([big, small], capacity, ALIGNMENT)
        self.assertEqual(_placed(result), {"small"})
        self.assertEqual(
            _obj_units(result, ALIGNMENT),
            _obj_units(cpsat_only, ALIGNMENT),
        )

    # -- 5. zero-spill-cost buffer semantics --
    def test_zero_spill_cost_buffer(self):
        """A single-use graph input has ``read_count = 1``,
        ``first_use_is_read = True`` so ``reads_served = 0`` and
        ``is_intermediate = 0``: ``spill_cost = 0``. Its presence or
        absence in the placed set does not change the objective; the seed
        must still certify when the objective already at the floor."""
        buffers = [
            _mk("live_input", 10, [0], first_use_is_read=True),
            _mk("useful", 20, [0, 3]),
        ]
        result = _hybrid(buffers, LARGE_SIZE, ALIGNMENT)
        cpsat_only = _cpsat_only(buffers, LARGE_SIZE, ALIGNMENT)
        self.assertEqual(
            _obj_units(result, ALIGNMENT),
            _obj_units(cpsat_only, ALIGNMENT),
        )

    # -- 5b. zero-cost non-excluded buffer may remain spilled --
    def test_zero_cost_non_excluded_buffer_may_remain_spilled(self):
        """The certificate bounds the *objective*, not the placement set.
        A non-excluded buffer whose ``spill_cost == 0`` can legally be
        left spilled on a certified plan: it neither raises the sum
        above the forced-spill floor nor lowers it. Regression test
        against the incorrect claim "reaching the floor is equivalent
        to placing every non-excluded buffer" -- a claim that assumes
        strictly positive spill costs.

        Fixture. ``hot`` has ``spill_cost = 40`` (a nonzero-cost
        intermediate with two reads) and lives across [0, 5). ``zero``
        is a single-use graph input (``read_count=1``,
        ``first_use_is_read=True`` => ``spill_cost = 0``) that arrives
        at t=1 while ``hot`` still occupies the entire capacity.
        Greedy places ``hot`` first and cannot fit ``zero``, so
        ``zero`` remains spilled. ``zero`` is not in
        ``record_exclusions()`` (its ``residency_reason`` is unset and
        ``min_footprint == size <= limit``), yet the plan is still
        certifiable: the residency objective evaluates to 0 -- the
        forced-spill floor for this input.
        """
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            CpSatLayoutSolver,
        )

        capacity = 10
        buffers = [
            _mk("hot", 10, [0, 2, 4]),
            _mk("zero", 10, [1], first_use_is_read=True),
        ]

        # Verify the fixture predicts what we're asserting: zero has
        # spill_cost == 0, hot has spill_cost > 0.
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            _hbm_spill_cost,
        )

        self.assertEqual(_hbm_spill_cost(buffers[1]), 0)
        self.assertGreater(_hbm_spill_cost(buffers[0]), 0)

        # Hybrid certifies and returns hot placed, zero spilled.
        result = _hybrid(buffers, capacity, ALIGNMENT)
        placed = _placed(result)
        self.assertEqual(placed, {"hot"})
        # Certificate was valid: objective matches standalone CP-SAT.
        cpsat_only = _cpsat_only(buffers, capacity, ALIGNMENT)
        self.assertEqual(
            _obj_units(result, ALIGNMENT),
            _obj_units(cpsat_only, ALIGNMENT),
        )
        # zero was spilled but was NOT in the forced-spill set: it had
        # no ``residency_reason`` and its ``min_footprint <= limit``.
        # The seed still committed a plan with zero as address=None, so
        # ``spill_reasons`` must fall back to the solver-chose-spill
        # sentinel for zero.
        solver = CpSatLayoutSolver(
            copy.deepcopy(buffers),
            capacity,
            ALIGNMENT,
        )
        solver.plan_layout()
        self.assertIn("zero", solver.spill_reasons)
        forced = dict(
            CpSatLayoutSolver(
                copy.deepcopy(buffers),
                capacity,
                ALIGNMENT,
            ).record_exclusions()
        )
        self.assertNotIn("zero", forced)

    # -- 6. graph-input spill-cost semantics --
    def test_graph_input_spill_cost_semantics(self):
        """A multi-use graph input's spill_cost is
        ``(read_count - 1) * size`` (drop the clone-in read pinning cannot
        avoid). Certificate arithmetic must match CP-SAT's."""
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            _hbm_spill_cost,
        )

        b = _mk("input", 8, [0, 2, 4, 6], first_use_is_read=True)
        # read_count = 4, first_use_is_read discounts 1 => reads_served=3.
        # is_intermediate = not first_use_is_read = False => 0.
        # spill_cost = (3 + 0) * 8 = 24.
        self.assertEqual(_hbm_spill_cost(b), 24)

    # -- 7. intermediate spill-cost semantics --
    def test_intermediate_spill_cost_semantics(self):
        """A computed intermediate's first use is the producing write, so
        ``read_count`` already excludes it and ``is_intermediate = 1``:
        ``spill_cost = (read_count + 1) * size``."""
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            _hbm_spill_cost,
        )

        b = _mk("intermediate", 8, [0, 2, 4], first_use_is_read=False)
        # read_count = len(uses)-1 = 2, reads_served = 2, is_intermediate=1
        # spill_cost = (2 + 1) * 8 = 24.
        self.assertEqual(_hbm_spill_cost(b), 24)

    # -- 8. non-aligned buffer sizes use the same objective as CP-SAT --
    def test_non_aligned_buffer_sizes_match_cpsat_objective(self):
        """CP-SAT wraps every buffer with ``ceil_div(size, alignment)``.
        Certificate must use the same scaling, else the seed can accept a
        plan whose greedy objective differs from what CP-SAT would compute."""
        # size 129, alignment 128 -> ceil_div = 2 unit-sized in CP-SAT.
        buffers = [_mk("odd", 129, [0, 1])]
        alignment = 128
        # Capacity plenty; both solvers place it.
        capacity = 4096
        hybrid = _hybrid(buffers, capacity, alignment)
        cpsat_only = _cpsat_only(buffers, capacity, alignment)
        self.assertEqual(_placed(hybrid), _placed(cpsat_only))
        # Both objectives equal 0 here (only buffer placed). More
        # interesting: force a spill and verify the unit-domain matches.
        buffers2 = [_mk("odd", 129, [0, 1]) for _ in range(3)]
        for i, b in enumerate(buffers2):
            b.name = f"odd_{i}"
        # 3 buffers of 2 units each, capacity 2 units => spill 2.
        capacity_units_2 = 2 * alignment
        h2 = _hybrid(buffers2, capacity_units_2, alignment)
        c2 = _cpsat_only(buffers2, capacity_units_2, alignment)
        self.assertEqual(_obj_units(h2, alignment), _obj_units(c2, alignment))

    # -- 9. non-aligned capacity cannot certify unrepresentable layouts --
    def test_limit_below_alignment_never_certifies(self):
        """When ``_capacity_units = limit // alignment == 0``, CP-SAT has
        no addressable slots; the seed must not certify anything (greedy
        might place a small buffer with byte capacity, but CP-SAT's model
        would place none). Falls through to CP-SAT."""
        buffers = [_mk("a", 3, [0, 1]), _mk("b", 3, [0, 1])]
        limit = 10  # < alignment=128 -> _capacity_units = 0
        alignment = 128
        result = _hybrid(buffers, limit, alignment)
        cpsat_only = _cpsat_only(buffers, limit, alignment)
        # Whatever cpsat does on its own, hybrid must match it.
        self.assertEqual(_placed(result), _placed(cpsat_only))
        self.assertEqual(
            _obj_units(result, alignment),
            _obj_units(cpsat_only, alignment),
        )

    def test_limit_not_divisible_by_alignment_uses_cpsat_top(self):
        """``_capacity_units = limit // alignment`` rounds down, so if
        greedy places a buffer whose top-of-buffer exceeds
        ``_capacity_units * alignment`` the seed must reject."""
        # limit=200, alignment=128 -> _capacity_units = 1, cpsat_top = 128
        # Greedy on byte capacity 200 might place a 130-byte buffer at 0
        # (top=130 > 128), which CP-SAT can't represent.
        buffers = [_mk("a", 130, [0, 1])]
        limit = 200
        alignment = 128
        result = _hybrid(buffers, limit, alignment)
        cpsat_only = _cpsat_only(buffers, limit, alignment)
        self.assertEqual(
            _obj_units(result, alignment),
            _obj_units(cpsat_only, alignment),
        )

    # -- 10. in-place reuse remains legal --
    def test_in_place_reuse_still_certifies(self):
        """In-place reuse: parent and child at the same LX offset for the
        handoff tick. Certificate must accept and CP-SAT must be skipped
        when greedy already places both."""
        buffers = [
            _mk("parent", 40, [0, 1]),
            _mk("child", 20, [1, 3], in_place_parents=["parent"]),
            _mk("stranger", 10, [2]),
        ]
        with patch.object(
            cp_model.CpSolver,
            "Solve",
            side_effect=AssertionError("Solve must not run"),
        ):
            result = _hybrid(buffers, LARGE_SIZE, ALIGNMENT)
        self.assertEqual(_placed(result), {"parent", "child", "stranger"})
        # child shares parent's address on the handoff.
        by_name = {b.name: b for b in result}
        self.assertEqual(by_name["parent"].address, by_name["child"].address)

    # -- 11. rejected greedy probe leaves originals untouched --
    def test_rejected_seed_leaves_originals_untouched_before_cpsat(self):
        """``_plan_layout_generic`` asserts every ``b.address is None`` on
        entry. Snapshot the caller's list right before that assertion; it
        must be all-None even after greedy ran its probe."""
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            CpSatLayoutSolver,
        )

        buffers = [_mk("a", 10, [0, 3]), _mk("b", 20, [0, 3]), _mk("c", 30, [0, 3])]
        solver = CpSatLayoutSolver(buffers, 50, ALIGNMENT)
        orig = solver._plan_layout_generic
        seen: dict = {}

        def _spy(*a, **k):
            seen["before_cpsat"] = [b.address for b in buffers]
            return orig(*a, **k)

        solver._plan_layout_generic = _spy  # type: ignore[method-assign]
        solver.plan_layout()
        self.assertEqual(seen["before_cpsat"], [None, None, None])

    # -- 12. accepted seed commits only the placement state expected --
    def test_accepted_seed_commits_addresses_only(self):
        """A certified seed writes ``address`` on the caller's buffers and
        sets ``spill_reasons`` in the same shape ``_plan_layout_generic``
        would; nothing else is mutated."""
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            CpSatLayoutSolver,
        )

        buffers = [_mk("a", 10, [0, 1]), _mk("b", 10, [0, 1])]
        # Preserve identity + starting fields we don't expect to be touched.
        pre_names = [b.name for b in buffers]
        pre_uses = [list(b.uses) for b in buffers]
        pre_sizes = [b.size for b in buffers]
        solver = CpSatLayoutSolver(buffers, LARGE_SIZE, ALIGNMENT)
        result = solver.plan_layout()
        # Same identity, same list.
        self.assertIs(result[0], buffers[0])
        self.assertIs(result[1], buffers[1])
        # Addresses populated.
        for b in buffers:
            self.assertIsNotNone(b.address)
        # Other fields untouched.
        self.assertEqual([b.name for b in buffers], pre_names)
        self.assertEqual([list(b.uses) for b in buffers], pre_uses)
        self.assertEqual([b.size for b in buffers], pre_sizes)
        # spill_reasons matches the shape _plan_layout_generic would emit.
        self.assertIsInstance(solver.spill_reasons, dict)
        # No buffer was excluded, so no spill reasons expected here.
        self.assertEqual(solver.spill_reasons, {})

    # -- 13. spill_reasons match allocator expectations --
    def test_spill_reasons_use_forced_reason_for_excluded_buffer(self):
        """A buffer forced non-resident by ``residency_reason`` must show
        up in ``spill_reasons`` with that same reason after a certified
        seed, matching what CP-SAT's own tail would emit."""
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            CpSatLayoutSolver,
        )

        buffers = [
            _mk("barred", 40, [0, 1], residency_reason="unit-test barred"),
            _mk("free", 40, [0, 1]),
        ]
        solver = CpSatLayoutSolver(buffers, LARGE_SIZE, ALIGNMENT)
        solver.plan_layout()
        self.assertEqual(solver.spill_reasons.get("barred"), "unit-test barred")

    # -- 14. joint plan_layout_and_core_divisions never uses the seed --
    def test_joint_path_does_not_use_seed_behavioral(self):
        """Behavioral test: patch ``_try_certified_greedy_seed`` to raise
        if called, then invoke ``plan_layout_and_core_divisions`` on a
        minimal ``CoreDivisionBuffer`` fixture. The joint entry must not
        touch the seed, so the call succeeds without hitting the patched
        raise.

        Uses the same ``_whole()`` single-division fixture pattern
        ``TestCpSatJointDivision`` uses in ``test_scratchpad_solver`` so
        the joint model has valid ``core_divisions`` to select from.
        """
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            CpSatLayoutSolver,
        )
        from torch_spyre._inductor.scratchpad.plan_solver import (
            CoreDivision,
            CoreDivisionBuffer,
        )

        whole = [CoreDivision()]
        # A single computed buffer with the whole-only division; keeps
        # the joint model well-formed without needing edge structure.
        buffers = [CoreDivisionBuffer("a", 8, [0, 1], core_divisions=whole)]
        with patch.object(
            CpSatLayoutSolver,
            "_try_certified_greedy_seed",
            side_effect=AssertionError("certified seed must not run on the joint path"),
        ):
            solver = CpSatLayoutSolver(buffers, LARGE_SIZE, alignment=128)
            solver.plan_layout_and_core_divisions()

    # -- 15. explicit LAYOUT_SOLVER=greedy behavior unchanged --
    def test_explicit_greedy_solver_unchanged(self):
        """``LAYOUT_SOLVER=greedy`` calls ``GreedyLayoutSolver`` directly;
        the fast path only lives in ``CpSatLayoutSolver`` so nothing about
        this call goes through the seed."""
        buffers = [_mk("a", 10, [0, 3]), _mk("b", 20, [0, 3]), _mk("c", 30, [0, 3])]
        greedy = _greedy(buffers, 50, ALIGNMENT)
        # Deterministic: greedy at capacity 50 spills c (largest last).
        self.assertEqual(_placed(greedy), {"a", "b"})
        self.assertEqual(_obj_units(greedy, ALIGNMENT), 60)


@unittest.skipUnless(_HAS_ORTOOLS, "seed helper is a CP-SAT feature")
class TestSharedSpillCostFormula(unittest.TestCase):
    """Both the CP-SAT wrapper and the certified seed call the same
    :func:`_hbm_spill_cost`. Verify the wrapper method delegates."""

    def test_wrapper_spill_cost_delegates_to_shared_helper(self):
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            _LifetimeBufferWithCpVars,
            _hbm_spill_cost,
        )

        buf = _mk("b", 8, [0, 1, 2], first_use_is_read=False)
        model = cp_model.CpModel()
        wrapper = _LifetimeBufferWithCpVars(
            buffer=buf, capacity_units=1024, model=model
        )
        self.assertEqual(wrapper.spill_cost(), _hbm_spill_cost(buf))


if __name__ == "__main__":
    unittest.main()

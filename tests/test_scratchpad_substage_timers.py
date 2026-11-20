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

"""The allocator's substage timers: what ``_time_hooks`` wraps, and what it does not.

``_time_hooks`` names the hooks it wraps rather than wrapping them at their call
sites, so that main can reshape ``plan_allocation``'s skeleton without conflicting
here. The cost of that choice is silence: a hook that upstream renames or removes
is simply skipped, and the metric disappears from the nightly sweep with nothing
raised and no test failing.

It sits here rather than under ``tests/inductor`` deliberately: that directory's
``conftest.py`` poisons device HBM in a session-scoped autouse fixture, which
makes every test in it need a device. Nothing here does -- the decorator is plain
Python and the solver is driven with mocks -- so it keeps company with
``test_frontend_timing_summarize.py`` instead.

So the coverage tests below assert, for **every** entry in the table, that some
allocator class defines that hook and that the decorator wrapped it. They are the
point of this file. A rename should fail here, on the change that made it, rather
than show up as a flat line on a chart three weeks later -- which is what happened
when ``pre_passes`` and ``post_passes`` became ``run_passes``: twelve nights of
sweeps hold the old names and nothing said so.

Strict rather than tolerant, and deliberately not keyed to the subset the nightly
report charts: that list lives in ``tools/frontend_timing/nightly``, which is not
in this repository, so a copy of it here would drift with nothing to catch it.
"""

from unittest import mock

from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch_spyre._inductor import config
from torch_spyre._inductor.timing_recorder import RECORDER
from torch_spyre._inductor.scratchpad.allocator import (
    _TIMED_HOOKS,
    CoOptimizingAllocator,
    ScratchpadAllocator,
    _time_hooks,
)

TIMED_CLASSES = (ScratchpadAllocator, CoOptimizingAllocator)


def regions(events):
    return [event.name for event in events]


def is_timed(attr):
    """Whether this class attribute is a hook ``_time_hooks`` has wrapped.

    Unwraps the descriptor first: a staticmethod carries the marker on the
    function it holds, not on itself.
    """
    held = attr.__func__ if isinstance(attr, (staticmethod, classmethod)) else attr
    return getattr(held, "_ts_timed", False)


@instantiate_parametrized_tests
class TestHookCoverage(TestCase):
    """What the decorator actually wrapped on the classes as they stand."""

    @parametrize("hook", sorted(_TIMED_HOOKS))
    def test_every_entry_in_the_table_is_a_hook_that_gets_timed(self, hook):
        """Strict on purpose: an entry that times nothing is a lost metric.

        The decorator skips a hook the class does not define, which is what lets
        this branch survive main reshaping ``plan_allocation``. That tolerance
        belongs there and not here: skipping keeps the merge clean and the
        compile working, while this test fails CI on the change that renamed the
        hook, for the author who still remembers why.
        """
        suffix = _TIMED_HOOKS[hook]
        defining = [c for c in TIMED_CLASSES if hook in c.__dict__]
        self.assertTrue(
            defining,
            f"no allocator class defines {hook!r}, so stage:Scratchpad:{suffix} is "
            f"recorded nowhere. If the hook was renamed, point _TIMED_HOOKS at the "
            f"new name and keep {suffix!r} as the suffix, so the metric stays "
            f"comparable with the nights already recorded under it. If the hook is "
            f"gone for good, delete the entry in the same change.",
        )
        for cls in defining:
            self.assertTrue(
                is_timed(cls.__dict__[hook]),
                f"{cls.__name__}.{hook} is defined but not wrapped",
            )

    @parametrize("cls", TIMED_CLASSES, name_fn=lambda c: c.__name__)
    def test_every_hook_the_class_defines_is_wrapped(self, cls):
        defined = {n: a for n in _TIMED_HOOKS if (a := cls.__dict__.get(n)) is not None}
        untimed = sorted(n for n, attr in defined.items() if not is_timed(attr))
        self.assertEqual(untimed, [], f"{cls.__name__} defines but does not time these")


class TestTimeHooks(TestCase):
    """The decorator's own contract, on classes built for the test."""

    def setUp(self):
        super().setUp()
        RECORDER._reset()
        self.addCleanup(RECORDER._reset)

    def test_a_hook_the_class_does_not_define_is_skipped(self):
        @_time_hooks
        class Bare:
            pass

        for name in _TIMED_HOOKS:
            self.assertNotIn(name, Bare.__dict__, f"{name} invented on a bare class")

    def test_a_wrapped_hook_records_its_region_and_returns_its_value(self):
        @_time_hooks
        class One:
            def _solve(self, value):
                return value * 2

        with config.patch({"timing": True}):
            self.assertEqual(One()._solve(21), 42)
        self.assertEqual(regions(RECORDER.events), ["stage:Scratchpad:solve"])

    def test_applying_twice_does_not_double_wrap(self):
        @_time_hooks
        class One:
            def _solve(self):
                return "once"

        twice = _time_hooks(One)
        with config.patch({"timing": True}):
            self.assertEqual(twice()._solve(), "once")
        self.assertEqual(regions(RECORDER.events), ["stage:Scratchpad:solve"])

    def test_a_staticmethod_stays_a_staticmethod(self):
        """``_run_passes`` is one, and wrapping the bare function adds a ``self``."""

        @_time_hooks
        class One:
            @staticmethod
            def _run_passes(graph):
                return f"ran {graph}"

        self.assertIsInstance(One.__dict__["_run_passes"], staticmethod)
        with config.patch({"timing": True}):
            self.assertEqual(One._run_passes("g"), "ran g")
            self.assertEqual(One()._run_passes("g"), "ran g")
        self.assertEqual(
            regions(RECORDER.events),
            ["stage:Scratchpad:run_passes"] * 2,
        )

    def test_an_override_in_a_subclass_is_timed_as_well_as_the_base(self):
        @_time_hooks
        class Base:
            def _solve(self):
                return "base"

        @_time_hooks
        class Sub(Base):
            def _solve(self):
                return "sub"

        with config.patch({"timing": True}):
            self.assertEqual(Sub()._solve(), "sub")
            self.assertEqual(Base()._solve(), "base")
        self.assertEqual(
            regions(RECORDER.events),
            ["stage:Scratchpad:solve"] * 2,
        )

    def test_nothing_is_recorded_when_timing_is_off(self):
        @_time_hooks
        class One:
            def _solve(self):
                return "value"

        with config.patch({"timing": False}):
            self.assertEqual(One()._solve(), "value")
        self.assertEqual(RECORDER.events, ())


class TestCpSatSolveStage(TestCase):
    """The solver's own region, which carries the model size the sweep tracks."""

    def setUp(self):
        super().setUp()
        RECORDER._reset()
        self.addCleanup(RECORDER._reset)

    def test_the_solve_region_carries_the_model_size_and_the_status(self):
        from torch_spyre._inductor.scratchpad import ilp_solver_ortools as ilp

        solver_cls = ilp.CpSatLayoutSolver
        proto = mock.Mock(variables=[0] * 1200, constraints=[0] * 900)
        model = mock.Mock(proto=proto)
        solver = mock.Mock()
        solver.Solve.return_value = "OPTIMAL_SENTINEL"
        solver.StatusName.return_value = "OPTIMAL"
        solver.WallTime.return_value = 1.2345
        # `buffers` has to be a real sequence: MemoryPlanSolver.__init__ sets it
        # to list(buffers), and a bare Mock attribute has no len().
        this = mock.Mock(spec=solver_cls)
        this.buffers = [object()] * 48

        with config.patch({"timing": True}):
            solver_cls._solve_and_record(this, solver, model, objective=True)

        events = {event.name: event for event in RECORDER.events}
        self.assertIn("stage:Scratchpad:cpsat_solve", events)
        meta = events["stage:Scratchpad:cpsat_solve"].meta
        self.assertEqual(meta["buffers"], 48)
        self.assertEqual(meta["variables"], 1200)
        self.assertEqual(meta["constraints"], 900)
        self.assertEqual(meta["status"], "OPTIMAL")
        self.assertEqual(meta["solver_wall_s"], 1.234)
        # ``objective`` is the bool the parameter declares, and the summarizer
        # drops booleans, so it stays provenance in the raw record rather than
        # becoming a counter series. Asserted as a bool to keep this test from
        # claiming a contract the callers cannot produce: they pass True or
        # nothing.
        self.assertIs(meta["objective"], True)

    def test_the_solve_region_defaults_the_objective_flag_to_false(self):
        from torch_spyre._inductor.scratchpad import ilp_solver_ortools as ilp

        solver_cls = ilp.CpSatLayoutSolver
        model = mock.Mock(proto=mock.Mock(variables=[0] * 3, constraints=[0] * 4))
        solver = mock.Mock()
        solver.StatusName.return_value = "FEASIBLE"
        solver.WallTime.return_value = 0.5
        this = mock.Mock(spec=solver_cls)
        this.buffers = []

        with config.patch({"timing": True}):
            solver_cls._solve_and_record(this, solver, model)

        meta = {e.name: e for e in RECORDER.events}["stage:Scratchpad:cpsat_solve"].meta
        self.assertIs(meta["objective"], False)
        self.assertEqual(meta["variables"], 3)
        # A measured zero, not a gap -- the summarizer keeps it.
        self.assertEqual(meta["buffers"], 0)


if __name__ == "__main__":
    run_tests()

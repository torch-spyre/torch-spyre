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


"""Tests for mirroring upstream compile phases into the timing record."""

import logging
import threading
import time

import pytest
import torch
from torch._dynamo.utils import dynamo_timed, get_chromium_event_logger

from torch_spyre._inductor import config, timing_recorder, torch_phases


@pytest.fixture(autouse=True)
def _clean_recorder():
    timing_recorder.RECORDER._reset()
    # Installed at import; a test that uninstalls must leave it installed.
    torch_phases.install()
    yield
    timing_recorder.RECORDER._reset()
    torch_phases.install()


def _by_name(recorder):
    return {event.name: event for event in recorder.events}


class TestBridge:
    """Phase mirroring, driven through upstream's own timing API."""

    def test_phases_nest_under_an_enclosing_spyre_region(self):
        with config.patch({"timing": True}):
            with timing_recorder.stage("stage:compile_fx:spyre_compile"):
                with dynamo_timed("GraphLowering.run"):
                    pass
        timing_recorder.RECORDER.finalize()

        events = _by_name(timing_recorder.RECORDER)
        compile_event = events["stage:compile_fx:spyre_compile"]
        lowering = events[torch_phases.LOWERING_PHASE]
        assert lowering.parent_ordinal == compile_event.ordinal
        assert lowering.is_closed
        # The point of the bridge: what used to be the parent's unattributed self
        # time is now a named child.
        assert compile_event.self_ns < compile_event.inclusive_ns

    def test_phase_name_override_is_what_gets_recorded(self):
        """``phase_name`` displaces the key upstream, and so here."""
        key = "_compile.compile_inner"
        with config.patch({"timing": True}):
            with dynamo_timed(key, phase_name="entire_frame_compile"):
                pass

        events = _by_name(timing_recorder.RECORDER)
        assert torch_phases.DYNAMO_PHASE in events
        assert f"stage:torch:{key}" not in events
        # The displaced key is not lost, it moves to metadata.
        assert events[torch_phases.DYNAMO_PHASE].meta["fn_name"] == key

    def test_metadata_is_narrowed_to_what_a_reader_uses(self):
        with config.patch({"timing": True}):
            with dynamo_timed(
                "GraphLowering.run", metadata={"noise": "x" * 4096}, is_backward=False
            ):
                pass

        meta = _by_name(timing_recorder.RECORDER)[torch_phases.LOWERING_PHASE].meta
        assert "noise" not in meta
        assert meta["is_backward"] is False
        assert "compile_id" in meta

    def test_abandoned_inner_phase_stays_open_and_leaves_the_parent_sound(self):
        """Upstream truncates its event stack on a Dynamo restart.

        It pops intermediate events without ever ending them, so an inner phase
        can vanish. It must not take its parent's arithmetic with it.
        """
        chromium = get_chromium_event_logger()
        with config.patch({"timing": True}):
            start = time.time_ns()
            chromium.log_event_start("outer", start, {}, False)
            chromium.log_event_start("inner", time.time_ns(), {}, False)
            chromium.log_event_end("outer", time.time_ns(), {}, start, False)
        timing_recorder.RECORDER.finalize()

        events = _by_name(timing_recorder.RECORDER)
        outer = events["stage:torch:outer"]
        inner = events["stage:torch:inner"]
        assert not inner.is_closed
        assert inner.self_ns == 0
        assert outer.is_closed
        # An open child contributes nothing, so the parent cannot go negative.
        assert outer.self_ns == outer.inclusive_ns
        # And the bridge's own stack is clean afterwards.
        assert torch_phases._open_phases() == []

    def test_a_phase_nested_in_itself_nests_rather_than_flattens(self):
        """``GraphLowering.run`` recurses into while-loop bodies and subgraphs.

        A record with three of them has one real lowering span and two inside it,
        so a reader totalling ``inclusive_ns`` by name overstates it. Nesting is
        what makes the distinction available.
        """
        with config.patch({"timing": True}):
            with dynamo_timed("GraphLowering.run"):
                with dynamo_timed("GraphLowering.run"):
                    pass
        timing_recorder.RECORDER.finalize()

        runs = [
            event
            for event in timing_recorder.RECORDER.events
            if event.name == torch_phases.LOWERING_PHASE
        ]
        assert len(runs) == 2
        outer, inner = sorted(runs, key=lambda event: event.ordinal)
        assert inner.parent_ordinal == outer.ordinal
        assert outer.self_ns == outer.inclusive_ns - inner.inclusive_ns

    def test_end_without_a_start_is_ignored(self):
        """Timing switched on mid-compile leaves ends with nothing to close."""
        chromium = get_chromium_event_logger()
        now = time.time_ns()
        chromium.log_event_start("orphan", now, {}, False)
        with config.patch({"timing": True}):
            chromium.log_event_end("orphan", time.time_ns(), {}, now, False)

        assert timing_recorder.RECORDER.events == ()

    def test_a_phase_closes_if_timing_is_turned_off_before_it_ends(self):
        chromium = get_chromium_event_logger()
        start = time.time_ns()
        with config.patch({"timing": True}):
            chromium.log_event_start("toggled", start, {}, False)
        chromium.log_event_end("toggled", time.time_ns(), {}, start, False)

        assert torch_phases._open_phases() == []
        # A later phase must not nest under the one that ended with timing off.
        with config.patch({"timing": True}):
            with dynamo_timed("GraphLowering.run"):
                pass

        events = _by_name(timing_recorder.RECORDER)
        assert events["stage:torch:toggled"].is_closed
        assert events[torch_phases.LOWERING_PHASE].parent_ordinal is None

    def test_repeated_failures_are_logged_at_debug(self, monkeypatch, caplog):
        def broken(event_name, metadata):
            raise RuntimeError("bridge broke")

        monkeypatch.setattr(torch_phases, "_begin", broken)
        monkeypatch.setattr(torch_phases, "_warned", set())
        with caplog.at_level(logging.DEBUG, logger=torch_phases.logger.name):
            with config.patch({"timing": True}):
                for _ in range(2):
                    with dynamo_timed("GraphLowering.run"):
                        pass

        ours = [r for r in caplog.records if r.name == torch_phases.logger.name]
        assert [r.levelno for r in ours] == [logging.WARNING, logging.DEBUG]
        assert "bridge broke" in ours[0].getMessage()
        assert ours[1].exc_info[1].args == ("bridge broke",)

    def test_a_different_failure_still_warns(self, monkeypatch, caplog):
        errors = iter([RuntimeError("first"), ValueError("second")])

        def broken(event_name, metadata):
            raise next(errors)

        monkeypatch.setattr(torch_phases, "_begin", broken)
        monkeypatch.setattr(torch_phases, "_warned", set())
        name = torch_phases.logger.name
        with caplog.at_level(logging.WARNING, logger=name):
            with config.patch({"timing": True}):
                for _ in range(2):
                    with dynamo_timed("GraphLowering.run"):
                        pass

        ours = [r.getMessage() for r in caplog.records if r.name == name]
        assert len(ours) == 2
        assert "first" in ours[0] and "second" in ours[1]

    def test_records_nothing_when_timing_is_off(self):
        with dynamo_timed("GraphLowering.run"):
            pass
        assert timing_recorder.RECORDER.events == ()

    def test_install_is_idempotent(self):
        from torch._dynamo.utils import ChromiumEventLogger

        before = ChromiumEventLogger.log_event_start
        torch_phases.install()
        torch_phases.install()
        assert ChromiumEventLogger.log_event_start is before

        with config.patch({"timing": True}):
            with dynamo_timed("GraphLowering.run"):
                pass
        # Double wrapping would record the phase twice.
        names = [event.name for event in timing_recorder.RECORDER.events]
        assert names.count(torch_phases.LOWERING_PHASE) == 1

    def test_uninstall_restores_upstream(self):
        torch_phases.uninstall()
        try:
            with config.patch({"timing": True}):
                with dynamo_timed("GraphLowering.run"):
                    pass
            assert timing_recorder.RECORDER.events == ()
        finally:
            torch_phases.install()

    def test_uninstall_clears_every_threads_stack(self):
        opened = threading.Event()
        uninstalled = threading.Event()
        seen = []

        def work():
            chromium = get_chromium_event_logger()
            with config.patch({"timing": True}):
                chromium.log_event_start("left_open", time.time_ns(), {}, False)
                opened.set()
                uninstalled.wait(timeout=30)
                seen.append(len(torch_phases._open_phases()))

        thread = threading.Thread(target=work)
        thread.start()
        assert opened.wait(timeout=30)
        torch_phases.uninstall()
        uninstalled.set()
        thread.join(timeout=30)
        torch_phases.install()

        assert seen == [0]

    def test_threads_do_not_share_the_phase_stack(self):
        """Async compile fans codegen across threads; each nests on its own."""
        seen = {}

        def work(tag):
            with config.patch({"timing": True}):
                with dynamo_timed(f"phase_{tag}"):
                    seen[tag] = len(torch_phases._open_phases())

        with config.patch({"timing": True}):
            threads = [threading.Thread(target=work, args=(t,)) for t in ("a", "b")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert seen == {"a": 1, "b": 1}
        events = _by_name(timing_recorder.RECORDER)
        for tag in ("a", "b"):
            assert events[f"stage:torch:phase_{tag}"].parent_ordinal is None


class TestBackendNameIsDistinct:
    """Upstream has a phase called backend_compile and it is not ours."""

    def test_async_compile_uses_the_shared_constant(self):
        from torch_spyre.execution import async_compile

        assert async_compile._BACKEND_STAGE is timing_recorder.BACKEND_COMPILE_EVENT


class TestRequiredPhases:
    """The four phase names a baseline reports, checked against a real compile.

    Deliberately a CPU compile: lowering, codegen, AOT tracing and Dynamo are all
    upstream, so their names can be verified on any runner. A torch bump that
    renames one fails here instead of silently emptying a committed baseline.
    """

    def test_a_compile_emits_every_required_phase(self):
        torch._dynamo.reset()

        def fn(x):
            return (x * 2).sum()

        with (
            config.patch({"timing": True}),
            torch._inductor.config.patch({"force_disable_caches": True}),
        ):
            torch.compile(fn)(torch.randn(8, 8))

        names = {event.name for event in timing_recorder.RECORDER.events}
        missing = set(torch_phases.REQUIRED_PHASES) - names
        assert not missing, f"upstream phases missing or renamed: {sorted(missing)}"

    def test_lowering_and_codegen_are_inside_the_frame_compile(self):
        torch._dynamo.reset()

        with (
            config.patch({"timing": True}),
            torch._inductor.config.patch({"force_disable_caches": True}),
        ):
            torch.compile(lambda x: x.relu())(torch.randn(8, 8))
        timing_recorder.RECORDER.finalize()

        events = {e.name: e for e in timing_recorder.RECORDER.events}
        by_ordinal = {e.ordinal: e for e in timing_recorder.RECORDER.events}

        def ancestors(event):
            while event.parent_ordinal is not None:
                event = by_ordinal[event.parent_ordinal]
                yield event.name

        for phase in (torch_phases.LOWERING_PHASE, torch_phases.CODEGEN_PHASE):
            assert torch_phases.DYNAMO_PHASE in set(ancestors(events[phase])), phase

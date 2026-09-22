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


"""Tests for the structured frontend timing recorder."""

import contextlib
import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_spyre._inductor import config, timing_recorder
from torch_spyre._inductor.timing_recorder import TimingRecorder


@pytest.fixture(autouse=True)
def _clean_recorder():
    """Keep the process-wide recorder from carrying events between tests."""
    timing_recorder.RECORDER._reset()
    yield
    timing_recorder.RECORDER._reset()


def _by_name(recorder):
    return {event.name: event for event in recorder.events}


class TestRecorder:
    """The recorder itself, independent of the config gate."""

    def test_nesting_and_self_time(self):
        recorder = TimingRecorder()
        with recorder.stage("pipeline", passes=2) as pipeline:
            with recorder.stage("pass:a") as first:
                first.meta["output_operations"] = 7
            with recorder.stage("pass:b"):
                pass
        recorder.finalize()

        events = _by_name(recorder)
        assert set(events) == {"pipeline", "pass:a", "pass:b"}
        assert events["pass:a"].parent_ordinal == pipeline.ordinal
        assert events["pass:b"].parent_ordinal == pipeline.ordinal
        assert events["pass:a"].meta["output_operations"] == 7

        children = events["pass:a"].inclusive_ns + events["pass:b"].inclusive_ns
        assert events["pipeline"].self_ns == (
            events["pipeline"].inclusive_ns - children
        )
        # A leaf has no children, so self time is its whole span.
        assert events["pass:a"].self_ns == events["pass:a"].inclusive_ns

    def test_raising_pass_is_recorded_and_leaves_stack_clean(self):
        recorder = TimingRecorder()
        with pytest.raises(ValueError):
            with recorder.stage("boom"):
                raise ValueError("kaboom")
        with recorder.stage("after"):
            pass

        events = _by_name(recorder)
        assert events["boom"].error == "ValueError: kaboom"
        assert events["boom"].inclusive_ns > 0
        # "after" is a sibling, not a child of the region that raised.
        assert events["after"].parent_ordinal is None

    def test_open_region_does_not_go_negative(self):
        """A dump taken mid-region must not report negative self time."""
        recorder = TimingRecorder()
        outer = recorder.stage("outer")
        outer.__enter__()
        with recorder.stage("inner"):
            pass
        recorder.finalize()

        events = _by_name(recorder)
        assert not events["outer"].is_closed
        assert events["outer"].self_ns == 0
        assert events["outer"].to_dict()["open"] is True
        # The closed child keeps its own measurement.
        assert events["inner"].self_ns == events["inner"].inclusive_ns
        assert "open" not in events["inner"].to_dict()

        outer.__exit__(None, None, None)
        recorder.finalize()
        assert events["outer"].self_ns == (
            events["outer"].inclusive_ns - events["inner"].inclusive_ns
        )

    def test_concurrent_regions_nest_per_thread(self):
        recorder = TimingRecorder()
        started = threading.Barrier(2)

        def worker(tag):
            with recorder.stage(f"outer:{tag}"):
                started.wait(timeout=10)
                with recorder.stage(f"inner:{tag}"):
                    pass

        threads = [threading.Thread(target=worker, args=(tag,)) for tag in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        recorder.finalize()

        events = _by_name(recorder)
        assert len(events) == 4
        # Each thread's inner region belongs to that thread's outer region,
        # even though the two overlapped in time.
        for tag in ("a", "b"):
            inner, outer = events[f"inner:{tag}"], events[f"outer:{tag}"]
            assert inner.parent_ordinal == outer.ordinal
        # Ordinals are unique across threads.
        ordinals = [event.ordinal for event in recorder.events]
        assert len(set(ordinals)) == len(ordinals)

    def test_dump_writes_one_record_and_no_leftovers(self, tmp_path):
        recorder = TimingRecorder()
        with recorder.stage("only"):
            pass
        recorder.set_run_meta(workload="flash", Lq=512)
        recorder.finalize()

        path = os.fspath(tmp_path / "record.json")
        recorder.dump_json(path)
        # The write goes to a temp name and is renamed into place.
        assert os.listdir(tmp_path) == ["record.json"]

        record = json.loads((tmp_path / "record.json").read_text())
        assert [event["name"] for event in record["events"]] == ["only"]
        meta = record["meta"]
        assert meta["workload"] == "flash" and meta["Lq"] == 512
        assert meta["recorder_version"] == timing_recorder.RECORDER_VERSION
        assert meta["clock"] == "time.perf_counter_ns"
        # A record that cannot name the substrate it measured is not comparable.
        assert meta["torch_version"] == torch.__version__
        assert meta["torch_spyre_version"]
        assert meta["python_version"]
        assert meta["pid"] == os.getpid()
        # The version string carries the sha only for a source checkout, so the
        # sha and the package path it describes are recorded separately.
        assert "git_sha" in meta
        assert meta["torch_spyre_path"].endswith("torch_spyre")

    def test_unserializable_meta_does_not_lose_the_record(self, tmp_path):
        recorder = TimingRecorder()
        with recorder.stage("only", dtype=torch.float16, shape=object()):
            pass
        recorder.finalize()
        path = os.fspath(tmp_path / "record.json")
        recorder.dump_json(path)

        event = json.loads((tmp_path / "record.json").read_text())["events"][0]
        assert event["meta"]["dtype"] == "torch.float16"
        assert "object object at" in event["meta"]["shape"]

    def test_reset_clears_events_and_metadata(self):
        recorder = TimingRecorder()
        with recorder.stage("first"):
            pass
        recorder.set_run_meta(workload="flash")
        recorder._reset()
        assert recorder.events == ()
        assert recorder.run_meta == {}


class TestRunMetadata:
    def test_git_sha_of_a_non_repository_is_empty(self, tmp_path):
        assert timing_recorder._git_sha(os.fspath(tmp_path)) == ""

    def test_git_sha_of_this_checkout(self):
        from torch_spyre import version

        sha = timing_recorder._git_sha(os.path.dirname(version.__file__))
        # Empty for a wheel install; a short sha for a source checkout.
        assert sha == "" or sha.isalnum()


class TestRecordPath:
    """One destination setting must not have two processes fighting over it."""

    def test_pid_is_inserted_before_the_suffix(self):
        assert timing_recorder.record_path("/tmp/rec.json", pid=7) == "/tmp/rec.7.json"
        assert timing_recorder.record_path("/tmp/rec", pid=7) == "/tmp/rec.7"
        assert timing_recorder.record_path("rec.a.json", pid=7) == "rec.a.7.json"

    def test_two_processes_do_not_collide(self):
        first = timing_recorder.record_path("/tmp/rec.json", pid=11)
        second = timing_recorder.record_path("/tmp/rec.json", pid=12)
        assert first != second


class TestGate:
    """Timing is off unless asked for, and records nothing when off."""

    def test_records_nothing_when_disabled(self):
        with config.patch({"timing": False}):
            assert not timing_recorder.is_enabled()
            with timing_recorder.stage("ignored"):
                pass
            assert timing_recorder.RECORDER.events == ()

    def test_records_when_enabled(self):
        with config.patch({"timing": True}):
            assert timing_recorder.is_enabled()
            with timing_recorder.stage("recorded", passes=1):
                pass
        events = _by_name(timing_recorder.RECORDER)
        assert events["recorded"].meta == {"passes": 1}

    def test_annotating_the_disabled_event_retains_nothing(self):
        with config.patch({"timing": False}):
            with timing_recorder.stage("ignored") as event:
                event.meta["output_operations"] = 3
                assert event.meta == {}
            with timing_recorder.stage("ignored") as event:
                assert event.meta.setdefault("input_operations", 3) == 3
                assert event.meta == {}
            with timing_recorder.stage("ignored_too") as event:
                assert event.meta == {}

    def test_no_dump_without_a_destination(self, tmp_path):
        with config.patch({"timing": True, "timing_out": ""}):
            assert timing_recorder.dump_and_finalize() is None
        with config.patch({"timing": False, "timing_out": os.fspath(tmp_path / "x")}):
            assert timing_recorder.dump_and_finalize() is None
        assert os.listdir(tmp_path) == []

    def test_dump_uses_configured_destination_with_pid(self, tmp_path):
        configured = os.fspath(tmp_path / "record.json")
        with config.patch({"timing": True, "timing_out": configured}):
            with timing_recorder.stage("recorded"):
                pass
            written = timing_recorder.dump_and_finalize()
        assert written == timing_recorder.record_path(configured)
        assert os.listdir(tmp_path) == [f"record.{os.getpid()}.json"]
        assert json.loads(open(written).read())["events"]

    def test_explicit_path_is_written_verbatim(self, tmp_path):
        exact = os.fspath(tmp_path / "exact.json")
        with config.patch({"timing": True, "timing_out": ""}):
            with timing_recorder.stage("recorded"):
                pass
            assert timing_recorder.dump_and_finalize(exact) == exact
        assert os.listdir(tmp_path) == ["exact.json"]

    def test_atexit_hook_writes_the_record(self, tmp_path):
        configured = os.fspath(tmp_path / "record.json")
        with config.patch({"timing": True, "timing_out": configured}):
            with timing_recorder.stage("recorded"):
                pass
            timing_recorder._dump_at_exit()
        assert os.listdir(tmp_path) == [f"record.{os.getpid()}.json"]

    def test_atexit_hook_warns_instead_of_raising(self, tmp_path):
        unwritable = os.fspath(tmp_path / "no_such_dir" / "record.json")
        with config.patch({"timing": True, "timing_out": unwritable}):
            with patch.object(timing_recorder.logger, "warning") as warning:
                timing_recorder._dump_at_exit()
        assert warning.call_count == 1


def _make_spyre_graph(n_operations=1):
    """Minimal mock GraphLowering carrying Spyre-device operations."""
    graph = MagicMock()
    graph.operations = []
    for index in range(n_operations):
        operation = MagicMock()
        operation.get_device.return_value = torch.device("spyre")
        operation.get_name.return_value = f"buf{index}"
        graph.operations.append(operation)
    return graph


@contextlib.contextmanager
def _only_the_pass_loop():
    """Stub the steps the pre-scheduling entry point runs around its passes.

    They are not what these tests measure, and driving the real cost model over
    mock operations would make a timing test fail for unrelated reasons.
    """
    with (
        patch("torch_spyre._inductor.passes.cost_model_pass"),
        patch("torch_spyre._inductor.passes.dump_cost_model"),
        patch("torch_spyre._inductor.passes.finalize_work_division_for_scheduler"),
    ):
        yield


class TestPipelineHooks:
    """Every pipeline emits a pipeline event and one event per pass."""

    def test_pre_scheduling_records_operation_counts(self):
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses

        graph = _make_spyre_graph(n_operations=2)

        def my_pass(g):
            # Passes mutate graph.operations, so input and output counts differ.
            g.operations.append(MagicMock())

        pipeline = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        pipeline.passes = [my_pass]

        with config.patch({"timing": True}), _only_the_pass_loop():
            pipeline(graph)

        events = _by_name(timing_recorder.RECORDER)
        pipeline_event = events["pipeline:CustomPreSchedulingPasses"]
        loop_event = events["stage:CustomPreSchedulingPasses:pass_loop"]
        pass_event = events["pass:CustomPreSchedulingPasses:my_pass"]
        assert pipeline_event.meta == {
            "passes": 1,
            "input_operations": 2,
            "output_operations": 3,
        }
        assert pass_event.meta == {"input_operations": 2, "output_operations": 3}
        assert loop_event.parent_ordinal == pipeline_event.ordinal
        assert pass_event.parent_ordinal == loop_event.ordinal

    def test_pre_scheduling_times_the_work_around_the_passes(self):
        """The entry point does more than run its pass list; all of it is timed."""
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses

        graph = _make_spyre_graph()
        pipeline = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        pipeline.passes = []

        with config.patch({"timing": True}), _only_the_pass_loop():
            pipeline(graph)

        events = _by_name(timing_recorder.RECORDER)
        pipeline_ordinal = events["pipeline:CustomPreSchedulingPasses"].ordinal
        for what in ("pass_loop", "cost_model", "cost_dump", "finalize_work_division"):
            name = f"stage:CustomPreSchedulingPasses:{what}"
            assert name in events, name
            assert events[name].parent_ordinal == pipeline_ordinal

    def test_ir_dumps_are_attributed_when_logging_is_on(self):
        """Formatting the IR is real work; it must not land in pipeline self time."""
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses

        graph = _make_spyre_graph()
        pipeline = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        pipeline.passes = []

        with config.patch({"timing": True}), _only_the_pass_loop():
            with patch("torch_spyre._inductor.passes.logger") as mock_logger:
                mock_logger.isEnabledFor.return_value = True
                pipeline(graph)

        events = _by_name(timing_recorder.RECORDER)
        pipeline_ordinal = events["pipeline:CustomPreSchedulingPasses"].ordinal
        for what in ("log_before", "log_after"):
            name = f"stage:CustomPreSchedulingPasses:{what}"
            assert name in events, name
            assert events[name].parent_ordinal == pipeline_ordinal

    def test_no_dump_regions_at_default_verbosity(self):
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses

        graph = _make_spyre_graph()
        pipeline = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        pipeline.passes = []

        with config.patch({"timing": True}), _only_the_pass_loop():
            with patch("torch_spyre._inductor.passes.logger") as mock_logger:
                mock_logger.isEnabledFor.return_value = False
                pipeline(graph)

        names = {event.name for event in timing_recorder.RECORDER.events}
        assert not [n for n in names if n.endswith(":log_before")]
        assert not [n for n in names if n.endswith(":log_after")]

    def test_pre_scheduling_records_a_pass_that_raises(self):
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses

        graph = _make_spyre_graph()

        def bad_pass(g):
            raise RuntimeError("pass exploded")

        pipeline = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        pipeline.passes = [bad_pass]

        with config.patch({"timing": True}), _only_the_pass_loop():
            with pytest.raises(RuntimeError):
                pipeline(graph)

        events = _by_name(timing_recorder.RECORDER)
        pass_event = events["pass:CustomPreSchedulingPasses:bad_pass"]
        assert pass_event.error == "RuntimeError: pass exploded"
        # The failure propagates, so the enclosing regions record it too.
        assert events["pipeline:CustomPreSchedulingPasses"].error is not None

    def test_graph_pipeline_records_node_counts(self):
        from torch_spyre._inductor.passes import CustomPreGradPasses

        graph = MagicMock()
        graph.nodes = [MagicMock(), MagicMock()]

        def my_pass(g):
            g.nodes.append(MagicMock())

        pipeline = CustomPreGradPasses.__new__(CustomPreGradPasses)
        pipeline.passes = [my_pass]

        with config.patch({"timing": True}):
            with patch(
                "torch_spyre._inductor.passes._graph_has_spyre_device",
                return_value=True,
            ):
                pipeline(graph)

        events = _by_name(timing_recorder.RECORDER)
        pipeline_event = events["pipeline:CustomPreGradPasses"]
        pass_event = events["pass:CustomPreGradPasses:my_pass"]
        assert pipeline_event.meta == {
            "passes": 1,
            "input_nodes": 2,
            "output_nodes": 3,
        }
        assert pass_event.meta == {"input_nodes": 2, "output_nodes": 3}
        assert pass_event.parent_ordinal == pipeline_event.ordinal

    def test_graph_pipeline_skips_counting_when_disabled(self):
        from torch_spyre._inductor.passes import CustomPreGradPasses

        graph = MagicMock()
        graph.nodes = [MagicMock()]
        pipeline = CustomPreGradPasses.__new__(CustomPreGradPasses)
        pipeline.passes = [lambda g: None]

        with config.patch({"timing": False}):
            with patch(
                "torch_spyre._inductor.passes._graph_has_spyre_device",
                return_value=True,
            ):
                pipeline(graph)

        assert timing_recorder.RECORDER.events == ()

    def test_node_pipeline_records_node_counts(self):
        from torch_spyre._inductor.passes import CustomPreFusionPasses

        nodes = [MagicMock(), MagicMock()]

        def my_pass(target):
            return target + [MagicMock()]

        pipeline = CustomPreFusionPasses.__new__(CustomPreFusionPasses)
        pipeline.passes = [my_pass]

        with config.patch({"timing": True}):
            with (
                patch(
                    "torch_spyre._inductor.passes._nodes_have_spyre_device",
                    return_value=True,
                ),
                patch(
                    "torch_spyre._inductor.passes.SpyreGraphTransformObserver"
                ) as observer,
            ):
                observer.return_value = contextlib.nullcontext(MagicMock())
                result = pipeline(nodes)

        assert len(result) == 3
        events = _by_name(timing_recorder.RECORDER)
        pipeline_event = events["pipeline:CustomPreFusionPasses"]
        pass_event = events["pass:CustomPreFusionPasses:my_pass"]
        assert pipeline_event.meta == {
            "passes": 1,
            "input_nodes": 2,
            "output_nodes": 3,
        }
        assert pass_event.meta == {"input_nodes": 2, "output_nodes": 3}
        assert pass_event.parent_ordinal == pipeline_event.ordinal

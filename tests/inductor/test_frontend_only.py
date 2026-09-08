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


"""Tests for the frontend-only measurement mode (TORCH_SPYRE_FRONTEND_ONLY)."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch  # noqa: F401

from torch_spyre._inductor import config, timing_recorder
from torch_spyre.execution import async_compile as ac
from torch_spyre.execution.kernel_runner import (
    SpyreFrontendOnlyRunner,
    SpyreSDSCKernelRunner,
)


@pytest.fixture(autouse=True)
def _clean_recorder():
    timing_recorder.RECORDER._reset()
    # The frontend-only warning fires once per process, so the first skipping
    # test would otherwise decide whether any later one sees it.
    ac._warned_frontend_only = False
    yield
    timing_recorder.RECORDER._reset()
    ac._warned_frontend_only = False


def _compile_one_kernel(tmp_path):
    """Drive SpyreAsyncCompile.sdsc with the real backend seam stubbed out.

    Everything the mode is supposed to preserve -- bundle generation and kernel
    provenance -- is patched so the test can assert it still ran, and
    subprocess.run is patched so a test failure cannot invoke the real backend.
    """
    compiler = ac.SpyreAsyncCompile.__new__(ac.SpyreAsyncCompile)
    compiler._provenance_attempt_count = 0
    compiler._provenance_failure_count = 0

    with (
        patch.object(ac, "get_output_dir", return_value=str(tmp_path)),
        patch.object(ac, "generate_bundle") as generate_bundle,
        patch.object(
            ac, "build_kernel_provenance_descriptor", return_value=None
        ) as provenance,
        patch.object(ac, "find_unimplemented", return_value=None),
        patch.object(ac.subprocess, "run") as subprocess_run,
        patch.object(ac, "SpyreSDSCKernelRunner", autospec=True) as sdsc_runner,
    ):
        runner = compiler.sdsc("fused_add_0", [MagicMock()])
        return runner, generate_bundle, provenance, subprocess_run, sdsc_runner


def _compile_ktir_one_kernel(tmp_path):
    """Drive _compile_ktir_with_dbo with its backend seam stubbed out.

    The KTIR emitter reaches dbo-opt, a per-kernel backend invocation of the same
    class as dxp_standalone, so the mode has to stop here too.
    """
    compiler = ac.SpyreAsyncCompile.__new__(ac.SpyreAsyncCompile)
    ktir_path = tmp_path / "fused_add_0.ktir"
    ktir_path.write_text("// ktir")

    with (
        patch.object(ac, "_check_ktir_device_prerequisites") as prereqs,
        patch.object(ac.subprocess, "run") as subprocess_run,
        patch.object(ac, "SpyreSDSCKernelRunner", autospec=True) as sdsc_runner,
        patch.object(ac.os.path, "exists", return_value=True),
    ):
        runner = compiler._compile_ktir_with_dbo(
            "fused_add_0", str(ktir_path), str(tmp_path)
        )
        return runner, prereqs, subprocess_run, sdsc_runner


class TestKtirBoundary:
    """The second emitter's backend is dbo-opt, and the mode must stop there too."""

    def test_ktir_backend_is_skipped(self, tmp_path):
        with config.patch({"frontend_only": True, "timing": True}):
            runner, prereqs, subprocess_run, sdsc_runner = _compile_ktir_one_kernel(
                tmp_path
            )

        subprocess_run.assert_not_called()
        sdsc_runner.assert_not_called()
        assert isinstance(runner, SpyreFrontendOnlyRunner)
        # The toolchain is the backend's business: a frontend measurement must be
        # possible on a box with no dbo-opt and no KTIR_DEVICE_MLIR.
        prereqs.assert_not_called()

        meta = timing_recorder.RECORDER.run_meta
        assert meta["backend_skipped_kernels"] == ["fused_add_0"]
        events = {e.name: e for e in timing_recorder.RECORDER.events}
        skipped = events["stage:SpyreAsyncCompile:backend_skipped"]
        assert skipped.meta["tool"] == "dbo-opt"

    def test_ktir_backend_runs_when_the_mode_is_off(self, tmp_path):
        with config.patch({"frontend_only": False, "timing": True}):
            _, prereqs, subprocess_run, _ = _compile_ktir_one_kernel(tmp_path)

        subprocess_run.assert_called_once()
        assert subprocess_run.call_args.args[0][0] == "dbo-opt"
        prereqs.assert_called_once()

        events = {e.name: e for e in timing_recorder.RECORDER.events}
        backend = events["stage:SpyreAsyncCompile:backend_compile"]
        # One event name for both emitters; the tool is what tells them apart, so
        # a frontend total stays one subtraction.
        assert backend.meta["tool"] == "dbo-opt"
        assert backend.meta["kernel"] == "fused_add_0"

    def test_ktir_emission_is_timed(self, tmp_path):
        """generate_ktir is backend-input generation, like generate_bundle."""
        import torch_spyre._inductor.codegen.ktir as ktir_module

        compiler = ac.SpyreAsyncCompile.__new__(ac.SpyreAsyncCompile)
        emitted = patch.object(ktir_module, "generate_ktir", return_value="// ktir")
        with (
            config.patch({"frontend_only": True, "timing": True}),
            patch.object(ac, "find_unimplemented", return_value=None),
            patch.object(ac, "get_output_dir", return_value=str(tmp_path)),
            emitted,
            patch.object(ac.subprocess, "run") as subprocess_run,
        ):
            runner = compiler.ktir("fused_add_0", [MagicMock()])

        subprocess_run.assert_not_called()
        assert isinstance(runner, SpyreFrontendOnlyRunner)
        events = {e.name: e for e in timing_recorder.RECORDER.events}
        emit = events["stage:SpyreAsyncCompile:generate_ktir"]
        assert emit.meta == {"kernel": "fused_add_0", "specs": 1}


class TestBoundary:
    def test_backend_is_skipped_and_frontend_still_runs(self, tmp_path):
        with config.patch({"frontend_only": True}):
            runner, generate_bundle, provenance, subprocess_run, sdsc_runner = (
                _compile_one_kernel(tmp_path)
            )

        # The backend never ran, and no kernel was prepared from its output.
        subprocess_run.assert_not_called()
        sdsc_runner.assert_not_called()
        # Backend-input generation did, which is the work being measured.
        generate_bundle.assert_called_once()
        provenance.assert_called_once()
        assert isinstance(runner, SpyreFrontendOnlyRunner)

    def test_backend_runs_when_the_mode_is_off(self, tmp_path):
        with config.patch({"frontend_only": False}):
            runner, generate_bundle, _, subprocess_run, sdsc_runner = (
                _compile_one_kernel(tmp_path)
            )

        subprocess_run.assert_called_once()
        assert subprocess_run.call_args.args[0][0] == "dxp_standalone"
        generate_bundle.assert_called_once()
        sdsc_runner.assert_called_once()
        assert not isinstance(runner, SpyreFrontendOnlyRunner)

    def test_mode_is_off_by_default(self):
        # Reading config.frontend_only here would fail for anyone who has the
        # variable exported -- exactly the people running this code -- so assert
        # what the default actually is: absent means off.
        from torch_spyre._inductor.logging_utils import _get_env_bool

        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_bool("TORCH_SPYRE_FRONTEND_ONLY", False) is False
        with patch.dict(os.environ, {"TORCH_SPYRE_FRONTEND_ONLY": "1"}):
            assert _get_env_bool("TORCH_SPYRE_FRONTEND_ONLY", False) is True


class TestBoundaryMarker:
    """A skipped backend must be distinguishable from a compile that died."""

    def test_record_names_every_skipped_kernel(self, tmp_path):
        with config.patch({"frontend_only": True, "timing": True}):
            _compile_one_kernel(tmp_path)
            _compile_one_kernel(tmp_path)
            # The mode is stated by the record's metadata whether or not any
            # kernel was skipped. Read inside the patch: the metadata is
            # assembled when the record is written.
            assert timing_recorder.RECORDER.to_dict()["meta"]["frontend_only"] is True

        meta = timing_recorder.RECORDER.run_meta
        assert meta["backend_skipped_kernels"] == ["fused_add_0", "fused_add_0"]

        names = [event.name for event in timing_recorder.RECORDER.events]
        assert names.count("stage:SpyreAsyncCompile:backend_skipped") == 2
        assert "stage:SpyreAsyncCompile:backend_compile" not in names

    def test_a_normal_compile_records_the_backend_instead(self, tmp_path):
        with config.patch({"frontend_only": False, "timing": True}):
            _compile_one_kernel(tmp_path)
            # A normal run says so rather than staying silent about the mode.
            assert timing_recorder.RECORDER.to_dict()["meta"]["frontend_only"] is False

        names = [event.name for event in timing_recorder.RECORDER.events]
        assert "stage:SpyreAsyncCompile:backend_compile" in names
        assert "stage:SpyreAsyncCompile:backend_skipped" not in names
        assert "backend_skipped_kernels" not in timing_recorder.RECORDER.run_meta

    def test_bundle_generation_is_timed_either_way(self, tmp_path):
        for frontend_only in (True, False):
            timing_recorder.RECORDER._reset()
            with config.patch({"frontend_only": frontend_only, "timing": True}):
                _compile_one_kernel(tmp_path)
            events = {e.name: e for e in timing_recorder.RECORDER.events}
            bundle = events["stage:SpyreAsyncCompile:generate_bundle"]
            assert bundle.meta["kernel"] == "fused_add_0"
            assert bundle.meta["specs"] == 1


class TestNesting:
    """The frontend total is a subtraction, so the nesting has to hold."""

    def test_backend_events_nest_under_the_compile(self, tmp_path):
        with config.patch({"frontend_only": False, "timing": True}):
            with timing_recorder.stage("stage:compile_fx:spyre_compile") as compile_ev:
                _compile_one_kernel(tmp_path)

        events = {e.name: e for e in timing_recorder.RECORDER.events}
        backend = events["stage:SpyreAsyncCompile:backend_compile"]
        # If sdsc ever moves off the compiling thread this orphans to None and
        # every derived frontend number silently becomes wrong.
        assert backend.parent_ordinal == compile_ev.ordinal

    def test_skip_markers_nest_under_the_compile(self, tmp_path):
        with config.patch({"frontend_only": True, "timing": True}):
            with timing_recorder.stage("stage:compile_fx:spyre_compile") as compile_ev:
                _compile_one_kernel(tmp_path)

        events = {e.name: e for e in timing_recorder.RECORDER.events}
        assert (
            events["stage:SpyreAsyncCompile:backend_skipped"].parent_ordinal
            == compile_ev.ordinal
        )


class TestFailsLoudly:
    def test_calling_the_kernel_raises_with_an_actionable_message(self):
        runner = SpyreFrontendOnlyRunner("fused_add_0", "/tmp/bundle")
        with pytest.raises(RuntimeError) as excinfo:
            runner.run()
        message = str(excinfo.value)
        assert "TORCH_SPYRE_FRONTEND_ONLY" in message
        assert "fused_add_0" in message
        assert "/tmp/bundle" in message

    def test_it_is_not_mistaken_for_a_real_runner(self):
        runner = SpyreFrontendOnlyRunner("fused_add_0", "/tmp/bundle")
        assert not isinstance(runner, SpyreSDSCKernelRunner)
        # No spyreCodeDir was prepared, so there is no jobplan to launch.
        assert not hasattr(runner, "jobplan")

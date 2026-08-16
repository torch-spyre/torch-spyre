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

"""Unit tests for the OpSpec->KTIR backend-compiler hook.

Covers ``_check_ktir_device_prerequisites`` and
``SpyreAsyncCompile._compile_ktir_with_dbo`` -- the step between the emitter
(covered by ``test_ktir_emitter.py``) and the loaded kernel.

No device and no real ``dbo-opt``: ``subprocess.run`` and ``shutil.which`` are
mocked throughout, so unlike the emitter's golden test this file needs neither
``mlir_ktdp`` nor deeptools and runs anywhere.
"""

import os
import subprocess
import tempfile
import unittest
from unittest import mock

from torch_spyre.execution import async_compile as ac

_MODULE = "torch_spyre.execution.async_compile"
_CONFIG = "torch_spyre._inductor.config"


def _compiler():
    """A ``SpyreAsyncCompile`` that has not started AsyncCompile's worker pool.

    ``_compile_ktir_with_dbo`` touches no instance state, and constructing the
    real object would spin up compile workers for tests that never compile.
    """
    return ac.SpyreAsyncCompile.__new__(ac.SpyreAsyncCompile)


def _prereqs_met():
    """Patches under which every device prerequisite is satisfied.

    Each test then unsatisfies exactly the one it is about.
    """
    settings = {
        "bundle_symbolic_args": False,
        "ktir_device_mlir": "/nonexistent/device.mlir",
    }
    patches = [mock.patch(f"{_CONFIG}.{k}", v) for k, v in settings.items()]
    patches.append(mock.patch(f"{_MODULE}.shutil.which", return_value="/bin/dbo-opt"))
    return patches


class _PrereqCase(unittest.TestCase):
    """Base class applying/removing the prerequisite patches per test."""

    def setUp(self):
        for patcher in _prereqs_met():
            patcher.start()
            self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output_dir = tmp.name
        self.ktir_path = os.path.join(self.output_dir, "k.ktir")
        with open(self.ktir_path, "w") as fh:
            fh.write("module {}\n")

    def compile(self):
        return _compiler()._compile_ktir_with_dbo(
            "ktir_fused_add_0", self.ktir_path, self.output_dir
        )

    def _write_spyrecode(self):
        code_dir = os.path.join(self.output_dir, "spyreCodeDir")
        os.makedirs(code_dir, exist_ok=True)
        with open(os.path.join(code_dir, "spyrecode.json"), "w") as fh:
            fh.write("{}")


class TestKtirPrerequisites(_PrereqCase):
    """Every unmet prerequisite is named, and named before dbo-opt is run."""

    def test_missing_device_mlir_fails_fast(self):
        with (
            mock.patch(f"{_CONFIG}.ktir_device_mlir", ""),
            mock.patch(f"{_MODULE}.subprocess.run") as run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile()
        self.assertIn("KTIR_DEVICE_MLIR", str(ctx.exception))
        run.assert_not_called()

    def test_symbolic_args_is_a_prerequisite_failure(self):
        with (
            mock.patch(f"{_CONFIG}.bundle_symbolic_args", True),
            mock.patch(f"{_MODULE}.subprocess.run") as run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile()
        message = str(ctx.exception)
        # The footgun this replaced was a dbo-opt exit-1 dump that never
        # mentioned the knob responsible.
        self.assertIn("BUNDLE_SYMBOLIC_ARGS=0", message)
        run.assert_not_called()

    def test_all_unmet_prerequisites_reported_at_once(self):
        """One error naming every unmet prerequisite, not the first one found."""
        with (
            mock.patch(f"{_CONFIG}.bundle_symbolic_args", True),
            mock.patch(f"{_CONFIG}.ktir_device_mlir", ""),
            mock.patch(f"{_MODULE}.shutil.which", return_value=None),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                ac._check_ktir_device_prerequisites()
        message = str(ctx.exception)
        for expected in ("BUNDLE_SYMBOLIC_ARGS=0", "KTIR_DEVICE_MLIR", "dbo-opt"):
            self.assertIn(expected, message)
        # All three, not just the first one found.
        self.assertEqual(message.count("\n  - "), 3)


class TestKtirDboFailures(_PrereqCase):
    def test_nonzero_exit_surfaces_command_and_stderr(self):
        exc = subprocess.CalledProcessError(
            returncode=1, cmd=["dbo-opt"], stderr="error: could not translate\n"
        )
        with mock.patch(f"{_MODULE}.subprocess.run", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile()
        message = str(ctx.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("error: could not translate", message)
        # The full command line, not just the program name.
        self.assertIn("dbo-opt --from-ktir --device=", message)
        self.assertIn(self.ktir_path, message)

    def test_exit_zero_without_spyrecode_is_a_failure(self):
        proc = subprocess.CompletedProcess(
            args=["dbo-opt"], returncode=0, stdout="", stderr="warning: nothing to do\n"
        )
        with mock.patch(f"{_MODULE}.subprocess.run", return_value=proc):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile()
        message = str(ctx.exception)
        self.assertIn("exited 0 but wrote no", message)
        self.assertIn("spyrecode.json", message)
        self.assertIn("warning: nothing to do", message)

    def test_timeout_is_reported_as_a_timeout(self):
        exc = subprocess.TimeoutExpired(cmd=["dbo-opt"], timeout=ac._COMPILE_TIMEOUT_S)
        with mock.patch(f"{_MODULE}.subprocess.run", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile()
        message = str(ctx.exception)
        self.assertIn(f"timed out after {ac._COMPILE_TIMEOUT_S}s", message)
        self.assertIn("dbo-opt --from-ktir --device=", message)


class TestKtirDboSuccess(_PrereqCase):
    def _run_ok(self):
        self._write_spyrecode()
        return subprocess.CompletedProcess(
            args=["dbo-opt"], returncode=0, stdout="", stderr=""
        )

    def test_dbo_opt_inherits_this_process_environment(self):
        """dbo-opt is spawned with no ``env`` override, so it inherits ours.

        Library paths are the user's to export before the run. Passing an
        explicit ``env`` here -- even one built from ``os.environ`` -- is how
        that silently regresses into a stripped or reordered search path, so
        pin the absence of the argument rather than its contents.
        """
        with (
            mock.patch(f"{_MODULE}.SpyreSDSCKernelRunner"),
            mock.patch(
                f"{_MODULE}.subprocess.run", side_effect=lambda *a, **k: self._run_ok()
            ) as run,
        ):
            self.compile()
        self.assertIsNone(run.call_args[1].get("env"))


if __name__ == "__main__":
    unittest.main()

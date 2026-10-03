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

    ``ktir_device_mlir`` is no longer a prerequisite, but is set here so the
    default case carries the ``--device`` override: the command-line assertions
    below would otherwise be testing the flag's absence by accident.
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

    def test_missing_device_mlir_is_not_a_prerequisite_failure(self):
        """An unset device .mlir is compiled, not refused.

        This check used to demand ``KTIR_DEVICE_MLIR``, which duplicated a
        default dbo-opt already has: unset, it takes
        ``sys-arch-spec/KTDFArchGraphDevice/spyre_dd2_basic.mlir`` from under
        ``DEEPTOOLS_PATH``.  As with the symbolic-args prerequisite below, the
        thing worth pinning is the reverse of what was pinned before -- that the
        check stays quiet and dbo-opt is reached.
        """
        with mock.patch(f"{_CONFIG}.ktir_device_mlir", ""):
            ac._check_ktir_device_prerequisites()  # does not raise

    def test_symbolic_args_are_not_a_prerequisite_failure(self):
        """A symbolic base address is compiled, not refused.

        This check used to demand ``BUNDLE_SYMBOLIC_ARGS=0``, because dbo-opt
        needed base addresses baked into constants and the footgun it replaced was
        a dbo-opt exit-1 dump that never mentioned the knob responsible.  The
        backend takes a symbolic start address itself now, so the prerequisite is
        gone -- and what is worth pinning is the reverse of what was pinned before:
        that the check stays quiet and dbo-opt is actually reached.
        """
        with mock.patch(f"{_CONFIG}.bundle_symbolic_args", True):
            ac._check_ktir_device_prerequisites()  # does not raise

    def test_dbo_opt_on_path_is_the_only_prerequisite(self):
        """One entry, and KTIR_DEVICE_MLIR is not it.

        The message keeps its list shape so a future prerequisite reads the same
        way, but the device .mlir must not reappear in it: naming an optional
        setting in a "cannot compile" error sends the reader to configure
        something that was never the problem.  One rather than three now that
        both the symbolic-args and device-.mlir prerequisites are lifted.
        """
        with (
            mock.patch(f"{_CONFIG}.ktir_device_mlir", ""),
            mock.patch(f"{_MODULE}.shutil.which", return_value=None),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                ac._check_ktir_device_prerequisites()
        message = str(ctx.exception)
        self.assertIn("dbo-opt", message)
        self.assertNotIn("KTIR_DEVICE_MLIR", message)
        self.assertEqual(message.count("\n  - "), 1)


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

    def test_unset_device_mlir_omits_the_flag_entirely(self):
        """No ``--device`` argument at all, rather than an empty one.

        ``--device=`` would override dbo-opt's default with the empty string
        instead of leaving it alone, so the absence of the argument is the
        behaviour, not just the absence of a path.
        """
        with (
            mock.patch(f"{_CONFIG}.ktir_device_mlir", ""),
            mock.patch(f"{_MODULE}.SpyreSDSCKernelRunner"),
            mock.patch(
                f"{_MODULE}.subprocess.run", side_effect=lambda *a, **k: self._run_ok()
            ) as run,
        ):
            self.compile()
        cmd = run.call_args[0][0]
        self.assertFalse([a for a in cmd if a.startswith("--device")])
        # The rest of the command is unaffected by the omission.
        self.assertEqual(cmd[:2], ["dbo-opt", "--from-ktir"])
        self.assertIn("--kEmitSpyreCode", cmd)
        self.assertEqual(cmd[-1], self.ktir_path)

    def test_set_device_mlir_is_passed_through(self):
        """The override still reaches dbo-opt when config names a path."""
        with (
            mock.patch(f"{_CONFIG}.ktir_device_mlir", "/some/device.mlir"),
            mock.patch(f"{_MODULE}.SpyreSDSCKernelRunner"),
            mock.patch(
                f"{_MODULE}.subprocess.run", side_effect=lambda *a, **k: self._run_ok()
            ) as run,
        ):
            self.compile()
        self.assertIn("--device=/some/device.mlir", run.call_args[0][0])

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

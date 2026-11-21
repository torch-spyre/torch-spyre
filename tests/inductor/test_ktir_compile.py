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

Covers early core-placement validation, ``_check_ktir_device_prerequisites``,
and ``SpyreAsyncCompile._compile_ktir_with_dbo``. The latter is the step
between the emitter (covered by ``test_ktir_emitter.py``) and the loaded kernel.

No device and no real ``dbo-opt``: ``subprocess.run`` and ``shutil.which`` are
mocked throughout, so unlike the emitter's golden test this file needs neither
``mlir_ktdp`` nor deeptools and runs anywhere.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch._inductor.compile_fx as cfx

from torch_spyre import _inductor as spyre_inductor
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.errors import Unsupported
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
        self.sdsc_bundle_dir_prefix = None
        with open(self.ktir_path, "w") as fh:
            fh.write("module {}\n")

    def compile(self):
        return _compiler()._compile_ktir_with_dbo(
            "ktir_fused_add_0",
            self.ktir_path,
            self.output_dir,
            self.sdsc_bundle_dir_prefix,
        )

    def _write_spyrecode(self):
        code_dir = os.path.join(self.output_dir, "spyreCodeDir")
        os.makedirs(code_dir, exist_ok=True)
        with open(os.path.join(code_dir, "spyrecode.json"), "w") as fh:
            fh.write("{}")


class TestCorePlacementConfiguration(unittest.TestCase):
    """Refuse invalid placement before device initialization or graph lowering."""

    def setUp(self):
        self.enterContext(
            spyre_config.patch(
                sencores=32,
                sencores_physical=32,
                core_placement_mode="dense",
                ktir_emitter=False,
                ktir_device_mlir="",
            )
        )
        self.enterContext(
            mock.patch.dict(
                os.environ,
                {
                    "SENCORES": "32",
                    "SENCORES_PHYSICAL": "32",
                    "SPYRE_CORE_PLACEMENT": "dense",
                    "TORCH_SPYRE_KTIR": "0",
                    "KTIR_DEVICE_MLIR": "",
                },
            )
        )
        self.downstream = self.enterContext(mock.patch.object(cfx, "compile_fx"))
        self.enterContext(mock.patch.object(cfx, "_spyre_wrapped", False, create=True))
        self.enterContext(mock.patch.object(spyre_inductor, "patch_inductor_fusions"))
        self.enterContext(mock.patch.object(spyre_inductor, "try_collect"))
        self.lazy_init = self.enterContext(mock.patch("torch.spyre._impl._lazy_init"))
        spyre_inductor.enable_spyre_compile_fx_wrapper()
        graph = torch.fx.Graph()
        result = graph.call_function(torch.empty, ([1],), {"device": "spyre"})
        graph.output(result)
        self.gm = torch.fx.GraphModule({}, graph)

    def _load_config(self):
        # Match test_async_compile.py's isolated loader. Do not reload the module
        # used by the other tests or start a subprocess for each environment.
        name = "_test_core_placement_config"
        spec = importlib.util.spec_from_file_location(name, spyre_config.__file__)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {name: module}):
            spec.loader.exec_module(module)
            return sys.modules[name]

    def test_environment_rejects_logical_budget_above_physical(self):
        with (
            mock.patch.dict(os.environ, {"SENCORES_PHYSICAL": "16"}),
            self.assertRaisesRegex(
                ValueError,
                "SENCORES must be between 1 and SENCORES_PHYSICAL, "
                "got SENCORES=32, SENCORES_PHYSICAL=16",
            ),
        ):
            self._load_config()

    def test_compile_rejects_patched_logical_budget_above_physical(self):
        with (
            spyre_config.patch(sencores_physical=16),
            self.assertRaisesRegex(
                Unsupported,
                "Spyre backend does not support: SENCORES must be between "
                "1 and SENCORES_PHYSICAL",
            ),
        ):
            cfx.compile_fx(self.gm, [])
        self.lazy_init.assert_not_called()
        self.downstream.assert_not_called()

    def test_environment_rejects_ktir_clamshell(self):
        with (
            mock.patch.dict(
                os.environ,
                {"TORCH_SPYRE_KTIR": "1", "SPYRE_CORE_PLACEMENT": "clamshell"},
            ),
            self.assertRaisesRegex(
                ValueError, "KTIR emission with SPYRE_CORE_PLACEMENT=clamshell"
            ),
        ):
            self._load_config()

    def test_compile_rejects_patched_ktir_clamshell_for_any_core_count(self):
        for cores in (1, 16, 32):
            with (
                self.subTest(cores=cores),
                spyre_config.patch(
                    sencores=cores, core_placement_mode="clamshell", ktir_emitter=True
                ),
                self.assertRaisesRegex(
                    Unsupported,
                    "Spyre backend does not support: KTIR emission with "
                    "SPYRE_CORE_PLACEMENT=clamshell",
                ),
            ):
                cfx.compile_fx(self.gm, [])
        self.lazy_init.assert_not_called()
        self.downstream.assert_not_called()

    def test_device_override_without_ktir_allows_clamshell(self):
        with mock.patch.dict(
            os.environ,
            {
                "SPYRE_CORE_PLACEMENT": "clamshell",
                "KTIR_DEVICE_MLIR": "/some/device.mlir",
            },
        ):
            self.assertFalse(self._load_config().ktir_emitter)
        with (
            spyre_config.patch(
                core_placement_mode="clamshell", ktir_device_mlir="/some/device.mlir"
            ),
            mock.patch.object(spyre_inductor, "_declared_dims_of", return_value={}),
            mock.patch.object(spyre_inductor, "enable_spyre_context"),
        ):
            self.assertIs(cfx.compile_fx(self.gm, []), self.downstream.return_value)
        self.lazy_init.assert_called_once()
        self.downstream.assert_called_once()


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
        exc = subprocess.TimeoutExpired(cmd=["dbo-opt"], timeout=7.0)
        with (
            mock.patch(f"{_CONFIG}.backend_compile_timeout_seconds", 7.0),
            mock.patch(f"{_MODULE}.subprocess.run", side_effect=exc),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile()
        message = str(ctx.exception)
        self.assertIn("timed out after 7.0s", message)
        # Names the knob that relaxes it.
        self.assertIn("SPYRE_BACKEND_COMPILE_TIMEOUT_SECONDS", message)
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

    def test_timeout_comes_from_config(self):
        for configured, expected in ((90.0, 90.0), (0.0, None)):
            with (
                self.subTest(configured=configured),
                mock.patch(f"{_CONFIG}.backend_compile_timeout_seconds", configured),
                mock.patch(f"{_MODULE}.SpyreSDSCKernelRunner"),
                mock.patch(
                    f"{_MODULE}.subprocess.run",
                    side_effect=lambda *a, **k: self._run_ok(),
                ) as run,
            ):
                self.compile()
                self.assertEqual(run.call_args[1]["timeout"], expected)


class TestSdscBundleDirPrefix(_PrereqCase):
    def test_prefix_is_forwarded_to_runner(self):
        self.sdsc_bundle_dir_prefix = "ab12cd34"
        self._write_spyrecode()

        with mock.patch(f"{_MODULE}.subprocess.run"):
            runner = self.compile()

        self.assertEqual(runner.sdsc_bundle_dir_prefix, "ab12cd34")


if __name__ == "__main__":
    unittest.main()

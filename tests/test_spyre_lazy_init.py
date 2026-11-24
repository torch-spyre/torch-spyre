# Copyright 2025 The Torch-Spyre Authors.
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

# Owner(s): ["module: cpp"]

# NOTE: All tests here should run on their own subprocess. Never initialize
# the Spyre runtime in the parent process: opening the card takes exclusive
# VFIO access, so a child subprocess that then tries to open it fails with
# "Device or resource busy".
import os
from unittest import mock

from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import TestCase, run_tests


class TestSpyre(TestCase):
    def setUp(self):
        # TestCase.setUp seeds every device's RNG, and seeding Spyre starts its
        # runtime in this process, which takes the card from the subprocesses.
        with mock.patch.object(common_utils, "set_rng_seed"):
            super().setUp()

    def test_device_layout_available_before_runtime_init(self):
        """The tensor monkey-patch (`.to(device_layout=)` etc.) must be applied
        at import time, before the device runtime is started.

        This is the contract the autoload/lazy-init split relies on: importing
        ``torch_spyre`` installs ``Tensor.to(device_layout=...)``,
        ``Tensor.device_tensor_layout()`` and ``torch.empty(device_layout=)``
        without calling ``start_runtime()``. Run in a fresh process so the
        result is not contaminated by other tests that already started the
        runtime in this session.
        """
        import sys
        import subprocess
        import textwrap

        script = textwrap.dedent("""
            import inspect
            import torch

            # Runtime must NOT have been started just by importing.
            assert torch.spyre.is_initialized() is False, (
                "runtime was initialized during import"
            )

            # The monkey-patch must already be installed.
            assert getattr(torch.Tensor, "_spyre_tensor_patched", False) is True, (
                "tensor patch not applied at import time"
            )

            # `.to` must accept the device_layout kwarg (i.e. it is spyre_to,
            # not the stock Tensor.to).
            assert "device_layout" in inspect.signature(torch.Tensor.to).parameters, (
                "Tensor.to does not accept device_layout after import"
            )

            # device_tensor_layout() must be installed and callable on a plain
            # CPU tensor without touching the runtime (returns None for non-spyre).
            assert hasattr(torch.Tensor, "device_tensor_layout"), (
                "device_tensor_layout not installed at import time"
            )
            assert torch.ones(2).device_tensor_layout() is None

            # torch.empty must accept device_layout too.
            assert "device_layout" in inspect.signature(torch.empty).parameters, (
                "torch.empty does not accept device_layout after import"
            )

            # Still must not have started the runtime as a side effect.
            assert torch.spyre.is_initialized() is False, (
                "runtime was initialized while exercising the patched API"
            )

            print("OK")
        """)

        env = os.environ.copy()
        env["DT_DEEPRT_VERBOSE"] = "-1"
        env["DTLOG_LEVEL"] = "error"

        proc = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=130,
            text=True,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        assert proc.returncode == 0, (
            f"subprocess failed (rc={proc.returncode}).\nstdout:\n{out}\nstderr:\n{err}"
        )
        assert out.endswith("OK"), f"unexpected stdout:\n{out}\nstderr:\n{err}"

    def test_profiler_construction_does_not_start_runtime(self):
        """Constructing ``torch.profiler.profile`` must not start the runtime.

        Autoload patches ``torch.profiler.profile.__init__`` so the first
        construction imports ``_C``, whose static
        ``REGISTER_PRIVATEUSE1_PROFILER`` initializer registers the activity
        profiler. Importing ``_C`` is sufficient; starting the device runtime is
        not, and doing so breaks processes that construct a profiler without
        owning a card -- the vLLM API-server front-end builds a CPU-only
        profiler and would fail to open the VFIO device already held by the
        worker process. Run in a fresh process so an already-started runtime
        from another test cannot mask a regression.
        """
        import sys
        import subprocess
        import textwrap

        script = textwrap.dedent("""
            import torch

            assert torch.spyre.is_initialized() is False, (
                "runtime was initialized during import"
            )

            # CPU-only profiler: nothing here justifies touching the device.
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
            )
            assert torch.spyre.is_initialized() is False, (
                "constructing torch.profiler.profile started the device runtime"
            )

            # The patch is one-shot and restores the original __init__; a second
            # construction must leave the runtime alone too.
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
            )
            assert torch.spyre.is_initialized() is False, (
                "second profile() construction started the device runtime"
            )

            print("OK")
        """)

        env = os.environ.copy()
        env["DT_DEEPRT_VERBOSE"] = "-1"
        env["DTLOG_LEVEL"] = "error"

        proc = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=130,
            text=True,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        assert proc.returncode == 0, (
            f"subprocess failed (rc={proc.returncode}).\nstdout:\n{out}\nstderr:\n{err}"
        )
        assert out.endswith("OK"), f"unexpected stdout:\n{out}\nstderr:\n{err}"

    def test_to_device_layout_first_op_initializes_runtime(self):
        """``tensor.to("spyre", device_layout=...)`` must initialize the Spyre
        runtime even when it is the first Spyre device operation in a fresh
        process (issue #6129).

        The ``device_layout`` branch of ``spyre_to`` calls
        ``spyre_empty_with_layout()`` directly via pybind11, bypassing the ATen
        dispatcher path that ordinarily self-inits the runtime. Without an
        explicit ``_lazy_init()`` first, a fresh process raises
        ``RuntimeError: RuntimeContext not created``.

        Runs in a subprocess so an already-initialized runtime from another
        test cannot mask the regression; a parent-process probe would itself
        start the runtime (and take the VFIO device) before the child runs.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import torch
            import torch_spyre
            from torch_spyre._C import SpyreTensorLayout

            # This must be the first Spyre device operation: do not call
            # anything that touches the runtime before this.
            assert torch.spyre.is_initialized() is False, (
                "runtime was initialized before the device_layout transfer"
            )

            x = torch.ones(4, 64, dtype=torch.float16)
            stl = SpyreTensorLayout([4, 64], torch.float16)
            y = x.to("spyre", device_layout=stl)

            assert y.device.type == "spyre", f"expected spyre device, got {y.device}"
            assert torch.equal(y.cpu(), x), "device_layout transfer corrupted data"
            assert torch.spyre.is_initialized() is True, (
                "device_layout transfer did not initialize the runtime"
            )
            print("OK")
        """)

        env = os.environ.copy()
        env["DT_DEEPRT_VERBOSE"] = "-1"
        env["DTLOG_LEVEL"] = "error"

        proc = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=130,
            text=True,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        assert proc.returncode == 0, (
            f"subprocess failed (rc={proc.returncode}).\nstdout:\n{out}\nstderr:\n{err}"
        )
        assert out.endswith("OK"), f"unexpected stdout:\n{out}\nstderr:\n{err}"

    def test_empty_device_layout_first_op_initializes_runtime(self):
        """``torch.empty(..., device_layout=...)`` must initialize the Spyre
        runtime even when it is the first Spyre device operation in a fresh
        process (issue #6129).

        ``spyre_empty`` calls ``empty_with_layout()`` directly via pybind11,
        bypassing the ATen dispatcher path that ordinarily self-inits the
        runtime. Without an explicit ``_lazy_init()`` first, a fresh process
        raises ``RuntimeError: RuntimeContext not created``.

        Runs in a subprocess so an already-initialized runtime from another
        test cannot mask the regression; a parent-process probe would itself
        start the runtime (and take the VFIO device) before the child runs.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import torch
            import torch_spyre
            from torch_spyre._C import SpyreTensorLayout

            # This must be the first Spyre device operation: do not call
            # anything that touches the runtime before this.
            assert torch.spyre.is_initialized() is False, (
                "runtime was initialized before torch.empty(device_layout=)"
            )

            stl = SpyreTensorLayout([4, 64], torch.float16)
            y = torch.empty(
                (4, 64),
                dtype=torch.float16,
                device=torch.device("spyre"),
                device_layout=stl,
            )

            assert y.device.type == "spyre", f"expected spyre device, got {y.device}"
            assert tuple(y.shape) == (4, 64), f"unexpected shape {tuple(y.shape)}"
            assert y.dtype == torch.float16, f"unexpected dtype {y.dtype}"
            assert torch.spyre.is_initialized() is True, (
                "torch.empty(device_layout=) did not initialize the runtime"
            )
            print("OK")
        """)

        env = os.environ.copy()
        env["DT_DEEPRT_VERBOSE"] = "-1"
        env["DTLOG_LEVEL"] = "error"

        proc = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=130,
            text=True,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        assert proc.returncode == 0, (
            f"subprocess failed (rc={proc.returncode}).\nstdout:\n{out}\nstderr:\n{err}"
        )
        assert out.endswith("OK"), f"unexpected stdout:\n{out}\nstderr:\n{err}"


if __name__ == "__main__":
    run_tests()

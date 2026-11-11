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

# Owner(s): ["module: cpp"]

"""start_runtime() releases the GIL, and device ops racing it wait for it.

The hardware cases run in a fresh process each, so the runtime is not already
started by another test in the session.
"""

import os
import subprocess
import sys
import textwrap

import pytest


def test_start_runtime_releases_gil_marker():
    try:
        from torch_spyre import _C
    except ImportError as e:
        pytest.skip(f"torch_spyre._C not built: {e}")
    assert getattr(_C, "_start_runtime_releases_gil", False) is True


_PRELUDE = """
import sys, threading, time
import torch
import torch_spyre  # noqa: F401
if torch.spyre.device_count() < 1:
    print("NOHW")
    sys.exit(0)
assert not torch.spyre.is_initialized()
errors = []
def init():
    try:
        torch.spyre._impl._lazy_init()
    except BaseException as e:
        errors.append(e)
t = threading.Thread(target=init)
"""


def _run(body: str) -> str:
    env = os.environ.copy()
    env["DT_DEEPRT_VERBOSE"] = "-1"
    env["DTLOG_LEVEL"] = "error"
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PRELUDE) + textwrap.dedent(body)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=300,
        text=True,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"
    if out.endswith("NOHW"):
        pytest.skip("no Spyre device")
    return out


def test_main_thread_runs_python_during_start_runtime():
    out = _run("""
        t0 = time.perf_counter()
        t.start()
        spins = 0
        while t.is_alive():
            spins += 1
        elapsed = time.perf_counter() - t0
        t.join()
        assert not errors, errors
        assert torch.spyre.is_initialized()
        print(f"SPINS {spins} {elapsed:.3f}")
    """)
    spins, elapsed = out.split()[-2:]
    # With the GIL held for the device open the main thread would barely run;
    # released, it spins freely (millions per second).
    if float(elapsed) > 0.5:
        assert int(spins) > 10_000, out


def test_device_ops_racing_start_runtime_wait_for_it():
    _run("""
        t.start()
        # Allocation, H2D copy and stream lookup from the main thread while the
        # runtime is still starting on t: each must wait for it, not race it.
        x = torch.zeros(64, dtype=torch.float16, device="spyre")
        y = torch.arange(64, dtype=torch.float16).to("spyre")
        torch.spyre.synchronize()
        t.join()
        assert not errors, errors
        assert torch.equal(y.cpu(), torch.arange(64, dtype=torch.float16))
        assert torch.count_nonzero(x.cpu()) == 0
        print("OK")
    """)

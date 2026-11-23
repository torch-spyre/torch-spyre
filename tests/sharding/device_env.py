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

"""Environment gating shared by the sharding device tests.

A device test in this suite needs Flex to admit sharded allocations:
``SENCORES`` in (8, 16) plus a device mode that routes to the multi-chunk
scheduler (``FLEX_DEVICE=EMUL1p5``) — mirroring flex's
``admitsShardedAllocations()``. Both are fixed at process start, so the
check is only meaningful in a process launched with that environment (see
``test_sharded_add.test_device_suite_in_emulation_shape``).
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

# Runs in a subprocess (see mode_usable): the smallest operation that forces
# Flex to open the device, so it fails exactly where the device tests would.
_PROBE = (
    "import torch\n"
    "import torch_spyre  # noqa: F401  (registers the spyre device)\n"
    "torch.zeros(64, dtype=torch.float16, device='spyre')\n"
)

# (mode env, SENCORES) -> (usable, reason), so the parametrized wrapper
# cases share one probe per shape.
_probe_cache: dict[tuple[str, int], tuple[bool, str]] = {}


def mode_usable(mode_env: dict[str, str], shards: int) -> tuple[bool, str]:
    """Whether a child process in this device mode can open the card.

    ``emulation_ready()`` only reads the environment; it cannot tell whether
    the machine can actually back the mode (e.g. a PF-only host has a device
    but no VFs bound, so ``FLEX_DEVICE=EMUL1p5`` dies opening the PCI path
    with ``read_symlink ... virtfn1``). This probe runs one device allocation
    in a throwaway process with the mode's environment and reports the
    failure instead of letting the child test suite crash.
    """
    key = (",".join(f"{k}={v}" for k, v in sorted(mode_env.items())), shards)
    if key not in _probe_cache:
        env = dict(os.environ)
        env.update(mode_env)
        env["SENCORES"] = str(shards)
        env.pop("PYTEST_ADDOPTS", None)
        try:
            result = subprocess.run(
                [sys.executable, "-c", _PROBE],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            _probe_cache[key] = (False, "device probe timed out")
            return _probe_cache[key]
        if result.returncode == 0:
            _probe_cache[key] = (True, "")
        else:
            last = (result.stderr or "").strip().splitlines()
            _probe_cache[key] = (
                False,
                last[-1] if last else f"exit {result.returncode}",
            )
    return _probe_cache[key]


def num_shards() -> int:
    return int(os.environ.get("SENCORES", "0") or 0)


def promote_vf_mode_to_emulation() -> bool:
    """Rewrite a VF-mode environment into the emulation shape.

    A VF-mode host can back the emulation device (the wrapper's probe and
    children rely on that), but a process started with ``FLEX_DEVICE=VF``
    loads the real-VF runtime instead of the emulation code. Call this
    before anything loads the Flex runtime — the sharding ``conftest.py``
    does so at import time — so a plain pytest run of ``tests/sharding/``
    runs the whole suite (device tests and device-less tests alike) on
    the emulator. ``SENCORES`` is forced to 8 when it is not already in
    (8, 16) — the shapes that admit sharded allocations — so a full-core
    VF environment (e.g. SENCORES=32) still yields the emulation shape.
    Returns whether anything was rewritten. No-op once the runtime is
    already open (the mode is fixed then), and for non-VF modes (PF stays
    PF: a PF-only host has no VFs to emulate on).
    """
    if os.environ.get("FLEX_DEVICE") != "VF":
        return False
    try:
        if torch.spyre.is_initialized():
            return False
    except Exception:
        # Not loadable here (e.g. a plain-CPU process): nothing is open,
        # so the rewrite is safe.
        pass
    os.environ["FLEX_DEVICE"] = "EMUL1p5"
    shards = num_shards()
    if shards not in (8, 16):
        os.environ["SENCORES"] = "8"
        shards = 8
    # torch_spyre._inductor.config captures SENCORES at import; the repo
    # root's tests/conftest.py can import it before this directory's
    # conftest runs. Fix the cached value too, or to_sharded would still
    # see the ambient core count and reject every shard count.
    config_module = sys.modules.get("torch_spyre._inductor.config")
    if config_module is not None:
        config_module.sencores = shards
    return True


def device_mode_admits_sharded() -> bool:
    # Mirror of flex's admitsShardedAllocations() (runtime_scheduler_factory):
    # the EMUL1p5 device always admits; anything else no.
    return os.environ.get("FLEX_DEVICE") == "EMUL1p5"


def emulation_ready() -> bool:
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0":
        return False
    return num_shards() in (8, 16) and device_mode_admits_sharded()


def device_available() -> bool:
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0":
        return False
    try:
        return torch.spyre.device_count() > 0
    except Exception:
        return False

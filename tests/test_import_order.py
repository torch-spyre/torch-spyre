# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Importing torch_spyre works whether or not torch was imported first (#2925).

Each case runs in a fresh interpreter, because import order only matters in a
process that has imported neither package yet. The test process itself has
already imported torch (and, through backend autoload, torch_spyre).
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_IMPORT_TIMEOUT_S = 300

# Run in the child after the imports under test. Prints one line per fact so a
# failure shows which one was wrong.
_REPORT = """
import sys
import torch
print("spyre_registered", hasattr(torch, "spyre"))
if hasattr(torch, "spyre"):
    print("device", torch.device("spyre").type)
    # Import alone must not start the device runtime (open the card).
    print("runtime_started", torch.spyre._impl._initialized)
ts = sys.modules["torch_spyre"]
print("deferred", ts._autoload_deferred)
print("ran", getattr(ts._autoload, "_ran", False))
"""


def _run(imports: str, autoload: bool) -> dict[str, str]:
    # Import the torch_spyre this process uses, not whatever the child's working
    # directory would shadow it with.
    spec = importlib.util.find_spec("torch_spyre")
    assert spec is not None and spec.origin is not None
    package_root = str(Path(spec.origin).parent.parent)
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="1" if autoload else "0")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (package_root, env.get("PYTHONPATH")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", imports + "\n" + _REPORT],
        capture_output=True,
        cwd=package_root,
        env=env,
        text=True,
        timeout=_IMPORT_TIMEOUT_S,
    )
    assert proc.returncode == 0, proc.stderr
    facts = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" ")
        facts[key] = value
    return facts


@pytest.mark.parametrize(
    "imports, deferred",
    [
        # The #2925 order: torch's autoload hook runs during the `import torch`
        # inside torch_spyre/__init__.py, so the load waits for the module.
        ("import torch_spyre", "True"),
        # torch imports the whole package, then calls the hook itself.
        ("import torch\nimport torch_spyre", "False"),
    ],
    ids=["torch_spyre_first", "torch_first"],
)
def test_import_registers_device(imports, deferred):
    facts = _run(imports, autoload=True)
    assert facts == {
        "spyre_registered": "True",
        "device": "spyre",
        "runtime_started": "False",
        "deferred": deferred,
        "ran": "True",
    }


@pytest.mark.parametrize(
    "imports",
    ["import torch_spyre", "import torch\nimport torch_spyre"],
    ids=["torch_spyre_first", "torch_first"],
)
def test_import_without_autoload_registers_nothing(imports):
    # The xdist controller and the sharding tests rely on this: with autoload
    # off, importing torch_spyre leaves torch.spyre unregistered.
    facts = _run(imports, autoload=False)
    assert facts == {
        "spyre_registered": "False",
        "deferred": "False",
        "ran": "False",
    }

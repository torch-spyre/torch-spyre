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

"""Shared side-effect checks for the CPU-only sharding modules and oracles."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def forbid_distributed(monkeypatch) -> None:
    """Fail any torch.distributed / DeviceMesh entry point called afterwards."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("runtime/distributed entry point called")

    from torch import distributed
    from torch.distributed import device_mesh

    for name in (
        "init_process_group",
        "new_group",
        "get_rank",
        "scatter",
        "gather",
        "all_reduce",
        "broadcast",
        "reduce_scatter",
        "reduce_scatter_tensor",
    ):
        if hasattr(distributed, name):
            monkeypatch.setattr(distributed, name, forbidden)
    monkeypatch.setattr(device_mesh, "DeviceMesh", forbidden)


def subprocess_import_side_effects(import_line: str) -> dict[str, bool]:
    """Import ``import_line`` in a clean subprocess and report side effects.

    Reports whether the import loaded the torch_spyre C++ extension,
    initialized torch.distributed, or registered the spyre runtime.
    """

    code = f"""
import json, sys, torch
import torch.distributed as dist
sys.path.insert(0, r"{Path(__file__).resolve().parent}")
{import_line}
print(json.dumps({{
    'extension': 'torch_spyre._C' in sys.modules,
    'distributed': dist.is_initialized(),
    'spyre_registered': hasattr(torch, 'spyre'),
    'runtime': getattr(torch, 'spyre', None) is not None and
               getattr(torch.spyre, '_initialized', False),
}}))
"""
    environment = os.environ.copy()
    environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    import json

    return json.loads(completed.stdout)

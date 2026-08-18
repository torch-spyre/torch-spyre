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
"""
Wrapper to access Spyre metric file in the old format.
"""

import sys
import sysconfig

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from typing import Iterable

# Assuming config json files are already loaded
from .section_types import MetricDataType

## Try to load libaiumonitor.so from a pre-defined paths
AIUMONITOR_NAME = "libaiumonitor"
AIUMONITOR_PATHS: list[Path] = [
    Path("~/.local/lib").expanduser(),
    Path("/opt/ibm/spyre/aiu-monitor/lib"),
    Path("/opt/ibm/spyre/runtime/lib"),
]

if AIUMONITOR_NAME not in sys.modules:
    fname = AIUMONITOR_NAME + sysconfig.get_config_var("EXT_SUFFIX")
    for libdir in AIUMONITOR_PATHS:
        libfile = libdir / fname
        if not libfile.exists():
            continue

        try:
            spec = spec_from_file_location(AIUMONITOR_NAME, libfile)
            if not spec or not spec.loader:
                continue

            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules[AIUMONITOR_NAME] = mod
            break
        except ImportError:
            # Try next libdir
            pass

try:
    import libaiumonitor as mon
except ImportError as e:
    print(
        "ERROR: Failed to load libaiuminotor.so. Check if ibm-aiu-monitor RPM is installed"
        " and/or PYTHONPATH is set correctly.",
        file=sys.stderr,
    )
    raise RuntimeError(
        "Missing libaiumonitor.so to handle metrics file in the old format"
    ) from e

dummy = mon.DcrHelper.calculate(None, None)
counters = mon.DcrHelper.setupCounters(None)

dtype_names = ["pwr", "tempr", "rdmem", "wrmem", "rxpci", "txpci", "rdrdma", "wrrdma"]
dtype_list = [MetricDataType.find_by_name(n) for n in dtype_names]


def read_old_metrics(path: Path | str) -> Iterable[tuple[MetricDataType, int | float]]:
    snapshot = mon.Snapshot(path)
    values = mon.DcrHelper.calculate(counters[0].value(snapshot), dummy)
    yield from zip(dtype_list, values)

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
"""The table lowering imports where ortools is not installed (s390x, ppc64le): the
solver imports it, so its import must not fail there. This file needs no ortools, so
it has no module-level skip."""

import os
import subprocess
import sys

_WITHOUT_ORTOOLS = """
import sys
sys.modules["ortools"] = None  # importing ortools raises ImportError
from torch_spyre._inductor.scratchpad import cpsat_table_lowering, ilp_solver_ortools
assert cpsat_table_lowering.cp_model is None
assert ilp_solver_ortools.cp_model is None
assert cpsat_table_lowering.lower_cost_with_tables is not None
"""


def test_the_table_module_imports_without_ortools():
    env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    done = subprocess.run(
        [sys.executable, "-c", _WITHOUT_ORTOOLS],
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr

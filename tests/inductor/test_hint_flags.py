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

"""Each kind of spyre_hint has its own switch to ignore it.

The flags are read from the environment when torch_spyre._inductor.config is
imported, so each case runs in a fresh interpreter.
"""

import os
import subprocess
import sys


def _config_value(expression: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "  # torch first: it autoloads the torch_spyre backend
                "from torch_spyre._inductor import config; "
                f"print({expression})"
            ),
        ],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_there_is_no_umbrella_hint_flag():
    assert _config_value("hasattr(config, 'ignore_wsr_hints')", {}) == "False"


def test_work_div_hints_have_their_own_switch():
    env = {"SPYRE_INDUCTOR_IGNORE_WORK_DIV_HINTS": "1"}
    assert _config_value("config.ignore_work_division_hints", env) == "True"
    assert _config_value("config.ignore_work_division_hints", {}) == "False"


def test_the_old_umbrella_variable_is_ignored():
    env = {"SPYRE_INDUCTOR_IGNORE_HINTS": "1"}
    assert _config_value("config.ignore_work_division_hints", env) == "False"
    # span-overflow tiling is opt-in, so it stays ignored by default
    assert _config_value("config.ignore_span_overflow_hints", env) == "True"


def test_span_overflow_switch_does_not_depend_on_the_work_div_switch():
    env = {
        "SPYRE_INDUCTOR_IGNORE_WORK_DIV_HINTS": "1",
        "SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS": "0",
    }
    assert _config_value("config.ignore_span_overflow_hints", env) == "False"
    assert _config_value("config.ignore_work_division_hints", env) == "True"

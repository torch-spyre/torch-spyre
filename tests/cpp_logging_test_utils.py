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

"""Shared utilities for C++ logging subprocess tests."""

import os
import subprocess
import sys
import textwrap


LOGGING_ENV_VARS = [
    "TORCH_SPYRE_DEBUG",
    "SPYRE_INDUCTOR_LOG",
    "SPYRE_INDUCTOR_LOG_LEVEL",
    "SPYRE_LOG_FILE",
    "TORCH_LOGS",
]


def clean_env(overrides: dict) -> dict:
    """Return a copy of the current environment with logging vars stripped."""
    env = os.environ.copy()
    for key in LOGGING_ENV_VARS:
        env.pop(key, None)
    env.update(overrides)
    return env


def run_subprocess(script: str, env_overrides: dict) -> subprocess.CompletedProcess:
    """Run a Python script in an isolated subprocess with controlled env."""
    env = clean_env(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

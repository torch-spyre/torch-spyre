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


"""Integration tests: compile every example and run it on a Spyre device.

Kernels are compiled once per session by examples/examples.py on the current
toolchain. Each example is then checked by examples/check.py, which launches it
through both the spyre launch CLI and the SDK. Every compile and check runs in
its own process, since a device fault poisons the stream for the rest of the
process it happens in. Skipped when no device is present.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from examples import EXAMPLES  # noqa: E402


def _spyre_available():
    probe = "import torch; print(torch.accelerator.current_accelerator())"
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "spyre" in result.stdout


pytestmark = pytest.mark.skipif(not _spyre_available(), reason="needs a Spyre device")


def _run(cmd, timeout):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=EXAMPLES_DIR,
        check=False,
    )
    return result, result.stdout + result.stderr


@pytest.fixture(scope="session")
def kernels(tmp_path_factory):
    """Compile every example once; return (output dir, compile output)."""
    out = tmp_path_factory.mktemp("kernels")
    _, output = _run([sys.executable, "examples.py", "--out", str(out)], 60 * 60)
    return out, output


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example(kernels, name):
    out, compile_output = kernels
    path = out / name
    if not (path / "spyreCodeDir").is_dir():
        lines = [line for line in compile_output.splitlines() if name in line]
        pytest.fail(f"examples.py did not build {name}:\n" + "\n".join(lines))
    result, output = _run([sys.executable, "check.py", name, str(path)], 1200)
    assert result.returncode == 0, output

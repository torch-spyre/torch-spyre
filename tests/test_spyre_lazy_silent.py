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

# NOTE: This test should run on its own, please don't add any other test here
import os

from torch.testing._internal.common_utils import run_tests, TestCase


class TestSpyre(TestCase):
    def test_lazy_import_and_silent(self):
        import sys
        import subprocess
        import textwrap

        # Build a tiny script that only imports torch
        script = textwrap.dedent("""
            import torch  # noqa: F401
        """)

        # Run in a clean environment to reduce noise
        env = os.environ.copy()
        os.environ["TORCH_SENDNN_LOG"] = "CRITICAL"
        os.environ["DT_DEEPRT_VERBOSE"] = "-1"
        os.environ["DTLOG_LEVEL"] = "error"

        # Run a fresh python process and capture both streams
        args = [sys.executable, "-c", script, "2>&1"]
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=130,
            text=True,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        assert "spyre" not in out, f"stdout not empty during import torch:\n{out}"
        assert "senlib" not in out, f"stdout not empty during import torch:\n{out}"
        assert "spyre" not in err, f"stdout not empty during import torch:\n{out}"
        assert "senlib" not in err, f"stdout not empty during import torch:\n{out}"


if __name__ == "__main__":
    run_tests()

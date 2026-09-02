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

import os
import subprocess
import sys
import textwrap

import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSpyreCCLBackend(TestCase):
    def _run_local_rank_script(
        self, local_rank: str
    ) -> subprocess.CompletedProcess[str]:
        script = textwrap.dedent(
            f"""
            import os

            os.environ[\"LOCAL_RANK\"] = {local_rank!r}

            import torch
            import torch_spyre

            torch.ones(1, device=\"spyre\")
            """
        )
        env = os.environ.copy()
        return subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=130,
            text=True,
            check=False,
        )

    def test_spyreccl_device_to_backend(self) -> None:
        # Make sure the module has been loaded
        assert dist.distributed_c10d.is_backend_available("spyreccl")
        # Make sure the module is the default for the spyre device
        assert "spyreccl" == dist.get_default_backend_for_device("spyre")

    def test_parse_local_rank_invalid_text_raises(self) -> None:
        proc = self._run_local_rank_script("invalid")
        output = f"{proc.stdout}\n{proc.stderr}"
        assert proc.returncode != 0, output
        # - Spyre Comms will catch and throw with the message:
        #   LOCAL_RANK must be a valid integer (no digits found): 'invalid'
        # - Torch Spyre will catch and throw with the message:
        #   LOCAL_RANK is not a valid integer: 'invalid'
        assert any(
            text in output
            for text in (
                "LOCAL_RANK must be a valid integer",
                "LOCAL_RANK is not a valid integer",
            )
        ), output

    def test_parse_local_rank_negative_raises(self) -> None:
        proc = self._run_local_rank_script("-1")
        output = f"{proc.stdout}\n{proc.stderr}"
        assert proc.returncode != 0, output
        # Two layers of the library can catch this and may throw different
        # exception messages (account for both):
        # - Spyre Comms will catch and throw with the message:
        #   LOCAL_RANK value overflows unsigned long long: -1
        # - Torch Spyre will catch and throw with the message:
        #   LOCAL_RANK value is out of range: -1
        assert any(
            text in output
            for text in (
                "LOCAL_RANK value is out of range",
                "LOCAL_RANK value overflows",
            )
        ), output

    def test_parse_local_rank_too_large_raises(self) -> None:
        proc = self._run_local_rank_script("128")
        output = f"{proc.stdout}\n{proc.stderr}"
        assert proc.returncode != 0, output
        # Two layers of the library can catch this and may throw different
        # exception messages (account for both):
        # - Spyre Comms will catch and throw with the message:
        #   LOCAL_RANK value 128 exceeds maximum allowed value 3 (range: [0, 3])
        # - Torch Spyre will catch and throw with the message:
        #   LOCAL_RANK value 128 exceeds the maximum supported device index (127)
        assert any(
            text in output
            for text in (
                "exceeds maximum allowed value",
                "exceeds the maximum supported device index",
            )
        ), output


if __name__ == "__main__":
    run_tests()

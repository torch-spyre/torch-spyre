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

"""A stand-in for `torchrun --no-python`: starts one process per rank behind the pin wrapper, LOCAL_RANK set."""

import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--ranks", type=int, required=True)
ap.add_argument("--wrapper", required=True)
ap.add_argument("command", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
procs = [
    subprocess.Popen(
        [a.wrapper, *cmd],
        env={**os.environ, "LOCAL_RANK": str(i), "RANK": str(i), "WORLD_SIZE": str(a.ranks)},
    )
    for i in range(a.ranks)
]
sys.exit(max(p.wait() for p in procs))

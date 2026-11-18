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


"""Run one built example kernel through spyre launch and the spyre_cli SDK.

Usage: python check.py <name> [kernel_dir]    (default: build/<name>)

Two checks, each reported on its own line:

- cli: ``spyre launch`` with the example's -i/-o arguments must exit 0 with no
  device error in its output.
- sdk: ``spyre_cli.launch`` with random inputs must match the CPU reference.

The CLI runs first, in its own process, before this one touches the device.
Run this script in a fresh process per example: a device fault leaves the
stream in an error state for the rest of the process.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from examples import EXAMPLES, cli_args, expected, make_inputs, torch_dtype

DEVICE_ERRORS = ("RAS::", "StreamInErrorState", "DtException")


def check_cli(name, path):
    spyre = shutil.which("spyre")
    if spyre is None:
        return False, "spyre CLI not on PATH"
    result = subprocess.run(
        [spyre, "launch", *cli_args(name), str(path)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    output = result.stdout + result.stderr
    errors = [err for err in DEVICE_ERRORS if err in output]
    if result.returncode != 0 or errors:
        return False, f"exit={result.returncode} errors={errors}\n{output}"
    return True, "exit=0"


def check_sdk(name, path):
    import spyre_cli
    import torch

    ex = EXAMPLES[name]
    inputs = make_inputs(name)
    out = torch.empty(ex.output, dtype=torch_dtype(name), device="spyre")

    runner = spyre_cli.launch(*[t.to("spyre") for t in inputs], out, path=path)
    got = out.cpu()
    del runner

    want = expected(name, inputs)
    max_diff = (got.float() - want.float()).abs().max().item()
    ok = torch.allclose(got, want, atol=ex.atol, rtol=ex.rtol)
    return ok, f"max|diff|={max_diff}"


def main():
    name = sys.argv[1]
    if len(sys.argv) > 2:
        path = Path(sys.argv[2])
    else:
        path = Path(__file__).parent / "build" / name

    ok = True
    for label, check in (("cli", check_cli), ("sdk", check_sdk)):
        passed, detail = check(name, path)
        print(f"{name} {label}: {'PASS' if passed else 'FAIL'} {detail}")
        ok = ok and passed
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

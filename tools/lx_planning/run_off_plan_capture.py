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

"""Capture one existing fixture on one source tree; never launch a matrix."""

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from lx_semantic_plan import capture_off_plan, validate_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("fixture", help="One exact existing pytest node ID")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    validate_contract(contract)
    if contract["fixture"] != args.fixture:
        raise ValueError("fixture does not match the capture contract")
    if args.output.exists():
        raise FileExistsError("preserve the existing capture instead of overwriting it")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError("capture requires a clean tracked source tree")

    class Capture:
        record = None

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            if item.nodeid != args.fixture:
                raise ValueError("unexpected test selection")
            with capture_off_plan(contract, compiler_head=head) as record:
                outcome = yield
                outcome.get_result()
            self.record = record

    plugin = Capture()
    result = pytest.main([args.fixture, "-q"], plugins=[plugin])
    if result != 0 or plugin.record is None:
        raise SystemExit("capture did not finish: no semantic equality is established")
    with args.output.open("x") as output:
        output.write(json.dumps(plugin.record, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()

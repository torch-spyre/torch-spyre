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


"""Update the generated residency-rule table without loading the backend."""

import argparse
from pathlib import Path
import runpy


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(__file__).resolve().parents[2]
    registry = runpy.run_path(
        str(root / "torch_spyre/_inductor/scratchpad/residency_rules.py")
    )
    path = root / "docs/source/compiler/scratchpad_planning.md"
    old = path.read_text()
    start = "<!-- BEGIN GENERATED BUFFER RESIDENCY RULES -->\n"
    end = "<!-- END GENERATED BUFFER RESIDENCY RULES -->"
    before, rest = old.split(start, 1)
    _, after = rest.split(end, 1)
    new = before + start + registry["residency_rules_markdown"]() + end + after
    ownership = runpy.run_path(str(root / "torch_spyre/_inductor/slice_ownership.py"))
    start = "<!-- BEGIN GENERATED OWNERSHIP PROOFS -->\n"
    end = "<!-- END GENERATED OWNERSHIP PROOFS -->"
    before, rest = new.split(start, 1)
    _, after = rest.split(end, 1)
    new = before + start + ownership["ownership_proofs_markdown"]() + end + after
    path.write_text(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

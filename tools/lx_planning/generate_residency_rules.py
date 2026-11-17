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
    path = root / "docs/source/compiler/scratchpad_planning.md"
    new = path.read_text()
    for module, marker, render in (
        (
            "scratchpad/residency_rules.py",
            "BUFFER RESIDENCY RULES",
            "residency_rules_markdown",
        ),
        ("slice_ownership.py", "OWNERSHIP PROOFS", "ownership_proofs_markdown"),
        (
            "scratchpad/residency_rules.py",
            "TRANSFER CERTIFICATES",
            "transfer_certificates_markdown",
        ),
    ):
        registry = runpy.run_path(str(root / "torch_spyre/_inductor" / module))
        start = f"<!-- BEGIN GENERATED {marker} -->\n"
        end = f"<!-- END GENERATED {marker} -->"
        before, rest = new.split(start, 1)
        _, after = rest.split(end, 1)
        new = before + start + registry[render]() + end + after
    path.write_text(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

#!/usr/bin/env python3
"""Extract the torch MAJOR.MINOR (e.g. "2.10") from a torch-spyre pyproject.toml.

Usage:
    get_torch_minor.py <path-to-pyproject.toml>
    get_torch_minor.py -                     # read from stdin

Used to derive the matching pytorch release branch (release/MAJOR.MINOR)
for the pre-cloned pytorch source in the torch-spyre image, keeping it in
sync with the `torch~=X.Y.Z` pin in torch-spyre's pyproject.toml.

This is a port of scripts/get_torch_minor.py in the spyre-runtimes repo —
keep both in sync (the runner image build pipeline also calls this to
pick TORCH_MINOR for the pytorch clone).
"""

from __future__ import annotations

import pathlib
import re  # stdlib; this script is exempted from the import-regex-as-re hook
import sys
import tomllib


def extract_torch_minor(pyproject_bytes: bytes) -> str:
    data = tomllib.loads(pyproject_bytes.decode("utf-8"))
    deps = data.get("project", {}).get("dependencies", [])

    # Match a `torch` requirement but not `torchvision`, `torchaudio`, etc.
    torch_re = re.compile(r"^\s*torch(?!\w)(\s|[<>=!~]|$)")
    torch_specs = [d for d in deps if torch_re.match(d)]

    if not torch_specs:
        raise SystemExit("no torch dependency found in [project.dependencies]")
    if len(torch_specs) > 1:
        raise SystemExit(f"multiple torch dependencies found: {torch_specs}")

    spec = torch_specs[0]
    m = re.search(r"(\d+)\.(\d+)", spec)
    if not m:
        raise SystemExit(f"could not extract MAJOR.MINOR from torch spec: {spec!r}")
    return f"{m.group(1)}.{m.group(2)}"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    arg = argv[1]
    if arg == "-":
        content = sys.stdin.buffer.read()
    else:
        content = pathlib.Path(arg).read_bytes()

    print(extract_torch_minor(content))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

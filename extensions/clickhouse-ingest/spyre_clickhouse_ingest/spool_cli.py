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

"""`spool [list | drop TOKEN]`: look at and clear batches parked in CH_WRITE_SPOOL.

A parked batch is one a write could not land. Nothing sends it again: check the table, send the
rows yourself if they are missing (the file is plain JSON), then `drop` it.
"""

import json
import os
import sys
from pathlib import Path

from .resilience import Spool


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    where = (os.environ.get("CH_WRITE_SPOOL") or "").strip()
    if not where or not Path(where).is_dir():
        print("CH_WRITE_SPOOL is not set to an existing directory", file=sys.stderr)
        return 2
    spool = Spool(where)
    cmd, rest = (argv[0] if argv else "list"), argv[1:]
    if cmd == "list" and not rest:
        for item in spool.items():
            print(json.dumps(item))
        return 0
    if cmd == "drop" and len(rest) == 1:
        try:
            n = spool.drop(rest[0])
        except ValueError as e:
            print(f"drop: {e}", file=sys.stderr)
            return 2
        print(f"drop: {n} item(s)")
        return 0 if n else 1
    print("usage: spool [list | drop TOKEN]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

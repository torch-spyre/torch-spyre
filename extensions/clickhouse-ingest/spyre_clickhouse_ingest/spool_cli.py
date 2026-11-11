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

"""`spool list|release TOKEN|drop TOKEN`: what a person does with held spool items.

A held item is a batch a failed or interrupted attempt may have stored; nothing resends it on its
own. Check the table first, then `release` it (the rows are NOT stored: it is resent on the next
drain) or `drop` it (the rows are stored, or not wanted).
"""

import json
import sys

from . import spool


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    s = spool.from_env("cli")
    if s is None:
        print("CH_WRITE_SPOOL is not set: no spool to look at", file=sys.stderr)
        return 2
    cmd, rest = (argv[0] if argv else "list"), argv[1:]
    if cmd == "list":
        for it in s.items():
            print(json.dumps(it))
        return 0
    if cmd in ("release", "drop") and len(rest) == 1:
        n = s.release_held(rest[0]) if cmd == "release" else s.drop(rest[0])
        print(f"{cmd}: {n} item(s)")
        return 0 if n else 1
    print("usage: spool [list | release TOKEN | drop TOKEN]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

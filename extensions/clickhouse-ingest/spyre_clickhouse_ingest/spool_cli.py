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

"""`spool [list [--fail-if-pending] | replay [TOKEN] | drop TOKEN]`: look at, resend and clear
batches parked in CH_WRITE_SPOOL (else $RUNNER_TEMP/ch-spool).

A parked batch is one a write could not land; nothing sends it again by itself. `replay` sends a
batch to a Replicated table under its original dedup token and removes the file once stored; it
refuses one that a resend could double (see `writes.replay`). For those, check the table, send
the rows yourself if they are missing (the file is plain JSON), then `drop` it.

`list` exits 0 whatever the spool holds, unless `--fail-if-pending` is given: then it exits 75
(EX_TEMPFAIL) when any batch is parked or any quarantined, unreadable or leftover temp file exists.
"""

import json
import sys
from pathlib import Path

from . import writes
from .resilience import Spool, safe

PENDING_EXIT = 75


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    where = writes.spool_dir()
    if not where or not Path(where).is_dir():
        print(
            "no spool: CH_WRITE_SPOOL (or RUNNER_TEMP) is not a directory",
            file=sys.stderr,
        )
        return 2
    spool = Spool(where)
    cmd, rest = (argv[0] if argv else "list"), argv[1:]
    if cmd == "list" and rest in ([], ["--fail-if-pending"]):
        items = spool.items()
        for item in items:
            print(json.dumps(item))
        bad = {k: v for k, v in spool.problems().items() if v}
        for kind, names in bad.items():
            print(
                f"spool: {len(names)} {kind} file(s): {', '.join(names)}",
                file=sys.stderr,
            )
        return PENDING_EXIT if rest and (items or bad) else 0
    if cmd == "drop" and len(rest) == 1:
        try:
            n = spool.drop(rest[0])
        except ValueError as e:
            print(f"drop: {e}", file=sys.stderr)
            return 2
        print(f"drop: {n} item(s)")
        return 0 if n else 1
    if cmd == "replay" and len(rest) <= 1:
        return replay(spool, rest[0] if rest else None)
    print(
        "usage: spool [list [--fail-if-pending] | replay [TOKEN] | drop TOKEN]",
        file=sys.stderr,
    )
    return 2


def replay(spool: Spool, token=None, connect=None) -> int:
    """0 when every selected batch was stored and dropped, 1 otherwise."""
    if connect is None:
        from .client import ClickHouse

        connect = ClickHouse.connect
    todo = [i for i in spool.pending() if token is None or i.token == token]
    if not todo:
        print("replay: nothing to replay", file=sys.stderr)
        return 1 if token else 0
    clients: dict = {}
    failed = 0
    for item in todo:
        try:
            db = item.database or ""
            if db not in clients:
                clients[db] = connect(database=db)
            result = writes.replay(clients[db], item)
        except Exception as err:  # noqa: BLE001 -- reported per batch; the rest still run
            result = f"failed: {safe(err)}"
        if result == "sent":
            spool.drop(item.token)
        else:
            failed += 1
        print(json.dumps({"token": item.token, "table": item.table, "result": result}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

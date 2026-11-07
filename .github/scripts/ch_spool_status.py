#!/usr/bin/env python3
"""Print the spool counters a job summary or a monitor needs: pending, oldest age, bytes."""

import json
import sys

from spyre_clickhouse_ingest import spool


def main(argv=None) -> int:
    s = spool.from_env("gha")
    if s is None:
        print(json.dumps({"ch_spool_pending": 0, "configured": False}))
        return 0
    print(json.dumps({**spool.metrics(s), "configured": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

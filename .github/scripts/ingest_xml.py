#!/usr/bin/env python3
"""Deprecated path of `python -m spyre_clickhouse_ingest results`: forwards argv to it.

Delete once no workflow or Jenkinsfile calls .github/scripts/ingest_xml.py; the grep that finds
the remaining callers is in extensions/clickhouse-ingest/README.md ("Moving off ingest_xml.py").
"""

import sys

from spyre_clickhouse_ingest.results import main

if __name__ == "__main__":
    print(
        "[deprecated] .github/scripts/ingest_xml.py: use "
        "`python -m spyre_clickhouse_ingest results` (same arguments)",
        file=sys.stderr,
    )
    sys.exit(main(sys.argv[1:]))

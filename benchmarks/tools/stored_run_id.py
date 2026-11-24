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

"""The run id the stored rows carry, for the verdict step; never a guess.

    stored_run_id.py RESULTS_DIR            first of run.json's stored_run_ids, else exit 2
    stored_run_id.py --check-verdict FILE   exit 2 when the verdict JSON has no rows

The engine run id is the stored id only when the ingest was handed it, so run.json's `stored_run_ids` (empty when
the engine did not ingest) or an explicit BENCH_RUN_ID are the only sources.
"""

import json
import sys
from pathlib import Path


def stored_run_id(results):
    ids = (
        json.loads((Path(results) / "run.json").read_text()).get("stored_run_ids") or []
    )
    if not ids:
        raise SystemExit(
            "no stored run id known: run.json has no stored_run_ids (the engine did not ingest); "
            "set BENCH_RUN_ID to the id the ingest wrote"
        )
    return ids[0]


def check_verdict(path):
    total = json.loads(Path(path).read_text()).get("summary", {}).get("total", 0)
    if not total:
        raise SystemExit(
            "no verdict rows for this run id (rows did not land, or the id is not the stored one)"
        )
    return total


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--check-verdict"]:
        print(f"verdict rows: {check_verdict(argv[1])}")
    else:
        print(stored_run_id(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

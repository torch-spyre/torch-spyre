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

"""Gather the report.xml of every case that ran ok into one flat directory for the XML ingest.

    collect_ingest_xml.py RESULTS_DIR OUT_DIR [--github-output]

Reads RESULTS_DIR/run.json (spyre-bench `run`), copies cases/<case>/raw/*.xml to OUT_DIR/<case>.xml and prints
`run_id`, `stored_run_id` (first of run.json's stored_run_ids) and `count`; with --github-output they are appended to $GITHUB_OUTPUT.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def collect(results, out):
    run = json.loads((results / "run.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for case in run["cases"]:
        if case.get("status") != "ok":
            continue
        for i, xml in enumerate(sorted((results / case["dir"] / "raw").glob("*.xml"))):
            suffix = "" if i == 0 else f"-{i}"
            shutil.copyfile(xml, out / f"{Path(case['dir']).name}{suffix}.xml")
            copied += 1
    stored = run.get("stored_run_ids") or [""]
    return run["run_id"], stored[0], copied


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--github-output", action="store_true")
    args = ap.parse_args(argv)
    run_id, stored_id, count = collect(args.results, args.out)
    print(f"run_id={run_id}\nstored_run_id={stored_id}\ncount={count}")
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"run_id={run_id}\nstored_run_id={stored_id}\ncount={count}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

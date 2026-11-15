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

"""Run command of an `ops` case: this case's rows of the ops perf report, in the layout the ingest reads.

The report comes from the repo's own perf entry point (`make tests TEST_TYPE=perf`, or BENCH_SUITE_CMD). It covers
every op, so it runs once per --suite-dir and each case keeps only the testcases of its op and shape: the ingest
then writes exactly one row per case and the cases cannot double-count.

    run_kernel_case.py --op OP --shape SHAPES --suite-dir DIR --raw-dir DIR
"""

import argparse
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    "wall_clock",
    "cpu",
    "spyre",
    "kernel",
    "memory_transfer",
    "runtime",
    "compiler",
    "compile",
    "mem_size",
)


def suite_report(suite_dir):
    report = suite_dir / "report.xml"
    if report.is_file():
        return report
    suite_dir.mkdir(parents=True, exist_ok=True)
    cmd = shlex.split(os.environ.get("BENCH_SUITE_CMD", "")) or [
        "make", "-C", str(ROOT), "tests", "TEST_TYPE=perf", f"RESULTS_DIR={suite_dir}",
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    if not report.is_file():
        raise SystemExit(f"the perf suite wrote no {report.name} in {suite_dir}")
    return report


def case_rows(name, op, shape):
    """Whether a testcase name is `perf_<op>_<metric>_<ms|MB>_<shape>` for this op and shape."""
    head = f"perf_{op}_"
    if not name.startswith(head) or not name.endswith(f"_{shape}"):
        return False
    rest = name[len(head) : -len(shape) - 1]
    return any(rest in (f"{m}_ms", f"{m}_MB") for m in METRICS)


def filtered(report, op, shape):
    tree = ET.parse(report)
    kept = 0
    for suite in tree.getroot().iter("testsuite"):
        for tc in list(suite.findall("testcase")):
            if case_rows(tc.get("name", ""), op, shape):
                kept += 1
            else:
                suite.remove(tc)
    return tree, kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--op", required=True)
    ap.add_argument(
        "--shape", required=True, help="the shape string as the report names it"
    )
    ap.add_argument("--suite-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    tree, kept = filtered(suite_report(args.suite_dir), args.op, args.shape)
    if not kept:
        print(
            f"no testcase for op {args.op} with shape {args.shape} in the perf report",
            file=sys.stderr,
        )
        return 3
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    tree.write(args.raw_dir / "report.xml", encoding="utf-8", xml_declaration=True)
    print(f"{args.op}: kept {kept} testcases")
    return 0


if __name__ == "__main__":
    sys.exit(main())

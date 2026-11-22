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

"""Which rows of a perf report the benchmarks/ catalog declares, and which it does not.

    check_report_coverage.py REPORT_DIR [--strict]

Reads REPORT_DIR/report.xml, groups its testcases the way the ingest does (op and shape; model, config and mode) and
compares the groups with the catalog's identity inputs. A report row no case declares would be dropped by a bench
ingest while the legacy ingest stores it, so --strict (the mode that replaces the legacy ingest) exits 1 on any;
without it they are listed and the exit is 0. Catalog cases the report lacks are listed too: a case with no rows
fails at run time. Prints names and counts only.
"""

import argparse
import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def report_rows(report, runner):
    """{row key} of the report: ("op", op, shapes) or ("model", model, config, mode)."""
    rows = set()
    for suite in ET.parse(report).getroot().iter("testsuite"):
        for tc in suite.findall("testcase"):
            m = runner.NAME.match(tc.get("name", ""))
            if not m:
                continue
            tags = runner.tags_of(tc)
            op = m["op"]
            if runner.is_model(tags, op):
                rows.add(
                    (
                        "model",
                        runner.model_name(tags, op),
                        tags.get("config", ""),
                        tags.get("mode", ""),
                    )
                )
            else:
                rows.add(("op", op, m["shapes"] or ""))
    return rows


def catalog_rows(cases):
    rows = set()
    for _, inp, _ in cases:
        disc, name = inp["disc"], inp["name"]
        if disc["record_type"] == "model":
            rows.add(("model", name, disc["config_name"], disc["run_mode"]))
        else:
            rows.add(("op", name, disc["input_shapes"]))
    return rows


def unclaimed_kernels(report_dir, runner, cases):
    """Kernel-breakdown testcases of the directory that no declared case claims (the legacy ingest stores them)."""
    owners = [
        runner.Case(op=inp["name"], shape=inp["disc"]["input_shapes"])
        if inp["disc"]["record_type"] != "model"
        else runner.Case(
            model=inp["name"],
            config=inp["disc"]["config_name"],
            mode=inp["disc"]["run_mode"],
        )
        for _, inp, _ in cases
    ]
    lost = 0
    for xml in sorted(report_dir.glob("*.xml")):
        root = ET.parse(xml).getroot()
        if xml.name == "report.xml":
            continue
        if not runner.is_kernel_xml(root):
            print(
                f"  not a kernel breakdown nor the report, not carried by any case: {xml.name}"
            )
            lost += 1
            continue
        lost += sum(
            not any(o.owns_tags(runner.tags_of(tc)) for o in owners)
            for tc in root.iter("testcase")
        )
    return lost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report_dir", type=Path)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument(
        "--cases-file",
        type=Path,
        help="write the names of the declared cases the report holds, one per line",
    )
    args = ap.parse_args(argv)
    runner = load("run_kernel_case")
    ids = load("check_identity_ids")
    report = args.report_dir / "report.xml"
    if not report.is_file():
        print(f"no report.xml in {args.report_dir}", file=sys.stderr)
        return 2
    have, declared = report_rows(report, runner), catalog_rows(ids.declared())
    cases = ids.declared()
    if args.cases_file:
        args.cases_file.write_text(
            "".join(
                f"{name}\n"
                for name, inp, _ in cases
                if next(iter(catalog_rows([(name, inp, None)]))) in have
            )
        )
    undeclared, absent = sorted(have - declared), sorted(declared - have)
    print(
        f"report rows {len(have)}; declared by the catalog {len(have & declared)}; "
        f"not declared {len(undeclared)}; declared but absent from the report {len(absent)}"
    )
    for label, rows in (("not declared", undeclared), ("absent", absent)):
        for row in rows:
            print(f"  {label}: {' | '.join(row)}")
    lost = unclaimed_kernels(args.report_dir, runner, cases)
    if lost:
        print(f"  {lost} kernel-breakdown testcases match no declared case")
    return 1 if (undeclared or lost) and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())

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

"""Run command of a perf case: this case's rows of the perf report, in the layout the ingest reads.

The report comes from the repo's own perf entry point (`make tests TEST_TYPE=perf`, or BENCH_SUITE_CMD). It covers
every op and model, so it runs once per --suite-dir and each case keeps only its own testcases: the ingest then
writes exactly one row per case and the cases cannot double-count. BENCH_SUITE_FROM names a directory that already
holds the report (and any per-kernel breakdown XMLs) of a finished perf run; it is copied instead of running the suite.

    run_kernel_case.py --op OP --shape SHAPES --suite-dir DIR --raw-dir DIR
    run_kernel_case.py --model MODEL --config CONFIG --mode MODE --suite-dir DIR --raw-dir DIR
"""

import argparse
import json
import os
import regex as re
import shlex
import shutil
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
# The ingest's own name pattern: perf_<op>_<metric>_<ms|MB>[_<shapes>].
NAME = re.compile(
    rf"^perf_(?P<op>.+?)_(?P<metric>{'|'.join(METRICS)})_(?:ms|MB)(?:_(?P<shapes>.+))?$"
)
KERNEL_CLASS = "kernel_benchmark"
# Report metric -> the measurement key the ingest stores it under.
SAMPLE_KEYS = {
    "wall_clock": "total_duration_ms",
    "cpu": "cpu_ms",
    "spyre": "spyre_ms",
    "kernel": "kernel_mean_ms",
    "memory_transfer": "memory_transfer_mean_ms",
    "runtime": "runtime_ms",
    "compiler": "compile_ms",
    "compile": "compile_ms",
    "mem_size": "mem_size_mb",
}
# The suite writes 0.0 for these when it captured nothing; the ingest drops them.
ZERO_WHEN_UNCAPTURED = ("pt_util_percent", "compile_ms", "runtime_ms")


def suite_report(suite_dir):
    report = suite_dir / "report.xml"
    if report.is_file():
        return report
    suite_dir.mkdir(parents=True, exist_ok=True)
    finished = os.environ.get("BENCH_SUITE_FROM")
    if finished:
        for xml in sorted(Path(finished).glob("*.xml")):
            shutil.copyfile(xml, suite_dir / xml.name)
    else:
        cmd = shlex.split(os.environ.get("BENCH_SUITE_CMD", "")) or [
            "make", "-C", str(ROOT), "tests", "TEST_TYPE=perf", f"RESULTS_DIR={suite_dir}",
        ]  # fmt: skip
        subprocess.run(cmd, check=True)
    if not report.is_file():
        raise SystemExit(f"the perf suite wrote no {report.name} in {suite_dir}")
    return report


def tags_of(tc):
    """The testcase's `tag` properties, `key__value` split on the first `__` (the ingest's reading)."""
    out = {}
    for p in tc.iter("property"):
        value = p.get("value", "").strip()
        if p.get("name") == "tag" and "__" in value:
            key, _, val = value.partition("__")
            out[key] = val
    return out


def is_model(tags, op):
    """The ingest's model test: the explicit model tag, or the granite_ op prefix of older reports."""
    return bool(tags.get("model")) or op.startswith("granite_")


def model_name(tags, op):
    return "granite" if op.startswith("granite_") else tags.get("model", "")


def ints(text):
    """The integers of a shape string: a name spells `[[1, 512], [512, 8]]` where a kernel tag may spell `1x512`."""
    return tuple(re.findall(r"\d+", text or ""))


class Case:
    """Which rows belong to one case: an op at a shape, or a model at a config and mode."""

    def __init__(self, op=None, shape=None, model=None, config=None, mode=None):
        self.op, self.shape = op, shape
        self.model, self.config, self.mode = model, config, mode

    def describe(self):
        if self.model:
            return f"model {self.model} config {self.config} mode {self.mode}"
        return f"op {self.op} with shape {self.shape}"

    def owns_tags(self, tags):
        """Whether a testcase's tags put it in this case (kernel breakdowns and model rows)."""
        op = tags.get("op", "")
        if self.model:
            return (
                is_model(tags, op)
                and model_name(tags, op) == self.model
                and tags.get("config") == self.config
                and tags.get("mode") == self.mode
            )
        return (
            not is_model(tags, op)
            and op == self.op
            and ints(tags.get("input_shape")) == ints(self.shape)
        )


def report_groups(suite, case):
    """The (op, shapes) groups of the report that are this case's."""
    groups = set()
    for tc in suite.findall("testcase"):
        m = NAME.match(tc.get("name", ""))
        if not m:
            continue
        key = (m["op"], m["shapes"] or "")
        if case.model:
            if case.owns_tags(tags_of(tc)):
                groups.add(key)
        elif key == (case.op, case.shape) or (
            key[0].lower() == case.op.lower() and key[1] == case.shape
        ):
            groups.add(key)
    return groups


def filtered(report, case):
    """(tree, kept): the report with only the case's testcases."""
    tree = ET.parse(report)
    kept = 0
    for suite in tree.getroot().iter("testsuite"):
        groups = report_groups(suite, case)
        for tc in list(suite.findall("testcase")):
            m = NAME.match(tc.get("name", ""))
            if m and (m["op"], m["shapes"] or "") in groups:
                kept += 1
            else:
                suite.remove(tc)
    return tree, kept


def uncaptured(tree):
    """Measurement keys the whole report zero-fills (the ingest drops them for every record of the file)."""
    seen = {}
    for tc in tree.getroot().iter("testcase"):
        m = NAME.match(tc.get("name", ""))
        key = SAMPLE_KEYS.get(m["metric"]) if m else None
        if key in ZERO_WHEN_UNCAPTURED:
            seen.setdefault(key, []).append(tc.get("time"))
        util = tags_of(tc).get("pt_util")
        if util is not None:
            seen.setdefault("pt_util_percent", []).append(util)
    zero = set()
    for key, vals in seen.items():
        try:
            if key in ZERO_WHEN_UNCAPTURED and all(float(v) == 0 for v in vals):
                zero.add(key)
        except ValueError:
            pass
    return zero


def samples_of(tree, drop=()):
    """{measurement key: [value]} of the kept testcases, as the ingest would store them."""
    out = {}
    for tc in tree.getroot().iter("testcase"):
        m = NAME.match(tc.get("name", ""))
        if not m:
            continue
        try:
            out.setdefault(SAMPLE_KEYS[m["metric"]], [float(tc.get("time", ""))])
        except ValueError:
            continue
        try:
            out.setdefault("pt_util_percent", [float(tags_of(tc)["pt_util"])])
        except (KeyError, ValueError):
            pass
    if "total_duration_ms" not in out and "cpu_ms" in out:
        out["total_duration_ms"] = out["cpu_ms"]
    return {k: v for k, v in out.items() if k not in drop}


def is_kernel_xml(root):
    cases = root.findall(".//testcase")
    return bool(cases) and all(KERNEL_CLASS in tc.get("classname", "") for tc in cases)


def kernel_breakdowns(suite_dir, report, case):
    """[(name, tree, kept)] for each per-kernel breakdown XML of the suite that holds rows of this case."""
    out = []
    for xml in sorted(suite_dir.glob("*.xml")):
        if xml == report:
            continue
        tree = ET.parse(xml)
        if not is_kernel_xml(tree.getroot()):
            continue
        kept = 0
        for suite in tree.getroot().iter("testsuite"):
            for tc in list(suite.findall("testcase")):
                if case.owns_tags(tags_of(tc)):
                    kept += 1
                else:
                    suite.remove(tc)
        if kept:
            out.append((xml.name, tree, kept))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--op")
    ap.add_argument(
        "--shape", help="the shape string as the report names it (with --op)"
    )
    ap.add_argument("--model")
    ap.add_argument("--config", help="the model's config_name (with --model)")
    ap.add_argument("--mode", help="the model's run_mode (with --model)")
    ap.add_argument("--suite-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument(
        "--repeat-dir",
        type=Path,
        help="write metrics.json here for the engine's own results (the bench-dir ingest)",
    )
    args = ap.parse_args(argv)
    if bool(args.op) == bool(args.model):
        ap.error("give --op with --shape, or --model with --config and --mode")
    if (
        args.op
        and args.shape is None
        or args.model
        and None in (args.config, args.mode)
    ):
        ap.error("--op needs --shape; --model needs --config and --mode")
    case = Case(args.op, args.shape, args.model, args.config, args.mode)
    report = suite_report(args.suite_dir)
    tree, kept = filtered(report, case)
    drop = uncaptured(ET.parse(report))
    if not kept:
        print(f"no testcase for {case.describe()} in the perf report", file=sys.stderr)
        return 3
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    tree.write(args.raw_dir / "report.xml", encoding="utf-8", xml_declaration=True)
    if args.repeat_dir:
        args.repeat_dir.mkdir(parents=True, exist_ok=True)
        (args.repeat_dir / "metrics.json").write_text(
            json.dumps({"samples": samples_of(tree, drop)})
        )
    kernels = kernel_breakdowns(args.suite_dir, report, case)
    for name, ktree, _ in kernels:
        ktree.write(args.raw_dir / name, encoding="utf-8", xml_declaration=True)
    print(
        f"{args.op or args.model}: kept {kept} testcases and "
        f"{sum(k for _, _, k in kernels)} kernel testcases"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

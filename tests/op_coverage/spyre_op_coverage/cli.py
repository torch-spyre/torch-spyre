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

"""Command line for the Spyre op-coverage report.

Run it through ``tests/op_coverage/op_coverage.py``, which imports ``torch`` before
``torch_spyre`` (the backend entry point needs that order)::

    # A YOLO model from its Ultralytics config (no weight download)
    python tests/op_coverage/op_coverage.py ultralytics yolov5n.yaml

    # A Hugging Face LLM, two layers, prefill plus a few decode steps
    python tests/op_coverage/op_coverage.py hf ibm-granite/granite-3.3-8b-instruct \\
        --opt layers=2 --opt mode=generate

    # Any model: a function returning (model, args)
    python tests/op_coverage/op_coverage.py module my_pkg.models:build

    # An op YAML from hf-adapters' utils/model_ops
    python tests/op_coverage/op_coverage.py model-ops-yaml \\
        tests/configs/model_ops_tests/Meta-Llama-3.1-8B-Instruct.yaml

    # Re-render a saved JSON report
    python tests/op_coverage/op_coverage.py json op_coverage_reports/yolov5n.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Optional

from . import recipes

SOURCES = {
    **{name: help_text for name, (_, help_text) in recipes.RECIPES.items()},
    "model-ops-yaml": "An op YAML written by hf-adapters utils/model_ops (target: path).",
    "json": "A report saved by this tool (target: path); re-rendered only.",
}


def _options(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--opt expects KEY=VALUE, got {pair!r}")
        out[key.strip()] = value.strip()
    return out


def _progress(quiet: bool):
    if quiet or not sys.stderr.isatty():
        return None

    def report(done: int, total: int, what: str) -> None:
        print(
            f"\r[{done + 1}/{total}] {what[:60]:60s}",
            end="",
            file=sys.stderr,
            flush=True,
        )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tests/op_coverage/op_coverage.py",
        description="Report which ops a model needs and how torch-spyre handles them.",
        epilog="Sources:\n"
        + "\n".join(f"  {name:15s} {text}" for name, text in SOURCES.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source", choices=sorted(SOURCES), help="where the model comes from"
    )
    parser.add_argument("target", help="model id, path, or package.module:function")
    parser.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="recipe option, repeatable (dtype=float16, layers=2, fuse=false, ...)",
    )
    parser.add_argument("-o", "--output-dir", default="op_coverage_reports")
    parser.add_argument("--name", help="model name in the report and file names")
    parser.add_argument(
        "--format", default="md,json", help="comma-separated: md, json (default both)"
    )
    parser.add_argument(
        "--notes",
        action="append",
        default=[],
        type=pathlib.Path,
        metavar="FILE",
        help="extra known issues, YAML or JSON: {op: {note: ..., links: [...]}}",
    )
    parser.add_argument(
        "--docs-table",
        type=pathlib.Path,
        help="supported_operations.md to read (default: the one in this checkout)",
    )
    parser.add_argument("--no-docs", action="store_true", help="ignore the docs table")
    parser.add_argument(
        "--cast-float",
        metavar="DTYPE",
        help="model-ops-yaml: treat float tensors as this dtype (e.g. float16)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="compile and run each signature on the device (needs a Spyre card)",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="compile and run the whole model on the device (recipe sources)",
    )
    parser.add_argument("--device", default="spyre", help="device for --probe/--e2e")
    parser.add_argument("--probe-timeout", type=float, default=300.0, metavar="SEC")
    parser.add_argument("--max-probes", type=int, metavar="N")
    parser.add_argument(
        "--fail-on",
        choices=[
            "unsupported",
            "unknown",
            "unverified",
            "cpu_fallback",
            "eager_fallback",
        ],
        help="exit 1 if any op is this bad or worse (for CI)",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    ns = build_parser().parse_args(argv)
    formats = {f.strip() for f in ns.format.split(",") if f.strip()}
    progress = _progress(ns.quiet)

    from .analyze import analyze_model, build_report
    from .capture import CaptureResult
    from .knowledge import Knowledge
    from .model import CoverageReport, Support
    from .render import to_json, to_markdown

    options = _options(ns.opt)
    if ns.source == "json":
        report = CoverageReport.from_dict(
            json.loads(pathlib.Path(ns.target).read_text())
        )
    else:
        knowledge = Knowledge.load(ns.docs_table, ns.notes, use_docs=not ns.no_docs)
        if ns.source == "model-ops-yaml":
            from . import model_ops_yaml
            from .serialize import dtype_of

            cast = dtype_of(ns.cast_float) if ns.cast_float else None
            name, calls, warnings = model_ops_yaml.load(pathlib.Path(ns.target), cast)
            settings = {"cast_float": ns.cast_float} if ns.cast_float else {}
            report = build_report(
                ns.name or name,
                f"model_ops YAML {ns.target}",
                CaptureResult(calls, 0, {}, collections.Counter()),
                settings=settings,
                knowledge=knowledge,
                progress=progress,
            )
            report.warnings.extend(warnings)
        else:
            spec = recipes.load(ns.source, ns.target, options)
            settings = {"recipe": ns.source, **options}
            if spec.notes:
                settings["prepared"] = "; ".join(spec.notes)
            report = analyze_model(
                spec.model,
                spec.args,
                spec.kwargs,
                name=ns.name or spec.name,
                run=spec.run,
                settings=settings,
                knowledge=knowledge,
                progress=progress,
            )
            report.source = f"{ns.source} {ns.target}"
        if ns.probe or ns.e2e:
            report.settings["device"] = ns.device
        if ns.probe:
            from .probe import run_probes

            run_probes(
                report,
                device=ns.device,
                timeout=ns.probe_timeout,
                limit=ns.max_probes,
                progress=progress,
            )
        if ns.e2e:
            if ns.source in recipes.RECIPES:
                from .probe import run_e2e

                report.e2e = run_e2e(ns.source, ns.target, options, device=ns.device)
            else:
                report.warnings.append("--e2e needs a recipe source; skipped.")
    if progress:
        print(file=sys.stderr)

    out_dir = pathlib.Path(ns.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in report.model)
    written = []
    if "md" in formats:
        path = out_dir / f"{stem}.md"
        path.write_text(to_markdown(report))
        written.append(path)
    if "json" in formats:
        path = out_dir / f"{stem}.json"
        path.write_text(to_json(report))
        written.append(path)
    if not ns.quiet:
        counts: dict[str, int] = {}
        for op in report.ops:
            counts[op.status.label] = counts.get(op.status.label, 0) + 1
        summary = ", ".join(f"{n} {label.lower()}" for label, n in counts.items())
        print(f"{report.model}: {len(report.ops)} ops ({summary})")
        for path in written:
            print(f"wrote {path}")

    if ns.fail_on:
        threshold = Support(ns.fail_on).severity
        if any(op.status.severity >= threshold for op in report.ops):
            return 1
    return 0

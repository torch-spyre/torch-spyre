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

"""Render a :class:`CoverageReport` as Markdown or JSON."""

from __future__ import annotations

import collections
import json
from typing import Iterable

from .model import CoverageReport, LeafOp, Support, TorchOpRecord


def to_json(report: CoverageReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=False) + "\n"


def short(op: str) -> str:
    """``aten.bmm.default`` -> ``aten.bmm``; other overloads stay."""
    return op[: -len(".default")] if op.endswith(".default") else op


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def badge(status: Support) -> str:
    return f"{status.icon} {status.label}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _link(url: str) -> str:
    tail = url.rstrip("/").rsplit("/", 2)
    if len(tail) == 3 and tail[1] in ("issues", "pull"):
        return f"[#{tail[2]}]({url})"
    return f"<{url}>"


def first_decompositions(record: TorchOpRecord) -> list[tuple[str, str]]:
    steps: dict[tuple[str, str], None] = {}
    for leaf in record.leaves():
        if leaf.path and leaf.path[0].source != "hop":
            steps[(leaf.path[0].op, leaf.path[0].source)] = None
    return list(steps)


def summary_notes(record: TorchOpRecord) -> list[str]:
    """Short notes for the main table."""
    notes = []
    decomps = first_decompositions(record)
    by_source: dict[str, list[str]] = collections.defaultdict(list)
    for op, source in decomps:
        by_source[source].append(f"`{short(op)}`")
    if by_source.get("spyre"):
        notes.append("Spyre decomposition of " + ", ".join(by_source["spyre"]))
    if by_source.get("inductor"):
        notes.append("Inductor decomposition of " + ", ".join(by_source["inductor"]))
    if by_source.get("composite"):
        notes.append("PyTorch composite of " + ", ".join(by_source["composite"]))

    worst: dict[Support, list[str]] = collections.defaultdict(list)
    for leaf in record.leaves():
        status = leaf.verdict.status
        if status in (Support.SPYRE, Support.VIEW):
            continue
        name = f"`{short(leaf.op)}`"
        if name not in worst[status]:
            worst[status].append(name)
    for status in sorted(worst, key=lambda s: -s.severity):
        notes.append(f"{status.label}: " + ", ".join(worst[status][:4]))

    statuses = {sig.status for sig in record.signatures}
    if len(statuses) > 1:
        bad = sum(1 for sig in record.signatures if sig.status is record.status)
        notes.append(
            f"varies by shape ({bad} of {len(record.signatures)} signatures "
            f"{record.status.label.lower()})"
        )
    errors = {sig.error for sig in record.signatures if sig.error}
    if errors and record.status is Support.UNKNOWN:
        notes.append(sorted(errors)[0])
    dtype_notes = {n for leaf in record.leaves() for n in leaf.verdict.notes}
    notes.extend(sorted(dtype_notes))
    for issue in record.known_issues:
        links = " ".join(_link(u) for u in issue.links)
        notes.append(f"{issue.note} {links}".strip())
    return notes


def probe_summary(record: TorchOpRecord) -> str:
    """``3/4 pass, 1 compile_error`` over a torch op's probed signatures."""
    outcomes = collections.Counter(
        sig.probe.outcome for sig in record.signatures if sig.probe
    )
    if not outcomes:
        return "—"
    total = sum(outcomes.values())
    parts = [f"{outcomes.pop('pass', 0)}/{total} pass"]
    parts += [f"{n} {outcome}" for outcome, n in outcomes.most_common()]
    return ", ".join(parts)


def _signature_rows(record: TorchOpRecord, limit: int) -> list[str]:
    rows = [
        "| Signature | Calls | Compiled | Leaf ops |",
        "|---|---:|---|---|",
    ]
    for sig in sorted(record.signatures, key=lambda s: -s.count)[:limit]:
        leaves = ", ".join(
            f"{leaf.verdict.status.icon} `{short(leaf.op)}`"
            for leaf in _unique(sig.leaves)
        )
        if sig.error and not sig.leaves:
            leaves = _cell(sig.error)
        rows.append(
            f"| `{_cell(sig.inputs)}` | {sig.count} | {badge(sig.status)} | {leaves} |"
        )
    hidden = len(record.signatures) - limit
    if hidden > 0:
        rows.append(f"| … {hidden} more | | | |")
    return rows


def _unique(leaves: Iterable[LeafOp]) -> list[LeafOp]:
    seen: dict[str, LeafOp] = {}
    for leaf in leaves:
        current = seen.get(leaf.op)
        if (
            current is None
            or leaf.verdict.status.severity > current.verdict.status.severity
        ):
            seen[leaf.op] = leaf
    return list(seen.values())


def _tree(record: TorchOpRecord) -> list[str]:
    """Decomposition tree merged over all signatures."""
    root: dict = {}
    leaf_info: dict[tuple, LeafOp] = {}
    for leaf in record.leaves():
        node = root
        for step in leaf.path:
            node = node.setdefault((step.op, step.source), {})
        key = ("leaf", leaf.op)
        prev = leaf_info.get(tuple(s.op for s in leaf.path) + key)
        if prev is None or leaf.verdict.status.severity > prev.verdict.status.severity:
            leaf_info[tuple(s.op for s in leaf.path) + key] = leaf
        node.setdefault(key, {})

    lines: list[str] = []

    def walk(node: dict, depth: int, prefix: tuple) -> None:
        for key, child in node.items():
            indent = "  " * depth
            if key[0] == "leaf":
                leaf = leaf_info[prefix + key]
                line = (
                    f"{indent}- {leaf.verdict.status.icon} `{short(leaf.op)}` — "
                    f"{leaf.verdict.mechanism}"
                )
                if leaf.verdict.notes:
                    line += " (" + "; ".join(leaf.verdict.notes) + ")"
                lines.append(line)
            else:
                op, source = key
                how = {
                    "spyre": "Spyre decomposition",
                    "inductor": "Inductor decomposition",
                    "composite": "PyTorch composite (CompositeImplicitAutograd)",
                    "hop": "higher-order op body",
                }.get(source, source)
                lines.append(f"{indent}- `{short(op)}` → {how}")
                walk(child, depth + 1, prefix + (op,))

    walk(root, 0, ())
    return lines


def _details(record: TorchOpRecord, signature_limit: int) -> list[str]:
    out = [
        "<details>",
        f"<summary><code>{record.name}</code> — {badge(record.status)} — "
        f"{_plural(record.count, 'call')}, "
        f"{_plural(len(record.signatures), 'signature')}</summary>",
        "",
    ]
    top = sorted({op for sig in record.signatures for op in sig.top_level})
    if top:
        out.append("Dispatches to " + ", ".join(f"`{short(op)}`" for op in top) + ".")
        out.append("")
    out.extend(_tree(record))
    out.append("")
    eager = [sig.eager for sig in record.signatures if sig.eager]
    if eager:
        worst = max(eager, key=lambda v: v.status.severity)
        out.append(f"Eager: {badge(worst.status)} — {worst.mechanism}")
        out.append("")
    if record.documented:
        doc = record.documented
        compiled = "compiled" if doc.compiled else "not compiled"
        eager_doc = "eager" if doc.eager else "no eager"
        line = f"Documented ({doc.category}): {compiled}, {eager_doc}, runs on {doc.execution or '?'}"
        if doc.notes:
            line += f" — {doc.notes}"
        out.append(line)
        out.append("")
    if record.modules:
        out.append("Modules: " + ", ".join(f"`{m}`" for m in record.modules))
    if record.locations:
        out.append("Called from: " + "; ".join(f"`{loc}`" for loc in record.locations))
    if record.modules or record.locations:
        out.append("")
    out.extend(_signature_rows(record, signature_limit))
    probes = [sig for sig in record.signatures if sig.probe]
    if probes:
        out.append("")
        out.append("| Probe signature | Outcome | Detail |")
        out.append("|---|---|---|")
        for sig in probes[:signature_limit]:
            assert sig.probe is not None
            detail = sig.probe.detail
            if sig.probe.fallbacks:
                detail += " Fallbacks: " + ", ".join(sig.probe.fallbacks)
            out.append(
                f"| `{_cell(sig.inputs)}` | {sig.probe.outcome} | {_cell(detail)} |"
            )
    out += ["", "</details>", ""]
    return out


def _gap_section(report: CoverageReport) -> list[str]:
    from .analyze import blockers

    items = blockers(report)
    if not items:
        return ["Every op the model calls lowers to Spyre.", ""]
    out = []
    by_status: dict[Support, list[TorchOpRecord]] = collections.defaultdict(list)
    for record in items:
        by_status[record.status].append(record)
    for status in sorted(by_status, key=lambda s: -s.severity):
        out.append(f"### {status.icon} {status.label}")
        out.append("")
        out.append(status.description)
        out.append("")
        for record in by_status[status]:
            out.append(f"- **`{record.name}`** ({_plural(record.count, 'call')})")
            shown = set()
            for leaf in record.leaves():
                if leaf.verdict.status is not status or leaf.op in shown:
                    continue
                shown.add(leaf.op)
                via = " → ".join(f"`{short(s.op)}`" for s in leaf.path)
                via = f" via {via}" if via else ""
                out.append(f"  - `{short(leaf.op)}`{via}: {leaf.verdict.mechanism}")
            if status is Support.UNKNOWN:
                for err in sorted({s.error for s in record.signatures if s.error})[:2]:
                    out.append(f"  - {err}")
            examples = [s.inputs for s in record.signatures if s.status is status][:2]
            for example in examples:
                out.append(f"  - e.g. `{example}`")
            if record.modules:
                out.append("  - in " + ", ".join(f"`{m}`" for m in record.modules[:3]))
            for issue in record.known_issues:
                links = " ".join(_link(u) for u in issue.links)
                out.append(f"  - {issue.note} {links}".rstrip())
        out.append("")
    return out


def to_markdown(report: CoverageReport, *, signature_limit: int = 12) -> str:
    from .analyze import summarize

    env = report.environment
    out = [f"# Spyre op coverage: {report.model}", ""]
    meta = [
        ("Source", report.source),
        (
            "Captured",
            f"{_plural(report.graphs, 'graph')}, "
            f"{_plural(sum(report.graph_breaks.values()), 'graph break')}",
        ),
    ]
    if report.settings:
        meta.append(
            ("Settings", ", ".join(f"{k}={v}" for k, v in report.settings.items()))
        )
    versions = (
        f"torch {env.get('torch', '?')}, torch-spyre {env.get('torch_spyre', '?')}"
    )
    if env.get("torch_spyre_commit"):
        versions += f" ({env['torch_spyre_commit']})"
    meta.append(("Versions", versions))
    meta.append(("Generated", report.created))
    out += ["| | |", "|---|---|"]
    out += [f"| {k} | {_cell(v)} |" for k, v in meta]
    out.append("")

    counts = summarize(report)
    calls = sum(op.count for op in report.ops)
    out += ["## Summary", ""]
    out.append(
        f"The model calls {_plural(len(report.ops), 'distinct torch op')} "
        f"({_plural(calls, 'call')}), which reach Inductor as "
        f"{_plural(len(report.aten_ops), 'distinct ATen op')}."
    )
    blocked = [s for s in (Support.UNSUPPORTED, Support.UNKNOWN) if counts[s]]
    if blocked:
        out.append(
            "Compiling it on Spyre is expected to fail until the "
            + " and ".join(f"{s.label.lower()} ops" for s in blocked)
            + " below are resolved."
        )
    elif counts[Support.UNVERIFIED]:
        out.append("Some ops use in-tree lowerings that are not validated on Spyre.")
    out.append("")
    out += ["| Status | Ops | Meaning |", "|---|---:|---|"]
    for status in Support:
        if counts[status]:
            out.append(f"| {badge(status)} | {counts[status]} | {status.description} |")
    out.append("")
    for warning in report.warnings:
        out.append(f"> {warning}")
        out.append("")
    device = report.settings.get("device", "spyre")
    if report.e2e:
        out.append(
            f"Whole model compiled and run on `{device}` (`--e2e`): "
            f"**{report.e2e.outcome}**"
            + (f" — {_cell(report.e2e.detail)}" if report.e2e.detail else "")
        )
        out.append("")

    probed = any(sig.probe for op in report.ops for sig in op.signatures)
    out += [f"## What does {report.model} require?", ""]
    header = "| Op | Calls | Compiled path | Eager | Runs on |"
    rule = "|---|---:|---|---|---|"
    if probed:
        header += f" Probe on `{device}` |"
        rule += "---|"
    out += [header + " Notes |", rule + "---|"]
    for record in report.ops:
        notes = "<br>".join(_cell(n) for n in summary_notes(record))
        row = (
            f"| `{record.name}` | {record.count} | {badge(record.status)} | "
            f"{badge(record.eager_status)} | {record.execution} |"
        )
        if probed:
            row += f" {probe_summary(record)} |"
        out.append(row + f" {notes} |")
    out.append("")

    out += ["## Gaps", ""]
    out += _gap_section(report)

    out += ["## Op details", ""]
    for record in report.ops:
        out += _details(record, signature_limit)

    out += ["## ATen ops that reach Inductor", ""]
    out += [
        "| ATen op | Status | How | Signature uses | Used by |",
        "|---|---|---|---:|---|",
    ]
    for aten in report.aten_ops:
        used = ", ".join(f"`{u}`" for u in aten.used_by[:4])
        if len(aten.used_by) > 4:
            used += f" +{len(aten.used_by) - 4}"
        out.append(
            f"| `{short(aten.op)}` | {badge(aten.verdict.status)} | "
            f"{_cell(aten.verdict.mechanism)} | {aten.count} | {used} |"
        )
    out.append("")

    if report.graph_breaks:
        out += ["## Graph breaks", "", "| Reason | Count |", "|---|---:|"]
        for reason, count in report.graph_breaks.items():
            out.append(f"| {_cell(reason[:200])} | {count} |")
        out.append("")

    out += [
        "## How to read this report",
        "",
        "Each torch-level call is traced on fake Spyre tensors through Spyre's "
        "decomposition table; every ATen op left is then checked against the "
        "lowerings, CPU fallbacks and eager kernels torch-spyre registers. A "
        "torch op gets the worst status of its ATen ops. *Needs validation* "
        "means an in-tree Inductor lowering exists that no Spyre table or the "
        "supported-operations page vouches for; run with `--probe` on a Spyre "
        "host to settle those. Graph-level FX passes are not modeled per op.",
        "",
    ]
    return "\n".join(out)

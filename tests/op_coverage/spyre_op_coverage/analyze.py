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

"""Stage 3: group the captured calls and build the report."""

from __future__ import annotations

import collections
import dataclasses
import datetime
import platform
import subprocess
from typing import Any, Callable, Iterable, Optional

import torch

from .capture import CapturedCall, CaptureResult, capture
from .knowledge import Knowledge
from .model import AtenOpRecord, CoverageReport, Support, TorchOpRecord, Verdict
from .registry import SpyreRegistry
from .trace import Tracer

Progress = Callable[[int, int, str], None]


@dataclasses.dataclass
class _Group:
    name: str
    count: int = 0
    signatures: dict[str, list[Any]] = dataclasses.field(default_factory=dict)
    modules: collections.Counter = dataclasses.field(
        default_factory=collections.Counter
    )
    locations: collections.Counter = dataclasses.field(
        default_factory=collections.Counter
    )


@dataclasses.dataclass
class Analysis:
    ops: list[TorchOpRecord]
    aten_ops: list[AtenOpRecord]
    # Calls that do no tensor work (torch._C._log_api_usage_once, ...).
    skipped: list[str]


def _has_tensor_args(call: CapturedCall) -> bool:
    from torch.utils import _pytree as pytree

    from .capture import TensorMeta

    leaves = pytree.tree_leaves((call.args, call.kwargs))
    return any(isinstance(x, TensorMeta) for x in leaves)


def analyze_calls(
    calls: Iterable[CapturedCall],
    *,
    registry: Optional[SpyreRegistry] = None,
    knowledge: Optional[Knowledge] = None,
    progress: Optional[Progress] = None,
) -> Analysis:
    """Trace each unique call signature once and roll the results up."""
    registry = registry or SpyreRegistry()
    knowledge = knowledge or Knowledge.load()
    tracer = Tracer(registry)

    groups: dict[str, _Group] = {}
    for call in calls:
        group = groups.setdefault(call.name, _Group(call.name))
        group.count += 1
        entry = group.signatures.setdefault(call.signature_key(), [call, 0])
        entry[1] += 1
        if call.module:
            group.modules[call.module] += 1
        if call.location:
            group.locations[call.location] += 1

    total = sum(len(g.signatures) for g in groups.values())
    done = 0
    records = []
    skipped = []
    for group in groups.values():
        record = TorchOpRecord(
            name=group.name,
            count=group.count,
            modules=[m for m, _ in group.modules.most_common(5)],
            locations=[loc for loc, _ in group.locations.most_common(3)],
        )
        for call, count in group.signatures.values():
            if progress:
                progress(done, total, group.name)
            analysis = tracer.analyze(call, count)
            done += 1
            if not (analysis.leaves or analysis.error or _has_tensor_args(call)):
                continue
            analysis.replay = replay_spec(call)
            for leaf in analysis.leaves:
                knowledge.refine(leaf)
            record.signatures.append(analysis)
        if not record.signatures:
            skipped.append(group.name)
            continue
        record.documented = knowledge.doc_for(group.name)
        leaf_names = {leaf.op for leaf in record.leaves()}
        record.known_issues = knowledge.issues_for([group.name, *sorted(leaf_names)])
        records.append(record)
    return Analysis(records, _aten_records(records), skipped)


def _aten_records(records: list[TorchOpRecord]) -> list[AtenOpRecord]:
    by_op: dict[str, AtenOpRecord] = {}
    for record in records:
        for sig in record.signatures:
            for leaf in sig.leaves:
                current = by_op.get(leaf.op)
                if current is None:
                    current = by_op[leaf.op] = AtenOpRecord(leaf.op, leaf.verdict, 0)
                elif leaf.verdict.status.severity > current.verdict.status.severity:
                    current.verdict = leaf.verdict
                current.count += sig.count
                if record.name not in current.used_by:
                    current.used_by.append(record.name)
    return sorted(by_op.values(), key=lambda r: (-r.verdict.status.severity, r.op))


def replay_spec(call: CapturedCall) -> Optional[dict[str, Any]]:
    """Enough of a call to rebuild it in another process (see probe)."""
    from .serialize import encode

    target = call.target
    if target is None:
        return None
    if call.kind == "call_function":
        target = torch.fx.node._get_qualified_name(target)
    return {
        "name": call.name,
        "kind": call.kind,
        "target": target,
        "args": encode(call.args),
        "kwargs": encode(call.kwargs),
    }


def environment() -> dict[str, str]:
    import torch_spyre

    env = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
    }
    try:
        from importlib.metadata import version

        env["torch_spyre"] = version("torch_spyre")
    except Exception:  # noqa: BLE001 - not installed as a distribution
        env["torch_spyre"] = "source checkout"
    try:
        root = torch_spyre.__file__.rsplit("/torch_spyre/", 1)[0]
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if sha:
            env["torch_spyre_commit"] = sha
    except Exception:  # noqa: BLE001 - not a git checkout
        pass
    return env


def build_report(
    model: str,
    source: str,
    captured: CaptureResult,
    *,
    settings: Optional[dict[str, str]] = None,
    registry: Optional[SpyreRegistry] = None,
    knowledge: Optional[Knowledge] = None,
    progress: Optional[Progress] = None,
) -> CoverageReport:
    analysis = analyze_calls(
        captured.calls, registry=registry, knowledge=knowledge, progress=progress
    )
    report = CoverageReport(
        model=model,
        source=source,
        created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        environment=environment(),
        settings=dict(settings or {}),
        graphs=captured.graphs,
        graph_breaks=dict(captured.graph_breaks),
        ops=analysis.ops,
        aten_ops=analysis.aten_ops,
    )
    if analysis.skipped:
        report.warnings.append(
            "Skipped calls that do no tensor work: "
            + ", ".join(f"`{name}`" for name in analysis.skipped)
        )
    if captured.graph_breaks:
        report.warnings.append(
            f"Dynamo broke the model into {captured.graphs} graphs (see Graph "
            "breaks). The code at each break runs eagerly, outside any graph, "
            "and its ops are not in this report."
        )
    return report


def analyze_model(
    model: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: Optional[dict[str, Any]] = None,
    *,
    name: Optional[str] = None,
    run: Optional[Callable[[Callable[..., Any]], Any]] = None,
    settings: Optional[dict[str, str]] = None,
    knowledge: Optional[Knowledge] = None,
    progress: Optional[Progress] = None,
) -> CoverageReport:
    """Capture ``model``'s ops on the CPU and report how Spyre handles them.

    By default the compiled model is called once with ``args`` and ``kwargs``.
    To cover a ``generate`` loop or similar, pass ``run``: it gets the model
    with its ``forward`` compiled (or the compiled function, if ``model`` is
    not a module) and drives it.
    """
    kwargs = kwargs or {}
    if isinstance(model, torch.nn.Module):
        module = model
        original = module.forward

        def drive(compiled: Callable[..., Any]) -> Any:
            module.forward = compiled  # type: ignore[method-assign]
            try:
                return run(module) if run else module(*args, **kwargs)
            finally:
                module.forward = original  # type: ignore[method-assign]

        captured = capture(drive, original, root=module)
    else:
        captured = capture(run or (lambda compiled: compiled(*args, **kwargs)), model)
    label = name or type(model).__name__
    return build_report(
        label,
        "torch.compile capture",
        captured,
        settings=settings,
        knowledge=knowledge,
        progress=progress,
    )


def summarize(report: CoverageReport) -> dict[Support, int]:
    counts: collections.Counter = collections.Counter(op.status for op in report.ops)
    return {status: counts.get(status, 0) for status in Support}


def blockers(report: CoverageReport) -> list[TorchOpRecord]:
    """Ops that are not plainly on Spyre, worst first."""
    risky = [op for op in report.ops if op.status not in (Support.SPYRE, Support.VIEW)]
    return sorted(risky, key=lambda op: (-op.status.severity, -op.count))


def leaf_verdicts(record: TorchOpRecord) -> dict[str, Verdict]:
    """The worst verdict per leaf op of a torch-level op."""
    out: dict[str, Verdict] = {}
    for leaf in record.leaves():
        current = out.get(leaf.op)
        if current is None or leaf.verdict.status.severity > current.status.severity:
            out[leaf.op] = leaf.verdict
    return out

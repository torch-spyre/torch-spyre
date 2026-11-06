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

"""Data model of an op-coverage report.

Everything here is plain data, so a report can be written to JSON, read back
and rendered again (or diffed against an earlier one) without torch-spyre.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Optional


class Support(enum.Enum):
    """How an op reaches Spyre, from best to worst.

    The order is the roll-up order: an op that decomposes into several ATen
    ops gets the worst status among them.
    """

    SPYRE = "spyre"
    VIEW = "view"
    EAGER_FALLBACK = "eager_fallback"
    CPU_FALLBACK = "cpu_fallback"
    UNVERIFIED = "unverified"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"

    @property
    def severity(self) -> int:
        return list(Support).index(self)

    @property
    def label(self) -> str:
        return _LABELS[self][0]

    @property
    def icon(self) -> str:
        return _LABELS[self][1]

    @property
    def description(self) -> str:
        return _LABELS[self][2]

    @staticmethod
    def worst(statuses) -> "Support":
        statuses = list(statuses)
        if not statuses:
            return Support.UNKNOWN
        return max(statuses, key=lambda s: s.severity)

    @staticmethod
    def rollup(statuses) -> "Support":
        """The worst status, except that views only count when alone.

        ``linear`` lowers to a transpose and a matmul; it is a Spyre op, not
        a view op.
        """
        statuses = list(statuses)
        compute = [s for s in statuses if s is not Support.VIEW]
        return Support.worst(compute) if compute else Support.worst(statuses)


_LABELS = {
    Support.SPYRE: (
        "Spyre",
        "✅",
        "Lowered to a Spyre kernel (a Spyre lowering, or an Inductor lowering "
        "whose ops the Spyre code generator implements).",
    ),
    Support.VIEW: (
        "View",
        "🟦",
        "View or layout op handled by Inductor's view lowering and Spyre layout "
        "propagation. Some shapes are not stick-aligned and can be rejected.",
    ),
    Support.EAGER_FALLBACK: (
        "Eager fallback",
        "🟨",
        "No lowering; Inductor calls the op's Spyre eager kernel, which splits "
        "the compiled graph.",
    ),
    Support.CPU_FALLBACK: (
        "CPU fallback",
        "🟧",
        "Runs on the CPU: tensors are copied to the host and back.",
    ),
    Support.UNVERIFIED: (
        "Needs validation",
        "⚠️",
        "An upstream Inductor lowering exists, but nothing shows that Spyre "
        "supports it.",
    ),
    Support.UNKNOWN: (
        "Unknown",
        "❓",
        "The analysis could not trace the op; see the error.",
    ),
    Support.UNSUPPORTED: (
        "Not supported",
        "❌",
        "No lowering and no Spyre kernel, or the lowering emits an op that "
        "Spyre code generation does not implement.",
    ),
}


@dataclasses.dataclass
class Verdict:
    """The status of one ATen op, with the reason."""

    status: Support
    mechanism: str
    notes: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DecompStep:
    """One decomposition applied on the way from a torch op to a leaf."""

    op: str
    source: str  # "spyre" or "inductor"


@dataclasses.dataclass
class LeafOp:
    """An ATen op that reaches Inductor lowering after all decompositions."""

    op: str
    verdict: Verdict
    path: list[DecompStep] = dataclasses.field(default_factory=list)
    input_dtypes: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ProbeResult:
    """Outcome of compiling one op signature on a Spyre device."""

    outcome: str  # pass, mismatch, compile_error, runtime_error, crash, skipped
    detail: str = ""
    max_abs_diff: Optional[float] = None
    fallbacks: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SignatureAnalysis:
    """One unique call signature of a torch-level op, and how it lowers."""

    inputs: str
    count: int
    top_level: list[str] = dataclasses.field(default_factory=list)
    leaves: list[LeafOp] = dataclasses.field(default_factory=list)
    eager: Optional[Verdict] = None
    error: Optional[str] = None
    probe: Optional[ProbeResult] = None
    # Positional args as JSON-able data, for re-running the signature (probe).
    replay: Optional[dict[str, Any]] = None

    @property
    def status(self) -> Support:
        if self.error and not self.leaves:
            return Support.UNKNOWN
        if not self.leaves:
            # Traced to nothing: an identity view such as x.view(x.shape).
            return Support.VIEW
        return Support.rollup(leaf.verdict.status for leaf in self.leaves)


@dataclasses.dataclass
class DocEntry:
    """A row of docs/source/user_guide/supported_operations.md."""

    names: list[str]
    category: str
    eager: bool
    compiled: bool
    execution: str
    notes: str


@dataclasses.dataclass
class KnownIssue:
    """A curated note about an op: an open issue, a PR, a caveat."""

    note: str
    links: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TorchOpRecord:
    """A torch-level op as the model calls it, over all its signatures."""

    name: str
    count: int
    signatures: list[SignatureAnalysis] = dataclasses.field(default_factory=list)
    modules: list[str] = dataclasses.field(default_factory=list)
    locations: list[str] = dataclasses.field(default_factory=list)
    documented: Optional[DocEntry] = None
    known_issues: list[KnownIssue] = dataclasses.field(default_factory=list)

    @property
    def status(self) -> Support:
        return Support.rollup(sig.status for sig in self.signatures)

    @property
    def eager_status(self) -> Support:
        return Support.worst(
            sig.eager.status if sig.eager else Support.UNKNOWN
            for sig in self.signatures
        )

    @property
    def execution(self) -> str:
        """Where the op runs when compiled: Spyre, CPU, or both."""
        places = set()
        for sig in self.signatures:
            for leaf in sig.leaves:
                status = leaf.verdict.status
                if status is Support.CPU_FALLBACK:
                    places.add("CPU")
                elif status in (Support.SPYRE, Support.VIEW, Support.EAGER_FALLBACK):
                    places.add("Spyre")
                else:
                    places.add("?")
        if not places:
            # Only identity calls (x.contiguous() on a contiguous x).
            return "—"
        if places == {"?"}:
            return "?"
        return " + ".join(sorted(places - {"?"}))

    def leaves(self) -> list[LeafOp]:
        return [leaf for sig in self.signatures for leaf in sig.leaves]


@dataclasses.dataclass
class AtenOpRecord:
    """A unique leaf ATen op over the whole model."""

    op: str
    verdict: Verdict
    count: int
    used_by: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CoverageReport:
    """The op coverage of one model on Spyre."""

    model: str
    source: str
    created: str
    environment: dict[str, str] = dataclasses.field(default_factory=dict)
    settings: dict[str, str] = dataclasses.field(default_factory=dict)
    graphs: int = 0
    graph_breaks: dict[str, int] = dataclasses.field(default_factory=dict)
    ops: list[TorchOpRecord] = dataclasses.field(default_factory=list)
    aten_ops: list[AtenOpRecord] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)
    e2e: Optional[ProbeResult] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoverageReport":
        return _from_jsonable(cls, data)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {
            f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
        # Derived fields are written too, so a JSON consumer needs no logic.
        for derived in ("status", "eager_status", "execution"):
            if isinstance(getattr(type(obj), derived, None), property):
                out[derived] = _to_jsonable(getattr(obj, derived))
        return out
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable(cls: Any, data: Any) -> Any:
    import typing

    if data is None:
        return None
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _convert(hints[f.name], data[f.name])
    return cls(**kwargs)


def _convert(hint: Any, value: Any) -> Any:
    import typing

    if value is None:
        return None
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Union:
        inner = [a for a in args if a is not type(None)]
        return _convert(inner[0], value)
    if origin is list:
        return [_convert(args[0], v) for v in value]
    if origin is dict:
        return {k: _convert(args[1], v) for k, v in value.items()}
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return hint(value)
    if dataclasses.is_dataclass(hint):
        return _from_jsonable(hint, value)
    return value

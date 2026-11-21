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

"""Human knowledge layered over the registry verdicts.

Two sources, both optional:

* The supported-operations table in
  ``docs/source/user_guide/supported_operations.md``. Its notes are shown next
  to the op, and an in-tree lowering the table lists as compiled on Spyre is
  reported as supported rather than as needing validation.
* Known issues: open issues, PRs in flight, caveats. ``KNOWN_ISSUES`` below
  holds the shared ones; ``--notes FILE`` (YAML or JSON with the same shape)
  adds team-local ones. Remove an entry when its issue closes.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable, Optional

import regex as re

from .model import DocEntry, KnownIssue, LeafOp, Support, Verdict

_TS = "https://github.com/torch-spyre/torch-spyre"

# Keyed by torch-level name (as captured) or ATen op name, with or without the
# overload (``aten.max_pool2d_with_indices``). Keep entries short and linked.
KNOWN_ISSUES: dict[str, list[KnownIssue]] = {
    "torch.nn.functional.max_pool2d": [
        KnownIssue("Spyre max_pool2d support is in review.", [f"{_TS}/pull/4348"])
    ],
    "aten.max_pool2d_with_indices": [
        KnownIssue("Spyre max_pool2d support is in review.", [f"{_TS}/pull/4348"])
    ],
    "spyre.reshape_via_cpu": [
        KnownIssue(
            "Reshapes whose new innermost dim is not stick-aligned run on the CPU; "
            "general views are tracked in #1353.",
            [f"{_TS}/issues/1353"],
        )
    ],
}


def _default_docs_path() -> Optional[pathlib.Path]:
    # <repo>/tests/op_coverage/spyre_op_coverage/knowledge.py
    root = pathlib.Path(__file__).resolve().parents[3]
    path = root / "docs" / "source" / "user_guide" / "supported_operations.md"
    return path if path.is_file() else None


_ROW = re.compile(
    r"^\|\s*(?P<ops>[^|]+?)\s*\|\s*(?P<eager>Y?)\s*\|\s*(?P<compiled>Y?)\s*"
    r"\|\s*(?P<execution>[^|]*?)\s*\|\s*(?P<notes>.*?)\s*\|\s*$"
)
_CATEGORY = re.compile(r"^\|\s*\*\*(?P<name>[^*]+)\*\*")
_NAME = re.compile(r"`([^`]+)`")


def parse_docs_table(text: str) -> list[DocEntry]:
    """Rows of the supported-operations table."""
    entries = []
    category = ""
    for line in text.splitlines():
        heading = _CATEGORY.match(line)
        if heading:
            category = heading.group("name").strip()
            continue
        row = _ROW.match(line)
        if not row or "`" not in row.group("ops"):
            continue
        names = [n.strip() for n in _NAME.findall(row.group("ops"))]
        entries.append(
            DocEntry(
                names=names,
                category=category,
                eager=row.group("eager") == "Y",
                compiled=row.group("compiled") == "Y",
                execution=row.group("execution").strip(),
                notes=row.group("notes").strip(),
            )
        )
    return entries


def name_aliases(name: str) -> list[str]:
    """Spellings under which a torch-level or ATen op may be documented."""
    if name.startswith(("aten.", "prims.")):
        packet = name.split(".")[1]
        bases = [packet, packet.lstrip("_")]
    else:
        base = name
        for prefix in ("torch.nn.functional.", "torch.Tensor.", "torch."):
            if base.startswith(prefix):
                base = base[len(prefix) :]
                break
        bases = [base, base.rstrip("_")]
    aliases = [name]
    for base in bases:
        for prefix in ("torch.", "torch.nn.functional.", "torch.Tensor."):
            aliases.append(prefix + base)
    return list(dict.fromkeys(aliases))


class Knowledge:
    """Docs-table entries and known issues, looked up by op name."""

    def __init__(
        self,
        docs: Iterable[DocEntry] = (),
        issues: Optional[dict[str, list[KnownIssue]]] = None,
    ) -> None:
        self.docs: dict[str, DocEntry] = {}
        for entry in docs:
            for name in entry.names:
                self.docs.setdefault(name, entry)
        self.issues = dict(KNOWN_ISSUES if issues is None else issues)

    @classmethod
    def load(
        cls,
        docs_path: Optional[pathlib.Path] = None,
        notes_paths: Iterable[pathlib.Path] = (),
        use_docs: bool = True,
    ) -> "Knowledge":
        docs: list[DocEntry] = []
        path = docs_path or (_default_docs_path() if use_docs else None)
        if path is not None:
            docs = parse_docs_table(path.read_text())
        issues = {k: list(v) for k, v in KNOWN_ISSUES.items()}
        for notes_path in notes_paths:
            for key, items in load_notes(notes_path).items():
                issues.setdefault(key, []).extend(items)
        return cls(docs, issues)

    def doc_for(self, name: str) -> Optional[DocEntry]:
        for alias in name_aliases(name):
            if alias in self.docs:
                return self.docs[alias]
        return None

    def issues_for(self, names: Iterable[str]) -> list[KnownIssue]:
        found: list[KnownIssue] = []
        for name in names:
            keys = [name]
            if name.count(".") >= 2 and name.split(".")[0] in (
                "aten",
                "prims",
                "spyre",
            ):
                keys.append(name.rsplit(".", 1)[0])
            for key in keys:
                for issue in self.issues.get(key, []):
                    if issue not in found:
                        found.append(issue)
        return found

    def refine(self, leaf: LeafOp) -> None:
        """Trust the docs table for in-tree lowerings it lists as compiled."""
        if leaf.verdict.status is not Support.UNVERIFIED:
            return
        entry = self.doc_for(leaf.op)
        if entry and entry.compiled and entry.execution.lower().startswith("spyre"):
            leaf.verdict = Verdict(
                Support.SPYRE,
                leaf.verdict.mechanism.replace(", not verified on Spyre", "")
                + "; listed as compiled in supported_operations.md",
                leaf.verdict.notes,
            )


def load_notes(path: pathlib.Path) -> dict[str, list[KnownIssue]]:
    """Read a notes file: ``{op: {note: str, links: [str]} | [ ... ]}``."""
    text = path.read_text()
    data: Any
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    out: dict[str, list[KnownIssue]] = {}
    for key, value in data.items():
        items = value if isinstance(value, list) else [value]
        out[str(key)] = [
            KnownIssue(
                str(item.get("note", "")), [str(x) for x in item.get("links", [])]
            )
            for item in items
        ]
    return out

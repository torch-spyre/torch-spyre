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

"""Compare the private Inductor code the LoopNode scheduler patch depends on.

``torch_spyre/_inductor/loop_scheduler.py`` replaces four ``Scheduler`` methods
and relies on a number of others (``PATCHED_SURFACE`` lists them all).  Inductor
is private API, so a torch upgrade can change any of them.  This tool answers
"which of those functions changed between the old and the new torch?" so that an
upgrade re-verifies only those.

Usage:
    diff_inductor_surface.py <old torch/_inductor dir> <new torch/_inductor dir>

Each argument is a directory holding ``scheduler.py`` and ``ir.py``: the
``site-packages/torch/_inductor`` of an environment, or the same directory of a
checkout of the matching ``vX.Y.Z`` tag.  To get the old release's files without
an old environment::

    for f in scheduler.py ir.py; do
      curl -sSfLo old/$f \\
        https://raw.githubusercontent.com/pytorch/pytorch/vX.Y.0/torch/_inductor/$f
    done

``PATCHED_SURFACE`` is read from ``loop_scheduler.py`` as source, not imported,
so the tool works while the new torch is installed and the native extension is
not rebuilt yet (``import torch_spyre`` fails in that window).

Output: one line per entry, ``SAME`` or ``CHANGED`` with a unified diff, then
``NOT FOUND`` for names present in neither tree, then a summary.  Exit status is
0 when nothing changed, 1 when at least one entry changed, 2 on a usage error.

PyTorch 2.14 rewrote every ``assert C, msg`` as ``if not C: raise
AssertionError(msg)``.  That is not a change in behaviour, so both forms are
normalised to one before comparing, and comments, blank lines and docstrings
are dropped.  The normaliser is a heuristic for exactly that rewrite; read any
CHANGED diff before concluding something moved.
"""

from __future__ import annotations

import ast
import difflib
import sys
import textwrap
from pathlib import Path

import regex as re

LOOP_SCHEDULER = (
    Path(__file__).resolve().parent.parent
    / "torch_spyre"
    / "_inductor"
    / "loop_scheduler.py"
)

_INVERSE_OPS = {
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


class _Canon(ast.NodeTransformer):
    """Rewrite ``assert C`` and ``if <cmp>: raise AssertionError`` to one form."""

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        self.generic_visit(node)
        return ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=node.test),
            body=[
                ast.Raise(exc=ast.Name(id="AssertionError", ctx=ast.Load()), cause=None)
            ],
            orelse=[],
        )

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        raises_assertion = (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Raise)
            and not node.orelse
            and isinstance(node.body[0].exc, ast.Call)
            and getattr(node.body[0].exc.func, "id", None) == "AssertionError"
        )
        if not raises_assertion:
            return node
        node.body = [
            ast.Raise(exc=ast.Name(id="AssertionError", ctx=ast.Load()), cause=None)
        ]
        test = node.test
        # ``if not C: raise`` is ``assert C`` and already canonical.  A single
        # comparison ``if A <op> B: raise`` is ``assert A <inverse op> B``.
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return node
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and type(test.ops[0]) in _INVERSE_OPS
        ):
            inverse = _INVERSE_OPS[type(test.ops[0])]()
            node.test = ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Compare(
                    left=test.left, ops=[inverse], comparators=test.comparators
                ),
            )
        else:
            node.test = ast.UnaryOp(
                op=ast.Not(), operand=ast.UnaryOp(op=ast.Not(), operand=test)
            )
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        # Docstrings and other bare string expressions.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return node


def _normalise(src: str) -> str:
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return "\n".join(
            line.rstrip()
            for line in src.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return ast.unparse(ast.fix_missing_locations(_Canon().visit(tree)))


def _definitions(path: Path) -> dict[str, str]:
    """Source of every function and class in ``path``, keyed by name and ``Class.name``."""
    text = path.read_text()
    tree = ast.parse(text)
    lines = text.splitlines()
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.setdefault(
                node.name, "\n".join(lines[node.lineno - 1 : node.end_lineno])
            )
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for item in cls.body:
            if isinstance(item, ast.FunctionDef):
                out[f"{cls.name}.{item.name}"] = "\n".join(
                    lines[item.lineno - 1 : item.end_lineno]
                )
    return out


def patched_surface(path: Path = LOOP_SCHEDULER) -> list[str]:
    """The ``Owner.name`` tokens named by ``PATCHED_SURFACE`` in ``loop_scheduler.py``.

    An entry reads ``[patched: ]Owner.name[ / name2 / name3] -- description``.
    Take the part before `` -- ``, drop a leading ``patched:``, split on ``/`` and
    let ``name2`` inherit the owner of ``Owner.name``.  Entries that are prose
    rather than a name are skipped.
    """
    tree = ast.parse(path.read_text())
    entries: tuple[str, ...] | None = None
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if target == "PATCHED_SURFACE":
            entries = ast.literal_eval(node.value)  # type: ignore[arg-type]
            break
    if entries is None:
        raise SystemExit(f"PATCHED_SURFACE not found in {path}")

    names: list[str] = []
    for entry in entries:
        head = re.sub(r"^patched:\s*", "", str(entry).split(" -- ")[0]).strip()
        owner = None
        for part in (p.strip().strip("`,:;()") for p in head.split("/")):
            part = re.sub(r"\s*\(.*$", "", part).strip()  # "name (2.14 only)"
            if part.startswith("ir."):
                part = part[len("ir.") :]  # ir.py defines Operation directly
            if not part or not re.fullmatch(
                r"[A-Za-z_][A-Za-z_0-9]*(\.[A-Za-z_][A-Za-z_0-9]*)?", part
            ):
                continue
            if "." in part:
                owner = part.rsplit(".", 1)[0]
                full = part
            else:
                full = f"{owner}.{part}" if owner else part
            if full not in names:
                names.append(full)
    return names


def compare(old: Path, new: Path, names: list[str]) -> tuple[int, int, list[str]]:
    """Print the comparison; return (changed, compared, not_found)."""
    changed = compared = 0
    seen: set[str] = set()
    for fname in ("scheduler.py", "ir.py"):
        defs_old, defs_new = _definitions(old / fname), _definitions(new / fname)
        for name in names:
            if name not in defs_old and name not in defs_new:
                continue
            seen.add(name)
            compared += 1
            a = _normalise(defs_old.get(name, ""))
            b = _normalise(defs_new.get(name, ""))
            if a == b:
                print(f"SAME     {fname}:{name}")
                continue
            changed += 1
            side = "" if a and b else "   (missing in one side)"
            print(f"CHANGED  {fname}:{name}{side}")
            for line in difflib.unified_diff(
                a.splitlines(), b.splitlines(), "old", "new", lineterm="", n=1
            ):
                print("    " + line)
    return changed, compared, [n for n in names if n not in seen]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    old, new = Path(argv[1]), Path(argv[2])
    for d in (old, new):
        for fname in ("scheduler.py", "ir.py"):
            if not (d / fname).is_file():
                print(f"error: {d / fname} not found", file=sys.stderr)
                return 2
    names = patched_surface()
    changed, compared, missing = compare(old, new, names)
    for name in missing:
        print(f"NOT FOUND in either tree: {name}")
    print(
        f"\n{changed} changed; {len(names)} names parsed, {compared} compared, "
        f"{len(missing)} not found"
    )
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Fail if this repo's copy of v2_schema.py has drifted from the others in MEANING.

The file is copied, not imported: the baked-image ingest runs under
`uv run --no-project`, so there is no sys.path beyond the script's own directory and no
package to install. This check is what keeps the copies honest.

It compares parsed ASTs, not bytes. The repos pin different ruff line lengths (88 here, 100 in
spyre-inference), so byte equality is unachievable while semantic equality is exactly what
matters -- and a formatting-only difference is precisely what made one earlier fix need two
different patches.

Usage: check_v2_schema_drift.py <other copy> [<other copy> ...]
"""

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "v2_schema.py"


def fingerprint(path: Path) -> str:
    """A signature of the module's MEANING, insensitive to formatting and import order.

    Import statements are sorted before dumping: each repo pins its own ruff, and their
    isort rules order `typing` against `collections.abc` differently, so the raw AST differs
    on files that are otherwise the same code. Comparing raw dumps reported those as drift.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    rest = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    tree.body = sorted(imports, key=ast.dump) + rest
    return ast.dump(tree)


def main(argv):
    if not HERE.exists():
        print(f"ERROR: {HERE} not found", file=sys.stderr)
        return 2
    mine = fingerprint(HERE)
    bad = 0
    for other in argv:
        p = Path(other)
        if not p.exists():
            print(f"SKIP  {p} (not present)")
            continue
        if fingerprint(p) == mine:
            print(f"OK    {p} matches")
        else:
            print(f"DRIFT {p} differs in meaning from {HERE}", file=sys.stderr)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

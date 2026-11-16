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

"""Exercise every code block of docs/dev/*.md.

A fenced block of language yaml, json, python or bash must carry `validated` in its info string, plus one mode:

  python       exec        run the block in the doc's workspace; it must exit 0
  yaml/python  file=P      the block equals examples/P (leading comment lines ignored)
               packbase=N as=R cat=C
                           copy examples/packs/N, write the block at R, validate catalog examples/C with that pack
               in=P        the block occurs verbatim in examples/P
               base=C as=R [central=A,B] [pack=A,B] [expect=error:TEXT]
                           copy examples/C, write the block at R, run `validate`; pass unless expect says otherwise
  json         file=P      the block parses to the same JSON as examples/P
               sidecar     the block is a metrics.json the engine reads without notes
               env-file    the block is a --env-file the engine accepts
               subset-of=G the block is a recursive subset of the JSON file matching glob G in the doc's workspace
  bash         every command is `spyre-bench ...`, run in the doc's workspace (a temp dir that links `examples/`);
               a trailing `# exit=N` states a non-zero expected exit

Other languages (sh, text, make, console) are not run; an `sh` block may not contain a spyre-bench command.

    python -m pytest -q -p no:cacheprovider tests/test_docs_snippets.py
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXT = Path(__file__).resolve().parent.parent
X = EXT / "examples"
DOCS = EXT / "docs"
DEV = DOCS / "dev"
MUST_VALIDATE = {"yaml", "yml", "json", "python", "bash"}
INGEST = EXT.parent / "clickhouse-ingest"
ENV = {"PATH": "/usr/bin:/bin", "PYTHONPATH": os.pathsep.join([str(EXT), *([str(INGEST)] if INGEST.is_dir() else [])])}


class Block:
    def __init__(self, doc, line, lang, attrs, text):
        self.doc, self.line, self.lang, self.attrs, self.text = doc, line, lang, attrs, text

    @property
    def where(self):
        return f"{self.doc.relative_to(DOCS)}:{self.line}"


def parse_info(info):
    toks = shlex.split(info)
    attrs = {}
    for t in toks[1:]:
        k, _, v = t.partition("=")
        attrs[k] = v or True
    return (toks[0] if toks else ""), attrs


def blocks(doc):
    out, lines, i = [], doc.read_text().splitlines(), 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("```"):
            fence = s[: len(s) - len(s.lstrip("`"))]
            lang, attrs = parse_info(s[len(fence) :])
            start, body = i + 1, []
            i += 1
            while not lines[i].strip() == fence:
                body.append(lines[i])
                i += 1
            indent = len(lines[start - 1]) - len(lines[start - 1].lstrip())
            out.append(Block(doc, start, lang, attrs, "\n".join(ln[indent:] for ln in body)))
        i += 1
    return out


def docs():
    return sorted(DEV.glob("*.md"))


def uncomment(text):
    lines = text.strip("\n").splitlines()
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return "\n".join(lines).strip()


def run_verb(args, cwd, env=None):
    p = subprocess.run(
        [sys.executable, "-m", "spyre_bench", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={**ENV, **(env or {})},
    )
    return p.returncode, p.stdout + p.stderr


def logical_commands(text):
    cur, out = "", []
    for ln in text.splitlines():
        if not cur and (not ln.strip() or ln.lstrip().startswith("#")):
            continue
        cur += ln[:-1] + " " if ln.rstrip().endswith("\\") else ln
        if not ln.rstrip().endswith("\\"):
            out.append(cur)
            cur = ""
    return out


def subset(want, got, path="$"):
    """Problems where `want` is not a recursive subset of `got` (lists compare positionally, want may be shorter)."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return [f"{path}: expected an object"]
        return [
            p
            for k, v in want.items()
            for p in (subset(v, got[k], f"{path}.{k}") if k in got else [f"{path}.{k}: missing"])
        ]
    if isinstance(want, list):
        if not isinstance(got, list) or len(want) > len(got):
            return [f"{path}: expected a list of at least {len(want)}"]
        return [p for n, v in enumerate(want) for p in subset(v, got[n], f"{path}[{n}]")]
    return [] if want == got else [f"{path}: {want!r} != {got!r}"]


class Workspace:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "examples").symlink_to(X)

    def close(self):
        self.tmp.cleanup()


class DocsSnippets(unittest.TestCase):
    def test_there_are_guides(self):
        names = {d.name for d in docs()}
        self.assertIn("README.md", names)
        self.assertGreaterEqual(len(names), 12, sorted(names))

    def test_every_code_block_is_validated_or_not_code(self):
        bad = []
        for d in docs():
            for b in blocks(d):
                if not b.lang:
                    bad.append(f"{b.where}: fence without a language")
                elif b.lang in MUST_VALIDATE and "validated" not in b.attrs:
                    bad.append(f"{b.where}: {b.lang} block is not marked validated")
                elif b.lang not in MUST_VALIDATE and any(
                    c.split()[:1] == ["spyre-bench"] for c in logical_commands(b.text)
                ):
                    bad.append(f"{b.where}: a spyre-bench command belongs in a validated bash block")
        self.assertEqual(bad, [])

    def test_relative_links_resolve(self):
        bad = []
        for d in docs():
            fenced = False
            for ln in d.read_text().splitlines():
                if ln.startswith("```"):
                    fenced = not fenced
                if fenced:
                    continue
                rest = "".join(part for n, part in enumerate(ln.split("`")) if n % 2 == 0)
                while "](" in rest:
                    _, _, rest = rest.partition("](")
                    target, _, rest = rest.partition(")")
                    if target.startswith(("http", "#", "mailto:")):
                        continue
                    if not (d.parent / target.split("#")[0]).exists():
                        bad.append(f"{d.name}: {target}")
        self.assertEqual(bad, [])

    def test_validated_blocks(self):
        failures = []
        for d in docs():
            ws = Workspace()
            try:
                for b in blocks(d):
                    if "validated" not in b.attrs:
                        continue
                    why = self.check(b, ws)
                    if why:
                        failures.append(f"{b.where}: {why}")
            finally:
                ws.close()
        self.assertEqual(failures, [], "\n".join(failures))

    def check(self, b, ws):
        a = b.attrs
        if b.lang == "python" and "exec" in a:
            p = subprocess.run([sys.executable, "-c", b.text], capture_output=True, text=True, cwd=ws.dir, env=ENV)
            return None if p.returncode == 0 else f"exited {p.returncode}: {(p.stdout + p.stderr).strip()[:400]}"
        if b.lang in ("yaml", "yml", "python"):
            if "file" in a or "in" in a:
                f = X / (a.get("file") or a["in"])
                if not f.is_file():
                    return f"{f.relative_to(EXT)} does not exist"
                if "file" in a:
                    return None if uncomment(b.text) == uncomment(f.read_text()) else "differs from " + a["file"]
                return None if b.text.strip("\n") in f.read_text() else "not found verbatim in " + a["in"]
            if "base" in a and "as" in a:
                return self.check_new_file(b)
            if "packbase" in a and "as" in a and "cat" in a:
                return self.check_new_file(b)
            return "needs file=, in= or base= and as="
        if b.lang == "json":
            return self.check_json(b, ws)
        if b.lang == "bash":
            return self.check_bash(b, ws)
        return f"unknown language {b.lang}"

    def check_new_file(self, b):
        a = b.attrs
        with tempfile.TemporaryDirectory() as t:
            extra = []
            if "packbase" in a:
                cat = X / a["cat"]
                target = Path(t) / "pack" / a["as"]
                shutil.copytree(X / "packs" / a["packbase"], Path(t) / "pack")
                extra += ["--pack", str(Path(t) / "pack")]
            else:
                cat = Path(t) / a["base"]
                shutil.copytree(X / a["base"], cat)
                target = cat / a["as"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(b.text + "\n")
            for c in str(a.get("central", "")).split(","):
                extra += ["--central", str(X / c)] if c else []
            for p in str(a.get("pack", "")).split(","):
                extra += ["--pack", str(X / p)] if p else []
            rc, out = run_verb(["validate", str(cat), *extra], t)
        want = str(a.get("expect", ""))
        if want.startswith("error:"):
            if rc != 2:
                return f"expected a validation error, got exit {rc}"
            return None if want[6:] in out else f"error {want[6:]!r} not in: {out.strip()[:400]}"
        return None if rc == 0 else f"validate failed: {out.strip()[:400]}"

    def check_json(self, b, ws):
        a = b.attrs
        try:
            data = json.loads(b.text)
        except ValueError as e:
            return f"not JSON: {e}"
        if "file" in a:
            f = X / a["file"]
            return None if f.is_file() and json.loads(f.read_text()) == data else f"differs from {a['file']}"
        if "sidecar" in a:
            from spyre_bench import run

            d = ws.dir / "sidecar"
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir()
            (d / "metrics.json").write_text(b.text)
            samples, _, _, notes = run.read_sidecar(d, 1 << 20)
            return (
                None
                if samples and not notes and set(samples) == set(data["samples"])
                else f"sidecar not accepted: {notes}"
            )
        if "env-file" in a:
            from spyre_bench import run

            f = ws.dir / "env.json"
            f.write_text(b.text)
            try:
                run.load_env_file(str(f))
            except Exception as e:  # the engine raises its own UsageError
                return f"env file refused: {e}"
            return None
        if "subset-of" in a:
            hits = sorted(ws.dir.glob(a["subset-of"]))
            if not hits:
                return f"no file matches {a['subset-of']} (an earlier bash block must create it)"
            problems = subset(data, json.loads(hits[0].read_text()))
            return "; ".join(problems[:5]) or None
        return "json needs file=, sidecar, env-file or subset-of="

    def check_bash(self, b, ws):
        for cmd in logical_commands(b.text):
            code = 0
            if "# exit=" in cmd:
                cmd, _, tail = cmd.partition("# exit=")
                code = int(tail.split()[0])
            toks = shlex.split(cmd)
            env = {}
            while toks and "=" in toks[0] and not toks[0].startswith("-"):
                k, _, v = toks.pop(0).partition("=")
                env[k] = v
            if not toks or toks[0] != "spyre-bench":
                return f"only spyre-bench commands run here: {cmd.strip()[:80]}"
            rc, out = run_verb(toks[1:], ws.dir, env)
            if rc != code:
                return f"`{cmd.strip()[:100]}` exited {rc}, expected {code}: {out.strip()[:400]}"
        return None


if __name__ == "__main__":
    unittest.main()

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

"""`spyre-bench dataset add|list|check` against a filesystem store, plus the https path against a local fake server.
python -m pytest -q -p no:cacheprovider tests/test_dataset_add.py
"""

import contextlib
import hashlib
import http.server
import io
import json
import os
import shutil
import tarfile
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

import yaml

from spyre_bench import dataset_add as da
from spyre_bench import dataset_store as ds
from spyre_bench import engine
from spyre_bench.cli import main as cli_main

EXT = Path(__file__).resolve().parent.parent
HF = EXT / "examples" / "hf-demo"
CENTRAL = EXT / "examples" / "central"
TMP = Path(tempfile.mkdtemp(prefix="spyre-bench-dsadd-"))
SECRET = "Zeta Private Questions"
TOKEN = "tok-SECRET-0123456789"
CASE = "eval_slot-01_demoa_fp16_l50"
_n = [0]


def fresh(name):
    _n[0] += 1
    root = TMP / f"{name}-{_n[0]}"
    root.mkdir(parents=True)
    return root


def make_catalog(root, store_doc):
    """hf-demo with its slot-91 definitions replaced by slot-01 references and a `datasets` Store."""
    cat = root / "cat"
    shutil.copytree(HF, cat)
    docs = [d for d in yaml.safe_load_all((cat / "project.yaml").read_text())]
    drop = {("Asset", "slot-91-files"), ("Dataset", "slot-91"), ("Store", "demo-store"), ("Selection", "lm-eval-pr")}
    keep = [d for d in docs if (d["kind"], d["metadata"]["name"]) not in drop]
    (cat / "project.yaml").write_text(yaml.safe_dump_all(keep, sort_keys=False))
    (cat / "lm-eval" / "eval-slot-91.yaml").unlink()
    if store_doc:
        (cat / "stores.yaml").write_text(store_doc)
    return cat


def install_benchmark(cat, slug):
    """The Benchmark that reads the new dataset, written after the add as a user would."""
    text = (HF / "lm-eval" / "eval-slot-91.yaml").read_text().replace("slot-91", slug)
    (cat / "lm-eval" / f"eval-{slug}.yaml").write_text(text)


def fs_store_doc(path):
    return (
        "apiVersion: bench/v1alpha1\nkind: Store\nmetadata: {name: datasets}\n"
        f"spec: {{type: core/filesystem, path: {path}, visibility: internal}}\n"
    )


def run(*argv, env=None, urlopen=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = da.main(list(argv), env=env if env is not None else {"USER": "tester"}, urlopen=urlopen, out=out)
    return rc, out.getvalue(), err.getvalue()


def write_jsonl(path, lines=3):
    Path(path).write_text("".join(json.dumps({"q": i}) + "\n" for i in range(lines)))
    return Path(path)


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


class FsCase(unittest.TestCase):
    def setUp(self):
        self.root = fresh(self._testMethodName)
        self.store = self.root / "store"
        self.store.mkdir()
        self.cat = make_catalog(self.root, fs_store_doc(self.store))
        self.src = write_jsonl(self.root / "gsm-real-name.jsonl")

    def add(self, *extra, src=None, apply=True, name=SECRET):
        args = [
            "add",
            str(src or self.src),
            "--display-name",
            name,
            "--benchmarks-dir",
            str(self.cat),
            "--central",
            str(CENTRAL),
            "--subset",
            "test-first-50=*.jsonl",
            *extra,
        ]
        return run(*args, *(["--apply"] if apply else []))

    def lst(self, *extra):
        return run("list", "--benchmarks-dir", str(self.cat), "--central", str(CENTRAL), *extra)

    def check(self):
        return run("check", "--benchmarks-dir", str(self.cat), "--central", str(CENTRAL))

    def slots(self):
        return sorted(p.name for p in self.store.iterdir() if not p.name.startswith("."))


class EndToEnd(FsCase):
    def test_add_list_check_validate_explain(self):
        rc, out, err = self.add(apply=False)
        self.assertEqual(rc, 0, err)
        self.assertIn("slug: slot-01 (new)", out)
        self.assertIn("dry run", out)
        self.assertEqual(self.slots(), [])
        self.assertFalse((self.cat / "datasets.yaml").exists())

        rc, out, err = self.add()
        self.assertEqual(rc, 0, err)
        self.assertIn("dataset: slot-01", out)
        self.assertEqual(self.slots(), ["slot-01"])
        self.assertEqual(sorted(p.name for p in (self.store / "slot-01").iterdir()), ["data.jsonl", "dataset.json"])
        meta = json.loads((self.store / "slot-01" / "dataset.json").read_text())
        sha = hashlib.sha256(self.src.read_bytes()).hexdigest()
        self.assertEqual(
            (meta["sha256"], meta["public"], meta["schema_version"], meta["created_by"]), (sha, False, 1, "tester")
        )
        self.assertEqual(meta["display_name"], SECRET)

        install_benchmark(self.cat, "slot-01")
        cat = engine.Catalog([self.cat], [], [CENTRAL])
        self.assertEqual(cat.errors, [])
        self.assertTrue(any(c["name"] == CASE for c in cat.cases))
        spec = cat.by[("Asset", "slot-01-files")].spec
        self.assertEqual(spec["source"], {"store": "datasets", "slug": "slot-01"})
        self.assertEqual(
            spec["integrity"]["files"], [{"path": "data.jsonl", "sha256": sha, "size": self.src.stat().st_size}]
        )

        rc, out, err = self.lst()
        self.assertEqual(rc, 0, err)
        self.assertIn("slot-01", out)
        self.assertIn(sha[:12], out)
        self.assertIn("eval-slot-01", out)
        rc, out, err = self.lst("--remote", "--json")
        row = json.loads(out)[0]
        self.assertEqual((row["slug"], row["public"], row["in_store"]), ("slot-01", False, True))

        rc, out, err = self.check()
        self.assertEqual((rc, err), (0, ""))
        self.assertIn("PASS: 1 dataset(s)", out)

        rc = cli_main(
            ["explain", CASE, "--catalog", str(self.cat), "--central", str(CENTRAL), "--dataset-store", "datasets"]
        )
        self.assertEqual(rc, 0)

    def test_public_shows_name_in_explain(self):
        rc, out, err = self.add("--public", "--license", "MIT", name="Open Questions")
        self.assertEqual(rc, 0, err)
        self.assertIn("Open Questions", out)
        install_benchmark(self.cat, "slot-01")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli_main(
                ["explain", CASE, "--catalog", str(self.cat), "--central", str(CENTRAL), "--dataset-store", "datasets"]
            )
        self.assertIn("Open Questions", buf.getvalue())
        self.assertEqual(self.lst("--remote", "--json")[1].count('"public": true'), 1)

    def test_public_needs_license(self):
        rc, _, err = self.add("--public")
        self.assertEqual(rc, 2)
        self.assertIn("--license", err)

    def test_directory_manifest_and_order_independence(self):
        a, b = fresh("d1"), fresh("d2")
        for d, order in ((a, ("x.jsonl", "sub/y.jsonl")), (b, ("sub/y.jsonl", "x.jsonl"))):
            for rel in order:
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / rel).write_text(rel)
        pa, pb = da.scan(str(a)), da.scan(str(b))
        self.assertEqual(pa.sha256, pb.sha256)
        self.assertEqual([i.rel for i in pa.items], ["sub/y.jsonl", "x.jsonl", "SHA256SUMS"])
        rc, _, err = self.add("--role", "eval", src=a)
        self.assertEqual(rc, 0, err)
        spec = engine.Catalog([self.cat], [], [CENTRAL]).by[("Asset", "slot-01-files")].spec
        self.assertEqual(spec["integrity"]["manifest"], {"path": "SHA256SUMS", "sha256": pa.sha256})
        self.assertEqual(self.check()[0], 0)
        (self.store / "slot-01" / "x.jsonl").write_text("tampered")
        rc, _, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("slot-01:", err)
        self.assertNotIn("x.jsonl", err)
        self.assertNotIn(SECRET, err)


class Idempotency(FsCase):
    def test_second_run_is_a_noop(self):
        self.assertEqual(self.add()[0], 0)
        before = (tree(self.cat), tree(self.store))
        rc, out, err = self.add()
        self.assertEqual(rc, 0, err)
        self.assertIn("already in the store, identical", out)
        self.assertEqual((tree(self.cat), tree(self.store)), before)
        self.assertEqual(self.slots(), ["slot-01"])

    def test_yaml_missing_but_store_present_is_repaired(self):
        self.assertEqual(self.add()[0], 0)
        for f in ("datasets.yaml", "assets-datasets.yaml"):
            (self.cat / f).unlink()
        rc, out, err = self.add("--slug", "slot-01")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.check()[0], 0)
        self.assertEqual(self.slots(), ["slot-01"])

    def test_same_slug_different_data_refused(self):
        self.assertEqual(self.add()[0], 0)
        other = write_jsonl(self.root / "other.jsonl", 5)
        before = (tree(self.cat), tree(self.store))
        rc, out, err = self.add("--slug", "slot-01", src=other)
        self.assertEqual(rc, 2)
        self.assertIn("slot-01", err)
        self.assertEqual((tree(self.cat), tree(self.store)), before)

    def test_same_data_different_metadata_refused(self):
        self.assertEqual(self.add()[0], 0)
        before = (tree(self.cat), tree(self.store))
        rc, _, err = self.add(name="Another Name")
        self.assertEqual(rc, 2)
        self.assertIn("different metadata", err)
        self.assertNotIn("Another Name", err)
        rc, _, err = self.add("--public", "--license", "MIT")
        self.assertEqual(rc, 2)
        self.assertEqual((tree(self.cat), tree(self.store)), before)

    def test_second_dataset_gets_next_slot_and_yaml_appends(self):
        self.assertEqual(self.add()[0], 0)
        other = write_jsonl(self.root / "other.jsonl", 5)
        rc, out, err = self.add("--subset", "x=*.jsonl", src=other)
        self.assertEqual(rc, 0, err)
        self.assertIn("slot-02", out)
        text = (self.cat / "datasets.yaml").read_text()
        self.assertEqual(text.count("kind: Dataset"), 2)
        self.assertTrue(text.index("slot-01") < text.index("slot-02"))


class Allocation(FsCase):
    def seed(self, *slugs):
        for s in slugs:
            (self.store / s).mkdir()
            (self.store / s / "dataset.json").write_text(json.dumps({"display_name": "x", "public": False}))

    def test_gaps_hidden_and_demo_slots(self):
        self.seed("slot-01", "slot-02", "slot-05", "slot-91", "slot-93")
        (self.store / ".stage-abc").mkdir()
        (self.store / "notes").mkdir()
        self.assertEqual(
            ds.DirStore(self.store).slugs(), {"slot-01", "slot-02", "slot-05", "slot-91", "slot-93", "notes"}
        )
        self.assertEqual(ds.next_slug(ds.DirStore(self.store).slugs()), "slot-06")
        _, out, _ = self.add("--subset", "test-first-50=*.jsonl")
        self.assertIn("slug: slot-06 (new)", out)

    def test_empty_and_full(self):
        self.assertEqual(ds.next_slug(set()), "slot-01")
        self.assertEqual(ds.next_slug({"slot-88"}), "slot-89")
        with self.assertRaises(ds.Collision):
            ds.next_slug({"slot-89"})

    def test_explicit_slug_taken_by_foreign_folder(self):
        (self.store / "slot-07").mkdir()
        rc, _, err = self.add("--slug", "slot-07")
        self.assertEqual(rc, 2)
        self.assertEqual(self.slots(), ["slot-07"])

    def test_slug_formats(self):
        for bad in ("slot-1", "slot-90", "slot-00", "slot-001", "Slot-01", "foo", "../x", "slot-01/x", "/abs"):
            rc, _, err = self.add("--slug", bad, apply=False)
            self.assertEqual(rc, 2, bad)
            self.assertIn("--slug", err)

    def test_slug_in_repo_but_unstaged_is_not_reallocated(self):
        (self.cat / "extra.yaml").write_text(
            "apiVersion: bench/v1alpha1\nkind: Asset\nmetadata: {name: slot-03-files}\nspec:\n  type: fileset\n"
            '  source: {store: datasets, slug: slot-03}\n  integrity: {files: [{path: data.jsonl, sha256: "'
            + "1" * 64
            + '"}]}\n'
        )
        rc, out, err = self.add()
        self.assertEqual(rc, 0, err)
        self.assertIn("slot-04", out)

    def test_concurrent_adds_get_distinct_slugs(self):
        cats = []
        for i in range(4):
            root = fresh("conc")
            cats.append((make_catalog(root, fs_store_doc(self.store)), write_jsonl(root / "d.jsonl", 3 + i)))
        results = []

        def go(cat, src):
            out = io.StringIO()
            rc = da.main(
                [
                    "add",
                    str(src),
                    "--display-name",
                    "n",
                    "--benchmarks-dir",
                    str(cat),
                    "--central",
                    str(CENTRAL),
                    "--subset",
                    "test-first-50=*.jsonl",
                    "--apply",
                ],
                env={"USER": "t"},
                out=out,
            )
            results.append((rc, out.getvalue()))

        ts = [threading.Thread(target=go, args=c) for c in cats]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual([r[0] for r in results], [0] * 4, results)
        self.assertEqual(self.slots(), ["slot-01", "slot-02", "slot-03", "slot-04"])


class Atomicity(FsCase):
    def test_failure_before_publish_leaves_nothing_visible(self):
        with mock.patch.object(ds, "write_meta_file", side_effect=OSError("disk full")):
            rc, _, err = self.add()
        self.assertEqual(rc, 5)
        self.assertEqual(self.slots(), [])
        self.assertEqual([p.name for p in self.store.iterdir() if p.name.startswith(".stage-")], [])
        self.assertFalse((self.cat / "datasets.yaml").exists())
        rc, out, err = self.lst("--remote")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.add()[0], 0)
        self.assertEqual(self.check()[0], 0)

    def test_copy_failure_midway_is_cleaned(self):
        real = ds.copy_verified
        calls = []

        def boom(item, dst):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("io error")
            return real(item, dst)

        d = fresh("dir")
        (d / "a.jsonl").write_text("a")
        (d / "b.jsonl").write_text("b")
        with mock.patch.object(ds, "copy_verified", boom):
            rc, _, _ = self.add(src=d)
        self.assertEqual(rc, 5)
        self.assertEqual([p.name for p in self.store.iterdir() if p.name != ".add.lock"], [])

    def test_source_changing_while_copying_is_refused(self):
        plan = da.scan(str(self.src))
        self.src.write_text("changed after hashing\n")
        with self.assertRaises(ds.StoreError):
            ds.copy_verified(plan.items[0], self.root / "out")

    def test_half_state_folder_is_not_a_dataset_and_check_flags_it(self):
        self.assertEqual(self.add()[0], 0)
        (self.store / "slot-01" / "dataset.json").unlink()
        rc, _, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("slot-01", err)

    def test_yaml_write_failure_after_store_publish_is_repaired_by_rerun(self):
        real = da.append_docs

        def only_real(path, texts):
            if Path(path).parent == self.cat:
                raise OSError("ro fs")
            return real(path, texts)

        with mock.patch.object(da, "append_docs", only_real):
            rc, _, _ = self.add()
        self.assertEqual(rc, 5)
        self.assertEqual(self.slots(), ["slot-01"])
        rc, out, err = self.add()
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.slots(), ["slot-01"])
        self.assertEqual(self.check()[0], 0)


class Leaks(FsCase):
    def everything(self):
        text = ""
        for p in list(self.cat.rglob("*")) + list(self.store.rglob("*")):
            text += p.name + "\n"
            if p.is_file() and p.name != "dataset.json":
                text += p.read_text(errors="replace")
        return text

    def test_private_name_never_printed_or_written_to_the_repo(self):
        seen = []
        seen += self.add(apply=False)[1:]
        seen += self.add()[1:]
        seen += self.add()[1:]
        seen += self.add(name=SECRET + " 2")[1:]
        seen += self.add("--slug", "slot-09")[1:]
        seen += self.add("--slug", "bad")[1:]
        seen += self.lst("--remote")[1:]
        seen += self.lst("--remote", "--json")[1:]
        (self.store / "slot-01" / "data.jsonl").write_text("tamper")
        seen += self.check()[1:]
        seen += run("add", str(self.root / "missing"), "--display-name", SECRET)[1:]
        seen += run("add", str(self.src), "--display-name", SECRET, "--benchmarks-dir", str(self.root / "nope"))[1:]
        for s in seen:
            self.assertNotIn(SECRET, s)
            self.assertNotIn("Zeta", s)
        self.assertNotIn("Zeta", self.everything())
        self.assertNotIn("gsm-real-name", self.everything())
        self.assertIn(SECRET, (self.store / "slot-01" / "dataset.json").read_text())

    def test_display_name_rules(self):
        for bad in ("", "   ", "x" * 121, "tab\there", "nl\nx", "\x1b[31m"):
            rc, out, err = self.add(name=bad, apply=False)
            self.assertEqual(rc, 2, repr(bad))
            self.assertNotIn(bad.strip() or "@@", err + out)


class Refusals(FsCase):
    def test_symlink_inside_directory(self):
        d = fresh("sym")
        (d / "a.jsonl").write_text("a")
        os.symlink(self.src, d / "link.jsonl")
        rc, _, err = self.add(src=d)
        self.assertEqual(rc, 2)
        self.assertIn("symlink", err)
        self.assertEqual(self.slots(), [])

    def test_symlinked_directory_entry_and_top_level(self):
        d = fresh("symdir")
        (d / "a.jsonl").write_text("a")
        os.symlink(self.root, d / "sub")
        self.assertEqual(self.add(src=d)[0], 2)
        top = self.root / "toplink.jsonl"
        os.symlink(self.src, top)
        self.assertEqual(self.add(src=top)[0], 2)

    def test_bad_names_and_reserved_files(self):
        for name in ("..hidden", ".hidden", "sp ace.jsonl", "wild*.jsonl", "ünï.jsonl"):
            d = fresh("names")
            try:
                (d / name).write_text("x")
            except OSError:
                continue
            self.assertEqual(self.add(src=d)[0], 2, name)
        for name in ("dataset.json", "SHA256SUMS"):
            d = fresh("reserved")
            (d / name).write_text("x")
            self.assertEqual(self.add(src=d)[0], 2, name)

    def test_open_regular_rejects_symlink_and_dirs(self):
        link = self.root / "l"
        os.symlink(self.src, link)
        with self.assertRaises(OSError):
            ds.open_regular(link)
        with self.assertRaises(ds.StoreError):
            ds.open_regular(self.root)

    def test_zip_slip_archives(self):
        z = self.root / "evil.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("../../escape.txt", "x")
        self.assertEqual(self.add(src=z)[0], 2)
        z2 = self.root / "abs.zip"
        with zipfile.ZipFile(z2, "w") as f:
            f.writestr("/etc/escape", "x")
        self.assertEqual(self.add(src=z2)[0], 2)
        z3 = self.root / "link.zip"
        with zipfile.ZipFile(z3, "w") as f:
            info = zipfile.ZipInfo("l")
            info.external_attr = (0o120777) << 16
            f.writestr(info, "/etc/passwd")
        self.assertEqual(self.add(src=z3)[0], 2)

    def test_tar_slip_and_links(self):
        for kind in ("dotdot", "symlink", "abs"):
            t = self.root / f"{kind}.tar.gz"
            with tarfile.open(t, "w:gz") as f:
                info = tarfile.TarInfo({"dotdot": "../x", "symlink": "ok", "abs": "/etc/x"}[kind])
                if kind == "symlink":
                    info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                f.addfile(info)
            self.assertEqual(self.add(src=t)[0], 2, kind)
        self.assertEqual(self.slots(), [])

    def test_good_archive_is_stored_under_a_neutral_name(self):
        t = self.root / "real-dataset-name.tar.gz"
        with tarfile.open(t, "w:gz") as f:
            f.add(self.src, arcname="inner/data.jsonl")
        rc, out, err = self.add(src=t)
        self.assertEqual(rc, 0, err)
        self.assertTrue((self.store / "slot-01" / "data.tar.gz").is_file())
        self.assertEqual(da.neutral_name("Foo.Bar.JSONL"), "data.jsonl")
        self.assertEqual(da.neutral_name("name.weird"), "data")
        self.assertEqual(da.neutral_name("name"), "data")

    def test_subset_globs(self):
        for bad in ("x=../*", "x=/abs", "nogrob", "BAD=*.jsonl", "x="):
            rc, _, err = run(
                "add",
                str(self.src),
                "--display-name",
                "n",
                "--benchmarks-dir",
                str(self.cat),
                "--central",
                str(CENTRAL),
                "--subset",
                bad,
            )
            self.assertEqual(rc, 2, bad)

    def test_size_and_file_caps(self):
        rc, _, err = self.add("--max-bytes", "5")
        self.assertEqual(rc, 2)
        self.assertIn("size cap", err)
        with mock.patch.object(da, "MAX_FILES", 1):
            d = fresh("many")
            for n in "ab":
                (d / f"{n}.jsonl").write_text(n)
            self.assertEqual(self.add(src=d)[0], 2)

    def test_catalog_must_exist_and_be_valid(self):
        rc, _, err = run("add", str(self.src), "--display-name", "n", "--benchmarks-dir", str(self.root / "none"))
        self.assertEqual(rc, 2)
        (self.cat / "bad.yaml").write_text("apiVersion: nope\n")
        self.assertEqual(self.add()[0], 2)

    def test_conflicting_existing_definition_refused(self):
        (self.cat / "datasets.yaml").write_text(
            "apiVersion: bench/v1alpha1\nkind: Asset\nmetadata: {name: slot-01-files}\nspec:\n  type: fileset\n"
            '  source: {store: datasets, slug: slot-01}\n  integrity: {files: [{path: data.jsonl, sha256: "'
            + "2" * 64
            + '"}]}\n'
        )
        before = tree(self.store)
        rc, _, err = self.add("--slug", "slot-01")
        self.assertEqual(rc, 2)
        self.assertEqual(tree(self.store), before)


class CheckCommand(FsCase):
    def test_detects_each_kind_of_drift_with_slug_only_messages(self):
        self.assertEqual(self.add()[0], 0)
        meta = self.store / "slot-01" / "dataset.json"
        good = meta.read_text()
        cases = {
            "bad json": "{",
            "public not bool": json.dumps({"display_name": "n", "public": "yes"}),
            "digest not declared": json.dumps({"display_name": "n", "public": False, "sha256": "e" * 64}),
            "slug mismatch": json.dumps({"slug": "slot-02", "display_name": "n", "public": False}),
        }
        for label, body in cases.items():
            meta.write_text(body)
            rc, _, err = self.check()
            self.assertEqual(rc, 2, label)
            self.assertTrue(err.startswith("error: slot-01"), (label, err))
        meta.write_text(good)
        (self.store / "slot-01" / "data.jsonl").unlink()
        rc, _, err = self.check()
        self.assertEqual(rc, 2)
        self.assertIn("file #1", err)
        shutil.rmtree(self.store / "slot-01")
        rc, _, err = self.check()
        self.assertEqual(rc, 2)
        self.assertNotIn(SECRET, err)


class ReviewRound1(FsCase):
    def test_slug_only_in_catalog_dry_run_and_apply_agree(self):
        self.assertEqual(self.add()[0], 0)
        shutil.rmtree(self.store / "slot-01")
        dry = self.add("--slug", "slot-01", apply=False)
        real = self.add("--slug", "slot-01")
        self.assertEqual((dry[0], real[0]), (0, 0), (dry, real))
        self.assertEqual(self.check()[0], 0)

    def test_store_name_is_validated(self):
        for bad in ("x, labels: {a: b}", "UP", "a b", "x\ny", ""):
            rc, _, err = self.add("--store", bad, apply=False)
            self.assertEqual(rc, 2, bad)
            self.assertIn("--store", err)

    def test_yaml_keywords_stay_strings(self):
        for lic in ("no", "on", "null", "123"):
            root = fresh("kw")
            cat = make_catalog(root, fs_store_doc(self.store))
            src = write_jsonl(root / "d.jsonl", 2 + len(lic))
            rc, _, err = run(
                "add",
                str(src),
                "--display-name",
                "n",
                "--license",
                lic,
                "--benchmarks-dir",
                str(cat),
                "--central",
                str(CENTRAL),
                "--subset",
                "on=*.jsonl",
                "--subset",
                "123=*.jsonl",
                "--apply",
            )
            self.assertIn(rc, (0, 2), err)
            if rc == 0:
                found = engine.Catalog([cat], [], [CENTRAL])
                self.assertEqual(found.errors, [])
                specs = [d.spec for d in found.of("Asset") if d.name.endswith("-files")]
                self.assertEqual(specs[-1]["license"], lic)
                self.assertEqual(sorted(found.of("Dataset")[-1].spec["subsets"]), ["123", "on"])

    def test_reserved_names_any_case_and_tmp(self):
        for name in ("Dataset.JSON", "sha256sums"):
            d = fresh("res")
            (d / name).write_text("x")
            self.assertEqual(self.add(src=d)[0], 2, name)
        d = fresh("tmp")
        (d / "a.txt").write_text("a")
        (d / "dataset.json.tmp").write_text("keep")
        self.assertEqual(self.add(src=d)[0], 0)
        self.assertEqual((self.store / "slot-01" / "dataset.json.tmp").read_text(), "keep")
        self.assertEqual(self.check()[0], 0)

    def test_check_survives_unreadable_file_without_path_leak(self):
        self.assertEqual(self.add()[0], 0)
        f = self.store / "slot-01" / "data.jsonl"
        f.chmod(0)
        try:
            if os.access(f, os.R_OK):
                self.skipTest("running as a user that ignores modes")
            rc, _, err = self.check()
        finally:
            f.chmod(0o644)
        self.assertEqual(rc, 2)
        self.assertNotIn(str(self.store), err)

    def test_catalog_restored_when_validation_fails_after_write(self):
        self.assertEqual(self.add()[0], 0)
        before = tree(self.cat)
        real = engine.Catalog
        calls = []

        def flaky(*a, **k):
            c = real(*a, **k)
            calls.append(1)
            if len(calls) >= 3:
                c.errors.append("boom")
            return c

        other = write_jsonl(self.root / "o.jsonl", 9)
        with mock.patch.object(engine, "Catalog", flaky):
            rc, _, err = self.add(src=other)
        self.assertEqual(rc, 2)
        self.assertEqual(tree(self.cat), before)

    def test_append_does_not_double_separator(self):
        f = self.root / "x.yaml"
        f.write_text("a: 1\n---\n")
        da.append_docs(f, ["b: 2\n"])
        self.assertEqual(f.read_text(), "a: 1\n---\nb: 2\n")

    def test_unicode_digit_slug_refused(self):
        rc, _, err = self.add("--slug", "slot-\u0660\u0665", apply=False)
        self.assertEqual(rc, 2)
        self.assertIn("--slug", err)


# ---------------------------------------------------------------- https against a local fake server
class Fake(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _record(self):
        self.server.log.append(
            (self.command, self.path, self.headers.get("Authorization"), self.headers.get("X-Checksum-Sha256"))
        )

    def do_PUT(self):
        self._record()
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.server.fail_put_for and self.server.fail_put_for in self.path:
            self.send_response(500)
            self.end_headers()
            return
        self.server.files[self.path] = body
        self.send_response(201)
        self.end_headers()

    def do_DELETE(self):
        self._record()
        self.server.files.pop(self.path, None)
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        self._record()
        files = self.server.files
        prefix = "/artifactory/api/storage/"
        if self.path.startswith(prefix):
            tail = self.path[len(prefix) :].rstrip("/")
            data_path = "/artifactory/" + tail
            if data_path in files:
                body = json.dumps(
                    {
                        "checksums": {"sha256": hashlib.sha256(files[data_path]).hexdigest()},
                        "size": len(files[data_path]),
                    }
                )
            else:
                kids = {}
                for k in files:
                    if k.startswith(data_path + "/"):
                        rest = k[len(data_path) + 1 :]
                        kids[rest.split("/")[0]] = "/" in rest
                if not kids and tail:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps({"children": [{"uri": "/" + n, "folder": f} for n, f in kids.items()]})
            self._send(body.encode())
        elif self.path in files:
            self._send(files[self.path])
        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, body):
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Https(unittest.TestCase):
    def setUp(self):
        self.root = fresh(self._testMethodName)
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        self.srv.files, self.srv.log, self.srv.fail_put_for = {}, [], None
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.port = self.srv.server_address[1]
        self.cat = make_catalog(self.root, None)
        self.src = write_jsonl(self.root / "real-name.jsonl")
        self.env = {
            "USER": "tester",
            "STORE_BASE_URL": "https://store.invalid/artifactory",
            "STORE_DATASETS_PATH": "repo/datasets",
            "STORE_TOKEN": TOKEN,
        }
        self.seen_urls = []

    def urlopen(self, req, timeout=None):
        self.seen_urls.append(req.full_url)
        assert req.full_url.startswith("https://store.invalid/"), req.full_url
        real = urllib.request.Request(
            req.full_url.replace("https://store.invalid", f"http://127.0.0.1:{self.port}"),
            data=req.data,
            method=req.get_method(),
            headers=dict(req.header_items()),
        )
        return urllib.request.urlopen(real, timeout=timeout or 10)

    def add(self, *extra, src=None, env=None):
        return run(
            "add",
            str(src or self.src),
            "--display-name",
            SECRET,
            "--benchmarks-dir",
            str(self.cat),
            "--central",
            str(CENTRAL),
            "--subset",
            "test-first-50=*.jsonl",
            "--apply",
            *extra,
            env=env or self.env,
            urlopen=self.urlopen,
        )

    def test_put_order_store_reference_and_no_token_in_output(self):
        rc, out, err = self.add()
        self.assertEqual(rc, 0, err)
        puts = [p for m, p, a, c in self.srv.log if m == "PUT"]
        base = "/artifactory/repo/datasets/slot-01/"
        self.assertTrue(puts[0].startswith(base + ".claim-"))
        self.assertEqual(puts[-1], base + "dataset.json")
        self.assertIn(base + "data.jsonl", puts)
        self.assertLess(puts.index(base + "data.jsonl"), puts.index(base + "dataset.json"))
        self.assertTrue(all(a == f"Bearer {TOKEN}" for m, p, a, c in self.srv.log))
        sha = hashlib.sha256(self.src.read_bytes()).hexdigest()
        self.assertIn(sha, [c for m, p, a, c in self.srv.log if p.endswith("data.jsonl")])
        self.assertEqual([k for k in self.srv.files if ".claim-" in k], [])
        stores = (self.cat / "stores.yaml").read_text()
        self.assertIn("STORE_DATASETS_PATH", stores)
        self.assertIn("{env: STORE_TOKEN}", stores)
        for text in (out, err, stores, *(p.read_text() for p in self.cat.rglob("*.yaml"))):
            self.assertNotIn(TOKEN, text)
            self.assertNotIn(SECRET, text)
            self.assertNotIn("store.invalid", text)
        self.assertEqual(engine.Catalog([self.cat], [], [CENTRAL]).errors, [])

        rc, out, err = run(
            "check", "--benchmarks-dir", str(self.cat), "--central", str(CENTRAL), env=self.env, urlopen=self.urlopen
        )
        self.assertEqual((rc, err), (0, ""))
        rc, out, err = run(
            "list",
            "--remote",
            "--benchmarks-dir",
            str(self.cat),
            "--central",
            str(CENTRAL),
            env=self.env,
            urlopen=self.urlopen,
        )
        self.assertIn("slot-01", out)
        before = dict(self.srv.files)
        rc, out, err = self.add()
        self.assertEqual(rc, 0, err)
        self.assertIn("identical", out)
        self.assertEqual(self.srv.files, before)

    def test_no_token_refuses_to_write(self):
        env = {k: v for k, v in self.env.items() if k != "STORE_TOKEN"}
        rc, out, err = self.add(env=env)
        self.assertEqual(rc, 5)
        self.assertIn("STORE_TOKEN", err)
        self.assertEqual(self.srv.files, {})

    def test_http_error_and_failed_upload_leave_no_metadata(self):
        self.srv.fail_put_for = "data.jsonl"
        rc, out, err = self.add()
        self.assertEqual(rc, 5)
        self.assertIn("HTTP 500", err)
        self.assertNotIn(TOKEN, err + out)
        self.assertNotIn("store.invalid", err + out)
        self.assertFalse(any(k.endswith("dataset.json") for k in self.srv.files))
        self.assertFalse((self.cat / "datasets.yaml").exists())

    def test_foreign_claim_and_foreign_files_are_collisions(self):
        self.srv.files["/artifactory/repo/datasets/slot-01/.claim-0000"] = b""
        self.srv.files["/artifactory/repo/datasets/slot-01/other.bin"] = b"x"
        rc, out, err = self.add("--slug", "slot-01")
        self.assertEqual(rc, 2)
        self.assertEqual(self.srv.files["/artifactory/repo/datasets/slot-01/other.bin"], b"x")
        self.assertFalse(any(k.endswith("dataset.json") for k in self.srv.files))

    def test_lower_foreign_claim_wins_the_race(self):
        # An empty-but-claimed slug: the other adder holds the lowest marker, so this one stands down.
        self.srv.files["/artifactory/repo/datasets/slot-01/.claim-0000"] = b""
        rc, out, err = self.add("--slug", "slot-01")
        self.assertEqual(rc, 2)
        self.assertEqual([k for k in self.srv.files if k.endswith("data.jsonl")], [])

    def test_plain_http_endpoint_refused(self):
        env = dict(self.env, STORE_BASE_URL="http://store.invalid/artifactory")
        rc, out, err = self.add(env=env)
        self.assertEqual(rc, 5)
        self.assertEqual(self.srv.log, [])

    def test_slug_allocation_from_remote_listing_with_gaps(self):
        for s in ("slot-01", "slot-04", "slot-91"):
            self.srv.files[f"/artifactory/repo/datasets/{s}/dataset.json"] = b"{}"
        rc, out, err = self.add()
        self.assertEqual(rc, 0, err)
        self.assertTrue(any(k.endswith("slot-05/dataset.json") for k in self.srv.files))


if __name__ == "__main__":
    unittest.main()

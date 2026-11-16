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

"""Dataset slugs and `<slug>/dataset.json`: the real name shows only for `public: true`; every failure is explicit.
python -m pytest -q -p no:cacheprovider tests/test_dataset_meta.py
"""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
import urllib.error
from pathlib import Path

from spyre_bench import dataset_meta as dm
from spyre_bench.cli import main as cli_main

EXT = Path(__file__).resolve().parent.parent
HF = EXT / "examples" / "hf-demo"
CENTRAL = EXT / "examples" / "central"
META = EXT / "examples" / "dataset-meta"
TMP = Path(tempfile.mkdtemp(prefix="spyre-bench-slugs-"))
SECRET = "Zeta Private Questions"
DIGEST = "000000000000000000000000000000000000000000000000000000000000000d"


def meta_dir(name, **doc):
    root = TMP / name
    shutil.rmtree(root, ignore_errors=True)
    (root / "slot-91").mkdir(parents=True)
    body = {"slug": "slot-91", "display_name": SECRET, "public": False, "sha256": DIGEST, **doc}
    (root / "slot-91" / "dataset.json").write_text(json.dumps({k: v for k, v in body.items() if v is not None}))
    return root


def explain(*extra):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli_main(
            ["explain", "eval_slot-91_demoa_fp16_l50", "--catalog", str(HF), "--central", str(CENTRAL), *extra]
        )
    return rc, out.getvalue(), err.getvalue()


class Metadata(unittest.TestCase):
    def test_parse_and_label(self):
        pub = dm.parse(json.dumps({"display_name": "Open Set", "public": True, "license": "MIT"}), "slot-90")
        self.assertEqual((pub.label(), pub.license), ("Open Set", "MIT"))
        priv = dm.parse(json.dumps({"display_name": SECRET, "public": False}), "slot-90")
        self.assertEqual(priv.label(), "slot-90")

    def test_malformed_files_are_errors_that_do_not_echo_content(self):
        bad = [
            "not json",
            "[]",
            json.dumps({"display_name": SECRET}),
            json.dumps({"display_name": SECRET, "public": "yes"}),
            json.dumps({"display_name": "", "public": True}),
            json.dumps({"display_name": SECRET, "public": False, "slug": "slot-99"}),
            json.dumps({"display_name": SECRET, "public": False, "sha256": "abc"}),
            json.dumps({"display_name": SECRET, "public": False, "license": 3}),
        ]
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(dm.DatasetMetaError) as cm:
                    dm.parse(text, "slot-91")
                self.assertNotIn(SECRET, str(cm.exception))

    def test_bad_slug_is_refused(self):
        for slug in ("", "../x", "A", "a/b", None):
            with self.subTest(slug=slug), self.assertRaises(dm.DatasetMetaError):
                dm.DirSource(TMP).fetch(slug)

    def test_missing_file_is_an_error_not_a_fallback(self):
        with self.assertRaises(dm.DatasetMetaError) as cm:
            dm.read(dm.DirSource(TMP / "empty"), "slot-91")
        self.assertIn("not found", str(cm.exception))

    def test_digest_must_be_one_the_asset_declares(self):
        meta = dm.parse(json.dumps({"display_name": "n", "public": True, "sha256": DIGEST}), "slot-91")
        ok = {"integrity": {"files": [{"path": "a", "sha256": DIGEST}]}}
        dm.check_assets(meta, [ok])
        dm.check_assets(meta, [{"integrity": {"manifest": {"path": "m", "sha256": DIGEST}}}])
        dm.check_assets(meta, [{"integrity": {"revision_pinned": True}}])
        with self.assertRaises(dm.DatasetMetaError):
            dm.check_assets(meta, [{"integrity": {"files": [{"path": "a", "sha256": "1" * 64}]}}])

    def test_http_source(self):
        seen = []

        class Resp:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return self.body

        def urlopen(req, timeout):
            seen.append((req.full_url, req.get_header("Authorization")))
            if "slot-404" in req.full_url:
                raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
            return Resp(json.dumps({"display_name": "Open Set", "public": True}).encode())

        store = {
            "type": "core/artifactory",
            "endpoint": {"env": "S_URL"},
            "root": {"env": "S_ROOT"},
            "auth": {"mode": "token", "refs": [{"env": "S_TOKEN"}]},
        }
        env = {"S_URL": "https://store.invalid/", "S_ROOT": "/repo/datasets", "S_TOKEN": "tok"}
        labels = dm.Labels(dm.source_for_store(store, env, urlopen))
        self.assertEqual(labels.label("slot-90"), "Open Set")
        self.assertEqual(seen, [("https://store.invalid/repo/datasets/slot-90/dataset.json", "Bearer tok")])
        with self.assertRaises(dm.DatasetMetaError) as cm:
            labels.label("slot-404")
        self.assertIn("not found", str(cm.exception))
        with self.assertRaises(dm.DatasetMetaError):
            dm.source_for_store(store, {}, urlopen)
        with self.assertRaises(dm.DatasetMetaError):
            dm.source_for_store({"type": "core/hf"}, env)


class ToolsRespectPublic(unittest.TestCase):
    def test_private_name_never_reaches_any_output(self):
        root = meta_dir("private")
        for extra in ((), ("--json",), ("--set", "run.timeout=121m")):
            with self.subTest(extra=extra):
                rc, out, err = explain("--dataset-meta-dir", str(root), *extra)
                self.assertEqual(rc, 0, err)
                self.assertNotIn(SECRET, out + err)
                self.assertIn("slot-91", out)

    def test_public_name_is_shown(self):
        rc, out, err = explain("--dataset-meta-dir", str(meta_dir("public", public=True)))
        self.assertEqual(rc, 0, err)
        self.assertIn(SECRET, out)

    def test_without_a_store_only_the_slug_is_shown(self):
        rc, out, _ = explain("--json")
        doc = json.loads(out)[0]
        self.assertEqual((rc, doc["inputs"][0]["dataset_label"]), (0, "slot-91"))

    def test_missing_metadata_fails_explicitly(self):
        rc, out, err = explain("--dataset-meta-dir", str(TMP / "nowhere"))
        self.assertEqual(rc, 2)
        self.assertEqual(out, "")
        self.assertIn("slot-91/dataset.json: not found", err)
        self.assertNotIn("Traceback", err)

    def test_malformed_metadata_fails_without_leaking(self):
        rc, out, err = explain("--dataset-meta-dir", str(meta_dir("badpublic", public="maybe")))
        self.assertEqual((rc, out), (2, ""))
        self.assertNotIn(SECRET, err)

    def test_digest_mismatch_fails(self):
        rc, _, err = explain("--dataset-meta-dir", str(meta_dir("baddigest", sha256="1" * 64)))
        self.assertEqual(rc, 2)
        self.assertIn("sha256", err)

    def test_example_metadata_matches_the_example_catalog(self):
        rc, out, err = explain("--dataset-meta-dir", str(META), "--json")
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out)[0]["inputs"][0]["dataset_label"], "Demo Questions")

    def test_store_flag_needs_a_store_of_the_catalog(self):
        rc, _, err = explain("--dataset-store", "no-such-store")
        self.assertEqual(rc, 2)
        self.assertIn("no Store of that name", err)


class OfflineValidate(unittest.TestCase):
    def test_validate_needs_no_store_and_no_environment(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli_main(["validate", str(HF), "--central", str(CENTRAL)])
        self.assertEqual(rc, 0, err.getvalue())

    def test_path_and_slug_together_are_refused(self):
        d = TMP / "pathslug"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(HF, d)
        p = d / "project.yaml"
        p.write_text(p.read_text().replace("slug: slot-91}", "slug: slot-91, path: x}"))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli_main(["validate", str(d), "--central", str(CENTRAL)])
        self.assertEqual(rc, 2)
        self.assertIn("path or slug", out.getvalue() + err.getvalue())


if __name__ == "__main__":
    unittest.main()

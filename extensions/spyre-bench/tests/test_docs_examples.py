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

"""The catalogs behind docs/dev/: each one validates, plans and (with its stub command) runs end to end.

python -m pytest -q -p no:cacheprovider tests/test_docs_examples.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spyre_bench import engine

EXT = Path(__file__).resolve().parent.parent
X = EXT / "examples"
SUBJECT = "9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2"
PACKS = {"docs-pack": [X / "packs" / "docs-queue"]}
# catalog -> (selection, environment or None, metrics every ok case must carry)
RUNNABLE = {
    "docs-benchmark": ("smoke", None, ["kernel.time.mean"]),
    "docs-environment": ("smoke", "two-cards", ["kernel.time.mean"]),
    "docs-analysis": ("smoke", None, ["kernel.time.mean"]),
    "docs-data": ("smoke", None, ["llm.ttft.p99", "llm.itl.median"]),
    "docs-pack": ("smoke", None, ["mq.throughput", "mq.latency.p99", "mq.depth.max"]),
}
VALIDATE_ONLY = ["docs-ingest-identity"]


def verb(*args, cwd=None):
    p = subprocess.run(
        [sys.executable, "-m", "spyre_bench", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(EXT)},
    )
    return p.returncode, p.stdout, p.stderr


def pack_args(name):
    return [a for p in PACKS.get(name, []) for a in ("--pack", p)]


class DocsCatalogs(unittest.TestCase):
    def test_every_docs_catalog_validates(self):
        for name in [*RUNNABLE, *VALIDATE_ONLY]:
            with self.subTest(name):
                rc, out, err = verb("validate", X / name, *pack_args(name))
                self.assertEqual(rc, 0, out + err)

    def test_docs_catalogs_plan_run_and_recompute(self):
        for name, (sel, env, metrics) in RUNNABLE.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as t:
                plan = Path(t) / "plan.json"
                extra = ["--environment", env] if env else []
                rc, out, err = verb(
                    "plan", "--catalog", X / name, *pack_args(name), "--purpose", "pr", "--select", sel,
                    "--subject-id", SUBJECT, "--arch", "x86_64", "--out", plan, *extra,
                )  # fmt: skip
                self.assertEqual(rc, 0, out + err)
                summary = json.loads(plan.read_text())["summary"]
                self.assertGreaterEqual(summary["run"], 1)
                rc, out, err = verb(
                    "run", "--plan", plan, "--catalog", X / name, *pack_args(name), "--workdir", X / name,
                    "--out", Path(t) / "out",
                )  # fmt: skip
                self.assertEqual(rc, 0, out + err)
                for f in sorted((Path(t) / "out" / "cases").glob("*/result.json")):
                    result = json.loads(f.read_text())
                    self.assertEqual(engine.recompute(result), [])
                    got = result["status"]["aggregate"]["metrics"]
                    for m in metrics:
                        self.assertIn(m, got)

    def test_environments_plan_with_the_expected_requests(self):
        want = {
            "two-cards": None,
            "two-cards-local": "device_local",
            "two-cards-explicit": "explicit",
        }
        for env, policy in want.items():
            with self.subTest(env), tempfile.TemporaryDirectory() as t:
                out = Path(t) / "plan.json"
                rc, so, se = verb(
                    "plan", "--catalog", X / "docs-environment", "--environment", env, "--purpose", "reference",
                    "--select", "smoke", "--subject-id", SUBJECT, "--arch", "x86_64", "--reference-node",
                    "--out", out,
                )  # fmt: skip
                self.assertEqual(rc, 0, so + se)
                case = json.loads(out.read_text())["cases"][0]
                self.assertEqual((case["pinning"] or {}).get("policy"), policy)
                self.assertIs(case["clean"]["requested"], False)

    def test_a_clean_request_without_exclusive_access_is_skipped(self):
        for flags, reason in (([], "reference_node_required"), (["--reference-node"], "capacity")):
            with self.subTest(reason), tempfile.TemporaryDirectory() as t:
                out = Path(t) / "plan.json"
                rc, so, se = verb(
                    "plan", "--catalog", X / "docs-environment", "--environment", "two-cards-clean", "--purpose",
                    "reference", "--select", "smoke", "--subject-id", SUBJECT, "--arch", "x86_64", "--out", out,
                    *flags,
                )  # fmt: skip
                self.assertEqual(rc, 4, so + se)
                case = json.loads(out.read_text())["cases"][0]
                self.assertEqual((case["action"], case["reason"]), ("skip", reason))
                self.assertIs(case["clean"]["requested"], True)

    def test_changing_a_pinned_digest_is_a_new_benchmark(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "docs-data"
            shutil.copytree(X / "docs-data", d)
            ids = []
            for _ in range(2):
                rc, out, err = verb("ids", "--catalog", d)
                self.assertEqual(rc, 0, out + err)
                ids.append(out)
                f = d / "assets.yaml"
                f.write_text(f.read_text().replace('"1' * 1 + "1" * 63 + '"', '"' + "a" * 64 + '"'))
            self.assertNotEqual(ids[0], ids[1])

    def test_analysis_results_are_current(self):
        rc = subprocess.run(
            [sys.executable, str(EXT / "tests" / "refresh_docs_results.py"), "--check"],
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(EXT)},
        ).returncode
        self.assertEqual(rc, 0, "run: python tests/refresh_docs_results.py")

    def test_the_analysis_reads_the_results(self):
        rc, out, err = verb(
            "analyze", "pinning-effect", "--catalog", X / "docs-analysis", "--results",
            X / "results" / "docs-analysis.results.json", "--json",
        )  # fmt: skip
        self.assertEqual(rc, 0, out + err)
        cells = {c["run.exec.cpu_pinning"]: c for c in json.loads(out)["compare"]["cells"]}
        self.assertAlmostEqual(cells["device_local"]["effect_pct"], -10.0, places=6)
        self.assertIs(cells["device_local"]["within_noise"], False)


if __name__ == "__main__":
    unittest.main()

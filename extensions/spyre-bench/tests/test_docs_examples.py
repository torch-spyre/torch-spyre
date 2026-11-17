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
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import regex as re
import yaml

from spyre_bench import engine

EXT = Path(__file__).resolve().parent.parent
X = EXT / "examples"
DEV = EXT / "docs" / "dev"
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
INGEST = [p for p in [EXT.parent / "clickhouse-ingest"] if p.is_dir()]


def verb(*args, cwd=None):
    p = subprocess.run(
        [sys.executable, "-m", "spyre_bench", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": os.pathsep.join([str(EXT), *map(str, INGEST)])},
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


# catalog -> [(selection, purpose, cases)]: the nightly and weekly Selections of docs/dev/12
REAL = {
    "real-vllm-bench": [("smoke", "pr", 1), ("nightly-small", "nightly", 4), ("weekly-full", "weekly", 12)],
    "real-perf-suite": [("smoke", "pr", 1), ("ops-core", "nightly", 19), ("ops-extended", "weekly", 19)],
    "real-perf-suite-models": [("smoke", "pr", 1), ("models-nightly", "nightly", 12), ("models-weekly", "weekly", 36)],
}
# producer checkout of benchmarks/ -> env var that points at it; torch-spyre defaults to this repository
PRODUCERS = {
    "real-vllm-bench": "SPYRE_BENCH_PRODUCER_SPYRE_INFERENCE",
    "real-perf-suite": "SPYRE_BENCH_PRODUCER_TORCH_SPYRE",
    "real-perf-suite-models": "SPYRE_BENCH_PRODUCER_HF_ADAPTERS",
}
# what a public file must not carry: internal tokens, any host but the public ones, addresses, measurements
FORBIDDEN = re.compile(
    "|".join(
        [
            "i" + "bm",
            "spyre" + "[23]",
            r"https?://(?!(?:huggingface\.co|raw\.githubusercontent\.com|github\.com)/)",
            r"\b(?!(?:huggingface|github|githubusercontent)\b)[a-z0-9-]+\.(?:com|net|org|io|cloud)\b",
            r"s390x|ppc64le|\bz1\d\b|\bpower\d?\b",
            r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
            r"\b\d+(?:\.\d+)?\s?(?:ms|us|tok/s|req/s|GiB|GB|MB|%)(?![\w|])",
        ]
    ),
    re.IGNORECASE,
)


def docs_of(path):
    return [d for d in yaml.safe_load_all(Path(path).read_text()) if d]


class RealCatalogs(unittest.TestCase):
    def test_validate_and_plan_the_nightly_and_weekly_selections(self):
        for name, plans in REAL.items():
            rc, out, err = verb("validate", X / name)
            self.assertEqual(rc, 0, name + out + err)
            for sel, purpose, n in plans:
                with self.subTest(name, selection=sel), tempfile.TemporaryDirectory() as t:
                    plan = Path(t) / "plan.json"
                    rc, out, err = verb(
                        "plan", "--catalog", X / name, "--purpose", purpose, "--select", sel,
                        "--subject-id", SUBJECT, "--arch", "x86_64", "--out", plan,
                    )  # fmt: skip
                    self.assertEqual(rc, 0, out + err)
                    doc = json.loads(plan.read_text())
                    self.assertEqual(doc["summary"], {**doc["summary"], "cases": n, "run": n, "skip": 0})
                    self.assertEqual({c["run_props"]["run.purpose"] for c in doc["cases"]}, {purpose})

    def test_the_vllm_identity_blocks_agree_with_the_ingest_name_parser(self):
        try:
            from spyre_clickhouse_ingest import vllm
        except ImportError as e:
            self.skipTest(f"needs the ingest package: {e}")
        cat = engine.Catalog([X / "real-vllm-bench"], [], [])
        self.assertEqual(cat.errors, [])
        for c in cat.cases:
            ing = cat.ingest_inputs(c)
            name, disc = ing["name"], ing["disc"]
            with self.subTest(name):
                self.assertEqual(c["name"], name)
                self.assertEqual(disc["run_mode"], name.split("_")[0])
                parsed = vllm.parse_input_shapes(name)
                self.assertEqual({k: v for k, v in disc.items() if v and k not in ("record_type", "run_mode")}, parsed)
                self.assertEqual(tuple(ing["disc_keys"]), engine.ID_PROFILES[ing["profile"]])
                sweep = set(parsed) - set(vllm.BENCH_ID_KEYS)
                self.assertEqual(ing["profile"], "vllm-sweep" if sweep else "vllm")

    def test_the_perf_kernel_cases_keep_the_ingest_key_list(self):
        cat = engine.Catalog([X / "real-perf-suite"], [], [])
        self.assertEqual(cat.errors, [])
        for c in cat.cases:
            ing = cat.ingest_inputs(c)
            self.assertEqual((ing["profile"], tuple(ing["disc_keys"])), ("perf-kernel", engine.INGEST_DISC_KEYS))
            self.assertEqual(ing["disc"]["input_shapes"], c["workload"]["flags"].get("input_shapes", ""))

    def test_shared_definitions_match_their_producers(self):
        checked = 0
        for name, var in PRODUCERS.items():
            root = os.environ.get(var) or (EXT.parent.parent / "benchmarks" if name == "real-perf-suite" else "")
            if not root or not Path(root).is_dir():
                continue
            checked += 1
            with self.subTest(name):
                getattr(self, "same_" + name.replace("-", "_"))(X / name, Path(root))
        if not checked:
            self.skipTest("no producer checkout: set " + ", ".join(PRODUCERS.values()))

    def same_real_perf_suite(self, ours, theirs):
        for f in (
            "project.yaml",
            "systems.yaml",
            *sorted(str(p.relative_to(ours)) for p in (ours / "ops").glob("*.yaml")),
        ):
            self.assertEqual((ours / f).read_text(), (theirs / f).read_text(), f)

    def same_real_perf_suite_models(self, ours, theirs):
        for f in sorted((ours / "hf-models").glob("*.yaml")):
            mine, orig = docs_of(f)[0], docs_of(theirs / "hf-models" / f.name)[0]
            self.assertEqual(mine["spec"], orig["spec"], f.name)
            self.assertEqual(mine["metadata"]["name"], orig["metadata"]["name"], f.name)
        self.assertEqual(docs_of(ours / "systems.yaml"), docs_of(theirs / "systems.yaml"))

    def same_real_vllm_bench(self, ours, theirs):
        shared = {}
        for f in theirs.rglob("*.yaml"):
            for d in docs_of(f):
                shared[(d["kind"], d["metadata"]["name"])] = d
        for f in ours.rglob("*.yaml"):
            for d in docs_of(f):
                twin = shared.get((d["kind"], d["metadata"]["name"]))
                if d["kind"] in ("Environment", "Model", "Asset") and twin and f.name != "assets-datasets.yaml":
                    self.assertEqual(d["spec"], twin["spec"], d["metadata"]["name"])
        for kind in ("latency", "throughput"):
            mine = docs_of(ours / "decoder" / f"{kind}-qwen05b-tp1-in64-out64.yaml")[0]["spec"]
            orig = docs_of(theirs / "decoder" / f"{kind}-granite8b-tp1-in64-out64.yaml")[0]["spec"]
            for key in ("type", "level", "tier", "env", "run", "est_hours", "workload"):
                self.assertEqual(mine[key], orig[key], (kind, key))
            self.assertEqual(
                [d["key"] for d in mine["identity"]["discriminators"]],
                [d["key"] for d in orig["identity"]["discriminators"]],
            )

    def test_public_files_carry_no_internal_names_or_measurements(self):
        guides = ("12-worked-example-nightly-weekly", "README", "02-define-a-benchmark", "10-contribute")
        files = [DEV / f"{g}.md" for g in guides] + [p for name in REAL for p in (X / name).rglob("*") if p.is_file()]
        bad = [
            f"{p.relative_to(EXT)}:{n}: {line.strip()[:80]}"
            for p in files
            for n, line in enumerate(p.read_text().splitlines(), 1)
            if FORBIDDEN.search(line)
        ]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()

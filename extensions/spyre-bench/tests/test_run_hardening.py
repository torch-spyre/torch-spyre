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

"""Hardening of plan, run and verdict: paths, ingest hand-off, environment, limits, signals, and the store SQL on a
real ClickHouse engine (chdb) when one is importable.
python -m pytest -q -p no:cacheprovider tests/test_run_hardening.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import regex as re

from spyre_bench import cliutil, engine, planfile, store as stores
from spyre_bench import plan as plan_cmd
from spyre_bench import run as run_cmd
from test_run import (
    CENTRAL,
    DEMO,
    EXT,
    NOW,
    OTHER,
    RUN,
    SUBJECT,
    capture,
    clean_env,
    ingest_ctx,
    make_plan,
    manifest,
    plan_doc,
    plan_main,
    run_main,
    sibling_results_parser,
    still_running,
    stored,
    subprocess_popen,
)

HTMP = Path(tempfile.mkdtemp(prefix="spyre-bench-hardening-"))
STORED = "7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"


def tearDownModule():
    shutil.rmtree(HTMP, ignore_errors=True)


def fresh(label):
    d = HTMP / label
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def edited_plan(plan, label, **changes):
    doc = json.loads(Path(plan).read_text())
    doc["cases"][0].update(changes)
    p = HTMP / f"{label}.json"
    p.write_text(json.dumps(doc))
    return p


def one_case_plan(label, **changes):
    plan = make_plan(HTMP / f"{label}-base.json", "--cases", "op_add_fp16")
    return edited_plan(plan, label, **changes)


class RelativePathTests(unittest.TestCase):
    def test_relative_out_with_another_workdir_works_for_commands_and_ingest(self):
        cwd = fresh("relcwd")
        stub = HTMP / "argv_ingest.py"
        stub.write_text("import json, sys\nopen(sys.argv[1], 'a').write(json.dumps(sys.argv[2:]) + '\\n')\n")
        seen = HTMP / "relative-ingest.log"
        seen.unlink(missing_ok=True)
        plan = make_plan(HTMP / "relplan.json")
        old = os.getcwd()
        os.chdir(cwd)
        self.addCleanup(os.chdir, old)
        with clean_env():
            code, _, err = run_main(
                plan, "rel/out", "--ingest", "--ingest-cmd", f"{sys.executable} {stub} {seen} {{raw_dir}} {{out_dir}}"
            )
        self.assertEqual(code, 0, err)
        m = manifest(cwd / "rel" / "out")
        self.assertEqual(m["counts"]["ok"], 2)
        for line in seen.read_text().splitlines():
            raw, out = json.loads(line)
            self.assertTrue(os.path.isabs(raw) and os.path.isabs(out), line)
            self.assertTrue(raw.startswith(str(cwd.resolve())) or raw.startswith(str(cwd)), raw)
            self.assertTrue(Path(raw).is_dir())

    def test_the_default_results_dir_with_another_workdir(self):
        cwd = fresh("defcwd")
        plan = make_plan(HTMP / "defplan.json")
        old = os.getcwd()
        os.chdir(cwd)
        self.addCleanup(os.chdir, old)
        with clean_env():
            code, _, err = capture(
                run_cmd.main,
                ["--plan", str(plan), "--workdir", str(DEMO), "--catalog", str(DEMO), "--central", str(CENTRAL)],
            )
        self.assertEqual(code, 0, err)
        self.assertTrue((cwd / "results" / "bench" / "run.json").is_file())


class IngestHandoffContractTests(unittest.TestCase):
    def report_ingest(self, run_id=STORED):
        stub = HTMP / "report_ingest.py"
        stub.write_text(
            "import json, sys\nprint('some progress')\n"
            f"print(json.dumps({{'run_id': {run_id!r}, 'bundle_sha256': 'abc', 'status': 'ingested'}}))\n"
        )
        return f"{sys.executable} {stub}"

    def test_report_of_reads_the_last_json_object_line(self):
        text = 'noise\n{"run_id": "A", "status": "ingested"}\n{"status": "rejected", "reason": "r"}\nmore noise\n'
        self.assertEqual(run_cmd.report_of(text), {"status": "rejected", "reason": "r"})  # the last one, not the first
        self.assertEqual(
            run_cmd.report_of('{"run_id": "A", "status": "ingested"}\n{broken'), {"run_id": "A", "status": "ingested"}
        )
        self.assertEqual(run_cmd.report_of("no json here\n[1, 2]\n"), {})
        self.assertEqual(run_cmd.report_of(None), {})

    def test_only_a_named_run_and_an_accepting_status_count(self):
        for doc, ok in (
            ({"run_id": "A", "status": "ingested"}, True),
            ({"run_id": "A", "status": "duplicate"}, True),
            ({"run_id": "A", "status": "would-ingest"}, True),
            ({"run_id": "A", "status": "rejected"}, False),
            ({"run_id": "A"}, False),
            ({"run_id": "", "status": "ingested"}, False),
            ({"run_id": 5, "status": "ingested"}, False),
            ({"status": "ingested"}, False),
        ):
            self.assertEqual(run_cmd.accepted_report(doc), ok, doc)

    def test_landed_check_and_manifest_use_the_stored_run_id(self):
        plan = make_plan(HTMP / "stored-plan.json")
        ids = [c["benchmark_id"] for c in json.loads(plan.read_text())["cases"]]
        out = fresh("stored-out")
        st = stores.MemoryStore(landed={STORED: set(ids)})
        code, _, err = run_main(
            plan,
            out,
            "--run-id",
            RUN,
            "--ingest",
            "--ingest-cmd",
            self.report_ingest(),
            "--store",
            "clickhouse",
            "--landed-wait",
            "0",
            st=st,
        )
        self.assertEqual(code, 0, err)
        m = manifest(out)
        self.assertEqual((m["engine_run_id"], m["run_id"], m["stored_run_ids"]), (RUN, RUN, [STORED]))
        self.assertEqual(m["ingest"]["invocations"][0]["stored_run_id"], STORED)
        self.assertEqual(m["ingest"]["invocations"][0]["bundle_sha256"], "abc")
        # rows under the engine's id alone do not satisfy a run whose rows were stored under another
        out = fresh("stored-out2")
        st = stores.MemoryStore(landed={RUN: set(ids)})
        code, _, _ = run_main(
            plan,
            out,
            "--run-id",
            RUN,
            "--ingest",
            "--ingest-cmd",
            self.report_ingest(),
            "--store",
            "clickhouse",
            "--landed-wait",
            "0",
            st=st,
        )
        self.assertEqual(code, 1)
        self.assertEqual(manifest(out)["exit_reason"], "no_rows_landed")

    def test_without_a_report_a_custom_ingest_keeps_the_engine_run_id(self):
        cmd = f"{sys.executable} -c pass"
        out = fresh("noreport-out")
        code, _, _ = run_main(make_plan(HTMP / "noreport.json"), out, "--run-id", RUN, "--ingest", "--ingest-cmd", cmd)
        self.assertEqual(code, 0)
        self.assertEqual(manifest(out)["stored_run_ids"], [RUN])

    def test_a_default_bundle_ingest_must_report_its_run_id(self):
        done = run_cmd.run_ingest(
            [{"argv": [sys.executable, "-c", "pass"], "cases": ["c"], "default": True, "threads": False}],
            {"out": str(fresh("nr-out")), "ingest_timeout": 30, "run_id": RUN},
        )
        self.assertEqual((done[0]["returncode"], done[0]["error"], done[0]["stored_run_id"]), (0, "no_report", None))
        with mock.patch.dict(run_cmd.INGEST_DEFAULTS, {"generic": ["{python}", "-c", "pass"]}):
            code, _, _ = run_main(make_plan(HTMP / "nr-plan.json"), fresh("nr-out2"), "--ingest")
        self.assertEqual(code, 5)

    def test_the_perf_kernel_default_threads_the_engine_run_id(self):
        (cmd,) = run_cmd.ingest_commands(
            [{"name": "c", "status": "ok", "dir": "cases/c"}],
            {"c": {"name": "c", "id_profile": "perf-kernel", "ingest": None}},
            ingest_ctx(),
        )
        self.assertTrue(cmd["threads"] and cmd["default"])
        (cmd,) = run_cmd.ingest_commands(
            [{"name": "c", "status": "ok", "dir": "cases/c"}],
            {"c": {"name": "c", "id_profile": "vllm", "ingest": None}},
            ingest_ctx(),
        )
        self.assertFalse(cmd["threads"])
        (cmd,) = run_cmd.ingest_commands(
            [{"name": "c", "status": "ok", "dir": "cases/c"}],
            {"c": {"name": "c", "id_profile": "vllm", "ingest": ["x"]}},
            ingest_ctx(),
        )
        self.assertTrue(cmd["threads"] and not cmd["default"])

    def test_default_argv_builds_a_valid_bundle_with_a_stable_run_id(self):
        parser, why = sibling_results_parser()
        if parser is None:
            self.skipTest(why)
        sys.path.insert(0, str(EXT.parent / "clickhouse-ingest"))
        try:
            from spyre_clickhouse_ingest import offline
            from spyre_clickhouse_ingest.identity import RunId
        finally:
            sys.path.remove(str(EXT.parent / "clickhouse-ingest"))
        raw = fresh("bundle-raw")
        (raw / "ops.json").write_text(
            json.dumps(
                {"benchmarks": [{"name": "op_add", "backend": "spyre", "measurements": {"kernel_mean_ms": [1.0, 1.1]}}]}
            )
        )
        ids, keys = {}, {}
        for n, run_id in enumerate((RUN, RUN, OTHER)):
            ctx = ingest_ctx(run_id=run_id, out=str(HTMP))
            (cmd,) = run_cmd.ingest_commands(
                [{"name": "c", "status": "ok", "dir": "cases/c"}],
                {"c": {"name": "c", "id_profile": "generic", "ingest": None}},
                ctx,
            )
            argv = [str(raw) if a.endswith("/cases/c/raw") else a for a in cmd["argv"]]
            keys[n] = argv[argv.index("--jenkins-run-key") + 1]
            args = parser.parse_args(argv[5:] + ["--offline", "--out", str(fresh(f"bundle-{n}") / "b")])
            out = offline.write(args)
            ids[n] = offline.report(out)["run_id"]
        self.assertEqual(ids[0], ids[1])  # the same engine run id and case give the same run key and stored run id
        self.assertNotEqual(ids[0], ids[2])
        self.assertEqual(ids[0], RunId.derive("bundle", keys[0], "x86_64", "perf"))
        self.assertRegex(keys[0], rf"^manual:spyre-bench:{RUN}-[0-9a-f]{{16}}$")

    def test_vllm_default_argv_builds_a_valid_bundle(self):
        parser, why = sibling_results_parser()
        fixtures = EXT.parent / "clickhouse-ingest" / "tests" / "data" / "vllm_bundle" / "results"
        if parser is None or not fixtures.is_dir():
            self.skipTest(why or "the ingest's vLLM fixtures are not alongside this package")
        sys.path.insert(0, str(EXT.parent / "clickhouse-ingest"))
        try:
            from spyre_clickhouse_ingest import offline
        finally:
            sys.path.remove(str(EXT.parent / "clickhouse-ingest"))
        for profile in ("vllm", "vllm-sweep"):
            raw = fresh(f"vllm-raw-{profile}")
            for f in fixtures.glob("*.json"):
                shutil.copy(f, raw)
            ctx = ingest_ctx(out=str(HTMP))
            (cmd,) = run_cmd.ingest_commands(
                [{"name": "c", "status": "ok", "dir": "cases/c"}],
                {"c": {"name": "c", "id_profile": profile, "ingest": None}},
                ctx,
            )
            argv = [str(raw) if a.endswith("/cases/c/raw") else a for a in cmd["argv"]]
            args = parser.parse_args(argv[5:] + ["--offline", "--out", str(fresh(f"vllm-b-{profile}") / "b")])
            with self.subTest(profile=profile):
                out = offline.write(args)
                self.assertEqual(offline.report(out)["test_type"], "perf")


class BudgetTests(unittest.TestCase):
    def test_a_reused_case_costs_nothing_against_the_budget(self):
        first = plan_doc("--purpose", "nightly", "--subject-id", SUBJECT)["cases"][0]
        st = stores.MemoryStore(results=[stored(first)])
        argv = [
            "--purpose",
            "nightly",
            "--subject-id",
            SUBJECT,
            "--cases",
            "op_add_fp16",
            "--reuse",
            "if_valid",
            "--env-id",
            "env-1",
        ]
        code, out, err = plan_main("--json", *argv, "--budget", "0.001", st=st)
        self.assertEqual(code, 0, err)  # the only case is reused, so there is nothing to pay for
        doc = json.loads(out)
        self.assertEqual(doc["summary"]["reuse"], 1)

    def test_a_run_case_still_hits_the_budget(self):
        code, _, err = plan_main("--purpose", "nightly", "--budget", "0.001")
        self.assertEqual(code, 2)
        self.assertIn("budget", err)

    def test_non_finite_or_negative_budgets_are_refused(self):
        for bad in ("nan", "inf", "-1", "-inf", "abc"):
            with self.subTest(budget=bad):
                self.assertEqual(plan_main("--purpose", "nightly", f"--budget={bad}")[0], 2)
        f = HTMP / "nanreq.yaml"
        f.write_text("purpose: nightly\nbudget: {device_hours: .nan}\n")
        self.assertEqual(plan_main("--request", str(f))[0], 2)


class EnvTwinTests(unittest.TestCase):
    def base(self, **env):
        return clean_env(BENCH_ARCH="x86_64", BENCH_REFERENCE_NODE="1", BENCH_PURPOSE="nightly", **env)

    def test_every_list_flag_has_a_working_env_twin(self):
        common = ["--catalog", str(DEMO), "--central", str(CENTRAL), "--now", NOW, "--json"]
        with self.base(BENCH_SUITES="nope"):
            code, _, err = capture(plan_cmd.main, common)
        self.assertEqual((code, "empty" in err), (2, True))  # the env twin was read
        with self.base(BENCH_SUITES="nope"):
            code, out, _ = capture(plan_cmd.main, [*common, "--suite", "ops"])
        self.assertEqual(code, 0)  # the flag wins
        first = plan_doc("--purpose", "nightly", "--subject-id", SUBJECT)["cases"][0]
        st = stores.MemoryStore(results=[stored(first)])
        reuse = [*common, "--subject-id", SUBJECT, "--cases", "op_add_fp16", "--reuse", "if_valid", "--env-id", "env-1"]
        with self.base(BENCH_FORCE="op_add_fp16"):
            code, out, _ = capture(plan_cmd.main, reuse, st=st)
        self.assertEqual(json.loads(out)["cases"][0]["reason"], "forced")
        with self.base(BENCH_FORCE="op_add_fp16"):
            code, out, _ = capture(plan_cmd.main, [*reuse, "--force", "other"], st=st)
        self.assertEqual(json.loads(out)["cases"][0]["action"], "reuse")  # the flag wins

    def test_set_merges_per_key_and_the_flag_wins(self):
        common = ["--catalog", str(DEMO), "--central", str(CENTRAL), "--now", NOW, "--json", "--cases", "op_add_fp16"]
        with self.base(BENCH_SET="run.repeats=4 run.cache=cold"):
            code, out, _ = capture(plan_cmd.main, common)
            rec = json.loads(out)["cases"][0]
            self.assertEqual((rec["repeats"], rec["cache"]), (4, "cold"))
            code, out, _ = capture(plan_cmd.main, [*common, "--set", "run.repeats=2"])
            rec = json.loads(out)["cases"][0]
            self.assertEqual((rec["repeats"], rec["cache"]), (2, "cold"))

    def test_central_and_pack_env_twins_and_flag_precedence(self):
        with clean_env(BENCH_CENTRAL=str(CENTRAL), BENCH_PACK=str(HTMP / "nopack")):
            self.assertEqual(cliutil.paths([], "BENCH_PACK"), [str(HTMP / "nopack")])
            self.assertEqual(cliutil.paths(["x"], "BENCH_CENTRAL"), ["x"])
            code, _, err = capture(
                plan_cmd.main, ["--catalog", str(DEMO), "--arch", "x86_64", "--purpose", "nightly", "--reference-node"]
            )
        self.assertEqual(code, 2)
        self.assertIn("no pack.yaml", err)  # BENCH_PACK was honoured

    def test_pick_treats_an_empty_list_as_unset(self):
        with clean_env(BENCH_X="from-env"):
            self.assertEqual(cliutil.pick([], "BENCH_X"), "from-env")
            self.assertEqual(cliutil.pick(["a"], "BENCH_X"), ["a"])
            self.assertEqual(cliutil.pick(None, "BENCH_X"), "from-env")

    def test_request_file_is_below_env_and_flags(self):
        f = HTMP / "suites-req.yaml"
        f.write_text("purpose: nightly\nselection: {suites: [nope]}\n")
        common = ["--catalog", str(DEMO), "--central", str(CENTRAL), "--now", NOW, "--request", str(f)]
        with clean_env(BENCH_ARCH="x86_64", BENCH_REFERENCE_NODE="1"):
            self.assertEqual(capture(plan_cmd.main, common)[0], 2)
        with clean_env(BENCH_ARCH="x86_64", BENCH_REFERENCE_NODE="1", BENCH_SUITES="ops"):
            self.assertEqual(capture(plan_cmd.main, common)[0], 0)
        with clean_env(BENCH_ARCH="x86_64", BENCH_REFERENCE_NODE="1", BENCH_SUITES="nope"):
            self.assertEqual(capture(plan_cmd.main, [*common, "--suite", "ops"])[0], 0)


class EnvScrubTests(unittest.TestCase):
    SECRETS = {
        "HF_TOKEN": "s1",
        "GH_TOKEN": "s2",
        "STORE_TOKEN": "s3",
        "CI_TOKEN": "s4",
        "AWS_SECRET_ACCESS_KEY": "s5",
        "BENCH_CLICKHOUSE_PASS": "s6",
        "CLICKHOUSE_PASS": "s7",
        "CLICKHOUSE_HOST": "h",
        "STORE_USER": "u",
        "MY_CUSTOM_SETTING": "c",
        "BENCH_OTHER": "1",
    }

    def dump_plan(self, label):
        script = HTMP / "dump_env.py"
        script.write_text(
            "import json, os, sys\nopen(sys.argv[1] + '/seen.json', 'w').write(json.dumps(sorted(os.environ)))\n"
        )
        return one_case_plan(label, command=["{python}", str(script), "{repeat_dir}"], repeats=1)

    def seen(self, out):
        return set(json.loads(next(Path(out).glob("cases/*/repeat-001/seen.json")).read_text()))

    def test_secret_looking_and_unknown_variables_do_not_reach_the_command(self):
        out = fresh("scrub-out")
        with clean_env(**self.SECRETS):
            run_main(self.dump_plan("scrub-plan"), out)
        seen = self.seen(out)
        for name in self.SECRETS:
            if name != "BENCH_OTHER":
                self.assertNotIn(name, seen, name)
        self.assertNotIn("BENCH_OTHER", seen)  # inherited BENCH_* is not passed either
        self.assertIn("PATH", seen)
        self.assertTrue({"BENCH_RUN_ID", "BENCH_CASE", "BENCH_REPEAT_DIR", "BENCH_RAW_DIR"} <= seen)

    def test_pass_env_names_a_variable_explicitly_even_if_secret_looking(self):
        out = fresh("pass-out")
        with clean_env(**self.SECRETS):
            run_main(self.dump_plan("pass-plan"), out, "--pass-env", "HF_TOKEN", "--pass-env", "MY_CUSTOM_SETTING")
        seen = self.seen(out)
        self.assertTrue({"HF_TOKEN", "MY_CUSTOM_SETTING"} <= seen)
        self.assertNotIn("GH_TOKEN", seen)
        out = fresh("pass-out2")
        with clean_env(BENCH_PASS_ENV="GH_TOKEN,AWS_SECRET_ACCESS_KEY", **self.SECRETS):
            run_main(self.dump_plan("pass-plan2"), out)
        self.assertTrue({"GH_TOKEN", "AWS_SECRET_ACCESS_KEY"} <= self.seen(out))

    def test_benchmark_env_function(self):
        env = {
            "PATH": "/bin",
            "LC_ALL": "C",
            "PYTHONPATH": "p",
            "SPYRE_DEVICES": "0",
            "HF_TOKEN": "t",
            "FOO": "1",
            "PYTHON_SECRET": "x",
        }
        self.assertEqual(
            planfile.benchmark_env(env), {"PATH": "/bin", "LC_ALL": "C", "PYTHONPATH": "p", "SPYRE_DEVICES": "0"}
        )
        self.assertEqual(planfile.benchmark_env(env, ["FOO", "HF_TOKEN"])["HF_TOKEN"], "t")

    def test_the_ingest_gets_its_credentials_and_nothing_else(self):
        stub = HTMP / "ingest_env.py"
        marker = HTMP / "ingest-env.json"
        marker.unlink(missing_ok=True)
        stub.write_text(f"import json, os\nopen({str(marker)!r}, 'w').write(json.dumps(sorted(os.environ)))\n")
        env = {
            "CLICKHOUSE_HOST": "h",
            "CLICKHOUSE_PASS": "p",
            "HF_TOKEN": "t",
            "GH_TOKEN": "g",
            "MY_CUSTOM_SETTING": "c",
            "EXTRA_FOR_INGEST": "e",
        }
        with clean_env(**env):
            run_main(
                make_plan(HTMP / "ie-plan.json"),
                fresh("ie-out"),
                "--ingest",
                "--ingest-cmd",
                f"{sys.executable} {stub}",
                "--ingest-pass-env",
                "EXTRA_FOR_INGEST",
            )
        seen = set(json.loads(marker.read_text()))
        self.assertTrue({"CLICKHOUSE_HOST", "CLICKHOUSE_PASS", "EXTRA_FOR_INGEST", "PATH", "PYTHONSAFEPATH"} <= seen)
        for leaked in ("HF_TOKEN", "GH_TOKEN", "MY_CUSTOM_SETTING"):
            self.assertNotIn(leaked, seen)

    def test_redact_argv(self):
        got = planfile.redact_argv(
            [
                "tool",
                "--token",
                "abc",
                "--password=xyz",
                "--jenkins-run-key",
                "k1",
                "AWS_SECRET_ACCESS_KEY=zzz",
                "PLAIN=1",
                "--arch",
                "x86_64",
            ]
        )
        self.assertEqual(
            got,
            [
                "tool",
                "--token",
                "<redacted>",
                "--password=<redacted>",
                "--jenkins-run-key",
                "k1",
                "AWS_SECRET_ACCESS_KEY=<redacted>",
                "PLAIN=1",
                "--arch",
                "x86_64",
            ],
        )

    def test_secrets_are_not_written_to_the_results_directory(self):
        out = fresh("redact-out")
        with clean_env(HF_TOKEN="topsecretvalue"):
            run_main(
                make_plan(HTMP / "redact-plan.json"),
                out,
                "--ingest",
                "--ingest-cmd",
                f"{sys.executable} -c pass --token ingestsecret --api-key=keysecret TOKEN=envsecret",
            )
        text = "\n".join(p.read_text(errors="replace") for p in Path(out).rglob("*.json"))
        for secret in ("topsecretvalue", "ingestsecret", "keysecret", "envsecret"):
            self.assertNotIn(secret, text)


class CaptureLimitTests(unittest.TestCase):
    def test_a_noisy_command_is_capped_and_cannot_block(self):
        script = HTMP / "noisy.py"
        script.write_text("import sys\nsys.stdout.write('x' * 3_000_000)\nsys.stderr.write('y' * 200_000)\n")
        out = fresh("noisy-out")
        t0 = time.monotonic()
        code, _, _ = run_main(
            one_case_plan("noisy", command=["{python}", str(script)], repeats=1, allow_empty=True),
            out,
            "--max-log-bytes",
            "1000",
        )
        self.assertLess(time.monotonic() - t0, 30)
        self.assertEqual(code, 0)
        case = manifest(out)["cases"][0]
        rdir = out / case["dir"] / "repeat-001"
        log = (rdir / "stdout.log").read_bytes()
        self.assertTrue(1000 < len(log) < 1200 and b"truncated at 1000 bytes" in log)
        self.assertTrue(b"truncated" in (rdir / "stderr.log").read_bytes())
        notes = json.loads((rdir / "status.json").read_text())["notes"]
        self.assertEqual(
            [n for n in notes if "log truncated" in n],
            ["stdout log truncated at 1000 bytes", "stderr log truncated at 1000 bytes"],
        )

    def test_a_small_log_is_not_marked(self):
        out = fresh("small-out")
        run_main(one_case_plan("small", command=["{python}", "-c", "print('hi')"], repeats=1, allow_empty=True), out)
        case = manifest(out)["cases"][0]
        self.assertEqual((out / case["dir"] / "repeat-001" / "stdout.log").read_text(), "hi\n")

    def test_an_oversized_sidecar_is_ignored_with_a_note(self):
        script = HTMP / "bigmetrics.py"
        script.write_text(
            "import json, sys\njson.dump({'samples': {'m': [1.0] * 5000}}, open(sys.argv[1] + '/metrics.json', 'w'))\n"
        )
        out = fresh("bigm-out")
        run_main(
            one_case_plan("bigm", command=["{python}", str(script), "{repeat_dir}"], repeats=1),
            out,
            "--max-metrics-bytes",
            "1000",
        )
        case = manifest(out)["cases"][0]
        st = json.loads((out / case["dir"] / "repeat-001" / "status.json").read_text())
        self.assertIn("exceeds", st["notes"][0])
        res = json.loads((out / case["result"]).read_text())
        self.assertEqual(res["status"]["repeats"][0]["samples"], {})

    def test_raw_output_over_the_threshold_is_noted(self):
        out = fresh("rawwarn-out")
        run_main(make_plan(HTMP / "rawwarn.json"), out, "--max-raw-bytes", "10")
        self.assertTrue(all("warning threshold" in n[0] for n in manifest(out)["result_notes"].values()))

    def test_limits_are_validated(self):
        plan = make_plan(HTMP / "limits.json")
        for flag in ("--max-log-bytes", "--max-metrics-bytes", "--max-raw-bytes"):
            self.assertEqual(run_main(plan, fresh("limits-out"), flag, "0")[0], 2, flag)


class ValidityTests(unittest.TestCase):
    def test_unknown_validity_words_never_satisfy_a_floor(self):
        self.assertFalse(engine.validity_at_least("bogus", "suspect"))
        self.assertTrue(engine.validity_at_least("valid", "suspect"))
        self.assertEqual(stores.validity_word("bogus"), "invalid")
        self.assertEqual(stores.validity_word("valid"), "valid")
        self.assertEqual(stores.validity_word(None), "invalid")

    def test_a_stored_bogus_validity_is_a_run_not_a_crash(self):
        first = plan_doc("--purpose", "nightly", "--subject-id", SUBJECT)["cases"][0]
        st = stores.MemoryStore(results=[stored(first, validity="bogus")])
        code, out, err = plan_main(
            "--json",
            "--purpose",
            "nightly",
            "--subject-id",
            SUBJECT,
            "--cases",
            "op_add_fp16,op_mul_fp16",
            "--reuse",
            "if_valid",
            "--env-id",
            "env-1",
            st=st,
        )
        self.assertEqual(code, 0, err)
        rec = json.loads(out)["cases"][0]
        self.assertEqual((rec["action"], rec["reason"]), ("run", "validity_below_request"))

    def test_the_clickhouse_store_normalises_it(self):
        ts = datetime(2026, 11, 14, tzinfo=timezone.utc)

        class Client:
            def query(self, sql, parameters=None):
                return mock.Mock(
                    column_names=["run_key", "benchmark_key", "stored_at", "props_all", "n"],
                    result_rows=[(RUN, SUBJECT, ts, [{"run.validity": "bogus", "env.id": "e"}], 3)],
                )

        row = stores.ClickHouseStore(client=Client(), database="d").stored_results([SUBJECT], SUBJECT)[0]
        self.assertEqual(row["validity"], "invalid")


class LowFixTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_a_non_object_env_file_exits_2_before_anything_is_written(self):
        out = HTMP / "envfile-never"
        shutil.rmtree(out, ignore_errors=True)
        for n, body in enumerate(
            ("[]", '"x"', "3", '{"qualified": "false"}', '{"fingerprint": []}', '{"node": 5}', '{"clean": "x"}')
        ):
            f = HTMP / f"env-{n}.json"
            f.write_text(body)
            with self.subTest(body=body):
                code, _, err = run_main(make_plan(HTMP / f"envplan{n}.json"), out, "--env-file", str(f))
                self.assertEqual(code, 2, err)
                self.assertFalse(out.exists())

    def test_qualified_is_strictly_a_boolean(self):
        f = HTMP / "env-q.json"
        f.write_text('{"qualified": true, "node": "node-a"}')
        out = fresh("qual-out")
        run_main(make_plan(HTMP / "qplan.json"), out, "--env-file", str(f))
        res = json.loads((out / manifest(out)["cases"][0]["result"]).read_text())["status"]
        self.assertNotIn("qualification_not_current", res["validity"]["reasons"])

    def test_malformed_run_records_are_refused_before_anything_runs(self):
        plan = make_plan(HTMP / "mal-base.json", "--cases", "op_add_fp16")
        bad = [
            {"timeout_min": float("nan")},
            {"timeout_min": 0},
            {"timeout_min": -3},
            {"timeout_min": "5"},
            {"timeout_min": None},
            {"pinning": "device_local"},
            {"pinning": {"policy": 3}},
            {"ingest": "tool"},
            {"ingest": []},
            {"labels": {"a": 1}},
            {"labels": ["a"]},
            {"allow_empty": "yes"},
        ]
        for n, changes in enumerate(bad):
            out = HTMP / f"mal-out-{n}"
            shutil.rmtree(out, ignore_errors=True)
            with self.subTest(changes=changes):
                code, _, err = run_main(edited_plan(plan, f"mal-{n}", **changes), out)
                self.assertEqual(code, 2, err)
                self.assertFalse(out.exists())

    def test_a_nan_timeout_does_not_disable_the_timeout(self):
        # JSON allows NaN; it must be refused rather than waiting forever
        p = HTMP / "nan-timeout.json"
        p.write_text(
            Path(make_plan(HTMP / "nan-base.json", "--cases", "op_add_fp16"))
            .read_text()
            .replace('"timeout_min": 5.0', '"timeout_min": NaN')
        )
        self.assertIn("NaN", p.read_text())
        self.assertEqual(run_main(p, fresh("nan-out"))[0], 2)

    def test_a_silent_command_is_not_a_run_unless_declared(self):
        silent = ["{python}", "-c", "pass"]
        out = fresh("silent-out")
        code, _, _ = run_main(one_case_plan("silent", command=silent, repeats=1), out)
        self.assertEqual(code, 1)
        case = manifest(out)["cases"][0]
        self.assertEqual((case["status"], case["reason"]), ("failed", "no_output"))
        code, _, _ = run_main(
            one_case_plan("silent2", command=silent, repeats=1, allow_empty=True), fresh("silent2-out")
        )
        self.assertEqual(code, 0)
        sidecar = ["{python}", "-c", "import sys; open(sys.argv[1] + '/metrics.json', 'w').write('{}')", "{repeat_dir}"]
        code, _, _ = run_main(one_case_plan("silent3", command=sidecar, repeats=1), fresh("silent3-out"))
        self.assertEqual(code, 0)

    def test_allow_empty_flows_from_the_benchmark_into_the_plan(self):
        d = fresh("allow-empty-cat")
        shutil.copytree(DEMO, d, dirs_exist_ok=True)
        f = d / "ops" / "op-add.yaml"
        f.write_text(f.read_text().replace("isolation: process", "isolation: process\n    allow_empty: true"))
        code, out, err = capture(
            plan_cmd.main,
            [
                "--catalog",
                str(d),
                "--central",
                str(CENTRAL),
                "--arch",
                "x86_64",
                "--reference-node",
                "--purpose",
                "nightly",
                "--json",
                "--now",
                NOW,
            ],
        )
        self.assertEqual(code, 0, err)
        flags = {c["name"]: c["allow_empty"] for c in json.loads(out)["cases"]}
        self.assertEqual(flags, {"op_add_fp16": True, "op_mul_fp16": False})

    def test_case_must_name_a_run_case(self):
        doc = json.loads(make_plan(HTMP / "cr-base.json").read_text())
        doc["cases"][0].update(action="reuse", reused_from="r")
        doc["cases"][1].update(action="skip", reason="capacity")
        p = HTMP / "cr.json"
        p.write_text(json.dumps(doc))
        for name in ("op_add_fp16", "op_mul_fp16"):
            code, _, err = run_main(p, fresh("cr-out"), "--case", name)
            self.assertEqual(code, 2, err)
            self.assertIn("not run", err)

    def test_repeats_token_is_the_total_number_of_repeats(self):
        script = HTMP / "repeats_tok.py"
        script.write_text(
            "import sys\nopen(sys.argv[1] + '/metrics.json', 'w').write('{}')\nopen(sys.argv[1] + '/n', 'w').write(sys.argv[2])\n"
        )
        out = fresh("tok-out")
        run_main(
            one_case_plan(
                "tok", command=["{python}", str(script), "{repeat_dir}", "{repeats}"], repeats=3, cache="both"
            ),
            out,
        )
        case = manifest(out)["cases"][0]
        self.assertEqual((out / case["dir"] / "repeat-002" / "n").read_text(), "6")

    def test_host_arch_spelling(self):
        for machine, want in (
            ("AMD64", "x86_64"),
            ("x86", "x86_64"),
            ("x86_64", "x86_64"),
            ("arm64", "aarch64"),
            ("armv7l", "armv7l"),
            ("riscv64", "riscv64"),
        ):
            with mock.patch("platform.machine", return_value=machine):
                self.assertEqual(cliutil.host_arch(), want, machine)


class SignalTests(unittest.TestCase):
    def start(self, label):
        pidfile = HTMP / f"{label}.pid"
        pidfile.unlink(missing_ok=True)
        code = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(120)"
        plan = one_case_plan(label, command=["{python}", "-c", code, str(pidfile)], repeats=1, allow_empty=True)
        out = fresh(f"{label}-out")
        env = {**os.environ, "PYTHONPATH": str(EXT)}
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "spyre_bench",
                "run",
                "--plan",
                str(plan),
                "--out",
                str(out),
                "--workdir",
                str(DEMO),
            ],
            env=env,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),  # a background shell may have ignored it
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(200):
            if pidfile.exists() and pidfile.read_text():
                break
            time.sleep(0.1)
        else:
            proc.kill()
            self.fail("the benchmark never started")
        return proc, int(pidfile.read_text()), out

    def alive(self, pid):
        for _ in range(50):
            if not still_running(pid):
                return False
            time.sleep(0.1)
        return True

    def test_sigterm_exits_143_and_kills_the_benchmark(self):
        proc, pid, out = self.start("sigterm")
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 143)
        self.assertNotIn(b"Traceback", err)
        self.assertFalse(self.alive(pid))
        self.assertEqual(json.loads((out / "run.json").read_text())["exit_reason"], "incomplete")

    def test_ctrl_c_exits_130_and_kills_the_benchmark(self):
        proc, pid, _ = self.start("sigint")
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 130)
        self.assertNotIn(b"Traceback", err)
        self.assertFalse(self.alive(pid))

    def test_a_benchmark_that_ignores_sigterm_does_not_survive_its_timeout(self):
        pidfile = HTMP / "trap.pid"
        pidfile.unlink(missing_ok=True)
        code = (
            "import os, signal, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(120)\n"
        )
        d = fresh("trap")
        t0 = time.monotonic()
        rc, reason, _ = run_cmd.execute_command(
            [sys.executable, "-c", code], os.environ.copy(), str(d), d / "o", d / "e", 1.5, None, "", subprocess_popen()
        )
        self.assertLess(time.monotonic() - t0, 20)  # a polite SIGTERM would leave this waiting for the sleep
        self.assertEqual((rc, reason), (None, "timeout"))
        pid = int(pidfile.read_text())
        self.assertFalse(self.alive(pid))

    def test_the_sigterm_handler_is_restored_after_a_run(self):
        before = signal.getsignal(signal.SIGTERM)
        run_main(make_plan(HTMP / "sigrestore.json"), fresh("sigrestore-out"))
        self.assertIs(signal.getsignal(signal.SIGTERM), before)
        with run_cmd.terminate_as_interrupt():
            self.assertIsNot(signal.getsignal(signal.SIGTERM), before)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)


class MutantTests(unittest.TestCase):
    def test_non_finite_samples_are_dropped(self):
        self.assertEqual(run_cmd.finite_floats([1, float("nan"), float("inf"), -float("inf"), True, 2.5]), [1.0, 2.5])
        samples, notes = run_cmd.clean_samples({"m": [1.0, float("nan")], "k": [float("inf")], "ok": [1, 2]})
        self.assertEqual(samples, {"ok": [1.0, 2.0]})
        self.assertEqual(len(notes), 2)

    def test_nan_in_a_sidecar_is_not_a_sample(self):
        script = HTMP / "nan_metrics.py"
        script.write_text(
            "import sys\nopen(sys.argv[1] + '/metrics.json', 'w').write('{\"samples\": {\"m\": [NaN, 1.0]}}')\n"
        )
        out = fresh("nan-sidecar")
        run_main(one_case_plan("nansc", command=["{python}", str(script), "{repeat_dir}"], repeats=1), out)
        res = json.loads((out / manifest(out)["cases"][0]["result"]).read_text())
        self.assertEqual(res["status"]["repeats"][0]["samples"], {})

    def test_the_ingest_timeout_is_enforced(self):
        t0 = time.monotonic()
        done = run_cmd.run_ingest(
            [
                {
                    "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
                    "cases": ["c"],
                    "default": False,
                    "threads": True,
                }
            ],
            {"out": str(fresh("ing-to")), "ingest_timeout": 1, "run_id": RUN},
        )
        self.assertLess(time.monotonic() - t0, 20)
        self.assertIsNone(done[0]["returncode"])
        code, _, _ = run_main(
            make_plan(HTMP / "ito-plan.json"),
            fresh("ito-out"),
            "--ingest",
            "--ingest-cmd",
            f"{sys.executable} -c 'import time; time.sleep(60)'",
            "--ingest-timeout",
            "1",
        )
        self.assertEqual(code, 5)

    def test_max_age_is_enforced(self):
        first = plan_doc("--purpose", "nightly", "--subject-id", SUBJECT)["cases"][0]
        old = {**stored(first), "ts": datetime(2026, 11, 1, tzinfo=timezone.utc)}
        argv = [
            "--json",
            "--purpose",
            "nightly",
            "--subject-id",
            SUBJECT,
            "--cases",
            "op_add_fp16",
            "--reuse",
            "if_valid",
            "--env-id",
            "env-1",
        ]
        code, out, _ = plan_main(*argv, "--max-age", "7d", st=stores.MemoryStore(results=[old]))
        self.assertEqual(json.loads(out)["cases"][0]["reason"], "too_old")
        code, out, _ = plan_main(*argv, "--max-age", "30d", st=stores.MemoryStore(results=[old]))
        self.assertEqual(json.loads(out)["cases"][0]["action"], "reuse")

    def test_max_age_must_be_a_positive_duration(self):
        for bad in ("7 days", "0d", "soon"):
            code, _, err = plan_main("--purpose", "nightly", "--reuse", "if_valid", "--max-age", bad)
            self.assertEqual(code, 2, bad)

    def test_verdict_artifact_without_verdict_rows_says_so(self):
        from spyre_bench import verdict as verdict_cmd

        st = stores.MemoryStore(runs={})
        code, _, err = capture(verdict_cmd.main, ["--artifact", SUBJECT], st=st)
        self.assertEqual(code, 2)
        self.assertIn("verdict rows", err)
        code, out, _ = capture(verdict_cmd.main, ["--run-id", RUN, "--json"], st=stores.MemoryStore())
        self.assertEqual((code, json.loads(out)["summary"]["total"]), (0, 0))  # an unknown run id is just empty


try:
    import chdb.session as chdb_session
except Exception:  # noqa: BLE001 - optional native dependency
    chdb_session = None

SQL_PARAM = re.compile(r"\{(\w+):([^}]+)\}")
A1, A2 = "11111111-1111-5111-8111-111111111111", "22222222-2222-5222-8222-222222222222"
R1, R2, R3 = (f"aaaaaaaa-0000-5000-8000-00000000000{i}" for i in (1, 2, 3))
B1, B2 = "bbbbbbbb-0000-5000-8000-000000000001", "bbbbbbbb-0000-5000-8000-000000000002"


class ChdbClient:
    """A clickhouse_connect-shaped client over a chdb session; typed {name:Type} parameters become literals."""

    def __init__(self, session):
        self.session = session

    @staticmethod
    def literal(kind, value):
        if kind == "UUID":
            return f"toUUID('{value}')"
        if kind == "Array(UUID)":
            return "[" + ",".join(f"toUUID('{v}')" for v in value) + "]"
        return "'" + str(value).replace("'", "''") + "'"

    def query(self, sql, parameters=None):
        sql = SQL_PARAM.sub(lambda m: self.literal(m.group(2), (parameters or {})[m.group(1)]), sql)
        lines = [ln for ln in str(self.session.query(sql, "JSONEachRow")).splitlines() if ln]
        rows = [json.loads(ln) for ln in lines]
        columns = list(rows[0]) if rows else []
        stamp = ("stored_at", "ts")
        return mock.Mock(
            column_names=columns,
            result_rows=[
                tuple(
                    datetime.strptime(r[c], "%Y-%m-%d %H:%M:%S") if c in stamp and isinstance(r[c], str) else r[c]
                    for c in columns
                )
                for r in rows
            ],
        )


@unittest.skipUnless(chdb_session, "chdb is not importable (run this class in the chdb image)")
class RealSqlTests(unittest.TestCase):
    """The store's SQL against the real DDL on a real engine; each assertion kills a plausible SQL mutation."""

    FILES = (
        "10-functional-tests.sql",
        "20-artifacts.sql",
        "30-benchmarks.sql",
        "49-result-uploads.sql",
        "50-artifact-views.sql",
        "60-benchmark-views.sql",
    )
    # Present once the benchmark schema stack is in the ingest; 62 reads a view that stack defines, so it goes last.
    STACK = (
        "81-bench-environments.sql",
        "82-run-placements.sql",
        "83-result-invalidations.sql",
        "84-bench-views.sql",
        "85-benchmark-verdict-views.sql",
    )
    LAST = ("62-benchmark-verdicts.sql",)

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(EXT.parent / "clickhouse-ingest"))
        try:
            from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"the ingest schema applier is not importable: {e}") from None
        finally:
            sys.path.remove(str(EXT.parent / "clickhouse-ingest"))
        cls.session = chdb_session.Session()
        pending = [
            stmt
            for name in cls.FILES + tuple(n for n in cls.STACK if (SCHEMA_DIR / n).exists()) + cls.LAST
            for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text())
        ]
        while (
            pending
        ):  # views of the stack and 62 reference each other across files: retry until a pass makes no progress
            failed = []
            for stmt in pending:
                try:
                    cls.session.query(stmt)
                except Exception:  # noqa: BLE001
                    failed.append(stmt)
            if len(failed) == len(pending):
                cls.session.query(failed[0])
            pending = failed
        cls.db = str(cls.session.query("SELECT currentDatabase()", "TSV")).strip()
        cls.seed()
        cls.store = stores.ClickHouseStore(client=ChdbClient(cls.session), database=cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    @classmethod
    def insert(cls, table, rows):
        body = "\n".join(json.dumps(r) for r in rows)
        cls.session.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")

    @classmethod
    def seed(cls):
        cls.insert(
            "benchmarks",
            [
                {"benchmark_id": B1, "component": "demo", "name": "op_add", "tags": [], "props": {}},
                {"benchmark_id": B2, "component": "demo", "name": "op_mul", "tags": [], "props": {}},
            ],
        )
        runs = [(R1, A1, "2026-11-10 03:00:00"), (R2, A1, "2026-11-11 03:00:00"), (R3, A2, "2026-11-12 03:00:00")]
        cls.insert(
            "artifact_results",
            [
                {
                    "artifact_id": a,
                    "run_id": r,
                    "result_kind": "performance",
                    "test_type": "perf",
                    "state": "passed",
                    "arch": "x86_64",
                    "duration_s": 1,
                }
                for r, a, _ in runs
            ],
        )
        rows = []
        for r, _, stamp in runs:
            for b in (B1, B2) if r == R1 else (B1,):
                rows.append(
                    {
                        "run_id": r,
                        "benchmark_id": b,
                        "component": "demo",
                        "backend": "spyre",
                        "measurements": {"kernel_mean_ms": [1.0, 1.1, 0.9]},
                        "iterations": 3,
                        "props": {
                            "run_utc": stamp,
                            "env.id": "e1",
                            "env.label.device.count": "1",
                            "run.validity": "valid",
                        },
                    }
                )
        cls.insert("benchmark_runs", rows)

    def test_stored_results_filters_by_artifact_and_benchmark_and_reads_props(self):
        rows = self.store.stored_results([B1], A1)
        self.assertEqual(sorted(r["run_id"] for r in rows), [R1, R2])  # not B2's row, not the other artifact's run
        self.assertEqual({r["benchmark_id"] for r in rows}, {B1})
        r = rows[0]
        self.assertEqual(
            (r["env_id"], r["labels"], r["validity"], r["n"]), ("e1", {"env.label.device.count": "1"}, "valid", 3)
        )
        self.assertEqual(r["subject_id"], A1)
        both = self.store.stored_results([B1, B2], A1)
        self.assertEqual(sorted((r["run_id"], r["benchmark_id"]) for r in both), [(R1, B1), (R1, B2), (R2, B1)])
        self.assertEqual(self.store.stored_results([B1], A2)[0]["run_id"], R3)
        self.assertEqual(self.store.stored_results([B2], A2), [])

    def test_ids_beyond_one_batch_are_still_found(self):
        ids = [f"{n:08x}-0000-5000-8000-0000000000ff" for n in range(2500)] + [B1]  # B1 sits in the third batch
        rows = self.store.stored_results(ids, A1)
        self.assertEqual(sorted(r["run_id"] for r in rows), [R1, R2])

    def test_landed_is_exactly_the_runs_benchmarks(self):
        self.assertEqual(self.store.landed(R1), {B1, B2})
        self.assertEqual(self.store.landed(R3), {B1})
        self.assertEqual(self.store.landed("cccccccc-0000-5000-8000-000000000009"), set())

    def test_latest_run_is_the_newest_with_verdict_rows_of_that_artifact(self):
        self.assertEqual(self.store.latest_run(A1), R2)  # newer than R1, and R3 (newest overall) is another artifact's
        self.assertEqual(self.store.latest_run(A2), R3)
        self.assertIsNone(self.store.latest_run("cccccccc-0000-5000-8000-000000000009"))

    def test_verdict_rows_belong_to_the_run(self):
        rows = self.store.verdict_rows(R1)
        self.assertEqual({r["run_key"] for r in rows}, {R1})
        self.assertEqual(sorted(r["benchmark_key"] for r in rows), [B1, B2])
        self.assertEqual({r["metric"] for r in rows}, {"kernel_mean_ms"})
        self.assertEqual(len(self.store.verdict_rows(R3)), 1)
        self.assertEqual(self.store.verdict_rows("cccccccc-0000-5000-8000-000000000009"), [])
        row = rows[0]
        self.assertIn(row["status"], ("no_baseline", "unchanged", "regressed", "suspected", "improved"))
        self.assertEqual(row["tier"], "blocking")


if __name__ == "__main__":
    unittest.main()

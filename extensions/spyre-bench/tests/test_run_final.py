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

"""Redaction forms, environment filtering, verdict run-id precedence, store batching, plan copies and the smaller
contracts of run: links, stale files, pumps, exit mapping.
python -m pytest -q -p no:cacheprovider tests/test_run_final.py
"""

import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from spyre_bench import cliutil, planfile, store as stores
from spyre_bench import run as run_cmd
from spyre_bench import verdict as verdict_cmd
from test_run import (
    DEMO,
    EXT,
    OTHER,
    RUN,
    SUBJECT,
    FakeClient,
    capture,
    clean_env,
    make_plan,
    manifest,
    run_main,
    verdict_row,
)

FTMP = Path(tempfile.mkdtemp(prefix="spyre-bench-final-"))


def tearDownModule():
    shutil.rmtree(FTMP, ignore_errors=True)


def fresh(label):
    d = FTMP / label
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def one_case_plan(label, **changes):
    doc = json.loads(make_plan(FTMP / f"{label}-base.json", "--cases", "op_add_fp16").read_text())
    doc["cases"][0].update(changes)
    p = FTMP / f"{label}.json"
    p.write_text(json.dumps(doc))
    return p


class CaseNameRedactionTests(unittest.TestCase):
    NAMES = [
        "throughput_granite8b_tp1_in64_out64",
        "latency_granite8b_tp1_in64_out64",
        "latency_model8b_tp4_in64_out64",
        "decode_throughput_model70b_tp8_batch32_in2048_out128",
        "0123456789abcdef0123456789abcdef01234567",
        "123e4567-e89b-12d3-a456-426614174000",
        "/data/runs/model8b_tp4_in64_out64/results/run0001.json",
    ]

    def test_names_ids_shas_uuids_and_paths_are_not_tokens(self):
        for w in self.NAMES:
            with self.subTest(w=w):
                self.assertEqual(planfile.redact_word(w), w)
                self.assertEqual(planfile.redact_argv(["bench", "--tests", w]), ["bench", "--tests", w])
                self.assertFalse(planfile.credential_looking(w))

    def test_opaque_tokens_are_still_hidden(self):
        for w in ["sk-" + "a1b2c3d4" * 3, "ghp_" + "Ab1" * 12, "Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0Lq"]:
            with self.subTest(w=w):
                self.assertEqual(planfile.redact_word(w), "<redacted>")


class RedactionFormTests(unittest.TestCase):
    CASES = [
        (["--env=HF_TOKEN=abc"], ["--env=HF_TOKEN=<redacted>"]),
        (["-eHF_TOKEN=abc"], ["-eHF_TOKEN=<redacted>"]),
        (["--set=password=abc"], ["--set=password=<redacted>"]),
        (["-DPASSWORD=abc"], ["-DPASSWORD=<redacted>"]),
        (["-u", "user:pass"], ["-u", "<redacted>"]),
        (["-uuser:pass"], ["-u<redacted>"]),
        (["--user", "u:p"], ["--user", "<redacted>"]),
        (["--user=u:p"], ["--user=<redacted>"]),
        (["--user", "joe", "--arch", "x86_64"], ["--user", "joe", "--arch", "x86_64"]),
        (["redis://:pw@host"], ["redis://<redacted>@host"]),
        (["https://u:p@host/x?y=1"], ["https://<redacted>@host/x?y=1"]),
        (["sk-abcdefghij12345"], ["<redacted>"]),
        (["ghp_abcdefghijklmnop"], ["<redacted>"]),
        (["eyJhbGciOi.eyJzdWIiOi.c2ln"], ["<redacted>"]),
        (["--data", '{"password":"x","name":"n"}'], ["--data", '{"password":"<redacted>","name":"n"}']),
        (["--data", '{"api_key": "x"}'], ["--data", '{"api_key": "<redacted>"}']),
        (["sh", "-c", "export HF_TOKEN=abc; run"], ["sh", "-c", "export HF_TOKEN=<redacted>; run"]),
        (["--connection-string", "Server=h;Password=abc"], ["--connection-string", "Server=h;Password=<redacted>"]),
        (["host=h password=abc"], ["host=h password=<redacted>"]),
        (["-p=secret"], ["-p=<redacted>"]),
        (["-p", "abc"], ["-p", "<redacted>"]),
        (["-pabc"], ["-p<redacted>"]),
        (["--header", "Authorization: Bearer abcdef123"], ["--header", "Authorization: <redacted>"]),
        (["sh", "-c", "export HF_TOKEN='a b'; run"], ["sh", "-c", "export HF_TOKEN=<redacted>; run"]),
        (["TOKEN='x'"], ["TOKEN=<redacted>"]),
        (["password='x'"], ["password=<redacted>"]),
        (["password: x"], ["password: <redacted>"]),
        (["{'password': 'x'}"], ["{'password': '<redacted>'}"]),
        (["-pass", "x"], ["-pass", "<redacted>"]),
        (["-u", "x"], ["-u", "<redacted>"]),
        (["python", "-u", "script.py"], ["python", "-u", "script.py"]),
        (["sh", "-c", "tool --token x"], ["sh", "-c", "tool --token <redacted>"]),
        (["sh", "-c", "curl -u u:x"], ["sh", "-c", "curl -u <redacted>"]),
        (["dXNlcjpwYXNzd29yZGFiY2RlZg=="], ["<redacted>"]),
        (["glpat-abcdefghijklmnop12"], ["<redacted>"]),
        (["npm_abcdefghijklmnop12"], ["<redacted>"]),
        (["AIzaSyAbcdefghijklmnop12"], ["<redacted>"]),
        (["ya29.abcdefghijklmnop12"], ["<redacted>"]),
        (["pypi-AgEIcHlwaS5vcmc"], ["<redacted>"]),
        (["SECRETKEY=abc"], ["SECRETKEY=<redacted>"]),
        (["PRIVKEY=abc"], ["PRIVKEY=<redacted>"]),
        (["ACCESSKEY=abc"], ["ACCESSKEY=<redacted>"]),
        # left alone
        (["--max-tokens", "128"], ["--max-tokens", "128"]),
        (["TOKENS=128"], ["TOKENS=128"]),
        (["-p", "no:cacheprovider"], ["-p", "no:cacheprovider"]),
        (["--auth-mode", "x"], ["--auth-mode", "x"]),
        (["--ssh-key-path", "/a/b"], ["--ssh-key-path", "/a/b"]),
        (["0123456789abcdef0123456789abcdef01234567"], ["0123456789abcdef0123456789abcdef01234567"]),
        (["--passes", "3"], ["--passes", "3"]),
        (["--author", "joe"], ["--author", "joe"]),
        (["MONKEY_PATCH=1"], ["MONKEY_PATCH=1"]),
        (["--jenkins-run-key", "manual:x:y"], ["--jenkins-run-key", "manual:x:y"]),
        (["550e8400-e29b-41d4-a716-446655440000"], ["550e8400-e29b-41d4-a716-446655440000"]),
        (
            ["a-long-dashed-host-name-for-testing-purposes.example.com"],
            ["a-long-dashed-host-name-for-testing-purposes.example.com"],
        ),
        (["--arch=x86_64", "--shape", "1,512"], ["--arch=x86_64", "--shape", "1,512"]),
    ]

    def test_forms(self):
        for argv, want in self.CASES:
            with self.subTest(argv=argv):
                self.assertEqual(planfile.redact_argv(argv), want)

    def test_dry_run_prints_redacted_argv_in_text_and_json(self):
        plan = one_case_plan(
            "dry-redact", command=["{python}", "-c", "pass", "--token", "hunter2", "https://u:pw0rd@h/x"], repeats=1
        )
        for extra in ((), ("--json",)):
            code, out, _ = run_main(plan, fresh("dry-redact-out"), "--dry-run", *extra)
            self.assertEqual(code, 0)
            for secret in ("hunter2", "pw0rd"):
                self.assertNotIn(secret, out)
            self.assertIn("redacted", out)

    def test_the_results_directory_names_its_plan_copy_as_redacted(self):
        secret = ["{python}", "-c", "pass", "--token", "hunter2"]
        plan = one_case_plan("plan-copy", command=secret, repeats=1, allow_empty=True)
        out = fresh("plan-copy-out")
        run_main(plan, out)
        self.assertFalse((out / "plan.json").exists())
        copy = out / "plan.redacted.json"
        self.assertNotIn("hunter2", copy.read_text())
        m = manifest(out)["plan"]
        self.assertEqual(m["path"], "plan.redacted.json")
        self.assertEqual(m["sha256"], hashlib.sha256(plan.read_bytes()).hexdigest())  # the plan that actually ran
        self.assertEqual(m["redacted_sha256"], hashlib.sha256(copy.read_bytes()).hexdigest())
        self.assertNotEqual(m["sha256"], m["redacted_sha256"])

    def test_a_redacted_copy_is_not_a_runnable_plan(self):
        plan = one_case_plan(
            "plan-copy2", command=["{python}", "-c", "pass", "--token", "hunter2"], repeats=1, allow_empty=True
        )
        out = fresh("plan-copy2-out")
        run_main(plan, out)
        out2 = fresh("plan-copy2-again")
        code, _, err = run_main(out / "plan.redacted.json", out2)
        self.assertEqual(code, 2)
        self.assertIn("redacted copy", err)
        self.assertFalse((out2 / "cases").exists())

    def test_duplicate_case_names_are_refused(self):
        doc = json.loads(make_plan(FTMP / "dup-base.json").read_text())
        doc["cases"][1]["name"] = doc["cases"][0]["name"]
        p = FTMP / "dup.json"
        p.write_text(json.dumps(doc))
        code, _, err = run_main(p, fresh("dup-out"))
        self.assertEqual(code, 2)
        self.assertIn("appears twice", err)


class EnvSegmentTests(unittest.TestCase):
    def test_secret_names_are_whole_segments(self):
        for name in (
            "SPYRE_BYPASS",
            "SPYRE_KEYBOARD",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "XDG_CACHE_HOME",
            "SPYRE_COMPASS",
        ):
            self.assertFalse(planfile.secret_env_name(name), name)
        for name in (
            "SPYRE_KEY",
            "spyre_api_key",
            "XDG_SESSION_ID",
            "SPYRE_AUTH",
            "VLLM_PRIVATE",
            "OMP_PAT",
            "SPYRE_DSN",
            "X_PW",
            "HF_TOKEN",
            "SPYRE_COOKIE",
            "SPYRE_BEARER",
            "SPYRE_CERTKEY",
        ):
            self.assertTrue(planfile.secret_env_name(name), name)

    def test_token_shapes(self):
        for value in (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.c2ln",
            "sk-abcdefghij1234",
            "ghp_" + "a" * 30,
            "xoxb-1234567890-abc",
            "AKIA" + "A" * 16,
            "q1" * 16,
            "Bearer abcdef123456",
            "dXNlcjpwYXNzd29yZGFiY2RlZg==",
            "glpat-abcdefghijklmnop12",
            "npm_abcdefghijklmnop12",
            "AIzaSyAbcdefghijklmnop12",
            "ya29.abcdefghijklmnop12",
            "pypi-AgEIcHlwaS5vcmc",
        ):
            self.assertTrue(planfile.credential_looking(value), value)
        for value in (
            "550e8400-e29b-41d4-a716-446655440000",
            "a-long-dashed-host-name-for-testing.example.com",
            "/usr/bin:/usr/local/bin",
            "x86_64",
            "a" * 40,
            "1" * 40,
            "0123456789abcdef0123456789abcdef01234567",  # a commit sha
        ):
            self.assertFalse(planfile.credential_looking(value), value)

    def test_every_withheld_name_is_reported_with_a_reason_and_never_a_value(self):
        env, dropped = planfile.env_filter(
            {"SPYRE_KEY": "plain", "HTTPS_PROXY": "http://u:pw0rd@h:1", "SPYRE_OK": "1", "UNLISTED_TOKEN": "x"}
        )
        self.assertEqual(env, {"SPYRE_OK": "1"})
        self.assertEqual(dropped, {"HTTPS_PROXY": "credential-looking value", "SPYRE_KEY": "secret-looking name"})

    def test_prefixes_compare_case_insensitively_but_secrets_stay_out(self):
        env, dropped = planfile.env_filter(
            {"acme_devices": "0,1", "ACME_SLOTS": "2", "acme_api_key": "k", "other_x": "1"}, (), ("acme_",)
        )
        self.assertEqual(env, {"acme_devices": "0,1", "ACME_SLOTS": "2"})
        self.assertEqual(dropped, {"acme_api_key": "secret-looking name"})
        env, _ = planfile.env_filter({"acme_devices": "0"}, (), ("ACME_",))
        self.assertEqual(env, {"acme_devices": "0"})

    def test_new_default_families(self):
        names = [
            "CUDA_VISIBLE_DEVICES", "TORCH_HOME", "VLLM_LOGGING_LEVEL", "KMP_AFFINITY", "GOMP_CPU_AFFINITY", "HF_HUB_CACHE",
            "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "CPPFLAGS", "LIBRARY_PATH", "CPATH", "CMAKE_BUILD_TYPE", "CCACHE_DIR",
        ]  # fmt: skip
        env, dropped = planfile.env_filter({n: "v" for n in names})
        self.assertEqual((sorted(env), dropped), (sorted(names), {}))

    def test_a_case_inherits_the_whole_spyre_device_environment(self):
        """Every name `/etc/profile.d/ibm-aiu-setup.sh` exports on a one-card host: drop one and the runtime
        refuses with CFGERR, which reaches the leg as `repeats_failed` and names no cause."""
        names = [
            "AIU_WORLD_SIZE", "AIU_WORLD_RANK_0", "AIU_DEVICE_0", "AIU_ENV_FILE", "AIU_TOPO_FILE",
            "AIU_AUTOGEN_SENLIB_CONFIG_FILE", "AIU_TIER_0_NUM_SETS", "AIU_TIER_0_SET_0_SIZE", "AIU_TIER_0_SET_0_RANK_0",
            "AIUCARDMGMT_INSTALL_DIR", "FLEX_DEVICE", "FLEX_COMPUTE", "SENCORES", "SENDNN_DIR", "SENDNN_INSTALL_DIR",
            "SENLIB_INSTALL_DIR", "SENTIENT_BASE_INSTALL_DIR", "SENTINYEXEC_INSTALL_DIR", "SEN_PROJECT_PACKAGE",
            "SEN_COMMON_HEADERS", "DT_DEEPRT_VERBOSE", "DEEPTOOLS_INSTALL_DIR", "DEEPTOOLS_PATH",
            "RUNTIME_INSTALL_DIR", "RUNTIME_FULL_INSTALL_DIR", "TVM_INSTALL_DIR", "MLSERVER_TVM_INSTALL_DIR",
            "_IBM_AIU_SETUP", "pybind11_DIR",
        ]  # fmt: skip
        env, dropped = planfile.env_filter({n: "v" for n in names})
        self.assertEqual((sorted(env), dropped), (sorted(names), {}))

    def test_a_secret_in_a_device_family_is_still_withheld(self):
        """The new families widen what a case inherits, not what a secret-looking name may carry."""
        env, dropped = planfile.env_filter({"AIU_PROFILING_CREDS": "v", "SENLIB_API_KEY": "v", "FLEX_DEVICE": "PF"})
        self.assertEqual(env, {"FLEX_DEVICE": "PF"})
        self.assertEqual(sorted(dropped), ["AIU_PROFILING_CREDS", "SENLIB_API_KEY"])

    def test_a_proxy_with_credentials_reaches_the_ingest_only_when_named(self):
        env = {"HTTPS_PROXY": "http://user:pw0rd@proxy:3128", "PATH": "/bin"}
        self.assertNotIn("HTTPS_PROXY", planfile.ingest_env(env, (), "/h"))
        self.assertEqual(planfile.ingest_env(env, ("HTTPS_PROXY",), "/h")["HTTPS_PROXY"], env["HTTPS_PROXY"])

    def test_the_ingest_environment_has_no_python_variables_and_a_fresh_home(self):
        env = {
            "HOME": "/real",
            "PYTHONPATH": "/x",
            "PYTHONHOME": "/y",
            "PYTHONUSERBASE": "/z",
            "PATH": "/bin",
            "CLICKHOUSE_HOST": "h",
        }
        got = planfile.ingest_env(env, (), "/fresh")
        self.assertEqual(got["HOME"], "/fresh")
        self.assertEqual({k for k in got if k.startswith("PYTHON")}, {"PYTHONSAFEPATH"})
        self.assertEqual(got["CLICKHOUSE_HOST"], "h")
        self.assertEqual(planfile.ingest_env(env, ("HOME", "PYTHONPATH"), "/fresh")["HOME"], "/real")

    def test_the_withheld_names_are_noted_in_status_json(self):
        script = FTMP / "noop.py"
        script.write_text("import sys\nopen(sys.argv[1] + '/metrics.json', 'w').write('{}')\n")
        plan = one_case_plan("notes", command=["{python}", str(script), "{repeat_dir}"], repeats=1)
        out = fresh("notes-out")
        with clean_env(SPYRE_KEY="plain", HTTPS_PROXY="http://u:pw0rd@h:1"):
            run_main(plan, out)
        case = manifest(out)["cases"][0]
        text = (out / case["dir"] / "repeat-001" / "status.json").read_text()
        self.assertIn("SPYRE_KEY withheld", text)
        self.assertIn("HTTPS_PROXY withheld", text)
        self.assertNotIn("pw0rd", text)


class VerdictPrecedenceTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.s1, self.s2 = "7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f", "8e1f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"
        self.dir = fresh("vp")
        (self.dir / "run.json").write_text(json.dumps({"engine_run_id": RUN, "stored_run_ids": [self.s1, self.s2]}))
        self.store = stores.MemoryStore(
            verdicts={self.s1: [verdict_row(run_id=self.s1)], self.s2: [], OTHER: [verdict_row(run_id=OTHER)]}
        )

    def call(self, *argv):
        return capture(verdict_cmd.main, list(argv), st=self.store)

    def test_an_explicit_run_dir_beats_either_ambient_run_id(self):
        for var in ("RUN_ID", "BENCH_RUN_ID"):
            with self.subTest(var=var), mock.patch.dict(os.environ, {var: OTHER}):
                code, out, err = self.call("--run-dir", str(self.dir), "--json")
                self.assertEqual(json.loads(out)["run_ids"], [self.s1, self.s2], err)

    def test_an_explicit_run_id_beats_an_ambient_run_dir(self):
        with mock.patch.dict(os.environ, {"BENCH_RUN_DIR": str(self.dir)}):
            self.assertEqual(json.loads(self.call("--run-id", OTHER, "--json")[1])["run_ids"], [OTHER])

    def test_every_empty_id_is_named_and_require_rows_fails_if_any_is_empty(self):
        code, out, err = self.call("--run-dir", str(self.dir), "--json")
        doc = json.loads(out)
        self.assertEqual((code, doc["summary"]["total"], doc["empty_run_ids"]), (0, 1, [self.s2]))
        self.assertIn(f"no verdict rows for {self.s2}", err)
        self.assertNotIn(f"no verdict rows for {self.s1}", err)
        self.assertEqual(self.call("--run-dir", str(self.dir), "--require-rows")[0], 1)
        self.assertEqual(self.call("--run-id", OTHER, "--require-rows")[0], 0)
        self.assertIn(self.s2, self.call("--run-dir", str(self.dir))[1])


class StoreBatchTests(unittest.TestCase):
    def test_thousands_of_ids_are_queried_in_chunks_of_a_thousand(self):
        ts = datetime(2026, 11, 14, tzinfo=timezone.utc)
        ids = [f"{n:08x}-0000-5000-8000-000000000000" for n in range(5000)]
        client = FakeClient(
            rows=[(RUN, ids[0], ts, [{"run.validity": "valid"}], 3)],
            columns=["run_key", "benchmark_key", "stored_at", "props_all", "n"],
        )
        rows = stores.ClickHouseStore(client=client, database="d").stored_results(ids, SUBJECT)
        sizes = [len(params["ids"]) for _, params in client.queries]
        self.assertEqual((sizes, len(rows)), ([1000] * 5, 5))  # the same fake row came back for each chunk
        self.assertTrue(all(len(p["ids"]) <= stores.ID_BATCH for _, p in client.queries))

    def test_one_weak_row_weakens_the_run_and_benchmark(self):
        ts = datetime(2026, 11, 14, tzinfo=timezone.utc)
        props = [
            {"run.validity": "valid", "env.id": "e1"},
            {"run.validity": "suspect", "env.id": "e1"},
            {"run.validity": "valid", "env.id": "e1"},
        ]
        client = FakeClient(
            rows=[(RUN, SUBJECT, ts, props, 3)], columns=["run_key", "benchmark_key", "stored_at", "props_all", "n"]
        )
        (row,) = stores.ClickHouseStore(client=client, database="d").stored_results([SUBJECT], SUBJECT)
        self.assertEqual((row["validity"], row["env_id"]), ("suspect", "e1"))
        props[1]["env.id"] = "e2"  # rows disagreeing about the environment: unknown
        (row,) = stores.ClickHouseStore(client=client, database="d").stored_results([SUBJECT], SUBJECT)
        self.assertIsNone(row["env_id"])

    def test_the_clickhouse_store_checks_its_settings_without_connecting(self):
        with clean_env():
            with self.assertRaises(stores.StoreError):
                stores.ClickHouseStore().check()
        with clean_env(CLICKHOUSE_HOST="h", CLICKHOUSE_PASS="p", BENCH_CLICKHOUSE_DB="d"):
            stores.ClickHouseStore().check()


class RunContractTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_a_missing_store_configuration_fails_before_anything_runs(self):
        out = fresh("preflight-out")
        code, _, err = capture(
            run_cmd.main,
            [
                "--plan",
                str(make_plan(FTMP / "pre.json")),
                "--out",
                str(out),
                "--workdir",
                str(DEMO),
                "--store",
                "clickhouse",
                "--ingest",
                "--ingest-cmd",
                f"{sys.executable} -c pass",
            ],
        )
        self.assertEqual(code, 2)
        self.assertIn("CLICKHOUSE_HOST", err)
        self.assertFalse((out / "cases").exists())

    def test_planted_links_in_the_case_directory_are_not_followed(self):
        plan = make_plan(FTMP / "links.json", "--cases", "op_add_fp16")
        out = fresh("links-out")
        run_main(plan, out)
        case = out / manifest(out)["cases"][0]["dir"]
        victim = fresh("links-victim")
        (victim / "keep.txt").write_text("keep")
        (victim / "sub").mkdir()
        shutil.rmtree(case / "raw")
        (case / "raw").symlink_to(victim)
        shutil.rmtree(case / "repeat-001")
        (case / "repeat-001").symlink_to(victim / "sub")
        (case / "result.json").unlink()
        (case / "result.json").symlink_to(victim / "keep.txt")
        code, _, err = run_main(plan, out)
        self.assertEqual(code, 0, err)
        self.assertNotIn("Traceback", err)
        self.assertEqual((victim / "keep.txt").read_text(), "keep")
        self.assertEqual(sorted(p.name for p in victim.iterdir()), ["keep.txt", "sub"])
        self.assertEqual(list((victim / "sub").iterdir()), [])
        self.assertFalse((case / "raw").is_symlink() or (case / "repeat-001").is_symlink())

    def test_a_planted_link_in_place_of_the_case_directory_is_replaced(self):
        plan = make_plan(FTMP / "links2.json", "--cases", "op_add_fp16")
        out = fresh("links2-out")
        run_main(plan, out)
        case = out / manifest(out)["cases"][0]["dir"]
        shutil.rmtree(case)
        victim = fresh("links2-victim")
        case.symlink_to(victim)
        self.assertEqual(run_main(plan, out)[0], 0)
        self.assertEqual(list(victim.iterdir()), [])
        self.assertTrue(case.is_dir() and not case.is_symlink())

    def test_a_stale_result_json_never_survives(self):
        plan = make_plan(FTMP / "stale.json", "--cases", "op_add_fp16")
        out = fresh("stale-out")
        run_main(plan, out)
        case = out / manifest(out)["cases"][0]["dir"]
        self.assertTrue((case / "result.json").is_file())
        argv = ["--plan", str(plan), "--out", str(out), "--workdir", str(DEMO)]  # no catalog: no Result is built
        self.assertEqual(capture(run_cmd.main, argv)[0], 0)
        self.assertFalse((case / "result.json").exists())

    def test_a_benchmark_reading_stdin_sees_end_of_file(self):
        plan = one_case_plan(
            "stdin", command=["{python}", "-c", "import sys; sys.stdin.read()"], repeats=1, allow_empty=True
        )
        t0 = time.monotonic()
        self.assertEqual(run_main(plan, fresh("stdin-out"))[0], 0)
        self.assertLess(time.monotonic() - t0, 20)

    def test_a_failed_ingests_reason_is_recorded(self):
        stub = FTMP / "failing_ingest.py"
        stub.write_text(
            "import json, sys\nprint(json.dumps({'status': 'failed', 'reason': 'warehouse unreachable'}))\nsys.exit(3)\n"
        )
        out = fresh("failing-out")
        code, _, _ = run_main(
            make_plan(FTMP / "failing.json", "--cases", "op_add_fp16"),
            out,
            "--ingest",
            "--ingest-cmd",
            f"{sys.executable} {stub}",
        )
        self.assertEqual(code, 5)
        inv = manifest(out)["ingest"]["invocations"][0]
        self.assertEqual((inv["returncode"], inv["reason"], inv["stored_run_id"]), (3, "warehouse unreachable", None))

    def test_a_rejected_report_carrying_a_run_id_is_not_a_stored_run(self):
        stub = FTMP / "rejecting_ingest.py"
        stub.write_text("import json\nprint(json.dumps({'run_id': 'R-1', 'status': 'rejected', 'reason': 'no'}))\n")
        out = fresh("rejecting-out")
        run_main(
            make_plan(FTMP / "rejecting.json", "--cases", "op_add_fp16"),
            out,
            "--ingest",
            "--ingest-cmd",
            f"{sys.executable} {stub}",
        )
        m = manifest(out)
        self.assertEqual((m["stored_run_ids"], m["ingest"]["stored_run_ids"]), ([], []))

    def test_only_an_ingesting_rerun_is_blocked_by_an_earlier_run_id(self):
        plan = make_plan(FTMP / "rerun.json")
        out = fresh("rerun-out")
        self.assertEqual(run_main(plan, out, "--run-id", RUN)[0], 0)
        self.assertEqual(run_main(plan, out, "--run-id", RUN)[0], 0)  # nothing was ingested, nothing to collide with
        ing = ["--ingest", "--ingest-cmd", f"{sys.executable} -c pass"]
        self.assertEqual(run_main(plan, out, "--run-id", RUN, *ing)[0], 0)  # the earlier run did not ingest
        code, _, err = run_main(plan, out, "--run-id", RUN, *ing)
        self.assertEqual(code, 2)
        self.assertIn("with an ingest", err)

    def test_io_failures_are_not_reported_as_invalid_input(self):
        with mock.patch.object(run_cmd, "execute", side_effect=OSError(errno.ENOSPC, "No space left on device")):
            code, _, err = capture(run_cmd.main, ["--plan", "x"])
        self.assertEqual(code, cliutil.EXIT_STORE)
        self.assertIn("I/O failure", err)
        with mock.patch.object(run_cmd, "execute", side_effect=FileNotFoundError(errno.ENOENT, "gone")):
            self.assertEqual(capture(run_cmd.main, ["--plan", "x"])[0], cliutil.EXIT_INVALID)

    def test_the_node_name_comes_only_from_the_probe_file_or_node_name(self):
        env = FTMP / "node-env.json"
        env.write_text("{}")
        out = fresh("node-out")
        with mock.patch.dict(os.environ, {"GHA_RUNNER_POD_NAME": "pod-a"}):
            run_main(make_plan(FTMP / "node.json", "--cases", "op_add_fp16"), out, "--env-file", str(env))
        res = json.loads((out / manifest(out)["cases"][0]["result"]).read_text())
        self.assertEqual(res["status"]["repeats"][0]["placement"]["nodes"], [])

    def test_the_ingest_python_flag_names_the_interpreter(self):
        cmds = run_cmd.ingest_commands(
            [{"name": "c", "status": "ok", "dir": "cases/c"}],
            {"c": {"name": "c", "id_profile": "generic", "ingest": None}},
            {
                "run_id": RUN,
                "subject_id": SUBJECT,
                "arch": "x86_64",
                "component": "x",
                "out": "/o",
                "ingest_cmd": None,
                "ingest_python": "/opt/py/bin/python",
            },
        )
        self.assertEqual(cmds[0]["argv"][:2], ["/opt/py/bin/python", "-I"])
        code, _, err = run_main(
            make_plan(FTMP / "badpy.json"), fresh("badpy-out"), "--ingest", "--ingest-python", str(FTMP / "nope")
        )
        self.assertEqual(code, 2)


class PumpContractTests(unittest.TestCase):
    def test_a_pump_drains_waiting_data_and_leaves_promptly_when_a_detached_holder_keeps_the_pipe(self):
        r, w = os.pipe()
        keep = os.dup(w)  # a detached process that never closes its end
        os.write(w, b"hello")
        os.close(w)
        out = FTMP / "pump.log"
        with os.fdopen(r, "rb") as stream:
            pump = run_cmd.Capture(stream, out, 1 << 20)
            pump.start()
            time.sleep(0.3)
            pump.stop.set()
            t0 = time.monotonic()
            pump.join(timeout=5)
            self.assertFalse(pump.is_alive())
            self.assertLess(time.monotonic() - t0, 2)
        os.close(keep)
        self.assertEqual(out.read_bytes(), b"hello")

    def test_finish_pumps_waits_a_bounded_time_and_does_not_close_pipes_under_a_live_pump(self):
        class Pump:
            def __init__(self, alive):
                self.alive, self.joined, self.stop = alive, [], mock.Mock()

            def join(self, timeout=None):
                self.joined.append(timeout)

            def is_alive(self):
                return self.alive

        proc = mock.Mock()
        live, dead = Pump(True), Pump(False)
        run_cmd.finish_pumps(proc, [live, dead])
        self.assertTrue(all(t is not None and 0 < t <= 5 for t in live.joined + dead.joined))
        live.stop.set.assert_called_once()
        proc.stdout.close.assert_not_called()
        proc.stderr.close.assert_not_called()
        run_cmd.finish_pumps(proc, [dead])
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()

    def test_a_pipe_with_a_high_file_descriptor_is_served(self):
        r, w = os.pipe()
        try:
            high = os.dup2(r, 1100)
        except OSError:
            os.close(r)
            os.close(w)
            self.skipTest("this process may not open descriptors above 1024")
        os.close(r)
        os.write(w, b"high")
        os.close(w)
        out = FTMP / "high.log"
        with os.fdopen(high, "rb") as stream:
            pump = run_cmd.Capture(stream, out, 1 << 20)
            pump.start()
            pump.join(timeout=10)
        self.assertEqual(out.read_bytes(), b"high")

    def test_the_last_json_line_is_found_past_trailing_chatter(self):
        r, w = os.pipe()
        os.write(w, b'{"run_id": "A", "status": "ingested"}\nnot json\n{broken\n')
        os.close(w)
        with os.fdopen(r, "rb") as stream:
            pump = run_cmd.Capture(stream, FTMP / "lastjson.log", 1 << 20)
            pump.start()
            pump.join(timeout=10)
        self.assertEqual(json.loads(pump.last_json), {"run_id": "A", "status": "ingested"})


class PlantedLinkTests(unittest.TestCase):
    """Links a benchmark plants while it runs (its own command creates them) must not make the engine write elsewhere."""

    PLANT = """
import os, sys
out, rdir, victim, vdir, what = sys.argv[1:6]
case = os.path.dirname(rdir)
cases = os.path.dirname(case)
open(os.environ["BENCH_RAW_DIR"] + "/r.json", "w").write("{}")


def link(path, target):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    os.symlink(target, path)


if what == "status":
    link(os.path.join(rdir, "status.json"), victim)
elif what == "case":
    link(os.path.join(case, "case.json"), victim)
elif what == "result":
    link(os.path.join(case, "result.json"), victim)
elif what == "run":
    link(os.path.join(out, "run.json"), victim)
elif what == "plan":
    link(os.path.join(out, "plan.redacted.json"), victim)
elif what == "cases":
    os.rename(cases, cases + ".moved")
    os.symlink(vdir, cases)
"""

    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.script = FTMP / "plant.py"
        self.script.write_text(self.PLANT)

    def plan(self, label, what, victim, vdir):
        command = ["{python}", str(self.script), "{out_dir}", "{repeat_dir}", str(victim), str(vdir), what]
        return one_case_plan(label, command=command, repeats=1)

    def test_nothing_planted_during_a_run_is_written_through(self):
        for what in ("status", "case", "result", "run", "plan", "cases"):
            with self.subTest(planted=what):
                victim = FTMP / f"victim-{what}.txt"
                victim.write_text("untouched")
                vdir = fresh(f"victim-dir-{what}")
                out = fresh(f"planted-{what}")
                code, _, err = run_main(self.plan(f"plant-{what}", what, victim, vdir), out)
                self.assertNotIn("Traceback", err)
                self.assertEqual(victim.read_text(), "untouched")
                self.assertEqual(list(vdir.iterdir()), [])
                self.assertEqual(code, 1 if what == "cases" else 0, err)  # the moved raw output is gone: no_output
                final = out / "run.json"
                self.assertTrue(final.is_file() and not final.is_symlink())
                self.assertEqual(json.loads(final.read_text())["engine_run_id"], manifest(out)["run_id"])
                for p in out.rglob("*"):
                    if p.is_symlink():
                        self.assertNotEqual(os.readlink(p), str(victim), p)

    def test_stale_links_from_an_earlier_run_are_replaced(self):
        plan = make_plan(FTMP / "stale-links.json", "--cases", "op_add_fp16")
        out = fresh("stale-links-out")
        run_main(plan, out)
        case = out / manifest(out)["cases"][0]["dir"]
        victim = FTMP / "victim-stale.txt"
        victim.write_text("untouched")
        for name in ("case.json", "result.json"):
            (case / name).unlink()
            (case / name).symlink_to(victim)
        (out / "run.json").unlink()
        (out / "run.json").symlink_to(victim)
        (out / "plan.redacted.json").unlink()
        (out / "plan.redacted.json").symlink_to(victim)
        self.assertEqual(run_main(plan, out)[0], 0)
        self.assertEqual(victim.read_text(), "untouched")
        for name in ("case.json", "result.json"):
            self.assertTrue((case / name).is_file() and not (case / name).is_symlink(), name)

    def test_a_cases_link_that_exists_before_the_run_is_replaced(self):
        out = fresh("cases-link-out")
        vdir = fresh("cases-link-victim")
        (out / "cases").symlink_to(vdir)
        self.assertEqual(run_main(make_plan(FTMP / "cases-link.json"), out)[0], 0)
        self.assertEqual(list(vdir.iterdir()), [])
        self.assertTrue((out / "cases").is_dir() and not (out / "cases").is_symlink())

    def test_write_file_replaces_links_and_directories_and_cleans_up(self):
        root = fresh("wf")
        victim = FTMP / "victim-wf.txt"
        victim.write_text("untouched")
        (root / "a.json").symlink_to(victim)
        (root / "b.json").mkdir()
        (root / "sub").symlink_to(fresh("wf-elsewhere"))
        for name in ("a.json", "b.json", "sub/c.json"):
            planfile.write_file(root, root / name, "{}")
            self.assertTrue((root / name).is_file() and not (root / name).is_symlink(), name)
        self.assertEqual(victim.read_text(), "untouched")
        self.assertEqual(list((FTMP / "wf-elsewhere").iterdir()), [])
        self.assertFalse([p for p in root.rglob("*.tmp")])


class IngestPreflightTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)

    def interpreter(self, label, exit_code):
        path = FTMP / f"{label}.sh"
        path.write_text(f"#!/bin/sh\nexit {exit_code}\n")
        path.chmod(0o755)
        return path

    def run_with(self, label, python, *extra):
        out = fresh(f"{label}-out")
        return run_main(make_plan(FTMP / f"{label}.json"), out, "--ingest", "--ingest-python", str(python), *extra), out

    def test_an_interpreter_that_cannot_import_the_package_is_refused_before_anything_runs(self):
        (code, _, err), out = self.run_with("pre-no", self.interpreter("py-no", 1))
        self.assertEqual(code, 2)
        self.assertIn("cannot be imported", err)
        self.assertFalse((out / "cases").exists())

    def test_an_interpreter_that_can_pass_the_preflight(self):
        (code, _, err), out = self.run_with("pre-yes", self.interpreter("py-yes", 0))
        self.assertEqual(code, 5, err)  # the preflight passed and the cases ran; this stand-in ingest reports nothing
        self.assertEqual(manifest(out)["counts"]["ok"], 2)

    def test_the_probe_uses_the_flags_and_environment_of_the_real_ingest(self):
        probe = FTMP / "probe-args.txt"
        script = FTMP / "py-record.sh"
        script.write_text(
            f"#!/bin/sh\necho \"$@\" >> {probe}\nenv | sort | grep -E '^(PYTHONPATH|HOME)=' >> {probe}\nexit 0\n"
        )
        script.chmod(0o755)
        pp = fresh("pre-pythonpath")
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/caller/path"}):
            self.run_with("pre-flags", script, "--ingest-pythonpath", str(pp))
            first = probe.read_text().splitlines()
            probe.unlink()
            self.run_with("pre-flags2", script)
            second = probe.read_text().splitlines()
        self.assertEqual(first[0], "-s -P -c import spyre_clickhouse_ingest")
        self.assertIn(f"PYTHONPATH={os.path.realpath(pp)}", first)
        self.assertEqual(second[0], "-I -c import spyre_clickhouse_ingest")
        self.assertNotIn("PYTHONPATH=/caller/path", second)

    def test_a_dry_run_and_a_custom_command_skip_the_probe_and_the_store_check(self):
        out = fresh("dry-store")
        code, _, err = capture(
            run_cmd.main,
            [
                "--plan",
                str(make_plan(FTMP / "dry-store.json")),
                "--out",
                str(out),
                "--workdir",
                str(DEMO),
                "--store",
                "clickhouse",
                "--ingest",
                "--dry-run",
            ],
        )
        self.assertEqual(code, 0, err)
        code, _, _ = run_main(
            make_plan(FTMP / "custom-skip.json"),
            fresh("custom-skip-out"),
            "--ingest",
            "--ingest-python",
            str(self.interpreter("py-no2", 1)),
            "--ingest-cmd",
            f"{sys.executable} -c pass",
        )
        self.assertEqual(code, 0)  # a custom command is not probed


class VerdictFlagPrecedenceTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.s1, self.s2 = "7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f", "8e1f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"
        self.store = stores.MemoryStore(
            verdicts={
                self.s1: [verdict_row(run_id=self.s1)],
                self.s2: [verdict_row(run_id=self.s2)],
                OTHER: [verdict_row(run_id=OTHER)],
            },
            runs={SUBJECT: self.s2},
        )

    def call(self, *argv):
        return capture(verdict_cmd.main, list(argv), st=self.store)

    def rundir(self, label, stored):
        d = fresh(label)
        (d / "run.json").write_text(json.dumps({"engine_run_id": RUN, "stored_run_ids": [stored]}))
        return d

    def test_an_explicit_run_id_ignores_a_stale_ambient_run_dir(self):
        with mock.patch.dict(os.environ, {"BENCH_RUN_DIR": str(FTMP / "does-not-exist")}):
            code, out, err = self.call("--run-id", self.s1, "--json")
        self.assertEqual((code, json.loads(out)["run_ids"]), (0, [self.s1]), err)

    def test_an_explicit_artifact_beats_ambient_run_ids_and_dirs(self):
        d = self.rundir("vfp-dir", self.s1)
        for env in ({"BENCH_RUN_ID": OTHER}, {"RUN_ID": OTHER}, {"BENCH_RUN_DIR": str(d)}):
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                code, out, err = self.call("--artifact", SUBJECT, "--json")
                self.assertEqual(json.loads(out)["run_ids"], [self.s2], err)

    def test_the_run_dir_flag_beats_the_ambient_run_dir(self):
        flag, ambient = self.rundir("vfp-flag", self.s1), self.rundir("vfp-ambient", self.s2)
        with mock.patch.dict(os.environ, {"BENCH_RUN_DIR": str(ambient)}):
            self.assertEqual(json.loads(self.call("--run-dir", str(flag), "--json")[1])["run_ids"], [self.s1])
            self.assertEqual(json.loads(self.call("--json")[1])["run_ids"], [self.s2])  # alone, the ambient one is used

    def test_an_explicit_run_id_beats_the_ambient_run_dir_and_run_id(self):
        d = self.rundir("vfp-both", self.s2)
        with mock.patch.dict(os.environ, {"BENCH_RUN_DIR": str(d), "BENCH_RUN_ID": OTHER}):
            self.assertEqual(json.loads(self.call("--run-id", self.s1, "--json")[1])["run_ids"], [self.s1])


class HeldOpenStdinTests(unittest.TestCase):
    """A benchmark and the ingest must see end-of-file on stdin even when the engine's own stdin is a live pipe."""

    def test_neither_blocks_on_the_engines_stdin(self):
        reader = FTMP / "read_stdin.py"
        reader.write_text("import sys\nsys.stdin.read()\nopen(sys.argv[1], 'w').write('eof')\n")
        marks = [FTMP / "stdin-bench.txt", FTMP / "stdin-ingest.txt"]
        for m in marks:
            m.unlink(missing_ok=True)
        code = (
            "import sys\nsys.stdin.read()\nopen(sys.argv[1], 'w').write('eof')\n"
            "import os\nos.makedirs(os.environ['BENCH_RAW_DIR'], exist_ok=True)\n"
            "open(os.environ['BENCH_RAW_DIR'] + '/r.json', 'w').write('{}')\n"
        )
        plan = one_case_plan("stdin2", command=["{python}", "-c", code, str(marks[0])], repeats=1)
        out = fresh("stdin2-out")
        proc = subprocess.Popen(
            [sys.executable, "-m", "spyre_bench", "run", "--plan", str(plan), "--out", str(out), "--workdir", str(DEMO),
             "--ingest", "--ingest-cmd", f"{sys.executable} {reader} {marks[1]}"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": str(EXT)},
        )  # fmt: skip
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            self.fail("a child waited on the held-open stdin")
        finally:
            proc.stdin.close()
        self.assertEqual([m.read_text() for m in marks], ["eof", "eof"])


class StoredIdTests(unittest.TestCase):
    def test_a_valid_report_with_a_failing_exit_is_not_a_stored_run(self):
        stub = FTMP / "valid_but_failing.py"
        stub.write_text("import json, sys\nprint(json.dumps({'run_id': 'R-1', 'status': 'ingested'}))\nsys.exit(1)\n")
        done = run_cmd.run_ingest(
            [{"argv": [sys.executable, str(stub)], "cases": ["c"], "default": False, "threads": True}],
            {"out": str(fresh("valid-fail")), "ingest_timeout": 30, "run_id": RUN},
        )
        self.assertEqual((done[0]["returncode"], done[0]["stored_run_id"]), (1, None))

    def test_capture_does_not_follow_a_link_for_a_benchmark_log(self):
        victim = FTMP / "victim-log.txt"
        victim.write_text("untouched")
        path = FTMP / "linked.log"
        path.unlink(missing_ok=True)
        path.symlink_to(victim)
        r, w = os.pipe()
        os.write(w, b"data")
        os.close(w)
        with os.fdopen(r, "rb") as stream:
            pump = run_cmd.Capture(stream, path, 1 << 20)
            pump.start()
            pump.join(timeout=10)
        self.assertIsNotNone(pump.error)
        self.assertEqual(victim.read_text(), "untouched")

    def test_pwd_is_inherited_and_not_reported_as_withheld(self):
        env, dropped = planfile.env_filter({"PWD": "/work", "PATH": "/bin"})
        self.assertEqual((env, dropped), ({"PWD": "/work", "PATH": "/bin"}, {}))
        self.assertTrue(planfile.redact_argv(["--pwd", "x"]) == ["--pwd", "<redacted>"])  # as a flag it is a password


if __name__ == "__main__":
    unittest.main()

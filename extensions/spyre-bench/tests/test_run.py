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

"""plan, run and verdict: fake stores, a stub benchmark command and the ingest mocked at the process boundary.
python -m pytest -q -p no:cacheprovider tests/test_run.py
"""

import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from spyre_bench import cliutil, engine, ingest_adapter, planfile, store as stores
from spyre_bench import plan as plan_cmd
from spyre_bench import run as run_cmd
from spyre_bench import verdict as verdict_cmd
from spyre_bench.cli import main as cli_main
from test_identity import sibling_ingest, without_ingest

EXT = Path(__file__).resolve().parent.parent
X = EXT / "examples"
DEMO = X / "run-demo"
CENTRAL = X / "central"
GOLDEN = Path(__file__).resolve().parent / "golden"
TMP = Path(tempfile.mkdtemp(prefix="spyre-bench-run-tests-"))
NOW = "2026-11-15T03:00:00Z"
SUBJECT = "9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2"
OTHER = "3b1e6a52-9c0d-5f47-8a21-7d4c0e9b5f10"
RUN = "5c0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"
BASE = ["--catalog", str(DEMO), "--central", str(CENTRAL), "--arch", "x86_64", "--reference-node", "--now", NOW]


def setUpModule():
    if not X.is_dir():
        raise unittest.SkipTest(f"examples/ not found at {X}: the tests need a source checkout")


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def capture(fn, argv, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(argv, **kw)
    return code, out.getvalue(), err.getvalue()


def plan_main(*argv, st=None):
    return capture(plan_cmd.main, [*BASE, *argv], st=st)


def plan_doc(*argv, st=None):
    code, out, err = plan_main("--json", *argv, st=st)
    assert code == 0, (code, err)
    return json.loads(out)


def clean_env(**extra):
    """An environment without BENCH_* / RUN_ID / RESULTS_DIR leaking in from the developer's shell."""
    keep = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("BENCH_", "CLICKHOUSE_")) and k not in ("RUN_ID", "RESULTS_DIR")
    }
    return mock.patch.dict(os.environ, {**keep, **extra}, clear=True)


def copy_demo(label):
    d = TMP / label
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(DEMO, d)
    return d


def stored(rec, *, subject=SUBJECT, env_id="env-1", validity="valid", n=3, labels=None, run_id="run-old"):
    return {
        "run_id": run_id,
        "benchmark_id": rec["benchmark_id"],
        "subject_id": subject,
        "env_id": env_id,
        "labels": rec["labels"] if labels is None else labels,
        "validity": validity,
        "n": n,
        "ts": datetime(2026, 11, 14, 3, 0, tzinfo=timezone.utc),
    }


class Tokens(unittest.TestCase):
    def test_case_tokens_resolve_run_tokens_stay(self):
        argv, errs = planfile.render_case_tokens(
            ["{python}", "{case.op}", "{repeat}", "{raw_dir}/{a}"], {"case.op": "add", "a": "x"}, "w"
        )
        self.assertEqual((argv, errs), (["{python}", "add", "{repeat}", "{raw_dir}/x"], []))

    def test_unknown_token_is_reported(self):
        _, errs = planfile.render_case_tokens(["{nope}"], {}, "case c")
        self.assertEqual(len(errs), 1)
        self.assertIn("{nope}", errs[0])

    def test_braces_in_values_and_escapes_survive_both_phases(self):
        argv, _ = planfile.render_case_tokens(["{v}", "{{lit}}"], {"v": "a{b}c"}, "w")
        vals = {k: "" for k in planfile.RUN_TOKENS}
        self.assertEqual(planfile.render_tokens(argv, vals, planfile.RUN_TOKENS, "w"), ["a{b}c", "{lit}"])

    def test_run_phase_rejects_unknown_tokens(self):
        with self.assertRaises(ValueError):
            planfile.render_tokens(["{case}"], {"case": "x"}, planfile.INGEST_TOKENS[:2], "w")

    def test_list_and_bool_values(self):
        self.assertEqual(planfile.token_value([1, 512]), "1,512")
        self.assertEqual(planfile.token_value(True), "true")

    def test_slug_is_collision_safe(self):
        self.assertNotEqual(planfile.slug("a/b"), planfile.slug("a_b"))
        self.assertRegex(planfile.slug("a=1,b/c"), r"^a_1_b_c-[0-9a-f]{8}$")


class PlanTests(unittest.TestCase):
    def test_golden_plan(self):
        doc = plan_doc("--purpose", "nightly", "--subject-id", SUBJECT)
        doc["producer"]["engine"]["version"] = "<version>"
        doc["producer"]["ingest"] = "<ingest>"
        want = json.loads((GOLDEN / "plan-run-demo.json").read_text())
        self.assertEqual(doc, want)

    def test_table_and_counts(self):
        code, out, _ = plan_main("--purpose", "nightly", "--subject-id", SUBJECT)
        self.assertEqual(code, 0)
        self.assertIn("run     op_add_fp16", out)
        self.assertIn("plan: run 2, reuse 0, skip 0", out)

    def test_out_writes_the_same_document(self):
        path = TMP / "plan-out.json"
        code, out, _ = plan_main("--purpose", "nightly", "--out", str(path), "--json")
        self.assertEqual(json.loads(path.read_text()), json.loads(out))

    def test_plan_schema_and_identity_version(self):
        doc = plan_doc("--purpose", "nightly")
        self.assertEqual(doc["plan_schema"], "bench.plan/1")
        eng = doc["producer"]["engine"]
        self.assertEqual((eng["plan_schema"], eng["identity_version"]), ("bench.plan/1", planfile.IDENTITY_VERSION))

    def test_command_is_rendered_with_case_values(self):
        rec = plan_doc("--purpose", "nightly")["cases"][0]
        self.assertEqual(
            rec["command"],
            [
                "{python}",
                "stub_bench.py",
                "--op",
                "add",
                "--shape",
                "1,512",
                "--raw",
                "{raw_dir}",
                "--repeat",
                "{repeat}",
                "--out",
                "{repeat_dir}",
            ],
        )
        self.assertEqual((rec["id_mode"], rec["id_profile"]), ("engine", None))

    def test_unknown_command_token_is_a_plan_error(self):
        d = copy_demo("bad-token")
        p = d / "ops" / "op-add.yaml"
        p.write_text(p.read_text().replace('"{case.op}"', '"{case.nope}"'))
        code, _, err = capture(
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
            ],
        )
        self.assertEqual(code, 2)
        self.assertIn("{case.nope}", err)

    def test_selection_by_name_and_cases(self):
        doc = plan_doc("--purpose", "nightly", "--select", "smoke")
        self.assertEqual([c["name"] for c in doc["cases"]], ["op_add_fp16"])
        doc = plan_doc("--purpose", "nightly", "--cases", "op_mul_fp16")
        self.assertEqual([c["name"] for c in doc["cases"]], ["op_mul_fp16"])

    def test_tier_flag(self):
        doc = plan_doc("--purpose", "nightly", "--tier", "sanity")
        self.assertEqual([c["name"] for c in doc["cases"]], ["op_add_fp16"])

    def test_set_overrides(self):
        doc = plan_doc("--purpose", "nightly", "--set", "run.repeats=5", "--set", "run.cache=both")
        self.assertEqual({(c["repeats"], c["cache"]) for c in doc["cases"]}, {(5, "both")})
        code, _, err = plan_main("--purpose", "nightly", "--set", "run.timeout=5")
        self.assertEqual(code, 2)
        self.assertIn("run.timeout", err)

    def test_env_twins_and_flag_precedence(self):
        base = ["--catalog", str(DEMO), "--central", str(CENTRAL), "--now", NOW, "--json"]
        with clean_env(
            BENCH_PURPOSE="nightly",
            BENCH_ARCH="x86_64",
            BENCH_REFERENCE_NODE="1",
            BENCH_CASES="op_mul_fp16",
            BENCH_SUBJECT_ID=SUBJECT,
            BENCH_SET="run.repeats=4",
        ):
            code, out, _ = capture(plan_cmd.main, base)
            doc = json.loads(out)
            self.assertEqual([(c["name"], c["repeats"]) for c in doc["cases"]], [("op_mul_fp16", 4)])
            self.assertEqual((doc["purpose"], doc["request"]["subject"]["artifact_id"]), ("nightly", SUBJECT))
            code, out, _ = capture(plan_cmd.main, [*base, "--cases", "op_add_fp16", "--set", "run.repeats=2"])
            doc = json.loads(out)
            self.assertEqual([(c["name"], c["repeats"]) for c in doc["cases"]], [("op_add_fp16", 2)])

    def test_catalog_env_twin(self):
        with clean_env(
            BENCH_DIR=str(DEMO),
            BENCH_CENTRAL=str(CENTRAL),
            BENCH_PURPOSE="nightly",
            BENCH_ARCH="x86_64",
            BENCH_REFERENCE_NODE="1",
        ):
            code, out, _ = capture(plan_cmd.main, [])
        self.assertEqual(code, 0)
        self.assertIn("plan: run 2", out)

    def test_request_file(self):
        f = TMP / "req.yaml"
        f.write_text(
            f"purpose: nightly\nsubject: {{artifact_id: {SUBJECT}}}\nselection: {{name: smoke}}\nbudget: {{accelerator_hours: 100}}\n"
        )
        doc = plan_doc("--request", str(f))
        self.assertEqual([c["name"] for c in doc["cases"]], ["op_add_fp16"])
        self.assertEqual(doc["request"]["budget"], {"device_hours": 100})
        self.assertEqual(doc["request"]["subject"]["artifact_id"], SUBJECT)

    def test_invalid_requests_exit_2(self):
        for argv in (
            ["--purpose", "no-such-purpose"],
            [],
            ["--purpose", "nightly", "--cases", "nope"],
            ["--purpose", "nightly", "--tier", "huge"],
            ["--purpose", "nightly", "--budget", "0.001"],
            ["--purpose", "nightly", "--reuse", "sometimes"],
            ["--purpose", "nightly", "--devices", "bad"],
        ):
            with self.subTest(argv=argv), clean_env():
                code, _, err = plan_main(*argv)
                self.assertEqual(code, 2, err)
                self.assertTrue(err.startswith("error:"), err)

    def test_missing_catalog_exits_2(self):
        with clean_env():
            code, _, err = capture(plan_cmd.main, ["--catalog", str(TMP / "nowhere"), "--purpose", "nightly"])
        self.assertEqual(code, 2)
        self.assertIn("not a directory", err)

    def test_every_case_skipped_exits_4(self):
        code, out, _ = capture(plan_cmd.main, [*BASE[:-3], "--arch", "x86_64", "--now", NOW, "--purpose", "nightly"])
        self.assertEqual(code, 4)
        self.assertIn("reference_node_required", out)

    def test_capacity_checked_only_when_stated(self):
        code, out, _ = plan_main("--json", "--purpose", "nightly", "--devices", "spyre/pf=0")
        self.assertEqual(code, 4)
        doc = json.loads(out)
        self.assertEqual({c["reason"] for c in doc["cases"]}, {"capacity"})
        self.assertEqual(doc["summary"]["reasons"], {"capacity": 2})
        doc = plan_doc("--purpose", "nightly", "--devices", "spyre/pf=1")
        self.assertEqual(doc["summary"]["run"], 2)


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.first = plan_doc("--purpose", "nightly", "--subject-id", SUBJECT)["cases"][0]

    def reuse_plan(self, *rows, extra=(), store=None):
        st = store or stores.MemoryStore(results=rows)
        doc = plan_doc(
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
            *extra,
            st=st,
        )
        return doc["cases"][0], st

    def test_matching_stored_result_is_reused(self):
        rec, st = self.reuse_plan(stored(self.first))
        self.assertEqual(
            (rec["action"], rec["reused_from"], rec["reason"]), ("reuse", "run-old", "same_subject+same_environment")
        )
        self.assertEqual(st.calls, ["stored_results"])

    def test_default_is_never_and_the_store_is_not_read(self):
        st = stores.MemoryStore(results=[stored(self.first)])
        doc = plan_doc("--purpose", "nightly", "--subject-id", SUBJECT, "--env-id", "env-1", st=st)
        self.assertEqual(doc["summary"]["reuse"], 0)
        self.assertEqual(st.calls, [])

    def test_each_mismatch_means_run_with_its_reason(self):
        cases = {
            "different_subject": stored(self.first, subject=OTHER),
            "different_environment": stored(self.first, env_id="env-2"),
            "validity_below_request": stored(self.first, validity="invalid"),
            "repeats_below_request": stored(self.first, n=1),
        }
        for why, row in cases.items():
            with self.subTest(why=why):
                st = stores.MemoryStore(results=[row])
                st.stored_results = lambda ids, subject, _r=row: [
                    _r
                ]  # bypass the subject filter to reach the engine rule
                rec, _ = self.reuse_plan(store=st)
                self.assertEqual((rec["action"], rec["reason"]), ("run", why))

    def test_pinning_label_mismatch_is_never_a_reuse(self):
        rec, _ = self.reuse_plan(
            stored(self.first, labels={**self.first["labels"], "env.label.cpu_pinning": "device_local"})
        )
        self.assertEqual((rec["action"], rec["reason"]), ("run", "different_environment"))

    def test_unknown_stored_data_means_run(self):
        row = stored(self.first, env_id=None, validity="invalid", labels={})
        rec, _ = self.reuse_plan(row)
        self.assertEqual(rec["action"], "run")

    def test_no_env_id_on_the_leg_means_run(self):
        st = stores.MemoryStore(results=[stored(self.first)])
        doc = plan_doc("--purpose", "nightly", "--subject-id", SUBJECT, "--reuse", "if_valid", st=st)
        self.assertEqual(doc["summary"]["reuse"], 0)

    def test_force_overrides_a_match(self):
        rec, _ = self.reuse_plan(stored(self.first), extra=["--force", "op_add_fp16"])
        self.assertEqual((rec["action"], rec["reason"]), ("run", "forced"))

    def test_reuse_needs_a_subject(self):
        doc = plan_doc("--purpose", "nightly", "--cases", "op_add_fp16", "--reuse", "if_valid", st=stores.MemoryStore())
        self.assertEqual(doc["cases"][0]["reason"], "reuse_needs_subject")

    def test_only_mode_reports_missing_as_skip(self):
        code, out, _ = plan_main(
            "--json",
            "--purpose",
            "nightly",
            "--subject-id",
            SUBJECT,
            "--cases",
            "op_add_fp16,op_mul_fp16",
            "--reuse",
            "only",
            "--env-id",
            "env-1",
            st=stores.MemoryStore(),
        )
        self.assertEqual(code, 4)  # nothing to run: every case is skipped
        rec = json.loads(out)["cases"][0]
        self.assertEqual(rec["action"], "skip")
        self.assertEqual(rec["reason"], "missing:no_result_for_case")

    def test_request_reuse_overrides_the_selection_only_when_asked(self):
        d = copy_demo("sel-reuse")
        p = d / "project.yaml"
        p.write_text(p.read_text().replace("run: {repeats_min: 1}", "run: {repeats_min: 1}\n  reuse: {mode: if_valid}"))
        base = [
            "--catalog",
            str(d),
            "--central",
            str(CENTRAL),
            "--arch",
            "x86_64",
            "--reference-node",
            "--now",
            NOW,
            "--json",
        ]
        args = ["--purpose", "nightly", "--subject-id", SUBJECT, "--select", "smoke", "--env-id", "env-1"]
        row = stored(self.first)
        code, out, _ = capture(plan_cmd.main, [*base, *args], st=stores.MemoryStore(results=[row]))
        self.assertEqual(json.loads(out)["cases"][0]["action"], "reuse")  # the Selection's own rule
        code, out, _ = capture(plan_cmd.main, [*base, *args, "--reuse", "never"], st=stores.MemoryStore(results=[row]))
        self.assertEqual(json.loads(out)["cases"][0]["action"], "run")  # the flag overrides it

    def test_force_and_max_age_need_a_reuse_mode(self):
        for extra in (["--force", "all"], ["--max-age", "7d"]):
            with self.subTest(extra=extra):
                code, _, err = plan_main("--purpose", "nightly", *extra)
                self.assertEqual(code, 2)
                self.assertIn("--reuse", err)
        code, _, _ = plan_main("--purpose", "nightly", "--reuse", "if_valid", "--force", "all", "--max-age", "7d")
        self.assertEqual(code, 0)

    def test_store_failure_exits_5_only_when_a_lookup_is_needed(self):
        st = stores.MemoryStore(fail="connection refused")
        code, _, err = plan_main(
            "--purpose", "nightly", "--subject-id", SUBJECT, "--reuse", "if_valid", "--env-id", "e", st=st
        )
        self.assertEqual(code, 5)
        self.assertIn("connection refused", err)
        code, _, _ = plan_main("--purpose", "nightly", "--subject-id", SUBJECT, st=st)
        self.assertEqual(code, 0)

    def test_unconfigured_clickhouse_store_exits_5(self):
        with clean_env():
            code, _, err = plan_main(
                "--purpose", "nightly", "--subject-id", SUBJECT, "--reuse", "if_valid", "--store", "clickhouse"
            )
        self.assertEqual(code, 5)
        self.assertIn("CLICKHOUSE_HOST", err)


class FakeClient:
    def __init__(self, rows=(), columns=(), fail=None):
        self.queries, self.rows, self.columns, self.fail = [], rows, columns, fail

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters))
        if self.fail:
            raise RuntimeError(self.fail)
        return mock.Mock(column_names=list(self.columns), result_rows=[tuple(r) for r in self.rows])


class ClickHouseStoreTests(unittest.TestCase):
    def store(self, **kw):
        c = FakeClient(**kw)
        return stores.ClickHouseStore(client=c, database="spyre_v2"), c

    def test_stored_results_maps_props_and_defaults_unknown_to_invalid(self):
        ts = datetime(2026, 11, 14, tzinfo=timezone.utc)
        rows = [
            (RUN, SUBJECT, ts, [{"env.id": "e1", "env.label.device.count": "1", "run.validity": "valid"}], 3),
            (OTHER, SUBJECT, ts, [{}], 0),
        ]
        st, c = self.store(rows=rows, columns=["run_key", "benchmark_key", "stored_at", "props_all", "n"])
        out = st.stored_results([SUBJECT], SUBJECT)
        self.assertEqual(
            (out[0]["env_id"], out[0]["labels"], out[0]["validity"], out[0]["n"]),
            ("e1", {"env.label.device.count": "1"}, "valid", 3),
        )
        self.assertEqual((out[1]["env_id"], out[1]["validity"], out[1]["labels"]), (None, "invalid", {}))
        sql, params = c.queries[0]
        self.assertIn("`spyre_v2`.v_benchmark_results_enriched", sql)
        self.assertEqual(params, {"subject": SUBJECT, "ids": [SUBJECT]})

    def test_naive_timestamps_are_utc_and_missing_ones_are_dropped(self):
        rows = [(RUN, SUBJECT, datetime(2026, 11, 14), [{}], 1), (OTHER, SUBJECT, None, [{}], 1)]
        st, _ = self.store(rows=rows, columns=["run_key", "benchmark_key", "stored_at", "props_all", "n"])
        out = st.stored_results([SUBJECT], SUBJECT)
        self.assertEqual([(r["run_id"], r["ts"].tzinfo) for r in out], [(RUN, timezone.utc)])

    def test_queries_use_typed_uuid_parameters(self):
        st, c = self.store(rows=[], columns=["benchmark_key"])
        st.stored_results([SUBJECT], SUBJECT)
        st.landed(RUN)
        st.latest_run(SUBJECT)
        st.verdict_rows(RUN)
        sqls = " ".join(q[0] for q in c.queries)
        self.assertNotIn("toString(run_id) =", sqls)
        self.assertNotIn("toString(artifact_id) =", sqls)
        self.assertIn("{subject:UUID}", c.queries[0][0])
        self.assertIn("v_benchmark_metric_verdicts", c.queries[2][0])

    def test_non_uuid_inputs_query_nothing(self):
        st, c = self.store()
        self.assertEqual(st.stored_results(["x"], SUBJECT), [])
        self.assertEqual(st.stored_results([SUBJECT], "x"), [])
        self.assertEqual(c.queries, [])

    def test_landed_latest_run_and_verdict_rows(self):
        st, c = self.store(rows=[(SUBJECT,)], columns=["benchmark_key"])
        self.assertEqual(st.landed(RUN), {SUBJECT})
        self.assertEqual(c.queries[0][1], {"run": RUN})
        st, c = self.store(rows=[(RUN,)], columns=["run_key"])
        self.assertEqual(st.latest_run(SUBJECT), RUN)
        st, c = self.store(rows=[], columns=["run_key"])
        self.assertIsNone(st.latest_run(SUBJECT))
        st, c = self.store(rows=[(RUN, "n")], columns=["run_id", "name"])
        self.assertEqual(st.verdict_rows(RUN), [{"run_id": RUN, "name": "n"}])
        self.assertIn("v_benchmark_metric_verdicts", c.queries[0][0])
        with self.assertRaises(stores.StoreError):
            st.verdict_rows("not-a-uuid")

    def test_query_failure_is_a_store_error(self):
        st, _ = self.store(fail="boom")
        with self.assertRaises(stores.StoreError) as cm:
            st.landed(RUN)
        self.assertIn("boom", str(cm.exception))

    def test_connection_settings(self):
        env = {"CLICKHOUSE_HOST": "h", "CLICKHOUSE_PASS": "p", "CLICKHOUSE_DB_V2": "v2", "CLICKHOUSE_SECURE": "0"}
        s = stores.connection_settings(env)
        self.assertEqual(
            (s["host"], s["port"], s["username"], s["secure"], s["database"]), ("h", 443, "default", False, "v2")
        )
        self.assertEqual(stores.connection_settings({**env, "BENCH_CLICKHOUSE_DB": "dev"})["database"], "dev")
        for bad in ({}, {**env, "CLICKHOUSE_PORT": "x"}, {**env, "CLICKHOUSE_DB_V2": "a;b"}, {"CLICKHOUSE_HOST": "h"}):
            with self.subTest(bad=bad), self.assertRaises(stores.StoreError):
                stores.connection_settings(bad)

    def test_open_store(self):
        self.assertIsInstance(stores.open_store("none"), stores.NullStore)
        self.assertIsInstance(stores.open_store("clickhouse"), stores.ClickHouseStore)
        with self.assertRaises(stores.StoreError):
            stores.open_store("mystery")

    def test_missing_driver_names_the_extra(self):
        with (
            clean_env(CLICKHOUSE_HOST="h", CLICKHOUSE_PASS="p", BENCH_CLICKHOUSE_DB="d"),
            mock.patch.dict(sys.modules, {"clickhouse_connect": None}),
        ):
            with self.assertRaises(stores.StoreError) as cm:
                stores.ClickHouseStore().landed(RUN)
        self.assertIn("spyre-bench[clickhouse]", str(cm.exception))


def still_running(pid):
    """True while the process exists and is not a zombie (a container's PID 1 may not reap orphans)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def ingest_ctx(**kw):
    return {
        "run_id": RUN,
        "run_key": f"manual:spyre-bench:{RUN}",
        "subject_id": SUBJECT,
        "arch": "x86_64",
        "component": "run-demo",
        "out": "/o",
        "ingest_cmd": None,
        **kw,
    }


def make_plan(path, *extra, subject=SUBJECT, catalog=DEMO):
    argv = [
        "--catalog", str(catalog), "--central", str(CENTRAL), "--arch", "x86_64", "--reference-node", "--now", NOW,
        "--purpose", "nightly", "--out", str(path), *(["--subject-id", subject] if subject else []), *extra,
    ]  # fmt: skip
    code, out, err = capture(plan_cmd.main, argv)
    assert code in (0, 4), (code, err)
    return path


def run_main(plan, out, *extra, st=None, catalog=DEMO, popen=None):
    argv = [
        "--plan",
        str(plan),
        "--out",
        str(out),
        "--workdir",
        str(catalog),
        "--catalog",
        str(catalog),
        "--central",
        str(CENTRAL),
        *extra,
    ]
    kw = {"st": st}
    if popen:
        kw["popen"] = popen
    return capture(run_cmd.main, argv, **kw)


def manifest(out):
    return json.loads((Path(out) / "run.json").read_text())


class RunTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.n = len(list(TMP.glob("run-*")))
        self.out = TMP / f"run-{self.n}-{self.id().split('.')[-1]}"
        self.plan = make_plan(TMP / f"plan-{self.out.name}.json")

    def test_all_ok_writes_the_documented_layout(self):
        code, out, err = run_main(self.plan, self.out, "--run-id", RUN)
        self.assertEqual(code, 0, err)
        m = manifest(self.out)
        self.assertEqual((m["run_schema"], m["run_id"], m["exit_reason"]), ("bench.run/1", RUN, "ok"))
        self.assertEqual(
            m["counts"], {"planned": 2, "run": 2, "ok": 2, "failed": 0, "reuse": 0, "skip": 0, "not_selected": 0}
        )
        case = self.out / m["cases"][0]["dir"]
        for rel in (
            "case.json",
            "raw/add-1.json",
            "repeat-001/status.json",
            "repeat-001/stdout.log",
            "repeat-003/metrics.json",
            "result.json",
        ):
            self.assertTrue((case / rel).is_file(), rel)
        self.assertTrue((self.out / "plan.redacted.json").is_file())
        st = json.loads((case / "repeat-002/status.json").read_text())
        self.assertEqual((st["status"], st["returncode"], st["repeat"], st["cache"]), ("ok", 0, 2, "warm"))
        self.assertEqual(st["argv"][1:3], ["stub_bench.py", "--op"])
        self.assertIn("ingest skipped", out)
        self.assertEqual(m["ingest"], {"requested": False, "status": "skipped", "invocations": []})
        self.assertEqual(m["rows_landed"]["status"], "unchecked")

    def test_result_is_an_engine_result_that_recomputes(self):
        run_main(self.plan, self.out, "--run-id", RUN)
        m = manifest(self.out)
        res = json.loads((self.out / m["cases"][0]["result"]).read_text())
        self.assertEqual(engine.shape_schema("result").errors(res), [])
        self.assertEqual(engine.recompute(res), [])
        st = res["status"]
        self.assertEqual(st["subject"]["artifact_id"], SUBJECT)
        self.assertEqual(st["aggregate"]["n"], 3)
        self.assertEqual(st["aggregate"]["metrics"]["kernel.time.mean"]["value"], 1.02)
        self.assertEqual(m["result_checks"], {"op_add_fp16": [], "op_mul_fp16": []})
        # unknown facts (no probe file) lower validity and are named
        self.assertEqual(st["validity"]["state"], "suspect")
        self.assertIn("qualification_not_current", st["validity"]["reasons"])

    def test_env_file_supplies_fingerprint_and_facts(self):
        f = TMP / "env.json"
        f.write_text(
            json.dumps(
                {
                    "fingerprint": {"env.fp.kernel": "6.8", "env.fp.bogus": "x"},
                    "node": "node-a",
                    "qualified": True,
                    "clean": {"exclusive": "observed"},
                }
            )
        )
        run_main(self.plan, self.out, "--env-file", str(f))
        m = manifest(self.out)
        res = json.loads((self.out / m["cases"][0]["result"]).read_text())["status"]
        self.assertEqual(res["environment"]["fingerprint"], {"env.fp.kernel": "6.8"})
        self.assertEqual(res["repeats"][0]["placement"]["nodes"], ["node-a"])
        self.assertNotIn("qualification_not_current", res["validity"]["reasons"])
        self.assertTrue(any("env.fp.bogus" in n for n in m["result_notes"]["op_add_fp16"]))
        self.assertEqual(m["result_checks"]["op_add_fp16"], [])

    def test_some_failed_exits_1_and_keeps_going(self):
        with mock.patch.dict(os.environ, {"STUB_FAIL": "add"}):
            code, out, _ = run_main(self.plan, self.out, "--pass-env", "STUB_FAIL")
        self.assertEqual(code, 1)
        m = manifest(self.out)
        self.assertEqual((m["counts"]["ok"], m["counts"]["failed"], m["exit_reason"]), (1, 1, "cases_failed"))
        bad = next(c for c in m["cases"] if c["name"] == "op_add_fp16")
        self.assertEqual((bad["status"], bad["reason"], bad["repeats"]["failed"]), ("failed", "repeats_failed", 3))
        st = json.loads((self.out / bad["dir"] / "repeat-001/status.json").read_text())
        self.assertEqual((st["status"], st["reason"], st["returncode"]), ("failed", "exit_3", 3))

    def test_nothing_ran_exits_4(self):
        plan = make_plan(TMP / "skipall.json", "--devices", "spyre/pf=0")
        code, out, _ = run_main(plan, self.out)
        self.assertEqual(code, 4)
        m = manifest(self.out)
        self.assertEqual((m["exit_reason"], m["counts"]["run"], m["counts"]["skip"]), ("nothing_ran", 0, 2))
        self.assertEqual({c["status"] for c in m["cases"]}, {"skipped"})

    def test_all_reused_is_ok(self):
        first = json.loads(self.plan.read_text())["cases"][0]
        st = stores.MemoryStore(results=[stored(first), stored(json.loads(self.plan.read_text())["cases"][1])])
        plan = make_plan(TMP / "reuseall.json", "--reuse", "if_valid", "--env-id", "env-1")
        # the plan above ran without a store: build the all-reuse plan by planning with the fake store
        code, out, err = capture(
            plan_cmd.main,
            [
                *BASE,
                "--purpose",
                "nightly",
                "--subject-id",
                SUBJECT,
                "--reuse",
                "if_valid",
                "--env-id",
                "env-1",
                "--out",
                str(plan),
            ],
            st=st,
        )
        self.assertEqual(json.loads(plan.read_text())["summary"]["reuse"], 2, err)
        code, _, _ = run_main(plan, self.out)
        self.assertEqual(code, 0)
        m = manifest(self.out)
        self.assertEqual((m["exit_reason"], {c["status"] for c in m["cases"]}), ("ok", {"reused"}))
        self.assertEqual(list((self.out / "cases").glob("*")) if (self.out / "cases").exists() else [], [])

    def test_reuse_plus_skip_is_not_green(self):
        doc = json.loads(self.plan.read_text())
        doc["cases"][0].update(action="reuse", reused_from="r")
        doc["cases"][1].update(action="skip", reason="capacity")
        p = TMP / "mixed.json"
        p.write_text(json.dumps(doc))
        self.assertEqual(run_main(p, self.out)[0], 4)

    def test_no_command_is_a_failed_case(self):
        doc = json.loads(self.plan.read_text())
        doc["cases"][0]["command"] = None
        p = TMP / "nocmd.json"
        p.write_text(json.dumps(doc))
        code, _, _ = run_main(p, self.out)
        self.assertEqual(code, 1)
        bad = next(c for c in manifest(self.out)["cases"] if c["name"] == "op_add_fp16")
        self.assertEqual((bad["status"], bad["reason"], bad["result"]), ("failed", "no_command", None))

    def test_timeout_kills_the_command(self):
        doc = json.loads(self.plan.read_text())
        doc["cases"] = doc["cases"][:1]
        doc["cases"][0]["command"] = ["{python}", "-c", "import time; time.sleep(60)"]
        doc["cases"][0]["repeats"] = 1
        p = TMP / "slow.json"
        p.write_text(json.dumps(doc))
        t0 = time.monotonic()
        code, _, _ = run_main(p, self.out, "--timeout", "1")
        self.assertLess(time.monotonic() - t0, 20)
        self.assertEqual(code, 1)
        case = manifest(self.out)["cases"][0]
        st = json.loads((self.out / case["dir"] / "repeat-001/status.json").read_text())
        self.assertEqual((st["status"], st["reason"], st["returncode"]), ("failed", "timeout", None))

    def test_missing_executable_is_a_failed_repeat(self):
        doc = json.loads(self.plan.read_text())
        doc["cases"][0]["command"] = ["/no/such/binary"]
        p = TMP / "nobin.json"
        p.write_text(json.dumps(doc))
        self.assertEqual(run_main(p, self.out)[0], 1)
        case = next(c for c in manifest(self.out)["cases"] if c["name"] == "op_add_fp16")
        st = json.loads((self.out / case["dir"] / "repeat-001/status.json").read_text())
        self.assertTrue(st["reason"].startswith("spawn_failed"))

    def test_cache_both_runs_cold_then_warm(self):
        plan = make_plan(
            TMP / "both.json", "--cases", "op_mul_fp16", "--set", "run.cache=both", "--set", "run.repeats=1"
        )
        run_main(plan, self.out)
        case = self.out / manifest(self.out)["cases"][0]["dir"]
        caches = [json.loads((case / f"repeat-00{i}/status.json").read_text())["cache"] for i in (1, 2)]
        self.assertEqual(caches, ["cold", "warm"])

    def test_command_receives_run_tokens_and_env(self):
        script = TMP / "probe_env.py"
        script.write_text(
            "import json, os, sys; open(sys.argv[1] + '/metrics.json', 'w').write(json.dumps({'props': {'run.seen': os.environ['BENCH_CASE'] + '|' + os.environ['BENCH_RUN_ID'] + '|' + os.environ['BENCH_REPEAT'] + '|' + os.environ.get('BENCH_QUIET', '')}}))"
        )
        doc = json.loads(self.plan.read_text())
        doc["cases"] = doc["cases"][:1]
        doc["cases"][0].update(command=["{python}", str(script), "{repeat_dir}", "{run_id}"], repeats=1)
        p = TMP / "envprobe.json"
        p.write_text(json.dumps(doc))
        run_main(p, self.out, "--run-id", RUN, "--quiet")
        case = manifest(self.out)["cases"][0]
        res = json.loads((self.out / case["result"]).read_text())
        self.assertEqual(res["status"]["repeats"][0]["props"]["run.seen"], f"op_add_fp16|{RUN}|1|1")
        st = json.loads((self.out / case["dir"] / "repeat-001/status.json").read_text())
        self.assertEqual(st["argv"][-1], RUN)

    def test_case_filter(self):
        run_main(self.plan, self.out, "--case", "op_mul_fp16")
        m = manifest(self.out)
        self.assertEqual(
            [(c["name"], c["status"]) for c in m["cases"]], [("op_mul_fp16", "ok"), ("op_add_fp16", "not_selected")]
        )
        self.assertEqual((m["counts"]["run"], m["counts"]["not_selected"]), (1, 1))
        code, _, err = run_main(self.plan, TMP / "x", "--case", "nope")
        self.assertEqual(code, 2)
        self.assertIn("not in the plan", err)

    def test_dry_run_creates_and_runs_nothing(self):
        out = TMP / "dry-out"
        shutil.rmtree(out, ignore_errors=True)
        with mock.patch("subprocess.Popen") as popen:
            code, text, _ = run_main(self.plan, out, "--dry-run", "--ingest")
        self.assertEqual(code, 0)
        popen.assert_not_called()
        self.assertFalse(out.exists())
        self.assertEqual(len(text.strip().splitlines()), 5)
        self.assertIn("stub_bench.py", text)
        code, text, _ = run_main(self.plan, out, "--dry-run", "--json")
        self.assertEqual(len(json.loads(text)["commands"]), 5)

    def test_dry_run_with_nothing_to_run_exits_4(self):
        plan = make_plan(TMP / "skipall2.json", "--devices", "spyre/pf=0")
        self.assertEqual(run_main(plan, TMP / "dry2", "--dry-run")[0], 4)

    def test_engine_skew_exits_2(self):
        for key, value in (("plan_schema", "bench.plan/2"), ("identity", 99)):
            doc = json.loads(self.plan.read_text())
            if key == "identity":
                doc["producer"]["engine"]["identity_version"] = value
            else:
                doc[key] = value
            p = TMP / "skew.json"
            p.write_text(json.dumps(doc))
            with self.subTest(key=key):
                code, _, err = run_main(p, self.out)
                self.assertEqual(code, 2)
                self.assertIn("engine_skew", err)

    def test_plan_catalog_mismatch_exits_2(self):
        doc = json.loads(self.plan.read_text())
        doc["cases"][0]["benchmark_id"] = OTHER
        p = TMP / "mismatch.json"
        p.write_text(json.dumps(doc))
        code, _, err = run_main(p, self.out)
        self.assertEqual(code, 2)
        self.assertIn("re-plan", err)

    def test_run_without_a_catalog_still_runs_but_builds_no_result(self):
        argv = [
            "--plan",
            str(self.plan),
            "--out",
            str(self.out),
            "--workdir",
            str(DEMO),
            "--catalog",
            str(TMP / "none"),
        ]
        code, _, _ = capture(run_cmd.main, argv)
        self.assertEqual(code, 0)
        self.assertEqual({c["result"] for c in manifest(self.out)["cases"]}, {None})

    def test_bad_plan_files_exit_2(self):
        bad = TMP / "bad.json"
        bad.write_text("[]")
        self.assertEqual(run_main(bad, self.out)[0], 2)
        self.assertEqual(run_main(TMP / "absent.json", self.out)[0], 2)
        self.assertEqual(capture(run_cmd.main, [])[0], 2)

    def test_run_id_must_be_a_uuid_and_falls_back_for_ci_numbers(self):
        self.assertEqual(run_main(self.plan, self.out, "--run-id", "12345")[0], 2)
        with mock.patch.dict(os.environ, {"RUN_ID": "12345"}):
            self.assertEqual(run_main(self.plan, self.out)[0], 0)
        uuid_ = json.loads((self.out / "run.json").read_text())["run_id"]
        self.assertNotEqual(uuid_, "12345")
        with mock.patch.dict(os.environ, {"RUN_ID": RUN}):
            run_main(self.plan, self.out)
        self.assertEqual(manifest(self.out)["run_id"], RUN)

    def test_results_dir_env_twins(self):
        out = TMP / "twin-out"
        shutil.rmtree(out, ignore_errors=True)
        with mock.patch.dict(
            os.environ, {"RESULTS_DIR": str(out), "BENCH_PLAN": str(self.plan), "BENCH_WORKDIR": str(DEMO)}
        ):
            code, _, _ = capture(run_cmd.main, ["--catalog", str(DEMO), "--central", str(CENTRAL)])
        self.assertEqual(code, 0)
        self.assertTrue((out / "run.json").is_file())

    def test_performance_numbers_are_never_printed(self):
        code, out, err = run_main(self.plan, self.out, "--json")
        self.assertNotIn("1.02", out + err)


class RunSafetyTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.out = TMP / f"safety-{len(list(TMP.glob('safety-*')))}"
        self.plan = make_plan(TMP / f"plan-{self.out.name}.json", "--cases", "op_add_fp16")

    def edited(self, **changes):
        doc = json.loads(self.plan.read_text())
        doc["cases"][0].update(changes)
        p = TMP / f"edited-{self.out.name}.json"
        p.write_text(json.dumps(doc))
        return p

    def test_a_second_run_in_the_same_out_does_not_see_the_first(self):
        run_main(self.plan, self.out)
        case = self.out / manifest(self.out)["cases"][0]["dir"]
        self.assertTrue(list((case / "raw").iterdir()))
        quiet = self.edited(command=["{python}", "-c", "pass"], repeats=1, allow_empty=True)
        code, _, _ = run_main(quiet, self.out)
        self.assertEqual(code, 0)
        self.assertEqual(list((case / "raw").iterdir()), [])
        self.assertEqual([p.name for p in case.glob("repeat-*")], ["repeat-001"])
        self.assertFalse((case / "repeat-001" / "metrics.json").exists())
        res = json.loads((case / "result.json").read_text())
        self.assertEqual(res["status"]["repeats"][0]["samples"], {})

    def test_a_command_that_writes_nothing_is_not_ingested(self):
        stub = TMP / "never_called.py"
        stub.write_text("import sys\nsys.exit(9)\n")
        quiet = self.edited(command=["{python}", "-c", "pass"])
        code, _, _ = run_main(quiet, self.out, "--ingest", "--ingest-cmd", f"{sys.executable} {stub}")
        self.assertEqual(code, 1)
        m = manifest(self.out)
        self.assertEqual((m["cases"][0]["status"], m["cases"][0]["reason"]), ("failed", "no_output"))
        self.assertEqual(m["ingest"]["invocations"], [])

    def test_zero_or_bad_repeats_are_refused(self):
        code, _, err = plan_main("--purpose", "nightly", "--set", "run.repeats=0")
        self.assertEqual(code, 2)
        for bad in (0, -1, "3", None, True):
            with self.subTest(repeats=bad):
                code, _, err = run_main(self.edited(repeats=bad), self.out)
                self.assertEqual(code, 2, err)
                self.assertIn("repeats", err)

    def test_malformed_plan_records_exit_2(self):
        for changes in (
            {"action": "maybe"},
            {"benchmark_id": ""},
            {"cache": "hot"},
            {"command": "echo hi"},
            {"command": []},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(run_main(self.edited(**changes), self.out)[0], 2)
        doc = json.loads(self.plan.read_text())
        del doc["producer"]
        p = TMP / "noproducer.json"
        p.write_text(json.dumps(doc))
        self.assertEqual(run_main(p, self.out)[0], 2)

    def test_warehouse_credentials_stay_out_of_benchmark_commands(self):
        script = TMP / "dump_env.py"
        script.write_text(
            "import json, os, sys\nopen(sys.argv[1] + '/seen.json', 'w').write(json.dumps(sorted(k for k in os.environ if k.startswith('CLICKHOUSE_'))))\n"
        )
        plan = self.edited(command=["{python}", str(script), "{repeat_dir}"], repeats=1)
        with mock.patch.dict(os.environ, {"CLICKHOUSE_PASS": "secret", "CLICKHOUSE_HOST": "h"}):
            run_main(plan, self.out)
        seen = next(self.out.glob("cases/*/repeat-001/seen.json"))
        self.assertEqual(json.loads(seen.read_text()), [])

    def test_interrupt_kills_the_running_command(self):
        spawned = []

        def popen(*a, **kw):
            spawned.append(subprocess_popen()(*a, **kw))
            return spawned[-1]

        out = TMP / "interrupt"
        out.mkdir(exist_ok=True)
        with mock.patch("spyre_bench.placement.verify_pinning", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_cmd.execute_command(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    os.environ.copy(),
                    str(out),
                    out / "o",
                    out / "e",
                    60,
                    {"policy": "device_local"},
                    "0",
                    popen,
                )
        self.assertIsNotNone(spawned[0].poll())

    def test_stragglers_of_a_finished_command_are_killed(self):
        out = TMP / "straggler"
        out.mkdir(exist_ok=True)
        pidfile = out / "pid"
        code = f"import subprocess, sys; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); open({str(pidfile)!r}, 'w').write(str(p.pid))"
        rc, _, _ = run_cmd.execute_command(
            [sys.executable, "-c", code],
            os.environ.copy(),
            str(out),
            out / "o",
            out / "e",
            30,
            None,
            "",
            subprocess_popen(),
        )
        self.assertEqual(rc, 0)
        pid = int(pidfile.read_text())
        for _ in range(50):
            if not still_running(pid):
                break
            time.sleep(0.1)
        else:
            self.fail("the straggler is still alive")

    def test_an_aborted_run_leaves_an_incomplete_manifest_not_the_previous_one(self):
        run_main(self.plan, self.out)
        self.assertEqual(manifest(self.out)["exit_reason"], "ok")

        def boom(*a, **kw):
            raise KeyboardInterrupt

        code, _, err = run_main(self.plan, self.out, popen=boom)
        self.assertEqual(code, 130)
        self.assertNotIn("Traceback", err)
        m = manifest(self.out)
        self.assertEqual((m["exit_reason"], m["exit_code"], m["ended_at"]), ("incomplete", 1, None))

    def test_ingest_logs_of_an_earlier_run_are_removed(self):
        stale = self.out / "ingest" / "ingest-9.log"
        stale.parent.mkdir(parents=True)
        stale.write_text("old")
        run_main(self.plan, self.out)
        self.assertFalse(stale.exists())

    def test_huge_integers_in_the_sidecar_cannot_crash_the_run(self):
        script = TMP / "huge.py"
        script.write_text(
            "import sys\nopen(sys.argv[1] + '/metrics.json', 'w').write('{\"samples\": {\"m\": [1' + '0' * 400 + ']}}')\n"
        )
        plan = self.edited(command=["{python}", str(script), "{repeat_dir}"], repeats=1)
        code, _, _ = run_main(plan, self.out)
        self.assertEqual(code, 0)
        case = manifest(self.out)["cases"][0]
        res = json.loads((self.out / case["result"]).read_text())
        self.assertEqual(res["status"]["repeats"][0]["samples"], {})
        st = json.loads((self.out / case["dir"] / "repeat-001/status.json").read_text())
        self.assertIn("dropped", st["notes"][0])

    def test_numbers_are_validated_before_anything_runs(self):
        marker = TMP / "ingest-ran"
        marker.unlink(missing_ok=True)
        cmd = f"{sys.executable} -c \"open({str(marker)!r}, 'w')\""
        for flag, value in (
            ("--landed-wait", "abc"),
            ("--landed-wait", "-1"),
            ("--landed-wait", "nan"),
            ("--ingest-timeout", "0"),
        ):
            with self.subTest(flag=flag, value=value):
                out = TMP / "numbers-out"
                shutil.rmtree(out, ignore_errors=True)
                code, _, err = run_main(
                    self.plan,
                    out,
                    "--ingest",
                    "--ingest-cmd",
                    cmd,
                    "--store",
                    "clickhouse",
                    flag,
                    value,
                    st=stores.MemoryStore(),
                )
                self.assertEqual(code, 2, err)
                self.assertFalse((out / "cases").exists())
        self.assertFalse(marker.exists())

    def test_an_interrupt_during_the_ingest_kills_it(self):
        killed = []

        class Hung:
            pid = 2**22 + 1  # never a live group; a stray killpg must not reach pid 1 (init) or the runner
            stdout = stderr = None

            def wait(self, timeout=None):
                raise KeyboardInterrupt

        class Pump:
            last_json = None

            def __init__(self, *a, **k):
                self.stop = threading.Event()

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        with (
            mock.patch("subprocess.Popen", return_value=Hung()),
            mock.patch.object(run_cmd, "kill_group", killed.append),
            mock.patch.object(run_cmd, "reap_group", lambda proc: None),
            mock.patch.object(run_cmd, "Capture", Pump),
            mock.patch("os.killpg", side_effect=AssertionError("the test must not signal a process group")),
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_cmd.run_ingest(
                    [{"argv": ["x"], "cases": ["c"]}],
                    {
                        "out": str(TMP / "ing-int"),
                        "ingest_timeout": 5,
                        "run_id": RUN,
                    },
                )
        self.assertEqual(len(killed), 1)

    def test_a_result_that_cannot_be_built_fails_the_run(self):
        with mock.patch.object(run_cmd, "build_case_result", side_effect=KeyError("env")):
            code, _, _ = run_main(self.plan, self.out)
        self.assertEqual(code, 1)
        m = manifest(self.out)
        self.assertEqual(m["exit_reason"], "result_invalid")
        self.assertIn("result not built", m["result_checks"]["op_add_fp16"][0])

    def test_dry_run_of_an_all_reuse_plan_matches_the_real_run(self):
        doc = json.loads(self.plan.read_text())
        doc["cases"][0].update(action="reuse", reused_from="r", reason="same_subject+same_environment")
        p = TMP / "allreuse-dry.json"
        p.write_text(json.dumps(doc))
        self.assertEqual(run_main(p, TMP / "dry4", "--dry-run")[0], 0)
        self.assertEqual(run_main(p, self.out)[0], 0)

    def test_the_sigterm_handler_is_restored(self):
        import signal

        before = signal.getsignal(signal.SIGTERM)
        run_main(self.plan, self.out)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_bad_numbers_are_usage_errors(self):
        self.assertEqual(run_main(self.plan, self.out, "--timeout", "0")[0], 2)
        self.assertEqual(run_main(self.plan, self.out, "--timeout", "soon")[0], 2)
        self.assertEqual(
            run_main(
                self.plan,
                self.out,
                "--landed-wait",
                "abc",
                "--ingest",
                "--ingest-cmd",
                "true",
                "--store",
                "clickhouse",
                st=stores.MemoryStore(),
            )[0],
            2,
        )
        self.assertEqual(plan_main("--purpose", "nightly", "--nodes", "0")[0], 2)

    def test_dry_run_flags_a_case_without_a_command(self):
        plan = self.edited(command=None)
        self.assertEqual(run_main(plan, TMP / "dry3", "--dry-run")[0], 1)

    def test_unreadable_request_or_env_file_exits_2(self):
        bad = TMP / "bad-req.yaml"
        bad.write_text("purpose: [unclosed")
        self.assertEqual(plan_main("--request", str(bad))[0], 2)
        self.assertEqual(plan_main("--request", str(TMP / "absent.yaml"))[0], 2)
        self.assertEqual(run_main(self.plan, self.out, "--env-file", str(bad))[0], 2)

    def test_result_problems_fail_the_run(self):
        with mock.patch.object(run_cmd, "check_result", return_value=["benchmark_id"]):
            code, _, _ = run_main(self.plan, self.out)
        self.assertEqual(code, 1)
        self.assertEqual(manifest(self.out)["exit_reason"], "result_invalid")


class IngestHandoffTests(unittest.TestCase):
    """The ingest is mocked at the process boundary: a stub command records its argv and exits as told."""

    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.out = TMP / f"ingest-{len(list(TMP.glob('ingest-*')))}"
        self.plan = make_plan(TMP / f"plan-{self.out.name}.json")
        self.log = TMP / f"{self.out.name}.argv"
        stub = TMP / "fake_ingest.py"
        stub.write_text(
            "import json, os, sys\n"
            "open(os.environ['FAKE_INGEST_LOG'], 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "sys.exit(int(os.environ.get('FAKE_INGEST_EXIT', '0')))\n"
        )
        self.cmd = f"{sys.executable} {stub} --run-id {{run_id}} --artifact-id {{subject_id}} --arch {{arch}} --raw {{raw_dir}} --case {{case}} --profile {{profile}}"
        self.env = mock.patch.dict(
            os.environ, {"FAKE_INGEST_LOG": str(self.log), "BENCH_INGEST_PASS_ENV": "FAKE_INGEST_LOG,FAKE_INGEST_EXIT"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def calls(self):
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_ingest_gets_the_run_id_subject_and_raw_dir(self):
        code, _, err = run_main(self.plan, self.out, "--run-id", RUN, "--ingest", "--ingest-cmd", self.cmd)
        self.assertEqual(code, 0, err)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        first = calls[0]
        self.assertEqual(first[first.index("--run-id") + 1], RUN)
        self.assertEqual(first[first.index("--artifact-id") + 1], SUBJECT)
        self.assertEqual(first[first.index("--arch") + 1], "x86_64")
        self.assertTrue(first[first.index("--raw") + 1].endswith("/raw"))
        self.assertEqual(first[first.index("--profile") + 1], "generic")
        m = manifest(self.out)
        self.assertEqual(m["ingest"]["status"], "ok")
        self.assertEqual(m["ingest"]["invocations"][0]["returncode"], 0)
        self.assertTrue((self.out / m["ingest"]["invocations"][0]["log"]).is_file())

    def test_identical_argv_runs_once(self):
        cmd = f"{sys.executable} {TMP / 'fake_ingest.py'} --run-id {{run_id}}"
        run_main(self.plan, self.out, "--run-id", RUN, "--ingest", "--ingest-cmd", cmd)
        self.assertEqual(len(self.calls()), 1)

    def test_only_ok_cases_are_ingested(self):
        with mock.patch.dict(os.environ, {"STUB_FAIL": "add"}):
            code, _, _ = run_main(self.plan, self.out, "--ingest", "--ingest-cmd", self.cmd, "--pass-env", "STUB_FAIL")
        self.assertEqual(code, 1)
        cases = [c[c.index("--case") + 1] for c in self.calls()]
        self.assertEqual(cases, ["op_mul_fp16"])

    def test_ingest_failure_exits_5(self):
        with mock.patch.dict(os.environ, {"FAKE_INGEST_EXIT": "7"}):
            code, _, _ = run_main(self.plan, self.out, "--ingest", "--ingest-cmd", self.cmd)
        self.assertEqual(code, 5)
        m = manifest(self.out)
        self.assertEqual((m["exit_reason"], m["ingest"]["status"]), ("ingest_failed", "failed"))
        self.assertEqual(m["rows_landed"]["status"], "unchecked")

    def test_missing_ingest_executable_exits_5(self):
        code, _, _ = run_main(self.plan, self.out, "--ingest", "--ingest-cmd", "/no/such/ingest {run_id}")
        self.assertEqual(code, 5)

    def test_ingest_needs_a_subject(self):
        plan = make_plan(TMP / "nosubject.json", subject=None)
        code, _, err = run_main(plan, self.out, "--ingest", "--ingest-cmd", self.cmd)
        self.assertEqual(code, 2)
        self.assertIn("subject", err)

    def test_default_commands_per_profile(self):
        recs = {
            "perf-kernel": ("--xml-dir", "--trigger-type"),
            "vllm": ("--vllm-results-dir", None),
            "vllm-sweep": ("--vllm-results-dir", None),
            "generic": ("--benchmarks-dir", None),
            None: ("--benchmarks-dir", None),
        }
        ctx = ingest_ctx()
        for profile, (flag, extra) in recs.items():
            with self.subTest(profile=profile):
                rec = {"name": "c", "id_profile": profile, "ingest": None}
                outs = [{"name": "c", "status": "ok", "dir": "cases/c"}]
                (cmd,) = run_cmd.ingest_commands(outs, {"c": rec}, ctx)
                argv = cmd["argv"]
                self.assertEqual(argv[:5], [sys.executable, "-I", "-m", "spyre_clickhouse_ingest", "results"])
                self.assertIn(flag, argv)
                if profile == "perf-kernel":
                    self.assertEqual(argv[argv.index("--run-id") + 1], RUN)
                else:
                    self.assertNotIn("--run-id", argv)
                    self.assertRegex(
                        argv[argv.index("--jenkins-run-key") + 1], rf"^manual:spyre-bench:{RUN}-[0-9a-f]{{16}}$"
                    )
                    if profile in ("generic", None):
                        self.assertEqual(argv[argv.index("--component") + 1], "run-demo")
                    else:
                        self.assertNotIn("--component", argv)
                    self.assertEqual(argv[argv.index("--trigger-type") + 1], "perf")
                self.assertEqual(argv[argv.index("--artifact-id") + 1], SUBJECT)
                self.assertEqual(argv[argv.index(flag) + 1], "/o/cases/c/raw")
                if extra:
                    self.assertIn(extra, argv)
                self.assertEqual(argv[-2:], ["--schema", "v2"])

    def test_case_ingest_override_and_cli_override_order(self):
        ctx = ingest_ctx()
        rec = {"name": "c", "id_profile": "vllm", "ingest": ["tool", "{case}", "{profile}"]}
        outs = [{"name": "c", "status": "ok", "dir": "cases/c"}]
        self.assertEqual(run_cmd.ingest_commands(outs, {"c": rec}, ctx)[0]["argv"], ["tool", "c", "vllm"])
        ctx["ingest_cmd"] = "other {case}"
        self.assertEqual(run_cmd.ingest_commands(outs, {"c": rec}, ctx)[0]["argv"], ["other", "c"])

    def test_unknown_ingest_token_exits_2(self):
        code, _, err = run_main(self.plan, self.out, "--ingest", "--ingest-cmd", "tool {nope}")
        self.assertEqual(code, 2)
        self.assertIn("{nope}", err)

    def test_no_ingest_wins_over_the_env_twin(self):
        with mock.patch.dict(os.environ, {"BENCH_INGEST": "1"}):
            code, _, _ = run_main(self.plan, self.out, "--no-ingest", "--ingest-cmd", self.cmd)
            self.assertEqual((code, self.calls()), (0, []))
            run_main(self.plan, self.out, "--ingest-cmd", self.cmd)
            self.assertEqual(len(self.calls()), 2)

    def test_no_ingest_mode_never_touches_the_store_or_ingest(self):
        st = stores.MemoryStore()
        with mock.patch("subprocess.run") as runner:
            code, _, _ = run_main(self.plan, self.out, "--store", "none", st=st)
        self.assertEqual(code, 0)
        runner.assert_not_called()
        self.assertEqual(st.calls, [])


def sibling_results_parser():
    """The ingest's real `results` argparse parser, or a skip reason."""
    sys.path.insert(0, str(EXT.parent / "clickhouse-ingest"))
    try:
        from spyre_clickhouse_ingest import results

        return results.build_parser(), None
    except Exception as x:  # noqa: BLE001 - optional sibling package
        return None, f"spyre_clickhouse_ingest results is not importable: {x}"
    finally:
        sys.path.remove(str(EXT.parent / "clickhouse-ingest"))


def ingest_demo_copy(label):
    d = TMP / label
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(X / "ingest-demo", d)
    p = d / "ops" / "op-softmax.yaml"
    p.write_text(
        p.read_text().replace(
            "run: {repeats: 1, timeout: 30m, est: 2m, isolation: process}",
            "run: {repeats: 1, timeout: 30m, est: 2m, isolation: process, allow_empty: true, command: ['{python}', '-c', 'print(1)']}",
        )
    )
    return d


class IngestModeTests(unittest.TestCase):
    ARGS = [
        "--central",
        str(CENTRAL),
        "--arch",
        "x86_64",
        "--reference-node",
        "--purpose",
        "nightly",
        "--subject-id",
        SUBJECT,
        "--cases",
        "op_softmax_fp16",
    ]

    def test_plan_needs_the_ingest_and_says_so(self):
        d = ingest_demo_copy("ingest-mode-missing")
        with without_ingest(), clean_env():
            code, _, err = capture(plan_cmd.main, ["--catalog", str(d), *self.ARGS])
        self.assertEqual(code, 2)
        self.assertIn("ingest", err)

    def test_ids_and_profile_come_from_the_ingest_derivation(self):
        identity, why = sibling_ingest()
        if identity is None:
            self.skipTest(why)
        d = ingest_demo_copy("ingest-mode")
        ingest_adapter.reset()  # an earlier plan may have cached "ingest unavailable" before the sibling was importable
        self.addCleanup(ingest_adapter.reset)
        with clean_env():
            code, out, err = capture(plan_cmd.main, ["--catalog", str(d), "--json", *self.ARGS])
        self.assertEqual(code, 0, err)
        rec = json.loads(out)["cases"][0]
        cat = engine.Catalog([d], [], [CENTRAL])
        case = next(c for c in cat.cases if c["name"] == "op_softmax_fp16")
        self.assertEqual((rec["id_mode"], rec["id_profile"]), ("ingest", "perf-kernel"))
        self.assertEqual(
            (rec["benchmark_id"], rec["engine_id"]), (cat.ids(case)["benchmark_id"], cat.ids(case)["engine_id"])
        )
        self.assertNotEqual(rec["benchmark_id"], rec["engine_id"])
        out_dir = TMP / "ingest-mode-out"
        plan = TMP / "ingest-mode-plan.json"
        plan.write_text(out)
        with clean_env():
            code, _, err = capture(
                run_cmd.main,
                [
                    "--plan",
                    str(plan),
                    "--out",
                    str(out_dir),
                    "--catalog",
                    str(d),
                    "--central",
                    str(CENTRAL),
                    "--workdir",
                    str(d),
                ],
            )
        self.assertEqual(code, 0, err)
        m = manifest(out_dir)
        self.assertEqual(m["cases"][0]["id_mode"], "ingest")
        res = json.loads((out_dir / m["cases"][0]["result"]).read_text())
        self.assertEqual(res["status"]["case"]["id_mode"], "ingest")
        self.assertEqual(engine.recompute(res), [])


class IngestFlagTests(unittest.TestCase):
    def test_default_argv_is_accepted_by_the_real_ingest_parser(self):
        parser, why = sibling_results_parser()
        if parser is None:
            self.skipTest(why)
        ctx = ingest_ctx()
        for profile in run_cmd.INGEST_DEFAULTS:
            with self.subTest(profile=profile):
                (cmd,) = run_cmd.ingest_commands(
                    [{"name": "c", "status": "ok", "dir": "cases/c"}],
                    {"c": {"name": "c", "id_profile": profile, "ingest": None}},
                    ctx,
                )
                args = parser.parse_args(cmd["argv"][5:])
                self.assertEqual((args.artifact_id, args.arch, args.schema), (SUBJECT, "x86_64", "v2"))
                self.assertEqual(args.run_id, RUN if profile == "perf-kernel" else "")
                self.assertEqual(
                    args.jenkins_run_key.startswith(f"manual:spyre-bench:{RUN}-"), profile != "perf-kernel"
                )


class PinningTests(unittest.TestCase):
    def test_worst_sample_wins(self):
        s = [{"run.exec.pinning_verified": v} for v in ("true", "unknown", "false", "true")]
        self.assertEqual(run_cmd.worst_pinning(s)["run.exec.pinning_verified"], "false")
        self.assertEqual(run_cmd.worst_pinning(s[:2])["run.exec.pinning_verified"], "unknown")
        self.assertIsNone(run_cmd.worst_pinning([]))

    def test_sampling_follows_the_requested_policy(self):
        out = TMP / "pin"
        out.mkdir(exist_ok=True)
        seen = []

        def fake_verify(request, pid, device_numa, **_):
            seen.append((request["policy"], device_numa))
            return {"run.exec.pinning_verified": "true", "run.exec.cpu_pinning": request["policy"]}

        with mock.patch("spyre_bench.placement.verify_pinning", fake_verify):
            rc, reason, pin = run_cmd.execute_command(
                [sys.executable, "-c", "pass"],
                os.environ.copy(),
                str(out),
                out / "o",
                out / "e",
                30,
                {"policy": "device_local"},
                "0",
                subprocess_popen(),
            )
        self.assertEqual((rc, reason, pin["run.exec.pinning_verified"]), (0, "", "true"))
        self.assertEqual(seen[0], ("device_local", "0"))
        rc, reason, pin = run_cmd.execute_command(
            [sys.executable, "-c", "pass"],
            os.environ.copy(),
            str(out),
            out / "o",
            out / "e",
            30,
            None,
            "",
            subprocess_popen(),
        )
        self.assertEqual((rc, pin), (0, None))


def subprocess_popen():
    import subprocess

    return subprocess.Popen


class RowsLandedTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env(FAKE_INGEST_LOG=str(TMP / "landed.log"))
        self._env.start()
        self.addCleanup(self._env.stop)
        self.out = TMP / f"landed-{len(list(TMP.glob('landed-*')))}"
        self.plan = make_plan(TMP / f"plan-{self.out.name}.json")
        stub = TMP / "ok_ingest.py"
        stub.write_text("import sys\nsys.exit(0)\n")
        self.cmd = f"{sys.executable} {stub} {{run_id}}"
        self.ids = [c["benchmark_id"] for c in json.loads(self.plan.read_text())["cases"]]

    def run_with(self, st, *extra, out=None):
        return run_main(
            self.plan,
            out or self.out,
            "--run-id",
            RUN,
            "--ingest",
            "--ingest-cmd",
            self.cmd,
            "--store",
            "clickhouse",
            "--landed-wait",
            "0",
            *extra,
            st=st,
        )

    def test_rows_present_is_ok(self):
        code, _, _ = self.run_with(stores.MemoryStore(landed={RUN: set(self.ids)}))
        self.assertEqual(code, 0)
        self.assertEqual(manifest(self.out)["rows_landed"]["status"], "ok")

    def test_no_rows_at_all_exits_1(self):
        code, out, _ = self.run_with(stores.MemoryStore(landed={}))
        self.assertEqual(code, 1)
        m = manifest(self.out)
        self.assertEqual((m["exit_reason"], m["rows_landed"]["status"]), ("no_rows_landed", "no_rows_landed"))
        self.assertIn("no_rows_landed", out)

    def test_engine_mode_cases_need_a_row_each(self):
        code, _, _ = self.run_with(stores.MemoryStore(landed={RUN: {"a", "b"}}))
        self.assertEqual(code, 0)
        other = TMP / f"{self.out.name}-partial"
        code, _, _ = self.run_with(stores.MemoryStore(landed={RUN: {"anything"}}), out=other)
        self.assertEqual(code, 1)  # a partial ingest: one row for two cases
        self.assertIn("too few", manifest(other)["rows_landed"]["reason"])

    def test_rows_landing_late_are_waited_for(self):
        seen = []

        def late():
            seen.append(1)
            return set(self.ids) if len(seen) > 2 else set()

        slept = []
        got = run_cmd.check_landed(
            stores.MemoryStore(landed={RUN: late}),
            {RUN: [{"status": "ok", "id_mode": "engine", "benchmark_id": "x"}]},
            60,
            sleep=slept.append,
            clock=iter(range(0, 1000, 1)).__next__,
        )
        self.assertEqual((got["status"], len(slept)), ("ok", 2))

    def test_ingest_mode_ids_must_all_land(self):
        outs = [
            {"status": "ok", "id_mode": "ingest", "benchmark_id": "a"},
            {"status": "ok", "id_mode": "ingest", "benchmark_id": "b"},
            {"status": "failed", "id_mode": "ingest", "benchmark_id": "c"},
        ]
        got = run_cmd.check_landed(stores.MemoryStore(landed={RUN: {"a"}}), {RUN: outs}, 0)
        self.assertEqual((got["status"], got["expected"], got["missing"]), ("no_rows_landed", ["a", "b"], ["b"]))
        self.assertEqual(
            run_cmd.check_landed(stores.MemoryStore(landed={RUN: {"a", "b", "z"}}), {RUN: outs}, 0)["status"], "ok"
        )

    def test_store_failure_exits_5(self):
        code, _, _ = self.run_with(stores.MemoryStore(fail="down"))
        self.assertEqual(code, 5)
        self.assertEqual(manifest(self.out)["rows_landed"]["status"], "store_error")

    def test_no_store_means_unchecked_not_passed(self):
        code, _, _ = run_main(self.plan, self.out, "--ingest", "--ingest-cmd", self.cmd)
        self.assertEqual(code, 0)
        rl = manifest(self.out)["rows_landed"]
        self.assertEqual((rl["status"], rl["reason"]), ("unchecked", "no store configured"))

    def test_exit_precedence(self):
        base = {
            "counts": {"failed": 0, "run": 1, "reuse": 0, "skip": 0},
            "ingest": {"status": "ok"},
            "rows_landed": {"status": "ok"},
            "result_checks": {},
        }

        def of(**kw):
            m = copy.deepcopy(base)
            for k, v in kw.items():
                m[k].update(v)
            return run_cmd.exit_of(m)

        self.assertEqual(of(), (0, "ok"))
        self.assertEqual(of(counts={"failed": 1}, rows_landed={"status": "no_rows_landed"}), (1, "cases_failed"))
        self.assertEqual(of(ingest={"status": "failed"}, counts={"failed": 1}), (5, "ingest_failed"))
        self.assertEqual(of(counts={"run": 0, "skip": 1}), (4, "nothing_ran"))
        self.assertEqual(of(counts={"run": 0, "reuse": 2}), (0, "ok"))
        self.assertEqual(of(rows_landed={"status": "store_error"}), (5, "store_error"))
        self.assertEqual(of(result_checks={"c": ["benchmark_id"]}), (1, "result_invalid"))


def verdict_row(**kw):
    row = {
        "run_id": RUN, "benchmark_id": SUBJECT, "name": "op_add", "component": "demo", "backend": "spyre", "arch": "x86_64",
        "metric": "kernel_mean_ms", "tier": "blocking", "status": "unchanged", "reason": "", "baseline_kind": "rolling",
        "n_baseline": 20, "n_current": 3, "baseline_value": 1.0, "current_value": 1.01, "delta_pct": 1.0, "worse_pct": 1.0, "floor_pct": 3.0,
    }  # fmt: skip
    return {**row, **kw}


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)

    def store(self, *rows):
        return stores.MemoryStore(verdicts={RUN: list(rows)}, runs={SUBJECT: RUN})

    def call(self, *argv, rows=(), st=None):
        return capture(verdict_cmd.main, list(argv), st=st or self.store(*rows))

    def test_statuses_are_mapped_and_summarised(self):
        rows = [
            verdict_row(),
            verdict_row(name="op_b", status="regressed", worse_pct=9.0),
            verdict_row(name="op_c", status="improved"),
            verdict_row(name="op_d", status="suspected", reason="single_run"),
            verdict_row(name="op_e", status="no_baseline", n_baseline=2),
        ]
        code, out, _ = self.call("--run-id", RUN, "--json", rows=rows)
        doc = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(doc["verdict_schema"], "bench.verdict/1")
        self.assertEqual(
            doc["summary"],
            {
                "total": 5,
                "pass": 1,
                "regressed": 1,
                "improved": 1,
                "suspected": 1,
                "no_baseline": 1,
                "worst": "regressed",
            },
        )
        self.assertEqual(
            [v["status"] for v in doc["verdicts"]], ["pass", "regressed", "improved", "suspected", "no_baseline"]
        )
        self.assertEqual({b["name"]: b["status"] for b in doc["benchmarks"]}["op_b"], "regressed")
        self.assertEqual(set(doc["verdicts"][0]), set(verdict_cmd.FIELDS) | {"anchored"})

    def test_per_benchmark_status_is_the_worst_metric(self):
        rows = [verdict_row(metric="m1"), verdict_row(metric="m2", status="suspected")]
        doc = json.loads(self.call("--run-id", RUN, "--json", rows=rows)[1])
        self.assertEqual(
            doc["benchmarks"], [{"benchmark_id": SUBJECT, "name": "op_add", "status": "suspected", "metrics": 2}]
        )

    def test_exit_0_unless_fail_on(self):
        rows = [verdict_row(status="regressed")]
        self.assertEqual(self.call("--run-id", RUN, rows=rows)[0], 0)
        self.assertEqual(self.call("--run-id", RUN, "--fail-on", "regressed", rows=rows)[0], 1)
        self.assertEqual(self.call("--run-id", RUN, "--fail-on", "suspected", rows=rows)[0], 0)
        self.assertEqual(
            self.call("--run-id", RUN, "--fail-on", "regressed,suspected", rows=[verdict_row(status="suspected")])[0], 1
        )
        self.assertEqual(self.call("--run-id", RUN, "--fail-on", "nope", rows=rows)[0], 2)

    def test_a_run_without_rows_is_zero_total_exit_0(self):
        code, out, _ = self.call("--run-id", RUN, "--json")
        doc = json.loads(out)
        self.assertEqual((code, doc["summary"]["total"], doc["summary"]["worst"], doc["verdicts"]), (0, 0, None, []))

    def test_artifact_resolves_to_its_newest_run(self):
        code, out, _ = self.call("--artifact", SUBJECT, "--json", rows=[verdict_row()])
        doc = json.loads(out)
        self.assertEqual((doc["run_id"], doc["artifact_id"]), (RUN, SUBJECT))
        code, _, err = self.call("--artifact", OTHER, rows=[])
        self.assertEqual(code, 2)
        self.assertIn("no run", err)

    def test_quiet_drops_values(self):
        doc = json.loads(self.call("--run-id", RUN, "--json", "--quiet", rows=[verdict_row()])[1])
        self.assertTrue(set(verdict_cmd.VALUE_FIELDS).isdisjoint(doc["verdicts"][0]))
        with mock.patch.dict(os.environ, {"BENCH_QUIET": "1"}):
            doc = json.loads(self.call("--run-id", RUN, "--json", rows=[verdict_row()])[1])
        self.assertNotIn("current_value", doc["verdicts"][0])

    def test_text_output(self):
        code, out, _ = self.call("--run-id", RUN, rows=[verdict_row(status="regressed")])
        self.assertIn("1 regressed", out)
        self.assertIn("regressed", out.splitlines()[1])

    def test_store_failure_is_never_a_pass(self):
        code, _, err = self.call("--run-id", RUN, st=stores.MemoryStore(fail="down"))
        self.assertEqual(code, 2)
        self.assertIn("down", err)

    def test_unknown_status_from_the_view_is_an_error(self):
        code, _, err = self.call("--run-id", RUN, rows=[verdict_row(status="odd")])
        self.assertEqual(code, 2)
        self.assertIn("odd", err)

    def test_selector_and_store_validation(self):
        self.assertEqual(self.call()[0], 2)
        self.assertEqual(self.call("--run-id", RUN, "--store", "none")[0], 2)
        with mock.patch.dict(os.environ, {"BENCH_RUN_ID": RUN}):
            self.assertEqual(self.call(rows=[verdict_row()])[0], 0)

    def test_dispatch_through_the_cli(self):
        out = io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            mock.patch("spyre_bench.store.open_store", return_value=self.store(verdict_row())),
        ):
            code = cli_main(["verdict", "--run-id", RUN, "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["summary"]["total"], 1)

    def test_view_status_words_are_the_engines(self):
        self.assertEqual(stores.STATUS_OF_VIEW, {"unchanged": "pass"})
        self.assertEqual(set(verdict_cmd.SEVERITY), set(engine.VERDICTS))


class Verbs(unittest.TestCase):
    def test_help_lists_the_new_verbs(self):
        code, out, _ = capture(cli_main, ["--help"])
        self.assertEqual(code, 0)
        for verb in ("plan", "run", "verdict"):
            self.assertIn(f"  {verb} ", out)

    def test_cliutil_helpers(self):
        self.assertEqual(cliutil.csv(["a,b", "c d"]), ["a", "b", "c", "d"])
        self.assertEqual(cliutil.csv(None), [])
        with clean_env(BENCH_PACK=f"x{os.pathsep}y"):
            self.assertEqual(cliutil.paths([], "BENCH_PACK"), ["x", "y"])
            self.assertEqual(cliutil.paths(["z"], "BENCH_PACK"), ["z"])


class CacheBothStatsTests(unittest.TestCase):
    def reps(self):
        def rep(cache, v, warm):
            return {"status": "ok", "samples": {"m": [v]}, "props": {"run.cache": cache}, "warmup_discarded": warm}

        return [rep("cold", 100.0, 0), rep("cold", 140.0, 0), rep("warm", 10.0, 2), rep("warm", 10.0, 1)]

    def test_cv_and_aggregate_ignore_cold_when_warm_exists(self):
        from spyre_bench import engine, run

        reps = self.reps()
        self.assertEqual(run.cv_pct(reps), 0.0)
        self.assertEqual(engine.aggregate(reps, {})["metrics"]["m"]["value"], 10.0)

    def test_cold_only_metric_still_aggregates(self):
        from spyre_bench import engine

        cold = [r for r in self.reps() if r["props"]["run.cache"] == "cold"]
        self.assertEqual(engine.aggregate(cold, {})["metrics"]["m"]["value"], 120.0)


if __name__ == "__main__":
    unittest.main()

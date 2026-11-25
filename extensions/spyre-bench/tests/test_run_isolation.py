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

"""Ingest hand-off with several cases (through the real bundle path), ingest isolation, environment families,
redaction, run-id reuse, verdict run ids, and the small contracts the earlier suites left implicit.
python -m pytest -q -p no:cacheprovider tests/test_run_isolation.py
"""

import contextlib
import errno
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from spyre_bench import cliutil, planfile, store as stores
from spyre_bench import plan as plan_cmd
from spyre_bench import run as run_cmd
from spyre_bench import verdict as verdict_cmd
from test_run import (
    CENTRAL,
    DEMO,
    EXT,
    OTHER,
    RUN,
    SUBJECT,
    capture,
    clean_env,
    ingest_ctx,
    make_plan,
    manifest,
    plan_main,
    run_main,
    sibling_results_parser,
    still_running,
    verdict_row,
)

ITMP = Path(tempfile.mkdtemp(prefix="spyre-bench-isolation-"))
INGEST = ["--ingest", "--ingest-cmd", f"{sys.executable} -c pass"]


def tearDownModule():
    shutil.rmtree(ITMP, ignore_errors=True)


def fresh(label):
    d = ITMP / label
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def one_case_plan(label, **changes):
    """A plan of op_add_fp16 alone with `changes` applied to its record; written under this module's temp dir."""
    doc = json.loads(make_plan(ITMP / f"{label}-base.json", "--cases", "op_add_fp16").read_text())
    doc["cases"][0].update(changes)
    p = ITMP / f"{label}.json"
    p.write_text(json.dumps(doc))
    return p


def ok_case(name, profile=None):
    return {"name": name, "status": "ok", "dir": f"cases/{name}"}, {"name": name, "id_profile": profile, "ingest": None}


class MultiCaseBundleTests(unittest.TestCase):
    """Several ok cases of one run through the REAL offline.write / bundle ingest, with the warehouse faked."""

    @classmethod
    def setUpClass(cls):
        parser, why = sibling_results_parser()
        fixtures = EXT.parent / "clickhouse-ingest" / "tests" / "data" / "vllm_bundle" / "results"
        if parser is None or not fixtures.is_dir():
            raise unittest.SkipTest(why or "the ingest's vLLM fixtures are not alongside this package")
        cls.parser, cls.fixtures = parser, fixtures
        sys.path.insert(0, str(EXT.parent / "clickhouse-ingest"))
        try:
            from spyre_clickhouse_ingest import bundle, client, results
        finally:
            sys.path.remove(str(EXT.parent / "clickhouse-ingest"))
        cls.bundle, cls.client, cls.results = bundle, client, results

    def setUp(self):
        self.db = {}
        self._patches = []
        art = SimpleNamespace(artifact_id=SUBJECT, arch="x86_64", component="demo-comp")
        fake_client = SimpleNamespace(set_client_setting=lambda *a: None)
        env = mock.patch.dict(os.environ, {"CLICKHOUSE_DB_V2": "db"})
        env.start()
        self.addCleanup(env.stop)
        for obj, name, value in (
            (
                self.client,
                "ClickHouse",
                SimpleNamespace(connect=lambda database=None: fake_client, target_database=lambda: "db"),
            ),
            (self.bundle, "existing_run", lambda c, d, rid, comp: (set(self.db.get(rid, set())), set())),
            (self.bundle, "resolve_artifact", lambda c, d, meta: (art, "id")),
            (self.bundle, "write_generic", self.fake_write),
            (self.bundle, "write_vllm", self.fake_write),
            (self.bundle, "verdict_recorded", lambda *a, **k: True),
        ):
            p = mock.patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)

    def fake_write(self, c, d, root, meta, aid, rid, props):
        self.db.setdefault(rid, set()).add(props["bundle_sha256"])
        return 1

    def raw_for(self, profile, label):
        raw = fresh(f"raw-{label}")
        if profile == "generic":
            (raw / "ops.json").write_text(
                json.dumps({"benchmarks": [{"name": label, "measurements": {"kernel_mean_ms": [1.0, 1.1]}}]})
            )
        else:
            for f in self.fixtures.glob("*.json"):
                shutil.copy(f, raw)
        return raw

    def ingest_all(self, profiles):
        """Run the engine's default ingest argv of one case per profile through the real ingest; the reports."""
        outs = [ok_case(f"case-{n}", prof)[0] for n, prof in enumerate(profiles)]
        recs = {f"case-{n}": ok_case(f"case-{n}", prof)[1] for n, prof in enumerate(profiles)}
        cmds = run_cmd.ingest_commands(outs, recs, ingest_ctx(out=str(ITMP)))
        reports = []
        for n, cmd in enumerate(cmds):
            raw = self.raw_for(profiles[n], f"{id(self)}-{n}")
            argv = [str(raw) if a.endswith("/raw") else a for a in cmd["argv"]]
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = self.results.main(argv[5:])
            reports.append((rc, run_cmd.report_of(out.getvalue()), err.getvalue()))
        return cmds, reports

    def test_two_and_three_cases_per_bundle_profile_all_ingest_under_their_own_run_ids(self):
        for profile in ("generic", "vllm", "vllm-sweep"):
            for count in (2, 3):
                with self.subTest(profile=profile, cases=count):
                    self.db.clear()
                    cmds, reports = self.ingest_all([profile] * count)
                    self.assertEqual(len(cmds), count)  # one invocation per case, never deduplicated to one
                    keys = [c["argv"][c["argv"].index("--jenkins-run-key") + 1] for c in cmds]
                    self.assertEqual(len(set(keys)), count)
                    self.assertTrue(all(k.startswith(f"manual:spyre-bench:{RUN}-") for k in keys))
                    for rc, report, err in reports:
                        self.assertEqual(rc, 0, err)
                        self.assertTrue(run_cmd.accepted_report(report), report)
                    self.assertEqual(len({r[1]["run_id"] for r in reports}), count)

    def test_a_mixed_profile_plan_ingests_every_case(self):
        _, reports = self.ingest_all(["generic", "vllm", "generic", "vllm-sweep"])
        self.assertEqual([r[0] for r in reports], [0, 0, 0, 0])
        self.assertEqual(len({r[1]["run_id"] for r in reports}), 4)
        self.assertEqual(len(self.db), 4)

    def test_the_old_shared_key_is_what_the_ingest_refuses(self):
        # negative control: two bundles of different content under one key are rejected by the real ingest
        same = ingest_ctx(out=str(ITMP))
        outs, recs = zip(*[ok_case(f"c{n}", "generic") for n in range(2)])
        seen = []
        for n, (o, r) in enumerate(zip(outs, recs)):
            (cmd,) = run_cmd.ingest_commands([o], {o["name"]: r}, same)
            argv = [str(self.raw_for("generic", f"neg-{n}")) if a.endswith("/raw") else a for a in cmd["argv"]]
            key = argv.index("--jenkins-run-key") + 1
            argv[key] = f"manual:spyre-bench:{RUN}"  # the same key for both
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                seen.append(self.results.main(argv[5:]))
        self.assertEqual(seen[0], 0)
        self.assertNotEqual(seen[1], 0)

    def test_a_partial_failure_names_the_cases_that_landed(self):
        stub = ITMP / "half_ingest.py"
        stub.write_text(
            "import json, sys\ncase = sys.argv[1]\n"
            "if case == 'op_mul_fp16':\n    print(json.dumps({'status': 'rejected', 'reason': 'already recorded'}))\n    sys.exit(1)\n"
            "print(json.dumps({'run_id': '7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f', 'status': 'ingested'}))\n"
        )
        out = fresh("partial-out")
        code, _, _ = run_main(
            make_plan(ITMP / "partial.json"), out, "--ingest", "--ingest-cmd", f"{sys.executable} {stub} {{case}}"
        )
        self.assertEqual(code, 5)
        m = manifest(out)
        self.assertEqual(
            (m["ingest"]["cases_ingested"], m["ingest"]["cases_failed"]), (["op_add_fp16"], ["op_mul_fp16"])
        )
        self.assertEqual(m["stored_run_ids"], ["7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"])
        self.assertEqual(m["exit_reason"], "ingest_failed")

    def test_a_rejected_report_with_exit_0_is_a_failure(self):
        stub = ITMP / "rejecting.py"
        stub.write_text(
            "import json\nprint(json.dumps({'run_id': 'R', 'status': 'rejected', 'reason': 'bad bundle'}))\n"
        )
        out = fresh("reject-out")
        code, _, _ = run_main(
            make_plan(ITMP / "reject.json"), out, "--ingest", "--ingest-cmd", f"{sys.executable} {stub}"
        )
        self.assertEqual(code, 5)
        inv = manifest(out)["ingest"]["invocations"][0]
        self.assertEqual((inv["error"], inv["reason"]), ("report_rejected", "bad bundle"))

    def test_mixed_type_report_values_do_not_crash(self):
        stub = ITMP / "oddids.py"
        stub.write_text("import json\nprint(json.dumps({'run_id': 7, 'status': 'ingested'}))\n")
        out = fresh("odd-out")
        code, _, _ = run_main(make_plan(ITMP / "odd.json"), out, "--ingest", "--ingest-cmd", f"{sys.executable} {stub}")
        self.assertEqual(code, 5)


def recorder(path):
    """Source of a planted module: appends one JSON line saying which kind of process ran it and what secret it saw."""
    return (
        "import json, os, sys\n"
        f"open({str(path)!r}, 'a').write(json.dumps({{'isolated': sys.flags.isolated, 'argv0': sys.argv[:1], "
        "'secret': os.environ.get('CLICKHOUSE_PASS')}) + '\\n')\n"
    )


class IngestIsolationTests(unittest.TestCase):
    """A benchmark must not be able to run code inside the ingest, which holds the warehouse credentials."""

    def setUp(self):
        self.pkg = fresh("fakepkg")
        pkg = self.pkg / "spyre_clickhouse_ingest"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            "import argparse, json, sys\n"
            "print(json.dumps({'run_id': '7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f', 'status': 'ingested'}))\n"
        )
        self.leak = ITMP / "LEAK.txt"
        self.leak.unlink(missing_ok=True)
        self.wd = fresh("hostile-wd")
        self.home = fresh("hostile-home")
        content = recorder(self.leak)
        code = (
            f"import os, site\nHOSTILE = {content!r}\n"
            "for n in ('argparse.py', 'sitecustomize.py'):\n    open(n, 'w').write(HOSTILE)\n"
            "user = site.getusersitepackages()\nos.makedirs(user, exist_ok=True)\n"
            "open(os.path.join(user, 'usercustomize.py'), 'w').write(HOSTILE)\n"
            "os.makedirs(os.environ['BENCH_RAW_DIR'], exist_ok=True)\n"
            "open(os.environ['BENCH_RAW_DIR'] + '/r.json', 'w').write('{}')\n"
        )
        self.plan = one_case_plan("hostile", command=["{python}", "-c", code], repeats=1)
        # the caller's own PYTHONPATH points at the benchmark's working directory: a benchmark-written module there
        # must not reach the ingest
        self._env = clean_env(CLICKHOUSE_PASS="warehouse-secret", HOME=str(self.home), PYTHONPATH=str(self.wd))
        self._env.start()
        self.addCleanup(self._env.stop)

    def run_it(self, label, *extra, pythonpath=True):
        out = fresh(label)
        more = ["--ingest-pythonpath", str(self.pkg)] if pythonpath else []
        return run_main(self.plan, out, "--workdir", str(self.wd), "--ingest", *more, *extra), out

    def test_the_benchmark_did_write_hostile_files_where_a_naive_ingest_would_import_them(self):
        (code, _, _), _ = self.run_it("hostile-control", "--ingest-cmd", f"{sys.executable} -c pass", pythonpath=False)
        self.assertTrue((self.wd / "argparse.py").is_file() and (self.wd / "sitecustomize.py").is_file())
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(self.pkg), str(self.wd)])}
        subprocess.run([sys.executable, "-m", "spyre_clickhouse_ingest"], cwd=self.wd, env=env, capture_output=True)
        self.assertTrue(self.leak.exists(), "the control must show the attack works from the benchmark's cwd")
        self.leak.unlink()

    def test_the_default_ingest_does_not_import_them(self):
        profile = json.loads(self.plan.read_text())
        self.assertIsNone(profile["cases"][0]["ingest"])
        (code, _, err), out = self.run_it("hostile-default")
        self.assertEqual(code, 0, err)
        self.assertFalse(self.leak.exists())
        self.assertEqual(manifest(out)["stored_run_ids"], ["7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"])

    def test_a_custom_ingest_gets_the_same_treatment(self):
        cmd = f"{sys.executable} -m spyre_clickhouse_ingest"  # no -I: the environment and directory must cover it
        (code, _, err), out = self.run_it("hostile-custom", "--ingest-cmd", cmd)
        self.assertEqual(code, 0, err)
        self.assertFalse(self.leak.exists())

    def test_the_default_isolated_mode_ignores_the_user_site_and_the_callers_pythonpath(self):
        base = getattr(sys, "_base_executable", sys.executable)
        probe = subprocess.run(
            [base, "-c", "import site; print(site.ENABLE_USER_SITE)"],
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        if probe.stdout.strip() != "True":
            self.skipTest("the base interpreter has no user site here, so the attack cannot be reproduced")
        user = subprocess.run(
            [base, "-c", "import site; print(site.getusersitepackages())"], capture_output=True, text=True
        ).stdout.strip()
        Path(user).mkdir(parents=True, exist_ok=True)
        (Path(user) / "usercustomize.py").write_text(recorder(self.leak))
        subprocess.run([base, "-c", "pass"], env={**os.environ})
        control = self.records()
        self.assertEqual([r["secret"] for r in control], ["warehouse-secret"], "the control must show the attack works")
        self.leak.unlink()
        # The caller's HOME is handed to the ingest on purpose (--ingest-pass-env HOME), so the planted module is
        # reachable: only -I keeps it out. The ingest is driven directly with this interpreter, bypassing the
        # preflight (the package is not necessarily installed there).
        out = fresh("usersite-out")
        ctx = ingest_ctx(out=str(out), ingest_python=base, ingest_pass_env=["HOME"], ingest_timeout=60)
        naive = [{"argv": [base, "-c", "pass"], "cases": ["c"], "default": False, "threads": True}]
        run_cmd.run_ingest(naive, ctx)
        self.assertIn("warehouse-secret", [r["secret"] for r in self.records()], "a non-isolated ingest must run it")
        self.leak.unlink()
        default = run_cmd.ingest_commands(
            [{"name": "c", "status": "ok", "dir": "cases/c"}],
            {"c": {"name": "c", "id_profile": "generic", "ingest": None}},
            ctx,
        )
        self.assertEqual(default[0]["argv"][:3], [base, "-I", "-m"])
        done = run_cmd.run_ingest(default, ctx)
        self.assertIsNotNone(done[0]["returncode"], "the ingest must have started")
        self.assertEqual(self.records(), [], "the isolated ingest must not run the planted module")

    def records(self):
        if not self.leak.exists():
            return []
        return [json.loads(line) for line in self.leak.read_text().splitlines() if line]

    def test_ingest_pythonpath_must_be_outside_the_workdir_and_results(self):
        inside = self.wd / "pkgs"
        inside.mkdir()
        for bad in (inside, self.wd):
            code, _, err = run_main(
                self.plan, fresh("pp-out"), "--workdir", str(self.wd), "--ingest", "--ingest-pythonpath", str(bad)
            )
            self.assertEqual(code, 2, err)
            self.assertIn("a benchmark can write there", err)
        out = fresh("pp-out2")
        (out / "libs").mkdir()
        code, _, err = run_main(
            self.plan, out, "--workdir", str(self.wd), "--ingest", "--ingest-pythonpath", str(out / "libs")
        )
        self.assertEqual(code, 2)
        link = ITMP / "pp-link"
        link.unlink(missing_ok=True)
        link.symlink_to(inside)  # judged by realpath
        code, _, err = run_main(
            self.plan, fresh("pp-out3"), "--workdir", str(self.wd), "--ingest", "--ingest-pythonpath", str(link)
        )
        self.assertEqual(code, 2)
        code, _, err = run_main(
            self.plan,
            fresh("pp-out4"),
            "--workdir",
            str(self.wd),
            "--ingest",
            "--ingest-pythonpath",
            str(ITMP / "absent"),
        )
        self.assertEqual(code, 2)

    def test_the_ingest_gets_an_empty_directory_a_fresh_home_and_a_safe_path(self):
        probe = ITMP / "where.json"
        probe.unlink(missing_ok=True)
        stub = ITMP / "where.py"
        stub.write_text(
            "import json, os\nopen(" + repr(str(probe)) + ", 'w').write(json.dumps("
            "[os.getcwd(), os.environ.get('HOME'), os.environ.get('PYTHONSAFEPATH'), sorted(k for k in os.environ if k.startswith('PYTHON'))]))\n"
        )
        (code, _, err), out = self.run_it("where", "--ingest-cmd", f"{sys.executable} {stub}", pythonpath=False)
        cwd, home, safe, python_vars = json.loads(probe.read_text())
        ingest = (Path(out) / "ingest").resolve()
        self.assertEqual((Path(cwd).resolve().parent, Path(home).resolve().parent), (ingest, ingest))
        self.assertTrue(Path(cwd).name.startswith("cwd-") and Path(home).name.startswith("home-"))
        self.assertEqual((safe, python_vars), ("1", ["PYTHONSAFEPATH"]))  # the caller's PYTHONPATH did not come along
        self.assertNotEqual(Path(home).resolve(), self.home.resolve())
        for d in (ingest, Path(cwd), Path(home)):
            self.assertEqual(os.stat(d).st_mode & 0o777, 0o700, d)

    def test_home_and_python_variables_reach_the_ingest_only_when_named(self):
        probe = ITMP / "where2.json"
        stub = ITMP / "where2.py"
        stub.write_text(
            "import json, os\nopen("
            + repr(str(probe))
            + ", 'w').write(json.dumps({k: os.environ.get(k) for k in ('HOME', 'PYTHONUSERBASE')}))\n"
        )
        with mock.patch.dict(os.environ, {"PYTHONUSERBASE": str(ITMP / "userbase")}):
            self.run_it(
                "named",
                "--ingest-cmd",
                f"{sys.executable} {stub}",
                "--ingest-pass-env",
                "HOME",
                "--ingest-pass-env",
                "PYTHONUSERBASE",
                pythonpath=False,
            )
        self.assertEqual(
            json.loads(probe.read_text()), {"HOME": str(self.home), "PYTHONUSERBASE": str(ITMP / "userbase")}
        )

    def test_the_default_argv_starts_with_the_isolated_flag(self):
        for profile in run_cmd.INGEST_DEFAULTS:
            self.assertEqual(run_cmd.INGEST_DEFAULTS[profile][:3], ["{python}", "-I", "-m"])
        ctx = ingest_ctx(ingest_pythonpath=[str(self.pkg)])
        argv = run_cmd.resolve_ingest({"id_profile": "generic"}, None, ctx)
        self.assertEqual(argv[:4], ["{python}", "-s", "-P", "-m"])  # -I would ignore the PYTHONPATH we set

    def test_a_planted_ingest_directory_or_link_is_replaced_not_followed(self):
        victim = ITMP / "victim.txt"
        victim.write_text("untouched")
        elsewhere = fresh("elsewhere")
        code = (
            "import os, sys\nout = sys.argv[1]\nos.symlink(sys.argv[2], os.path.join(out, 'ingest'))\n"
            "os.makedirs(sys.argv[2] + '/cwd-planted', exist_ok=True)\n"
            "os.symlink(sys.argv[3], sys.argv[2] + '/ingest-1.log')\n"
            "os.makedirs(os.environ['BENCH_RAW_DIR'], exist_ok=True)\n"
            "open(os.environ['BENCH_RAW_DIR'] + '/r.json', 'w').write('{}')\n"
        )
        plan = one_case_plan(
            "planted", command=["{python}", "-c", code, "{out_dir}", str(elsewhere), str(victim)], repeats=1
        )
        out = fresh("planted-out")
        stub = ITMP / "ok_report.py"
        stub.write_text(
            "import json\nprint(json.dumps({'run_id': '7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f', 'status': 'ingested'}))\n"
        )
        code, _, err = run_main(plan, out, "--ingest", "--ingest-cmd", f"{sys.executable} {stub}")
        self.assertEqual(code, 0, err)
        self.assertEqual(victim.read_text(), "untouched")
        root = out / "ingest"
        self.assertTrue(root.is_dir() and not root.is_symlink())
        self.assertEqual(os.stat(root).st_mode & 0o777, 0o700)
        self.assertFalse(any(p.name == "cwd-planted" for p in root.iterdir()))
        self.assertFalse((root / "ingest-1.log").is_symlink())

    def test_fresh_ingest_dirs_removes_files_links_and_directories(self):
        for kind in ("file", "link", "dir"):
            out = fresh(f"fresh-{kind}")
            target = out / "ingest"
            if kind == "file":
                target.write_text("x")
            elif kind == "link":
                target.symlink_to(ITMP)
            else:
                (target / "cwd").mkdir(parents=True)
            root, cwd, home = run_cmd.fresh_ingest_dirs(out)
            self.assertTrue(root.is_dir() and not root.is_symlink(), kind)
            self.assertEqual(sorted(p.name[:4] for p in root.iterdir()), ["cwd-", "home"], kind)
            self.assertTrue(ITMP.is_dir())

    def test_capture_refuses_an_existing_path_and_a_link_for_exclusive_logs(self):
        for kind in ("exists", "link"):
            path = ITMP / f"excl-{kind}.log"
            path.unlink(missing_ok=True)
            if kind == "exists":
                path.write_text("old")
            else:
                path.symlink_to(ITMP / "victim-target.txt")
            r, w = os.pipe()
            os.write(w, b"data")
            os.close(w)
            with os.fdopen(r, "rb") as stream:
                pump = run_cmd.Capture(stream, path, 1 << 20, exclusive=True)
                pump.start()
                pump.stop.set()
                pump.join(timeout=10)
            self.assertIsNotNone(pump.error, kind)
            self.assertFalse((ITMP / "victim-target.txt").exists())

    def test_a_grandchild_of_the_ingest_dies_with_the_ingest_timeout(self):
        pidfile = ITMP / "ingest-grandchild.pid"
        pidfile.unlink(missing_ok=True)
        stub = ITMP / "spawner.py"
        stub.write_text(
            "import subprocess, sys, time\n"
            f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"open({str(pidfile)!r}, 'w').write(str(p.pid))\ntime.sleep(120)\n"
        )
        out = fresh("grandchild-out")
        code, _, _ = run_main(
            make_plan(ITMP / "grandchild.json", "--cases", "op_add_fp16"),
            out,
            "--ingest",
            "--ingest-cmd",
            f"{sys.executable} {stub}",
            "--ingest-timeout",
            "2",
        )
        self.assertEqual(code, 5)
        pid = int(pidfile.read_text())
        for _ in range(60):
            if not still_running(pid):
                break
            time.sleep(0.1)
        else:
            self.fail("the ingest's grandchild outlived the timeout")

    def test_a_huge_ingest_log_is_capped_but_its_report_is_still_read(self):
        stub = ITMP / "chatty_ingest.py"
        stub.write_text(
            "import json, sys\nsys.stdout.write('x' * 500_000 + '\\n')\n"
            "print(json.dumps({'run_id': '7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f', 'status': 'ingested'}))\n"
            "print('trailing chatter that is not json')\n"
        )
        out = fresh("chatty-out")
        code, _, _ = run_main(
            make_plan(ITMP / "chatty.json", "--cases", "op_add_fp16"),
            out,
            "--ingest",
            "--ingest-cmd",
            f"{sys.executable} {stub}",
            "--max-log-bytes",
            "1000",
        )
        self.assertEqual(code, 0)
        self.assertLess((out / "ingest" / "ingest-1.log").stat().st_size, 1200)
        self.assertEqual(manifest(out)["stored_run_ids"], ["7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"])


class EnvFamilyTests(unittest.TestCase):
    def seen(self, label, *extra, **env):
        script = ITMP / "dump_env2.py"
        script.write_text(
            "import json, os, sys\nopen(sys.argv[1] + '/seen.json', 'w').write(json.dumps(dict(os.environ)))\n"
        )
        plan = one_case_plan(label, command=["{python}", str(script), "{repeat_dir}"], repeats=1)
        out = fresh(f"{label}-out")
        with clean_env(**env):
            run_main(plan, out, *extra)
        return json.loads(next(out.glob("cases/*/repeat-001/seen.json")).read_text()), out

    def test_each_default_family_reaches_the_command(self):
        env = {
            "http_proxy": "http://proxy.example:3128",
            "HTTPS_PROXY": "http://proxy.example:3128",
            "no_proxy": "localhost",
            "SSL_CERT_FILE": "/etc/ssl/cert.pem",
            "SSL_CERT_DIR": "/etc/ssl/certs",
            "REQUESTS_CA_BUNDLE": "/etc/ssl/cert.pem",
            "CURL_CA_BUNDLE": "/etc/ssl/cert.pem",
            "MAKEFLAGS": "-j4",
            "MAKELEVEL": "1",
            "MFLAGS": "-j4",
            "CC": "cc",
            "CXX": "c++",
            "CFLAGS": "-O2",
            "CXXFLAGS": "-O2",
            "LDFLAGS": "-L/opt",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_MAX_THREADS": "4",
            "XDG_CACHE_HOME": "/cache",
            "HF_HOME": "/hf",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NODE_NAME": "node-a",
            "HOSTNAME": "host-a",
            "TEMP": "/t",
        }
        seen, _ = self.seen("fam", **env)
        for name, value in env.items():
            self.assertEqual(seen.get(name), value, name)

    def test_tokens_in_the_same_families_are_still_withheld(self):
        seen, _ = self.seen(
            "fam-tokens", HF_TOKEN="x", HF_HOME="/hf", SPYRE_API_KEY="y", OMP_SECRET="z", XDG_CREDENTIALS="c"
        )
        self.assertEqual(seen.get("HF_HOME"), "/hf")
        for name in ("HF_TOKEN", "SPYRE_API_KEY", "OMP_SECRET", "XDG_CREDENTIALS"):
            self.assertNotIn(name, seen)
        seen, _ = self.seen("fam-explicit", "--pass-env", "SPYRE_API_KEY", SPYRE_API_KEY="y")
        self.assertEqual(seen.get("SPYRE_API_KEY"), "y")

    def test_credential_looking_values_are_dropped_with_a_note(self):
        env = {
            "HTTPS_PROXY": "http://user:hunter2@proxy.example:3128",
            "HF_HOME": "q1" * 20,
            "SPYRE_NOTE": "ghp_" + "b" * 30,
            "MAKEFLAGS": "-j4",
        }
        seen, out = self.seen("fam-values", **env)
        for name in ("HTTPS_PROXY", "HF_HOME", "SPYRE_NOTE"):
            self.assertNotIn(name, seen)
        self.assertEqual(seen["MAKEFLAGS"], "-j4")
        case = manifest(out)["cases"][0]
        notes = json.loads((out / case["dir"] / "repeat-001" / "status.json").read_text())["notes"]
        self.assertTrue(any("HTTPS_PROXY" in n and "credential-looking value" in n for n in notes))
        self.assertFalse(any("hunter2" in n for n in notes))  # the name is noted, never the value
        seen, _ = self.seen("fam-values2", "--pass-env", "HTTPS_PROXY", **env)
        self.assertEqual(seen["HTTPS_PROXY"], env["HTTPS_PROXY"])  # named explicitly: passed as is

    def test_pass_env_prefix_for_a_device_family(self):
        seen, _ = self.seen(
            "fam-prefix", "--pass-env-prefix", "ACME_", ACME_DEVICES="0,1", ACME_TOKEN="t", OTHER_DEVICES="2"
        )
        self.assertEqual(seen.get("ACME_DEVICES"), "0,1")
        self.assertNotIn("ACME_TOKEN", seen)  # a secret-looking name inside a passed family still needs --pass-env
        self.assertNotIn("OTHER_DEVICES", seen)
        seen, _ = self.seen("fam-prefix2", ACME_DEVICES="0,1", BENCH_PASS_ENV_PREFIXES="ACME_")
        self.assertEqual(seen.get("ACME_DEVICES"), "0,1")

    def test_helpers(self):
        self.assertTrue(planfile.credential_looking("https://u:p@h/x"))
        self.assertTrue(planfile.credential_looking("q1" * 16))
        self.assertFalse(planfile.credential_looking("a" * 40))  # no digit: not token-shaped
        self.assertFalse(planfile.credential_looking("550e8400-e29b-41d4-a716-446655440000"))
        self.assertFalse(planfile.credential_looking("a-very-long-dashed-host-name-for-testing.example.com"))
        self.assertTrue(planfile.credential_looking("eyJhbGciOi.eyJzdWIiOi.c2ln"))
        self.assertFalse(planfile.credential_looking("/usr/bin:/usr/local/bin"))
        self.assertFalse(planfile.credential_looking("x86_64"))
        env, dropped = planfile.env_filter({"PATH": "/bin", "https_proxy": "http://u:p@h", "no_proxy": "h"})
        self.assertEqual(
            (env, dropped), ({"PATH": "/bin", "no_proxy": "h"}, {"https_proxy": "credential-looking value"})
        )


class RedactionTests(unittest.TestCase):
    CASES = [
        (["-p", "abc"], ["-p", "<redacted>"]),
        (["-pabc"], ["-p<redacted>"]),
        (["TOKEN", "abc"], ["TOKEN", "<redacted>"]),
        (["--key", "abc"], ["--key", "<redacted>"]),
        (["--access-key", "abc"], ["--access-key", "<redacted>"]),
        (["--pw", "abc"], ["--pw", "<redacted>"]),
        (["--bearer", "abc"], ["--bearer", "<redacted>"]),
        (["--cookie", "a=b"], ["--cookie", "<redacted>"]),
        (["--dsn", "postgres://u:p@h/db"], ["--dsn", "<redacted>"]),
        (["https://user:pw@host/x"], ["https://<redacted>@host/x"]),
        (["--url=https://user:pw@host/x"], ["--url=https://<redacted>@host/x"]),
        (["--header", "Authorization: Bearer abcdef123"], ["--header", "Authorization: <redacted>"]),
        (["-H", "Cookie: sid=abc"], ["-H", "Cookie: <redacted>"]),
        (["--passes", "3"], ["--passes", "3"]),
        (["--author", "joe"], ["--author", "joe"]),
        (["MONKEY_PATCH=1"], ["MONKEY_PATCH=1"]),
        (["--jenkins-run-key", "manual:x:y"], ["--jenkins-run-key", "manual:x:y"]),
        (["AWS_SECRET_ACCESS_KEY=zz"], ["AWS_SECRET_ACCESS_KEY=<redacted>"]),
        (["--password=zz", "--arch", "x86_64"], ["--password=<redacted>", "--arch", "x86_64"]),
        (["--token", "--arch", "x86_64"], ["--token", "--arch", "x86_64"]),  # a following flag is not a value
    ]

    def test_redact_argv(self):
        for argv, want in self.CASES:
            with self.subTest(argv=argv):
                self.assertEqual(planfile.redact_argv(argv), want)

    def secret_command(self):
        return ["{python}", "-c", "pass", "--token", "hunter2", "https://user:pw0rd@host/x", "-p", "pwvalue"]

    def test_run_writes_no_secret_into_the_results_directory(self):
        out = fresh("redact-all")
        plan = one_case_plan("redact-all", command=self.secret_command(), repeats=1, allow_empty=True)
        run_main(plan, out)
        text = "\n".join(p.read_text(errors="replace") for p in Path(out).rglob("*.json"))
        for secret in ("hunter2", "pw0rd", "pwvalue"):
            self.assertNotIn(secret, text)
        self.assertIn("<redacted>", (out / "plan.redacted.json").read_text())
        self.assertIn("<redacted>", next(out.glob("cases/*/case.json")).read_text())

    def test_plan_json_output_is_redacted_but_the_executable_file_is_not(self):
        d = ITMP / "secret-cat"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(DEMO, d)
        f = d / "ops" / "op-add.yaml"
        f.write_text(
            f.read_text().replace('"--out", "{repeat_dir}"]', '"--out", "{repeat_dir}", "--token", "hunter2"]')
        )
        written = ITMP / "secret-plan.json"
        common = [
            "--catalog",
            str(d),
            "--central",
            str(CENTRAL),
            "--arch",
            "x86_64",
            "--reference-node",
            "--purpose",
            "nightly",
        ]
        code, out, err = capture(plan_cmd.main, [*common, "--json", "--out", str(written)])
        self.assertEqual(code, 0, err)
        self.assertNotIn("hunter2", out)
        self.assertIn("<redacted>", out)
        self.assertIn("hunter2", written.read_text())  # run must be able to execute it


class RunIdTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_reusing_a_run_id_over_an_existing_run_is_refused(self):
        plan = make_plan(ITMP / "rid.json")
        out = fresh("rid-out")
        self.assertEqual(run_main(plan, out, "--run-id", RUN, *INGEST)[0], 0)
        code, _, err = run_main(plan, out, "--run-id", RUN, *INGEST)
        self.assertEqual(code, 2)
        self.assertIn("already used", err)
        self.assertIn("--force-new-run", err)
        self.assertEqual(manifest(out)["run_id"], RUN)  # nothing was overwritten

    def test_force_new_run_mints_a_fresh_id(self):
        plan = make_plan(ITMP / "rid2.json")
        out = fresh("rid2-out")
        run_main(plan, out, "--run-id", RUN, *INGEST)
        code, _, _ = run_main(plan, out, "--run-id", RUN, "--force-new-run", *INGEST)
        self.assertEqual(code, 0)
        self.assertNotEqual(manifest(out)["run_id"], RUN)

    def test_the_shared_env_run_id_collides_the_same_way(self):
        plan = make_plan(ITMP / "rid3.json")
        out = fresh("rid3-out")
        with mock.patch.dict(os.environ, {"RUN_ID": RUN}):
            self.assertEqual(run_main(plan, out, *INGEST)[0], 0)
            self.assertEqual(run_main(plan, out, *INGEST)[0], 2)

    def test_another_out_dir_and_a_dry_run_are_not_blocked(self):
        plan = make_plan(ITMP / "rid4.json")
        run_main(plan, fresh("rid4-a"), "--run-id", RUN, *INGEST)
        self.assertEqual(run_main(plan, fresh("rid4-b"), "--run-id", RUN, *INGEST)[0], 0)
        self.assertEqual(run_main(plan, ITMP / "rid4-a", "--run-id", RUN, "--dry-run")[0], 0)


class VerdictRunIdTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.stored = "7d0f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"
        self.dir = fresh("vdir")
        (self.dir / "run.json").write_text(json.dumps({"engine_run_id": RUN, "stored_run_ids": [self.stored, OTHER]}))
        rows = {
            self.stored: [verdict_row(run_id=self.stored)],
            OTHER: [verdict_row(run_id=OTHER, name="op_b", status="regressed")],
        }
        self.store = stores.MemoryStore(verdicts=rows)

    def call(self, *argv, st=None):
        return capture(verdict_cmd.main, list(argv), st=st or self.store)

    def test_run_dir_reads_every_stored_run_id(self):
        code, out, _ = self.call("--run-dir", str(self.dir), "--json", "--fail-on", "regressed")
        doc = json.loads(out)
        self.assertEqual((code, doc["run_ids"], doc["summary"]["total"]), (1, [self.stored, OTHER], 2))

    def test_an_engine_run_id_of_a_known_run_is_translated(self):
        code, out, err = self.call("--run-id", RUN, "--run-dir", str(self.dir), "--json")
        self.assertEqual(json.loads(out)["run_ids"], [self.stored, OTHER])
        self.assertIn("engine run id", err)
        with mock.patch.dict(os.environ, {"RESULTS_DIR": str(self.dir)}):  # found through the default out dir too
            self.assertEqual(json.loads(self.call("--run-id", RUN, "--json")[1])["run_ids"], [self.stored, OTHER])

    def test_a_stored_id_is_not_translated(self):
        doc = json.loads(self.call("--run-id", self.stored, "--run-dir", str(self.dir), "--json")[1])
        self.assertEqual(doc["run_ids"], [self.stored])

    def test_zero_rows_are_loud_and_can_fail(self):
        code, out, err = self.call("--run-id", RUN, "--json")
        self.assertEqual(code, 0)
        self.assertIn(f"no verdict rows for {RUN}", err)
        self.assertEqual(self.call("--run-id", RUN, "--require-rows")[0], 1)
        self.assertIn("no verdict rows for", self.call("--run-id", RUN)[1])

    def test_run_id_env_twins_are_consistent_with_run(self):
        with mock.patch.dict(os.environ, {"RUN_ID": self.stored}):
            self.assertEqual(json.loads(self.call("--json")[1])["run_ids"], [self.stored])
        with mock.patch.dict(os.environ, {"RUN_ID": "12345"}):  # a CI number is not a run id
            self.assertEqual(self.call("--json")[0], 2)

    def test_run_dir_problems_exit_2(self):
        self.assertEqual(self.call("--run-dir", str(ITMP / "nowhere"))[0], 2)
        empty = fresh("vdir-empty")
        (empty / "run.json").write_text(json.dumps({"engine_run_id": RUN, "stored_run_ids": []}))
        code, _, err = self.call("--run-dir", str(empty))
        self.assertEqual(code, 2)
        self.assertIn("no stored run ids", err)

    def test_counts_are_integers_whatever_the_store_returns(self):  # n_* coercion
        row = verdict_row(n_baseline="20", n_current=None)
        v = verdict_cmd.normal(row, False)
        self.assertEqual((v["n_baseline"], v["n_current"]), (20, 0))
        self.assertIsInstance(verdict_cmd.normal(verdict_row(n_baseline=20.0), False)["n_baseline"], int)


class SmallContractTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_capture_survives_a_failing_log_write(self):
        r, w = os.pipe()
        os.write(w, b"x" * 50_000)
        os.close(w)

        class Full:
            def write(self, data):
                raise OSError(errno.ENOSPC, "no space")

            def close(self):
                pass

        with os.fdopen(r, "rb") as stream, mock.patch.object(run_cmd, "open", create=True, return_value=Full()):
            pump = run_cmd.Capture(stream, ITMP / "full.log", 1 << 20)
            pump.start()
            pump.stop.set()
            pump.join(timeout=10)
        self.assertFalse(pump.is_alive())
        self.assertEqual(pump.error.errno, errno.ENOSPC)

    def test_a_log_write_failure_fails_the_repeat_not_the_wait(self):
        class Dead(run_cmd.Capture):
            def run(self):
                super().run()
                self.error = OSError(errno.ENOSPC, "no space")

        d = fresh("enospc")
        with mock.patch.object(run_cmd, "Capture", Dead):
            rc, reason, _ = run_cmd.execute_command(
                [sys.executable, "-c", "print('hi')"],
                os.environ.copy(),
                str(d),
                d / "o",
                d / "e",
                30,
                None,
                "",
                subprocess.Popen,
            )
        self.assertEqual((rc, reason), (0, f"log_write_failed:{errno.ENOSPC}"))

    def test_a_detached_process_holding_the_pipe_does_not_cost_seconds(self):
        code = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], start_new_session=True)"
        d = fresh("detached")
        t0 = time.monotonic()
        rc, reason, _ = run_cmd.execute_command(
            [sys.executable, "-c", code], os.environ.copy(), str(d), d / "o", d / "e", 60, None, "", subprocess.Popen
        )
        self.assertEqual((rc, reason), (0, ""))
        self.assertLess(time.monotonic() - t0, 6)

    def test_flag_beats_env_for_select_and_cases_and_they_are_not_merged(self):
        common = [
            "--catalog",
            str(DEMO),
            "--central",
            str(CENTRAL),
            "--arch",
            "x86_64",
            "--reference-node",
            "--purpose",
            "nightly",
            "--json",
        ]
        with mock.patch.dict(os.environ, {"BENCH_CASES": "op_mul_fp16", "BENCH_SELECTION": "smoke"}):
            code, out, _ = capture(plan_cmd.main, [*common])
            self.assertEqual(
                sorted(c["name"] for c in json.loads(out)["cases"]), ["op_add_fp16", "op_mul_fp16"]
            )  # env: both
            code, out, _ = capture(plan_cmd.main, [*common, "--cases", "op_add_fp16"])
            self.assertEqual([c["name"] for c in json.loads(out)["cases"]], ["op_add_fp16"])  # env twins ignored
            code, out, _ = capture(plan_cmd.main, [*common, "--select", "op_mul_fp16"])
            self.assertEqual([c["name"] for c in json.loads(out)["cases"]], ["op_mul_fp16"])

    def test_run_does_not_pick_up_benchmarks_from_the_callers_directory(self):
        cwd = fresh("implicit-cwd")
        shutil.copytree(DEMO, cwd / "benchmarks")
        old = os.getcwd()
        os.chdir(cwd)
        self.addCleanup(os.chdir, old)
        plan = make_plan(ITMP / "implicit.json")
        out = fresh("implicit-out")
        code, _, err = capture(
            run_cmd.main, ["--plan", str(plan), "--out", str(out), "--workdir", str(DEMO), "--central", str(CENTRAL)]
        )
        self.assertEqual(code, 0, err)
        self.assertEqual({c["result"] for c in manifest(out)["cases"]}, {None})
        with mock.patch.dict(os.environ, {"BENCH_DIR": str(cwd / "benchmarks")}):  # named explicitly: used
            out2 = fresh("implicit-out2")
            capture(
                run_cmd.main,
                ["--plan", str(plan), "--out", str(out2), "--workdir", str(DEMO), "--central", str(CENTRAL)],
            )
            self.assertTrue(all(c["result"] for c in manifest(out2)["cases"]))

    def test_out_is_resolved_through_symlinks(self):
        real = fresh("real-out")
        link = ITMP / "link-out"
        link.unlink(missing_ok=True)
        link.symlink_to(real)
        run_main(make_plan(ITMP / "link.json"), link)
        self.assertTrue((real / "run.json").is_file())
        self.assertTrue(manifest(real)["cases"][0]["dir"].startswith("cases/"))

    # ---- mutants the earlier suites let through
    def test_ingest_with_a_sidecar_or_allow_empty_still_needs_raw_output(self):
        sidecar = ["{python}", "-c", "import sys; open(sys.argv[1] + '/metrics.json', 'w').write('{}')", "{repeat_dir}"]
        for label, changes in (
            ("side", {"command": sidecar}),
            ("empty", {"command": ["{python}", "-c", "pass"], "allow_empty": True}),
        ):
            out = fresh(f"mut11-{label}")
            code, _, _ = run_main(
                one_case_plan(f"mut11-{label}", repeats=1, **changes),
                out,
                "--ingest",
                "--ingest-cmd",
                f"{sys.executable} -c pass",
            )
            self.assertEqual(code, 1, label)
            self.assertEqual(manifest(out)["cases"][0]["reason"], "no_output", label)

    def test_engine_set_variables_reach_the_command_with_their_values(self):
        script = ITMP / "engine_vars.py"
        script.write_text(
            "import json, os, sys\nopen(sys.argv[1] + '/seen.json', 'w').write(json.dumps(dict(os.environ)))\n"
        )
        out = fresh("enginevars")
        run_main(
            one_case_plan("enginevars", command=["{python}", str(script), "{repeat_dir}"], repeats=1),
            out,
            "--run-id",
            RUN,
            "--quiet",
        )
        seen = json.loads(next(out.glob("cases/*/repeat-001/seen.json")).read_text())
        want = {"BENCH_RUN_ID": RUN, "BENCH_CASE": "op_add_fp16", "BENCH_REPEAT": "1", "BENCH_REPEATS": "1", "BENCH_CACHE": "warm",
                "BENCH_PROFILE_LEVEL": "off", "BENCH_SUBJECT_ID": SUBJECT, "BENCH_QUIET": "1"}  # fmt: skip
        for name, value in want.items():
            self.assertEqual(seen.get(name), value, name)
        self.assertTrue(seen["BENCH_RAW_DIR"].endswith("/raw") and seen["BENCH_REPEAT_DIR"].endswith("/repeat-001"))

    def test_the_raw_size_warning_boundary(self):
        out = fresh("rawsize")
        plan = make_plan(ITMP / "rawsize.json", "--cases", "op_mul_fp16")
        run_main(plan, out)
        raw = next(out.glob("cases/*/raw"))
        size = sum(f.stat().st_size for f in raw.rglob("*") if f.is_file())
        for limit, noted in ((size, False), (size - 1, True)):
            o = fresh(f"rawsize-{limit}")
            run_main(plan, o, "--max-raw-bytes", str(limit))
            self.assertEqual(bool(manifest(o)["result_notes"]), noted, limit)

    def test_flag_on_is_case_insensitive(self):
        for value in ("1", "true", "TRUE", "Yes", "ON"):
            with mock.patch.dict(os.environ, {"BENCH_X": value}):
                self.assertTrue(cliutil.flag_on(False, "BENCH_X"), value)
        for value in ("0", "false", "no", "", "off"):
            with mock.patch.dict(os.environ, {"BENCH_X": value}):
                self.assertFalse(cliutil.flag_on(False, "BENCH_X"), value)
        self.assertTrue(cliutil.flag_on(True, "BENCH_NOT_SET"))

    def test_an_empty_env_twin_is_unset(self):
        with mock.patch.dict(os.environ, {"BENCH_X": ""}):
            self.assertEqual(cliutil.env("BENCH_X", "dflt"), "dflt")
            self.assertEqual(cliutil.pick(None, "BENCH_X", "dflt"), "dflt")
        with mock.patch.dict(os.environ, {"BENCH_PURPOSE": ""}):
            code, _, err = plan_main()  # BASE flags, no purpose: the empty twin must not count as one
        self.assertEqual(code, 2)
        self.assertIn("purpose", err)


if __name__ == "__main__":
    unittest.main()

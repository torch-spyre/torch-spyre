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

"""Engine tests: schemas, packs, examples, identity, plan and reuse, invalidation, qualification, results, negatives.
python -m pytest -q -p no:cacheprovider tests
"""

import contextlib
import copy
import io
import json
import shutil
import statistics
import sys
import tempfile
import unittest
from unittest import mock
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from spyre_bench import analysis, demo, engine, gen_schemas, placement
from spyre_bench import explain as ex
from spyre_bench import validate as val
from spyre_bench.cli import main as cli_main

EXT = Path(__file__).resolve().parent.parent
PKG = EXT / "spyre_bench"
X = EXT / "examples"
PG = X / "packs" / "postgres-example"
TRACE_PACK = X / "packs" / "trace-example"
CENTRAL = X / "central"
TMP = Path(tempfile.mkdtemp(prefix="spyre-bench-tests-"))
CATALOGS = {
    "llm-demo": ([], [CENTRAL]),
    "hf-demo": ([], [CENTRAL]),
    "kernel-demo": ([], [CENTRAL]),
    "comms-demo": ([], [CENTRAL]),
    "central": ([], []),
    "postgres-demo": ([PG], []),
    "pinning-demo": ([], []),
    "trace-demo": ([TRACE_PACK], []),
}
NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
WITH_JSONSCHEMA = engine.Draft202012Validator is not None


def cat(name, packs=None, central=None):
    p, c = CATALOGS.get(name, ([], [CENTRAL]))
    return engine.Catalog(
        [X / name if isinstance(name, str) else name], p if packs is None else packs, c if central is None else central
    )


def copy_catalog(name, case):
    d = TMP / case.replace(" ", "-")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(X / name, d)
    return d


def edit(path, old, new):
    t = path.read_text()
    assert old in t, (path, old)
    path.write_text(t.replace(old, new, 1))


def setUpModule():
    if not X.is_dir():
        raise unittest.SkipTest(f"examples/ not found at {X}: the tests need a source checkout, not an installed wheel")


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def ids_by_name(c):
    return {x["name"]: c.ids(x)["benchmark_id"] for x in c.cases}


class Schemas(unittest.TestCase):
    @unittest.skipUnless(WITH_JSONSCHEMA, "needs jsonschema")
    def test_core_schemas_metaschema_subset_current(self):
        from jsonschema import Draft202012Validator

        files = sorted((PKG / "core" / "schemas").glob("*.schema.json"))
        self.assertEqual(len(files), len(engine.KINDS) + 1)
        shapes = sorted((PKG / "core" / "schemas" / "tool-output").glob("*.shape.json"))
        self.assertEqual(len(shapes), 2)
        for f in files + shapes:
            d = json.loads(f.read_text())
            Draft202012Validator.check_schema(d)
            self.assertEqual(engine.subset_keywords(d), [], f.name)
        self.assertEqual(gen_schemas.main(["--check"]), 0)

    def test_engine_name(self):
        self.assertEqual(
            (engine.TOOL, engine.PACKAGE, engine.CLI_ALIAS, engine.API),
            ("spyre-bench", "spyre_bench", "bench", "bench/v1alpha1"),
        )
        self.assertEqual(gen_schemas.TOOL, "spyre-bench")
        reg = engine.Registry()
        for d in ["llm-serving", "spyre", "kernel", "comms"]:
            reg.load(PKG / "packs" / d)
        cmds = [
            c for r in list(reg.runners.values()) + list(reg.probes.values()) for c in r.get("command", []) if "." in c
        ]
        self.assertTrue(cmds and all(c.startswith("spyre_bench.") for c in cmds), cmds)
        self.assertEqual(reg.probes["host"]["command"][0], "spyre-bench")

    def test_kind_set(self):
        self.assertEqual(len(engine.AUTHORED), 16)
        self.assertEqual(engine.AUTHORED[11:14], ("NodeClass", "ValidityPolicy", "Expectation"))
        for k in ("Run", "Result", "InvalidationEvent", "Metric"):
            self.assertNotIn(k, engine.KINDS)

    def test_every_pack_loads_clean(self):
        reg = engine.Registry()
        for d in ["llm-serving", "spyre", "kernel", "comms"]:
            reg.load(PKG / "packs" / d)
        reg.load(PG)
        reg.finish()
        self.assertEqual(reg.errors, [])
        self.assertEqual(sorted(reg.packs), ["comms", "core", "kernel", "llm", "pg", "spyre"])
        self.assertGreaterEqual(len(reg.types), 20)
        self.assertGreaterEqual(len(reg.metrics), 20)
        self.assertGreaterEqual(len(reg.attrs), 50)
        self.assertIn("spyre/pf", reg.device_classes)
        self.assertIn("cpu/socket", reg.device_classes)

    def test_pack_rules(self):
        d = TMP / "bad-pack"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(PG, d)
        edit(d / "pack.yaml", 'requires: {core: "0.2"}', 'requires: {core: "0.3"}')
        edit(
            d / "pack.yaml",
            "{key: env.label.storage.class, class: label,",
            "{key: env.label.storage.class, class: environment,",
        )
        edit(d / "pack.yaml", "{name: pg.tps,", "{name: llm.tps,")
        edit(d / "pack.yaml", "default_metrics: [pg.tps, pg.latency.mean]", "default_metrics: [pg.tps, pg.nope]")
        s = json.loads((d / "schemas/workload-pgbench.json").read_text())
        s["properties"]["inputs"] = {"type": "string"}
        (d / "schemas/workload-pgbench.json").write_text(json.dumps(s))
        reg = engine.Registry()
        reg.load(d)
        reg.finish()
        msgs = "\n".join(reg.errors)
        self.assertIn("requires core 0.3, engine core is 0.2.0", msgs)
        self.assertIn("class environment keys start with env.", msgs)
        self.assertIn("is outside the pack namespace pg", msgs)
        self.assertIn("redefines core fields ['inputs']", msgs)
        self.assertIn("default_metrics: 'pg.nope' is not a registered metric", msgs)

    def test_attribute_registry_lookup(self):
        reg = engine.Registry()
        reg.load(PKG / "packs" / "spyre")
        self.assertEqual(reg.lookup("subject.dep.dep-a")["derived_from"], "subject.id")
        self.assertEqual(reg.lookup("env.label.pool")["values"], ["a", "b"])
        self.assertEqual(reg.lookup("env.label.rack")["key"], "env.label.<name>")
        self.assertTrue(reg.matches_registry("case.*") and reg.matches_registry("subject.dep.*"))
        self.assertFalse(reg.matches_registry("env.alloc.*"))
        self.assertIn("env.fp.cpu.base_ghz", reg.fingerprint_keys())
        self.assertNotIn("env.placement_id", reg.fingerprint_keys())


class Examples(unittest.TestCase):
    def test_every_catalog_validates(self):
        for name in CATALOGS:
            self.assertEqual(cat(name).errors, [], name)

    def test_case_counts(self):
        want = {
            "llm-demo": 16,
            "hf-demo": 4,
            "kernel-demo": 5,
            "comms-demo": 5,
            "postgres-demo": 7,
            "pinning-demo": 4,
            "trace-demo": 3,
            "central": 0,
        }
        self.assertEqual({n: len(cat(n).cases) for n in want}, want)

    def test_cli(self):
        args = [str(X / n) for n in CATALOGS if n != "central"] + [
            *("--pack", str(PG)),
            *("--pack", str(TRACE_PACK)),
            *("--central", str(CENTRAL)),
        ]
        self.assertEqual(val.main(args), 0)
        self.assertEqual(val.main([str(CENTRAL)]), 0)

    def test_cli_verbs(self):
        self.assertEqual(cli_main([]), 0)
        self.assertEqual(cli_main(["nope"]), 2)
        self.assertEqual(cli_main(["gen-schemas", "--check"]), 0)
        self.assertEqual(cli_main(["validate", str(X / "llm-demo"), "--central", str(CENTRAL)]), 0)
        self.assertEqual(
            cli_main(
                ["explain", "serve_demoa_tp4_c5_all", "--catalog", str(X / "llm-demo"), "--central", str(CENTRAL)]
            ),
            0,
        )

    def test_cli_errors_exit_2_without_traceback(self):
        for argv in (
            ["validate", "/nonexistent-catalog"],
            ["explain", "x", "--catalog", "/nonexistent-catalog"],
            ["explain", "no-such-case", "--catalog", str(X / "llm-demo"), "--central", str(CENTRAL)],
            ["explain", "x", "--catalog", str(X / "llm-demo")],
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(cli_main(argv), 2, argv)
            self.assertNotIn("Traceback", err.getvalue(), argv)
            self.assertTrue(err.getvalue().strip(), argv)

    def test_unreadable_and_misplaced_inputs_exit_2(self):
        d = TMP / "bad-bytes"
        d.mkdir(exist_ok=True)
        (d / "x.yaml").write_bytes(b"\xff\xfe")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(err):
            self.assertEqual(cli_main(["validate", str(d)]), 2)
        self.assertIn("not valid UTF-8", out.getvalue())
        self.assertNotIn("Traceback", err.getvalue() + out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli_main(["validate", str(d / "x.yaml")]), 2)
        self.assertIn("expected a catalog directory", out.getvalue())

    def test_explain_bad_set_exits_2_without_traceback(self):
        base = ["explain", "serve_demoa_tp4_c5_all", "--catalog", str(X / "llm-demo"), "--central", str(CENTRAL)]
        for bad in (
            ["--set", "workload=3"],
            ["--set", "system=nope"],
            ["--set", "foo"],
            ["--set", "workload.load.executor=bogus"],
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(base + bad), 2, bad)
            self.assertNotIn("Traceback", err.getvalue(), bad)
            self.assertEqual(len(err.getvalue().strip().splitlines()), 1, (bad, err.getvalue()))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main(base + ["--set", "workload.load.concurrency=8"]), 0)

    def test_gen_schemas_flags_and_output_dir(self):
        self.assertEqual(cli_main(["gen-schemas", "--bogus"]), 2)
        out = TMP / "schemas-out"
        self.assertEqual(cli_main(["gen-schemas", "--out", str(out)]), 0)
        self.assertEqual(cli_main(["gen-schemas", "--check", "--out", str(out)]), 0)
        (out / "project.schema.json").write_text("{}")
        self.assertEqual(cli_main(["gen-schemas", "--check", "--out", str(out)]), 1)
        gen_schemas.SOURCE_CHECKOUT, saved = False, gen_schemas.SOURCE_CHECKOUT
        try:
            self.assertEqual(cli_main(["gen-schemas"]), 2)
        finally:
            gen_schemas.SOURCE_CHECKOUT = saved

    def test_mistyped_pack_dir_is_an_error(self):
        self.assertEqual(
            cli_main(["validate", str(X / "hf-demo"), "--central", str(CENTRAL), "--pack", "/nonexistent-pack"]), 2
        )
        empty = TMP / "empty-pack-dir"
        empty.mkdir(exist_ok=True)
        self.assertEqual(cli_main(["validate", str(X / "hf-demo"), "--pack", str(empty)]), 2)

    def test_tool_output_shapes(self):
        results = json.loads((X / "results" / "pg-nightly.results.json").read_text())
        self.assertEqual(len(results), 7)
        for r in results:
            self.assertEqual(engine.shape_schema("result").errors(r), [])
        for ev in yaml.safe_load((X / "tool-output" / "invalidations.yaml").read_text())["events"]:
            self.assertEqual(engine.shape_schema("invalidation").errors(ev), [], ev["id"])

    def test_tool_output_shapes_stdlib(self):
        engine.Schema.use_jsonschema = False
        engine._CORE.clear()
        try:
            for r in json.loads((X / "results" / "pg-nightly.results.json").read_text()):
                self.assertEqual(engine.shape_schema("result").errors(r), [])
            bad = json.loads((X / "results" / "pg-nightly.results.json").read_text())[0]
            del bad["status"]["aggregate"]
            self.assertTrue(engine.shape_schema("result").errors(bad))
        finally:
            engine.Schema.use_jsonschema = True
            engine._CORE.clear()

    def test_tool_written_kinds_are_not_authorable(self):
        for kind in ("Result", "InvalidationEvent", "Run"):
            doc = engine.Doc(
                "x.yaml", 0, {"apiVersion": engine.API, "kind": kind, "metadata": {"name": "x"}, "spec": {}}
            )
            self.assertTrue(any("tool output" in e for e in engine.envelope_errors(doc)), kind)
        doc = engine.Doc(
            "x.yaml", 0, {"apiVersion": "legacy.dev/v1alpha1", "kind": "Suite", "metadata": {"name": "x"}, "spec": {}}
        )
        self.assertTrue(any("apiVersion: must be 'bench/v1alpha1'" in e for e in engine.envelope_errors(doc)))

    def test_stdlib_fallback_agrees(self):
        engine.Schema.use_jsonschema = False
        engine._CORE.clear()
        try:
            for name in CATALOGS:
                self.assertEqual(cat(name).errors, [], name)
            d = copy_catalog("llm-demo", "fallback")
            edit(d / "decoder/serve-demo-llm-a.yaml", "  precision: fp16", "  precison: fp16")
            self.assertTrue(any("unknown key 'precison' (did you mean 'precision'?)" in e for e in cat(d).errors))
        finally:
            engine.Schema.use_jsonschema = True
            engine._CORE.clear()

    def test_postgres_pack_registers_at_run_time(self):
        c = cat("postgres-demo", packs=[])
        self.assertTrue(any("pack 'postgres-example' not found" in e and "--pack" in e for e in c.errors), c.errors)
        c = cat("postgres-demo")
        self.assertEqual(c.errors, [])
        self.assertIn("pg", c.reg.packs)

    def test_metrics_default_from_workload_type(self):
        c = cat("kernel-demo")
        op = next(x for x in c.cases if x["benchmark"] == "op-matmul")
        self.assertIsNone(op["metrics"])
        self.assertEqual(c.case_metrics(op)[0], ["kernel.time.mean"])
        si = cat("llm-demo")
        s = next(x for x in si.cases if x["name"] == "serve_demoa_tp4_c5_all")
        self.assertEqual(
            si.case_metrics(s), (["llm.ttft.p99", "llm.itl.median", "llm.output_throughput"], ["llm.itl.median"])
        )

    def test_load_scenario_preset(self):
        c = cat("llm-demo")
        wl = c.effective("Workload", "serve-scenario-server")
        load = c.resolved_load(wl, c.reg.types[("Workload", "llm/serve")])
        self.assertEqual((load["executor"], load["rate"], load["arrival"]), ("open", 1.0, "poisson"))


class Identity(unittest.TestCase):
    def test_benchmark_id_is_the_repo_rule(self):
        self.assertEqual(
            engine.benchmark_id("llm-demo", "serve-x", ["domain__a"], {"b": "1", "a": "X"}, ["a", "b"]),
            "eb48287f-0149-5f8b-b44a-c8968fffd3cb",
        )
        self.assertEqual(engine.benchmark_id("", "x", []), "")
        self.assertEqual(
            engine.split_tags(["cadence__nightly", "headline", "nightly"]), (["headline"], ["cadence__nightly"])
        )
        repo = EXT.parent / "clickhouse-ingest"
        if not repo.exists():
            self.skipTest("clickhouse-ingest not alongside this package")
        sys.path.insert(0, str(repo))
        try:
            from spyre_clickhouse_ingest.identity import BenchmarkId
        except Exception as x:  # noqa: BLE001 - optional cross-check
            self.skipTest(f"repo identity module not importable: {x}")
        finally:
            sys.path.remove(str(repo))
        from spyre_clickhouse_ingest import identity

        self.assertEqual(engine.ID_NAMESPACE, identity.ID_NAMESPACE)
        for args in [
            ("llm-demo", "serve-x", ["domain__a"], {"b": "1", "a": "X"}, ["a", "b"]),
            ("kernel-demo", "op", [], {"k": "v"}, ["k", "missing"]),
        ]:
            self.assertEqual(engine.benchmark_id(*args), BenchmarkId.derive(*args))

    def test_backend_is_not_identity(self):
        base = cat("postgres-demo")
        d = copy_catalog("postgres-demo", "backend")
        edit(d / "oltp/pgbench-tpcb.yaml", "backend: postgres,", "backend: b,")
        other = engine.Catalog([d], [PG])
        self.assertEqual(ids_by_name(other), ids_by_name(base))
        a, b = base.cases[0], other.cases[0]
        self.assertEqual(
            (base.identity(a)["props"]["system.backend"], other.identity(b)["props"]["system.backend"]),
            ("postgres", "b"),
        )
        self.assertNotIn("system.backend", other.identity(b)["disc"])

    def test_derived_startup_shares_the_decoder_system(self):
        c = cat("llm-demo")
        by = {x["name"]: x for x in c.cases}
        a, b = by["startup_demoa_tp4_len1000_seqs10"], by["serve_demoa_tp4_c5_all"]
        sys_keys = {k: v for k, v in c.identity(a)["disc"].items() if k.startswith("system.")}
        self.assertEqual(sys_keys, {k: v for k, v in c.identity(b)["disc"].items() if k.startswith("system.")})
        self.assertNotEqual(c.ids(a), c.ids(b))

    def test_what_is_a_run_not_a_benchmark(self):
        base = cat("postgres-demo")
        want = ids_by_name(base)
        d = copy_catalog("postgres-demo", "identity")
        f = d / "oltp/pgbench-tpcb.yaml"
        edit(f, "progress_s: 10", "progress_s: 30")  # non_identity field
        edit(f, "run: {repeats: 3,", "env: {PGOPTIONS: -c_jit=off}\n  run: {repeats: 5,")  # knobs and repeats
        edit(f, "  tier: core\n", "  tier: extended\n  environment: pg-oltp-host\n  profile: {default: light}\n")
        edit(f, "backend: postgres, token: default", "backend: postgres-fork, token: default")  # backend unhashed
        self.assertEqual(ids_by_name(engine.Catalog([d], [PG])), want)
        c = engine.Catalog([d], [PG])
        self.assertEqual(c.errors, [])
        self.assertEqual({x["groupings"]["tier"][0] for x in c.cases}, {"extended"})
        for x in c.cases:  # groupings are run props, never tags and never hashed
            self.assertEqual(c.run_props(x), {"run.tier": "extended", "run.suite.oltp": "true"})
            self.assertEqual(c.identity(x)["tags"], [])

    def test_what_is_a_new_benchmark(self):
        base = cat("postgres-demo")
        want = ids_by_name(base)
        d = copy_catalog("postgres-demo", "identity2")
        edit(d / "project.yaml", "params: {scale: 100}", "params: {scale: 200}")  # input asset -> inputs
        c = engine.Catalog([d], [PG])
        self.assertEqual(c.errors, [])
        self.assertTrue(all(ids_by_name(c)[n] != want[n] for n in want))
        changed, _ = val.identity_changes(base, c)
        self.assertEqual(len(changed), 7)
        self.assertIn("a new series starts", changed[0])
        d = copy_catalog("postgres-demo", "identity3")
        edit(d / "oltp/pgbench-tpcb.yaml", "major: 16,", "major: 17,")  # system
        c = engine.Catalog([d], [PG])
        self.assertTrue(all(ids_by_name(c)[n] != want[n] for n in want))
        d = copy_catalog("postgres-demo", "identity4")
        edit(d / "oltp/pgbench-tpcb.yaml", "duration: 5m", "duration: 10m")  # workload
        self.assertTrue(all(ids_by_name(engine.Catalog([d], [PG]))[n] != want[n] for n in want))
        d = copy_catalog("postgres-demo", "identity5")
        edit(d / "oltp/pgbench-tpcb.yaml", "duration: 5m", "duration: 300s")  # same quantity, other unit
        self.assertEqual(ids_by_name(engine.Catalog([d], [PG])), want)

    def test_model_revision_changes_the_benchmark(self):
        base = ids_by_name(cat("llm-demo"))
        d = copy_catalog("llm-demo", "revision")
        edit(
            d / "assets.yaml",
            "revision: 8f2c4a6e1b3d5f708192a3b4c5d6e7f8091a2b3c",
            "revision: 9f2c4a6e1b3d5f708192a3b4c5d6e7f8091a2b3c",
        )
        new = ids_by_name(cat(d))
        changed = {n for n in base if new[n] != base[n]}
        self.assertIn("serve_demoa_tp4_c5_all", changed)
        self.assertNotIn("latency_pinned_demo-llm-c_tp1_in64_out64", changed)
        self.assertNotIn("serve_demo-embed-a_tp1_c64_n200_in512", changed)

    def test_case_name_is_a_label_benchmark_name_is_hashed(self):
        base = cat("llm-demo")
        d = copy_catalog("llm-demo", "rename")
        edit(
            d / "decoder/serve-demo-llm-a.yaml",
            "serve_{system.token}_tp{system.tp}",
            "decode_{system.token}_tp{system.tp}",
        )
        edit(d / "project.yaml", "serve_demoa_tp4_c5_all", "decode_demoa_tp4_c5_all")
        edit(d / "project.yaml", "serve_demoa_tp4_c5_trace-y", "decode_demoa_tp4_c5_trace-y")
        c = cat(d)
        self.assertEqual(c.errors, [])
        self.assertEqual(
            base.ids(next(x for x in base.cases if x["name"] == "serve_demoa_tp4_c5_all")),
            c.ids(next(x for x in c.cases if x["name"] == "decode_demoa_tp4_c5_all")),
        )
        d2 = copy_catalog("llm-demo", "rename2")
        f = d2 / "decoder/serve-demo-llm-a.yaml"
        edit(f, "name: serve-demo-llm-a,", "name: serve-demo-llm-a2,")
        f.rename(d2 / "decoder/serve-demo-llm-a2.yaml")
        fb = d2 / "decoder/serve-demo-llm-a2.yaml"
        fb.write_text(fb.read_text().replace("benchmarks: [serve-demo-llm-a]", "benchmarks: [serve-demo-llm-a2]"))
        for a in ("card-to-card", "cpu-clock-sweep"):
            edit(d2 / "analysis" / f"{a}.yaml", "[serve-demo-llm-a]", "[serve-demo-llm-a2]")
        c2 = cat(d2)
        self.assertEqual(c2.errors, [])
        changed, notes = val.identity_changes(base, c2)
        self.assertTrue(
            any("Benchmark/serve-demo-llm-a removed or renamed without previous_names" in n for n in notes), notes
        )
        self.assertTrue(any("identity changed under the same case name" in w for w in changed))
        edit(
            d2 / "decoder/serve-demo-llm-a2.yaml",
            "  type: perf\n  level: service\n  tier: core\n",
            "  type: perf\n  level: service\n  tier: core\n  previous_names: [serve-demo-llm-a]\n",
        )
        changed, notes = val.identity_changes(base, cat(d2))
        self.assertTrue(any("renamed to serve-demo-llm-a2" in w and "previous_names" in w for w in changed), changed)

    def test_run_attribute_axis_is_not_a_case(self):
        d = copy_catalog("postgres-demo", "runaxis")
        edit(
            d / "oltp/pgbench-tpcb.yaml",
            '      clients: {path: workload.load.concurrency, values: [1, 8, 32, 64], token: "c{}"}\n',
            '      clients: {path: run.repeats, values: [1, 8, 32, 64], token: "c{}"}\n',
        )
        errs = engine.Catalog([d], [PG]).errors
        self.assertTrue(any("same benchmark_id as case" in e and "runs, not cases" in e for e in errs), errs)

    def test_adhoc_variant_and_identity_diff(self):
        c = cat("llm-demo")
        v, diff = c.variant("serve_demoa_tp4_c5_all", ["workload.load.concurrency=8"])
        self.assertTrue(diff["identity_changed"])
        self.assertEqual(diff["changed"], [("case.load.concurrency", "5", "8")])
        v, diff = c.variant("serve_demoa_tp4_c5_all", ["run.repeats=5", "profile.default=full"])
        self.assertFalse(diff["identity_changed"])
        self.assertEqual(diff["changed"], [])

    @unittest.skipUnless(
        (demo.ROOT / "examples").is_dir(),
        "demo.py builds from a source checkout; the installed package has no examples/",
    )
    def test_result_ids_recompute_from_the_result_alone(self):
        self.assertEqual(demo.main(["--check"]), 0)
        results = json.loads((X / "results" / "pg-nightly.results.json").read_text())
        for r in results:
            self.assertEqual(engine.recompute(r), [])
            self.assertEqual(engine.check_aggregate(r), [])
        r = results[0]
        t = copy.deepcopy(r)
        t["status"]["repeats"][1]["samples"]["pg.tps"] = [
            1.0
        ] * 5  # observations are not identity, the aggregate is derived
        self.assertEqual(engine.recompute(t), [])
        self.assertNotEqual(engine.check_aggregate(t), [])
        t["status"]["case"]["disc"]["case.load.concurrency"] = "2"
        t["status"]["environment"]["fingerprint"]["env.fp.kernel"] = "other"
        t["status"]["repeats"][1]["placement"]["nodes"] = ["db-node-09"]
        t["status"]["series"]["labels"]["env.label.node.class"] = "other"
        self.assertEqual(engine.recompute(t), ["benchmark_id", "env_id", "placement repeat 2", "series"])

    def test_placement_per_repeat_is_not_in_the_series_key(self):
        r = json.loads((X / "results" / "pg-nightly.results.json").read_text())[0]["status"]
        pl = [x["placement"]["id"] for x in r["repeats"]]
        self.assertEqual((pl[0] == pl[2], pl[0] != pl[1]), (True, True))
        self.assertEqual(
            r["series"]["labels"], {"env.label.node.class": "db-metal-nvme", "env.label.storage.class": "nvme-local"}
        )
        self.assertNotIn("placement", json.dumps(r["series"]))
        self.assertTrue(r["environment"]["env_id"] != pl[0])

    def test_build_result_refuses_non_fingerprint_attributes(self):
        c = cat("postgres-demo")
        with self.assertRaises(ValueError):
            engine.build_result(c, c.cases[0], "x", {"artifact_id": "a"}, {"env.node": "n1"}, {}, {}, [])


def existing_row(c, case, subject, env_id, **kw):
    return {
        "run_id": "run-1",
        "benchmark_id": c.ids(case)["benchmark_id"],
        "subject_id": subject,
        "env_id": env_id,
        "labels": c.env_labels(c.effective("Environment", case["environment"])),
        "validity": "valid",
        "ts": "2026-11-12T00:00:00Z",
        "repeats": [
            {"run.repeat": i, "placement_id": f"pl-{i}", "nodes": [f"n{i}"], "devices": [f"SN{i}"]} for i in (1, 2, 3)
        ],
        **kw,
    }


class Plan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = cat("llm-demo")
        cls.leg = {"arch": "x86_64", "devices": {"spyre/pf": 8}, "nodes": 1, "reference_node": True, "env_id": "env-1"}

    def test_sanity_plan(self):
        p = engine.plan(self.c, {"purpose": "pr", "selection": {"tier": "sanity"}}, self.leg, now=NOW)
        names = [r["name"] for r in p["cases"]]
        self.assertEqual(names[:3], ["serve_demoa_tp4_c1_all", "serve_demoa_tp4_c5_all", "serve_demoa_tp4_c16_all"])
        self.assertNotIn("parity_demoa_tp1_greedy_n64_out128", names)  # quality is another request
        self.assertTrue(all(r["action"] == "run" for r in p["cases"]))
        rec = p["cases"][1]
        self.assertEqual(
            rec["run_props"],
            {
                "run.tier": "sanity",
                "run.suite.decoder": "true",
                "run.headline": "true",
                "run.purpose": "pr",
                "run.profile_level": "off",
                "run.validity_policy": self.c.digest("ValidityPolicy", "demo-default"),
            },
        )
        self.assertEqual(rec["accelerators"], 8)
        self.assertEqual(rec["inputs"][0]["selection_digest"], "required")  # expect given, digest on_expect
        self.assertTrue(rec["series_key"])

    def test_capacity_reference_budget_errors(self):
        p = engine.plan(
            self.c,
            {"purpose": "nightly", "selection": {"suites": ["decoder", "grid"]}},
            {**self.leg, "devices": {"spyre/pf": 1}},
        )["cases"]
        self.assertEqual({r["reason"] for r in p}, {"capacity"})
        p = engine.plan(
            self.c, {"purpose": "nightly", "selection": {"suites": ["grid"]}}, {**self.leg, "reference_node": False}
        )["cases"]
        self.assertEqual({r["reason"] for r in p}, {"reference_node_required"})
        for req, leg, msg in [
            ({"selection": {"tier": "bogus"}}, self.leg, "tier"),
            ({"selection": {"cases": ["nope"]}}, self.leg, "unknown case"),
            ({"selection": {"suites": ["decoder"]}, "budget": {"device_hours": 1}}, self.leg, "budget"),
            ({"selection": {"suites": ["decoder"]}}, {**self.leg, "arch": "s390x"}, "selection is empty"),
        ]:
            with self.assertRaises(engine.PlanError) as e:
                engine.plan(self.c, {"purpose": "nightly", **req}, leg)
            self.assertIn(msg, str(e.exception))

    def test_selection_repeats_and_profile(self):
        p = engine.plan(self.c, {"purpose": "release", "selection": {"cases": ["release-parity"]}}, self.leg)["cases"]
        self.assertEqual([(r["selection"], r["repeats"]) for r in p], [("release-parity", 5)] * 2)
        p = engine.plan(
            self.c,
            {"selection": {"cases": ["release-parity"]}, "overrides": {"profile_level": "full", "repeats": 2}},
            self.leg,
        )["cases"]
        self.assertEqual({(r["profile_level"], r["repeats"]) for r in p}, {("full", 2)})
        self.assertEqual(
            p[0]["benchmark_id"],
            engine.plan(self.c, {"selection": {"cases": ["release-parity"]}}, self.leg)["cases"][0]["benchmark_id"],
        )
        p = engine.plan(
            self.c, {"selection": {"cases": ["release-parity"]}, "overrides": {"profile_level": "light"}}, self.leg
        )["cases"]
        self.assertEqual({r["reason"] for r in p}, {""})
        p = engine.plan(
            self.c,
            {"selection": {"cases": ["latency_pinned_demo-llm-c_tp1_in64_out64"]}, "environment": "ref-1card"},
            {**self.leg, "devices": {"spyre/pf": 1}},
        )["cases"]
        self.assertEqual(p[0]["accelerators"], 1)

    def test_selection_by_grouping(self):
        names = {
            x["name"] for x in self.c.select(next(d for d in self.c.docs if d.name == "headline").spec["selector"])
        }
        self.assertEqual(len(names), 6)  # the extended trace-x case is above the core ceiling
        self.assertTrue(all(n.startswith("serve_demoa_tp4_c") for n in names))

    def test_reuse(self):
        case = next(x for x in self.c.cases if x["name"] == "serve_demoa_tp4_c5_all")
        req = {"selection": {"cases": ["sanity-with-reuse"]}, "subject": {"artifact_id": "art-1"}}

        def pick(p):
            return next(r for r in p["cases"] if r["name"] == case["name"])

        def rec(rows, request=req, events=(), leg=None):
            return pick(engine.plan(self.c, request, leg or self.leg, rows, events, now=NOW))

        r = rec([existing_row(self.c, case, "art-1", "env-1")])
        self.assertEqual(
            (r["action"], r["reused_from"], r["reason"]), ("reuse", "run-1", "same_subject+same_environment")
        )
        self.assertEqual(r["reuse_key"]["subject_id"], "art-1")
        for row, why in [
            (existing_row(self.c, case, "art-2", "env-1"), "different_subject"),
            (existing_row(self.c, case, "art-1", "env-2"), "different_environment"),
            (existing_row(self.c, case, "art-1", "env-1", validity="suspect"), "validity_below_request"),
        ]:
            r = rec([row])
            self.assertEqual((r["action"], r["reason"]), ("run", why))
        self.assertEqual(rec([])["reason"], "no_result_for_case")
        r = rec([existing_row(self.c, case, "art-1", "env-1")], {**req, "subject": {}})
        self.assertEqual(r["reason"], "reuse_needs_subject")
        # the same result no longer counts once a repeat is invalidated (3 stored, 3 requested)
        ev = [
            {
                "id": "e1",
                "scope_kind": "device",
                "state": "invalid",
                "device": "SN2",
                "from": "2026-11-01T00:00:00Z",
                "reason": "example",
            }
        ]
        r = rec(
            [
                existing_row(
                    self.c,
                    case,
                    "art-1",
                    "env-1",
                    repeats=[
                        {
                            "run.repeat": i,
                            "placement_id": f"p{i}",
                            "nodes": ["n"],
                            "devices": [f"SN{i}"],
                            "ts": "2026-11-12T00:00:00Z",
                        }
                        for i in (1, 2, 3)
                    ],
                )
            ],
            events=ev,
        )
        self.assertEqual(r["reason"], "repeats_below_request")
        # forced rerun and deliberate relaxation to label keys
        orig = [d for d in self.c.docs if d.name == "sanity-with-reuse"][0]
        saved = orig.body["spec"]["reuse"]["force"]
        orig.body["spec"]["reuse"]["force"] = ["all"]
        try:
            self.assertEqual(rec([existing_row(self.c, case, "art-1", "env-1")])["reason"], "forced")
        finally:
            orig.body["spec"]["reuse"]["force"] = saved
        label_req = {**req, "selection": {"cases": ["nightly-reuse-by-label"]}}
        r = rec([existing_row(self.c, case, "art-1", "env-OTHER")], label_req)
        self.assertEqual((r["action"], r["reused_from"]), ("reuse", "run-1"))
        row = existing_row(self.c, case, "art-1", "env-OTHER")
        row["labels"] = {**row["labels"], "env.label.device.count": "4"}
        self.assertEqual(rec([row], label_req)["reason"], "different_environment")
        old = existing_row(self.c, case, "art-1", "env-OTHER", ts="2026-10-01T00:00:00Z")
        self.assertEqual(rec([old], label_req)["reason"], "too_old")

    def test_reuse_mode_only_reports_missing(self):
        req = {"selection": {"suites": ["pinned"]}, "subject": {"artifact_id": "a"}, "reuse": {"mode": "only"}}
        p = engine.plan(self.c, req, {**self.leg, "devices": {"spyre/pf": 1}}, now=NOW)["cases"]
        self.assertEqual([(r["action"], r["reason"]) for r in p], [("missing", "no_result_for_case")])


class Lifecycle(unittest.TestCase):
    def test_invalidation_scopes(self):
        events = yaml.safe_load((X / "tool-output" / "invalidations.yaml").read_text())["events"]
        for e in events:
            engine.check_invalidation(e)
        self.assertEqual(
            [e["id"] for e in engine.active_invalidations(events)], ["inv-1", "inv-2", "inv-5"]
        )  # inv-3 restored
        row = {
            "run_id": "7c1e0000-0000-5000-8000-000000000001",
            "benchmark_id": "4f2c91ab-03d7-5a11-9c20-7d3e8b1a64f0",
            "ts": "2026-11-05T00:00:00Z",
            "repeats": [
                {"run.repeat": 1, "placement_id": "pl-a1", "nodes": ["n1"], "devices": ["serial-a"]},
                {"run.repeat": 2, "placement_id": "pl-a2", "nodes": ["n2"], "devices": ["serial-c"]},
                {"run.repeat": 3, "placement_id": "pl-a3", "nodes": ["n3"], "devices": ["serial-b"]},
            ],
        }
        self.assertEqual(
            engine.excluded_repeats(row, events), {2, 3}
        )  # run_benchmark repeat 2 + placement pl-a2 + device window
        self.assertEqual(
            engine.excluded_repeats(
                {**row, "ts": "2026-11-12T00:00:00Z", "repeats": [{**row["repeats"][2], "ts": "2026-11-12T00:00:00Z"}]},
                events,
            ),
            set(),
        )
        self.assertEqual(
            engine.excluded_repeats({**row, "run_id": "other", "repeats": row["repeats"][:2]}, events[1:2]), set()
        )
        node = {
            "id": "n",
            "scope_kind": "node",
            "state": "suspect",
            "node": "n1",
            "from": "2026-11-01T00:00:00Z",
            "reason": "x",
        }
        self.assertEqual(engine.excluded_repeats(row, [node]), {1})
        whole = {"id": "r", "scope_kind": "run", "state": "invalid", "run_id": row["run_id"], "reason": "x"}
        self.assertEqual(engine.excluded_repeats(row, [whole]), {1, 2, 3})

    def test_invalidation_writer_rules(self):
        base = {
            "scope_kind": "device",
            "state": "invalid",
            "device": "SN1",
            "from": "2026-11-01T00:00:00Z",
            "reason": "x",
        }
        engine.check_invalidation(base)
        for bad, msg in [
            ({**base, "scope_kind": "query", "match": "env.fp.driver=5"}, "reserved"),
            ({k: v for k, v in base.items() if k != "device"}, "needs device"),
            ({**base, "state": "restored"}, "names the invalidation it restores"),
            ({**base, "reason": ""}, "reason"),
            ({**base, "scope_kind": "galaxy"}, "galaxy"),
        ]:
            with self.assertRaises(ValueError) as e:
                engine.check_invalidation(bad)
            self.assertIn(msg, str(e.exception))

    def test_qualification_and_reference_grade(self):
        rows = yaml.safe_load((X / "tool-output" / "qualification.yaml").read_text())["rows"]
        nc = cat("central").by[("NodeClass", "x86-ref-pool")].spec
        self.assertIsNotNone(engine.qualification_current(rows, nc, "host-a", NOW))
        self.assertIsNone(engine.qualification_current(rows, nc, "host-b", NOW))  # cv 3.0 > 2.0
        self.assertIsNone(engine.qualification_current(rows, nc, "host-c", NOW))  # older than 14 days
        self.assertIsNotNone(
            engine.qualification_current(rows, nc, "host-c", datetime(2026, 10, 10, tzinfo=timezone.utc))
        )
        env = {"host": {"reference_node": True, "exclusive": True}}
        clean = {"exclusive": "enforced", "other_load": 0, "throttled": False}
        self.assertEqual(engine.repeat_validity(env, True, clean), "valid")
        self.assertEqual(engine.repeat_validity(env, False, clean), "unqualified_env")
        self.assertEqual(engine.repeat_validity(env, True, {**clean, "exclusive": "none"}), "shared_env")
        self.assertEqual(engine.repeat_validity(env, True, clean, gates_pass=False), "suspect")
        self.assertTrue(
            engine.validity_at_least("valid", "shared_env") and not engine.validity_at_least("suspect", "valid")
        )


class Explain(unittest.TestCase):
    def test_explain(self):
        c = cat("llm-demo")
        a = Namespace(environment=None, subject="art-1", purpose="release", set=[])
        doc = ex.explain(c, "serve_demoa_tp4_c5_all", a)[0]
        self.assertEqual(doc["environment"]["labels"]["env.label.device.count"], "8")
        self.assertEqual(doc["environment"]["accelerators"], 8)
        self.assertEqual(doc["plan"]["action"], "run")
        self.assertEqual(doc["reuse_key"]["subject_id"], "art-1")
        self.assertEqual(doc["identity"]["disc"]["case.load.concurrency"], "5")
        self.assertEqual(doc["identity"]["props"]["input.prompts.subset"], "smoke-200")
        self.assertEqual(len(ex.explain(c, "serve-demo-llm-a", a)), 7)
        a = Namespace(environment="ref-1card", subject=None, purpose="release", set=[])
        doc = ex.explain(c, "serve_demoa_tp4_c5_all", a)[0]
        self.assertEqual((doc["plan"]["environment"], doc["plan"]["accelerators"]), ("ref-1card", 1))
        self.assertEqual(
            doc["series_key"]
            != ex.explain(
                c, "serve_demoa_tp4_c5_all", Namespace(environment=None, subject=None, purpose="investigate", set=[])
            )[0]["series_key"],
            True,
        )

    def test_explain_set(self):
        c = cat("llm-demo")
        a = Namespace(environment=None, subject=None, purpose="investigate", set=["workload.load.concurrency=8"])
        doc = ex.explain(c, "serve_demoa_tp4_c5_all", a)[0]
        self.assertTrue(doc["identity_diff"]["identity_changed"])
        self.assertEqual(doc["identity_diff"]["changed"], [("case.load.concurrency", "5", "8")])


class Environments(unittest.TestCase):
    def test_totals_and_labels(self):
        c = cat("llm-demo")
        v = c.env_view("cpu-2node")
        self.assertEqual((v["nodes"], v["accelerators"], v["device_totals"]), (2, 0, {"cpu/socket": 2, "net/link": 1}))
        v = c.env_view("ref-1card")
        self.assertEqual(
            (v["accelerators"], v["labels"]["env.label.device.count"], v["labels"]["env.label.node.class"]),
            (1, "1", "x86-ref-pool"),
        )
        self.assertTrue(v["reference_node"] and v["exclusive"])
        v = cat("postgres-demo").env_view("pg-oltp-host")
        self.assertEqual((v["accelerators"], v["device_totals"]), (0, {"cpu/socket": 2, "storage/nvme": 1}))

    def test_spec_digest_tracks_the_request(self):
        a = cat("llm-demo").env_view("ref-8card-one-node")["spec_digest"]
        d = copy_catalog("central", "digest")
        edit(d / "environments.yaml", "count: 8, selectors", "count: 8, per_node: true, selectors")
        b = engine.Catalog([d], []).env_view("ref-8card-one-node")["spec_digest"]
        self.assertNotEqual(a, b)

    def test_custom_device_class_needs_its_pack(self):
        d = copy_catalog("central", "custom")
        edit(d / "environments.yaml", "device_class: spyre/pf, count: 8", "device_class: acme/proto-npu, count: 8")
        errs = engine.Catalog([d], []).errors
        self.assertTrue(
            any("device_class 'acme/proto-npu' is not registered by a loaded pack" in e for e in errs), errs
        )

    def test_analyses_resolve(self):
        c = cat("llm-demo")
        self.assertEqual(sorted(d.name for d in c.of("Analysis")), ["card-to-card-spread", "cpu-clock-sweep"])

    def test_decompose_needs_part_of(self):
        d = copy_catalog("llm-demo", "decompose")
        (d / "analysis" / "decompose.yaml").write_text(
            "apiVersion: bench/v1alpha1\nkind: Analysis\nmetadata: {name: step-vs-parts}\nspec:\n"
            "  select: {benchmarks: [serve-demo-llm-a, startup-decoder]}\n  metric: llm.itl.median\n"
            "  vary: [subject.dep.*]\n  decompose: {from: serve-demo-llm-a, to: [startup-decoder], show: residual}\n"
        )
        errs = cat(d).errors
        self.assertTrue(any("not linked by a part_of relation" in e for e in errs), errs)
        edit(
            d / "startup" / "startup-decoder.yaml",
            "  run: {cache: both,",
            "  relations: [{type: part_of, target: serve-demo-llm-a, via: {decompose: sum}}]\n  run: {cache: both,",
        )
        self.assertEqual(cat(d).errors, [])


class UnhashedFieldsAndClean(unittest.TestCase):
    NAME = "serve_demoa_tp4_c5_all"

    def test_system_extension_is_unhashed_but_recoverable(self):
        base = cat("llm-demo")
        d = copy_catalog("llm-demo", "ext")
        edit(d / SI_DEC, "mode: a", "mode: b")
        other = cat(d)
        a, b = (next(x for x in c.cases if x["name"] == self.NAME) for c in (base, other))
        self.assertEqual(base.ids(a), other.ids(b))
        self.assertEqual(base.identity(a)["props"]["system.ext.spyre.mode"], "a")
        self.assertEqual(other.identity(b)["props"]["system.ext.spyre.mode"], "b")
        self.assertNotIn("system.ext.spyre.mode", base.identity(a)["disc"])
        self.assertEqual(base.reg.lookup("system.ext.spyre.mode")["class"], "run")
        diff = engine.identity_diff(base.identity(a), other.identity(b))
        self.assertEqual(diff["props_changed"], [("system.ext.spyre.mode", "a", "b")])
        self.assertFalse(diff["identity_changed"] and diff["changed"])
        leg = {"arch": "x86_64", "devices": {"spyre/pf": 8}, "nodes": 1, "reference_node": True}
        rec = engine.plan(other, {"selection": {"cases": [self.NAME]}}, leg)["cases"][0]
        self.assertEqual(rec["case_props"]["system.ext.spyre.mode"], "b")
        doc = ex.explain(other, self.NAME, Namespace(environment=None, subject=None, purpose="investigate", set=[]))[0]
        self.assertEqual(doc["identity"]["props"]["system.ext.spyre.mode"], "b")
        v, vd = base.variant(self.NAME, ["system={base: demo-llm-a-tp4-fp16, extensions: {spyre: {mode: b}}}"])
        self.assertFalse(vd["identity_changed"])
        self.assertEqual(vd["props_changed"][0][0], "system.ext.spyre.mode")

    def test_result_carries_props(self):
        c = cat("llm-demo")
        case = next(x for x in c.cases if x["name"] == self.NAME)
        fp = {"env.fp.kernel": "6.8"}
        r = engine.build_result(
            c,
            case,
            "r",
            {"artifact_id": "a"},
            fp,
            {},
            {},
            [{"run.repeat": 1, "placement": {"nodes": ["n"]}, "samples": {"llm.ttft.p99": [1.0]}}],
        )
        self.assertEqual(r["status"]["case"]["props"]["system.ext.spyre.mode"], "a")
        self.assertEqual(engine.shape_schema("result").errors(r), [])
        self.assertEqual(engine.recompute(r), [])

    def test_unregistered_unhashed_field_is_rejected(self):
        d = copy_catalog("llm-demo", "ext-unreg")
        edit(d / SI_DEC, "mode: a", "mode: a, tag: a=1")
        reg_file = PKG / "packs" / "spyre" / "pack.yaml"
        self.assertIn("system.ext.spyre.tag", reg_file.read_text())  # registered: accepted
        self.assertEqual(cat(d).errors, [])
        edit(d / SI_DEC, "tag: a=1", "tag: b")
        self.assertEqual(cat(d).errors, [])

    def test_clean_environment_view_and_examples(self):
        c = cat("llm-demo")
        ref = c.env_view("ref-8card-reference-clean")
        self.assertTrue(ref["clean"]["requested"])
        self.assertEqual(ref["clean"]["expected_exclusive"], "enforced")
        self.assertEqual(ref["clean"]["held_node_hours"]["upper_bound"], 7.0)
        self.assertIsNone(ref["clean"]["held_node_hours"]["expected"])
        pr = c.env_view("ref-8card-pr")
        self.assertEqual(pr["clean"], {"requested": False, "expected_exclusive": "observed"})
        self.assertFalse(pr["reference_node"] or pr["exclusive"])
        doc = ex.explain(
            c,
            self.NAME,
            Namespace(environment="ref-8card-reference-clean", subject=None, purpose="reference", set=[]),
        )[0]
        self.assertTrue(doc["environment"]["clean"]["requested"])
        self.assertEqual(doc["plan"]["clean"]["expected_exclusive"], "enforced")
        for k in ("access", "wait_s", "hold_s", "other_load", "exclusive"):
            self.assertIsNotNone(c.reg.lookup(f"run.clean.{k}"), k)
        self.assertEqual(c.reg.lookup("run.clean.exclusive")["values"], ["enforced", "observed", "none", "unknown"])

    def test_clean_planning_never_runs_unclean_while_claiming_clean(self):
        c = cat("llm-demo")
        base = {"arch": "x86_64", "devices": {"spyre/pf": 8}, "nodes": 1, "reference_node": True}
        req = {
            "environment": "ref-8card-reference-clean",
            "selection": {"cases": [self.NAME]},
            "overrides": {"repeats": 3},
        }

        def plan_with(clean, request=req):
            return engine.plan(c, request, {**base, **({"clean": clean} if clean is not None else {})})["cases"][0]

        r = plan_with({"expected_wait_s": 600})
        self.assertEqual((r["action"], r["reason"]), ("run", ""))
        self.assertEqual(r["clean"]["expected_exclusive"], "enforced")
        r = plan_with(None)
        self.assertEqual((r["action"], r["reason"]), ("skip", "capacity"))  # no exclusive access available
        r = plan_with({"expected_wait_s": 5 * 3600})
        self.assertEqual((r["action"], r["reason"]), ("skip", "capacity_timeout"))  # past max_wait 4h
        r = engine.plan(c, {**req, "overrides": {"repeats": 3}}, {**base, "clean": {"expected_wait_s": 0}})["cases"][0]
        self.assertEqual(r["reason"], "")  # 3 x 40m = 2h <= 3h hold
        r = engine.plan(c, {**req, "overrides": {"repeats": 5}}, {**base, "clean": {"expected_wait_s": 0}})["cases"][0]
        self.assertEqual(r["reason"], "max_hold_too_short")  # 5 x 40m > 3h
        pr = engine.plan(c, {**req, "environment": "ref-8card-pr"}, {**base, "reference_node": False})["cases"][0]
        self.assertEqual((pr["action"], pr["clean"]), ("run", {"requested": False, "expected_exclusive": "observed"}))

    def test_run_clean_levels(self):
        import itertools

        for excl, load, cards in itertools.product(
            ["enforced", "observed", "none", "unknown"], [True, False, None], ["enforced", "none", "unknown"]
        ):
            want = excl == "enforced" or (excl == "observed" and load is True and cards == "enforced")
            self.assertEqual(
                engine.run_clean({"exclusive": excl, "load_clean": load, "devices_exclusive": cards}),
                want,
                (excl, load, cards),
            )
        self.assertFalse(engine.run_clean({}))
        self.assertFalse(
            engine.run_clean({"exclusive": "declared", "load_clean": True, "devices_exclusive": "enforced"})
        )
        for lvl in ("none", "unknown"):
            self.assertFalse(engine.run_clean({"exclusive": lvl, "load_clean": True, "devices_exclusive": "enforced"}))
        env = {"host": {"reference_node": True}}
        obs = {"exclusive": "observed", "load_clean": True, "devices_exclusive": "enforced"}
        self.assertEqual(engine.repeat_validity(env, True, obs), "valid")
        self.assertEqual(engine.repeat_validity(env, True, {**obs, "devices_exclusive": "unknown"}), "shared_env")
        self.assertEqual(engine.repeat_validity(env, False, obs), "unqualified_env")

    def test_cleanliness_never_enters_env_id(self):
        c = cat("llm-demo")
        case = next(x for x in c.cases if x["name"] == self.NAME)
        fp = {"env.fp.kernel": "6.8", "env.fp.cpu.base_ghz": 2.4}

        def rep(clean):
            return [
                {"run.repeat": 1, "placement": {"nodes": ["n1"]}, "clean": clean, "samples": {"llm.ttft.p99": [1.0]}}
            ]

        exclusive_run = engine.build_result(
            c,
            case,
            "a",
            {"artifact_id": "a"},
            fp,
            {},
            {},
            rep(
                {
                    "exclusive": "enforced",
                    "access": "A-1",
                    "wait_s": 900.0,
                    "hold_s": 1200.0,
                    "other_load": "0/0",
                    "devices_exclusive": "enforced",
                }
            ),
        )
        plain = engine.build_result(
            c,
            case,
            "b",
            {"artifact_id": "a"},
            fp,
            {},
            {},
            rep(
                {"exclusive": "observed", "load_clean": True, "devices_exclusive": "enforced", "other_load": "unknown"}
            ),
        )
        self.assertEqual(exclusive_run["status"]["environment"]["env_id"], plain["status"]["environment"]["env_id"])
        self.assertEqual(
            exclusive_run["status"]["repeats"][0]["placement"]["id"], plain["status"]["repeats"][0]["placement"]["id"]
        )
        for key in (
            "run.clean.access",
            "run.clean.wait_s",
            "run.clean.hold_s",
            "run.clean.other_load",
            "run.clean.devices_exclusive",
            "run.clean.load_clean",
        ):
            self.assertIsNotNone(c.reg.lookup(key), key)
            self.assertFalse(c.reg.lookup(key).get("fingerprint"), key)

    def test_recompute_checks_case_props(self):
        c = cat("llm-demo")
        case = next(x for x in c.cases if x["name"] == self.NAME)
        r = engine.build_result(
            c,
            case,
            "r",
            {"artifact_id": "a"},
            {"env.fp.kernel": "6.8"},
            {},
            {},
            [{"run.repeat": 1, "placement": {"nodes": ["n"]}, "samples": {"llm.ttft.p99": [1.0]}}],
        )
        self.assertEqual(engine.recompute(r, c.reg), [])
        t = copy.deepcopy(r)
        t["status"]["case"]["props"]["system.ext.spyre.mode"] = "turbo"
        t["status"]["case"]["props"]["system.ext.spyre.nonsense"] = "x"
        t["status"]["case"]["props"]["system.tp"] = "8"
        t["status"]["case"]["disc"].setdefault("system.tp", "8")
        self.assertEqual(
            sorted(engine.recompute(t, c.reg)),
            [
                "props overlap system.tp",
                "props system.ext.spyre.mode",
                "props system.ext.spyre.nonsense",
            ],
        )


class ValidityAndVerdicts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = cat("llm-demo")
        cls.pol = cls.c.policy.spec
        cls.exp = cls.c.by[("Expectation", "itl-no-regression")].spec
        cls.good = {
            "qualified": True,
            "clean": {"exclusive": "enforced"},
            "repeats": 5,
            "cv_pct": 1.0,
            "warmup_discarded": 2,
            "placement": True,
        }

    def test_policy_loaded_and_registered_rows(self):
        self.assertEqual(self.c.policy.name, "demo-default")
        self.assertEqual(
            sorted(self.pol["purposes"]),
            ["investigate", "nightly", "pr", "qualification", "reference", "release", "weekly"],
        )
        rows = {(r["kind"], r["name"]): r["spec_digest"] for r in self.c.registered_rows()}
        for k in (
            ("ValidityPolicy", "demo-default"),
            ("Expectation", "itl-no-regression"),
            ("Environment", "ref-1card"),
            ("NodeClass", "x86-ref-pool"),
        ):
            self.assertTrue(rows[k])
        d = copy_catalog("llm-demo", "digest-exp")
        edit(d / SI_DEC, "tolerance: {pct: 5}", "tolerance: {pct: 6}")
        self.assertNotEqual(
            cat(d).digest("Expectation", "itl-no-regression"), rows[("Expectation", "itl-no-regression")]
        )
        self.assertEqual(cat(d).digest("ValidityPolicy", "demo-default"), rows[("ValidityPolicy", "demo-default")])

    def test_each_gate_caps_the_validity(self):
        def v(purpose, **kw):
            return engine.validity_of(self.pol, purpose, {**self.good, **kw})

        self.assertEqual(v("reference"), ("valid", []))
        self.assertEqual(v("reference", qualified=False)[0], "unqualified_env")
        self.assertEqual(
            v("reference", clean={"exclusive": "observed", "load_clean": True, "devices_exclusive": "enforced"})[0],
            "shared_env",
        )
        self.assertEqual(
            v("release", clean={"exclusive": "observed", "load_clean": True, "devices_exclusive": "enforced"}),
            ("valid", []),
        )
        self.assertEqual(v("release", clean={"exclusive": "none"})[0], "shared_env")
        self.assertEqual(v("nightly", clean={"exclusive": "none"}), ("valid", []))  # nightly does not need clean
        self.assertEqual(v("nightly", repeats=2), ("suspect", ["repeats_below_min"]))
        self.assertEqual(v("nightly", cv_pct=7.0), ("suspect", ["cv_above_max"]))
        self.assertEqual(v("nightly", cv_pct=None)[0], "suspect")
        self.assertEqual(v("nightly", warmup_discarded=0), ("suspect", ["warmup_not_discarded"]))
        self.assertEqual(v("nightly", placement=False), ("suspect", ["placement_missing"]))
        self.assertEqual(v("nightly", gates_pass=False), ("suspect", ["gate_failed"]))
        self.assertEqual(v("nightly", measured=False)[0], "invalid")
        state, reasons = v("nightly", qualified=False, repeats=1)  # the worst cap wins, every reason is listed
        self.assertEqual((state, sorted(reasons)), ("suspect", ["qualification_not_current", "repeats_below_min"]))
        self.assertEqual(v("pr", qualified=False, repeats=1, placement=False, cv_pct=3.0), ("valid", []))

    def test_floor_is_tightened_never_loosened(self):
        pol = self.pol
        self.assertEqual(engine.effective_floor(pol, "pr", None), "shared_env")
        self.assertEqual(engine.effective_floor(pol, "pr", {"policy": {"validity_floor": "valid"}}), "valid")
        self.assertEqual(engine.effective_floor(pol, "release", {"policy": {"validity_floor": "suspect"}}), "valid")
        self.assertEqual(engine.effective_floor(None, None, None), "valid")

    def test_external_never_reference_grade_or_baseline(self):
        state, reasons = engine.validity_of(self.pol, "reference", self.good, trust="external")
        self.assertEqual((state, reasons), ("external", ["trust_external"]))
        self.assertFalse(engine.reference_grade(self.pol, "reference", state, "external"))
        self.assertFalse(engine.reference_grade(self.pol, "reference", "valid", "external"))  # even claiming valid
        self.assertTrue(engine.reference_grade(self.pol, "reference", "valid"))
        self.assertTrue(engine.reference_grade(self.pol, "release", "valid"))
        self.assertFalse(engine.reference_grade(self.pol, "nightly", "valid"))
        self.assertFalse(
            engine.baseline_eligible(self.pol, {"purpose": "reference", "validity": "valid", "trust": "external"})
        )
        for purpose in self.pol["purposes"]:
            self.assertFalse(engine.reference_grade(self.pol, purpose, "external"), purpose)

    def test_invalid_suspect_and_pr_runs_never_become_baselines(self):
        ok = {"purpose": "nightly", "validity": "valid", "trust": "internal"}
        self.assertTrue(engine.baseline_eligible(self.pol, ok))
        for bad in ("invalid", "suspect", "shared_env", "unqualified_env", "external"):
            self.assertFalse(engine.baseline_eligible(self.pol, {**ok, "validity": bad}), bad)
        for purpose in ("pr", "investigate", "qualification"):
            self.assertFalse(
                engine.baseline_eligible(self.pol, {**ok, "purpose": purpose}), purpose
            )  # valid, but the purpose forbids it
        self.assertFalse(engine.baseline_eligible(self.pol, {**ok, "purpose": "unknown"}))
        rows = [{"run_id": "r1", "repeats": [{"run.repeat": 1, "placement_id": "p"}]}]
        ev = [{"id": "e", "scope_kind": "run", "state": "invalid", "run_id": "r1", "reason": "x"}]
        self.assertTrue(engine.baseline_eligible(self.pol, {**ok, **rows[0]}))
        self.assertFalse(engine.baseline_eligible(self.pol, {**ok, **rows[0]}, ev))

    def test_select_baseline(self):
        cur = {
            "run_id": "now",
            "benchmark_id": "b",
            "ts": "2026-11-13T00:00:00Z",
            "env_id": "e1",
            "labels": {"env.label.device.count": "8"},
        }

        def row(i, **kw):
            return {
                "run_id": f"r{i}",
                "benchmark_id": "b",
                "ts": f"2026-11-{i:02d}T00:00:00Z",
                "validity": "valid",
                "purpose": "nightly",
                "env_id": "e1",
                "labels": {"env.label.device.count": "8"},
                **kw,
            }

        rows = [
            row(1),
            row(2),
            row(3, validity="suspect"),
            row(4, purpose="pr"),
            row(5, env_id="e2"),
            row(6, trust="external"),
            row(7, benchmark_id="other"),
            row(8),
            {**row(9), "run_id": "now"},
            row(20),
        ]
        exp = {"baseline": {"kind": "rolling", "window": 2, "purposes": ["nightly", "release"]}}
        got = engine.select_baseline(self.pol, exp, rows, cur)
        self.assertEqual([r["run_id"] for r in got], ["r8", "r2"])  # newest first, window 2, none of the excluded
        exp["baseline"]["window"] = 20
        self.assertEqual([r["run_id"] for r in engine.select_baseline(self.pol, exp, rows, cur)], ["r8", "r2", "r1"])
        exp["baseline"]["match"] = {"environment": {"labels": ["env.label.device.count"]}}
        self.assertIn(
            "r5", [r["run_id"] for r in engine.select_baseline(self.pol, exp, rows, cur)]
        )  # env_id relaxed to labels

    def test_judge(self):
        exp, dg = self.exp, ("e" * 32, "v" * 32)
        base = [100.0] * 10

        def j(cur, base=base, **kw):
            return engine.judge(exp, *dg, "lower_is_better", cur, base, **kw)

        self.assertEqual(j([106, 106, 106])["status"], "regressed")
        self.assertEqual(j([106, 106, 106])["worse_pct"], 6.0)
        r = j([106])
        self.assertEqual((r["status"], r["reason"]), ("suspected", "single_run"))  # a single run is capped
        self.assertEqual(j([103, 103])["status"], "pass")
        self.assertEqual(j([90, 90])["status"], "improved")
        self.assertEqual(j([90])["status"], "pass")
        self.assertEqual(j([106, 106], base[:9])["status"], "no_baseline")  # min_n is 10
        r = engine.judge(exp, *dg, "higher_is_better", [90, 90], base)
        self.assertEqual((r["status"], r["delta_pct"], r["worse_pct"]), ("regressed", -10.0, 10.0))
        noisy = [100, 120, 80, 110, 90, 105, 95, 115, 85, 100]
        r = j([108, 108], noisy)
        self.assertGreater(r["floor_pct"], 5.0)  # the baseline's own MAD widens the floor
        self.assertEqual(r["status"], "pass")
        r = j([200, 200], validity="suspect", floor="shared_env")
        self.assertEqual((r["status"], r["reason"]), ("suspected", "validity_suspect_below_shared_env"))
        r = j([106, 106])
        self.assertEqual(r["policy_version"], f"{'e' * 12}.{'v' * 12}")
        self.assertEqual(
            (r["expectation_digest"], r["validity_digest"], r["severity"], r["baseline_kind"]),
            ("e" * 32, "v" * 32, "gating", "rolling"),
        )
        fixed = {
            "metric": "llm.goodput_at_slo",
            "baseline": {"kind": "fixed", "value": 20.0},
            "tolerance": {"pct": 10},
            "severity": "advisory",
        }
        r = engine.judge(fixed, *dg, "higher_is_better", [10.0, 10.0], [])
        self.assertEqual(
            (r["baseline_kind"], r["baseline_value"], r["n_baseline"], r["status"]), ("fixed", 20.0, 1, "regressed")
        )
        self.assertIn(r["status"], engine.VERDICTS)
        r = engine.judge(
            {**exp, "noise": {"sigmas": 2, "source": "series_cv"}, "tolerance": {"pct": 1}},
            *dg,
            "lower_is_better",
            [104, 104],
            base,
            noise_cv_pct=3.0,
        )
        self.assertEqual((r["floor_pct"], r["status"]), (6.0, "pass"))

    def test_expectations_apply_to_cases_and_explain_shows_them(self):
        c = self.c
        case = next(x for x in c.cases if x["name"] == "serve_demoa_tp4_c5_all")
        self.assertEqual(sorted(d.name for d in c.expectations_for(case)), ["itl-no-regression", "ttft-p99-advisory"])
        tracex = next(
            x for x in c.cases if x["name"] == "serve_demoa_tp4_c5_trace-x"
        )  # tier extended: above the advisory ceiling
        self.assertEqual([d.name for d in c.expectations_for(tracex)], ["itl-no-regression"])
        doc = ex.explain(c, case["name"], Namespace(environment=None, subject=None, purpose="release", set=[]))[0][
            "validity"
        ]
        self.assertEqual(doc["policy"]["floor"], "valid")
        self.assertEqual(doc["policy"]["requires"]["clean"], "clean")
        self.assertTrue(doc["policy"]["reference"])
        self.assertEqual([e["name"] for e in doc["expectations"]], ["itl-no-regression", "ttft-p99-advisory"])
        self.assertEqual(doc["expectations"][0]["severity"], "gating")
        doc = ex.explain(c, case["name"], Namespace(environment="ref-8card-pr", subject=None, purpose="pr", set=[]))[0][
            "validity"
        ]
        self.assertEqual((doc["policy"]["floor"], doc["policy"]["baseline"]), ("shared_env", False))

    def test_plan_records_the_policy(self):
        leg = {"arch": "x86_64", "devices": {"spyre/pf": 8}, "nodes": 1, "reference_node": True, "env_id": "e"}
        p = engine.plan(self.c, {"purpose": "reference", "selection": {"cases": ["serve_demoa_tp4_c5_all"]}}, leg)[
            "cases"
        ][0]
        dg = self.c.digest("ValidityPolicy", "demo-default")
        self.assertEqual(p["run_props"]["run.validity_policy"], dg)
        self.assertEqual((p["validity"]["policy"]["floor"], p["validity"]["policy"]["reference"]), ("valid", True))
        with self.assertRaises(engine.PlanError) as e:
            engine.plan(self.c, {"purpose": "pr-check", "selection": {"cases": ["serve_demoa_tp4_c5_all"]}}, leg)
        self.assertIn("is not in ValidityPolicy/demo-default", str(e.exception))
        # the PR environment's shared_env floor cannot loosen the policy's: reuse of a suspect result is refused
        req = {
            "purpose": "nightly",
            "environment": "ref-8card-pr",
            "selection": {"cases": ["sanity-with-reuse"]},
            "subject": {"artifact_id": "a"},
        }
        case = next(x for x in self.c.cases if x["name"] == "serve_demoa_tp4_c5_all")
        row = existing_row(self.c, case, "a", "e")
        row["validity"] = "unqualified_env"
        r = next(
            x
            for x in engine.plan(self.c, req, {**leg, "reference_node": False}, [row], now=NOW)["cases"]
            if x["name"] == case["name"]
        )
        self.assertEqual(r["reason"], "validity_below_request")

    def test_result_stores_validity_and_policy_digest(self):
        r = json.loads((X / "results" / "pg-nightly.results.json").read_text())[0]["status"]
        pg = cat("postgres-demo")
        self.assertEqual(r["validity"]["state"], "valid")
        self.assertEqual(r["validity"]["policy_digest"], pg.digest("ValidityPolicy", "pg-default"))
        self.assertEqual(
            r["run_props"],
            {
                "run.validity": "valid",
                "run.validity_policy": r["validity"]["policy_digest"],
                "run.purpose": "nightly",
                "run.trust": "internal",
            },
        )
        for k in ("run.validity", "run.validity_policy", "run.trust"):
            self.assertIsNotNone(pg.reg.lookup(k), k)
        self.assertEqual(
            engine.validity_of(
                pg.policy.spec,
                "nightly",
                {"qualified": True, "clean": {}, "repeats": 3, "cv_pct": 0.1, "warmup_discarded": 1, "placement": True},
            ),
            ("valid", []),
        )
        state, reasons = engine.validity_of(
            pg.policy.spec,
            "nightly",
            {"qualified": True, "clean": {}, "repeats": 3, "cv_pct": 0.1, "warmup_discarded": 1, "placement": True},
            trust="external",
        )
        self.assertEqual(state, "external")


SI_DEC = "decoder/serve-demo-llm-a.yaml"
CENTRAL_ENV = "environments.yaml"
NEG = [  # (case, catalog, file, old, new, expected message fragment)
    (
        "type typo",
        "llm-demo",
        "workloads.yaml",
        "type: llm/serve\n  endpoint",
        "type: llm/serv\n  endpoint",
        "Workload type 'llm/serv' is not registered (did you mean 'llm/serve'?)",
    ),
    (
        "unloaded pack",
        "llm-demo",
        "pinned/latency-pinned.yaml",
        "type: llm/latency",
        "type: pg/pgbench",
        "no pack with prefix 'pg' is loaded",
    ),
    (
        "typed field typo",
        "llm-demo",
        SI_DEC,
        "  precision: fp16",
        "  precison: fp16",
        "unknown key 'precison' (did you mean 'precision'?)",
    ),
    (
        "unknown extension",
        "llm-demo",
        SI_DEC,
        "spyre: {mode: a",
        "other: {mode: a",
        "no loaded pack extends System with prefix 'other'",
    ),
    ("extension value", "llm-demo", SI_DEC, "mode: a", "mode: zz", "is not one of"),
    (
        "metric typo",
        "llm-demo",
        SI_DEC,
        "llm.ttft.p99,",
        "llm.ttft.p95,",
        "metric 'llm.ttft.p95' is not registered (did you mean 'llm.ttft.p99'?)",
    ),
    (
        "metric category",
        "llm-demo",
        SI_DEC,
        "llm.ttft.p99,",
        "llm.eval.exact_match,",
        "category quality, not valid for type perf",
    ),
    (
        "headline outside required",
        "llm-demo",
        SI_DEC,
        "headline: [llm.itl.median]",
        "headline: [llm.ttft.mean]",
        "headline metric llm.ttft.mean is not in required",
    ),
    ("level", "llm-demo", SI_DEC, "level: service", "level: kernel", "is not registered (known:"),
    ("level of type", "llm-demo", SI_DEC, "level: service", "level: model", "not a level of workload type llm/serve"),
    (
        "status",
        "llm-demo",
        "pinned/latency-pinned.yaml",
        "    - name: latency_pinned_demo-llm-c_tp1_in64_out64",
        "    - name: latency_pinned_demo-llm-c_tp1_in64_out64\nstatus: {phase: Planned}",
        "tool-written state is never authored",
    ),
    (
        "tool kind",
        "llm-demo",
        "analysis/card-to-card.yaml",
        "noise: series_floor, chart: dots-per-vary-key}",
        "noise: series_floor, chart: dots-per-vary-key}\n---\napiVersion: bench/v1alpha1\nkind: Result\nmetadata: {name: r}\nspec: {}",
        "tool output",
    ),
    ("anchors", "llm-demo", "pinned/latency-pinned.yaml", "run: {cache", "run: &r {cache", "anchors and aliases"),
    ("name collision", "llm-demo", SI_DEC, "_{concurrency}_{data}", "_{data}", "also produced by"),
    (
        "axis conflict",
        "llm-demo",
        SI_DEC,
        "patch: {run: {est: 25m}, workload: {inputs: [{name: prompts, dataset: trace-y, subset: smoke-200}]}}",
        "patch: {run: {est: 25m}, workload: {load: {concurrency: 2}, inputs: [{name: prompts, dataset: trace-y, subset: smoke-200}]}}",
        "both set $.workload.load.concurrency",
    ),
    (
        "selection typo",
        "llm-demo",
        "project.yaml",
        "serve_demoa_tp4_c5_trace-y]",
        "serve_demoa_tp4_c5_trace-yx]",
        "is not a case (did you mean 'serve_demoa_tp4_c5_trace-y'?)",
    ),
    (
        "base cycle",
        "llm-demo",
        "encoder/serve-demo-embed-a.yaml",
        "metadata: {name: demo-embed-a-tp1-fp16}\nspec:\n  type: llm/vllm",
        "metadata: {name: demo-embed-a-tp1-fp16}\nspec:\n  base: demo-embed-a-tp1-fp16-pc\n  type: llm/vllm",
        "cyclic",
    ),
    (
        "base siblings",
        "llm-demo",
        "encoder/serve-demo-embed-a.yaml",
        "spec:\n  base: demo-embed-a-tp1-fp16\n  patch: {flags: {enable-prefix-caching: false}}",
        "spec:\n  base: demo-embed-a-tp1-fp16\n  precision: bf16\n  patch: {flags: {enable-prefix-caching: false}}",
        "with base, changes go under patch",
    ),
    (
        "suite type",
        "llm-demo",
        SI_DEC,
        "  type: perf\n  level: service\n  tier: core\n",
        "  type: quality\n  level: service\n  tier: core\n",
        "!= Suite/decoder type 'perf'",
    ),
    (
        "file name",
        "llm-demo",
        SI_DEC,
        "name: serve-demo-llm-a,",
        "name: serve-demo-llm-a2,",
        "file must be named serve-demo-llm-a2.yaml",
    ),
    (
        "numeric target",
        "llm-demo",
        "workloads.yaml",
        "disable-shuffle: true}",
        "disable-shuffle: true, threshold: 2}",
        "numeric targets do not belong in a Workload",
    ),
    (
        "grouping value",
        "llm-demo",
        SI_DEC,
        "  tier: core\n  groupings",
        "  tier: gold\n  groupings",
        "tier: 'gold' is not one of",
    ),
    (
        "grouping undeclared",
        "llm-demo",
        SI_DEC,
        "groupings: {headline: true}",
        "groupings: {headliner: true}",
        "grouping 'headliner' is not declared",
    ),
    (
        "series label",
        "llm-demo",
        "project.yaml",
        "env.label.node.class]",
        "env.label.node.class, env.label.rack]",
        "lacks series labels ['env.label.rack']",
    ),
    (
        "profile default",
        "llm-demo",
        SI_DEC,
        'supported: ["off", light, full]',
        'supported: ["off", light]',
        "profile.on_regression 'full' is not in supported",
    ),
    ("yaml 1.1 bool", "llm-demo", SI_DEC, 'default: "off"', "default: off", "YAML 1.1 reads off/on/yes/no as booleans"),
    (
        "pack version",
        "llm-demo",
        "project.yaml",
        '{name: llm-serving, version: "0.2"}',
        '{name: llm-serving, version: "0.3"}',
        "does not satisfy 0.3",
    ),
    ("pg typed", "postgres-demo", "oltp/pgbench-tpcb.yaml", "script: tpcb-like", "script: tpcc", "is not one of"),
    (
        "load needs concurrency",
        "llm-demo",
        "workloads.yaml",
        "load: {executor: closed, concurrency: 5, requests: 200, warmup_requests: 8}",
        "load: {executor: closed, requests: 200}",
        "load.executor closed needs concurrency",
    ),
    (
        "load field mismatch",
        "llm-demo",
        "workloads.yaml",
        "load: {executor: open, rate: inf, requests: 500}",
        "load: {executor: open, rate: inf, requests: 500, concurrency: 4}",
        "load.concurrency does not apply to executor open",
    ),
    (
        "load not accepted",
        "llm-demo",
        "startup/startup-decoder.yaml",
        "cache: [cold, warm]}",
        "cache: [cold, warm], load: {executor: batch, requests: 1}}",
        "takes no load block",
    ),
    (
        "scenario unknown",
        "llm-demo",
        "workloads.yaml",
        "scenario: server",
        "scenario: burst",
        "scenario 'burst' is not a preset",
    ),
    (
        "slo metric",
        "llm-demo",
        "workloads.yaml",
        "slo: {llm.ttft.p99: {lte: 2.0s}, llm.tpot.p99: {lte: 50ms}}",
        "slo: {llm.ttft.p95: {lte: 2.0s}}",
        "slo.llm.ttft.p95: metric is not registered",
    ),
    (
        "input refs",
        "llm-demo",
        "workloads.yaml",
        "inputs: [{name: prompts, dataset: random-isl-osl}]",
        "inputs: [{name: prompts, dataset: random-isl-osl, asset: random-prompts-gen}]",
        "needs exactly one of dataset, asset, model",
    ),
    (
        "subset unknown",
        "llm-demo",
        SI_DEC,
        "subset: smoke-200, expect",
        "subset: smoke-2000, expect",
        "is not a named subset",
    ),
    (
        "select seed",
        "llm-demo",
        "datasets.yaml",
        "stratified: {sample: {n: 1000, seed: 0, stratify_by: bucket}}",
        "stratified: {order: {shuffle: {}}}",
        "seed",
    ),
    (
        "select sample+limit",
        "llm-demo",
        "datasets.yaml",
        "stratified: {sample: {n: 1000, seed: 0, stratify_by: bucket}}",
        "stratified: {limit: 5, sample: {n: 1000, seed: 0}}",
        "sample excludes limit/offset",
    ),
    (
        "asset visibility",
        "llm-demo",
        "assets.yaml",
        'version: "1"\n  license: Apache-2.0\n  visibility: internal\n  source: {store: demo-cache',
        'version: "1"\n  license: Apache-2.0\n  visibility: public\n  source: {store: demo-cache',
        "a public Asset cannot be served from the internal Store",
    ),
    (
        "asset pin",
        "llm-demo",
        "assets.yaml",
        "revision: 8f2c4a6e1b3d5f708192a3b4c5d6e7f8091a2b3c",
        "revision: main",
        "revision_pinned needs source.revision as a 40-hex commit",
    ),
    (
        "asset integrity",
        "llm-demo",
        "assets.yaml",
        "  integrity: {revision_pinned: true}\n  access:",
        "  access:",
        "integrity needs files, a manifest or revision_pinned",
    ),
    (
        "model asset",
        "llm-demo",
        "models.yaml",
        "asset: demo-embed-a-weights",
        "asset: demo-embed-a-wights",
        "no Asset of that name",
    ),
    (
        "system model ref",
        "llm-demo",
        SI_DEC,
        "  model: demo-llm-a\n",
        "  model: demo-llm-8c\n",
        "no Model of that name",
    ),
    (
        "generated needs producer",
        "llm-demo",
        "assets.yaml",
        "produced_by: {benchmark: reference-outputs-demo-llm-a,",
        "produced_by: {benchmark: reference-outputs-demo-llm-8c,",
        "no Benchmark of that name",
    ),
    (
        "compares_to",
        "llm-demo",
        SI_DEC,
        "{type: see_also, target: serve-grid-demo-llm-a}",
        "{type: compares_to, target: serve-grid-demo-llm-a}",
        "compares_to is not a relation",
    ),
    (
        "relation target",
        "llm-demo",
        "quality/parity-demo-llm-a.yaml",
        "{type: depends_on, target: reference-outputs-demo-llm-a}",
        "{type: depends_on, target: reference-outputs-demo-llm-8c}",
        "no Benchmark of that name",
    ),
    (
        "depends_on cycle",
        "llm-demo",
        "quality/reference-outputs-demo-llm-a.yaml",
        "{type: reference_for, target: parity-demo-llm-a}",
        "{type: depends_on, target: parity-demo-llm-a}",
        "depends_on cycle",
    ),
    (
        "analysis attribute",
        "llm-demo",
        "analysis/card-to-card.yaml",
        '"env.placement_id"',
        '"env.placement_ib"',
        "matches no registered attribute (did you mean 'env.placement_id'?)",
    ),
    (
        "analysis metric",
        "llm-demo",
        "analysis/card-to-card.yaml",
        "metric: llm.itl.median",
        "metric: llm.itl.mediam",
        "is not registered by a loaded pack",
    ),
    (
        "analysis overlap",
        "llm-demo",
        "analysis/card-to-card.yaml",
        '"env.id"]',
        '"env.id", "env.node"]',
        "is in both hold and vary",
    ),
    (
        "analysis benchmark",
        "llm-demo",
        "analysis/card-to-card.yaml",
        "[serve-demo-llm-a]",
        "[serve-demo-llm-8c]",
        "no Benchmark of that name",
    ),
    (
        "reuse label",
        "llm-demo",
        "project.yaml",
        "labels: [env.label.device.count]}",
        "labels: [env.fp.kernel]}",
        "names no registered env.label attribute",
    ),
    ("reuse force", "llm-demo", "project.yaml", "force: []", "force: [nope]", "reuse.force: 'nope' is not a case"),
    (
        "clean needs exclusive",
        "central",
        CENTRAL_ENV,
        "patch: {host: {reference_node: false, exclusive: false}",
        "patch: {host: {reference_node: false, exclusive: false, clean: {mode: wait, max_wait: 1h, max_hold: 1h}}",
        "host.clean needs host.exclusive",
    ),
    ("clean mode", "central", CENTRAL_ENV, "clean: {mode: wait,", "clean: {mode: evict,", "is not one of"),
    ("clean max_hold", "central", CENTRAL_ENV, "max_hold: 3h", "max_hold: 0s", "host.clean.max_hold must be positive"),
    ("clean missing max_wait", "central", CENTRAL_ENV, "max_wait: 4h, ", "", "max_wait"),
    (
        "pr baseline",
        "central",
        "validity-policy.yaml",
        "      baseline: false\n      reference: false\n    investigate:",
        "      baseline: true\n      reference: false\n    investigate:",
        "a PR run can never be a baseline or reference",
    ),
    (
        "policy purpose",
        "central",
        "validity-policy.yaml",
        "    investigate:\n",
        "    investigatee:\n",
        "not a registered purpose",
    ),
    (
        "reference needs clean",
        "central",
        "validity-policy.yaml",
        "qualification: current, clean: enforced,",
        "qualification: current, clean: any,",
        "a reference purpose needs floor valid",
    ),
    (
        "baseline needs qualification",
        "central",
        "validity-policy.yaml",
        "requires: {qualification: current, clean: any, min_repeats: 3,",
        "requires: {qualification: any, clean: any, min_repeats: 3,",
        "baselines need qualification current",
    ),
    ("policy gate enum", "central", "validity-policy.yaml", "clean: enforced,", "clean: always,", "is not one of"),
    (
        "exp metric",
        "llm-demo",
        SI_DEC,
        "metric: llm.itl.median\n  baseline",
        "metric: llm.itl.mediam\n  baseline",
        "is not registered by a loaded pack",
    ),
    (
        "exp metric not recorded",
        "llm-demo",
        SI_DEC,
        "metric: llm.itl.median\n  baseline",
        "metric: llm.e2el.mean\n  baseline",
        "is not recorded by case",
    ),
    (
        "exp baseline purpose",
        "llm-demo",
        SI_DEC,
        "purposes: [nightly, weekly, release]\n    match: {environment: id}",
        "purposes: [nightly, pr]\n    match: {environment: id}",
        "baseline purpose 'pr' is not baseline-eligible",
    ),
    (
        "exp min_n",
        "llm-demo",
        SI_DEC,
        "window: 20\n    min_n: 10\n    purposes: [nightly, weekly, release]\n    match: {environment: id}",
        "window: 5\n    min_n: 10\n    purposes: [nightly, weekly, release]\n    match: {environment: id}",
        "min_n exceeds window",
    ),
    (
        "exp selector",
        "llm-demo",
        SI_DEC,
        "selector: {benchmarks: [serve-demo-llm-a]}\n  metric: llm.itl",
        "selector: {benchmarks: [serve-demo-llm-8c]}\n  metric: llm.itl",
        "no Benchmark of that name",
    ),
    (
        "exp fixed value",
        "llm-demo",
        "grid/serve-grid-demo-llm-a.yaml",
        "baseline: {kind: rolling, window: 20, min_n: 10, purposes: [nightly], match: {environment: id}}",
        "baseline: {kind: fixed}",
        "a fixed baseline needs value",
    ),
    (
        "exp fixed window",
        "llm-demo",
        "grid/serve-grid-demo-llm-a.yaml",
        "baseline: {kind: rolling, window: 20, min_n: 10, purposes: [nightly], match: {environment: id}}",
        "baseline: {kind: fixed, value: 1.0, window: 3}",
        "apply to a rolling baseline only",
    ),
    ("exp tolerance", "llm-demo", "grid/serve-grid-demo-llm-a.yaml", "tolerance: {pct: 10}", "tolerance: {}", "pct"),
    (
        "exp severity",
        "llm-demo",
        "grid/serve-grid-demo-llm-a.yaml",
        "severity: advisory",
        "severity: blocking",
        "is not one of",
    ),
    (
        "exp direction",
        "llm-demo",
        SI_DEC,
        "metric: llm.itl.median\n  baseline",
        "metric: spyre.compiled_graphs\n  baseline",
        "has no direction",
    ),
    (
        "two policies",
        "llm-demo",
        "project.yaml",
        "visibility: public\n  groupings",
        "visibility: public\n  validity_policy: nope\n  groupings",
        "no ValidityPolicy of that name",
    ),
    (
        "device class",
        "central",
        CENTRAL_ENV,
        "device_class: spyre/pf, count: 8",
        "device_class: spyre/pg, count: 8",
        "is not registered by a loaded pack (did you mean 'spyre/pf'?)",
    ),
    (
        "selector attribute",
        "central",
        CENTRAL_ENV,
        "attribute: spyre.profile, eq: a}]}",
        "attribute: spyre.profil, eq: a}]}",
        "is not an attribute of spyre/pf (did you mean 'spyre.profile'?)",
    ),
    (
        "selector value",
        "central",
        CENTRAL_ENV,
        "attribute: spyre.profile, eq: a}]}",
        "attribute: spyre.profile, eq: other}]}",
        "is not one of",
    ),
    (
        "selector type",
        "central",
        CENTRAL_ENV,
        "attribute: net.gbps, gte: 10",
        "attribute: net.gbps, gte: fast",
        "net.gbps is float",
    ),
    (
        "constraint attribute",
        "central",
        CENTRAL_ENV,
        "match_attribute: cpu.cores, requests: [sockets]",
        "match_attribute: cpu.cores9, requests: [sockets]",
        "is not an attribute of cpu/socket",
    ),
    (
        "constraint request",
        "central",
        CENTRAL_ENV,
        "match_attribute: cpu.cores, requests: [sockets]",
        "match_attribute: cpu.cores, requests: [socks]",
        "request 'socks' is not a devices request",
    ),
    (
        "multi-node fabric",
        "central",
        CENTRAL_ENV,
        "  interconnect:\n    requests:\n      - {name: fabric, device_class: net/link, count: 1, selectors: [{attribute: net.gbps, gte: 10}]}\n",
        "",
        "nodes > 1 needs an interconnect request",
    ),
    (
        "fabric class",
        "central",
        CENTRAL_ENV,
        "name: fabric, device_class: net/link",
        "name: fabric, device_class: spyre/pf",
        "interconnect requests must be network device classes",
    ),
    (
        "label count",
        "central",
        CENTRAL_ENV,
        "labels: {device.count: 8,",
        "labels: {device.count: 7,",
        "labels.device.count 7 != 8 accelerators requested",
    ),
    (
        "label value",
        "central",
        CENTRAL_ENV,
        "labels: {device.count: 8,",
        "labels: {pool: other, device.count: 8,",
        "is not one of",
    ),
    (
        "label key",
        "central",
        CENTRAL_ENV,
        "labels: {device.count: 8,",
        "labels: {Device.Count: 8,",
        "keys are dotted lowercase names",
    ),
    (
        "patch replaces list",
        "central",
        CENTRAL_ENV,
        "- {name: cards, device_class: spyre/pf, count: 1, selectors: [{attribute: spyre.profile, eq: a}]}\n    labels: {device.count: 1}",
        "- {name: cards, count: 1}\n    labels: {device.count: 1}",
        "device_class",
    ),
    (
        "node class arch",
        "central",
        CENTRAL_ENV,
        "node_class: x86-shared-pool, arch: x86_64",
        "node_class: x86-shared-pool, arch: aarch64",
        "arch x86_64 != aarch64",
    ),
    (
        "nodeclass extension",
        "central",
        "node-classes.yaml",
        "device_function: pf",
        "device_function: xx",
        "is not one of",
    ),
    (
        "duplicate request",
        "central",
        CENTRAL_ENV,
        "requests: [{name: cards, device_class: spyre/pf, count: 1,",
        "requests: [{name: cards, device_class: spyre/pf, count: 1}, {name: cards, device_class: spyre/pf, count: 1,",
        "duplicate request name",
    ),
    (
        "pack attribute",
        "postgres-demo",
        "project.yaml",
        "series_labels: [env.label.node.class, env.label.storage.class]",
        "series_labels: [env.fp.kernel]",
        "does not match",
    ),
    (
        "pg label value",
        "postgres-demo",
        "project.yaml",
        "storage.class: nvme-local",
        "storage.class: floppy",
        "is not one of",
    ),
    (
        "pg asset generator",
        "postgres-demo",
        "project.yaml",
        "source: {generator: {tool: pgbench-init, seed: 0, params: {scale: 100}}}",
        "source: {uri: 'x://y'}",
        "type generator needs source.generator",
    ),
]


# ------------------------------------------------------------------ host pinning, derive and compare
def fake_proc(root, procs, start=1000):
    """procs: {pid: (ppid, comm, {tid: cpu list})} (+ optional pgrp, session, cgroup) as a minimal /proc."""
    for pid, spec in procs.items():
        ppid, comm, tasks, *extra = spec
        opts = extra[0] if extra else {}
        d = root / str(pid)
        (d / "task").mkdir(parents=True, exist_ok=True)
        fields = ["S", ppid, opts.get("pgrp", pid), opts.get("session", pid)] + [0] * 15 + [opts.get("start", start)]
        (d / "stat").write_text(f"{pid} ({comm}) " + " ".join(map(str, fields)) + " 0 0\n")
        if "cgroup" in opts:
            (d / "cgroup").write_text(f"0::{opts['cgroup']}\n")
        for tid, cpus in tasks.items():
            (d / "task" / str(tid)).mkdir(exist_ok=True)
            (d / "task" / str(tid) / "status").write_text(f"Name:\tx\nCpus_allowed_list:\t{cpus}\nMems_allowed:\t1\n")


def fake_sys(root, nodes):
    for n, cpus in nodes.items():
        d = root / "devices" / "system" / "node" / f"node{n}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(cpus + "\n")


class PinningObservation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=TMP))
        self.proc, self.sysd = self.tmp / "proc", self.tmp / "sys"
        self.proc.mkdir()
        fake_proc(
            self.proc,
            {
                100: (5, "serve (x) y", {100: "0-3", 101: "0-3"}),
                200: (100, "worker", {200: "2-3"}),
                300: (5, "other", {300: "0-63"}),
            },
        )
        fake_sys(self.sysd, {0: "0-3", 1: "4-7"})

    def verify(self, request, pid=100, numa=()):
        return placement.verify_pinning(request, pid, numa, self.proc, self.sysd)

    def test_cpu_list_round_trip(self):
        self.assertEqual(placement.parse_cpu_list("0-2,5,7-8"), frozenset({0, 1, 2, 5, 7, 8}))
        self.assertEqual(placement.format_cpu_list({8, 0, 1, 2, 5, 7}), "0-2,5,7-8")
        self.assertEqual(placement.format_cpu_list(placement.parse_cpu_list("3,1-2,2")), "1-3")
        for bad in ("5-2", "a", "1-"):
            with self.assertRaises(ValueError):
                placement.parse_cpu_list(bad)

    def test_tree_includes_children_and_threads_but_not_unrelated_processes(self):
        seen = placement.read_affinities(100, self.proc)
        self.assertEqual(sorted((p, t) for p, t, _ in seen), [(100, 100), (100, 101), (200, 200)])

    def test_inside_the_set_verifies_and_records_the_applied_set(self):
        r = self.verify({"policy": "explicit", "cpus": "0-3"})
        self.assertEqual(
            r,
            {
                "run.exec.cpu_pinning": "explicit",
                "run.exec.cpu_set_source": "explicit",
                "run.exec.cpu_set": "0-3",
                "run.exec.pinning_verified": "true",
                "run.exec.pinning_violations": "0",
            },
        )

    def test_every_thread_outside_the_set_is_a_violation(self):
        r = self.verify({"policy": "explicit", "cpus": "2-3"})
        self.assertEqual((r["run.exec.pinning_verified"], r["run.exec.pinning_violations"]), ("false", "2"))
        self.assertEqual(r["run.exec.cpu_set"], "0-3")

    def test_cannot_observe_is_unknown_never_true(self):
        want = {"policy": "explicit", "cpus": "0-3"}
        gone = self.verify(want, pid=999)
        self.assertEqual(gone["run.exec.pinning_verified"], "unknown")
        self.assertNotIn("run.exec.pinning_violations", gone)
        self.assertNotIn("run.exec.cpu_set", gone)
        self.assertEqual(
            placement.verify_pinning(want, 100, (), self.tmp / "nope")["run.exec.pinning_verified"], "unknown"
        )
        with mock.patch.object(placement.sys, "platform", "darwin"):
            self.assertIsNone(placement.read_affinities(100))
        none = self.verify({"policy": "none"})
        self.assertEqual((none["run.exec.pinning_verified"], none["run.exec.cpu_set_source"]), ("unknown", "none"))
        self.assertEqual(none["run.exec.cpu_set"], "0-3")

    def test_device_local_resolves_the_device_numa_node(self):
        r = self.verify({"policy": "device_local"}, numa=[0])
        self.assertEqual((r["run.exec.cpu_set_source"], r["run.exec.pinning_verified"]), ("device_numa", "true"))
        r = self.verify({"policy": "device_local"}, numa=[1])
        self.assertEqual(r["run.exec.pinning_verified"], "false")
        for numa in ((), [-1], [7]):
            r = self.verify({"policy": "device_local"}, numa=numa)
            self.assertEqual((r["run.exec.cpu_set_source"], r["run.exec.pinning_verified"]), ("unresolved", "unknown"))

    def test_per_device_allows_one_device_set_per_thread_but_not_a_mix(self):
        fake_proc(self.proc, {400: (5, "ranks", {401: "0-3", 402: "4-7"}), 500: (5, "mixed", {501: "2-5"})})
        self.assertEqual(self.verify({"policy": "per_device"}, 400, [0, 1])["run.exec.pinning_verified"], "true")
        r = self.verify({"policy": "per_device"}, 500, [0, 1])
        self.assertEqual((r["run.exec.pinning_verified"], r["run.exec.pinning_violations"]), ("false", "1"))
        self.assertEqual(self.verify({"policy": "device_local"}, 500, [0, 1])["run.exec.pinning_verified"], "true")


class PinningValidity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pol = cat("llm-demo").policy.spec
        cls.good = {
            "qualified": True,
            "clean": {"exclusive": "enforced"},
            "repeats": 5,
            "cv_pct": 1.0,
            "warmup_discarded": 2,
            "placement": True,
        }

    def v(self, request, props):
        return engine.validity_of(self.pol, "reference", {**self.good, **placement.validity_facts(request, props)})

    def test_required_pinning_that_is_violated_or_unseen_caps_at_suspect(self):
        req = {"policy": "explicit", "cpus": "0-3", "require_verified": True}
        self.assertEqual(self.v(req, {"run.exec.pinning_verified": "true"}), ("valid", []))
        self.assertEqual(self.v(req, {"run.exec.pinning_verified": "false"}), ("suspect", ["pinning_violated"]))
        self.assertEqual(self.v(req, {}), ("suspect", ["pinning_unverified"]))
        self.assertEqual(self.v(req, {"run.exec.pinning_verified": "unknown"})[1], ["pinning_unverified"])

    def test_not_required_never_changes_validity(self):
        req = {"policy": "explicit", "cpus": "0-3"}
        self.assertEqual(self.v(req, {"run.exec.pinning_verified": "false"}), ("valid", []))
        self.assertEqual(self.v(None, {}), ("valid", []))

    def test_a_suspect_run_is_suspected_never_regressed(self):
        exp = cat("llm-demo").by[("Expectation", "itl-no-regression")].spec
        state, _ = self.v(
            {"policy": "explicit", "cpus": "0", "require_verified": True}, {"run.exec.pinning_verified": "false"}
        )
        j = engine.judge(exp, "e" * 32, "v" * 32, "lower_is_better", [2.0, 2.0, 2.0], [1.0] * 5, validity=state)
        self.assertEqual(j["status"], "suspected")
        self.assertNotEqual(j["status"], "regressed")


class PinningRequest(unittest.TestCase):
    D = "pinning-demo"

    def test_existing_ids_are_unchanged_by_the_block(self):
        c = cat("llm-demo")
        case = next(x for x in c.cases if x["name"] == "latency_pinned_demo-llm-c_tp1_in64_out64")
        ids = c.ids(case)
        self.assertEqual(ids["benchmark_id"], "57055cb0-da70-500e-963e-fe15615233f9")
        env = c.effective("Environment", case["environment"])
        self.assertNotIn("env.label.cpu_pinning", c.env_labels(env))
        series = {k: c.env_labels(env)[k] for k in c.project.spec["series_labels"]}
        self.assertEqual(engine.series_key(ids["benchmark_id"], series), "85eb2a6f-9f13-58a0-af9d-5af87ebbe25b")
        self.assertEqual(
            engine.env_id_of({"env.fp.kernel": "6.8", "env.fp.cpu.base_ghz": 2.4}),
            "c01117b6-9647-50f1-9012-21b539e2ba65",
        )
        self.assertEqual(c.env_view("ref-1card")["spec_digest"], "429b09d4-b2a9-5027-bb0f-f23e40b97f10")

    def test_vocabulary_is_registered(self):
        reg = cat(self.D).reg
        for k in (
            "host.sockets",
            "host.numa_nodes",
            "host.cores_per_socket",
            "host.smt",
            "host.llc_domains",
            "host.sub_numa",
            "device.numa_node",
            "device.pcie_width",
            "device.pcie_gen",
        ):
            self.assertTrue(reg.attrs[f"env.fp.{k}"]["fingerprint"], k)
        for k in ("cpu_pinning", "cpu_set", "cpu_set_source", "pinning_verified", "pinning_violations"):
            self.assertEqual(reg.attrs[f"run.exec.{k}"]["class"], "run", k)
        self.assertEqual(reg.attrs["run.exec.cpu_pinning"]["values"], list(placement.POLICIES))

    def test_label_and_view_follow_the_request(self):
        c = cat(self.D)
        self.assertEqual(c.env_view("node-a-unpinned")["labels"]["env.label.cpu_pinning"], "none")
        v = c.env_view("node-a-device-local")
        self.assertEqual(v["labels"]["env.label.cpu_pinning"], "device_local")
        self.assertEqual(v["cpu_pinning"], {"policy": "device_local", "require_verified": True})
        self.assertNotEqual(v["spec_digest"], c.env_view("node-a-unpinned")["spec_digest"])

    def test_pinned_and_unpinned_never_share_a_series_or_a_reuse(self):
        c = cat(self.D)
        case = c.cases[0]
        fp = {"env.fp.host.sockets": "2", "env.fp.kernel": "k"}
        keys, rows = {}, []
        for env in ("node-a-unpinned", "node-a-device-local"):
            v = c.env_view(env)
            res = engine.build_result(
                c,
                case,
                case["name"],
                {"artifact_id": "s"},
                fp,
                {},
                v["labels"],
                [{"run.repeat": 1, "placement": {"nodes": ["n"]}, "samples": {"llm.batch_latency.mean": [1.0]}}],
            )
            self.assertEqual(engine.recompute(res), [])
            keys[env] = (res["status"]["environment"]["env_id"], res["status"]["series"]["key"])
            rows.append(
                {
                    "benchmark_id": "b",
                    "subject_id": "s",
                    "env_id": keys[env][0],
                    "labels": v["labels"],
                    "validity": "valid",
                    "ts": "2026-11-13T00:00:00Z",
                    "repeats": [{}],
                }
            )
        self.assertEqual(keys["node-a-unpinned"][0], keys["node-a-device-local"][0])
        self.assertNotEqual(keys["node-a-unpinned"][1], keys["node-a-device-local"][1])
        want = {
            "benchmark_id": "b",
            "subject_id": "s",
            "env_id": keys["node-a-unpinned"][0],
            "repeats": 1,
            "environment": "id",
            "floor": "valid",
            "labels": c.env_view("node-a-unpinned")["labels"],
        }
        self.assertEqual(engine.reuse_lookup(rows, [], want, NOW)[0], rows[0])
        self.assertEqual(engine.reuse_lookup(rows[1:], [], want, NOW), (None, "different_environment"))
        old = {k: v for k, v in rows[0]["labels"].items() if k != "env.label.cpu_pinning"}
        self.assertIsNotNone(engine.reuse_lookup([{**rows[0], "labels": old}], [], want, NOW)[0])

    def test_series_default_covers_results_built_without_the_label(self):
        c = cat(self.D)
        case = c.cases[0]
        labels = {k: v for k, v in c.env_view("node-a-unpinned")["labels"].items() if k != "env.label.cpu_pinning"}
        res = engine.build_result(
            c,
            case,
            "x",
            {"artifact_id": "s"},
            {"env.fp.kernel": "k"},
            {},
            labels,
            [{"run.repeat": 1, "placement": {"nodes": ["n"]}, "samples": {}}],
        )
        self.assertEqual(res["status"]["series"]["labels"]["env.label.cpu_pinning"], "none")

    def test_request_errors(self):
        for case, old, new, want in [
            (
                "explicit-needs-cpus",
                "policy: device_local, require_verified: true}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "policy: explicit}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "cpus is required for, and only valid with, policy explicit",
            ),
            (
                "cpus-without-explicit",
                "policy: device_local, require_verified: true}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "policy: device_local, cpus: '0-3'}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "only valid with, policy explicit",
            ),
            (
                "require-verified-none",
                "policy: device_local, require_verified: true}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "policy: none, require_verified: true}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "require_verified needs a policy other than none",
            ),
            (
                "bad-policy",
                "policy: device_local, require_verified: true}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "policy: sticky}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "sticky",
            ),
            (
                "reversed-range",
                "policy: device_local, require_verified: true}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "policy: explicit, cpus: '5-2'}}}\n---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-b-unpinned",
                "bad CPU range",
            ),
            (
                "hand-set-label",
                "labels: {node_class: node-a, device.count: 2}",
                "labels: {node_class: node-a, device.count: 2, cpu_pinning: none}",
                "labels.cpu_pinning is derived",
            ),
        ]:
            with self.subTest(case):
                d = copy_catalog(self.D, "pin/" + case)
                edit(d / "project.yaml", old, new)
                errs = cat(d).errors
                self.assertTrue(any(want in e for e in errs), f"{case}: want {want!r}, got {errs}")


def mk_result(
    case,
    repeats,
    env_id="env-1",
    subject="s-1",
    node="n-1",
    pin="none",
    bench="decode-probe",
    metric="llm.batch_latency.mean",
    labels=None,
    repeat_ids=None,
    failed=(),
    props=True,
    run_props=None,
):
    """A minimal Result: repeats = [samples-list per repeat], numbered by repeat_ids; props=False omits repeat props."""
    ids = repeat_ids or list(range(1, len(repeats) + 1))
    vals = [statistics.median(s) for s in repeats]
    return {
        "metadata": {"name": case},
        "status": {
            "case": {"name": bench},
            "subject": {"artifact_id": subject},
            "environment": {
                "env_id": env_id,
                "labels": {"env.label.node_class": "node-a", "env.label.cpu_pinning": pin, **(labels or {})},
                "observations": {"env.node": node},
            },
            "repeats": [
                {
                    "run.repeat": i,
                    "status": "failed" if i in failed else "ok",
                    **({"props": {"run.exec.cpu_pinning": pin}} if props else {}),
                    "samples": {metric: s},
                }
                for i, s in zip(ids, repeats)
            ],
            **({"run_props": run_props} if run_props else {}),
            "aggregate": {
                "statistic": "median",
                "n": len(repeats),
                "metrics": {metric: {"value": statistics.median(vals), "unit": "s"}},
            },
        },
    }


SLOPE = {
    "name": "llm.decode_per_token",
    "op": "slope",
    "cases": {"first": "long", "second": "short"},
    "x": {"first": 129, "second": 1},
}
M = "llm.batch_latency.mean"


class DeriveAndCompare(unittest.TestCase):
    def pair(self, derive=SLOPE, **kw):
        long = mk_result("long", [[13.0], [13.2], [12.8]], **kw)
        short = mk_result("short", [[1.0], [1.0], [1.0]], **kw)
        return long, short

    def test_slope_per_repeat_pair(self):
        d = analysis.derive_pair(SLOPE, M, *self.pair())
        self.assertEqual(d["repeats"], [1, 2, 3])
        self.assertAlmostEqual(d["values"][0], 12.0 / 128)
        self.assertAlmostEqual(d["median"], 12.0 / 128)
        self.assertAlmostEqual(d["mean"], 12.0 / 128)
        self.assertAlmostEqual(d["cv_pct"], 0.2 / 12.0 * 100, places=6)
        self.assertAlmostEqual(d["of_aggregates"], 12.0 / 128)

    def test_ratio_and_difference_and_unmatched_repeats(self):
        long, short = self.pair()
        short["status"]["repeats"].pop()
        r = analysis.derive_pair({**SLOPE, "op": "ratio"}, M, long, short)
        self.assertEqual((r["n"], r["values"]), (2, [13.0, 13.2]))
        self.assertAlmostEqual(r["of_aggregates"], 13.0 / 1.0)
        r = analysis.derive_pair({**SLOPE, "op": "difference"}, M, long, short)
        self.assertAlmostEqual(r["median"], 12.1)

    def test_incomparable_cases_are_refused_with_the_reason(self):
        for kw, why in [
            ({"env_id": "env-2"}, "different env_id"),
            ({"subject": "s-2"}, "different subject"),
            ({"node": "n-2"}, "different node"),
            ({"pin": "device_local"}, "different cpu pinning"),
        ]:
            with self.subTest(why):
                long, _ = self.pair()
                short = mk_result("short", [[1.0]] * 3, **kw)
                with self.assertRaises(analysis.AnalysisError) as cm:
                    analysis.derive_pair(SLOPE, M, long, short)
                self.assertIn(why, str(cm.exception))
                self.assertIn("not comparable", str(cm.exception))
        long, _ = self.pair()
        with self.assertRaises(analysis.AnalysisError) as cm:
            analysis.derive_pair(SLOPE, M, long, mk_result("short", [[1.0]], labels={"env.label.node_class": "node-b"}))
        self.assertIn("different environment labels", str(cm.exception))

    def spec(self, **over):
        return {"select": {"benchmarks": ["decode-probe"]}, "metric": M, "derive": SLOPE, **over}

    def test_analyze_refuses_when_no_pair_shares_a_context(self):
        long, _ = self.pair()
        with self.assertRaises(analysis.AnalysisError) as cm:
            analysis.analyze(self.spec(), [long, mk_result("short", [[1.0]] * 3, env_id="env-2")])
        self.assertIn("different env_id", str(cm.exception))
        with self.assertRaises(analysis.AnalysisError) as cm:
            analysis.analyze(self.spec(), [long])
        self.assertIn("no Result for case 'short'", str(cm.exception))

    def test_analyze_counts_unpaired_results_and_rejects_ambiguity(self):
        long, short = self.pair()
        other = mk_result("long", [[20.0]] * 3, env_id="env-9")
        rep = analysis.analyze(self.spec(), [long, short, other])
        self.assertEqual((len(rep["derive"]["pairs"]), rep["derive"]["unpaired_results"]), (1, 1))
        self.assertEqual(rep["derive"]["unit"], "s")
        with self.assertRaises(analysis.AnalysisError) as cm:
            analysis.analyze(self.spec(), [long, short, mk_result("short", [[1.0]] * 3)])
        self.assertIn("select one run per context", str(cm.exception))

    def cells(self):
        res = []
        for node, base, local, spread_ in (("node-a", 0.0100, 0.0090, 0.0001), ("node-b", 0.0100, 0.00995, 0.0006)):
            for pin, tok in (("none", base), ("device_local", local)):
                long = mk_result(
                    "long",
                    [[1.0 + 128 * (tok + d)] for d in (-spread_, 0.0, spread_)],
                    pin=pin,
                    env_id=f"env-{node}",
                    node=f"{node}-h",
                    labels={"env.label.node_class": node},
                )
                short = mk_result(
                    "short",
                    [[1.0]] * 3,
                    pin=pin,
                    env_id=f"env-{node}",
                    node=f"{node}-h",
                    labels={"env.label.node_class": node},
                )
                res += [long, short]
        return res

    def test_compare_table_effect_and_noise_flag(self):
        cmp_ = {"by": "run.exec.cpu_pinning", "across": ["env.label.node_class"], "metric": "llm.decode_per_token"}
        rep = analysis.analyze(self.spec(compare=cmp_), self.cells())["compare"]
        self.assertEqual((rep["baseline"], rep["unit"], rep["by"]), ("none", "s", "run.exec.cpu_pinning"))
        t = {(c["env.label.node_class"], c["run.exec.cpu_pinning"]): c for c in rep["cells"]}
        self.assertEqual(
            sorted(t), [("node-a", "device_local"), ("node-a", "none"), ("node-b", "device_local"), ("node-b", "none")]
        )
        a = t[("node-a", "device_local")]
        self.assertEqual(a["n"], 3)
        self.assertAlmostEqual(a["median"], 0.0090)
        self.assertAlmostEqual(a["effect_pct"], -10.0)
        self.assertIs(a["within_noise"], False)
        b = t[("node-b", "device_local")]
        self.assertAlmostEqual(b["effect_pct"], -0.5)
        self.assertIs(b["within_noise"], True)
        base = t[("node-a", "none")]
        self.assertEqual((base["effect_pct"], base["within_noise"]), (None, None))
        self.assertIsNotNone(base["cv_pct"])

    def test_compare_a_plain_metric_and_a_single_value_cell(self):
        res = [
            mk_result("long", [[2.0], [2.2]], pin="none"),
            mk_result("long", [[1.5]], pin="device_local"),
        ]
        spec = {
            "select": {"benchmarks": ["decode-probe"]},
            "metric": M,
            "compare": {"by": "run.exec.cpu_pinning", "metric": M},
        }
        cells = analysis.analyze(spec, res)["compare"]["cells"]
        self.assertEqual([c["run.exec.cpu_pinning"] for c in cells], ["device_local", "none"])
        self.assertIsNone(cells[0]["cv_pct"])
        self.assertIsNone(cells[0]["within_noise"])
        self.assertAlmostEqual(cells[0]["effect_pct"], (1.5 - 2.1) / 2.1 * 100)

    def test_schema_and_catalog_checks(self):
        c = cat("pinning-demo")
        self.assertEqual(c.errors, [])
        base = "analysis/pinning-effect.yaml"
        for case, old, new, want in [
            ("slope-needs-x", "    x: {first: 129, second: 1}\n", "", "derive.x is required for"),
            ("equal-x", "x: {first: 129, second: 1}", "x: {first: 3, second: 3}", "first and second must differ"),
            ("unknown-case", "cases: {first: decode-probe/out=out129,", "cases: {first: nope,", "no case of that name"),
            (
                "same-case",
                "second: decode-probe/out=out1}",
                "second: decode-probe/out=out129}",
                "must be different cases",
            ),
            (
                "shadows-metric",
                "name: llm.decode_per_token",
                "name: llm.batch_latency.mean",
                "shadows a registered metric",
            ),
            ("bad-op", "op: slope", "op: product", "product"),
            (
                "compare-metric",
                "    metric: llm.decode_per_token",
                "    metric: llm.nothing",
                "neither registered nor derive.name",
            ),
            ("compare-attr", "by: run.exec.cpu_pinning", "by: run.exec.nothing", "not a registered attribute"),
        ]:
            with self.subTest(case):
                d = copy_catalog("pinning-demo", "dc/" + case)
                edit(d / base, old, new)
                errs = cat(d).errors
                self.assertTrue(any(want in e for e in errs), f"{case}: want {want!r}, got {errs}")


class TraceScalingExample(unittest.TestCase):
    """Scaling across card counts: the card count differs by design, so the Analysis relaxes the environment id and label."""

    LAT = {1: 330.0, 2: 210.0, 4: 160.0}

    def results(self, other_subject=False, node_class=None):
        c = cat("trace-demo")
        by = {x["name"]: x for x in c.cases}
        out = []
        for n, lat in self.LAT.items():
            env = f"cards-{n}"
            v = c.env_view(env)
            labels = {**v["labels"], **({"env.label.node_class": node_class} if node_class and n == 4 else {})}
            reps = [
                {"run.repeat": r, "placement": {"nodes": ["host-a"]}, "samples": {M: [lat + d]}, "props": {}}
                for r, d in enumerate((-0.5, 0.0, 0.5), 1)
            ]
            subject = "subject-2" if other_subject and n == 4 else "subject-1"
            out.append(
                engine.build_result(
                    c,
                    by[f"decode_scaling_tp{n}"],
                    f"decode_scaling_tp{n}",
                    {"artifact_id": subject},
                    {"env.fp.kernel": f"kernel-{n}"},
                    {"env.node": "host-a"},
                    labels,
                    reps,
                )
            )
        return out

    def analyze(self, results, spec_edit=None):
        doc = cat("trace-demo").by[("Analysis", "decode-scaling")]
        spec = copy.deepcopy(doc.spec)
        if spec_edit:
            spec_edit(spec)
        return analysis.analyze(spec, results, cat("trace-demo").reg.metrics)

    def test_scaling_against_one_card(self):
        cells = {c["env.label.device.count"]: c for c in self.analyze(self.results())["compare"]["cells"]}
        self.assertAlmostEqual(cells["4"]["effect_pct"], (160.0 - 330.0) / 330.0 * 100)
        self.assertIsNone(cells["1"]["effect_pct"])
        self.assertEqual(cells["2"]["reading"], "better")

    def test_without_relax_different_environments_are_refused(self):
        def drop(spec):
            del spec["relax"]

        with self.assertRaisesRegex(analysis.AnalysisError, "different env_id"):
            self.analyze(self.results(), drop)

    def test_relax_does_not_excuse_a_different_subject_or_node_class(self):
        with self.assertRaisesRegex(analysis.AnalysisError, "different subject"):
            self.analyze(self.results(other_subject=True))
        with self.assertRaisesRegex(analysis.AnalysisError, "different environment labels"):
            self.analyze(self.results(node_class="node-b"))

    def test_relax_accepts_a_trailing_wildcard_but_a_bare_star_relaxes_nothing(self):
        def labels(spec):
            spec["relax"] = ["env.id", "env.label.device.*"]

        def bare(spec):
            spec["relax"] = ["*"]

        self.assertEqual(len(self.analyze(self.results(), labels)["compare"]["cells"]), 3)
        with self.assertRaisesRegex(analysis.AnalysisError, "different env_id"):
            self.analyze(self.results(), bare)

    def test_relaxing_env_id_stops_comparing_the_fingerprint(self):
        """Documented limit: the per-card-count kernel fingerprints differ in the fixture and the comparison still runs."""
        r = self.results()
        self.assertNotEqual(r[0]["status"]["environment"]["env_id"], r[2]["status"]["environment"]["env_id"])
        self.assertEqual(len(self.analyze(r)["compare"]["cells"]), 3)

    def test_rank_spread_resolves_against_the_registered_rank_attribute(self):
        c = cat("trace-demo")
        self.assertEqual(c.errors, [])
        self.assertEqual(c.by[("Analysis", "rank-spread")].spec["vary"], ["run.rank"])
        d = copy_catalog("trace-demo", "rank-unregistered")
        edit(d / "analysis" / "rank-spread.yaml", "vary: [run.rank]", "vary: [run.rank_nope]")
        self.assertTrue(any("matches no registered attribute" in e for e in cat(d, [TRACE_PACK], []).errors))


class PinningExample(unittest.TestCase):
    """End to end on the synthetic example: build Results, evaluate the Analysis through the CLI."""

    def results(self, skip_env=None, other_subject=False):
        c = cat("pinning-demo")
        by = {x["name"]: x for x in c.cases}
        out = []
        # synthetic per-token cost per (environment, pinned?)
        per_token = {
            "node-a-unpinned": 0.0100,
            "node-a-device-local": 0.0090,
            "node-b-unpinned": 0.0100,
            "node-b-device-local": 0.00995,
        }
        for env, tok in per_token.items():
            if env == skip_env:
                continue
            v = c.env_view(env)
            sockets = "1" if env.startswith("node-a") else "2"
            node = "host-" + env.split("-")[1]
            fp = {"env.fp.host.sockets": sockets, "env.fp.kernel": "kernel-x"}
            pin = v["cpu_pinning"]["policy"]
            for name, lat in (
                ("decode-probe/out=out1", lambda t: 1.0),
                ("decode-probe/out=out129", lambda t: 1.0 + 128 * t),
            ):
                reps = []
                for r, d in enumerate((-0.0001, 0.0, 0.0001), 1):
                    reps.append(
                        {
                            "run.repeat": r,
                            "placement": {"nodes": [node]},
                            "samples": {M: [lat(tok + d)]},
                            "props": {
                                "run.exec.cpu_pinning": pin,
                                "run.exec.pinning_verified": "true" if pin != "none" else "unknown",
                            },
                        }
                    )
                out.append(
                    engine.build_result(
                        c,
                        by[name],
                        name,
                        {"artifact_id": "subject-1" if not other_subject or "out129" not in name else "subject-2"},
                        fp,
                        {"env.node": node},
                        v["labels"],
                        reps,
                    )
                )
        return out

    def run_cli(self, results, *extra):
        f = TMP / "pinning-results.json"
        f.write_text(json.dumps(results))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli_main(
                ["analyze", "pinning-effect", "--catalog", str(X / "pinning-demo"), "--results", str(f), *extra]
            )
        return rc, out.getvalue(), err.getvalue()

    def test_example_results_recompute(self):
        for r in self.results():
            self.assertEqual(engine.recompute(r), [])
            self.assertIn("run.exec.cpu_pinning", r["status"]["repeats"][0]["props"])

    def test_analysis_report_json_and_yaml(self):
        rc, out, _ = self.run_cli(self.results(), "--json")
        self.assertEqual(rc, 0)
        rep = json.loads(out)
        t = {(c["env.label.node_class"], c["run.exec.cpu_pinning"]): c for c in rep["compare"]["cells"]}
        self.assertAlmostEqual(t[("node-a", "device_local")]["effect_pct"], -10.0, places=6)
        self.assertIs(t[("node-a", "device_local")]["within_noise"], False)
        self.assertIs(t[("node-b", "device_local")]["within_noise"], True)
        self.assertEqual(len(rep["derive"]["pairs"]), 4)
        rc, out, _ = self.run_cli(self.results())
        self.assertEqual(rc, 0)
        self.assertEqual(yaml.safe_load(out)["metric"], M)

    def test_cli_refuses_incomparable_cases(self):
        rc, out, err = self.run_cli(self.results(other_subject=True))
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("different subject", err)
        rc, _, err = self.run_cli(self.results(skip_env="node-b-unpinned"))
        self.assertEqual(rc, 0)

    def test_cli_errors(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli_main(["analyze", "nope", "--catalog", str(X / "pinning-demo"), "--results", "x"]), 2)
            self.assertEqual(
                cli_main(
                    [
                        "analyze",
                        "pinning-effect",
                        "--catalog",
                        str(X / "pinning-demo"),
                        "--results",
                        str(TMP / "missing.json"),
                    ]
                ),
                2,
            )


EXPLICIT = {"policy": "explicit", "cpus": "0-3"}
PIN = "run.exec.cpu_pinning"
OLD_FP = {"env.fp.kernel": "6.8", "env.fp.cpu.base_ghz": 2.4}
NEW_FP = {**OLD_FP, "env.fp.host.sockets": "2", "env.fp.device.pcie_gen": "gen-x"}


class PlacementStrictness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=TMP))
        self.proc, self.sysd = self.tmp / "proc", self.tmp / "sys"
        self.fresh()

    def fresh(self, extra=None):
        shutil.rmtree(self.proc, ignore_errors=True)
        self.proc.mkdir()
        fake_proc(
            self.proc,
            {
                100: (1, "serve", {100: "0-3", 101: "0-3"}, {"cgroup": "/pod"}),
                200: (100, "worker", {200: "0-3"}),
                **(extra or {}),
            },
        )

    def verify(self, request=EXPLICIT, pid=100, numa=()):
        return placement.verify_pinning(request, pid, numa, self.proc, self.sysd)

    def test_complete_observation_is_true(self):
        self.assertEqual(self.verify()["run.exec.pinning_verified"], "true")

    def test_any_unreadable_or_malformed_piece_makes_it_unknown_not_true(self):
        def garbage_status():
            (self.proc / "200/task/200/status").write_text("Name:\tx\n")

        def empty_mask():
            (self.proc / "200/task/200/status").write_text("Cpus_allowed_list:\t\n")

        def bad_mask():
            (self.proc / "200/task/200/status").write_text("Cpus_allowed_list:\t0-x\n")

        def task_dir_is_a_file():
            shutil.rmtree(self.proc / "200/task")
            (self.proc / "200/task").write_text("")

        def status_is_a_directory():
            (self.proc / "100/task/101/status").unlink()
            (self.proc / "100/task/101/status").mkdir()

        def garbage_stat_elsewhere():
            fake_proc(self.proc, {300: (1, "other", {300: "0-63"})})
            (self.proc / "300/stat").write_text("garbage")

        for fn in (
            garbage_status,
            empty_mask,
            bad_mask,
            task_dir_is_a_file,
            status_is_a_directory,
            garbage_stat_elsewhere,
        ):
            with self.subTest(fn.__name__):
                self.fresh()
                fn()
                r = self.verify()
                self.assertEqual(r["run.exec.pinning_verified"], "unknown")
                self.assertNotIn("run.exec.pinning_violations", r)

    def test_a_thread_that_exits_mid_scan_is_skipped(self):
        (self.proc / "100/task/101/status").unlink()
        r = self.verify()
        self.assertEqual((r["run.exec.pinning_verified"], r["run.exec.pinning_violations"]), ("true", "0"))

    def test_a_reused_root_pid_is_unknown(self):
        real = placement._stat
        calls = []

        def stat(proc, pid):
            s = real(proc, pid)
            if int(pid) == 100:
                calls.append(1)
                return (*s[:3], s[3] + len(calls) - 1)
            return s

        with mock.patch.object(placement, "_stat", stat):
            self.assertEqual(self.verify()["run.exec.pinning_verified"], "unknown")

    def test_an_orphaned_worker_in_the_group_session_or_cgroup_is_judged(self):
        loose = (1, "orphan", {600: "0-63"})
        for name, opts, want in [
            ("same-group", {"pgrp": 100}, "false"),
            ("same-session", {"session": 100}, "false"),
            ("same-cgroup", {"cgroup": "/pod"}, "false"),
            ("unrelated", {"cgroup": "/elsewhere"}, "true"),
        ]:
            with self.subTest(name):
                self.fresh({600: (*loose, opts)})
                r = self.verify()
                self.assertEqual(r["run.exec.pinning_verified"], want)
                if want == "false":
                    self.assertEqual(r["run.exec.pinning_violations"], "1")

    def test_root_cgroup_is_not_used_for_membership(self):
        self.fresh({600: (5, "other", {600: "0-63"}, {"cgroup": "/"})})
        (self.proc / "100/cgroup").write_text("0::/\n")
        self.assertEqual(self.verify()["run.exec.pinning_verified"], "true")

    def test_one_unreadable_device_node_makes_the_requested_set_unresolved(self):
        fake_sys(self.sysd, {0: "0-3"})
        self.assertEqual(placement.resolve_requested({"policy": "device_local"}, [0], self.sysd)[1], "device_numa")
        self.assertEqual(
            placement.resolve_requested({"policy": "device_local"}, [0, 5], self.sysd), (None, "unresolved")
        )
        self.assertEqual(placement.resolve_requested({"policy": "per_device"}, [0, 5], self.sysd), (None, "unresolved"))
        r = self.verify({"policy": "device_local"}, numa=[0, 5])
        self.assertEqual((r["run.exec.cpu_set_source"], r["run.exec.pinning_verified"]), ("unresolved", "unknown"))
        fake_sys(self.sysd, {1: ""})
        self.assertEqual(placement.resolve_requested({"policy": "device_local"}, [1], self.sysd), (None, "unresolved"))

    def test_cpu_lists_are_strict_and_bounded(self):
        for bad in ("", " ", "1_0", "+1", "1,,2", "1-", "-1", "0-9999999999", "16384", "3-1", "a"):
            with self.subTest(bad), self.assertRaises(ValueError):
                placement.parse_cpu_list(bad)
        self.assertEqual(len(placement.parse_cpu_list("0-16383")), 16384)
        d = copy_catalog("pinning-demo", "pin/huge")
        edit(
            d / "project.yaml", "policy: device_local, require_verified: true", "policy: explicit, cpus: '0-9999999999'"
        )
        errs = cat(d).errors
        self.assertTrue(any("host.cpu_pinning.cpus: bad CPU range" in e for e in errs), errs)

    def test_validity_facts_take_the_worst_repeat(self):
        req = {"policy": "explicit", "cpus": "0", "require_verified": True}
        t, f, u = ({"run.exec.pinning_verified": v} for v in ("true", "false", "unknown"))
        for reps, want in [
            ([t, t], "true"),
            ([t, f], "false"),
            ([f, u], "false"),
            ([t, u], "unknown"),
            ([t, {}], "unknown"),
            ([t, None], "unknown"),
            ([], "unknown"),
            (t, "true"),
            (u, "unknown"),
        ]:
            self.assertEqual(placement.validity_facts(req, reps)["pinning_verified"], want, reps)
        self.assertTrue(placement.validity_facts(req, t)["pinning_required"])
        self.assertFalse(placement.validity_facts({"policy": "explicit", "cpus": "0"}, t)["pinning_required"])

    def test_build_result_caps_validity_from_repeat_props(self):
        c = cat("llm-demo")
        case = c.cases[0]
        req = {"policy": "explicit", "cpus": "0-3", "require_verified": True}
        key = "run.exec.pinning_verified"

        def build(verdicts, pinning=req, state="valid"):
            reps = [
                {
                    "run.repeat": i,
                    "placement": {"nodes": ["n"]},
                    "samples": {"llm.ttft.p99": [1.0]},
                    **({"props": {key: v}} if v else {}),
                }
                for i, v in enumerate(verdicts, 1)
            ]
            res = engine.build_result(
                c,
                case,
                "x",
                {"artifact_id": "a"},
                OLD_FP,
                {},
                {},
                reps,
                validity=(state, [], "nightly", "internal"),
                pinning=pinning,
            )["status"]
            return res["validity"]["state"], res["validity"]["reasons"], res["run_props"]["run.validity"]

        self.assertEqual(build(["true", "true"]), ("valid", [], "valid"))
        self.assertEqual(build(["true", "false"]), ("suspect", ["pinning_violated"], "suspect"))
        self.assertEqual(build(["true", "unknown"])[:2], ("suspect", ["pinning_unverified"]))
        self.assertEqual(build([None])[:2], ("suspect", ["pinning_unverified"]))
        self.assertEqual(build(["false"], pinning={**req, "require_verified": False})[0], "valid")
        self.assertEqual(build(["false"], pinning=None)[0], "valid")
        self.assertEqual(build(["false"], state="invalid")[0], "invalid")


class PinningMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pol = cat("llm-demo").policy.spec

    LAB = "env.label.device.count"

    def want(self, pin=None, patterns=None):
        labels = {self.LAB: "2", **({engine.PIN_LABEL: pin} if pin else {})}
        return {
            "benchmark_id": "b",
            "subject_id": "s",
            "env_id": "e1",
            "labels": labels,
            "repeats": 1,
            "floor": "valid",
            "environment": {"labels": patterns or [self.LAB]},
        }

    def row(self, pin=None, **kw):
        return {
            "run_id": "r",
            "benchmark_id": "b",
            "subject_id": "s",
            "env_id": "e1",
            "validity": "valid",
            "ts": "2026-11-12T00:00:00Z",
            "repeats": [{}],
            "purpose": "nightly",
            "labels": {self.LAB: "2", **({engine.PIN_LABEL: pin} if pin else {})},
            **kw,
        }

    def test_labels_mode_reuse_treats_a_missing_pinning_label_as_none(self):
        look = lambda row, want: engine.reuse_lookup([row], [], want, NOW)  # noqa: E731
        self.assertIsNotNone(look(self.row(), self.want("none"))[0])
        self.assertIsNotNone(look(self.row("none"), self.want())[0])
        self.assertEqual(look(self.row(), self.want("device_local"))[1], "different_environment")
        self.assertEqual(look(self.row("device_local"), self.want())[1], "different_environment")
        self.assertIsNotNone(look(self.row("device_local"), self.want("device_local"))[0])
        pin_only = self.want("none", [self.LAB, engine.PIN_LABEL])
        self.assertIsNotNone(look(self.row(), pin_only)[0])
        id_want = {**self.want(), "environment": "id"}
        self.assertIsNotNone(look(self.row(), id_want)[0])
        self.assertIsNone(look(self.row("explicit"), id_want)[0])

    def test_baselines_never_cross_pinning_policies(self):
        exp = {"baseline": {"kind": "rolling", "window": 20, "purposes": ["nightly"]}}
        for mode in ("id", {"labels": [self.LAB]}):
            exp["baseline"]["match"] = {"environment": mode}
            rows = [
                {**self.row(), "run_id": "plain"},
                {**self.row("none"), "run_id": "none"},
                {**self.row("device_local"), "run_id": "pinned"},
            ]
            cur = {"run_id": "now", "benchmark_id": "b", "ts": "2026-11-13T00:00:00Z", "env_id": "e1"}
            got = lambda labels: sorted(  # noqa: E731
                r["run_id"] for r in engine.select_baseline(self.pol, exp, rows, {**cur, "labels": labels})
            )
            with self.subTest(mode=str(mode)):
                self.assertEqual(got({self.LAB: "2"}), ["none", "plain"])
                self.assertEqual(got({self.LAB: "2", engine.PIN_LABEL: "none"}), ["none", "plain"])
                self.assertEqual(got({self.LAB: "2", engine.PIN_LABEL: "device_local"}), ["pinned"])
                self.assertEqual(got({self.LAB: "2", engine.PIN_LABEL: "explicit"}), [])


class FingerprintVersions(unittest.TestCase):
    GOLDEN = "c01117b6-9647-50f1-9012-21b539e2ba65"

    def test_new_attributes_do_not_change_v1_ids(self):
        self.assertEqual(engine.env_id_of(OLD_FP), self.GOLDEN)
        self.assertEqual(engine.env_id_of(NEW_FP), self.GOLDEN)
        self.assertEqual(engine.env_id_of(NEW_FP, 1), engine.env_id_of(OLD_FP, 1))

    def test_v2_hashes_them(self):
        self.assertNotEqual(engine.env_id_of(NEW_FP, 2), engine.env_id_of(OLD_FP, 2))
        self.assertNotEqual(engine.env_id_of(OLD_FP, 2), self.GOLDEN)
        self.assertNotEqual(engine.env_id_of({**NEW_FP, "env.fp.host.sockets": "4"}, 2), engine.env_id_of(NEW_FP, 2))

    def test_registry_and_engine_table_agree(self):
        reg = cat("pinning-demo").reg
        since = {k for k, a in reg.attrs.items() if a.get("fingerprint_since") == 2}
        self.assertEqual(since, set(engine.FP_SINCE))
        self.assertTrue(since <= reg.fingerprint_keys())
        self.assertFalse(since & reg.fingerprint_keys(1))
        self.assertEqual(reg.fingerprint_keys(2), reg.fingerprint_keys())
        self.assertIn("env.fp.kernel", reg.fingerprint_keys(1))

    def build(self, c, fp):
        reps = [{"run.repeat": 1, "placement": {"nodes": ["n"]}, "samples": {}}]
        return engine.build_result(c, c.cases[0], "x", {"artifact_id": "s"}, fp, {}, {}, reps)["status"]["environment"]

    def test_project_selects_the_version(self):
        env = self.build(cat("pinning-demo"), NEW_FP)
        self.assertEqual((env["fingerprint_version"], env["env_id"]), (1, self.GOLDEN))
        self.assertIn("env.fp.host.sockets", env["fingerprint"])
        d = copy_catalog("pinning-demo", "fpv2")
        edit(d / "project.yaml", "default_suites: [decode]", "default_suites: [decode]\n  fingerprint_version: 2")
        c2 = cat(d)
        self.assertEqual(c2.errors, [])
        env = self.build(c2, NEW_FP)
        self.assertEqual((env["fingerprint_version"], env["env_id"]), (2, engine.env_id_of(NEW_FP, 2)))
        res = engine.build_result(
            c2,
            c2.cases[0],
            "x",
            {"artifact_id": "s"},
            NEW_FP,
            {},
            {},
            [{"run.repeat": 1, "placement": {"nodes": ["n"]}, "samples": {}}],
        )
        self.assertEqual(engine.recompute(res), [])
        edit(d / "project.yaml", "fingerprint_version: 2", "fingerprint_version: 3")
        self.assertTrue(cat(d).errors)

    def test_none_policy_digests_like_an_absent_block(self):
        d = copy_catalog("pinning-demo", "pin/none-digest")
        with open(d / "project.yaml", "a") as f:
            f.write(
                "---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-a-none}\n"
                "spec: {base: node-a-unpinned, patch: {host: {cpu_pinning: {policy: none}}}}\n"
            )
        c = cat(d)
        self.assertEqual(c.errors, [])
        self.assertEqual(c.digest("Environment", "node-a-none"), c.digest("Environment", "node-a-unpinned"))
        self.assertEqual(c.env_view("node-a-none")["spec_digest"], c.env_view("node-a-unpinned")["spec_digest"])
        self.assertEqual(c.env_view("node-a-none")["labels"], c.env_view("node-a-unpinned")["labels"])


class AnalysisHardening(unittest.TestCase):
    def spec(self, **over):
        return {"select": {"benchmarks": ["decode-probe"]}, "metric": M, **over}

    def cmp_spec(self, **over):
        return self.spec(compare={"by": PIN, "metric": M, "baseline": "none", **over})

    def test_compare_uses_the_compared_metrics_own_values(self):
        r = mk_result("c", [[2.0], [2.0]])
        for rep in r["status"]["repeats"]:
            rep["samples"]["llm.itl.median"] = [0.5]
        r["status"]["aggregate"]["metrics"]["llm.itl.median"] = {"value": 0.5, "unit": "s"}
        cells = analysis.analyze(self.spec(compare={"by": PIN, "metric": "llm.itl.median"}), [r])["compare"]
        self.assertEqual((cells["cells"][0]["median"], cells["unit"]), (0.5, "s"))

    def test_a_cell_never_pools_incomparable_or_repeated_results(self):
        base = mk_result("c", [[1.0], [1.1]])
        for name, other, want in [
            ("node", mk_result("c", [[1.0]], node="n-2"), "different node"),
            ("subject", mk_result("c", [[1.0]], subject="s-2"), "different subject"),
            ("env", mk_result("c", [[1.0]], env_id="env-2"), "different env_id"),
            ("repeated", mk_result("c", [[1.0]]), "repeated Results of one case"),
            ("cases", mk_result("d", [[1.0]]), "different cases"),
        ]:
            with self.subTest(name), self.assertRaises(analysis.AnalysisError) as cm:
                analysis.analyze(self.cmp_spec(), [base, other])
            self.assertIn(want, str(cm.exception))
        rep = analysis.analyze(self.cmp_spec(across=["env.node"]), [base, mk_result("c", [[1.0]], node="n-2")])
        self.assertEqual(sorted(c["env.node"] for c in rep["compare"]["cells"]), ["n-1", "n-2"])

    def test_missing_repeat_props_count_as_the_label_policy(self):
        long = mk_result("long", [[13.0]] * 3, props=False)
        short = mk_result("short", [[1.0]] * 3)
        self.assertEqual(analysis.incomparable(long, short), [])
        self.assertIn(
            "different cpu pinning", analysis.incomparable(long, mk_result("short", [[1.0]] * 3, pin="explicit"))
        )
        self.assertEqual(analysis.derive_pair(SLOPE, M, long, short)["n"], 3)

    def test_repeat_props_override_run_props_and_fill_in_when_absent(self):
        both = mk_result("c", [[1.0], [1.0]], pin="device_local", run_props={PIN: "none"})
        only_run = mk_result("c", [[1.0]], props=False, run_props={PIN: "explicit"})
        cells = analysis.analyze(self.cmp_spec(baseline="explicit"), [both, only_run])["compare"]["cells"]
        self.assertEqual(sorted(c[PIN] for c in cells), ["device_local", "explicit"])

    def test_pairs_follow_run_repeat_not_position(self):
        first = mk_result("long", [[13.0], [13.2], [12.8]], repeat_ids=[1, 2, 4])
        second = mk_result("short", [[1.0], [2.0], [1.0]], repeat_ids=[1, 3, 4])
        d = analysis.derive_pair(SLOPE, M, first, second)
        self.assertEqual(d["repeats"], [1, 4])
        self.assertAlmostEqual(d["values"][0], 12.0 / 128)
        self.assertAlmostEqual(d["values"][1], 11.8 / 128)

    def test_failed_repeats_are_excluded(self):
        first = mk_result("long", [[13.0], [99.0], [13.0]], failed={2})
        second = mk_result("short", [[1.0]] * 3)
        self.assertEqual(analysis.repeat_values(first, M), {1: 13.0, 3: 13.0})
        self.assertEqual(analysis.derive_pair(SLOPE, M, first, second)["repeats"], [1, 3])

    def test_cv_uses_the_magnitude_of_the_mean(self):
        d = analysis.derive_pair(
            {**SLOPE, "op": "difference"}, M, mk_result("long", [[1.0], [1.2]]), mk_result("short", [[3.0], [3.0]])
        )
        self.assertAlmostEqual(d["mean"], -1.9)
        self.assertAlmostEqual(d["cv_pct"], statistics.stdev([-2.0, -1.8]) / 1.9 * 100)
        self.assertGreater(d["cv_pct"], 0)

    def test_zero_denominators_skip_the_pair_with_a_count_and_a_warning(self):
        ratio = {**SLOPE, "op": "ratio"}
        warn = []
        d = analysis.derive_pair(ratio, M, mk_result("long", [[2.0], [4.0]]), mk_result("short", [[0.0], [2.0]]), warn)
        self.assertEqual((d["n"], d["values"], d["skipped_pairs"], d["repeats"]), (1, [2.0], 1, [2]))
        self.assertTrue(any("zero denominator" in w for w in warn))
        rep = analysis.analyze(
            self.spec(derive=ratio), [mk_result("long", [[2.0], [4.0]]), mk_result("short", [[0.0], [2.0]])]
        )
        self.assertTrue(rep["warnings"])
        with self.assertRaises(analysis.AnalysisError) as cm:
            analysis.derive_pair(ratio, M, mk_result("long", [[2.0]]), mk_result("short", [[0.0]]))
        self.assertIn("no usable repeat pair", str(cm.exception))

    def test_non_finite_samples_are_dropped_with_a_warning(self):
        nan = float("nan")
        long = mk_result("long", [[13.0, nan], [nan], [12.8], [float("inf")]])
        short = mk_result("short", [[1.0]] * 4)
        warn = []
        self.assertEqual(analysis.repeat_values(long, M, warn), {1: 13.0, 3: 12.8})
        self.assertEqual(len(warn), 3)
        rep = analysis.analyze(self.spec(derive=SLOPE), [long, short])
        self.assertEqual(rep["derive"]["pairs"][0]["repeats"], [1, 3])
        self.assertTrue(any("non-finite" in w for w in rep["warnings"]))

    def test_baseline_must_exist(self):
        pinned = [mk_result("c", [[1.0], [1.1]], pin="device_local")]
        with self.assertRaises(analysis.AnalysisError) as cm:
            analysis.analyze(self.spec(compare={"by": PIN, "metric": M}), pinned)
        self.assertIn("set compare.baseline", str(cm.exception))
        with self.assertRaises(analysis.AnalysisError):
            analysis.analyze(self.cmp_spec(baseline="explicit"), pinned)
        self.assertEqual(
            analysis.analyze(self.cmp_spec(baseline="device_local"), pinned)["compare"]["baseline"], "device_local"
        )

    def test_reading_follows_the_metric_direction(self):
        def cells(direction):
            res = DeriveAndCompare.cells(self)
            metrics = {M: {"direction": direction}} if direction else None
            rep = analysis.analyze(
                self.spec(
                    derive=SLOPE,
                    compare={"by": PIN, "across": ["env.label.node_class"], "metric": "llm.decode_per_token"},
                ),
                res,
                metrics,
            )
            return rep["compare"]

        low = cells("lower_is_better")
        self.assertEqual(low["direction"], "lower_is_better")
        pinned = {c["env.label.node_class"]: c for c in low["cells"] if c[PIN] == "device_local"}
        self.assertEqual(pinned["node-a"]["reading"], "better")
        self.assertEqual(
            {
                c["env.label.node_class"]: c["reading"]
                for c in cells("higher_is_better")["cells"]
                if c[PIN] == "device_local"
            }["node-a"],
            "worse",
        )
        none = cells(None)
        self.assertIsNone(none["direction"])
        self.assertTrue(all(c["reading"] is None for c in none["cells"]))

    def test_noise_rule_uses_the_larger_cv_and_includes_the_boundary(self):
        def rows(values_by_pin):
            return [
                {
                    "labels": {},
                    "obs": {},
                    "props": {PIN: pin},
                    "values": v,
                    "results": [mk_result("c", [[1.0]], pin=pin)],
                }
                for pin, v in values_by_pin.items()
            ]

        data = rows({"none": [1.0, 2.0], "device_local": [3.0, 4.0]})

        def run(base_cv, cell_cv):
            table = {(1.0, 2.0): (100.0, 100.0, base_cv), (3.0, 4.0): (105.0, 105.0, cell_cv)}
            with mock.patch.object(analysis, "spread", lambda v: table[tuple(v)]):
                return analysis.compare({"by": PIN, "metric": M, "baseline": "none"}, data)["cells"]

        cell = next(c for c in run(1.0, 10.0) if c[PIN] == "device_local")
        self.assertAlmostEqual(cell["effect_pct"], 5.0)
        self.assertIs(cell["within_noise"], True)  # max(10, 1) = 10 >= 5; min would say False
        cell = next(c for c in run(10.0, 1.0) if c[PIN] == "device_local")
        self.assertIs(cell["within_noise"], True)  # the baseline's CV counts too
        edge = cell["effect_pct"]
        cell = next(c for c in run(0.5, edge) if c[PIN] == "device_local")
        self.assertIs(cell["within_noise"], True)  # |effect| == CV is within noise
        cell = next(c for c in run(0.5, edge - 1e-9) if c[PIN] == "device_local")
        self.assertIs(cell["within_noise"], False)


class PlacementRound2(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=TMP))
        self.proc, self.sysd = self.tmp / "proc", self.tmp / "sys"
        self.proc.mkdir()
        fake_proc(self.proc, {100: (5, "serve", {100: "0-3"}, {"cgroup": "/"}), 200: (100, "worker", {200: "0-3"})})
        (self.proc / "100/cgroup").write_text("0::/\n")

    def verify(self, request=EXPLICIT, **kw):
        return placement.verify_pinning(request, 100, (), self.proc, self.sysd, **kw)

    def test_root_cgroup_cannot_tie_a_reparented_orphan_so_it_is_unknown(self):
        self.assertEqual(self.verify()["run.exec.pinning_verified"], "true")
        fake_proc(self.proc, {600: (1, "daemonised", {600: "0-31"})})
        r = self.verify()
        self.assertEqual(r["run.exec.pinning_verified"], "unknown")
        self.assertNotIn("run.exec.pinning_violations", r)
        (self.proc / "100/cgroup").write_text("0::/pod\n")
        (self.proc / "600/cgroup").write_text("0::/pod\n")
        self.assertEqual(self.verify()["run.exec.pinning_verified"], "false")

    def test_spawn_time_starttime_must_match(self):
        self.assertEqual(self.verify(expected_starttime=1000)["run.exec.pinning_verified"], "true")
        self.assertEqual(self.verify(expected_starttime="1000")["run.exec.pinning_verified"], "true")
        self.assertEqual(self.verify(expected_starttime=999)["run.exec.pinning_verified"], "unknown")
        self.assertEqual(self.verify(expected_starttime=0)["run.exec.pinning_verified"], "unknown")

    def test_stat_starttime_is_field_22_and_short_stats_are_unreadable(self):
        self.assertEqual(placement._stat(self.proc, 100)[3], 1000)
        (self.proc / "100/stat").write_text("100 (a) S 5 100 100\n")
        with self.assertRaises(ValueError):
            placement._stat(self.proc, 100)
        self.assertEqual(self.verify()["run.exec.pinning_verified"], "unknown")

    def test_two_mask_lines_are_unreadable(self):
        (self.proc / "200/task/200/status").write_text("Cpus_allowed_list:\t0-3\nCpus_allowed_list:\t0-3\n")
        self.assertEqual(self.verify()["run.exec.pinning_verified"], "unknown")
        (self.proc / "200/task/200/status").write_text("Name:\tx\n")
        self.assertEqual(self.verify()["run.exec.pinning_verified"], "unknown")

    def test_multi_device_numa_encoding(self):
        fake_sys(self.sysd, {0: "0-3", 1: "4-7"})
        local = lambda numa: placement.resolve_requested({"policy": "device_local"}, numa, self.sysd)  # noqa: E731
        per = lambda numa: placement.resolve_requested({"policy": "per_device"}, numa, self.sysd)  # noqa: E731
        want = ([frozenset(range(8))], "device_numa")
        self.assertEqual(local("0,1"), want)
        self.assertEqual(local("0, 1"), want)
        self.assertEqual(local([0, 1]), want)
        self.assertEqual(local(0), ([frozenset(range(4))], "device_numa"))
        self.assertEqual(per("0,1"), ([frozenset(range(4)), frozenset(range(4, 8))], "device_numa"))
        for bad in ("0,x", "", "-1", "0,", "2", None, []):
            self.assertEqual(local(bad), (None, "unresolved"), bad)

    def test_trailing_newline_in_cpus_is_rejected_everywhere(self):
        with self.assertRaises(ValueError):
            placement.parse_cpu_list("0-3\n")
        d = copy_catalog("pinning-demo", "pin/newline")
        edit(d / "project.yaml", "policy: device_local, require_verified: true", 'policy: explicit, cpus: "0-3\\n"')
        errs = cat(d).errors
        self.assertTrue(any("bad CPU list" in e for e in errs), errs)


class EngineRound2(unittest.TestCase):
    def test_malformed_pinning_block_is_reported_not_a_crash(self):
        for name, new in [("no-policy", "{cpus: '0-3'}"), ("string", "device_local"), ("list", "[1]")]:
            with self.subTest(name):
                d = copy_catalog("pinning-demo", "pin/malformed-" + name)
                edit(d / "project.yaml", "{policy: device_local, require_verified: true}", new)
                errs = engine.Catalog([d], [], []).errors
                self.assertTrue(errs)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(val.main([str(d)]), 2)

    def test_none_policy_normalises_with_a_false_require_verified(self):
        d = copy_catalog("pinning-demo", "pin/none-forms")
        with open(d / "project.yaml", "a") as f:
            f.write(
                "---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: node-a-none-false}\n"
                "spec: {base: node-a-unpinned, patch: {host: {cpu_pinning: {policy: none, require_verified: false}}}}\n"
            )
        c = cat(d)
        self.assertEqual(c.errors, [])
        self.assertEqual(c.digest("Environment", "node-a-none-false"), c.digest("Environment", "node-a-unpinned"))

    def test_device_policies_need_an_accelerator_request(self):
        d = copy_catalog("pinning-demo", "pin/no-accel")
        cpu_only = (
            "devices: {requests: [{name: sockets, device_class: cpu/socket, count: 1}]}, labels: {device.count: 0}"
        )
        with open(d / "project.yaml", "a") as f:
            for name, pol in (("dl", "device_local"), ("pd", "per_device"), ("ex", "explicit, cpus: '0-3'")):
                f.write(
                    "---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: cpu-" + name + "}\n"
                    "spec: {base: node-a-unpinned, patch: {"
                    + cpu_only
                    + ", host: {cpu_pinning: {policy: "
                    + pol
                    + "}}}}\n"
                )
        errs = cat(d).errors
        self.assertTrue(any("cpu-dl" in e and "needs an accelerator request" in e for e in errs), errs)
        self.assertTrue(any("cpu-pd" in e and "needs an accelerator request" in e for e in errs), errs)
        self.assertFalse(any("cpu-ex" in e for e in errs), errs)

    def test_build_result_takes_the_pinning_request_from_the_environment(self):
        d = copy_catalog("central", "pin/central-pinned")
        with open(d / "environments.yaml", "a") as f:
            f.write(
                "---\napiVersion: bench/v1alpha1\nkind: Environment\nmetadata: {name: ref-1card-pinned}\n"
                "spec: {base: ref-1card, patch: {host: {cpu_pinning: {policy: explicit, cpus: '0-3', require_verified: true}}}}\n"
            )
        c = cat("llm-demo", central=[d])
        self.assertEqual(c.errors, [])
        case = c.cases[0]
        pinned = {**case, "environment": "ref-1card-pinned"}
        key = "run.exec.pinning_verified"

        def state(case_, verdict, **kw):
            reps = [
                {
                    "run.repeat": 1,
                    "placement": {"nodes": ["n"]},
                    "samples": {"llm.ttft.p99": [1.0]},
                    **({"props": {key: verdict}} if verdict else {}),
                }
            ]
            res = engine.build_result(
                c,
                case_,
                "x",
                {"artifact_id": "a"},
                OLD_FP,
                {},
                {},
                reps,
                validity=("valid", [], "nightly", "internal"),
                **kw,
            )["status"]["validity"]
            return res["state"], res["reasons"]

        self.assertEqual(state(pinned, "false"), ("suspect", ["pinning_violated"]))
        self.assertEqual(state(pinned, None), ("suspect", ["pinning_unverified"]))
        self.assertEqual(state(pinned, "true"), ("valid", []))
        self.assertEqual(state(pinned, "false", pinning={}), ("valid", []))
        self.assertEqual(state(case, "false"), ("valid", []))

    def test_a_placement_reason_never_yields_regressed_or_improved(self):
        c = cat("llm-demo")
        exp = c.by[("Expectation", "itl-no-regression")].spec

        def judge(reasons, current=(2.0, 2.0, 2.0)):
            return engine.judge(
                exp,
                "e" * 32,
                "v" * 32,
                "lower_is_better",
                list(current),
                [1.0] * 10,
                validity="suspect",
                floor="suspect",
                reasons=reasons,
            )

        self.assertEqual(judge(["cv_above_max"])["status"], "regressed")
        self.assertEqual(judge([])["status"], "regressed")
        for reason in ("pinning_violated", "pinning_unverified"):
            j = judge(["cv_above_max", reason])
            self.assertEqual((j["status"], j["reason"]), ("suspected", reason))
        self.assertEqual(judge(["pinning_violated"], (0.5, 0.5, 0.5))["status"], "suspected")


class AnalysisRound2(unittest.TestCase):
    def spec(self, **over):
        return {"select": {"benchmarks": ["decode-probe"]}, "metric": M, **over}

    def cmp_spec(self, **over):
        return self.spec(compare={"by": PIN, "metric": M, "baseline": "none", **over})

    def test_baseline_and_level_cells_must_be_comparable(self):
        base = dict(subject="s-1", env_id="e1", node="n1")
        for name, other in [
            ("everything", dict(subject="s-2", env_id="e2", node="n2")),
            ("subject", dict(subject="s-2", env_id="e1", node="n1")),
            ("env", dict(subject="s-1", env_id="e2", node="n1")),
            ("node", dict(subject="s-1", env_id="e1", node="n2")),
        ]:
            with self.subTest(name), self.assertRaises(analysis.AnalysisError) as cm:
                analysis.analyze(
                    self.cmp_spec(across=[]),
                    [
                        mk_result("c", [[1.0], [1.1]], **base),
                        mk_result("c", [[1.0], [1.1]], pin="device_local", **other),
                    ],
                )
            self.assertIn("not comparable", str(cm.exception))
        ok = analysis.analyze(
            self.cmp_spec(across=[]),
            [mk_result("c", [[1.0], [1.1]], **base), mk_result("c", [[1.0], [1.1]], pin="device_local", **base)],
        )
        self.assertEqual(len(ok["compare"]["cells"]), 2)

    def test_a_result_whose_repeats_disagree_on_pinning_is_refused(self):
        mixed = mk_result("c", [[1.0], [1.1]])
        mixed["status"]["repeats"][1]["props"] = {PIN: "device_local"}
        with self.assertRaises(analysis.AnalysisError) as cm:
            analysis.analyze(self.cmp_spec(), [mixed])
        self.assertIn("disagree on cpu pinning", str(cm.exception))
        long, short = mk_result("long", [[13.0], [13.1]]), mk_result("short", [[1.0], [1.0]])
        for r in (long, short):
            r["status"]["repeats"][1]["props"] = {PIN: "device_local"}
        with self.assertRaises(analysis.AnalysisError) as cm:
            analysis.derive_pair(SLOPE, M, long, short)
        self.assertIn("mixed cpu pinning", str(cm.exception))
        failed = mk_result("c", [[1.0], [1.1]], failed={2})
        failed["status"]["repeats"][1]["props"] = {PIN: "device_local"}
        self.assertEqual(analysis._pinnings(failed), {"none"})

    def test_failed_repeat_props_do_not_pick_the_cell(self):
        r = mk_result("c", [[1.0], [1.1]], pin="device_local", failed={1})
        r["status"]["repeats"][0]["props"] = {PIN: "explicit"}
        cells = analysis.analyze(self.cmp_spec(baseline="device_local"), [r])["compare"]["cells"]
        self.assertEqual([c[PIN] for c in cells], ["device_local"])

    def test_derived_overflow_is_a_skipped_pair(self):
        diff = {**SLOPE, "op": "difference"}
        warn = []
        d = analysis.derive_pair(
            diff, M, mk_result("long", [[1e308], [13.0]]), mk_result("short", [[-1e308], [1.0]]), warn
        )
        self.assertEqual((d["values"], d["skipped_pairs"], d["repeats"]), ([12.0], 1, [2]))
        self.assertTrue(any("non-finite result" in w for w in warn))

    def test_malformed_results_give_a_clean_cli_error(self):
        bad_props = mk_result("decode-probe/out=out129", [[13.0]])
        bad_props["status"]["repeats"][0]["props"] = "abc"
        no_labels = mk_result("decode-probe/out=out129", [[13.0]])
        no_labels["status"]["environment"]["labels"] = None
        short = mk_result("decode-probe/out=out1", [[1.0]])
        for name, results in [("scalars", [1, "x"]), ("props", [bad_props, short]), ("labels", [no_labels, short])]:
            with self.subTest(name):
                rc, out, err = PinningExample.run_cli(self, results)
                self.assertEqual((rc, out), (2, ""))
                self.assertTrue(err.startswith("error:"), err)
                self.assertNotIn("Traceback", err)

    def test_duplicate_repeat_numbers_are_refused(self):
        r = mk_result("c", [[1.0], [2.0]], repeat_ids=[1, 1])
        with self.assertRaises(analysis.AnalysisError) as cm:
            analysis.repeat_values(r, M)
        self.assertIn("appears twice", str(cm.exception))
        with self.assertRaises(analysis.AnalysisError):
            analysis.analyze(self.cmp_spec(), [r])

    def test_single_value_cells_report_effect_but_no_reading(self):
        res = [mk_result("c", [[2.0]]), mk_result("c", [[1.0]], pin="device_local")]
        cells = analysis.analyze(self.cmp_spec(), res, {M: {"direction": "lower_is_better"}})["compare"]["cells"]
        cell = next(c for c in cells if c[PIN] == "device_local")
        self.assertAlmostEqual(cell["effect_pct"], -50.0)
        self.assertEqual((cell["within_noise"], cell["reading"]), (None, None))

    def test_missing_pinning_label_is_none_when_comparing(self):
        a, b = mk_result("a", [[1.0]]), mk_result("b", [[1.0]])
        del a["status"]["environment"]["labels"][engine.PIN_LABEL]
        self.assertEqual(analysis.incomparable(a, b), [])
        self.assertEqual(
            analysis.incomparable(a, mk_result("b", [[1.0]], pin="device_local")).count("different environment labels"),
            1,
        )

    def test_pinning_falls_back_to_the_label_for_a_repeat_without_the_property(self):
        pinned = mk_result("a", [[1.0]], pin="device_local", props=False)
        self.assertEqual(analysis._pinnings(pinned), {"device_local"})
        self.assertEqual(analysis.incomparable(pinned, mk_result("b", [[1.0]], pin="device_local")), [])
        self.assertIn("different cpu pinning", analysis.incomparable(pinned, mk_result("b", [[1.0]], pin="none")))

    def test_numeric_edges(self):
        self.assertEqual(analysis.spread([-1.0, 1.0])[2], None)
        self.assertEqual(analysis.spread([0.0, 0.0])[2], None)
        self.assertEqual(analysis.spread([2.0])[2], None)
        self.assertIsNone(analysis._reading(0, "lower_is_better"))
        self.assertIsNone(analysis._reading(0.0, "higher_is_better"))
        self.assertEqual(analysis._reading(-1.0, "lower_is_better"), "better")
        self.assertEqual(analysis._reading(1.0, "higher_is_better"), "better")
        self.assertFalse(analysis._finite(True))
        self.assertTrue(analysis._finite(1))
        warn = []
        self.assertEqual(analysis.repeat_values(mk_result("c", [[True, 2.0]]), M, warn), {1: 2.0})
        self.assertEqual(len(warn), 1)
        self.assertEqual(analysis.repeat_values(mk_result("c", [[1.0, 2.0, 9.0]]), M), {1: 2.0})

    def test_a_negative_baseline_median_keeps_the_effect_sign(self):
        rows = [
            {"labels": {}, "obs": {}, "props": {PIN: pin}, "values": v, "results": [mk_result("c", [[1.0]], pin=pin)]}
            for pin, v in (("none", [-4.0, -2.0]), ("device_local", [-2.0, -1.0]))
        ]
        cell = next(c for c in analysis.compare({"by": PIN, "metric": M}, rows)["cells"] if c[PIN] == "device_local")
        self.assertAlmostEqual(cell["effect_pct"], 50.0)

    def test_a_non_finite_aggregate_falls_back_to_the_repeat_median(self):
        long, short = mk_result("long", [[13.0], [13.2], [12.8]]), mk_result("short", [[1.0]] * 3)
        long["status"]["aggregate"]["metrics"][M]["value"] = float("nan")
        self.assertAlmostEqual(analysis.derive_pair(SLOPE, M, long, short)["of_aggregates"], 12.0 / 128)

    def test_compare_by_must_not_also_be_across(self):
        d = copy_catalog("pinning-demo", "dc/by-in-across")
        edit(d / "analysis/pinning-effect.yaml", "across: [env.label.node_class]", "across: [run.exec.cpu_pinning]")
        errs = cat(d).errors
        self.assertTrue(any("compare.by must not also be in across" in e for e in errs), errs)


class Negative(unittest.TestCase):
    def test_negative_cases(self):
        for case, name, rel, old, new, want in NEG:
            with self.subTest(case=case):
                d = copy_catalog(name, "neg/" + case)
                edit(d / rel, old, new)
                packs, central = CATALOGS[name]
                errs = engine.Catalog([d], packs, central).errors
                self.assertTrue(any(want in e for e in errs), f"{case}: want {want!r}, got {errs}")


if __name__ == "__main__":
    unittest.main()

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
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from spyre_bench import demo, engine, gen_schemas
from spyre_bench import explain as ex
from spyre_bench import validate as val
from spyre_bench.cli import main as cli_main

EXT = Path(__file__).resolve().parent.parent
PKG = EXT / "spyre_bench"
X = EXT / "examples"
PG = X / "packs" / "postgres-example"
CENTRAL = X / "central"
TMP = Path(tempfile.mkdtemp(prefix="spyre-bench-tests-"))
CATALOGS = {
    "llm-demo": ([], [CENTRAL]),
    "hf-demo": ([], [CENTRAL]),
    "kernel-demo": ([], [CENTRAL]),
    "comms-demo": ([], [CENTRAL]),
    "central": ([], []),
    "postgres-demo": ([PG], []),
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
            "central": 0,
        }
        self.assertEqual({n: len(cat(n).cases) for n in want}, want)

    def test_cli(self):
        args = [str(X / n) for n in CATALOGS if n != "central"] + ["--pack", str(PG), "--central", str(CENTRAL)]
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

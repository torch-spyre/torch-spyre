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

"""Identity tests: engine ids stay put, the `identity` block validates, the ingest id conforms, nothing needs ingest.
python -m pytest -q -p no:cacheprovider tests
"""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from spyre_bench import engine, ingest_adapter
from spyre_bench import explain as ex
from spyre_bench.cli import main as cli_main

EXT = Path(__file__).resolve().parent.parent
X = EXT / "examples"
GOLDEN = Path(__file__).resolve().parent / "golden"
CENTRAL = X / "central"
PG = X / "packs" / "postgres-example"
TMP = Path(tempfile.mkdtemp(prefix="spyre-bench-identity-"))
CATALOGS = {
    "llm-demo": ([], [CENTRAL]),
    "hf-demo": ([], [CENTRAL]),
    "kernel-demo": ([], [CENTRAL]),
    "comms-demo": ([], [CENTRAL]),
    "central": ([], []),
    "postgres-demo": ([PG], []),
    "pinning-demo": ([], []),
}
INGEST_CASES = {
    "op_matmul_m4096-n4096-k4096",
    "op_matmul_m1-n4096-k4096",
    "op_softmax_fp16",
    "op_latency_fp16",
    "op_generic_add",
}


def setUpModule():
    if not X.is_dir():
        raise unittest.SkipTest(f"examples/ not found at {X}: the tests need a source checkout, not an installed wheel")


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def load(path, packs=(), central=()):
    return engine.Catalog([path], list(packs), list(central))


def ingest_demo(path=None):
    return load(path or X / "ingest-demo", central=[CENTRAL])


def copy_ingest_demo(label):
    d = TMP / label
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(X / "ingest-demo", d)
    return d


def edit_identity(label, change):
    """A copy of ingest-demo/op-matmul whose identity block went through `change(block)`."""
    d = copy_ingest_demo(label)
    f = d / "ops" / "op-matmul.yaml"
    docs = list(yaml.safe_load_all(f.read_text()))
    for doc in docs:
        if doc["kind"] == "Benchmark":
            change(doc["spec"]["identity"])
    f.write_text(yaml.safe_dump_all(docs, sort_keys=False))
    return d


def block_errors(label, change):
    """Only the identity-block messages (cases of the edited Benchmark may legitimately collide)."""
    return "\n".join(x for x in errors_of(label, change).splitlines() if "identity:" in x)


def errors_of(label, change):
    return "\n".join(ingest_demo(edit_identity(label, change)).errors)


def sibling_ingest():
    """BenchmarkId and the ingest modules from the sibling checkout, or a skip reason."""
    repo = EXT.parent / "clickhouse-ingest"
    try:
        from spyre_clickhouse_ingest import identity

        return identity, None
    except ImportError:
        pass
    if not repo.exists():
        return None, "spyre_clickhouse_ingest is not installed and clickhouse-ingest is not alongside this package"
    sys.path.insert(0, str(repo))
    try:
        from spyre_clickhouse_ingest import identity

        return identity, None
    except Exception as x:  # noqa: BLE001 - optional dependency of the conformance tests
        return None, f"spyre_clickhouse_ingest is not importable: {x}"
    finally:
        sys.path.remove(str(repo))


def table():
    rows = []
    for ln in (GOLDEN / "ingest-ids.tsv").read_text().splitlines()[1:]:
        case, producer, component, name, tags, keys, disc, bid = ln.split("\t")
        rows.append(
            {
                "example_case": case,
                "producer": producer,
                "component": component,
                "name": name,
                "tags": tags.split(",") if tags else [],
                "disc_keys": keys.split(",") if keys else [],
                "disc": json.loads(disc),
                "benchmark_id": bid,
            }
        )
    return rows


@contextlib.contextmanager
def without_ingest():
    """Behave as an install without the ingest package, whether or not it is importable here."""
    ingest_adapter.reset()
    blocked = {"spyre_clickhouse_ingest": None, "spyre_clickhouse_ingest.identity": None}
    try:
        with mock.patch.dict(sys.modules, blocked):
            yield
    finally:
        ingest_adapter.reset()


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


def engine_rows(only=None):
    lines = ["catalog\tcase\tbenchmark_id\tseries_key\tenv_spec_digest"]
    for name, (packs, central) in CATALOGS.items():
        cat = load(X / name, packs, central)
        for c in cat.cases:
            bid = cat.ids(c)["benchmark_id"]
            sk = dig = ""
            if c["environment"]:
                view = cat.env_view(c["environment"])
                dig = view["spec_digest"]
                series = {
                    k: view["labels"][k] for k in cat.project.spec.get("series_labels", []) if k in view["labels"]
                }
                sk = engine.series_key(bid, series) if series else ""
            lines.append("\t".join([name, c["name"], bid, sk, dig]))
    return "\n".join(lines) + "\n"


class EngineIdsStayPut(unittest.TestCase):
    def test_examples_without_a_block_keep_their_ids(self):
        self.assertEqual(engine_rows(), (GOLDEN / "engine-ids.tsv").read_text())

    def test_explicit_engine_mode_changes_nothing(self):
        for name, (packs, central) in CATALOGS.items():
            if name == "central":
                continue
            d = TMP / f"explicit-{name}"
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(X / name, d)
            for f in d.rglob("*.yaml"):
                docs = list(yaml.safe_load_all(f.read_text()))
                for doc in docs:
                    if doc and doc.get("kind") == "Benchmark":
                        doc["spec"]["identity"] = {"mode": "engine"}
                f.write_text(yaml.safe_dump_all(docs, sort_keys=False))
            base, explicit = load(X / name, packs, central), load(d, packs, central)
            self.assertEqual(explicit.errors, [], name)
            self.assertTrue(explicit.cases)
            for a, b in zip(base.cases, explicit.cases):
                self.assertEqual(base.ids(a), explicit.ids(b), (name, a["name"]))
                self.assertEqual(explicit.ids(b), {"benchmark_id": explicit.engine_id(b)})
                self.assertIsNone(explicit.ingest_inputs(b))

    def test_engine_id_of_an_ingest_case_is_the_id_it_has_without_the_block(self):
        d = copy_ingest_demo("no-block")
        f = d / "ops" / "op-matmul.yaml"
        docs = list(yaml.safe_load_all(f.read_text()))
        for doc in docs:
            if doc["kind"] == "Benchmark":
                del doc["spec"]["identity"]
        f.write_text(yaml.safe_dump_all(docs, sort_keys=False))
        plain, ing = ingest_demo(d), ingest_demo()
        for a in plain.cases:
            b = next(c for c in ing.cases if c["name"] == a["name"])
            if a["benchmark"] == "op-matmul":
                self.assertEqual(plain.ids(a)["benchmark_id"], ing.engine_id(b))

    def test_result_shape_of_an_engine_case_has_no_new_keys(self):
        rows = json.loads((X / "results" / "pg-nightly.results.json").read_text())
        for r in rows:
            self.assertFalse({"id_mode", "disc_keys", "engine_id"} & set(r["status"]["case"]))


class NoCoreSchedule(unittest.TestCase):
    def test_no_schedule_field_and_none_required(self):
        schema = json.loads((EXT / "spyre_bench" / "core" / "schemas" / "benchmark.schema.json").read_text())
        self.assertNotIn("schedule", json.dumps(schema))
        spec = schema["properties"]["spec"]
        self.assertIn("est_hours", spec["properties"])
        self.assertNotIn("est_hours", spec["required"])
        for p in sorted(X.rglob("*.yaml")):
            self.assertNotIn("schedule", p.read_text().lower(), p)

    def test_schedule_is_refused_and_est_hours_checked(self):
        def edited(label, change):
            d = copy_ingest_demo(label)
            f = d / "ops" / "op-softmax.yaml"
            doc = yaml.safe_load(f.read_text())
            change(doc["spec"])
            f.write_text(yaml.safe_dump(doc, sort_keys=False))
            return "\n".join(ingest_demo(d).errors)

        self.assertIn("unknown key 'schedule'", edited("sched", lambda s: s.update(schedule={"preferred": "x"})))
        self.assertEqual(edited("hours-ok", lambda s: s.update(est_hours=2.5)), "")
        self.assertIn("est_hours", edited("hours-bad", lambda s: s.update(est_hours=-1)))


class IdentityBlock(unittest.TestCase):
    def test_example_resolves_to_ingest_style_inputs(self):
        cat = ingest_demo()
        self.assertEqual(cat.errors, [])
        self.assertEqual({c["name"] for c in cat.cases}, INGEST_CASES)
        c = next(c for c in cat.cases if c["name"] == "op_softmax_fp16")
        self.assertEqual(
            cat.ingest_inputs(c),
            {
                "component": "ingest-demo",
                "name": "softmax",
                "tags": [],
                "disc": {
                    "record_type": "model",
                    "config_name": "fp16",
                    "input_shapes": "1_512_4096",
                    "run_mode": "inference",
                    "kernel_name": "",
                    "is_total": "",
                },
                "disc_keys": list(engine.INGEST_DISC_KEYS),
                "profile": "perf-kernel",
            },
        )

    def test_engine_id_is_available_without_ingest_and_differs(self):
        cat = ingest_demo()
        with without_ingest():
            for c in cat.cases:
                self.assertTrue(cat.engine_id(c))
                self.assertEqual(
                    cat.engine_id(c),
                    engine.benchmark_id(
                        **{
                            "component": cat.identity(c)["component"],
                            "name": cat.identity(c)["name"],
                            "tags": cat.identity(c)["tags"],
                            "disc": cat.identity(c)["disc"],
                            "disc_keys": sorted(cat.identity(c)["disc"]),
                        }
                    ),
                )

    def test_id_token_separates_what_the_ingest_id_separates(self):
        cat = ingest_demo()
        tokens = {cat.id_token(c) for c in cat.cases}
        self.assertEqual(len(tokens), len(INGEST_CASES))
        rows = table()
        by_token = {}
        for r in rows:
            probe = engine.hid(
                "ingest-inputs",
                [
                    engine.norm(r["component"]),
                    engine.norm(r["name"]),
                    engine.tag_part(r["tags"]),
                    engine.disc_part(r["disc"], r["disc_keys"]),
                ],
            )
            by_token.setdefault(probe, set()).add(r["benchmark_id"])
        self.assertTrue(all(len(v) == 1 for v in by_token.values()))
        self.assertEqual(len({r["benchmark_id"] for r in rows}), len(by_token))

    def test_negatives(self):
        def drop_key(k):
            return lambda b: b.update(discriminators=[d for d in b["discriminators"] if d["key"] != k])

        def set_disc(i, kw):
            return lambda b: b["discriminators"][i].update(kw)

        cases = {
            "unknown key": (
                lambda b: b["discriminators"].append({"key": "color", "value": "x"}),
                "unknown key 'color'",
            ),
            "backend key": (
                lambda b: b["discriminators"].append({"key": "backend", "value": "x"}),
                "backend is the comparison axis",
            ),
            "duplicate key": (lambda b: b["discriminators"].append(dict(b["discriminators"][0])), "listed twice"),
            "missing key": (drop_key("kernel_name"), "hashes exactly"),
            "order": (lambda b: b["discriminators"].reverse(), "hashes exactly"),
            "no discriminators": (lambda b: b.pop("discriminators"), "hashes exactly"),
            "value and from": (set_disc(0, {"from": "case.op"}), "exactly one of value, from"),
            "neither": (lambda b: b["discriminators"][0].pop("value"), "exactly one of value, from"),
            "join without from": (set_disc(0, {"join": "_"}), "join needs from"),
            "join of a scalar": (set_disc(2, {"from": "case.op"}), "join needs a list value"),
            "backend value": (set_disc(2, {"from": "system.backend"}), "system.backend cannot feed an id"),
            "unknown from": (set_disc(2, {"from": "case.nope"}), "is not an identity value of this case"),
            "from prefix": (set_disc(2, {"from": "run.tier"}), "must start with"),
            "tier tag": (lambda b: b.update(tags=["tier"]), "starts with a run-level grouping"),
            "tier-prefixed tag": (lambda b: b.update(tags=["tier__core"]), "starts with a run-level grouping"),
            "suite tag": (lambda b: b.update(tags=["suite.ops"]), "starts with a run-level grouping"),
            "run tag": (lambda b: b.update(tags=["run.profile"]), "starts with a run-level grouping"),
            "backend tag": (lambda b: b.update(tags=["backend__spyre"]), "backend"),
            "tier value tag": (lambda b: b.update(tags=["core"]), "grouping value"),
            "suite name tag": (lambda b: b.update(tags=["ops"]), "grouping value"),
            "tier hyphen tag": (lambda b: b.update(tags=["tier-core"]), "starts with a run-level grouping"),
            "tier underscore tag": (lambda b: b.update(tags=["tier_core"]), "starts with a run-level grouping"),
            "tier equals tag": (lambda b: b.update(tags=["tier=core"]), "starts with a run-level grouping"),
            "suite colon tag": (lambda b: b.update(tags=["suite:ops"]), "starts with a run-level grouping"),
            "suite dot tag": (lambda b: b.update(tags=["suite.ops"]), "starts with a run-level grouping"),
            "padded tag": (lambda b: b.update(tags=[" X "]), "does not match"),
            "list without join": (lambda b: b["discriminators"][2].pop("join"), "needs `join`"),
            "empty component": (lambda b: b.update(component=""), "component"),
            "empty name": (lambda b: b.update(name=""), "name"),
            "unknown profile": (lambda b: b.update(profile="other"), "profile"),
            "cadence tag": (lambda b: b.update(tags=["cadence__nightly"]), "run-context or result tag"),
            "legacy alias tag": (lambda b: b.update(tags=["nightly"]), "run-context or result tag"),
            "result tag": (lambda b: b.update(tags=["refcoverage__x"]), "run-context or result tag"),
            "duplicate tag": (lambda b: b.update(tags=["domain__a", "DOMAIN__A"]), "listed twice"),
            "bad tag": (lambda b: b.update(tags=["Bad Tag"]), "does not match"),
            "engine with extras": (lambda b: b.update(mode="engine"), "need mode: ingest"),
            "no mode": (lambda b: b.pop("mode"), "'mode'"),
            "unknown mode": (lambda b: b.update(mode="auto"), "mode"),
            "unknown field": (lambda b: b.update(schedule={}), "unknown key 'schedule'"),
        }
        for label, (change, want) in cases.items():
            with self.subTest(label):
                got = errors_of("neg-" + label.replace(" ", "-"), change)
                self.assertIn(want, got)

    def test_project_grouping_names_are_refused_as_tags(self):
        d = edit_identity("grouping-tag", lambda b: b.update(tags=["domain__a"]))
        p = d / "project.yaml"
        docs = list(yaml.safe_load_all(p.read_text()))
        docs[0]["spec"]["groupings"] = [
            {"name": "tier", "ordered": True, "cardinality": "one", "values": ["core", "sanity"]},
            {"name": "domain", "ordered": False, "cardinality": "many"},
        ]
        p.write_text(yaml.safe_dump_all(docs, sort_keys=False))
        got = "\n".join(ingest_demo(d).errors)
        self.assertIn("'domain__a' starts with a run-level grouping", got)

    def test_same_ingest_inputs_are_a_duplicate_even_if_engine_ids_differ(self):
        got = errors_of("dup", lambda b: b["discriminators"].__setitem__(2, {"key": "input_shapes", "value": "x"}))
        self.assertIn("same benchmark_id as case", got)

    def test_identity_change_is_reported_by_validate_without_ingest(self):
        base = TMP / "base-dir"
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir()
        shutil.copytree(X / "ingest-demo", base / "ingest-demo")
        cur = edit_identity(
            "changed", lambda b: b["discriminators"].__setitem__(3, {"key": "run_mode", "value": "other"})
        )
        shutil.rmtree(TMP / "changed-cur", ignore_errors=True)
        (TMP / "changed-cur").mkdir()
        shutil.move(str(cur), str(TMP / "changed-cur" / "ingest-demo"))
        with without_ingest():
            code, out, _ = run_cli(
                "validate", str(TMP / "changed-cur" / "ingest-demo"), "--central", str(CENTRAL), "--base-dir", str(base)
            )
        self.assertEqual(code, 0)
        self.assertIn("a new series starts", out)


class WithoutIngest(unittest.TestCase):
    """CI without the ingest package: everything but the stored id works."""

    def test_validate_and_explain(self):
        with without_ingest():
            self.assertFalse(ingest_adapter.available())
            code, out, _ = run_cli("validate", str(X / "ingest-demo"), "--central", str(CENTRAL))
            self.assertEqual((code, "PASS" in out), (0, True))
            docs = ex.explain(
                ingest_demo(),
                "op_softmax_fp16",
                mock.Mock(environment=None, purpose="investigate", subject=None, set=[]),
            )
            d = docs[0]
            self.assertIsNone(d["benchmark_id"])
            self.assertIn("spyre-bench[ingest]", d["id_unavailable"])
            self.assertEqual(d["ingest_identity"]["name"], "softmax")
            self.assertIn("error", d["plan"])
            self.assertIsNone(d["series_key"])
            code, out, err = run_cli(
                "explain", "op_softmax_fp16", "--catalog", str(X / "ingest-demo"), "--central", str(CENTRAL), "--json"
            )
            self.assertEqual(code, 0, err)
            self.assertIsNone(json.loads(out)[0]["benchmark_id"])

    def test_explain_with_set_needs_no_ingest(self):
        with without_ingest():
            code, out, err = run_cli(
                "explain",
                "op_softmax_fp16",
                "--catalog",
                str(X / "ingest-demo"),
                "--central",
                str(CENTRAL),
                "--set",
                "workload.shape=[1,1024,4096]",
                "--json",
            )
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)[0]["identity_diff"]["identity_changed"])

    def test_engine_mode_examples_need_nothing(self):
        with without_ingest():
            for name, (packs, central) in CATALOGS.items():
                self.assertEqual(load(X / name, packs, central).errors, [], name)
            code, out, _ = run_cli("ids", "--catalog", str(X / "pinning-demo"))
            self.assertEqual(code, 0)
            self.assertTrue(out.strip())

    def test_ids_verb(self):
        with without_ingest():
            code, out, _ = run_cli(
                "ids", "--catalog", str(X / "ingest-demo"), "--central", str(CENTRAL), "--inputs", "--json"
            )
            self.assertEqual(code, 0)
            self.assertEqual({r["case"] for r in json.loads(out)}, INGEST_CASES)
            code, _, err = run_cli("ids", "--catalog", str(X / "ingest-demo"), "--central", str(CENTRAL))
            self.assertEqual(code, 2)
            self.assertIn("spyre-bench[ingest]", err)
            code, out, _ = run_cli("ids", "--adapter")
            self.assertEqual((code, json.loads(out)["available"]), (2, False))

    def test_plan_and_recompute_refuse_loudly(self):
        cat = ingest_demo()
        leg = {"arch": "x86_64", "devices": {"spyre/pf": 8}, "nodes": 1, "reference_node": True}
        with without_ingest():
            with self.assertRaises(ingest_adapter.IngestUnavailable) as cm:
                engine.plan(cat, {"purpose": "pr", "selection": {"cases": ["op_softmax_fp16"]}}, leg)
            self.assertIn("uv pip install", str(cm.exception))
            result = {
                "status": {
                    "case": {
                        "benchmark_id": "x",
                        "id_mode": "ingest",
                        "component": "c",
                        "name": "n",
                        "tags": [],
                        "disc": {},
                        "disc_keys": [],
                    },
                    "environment": {"fingerprint": {}, "env_id": "e"},
                    "repeats": [],
                    "series": {"labels": {}, "key": "k"},
                }
            }
            with self.assertRaises(ingest_adapter.IngestUnavailable):
                engine.recompute(result)


def vllm_block(b):
    b["profile"] = "vllm"
    b["discriminators"] = [
        {"key": k, "value": "1"} for k in ("record_type", "run_mode", "tensor_parallel", "input_len", "output_len")
    ]


class Profiles(unittest.TestCase):
    def test_profile_key_lists(self):
        self.assertEqual(block_errors("vllm-ok", vllm_block), "")
        self.assertEqual(engine.ID_PROFILES["vllm-sweep"][:5], engine.ID_PROFILES["vllm"])
        self.assertIn(
            "profile vllm hashes exactly", errors_of("vllm-bad", lambda b: (vllm_block(b), b["discriminators"].pop()))
        )
        self.assertIn("never writes", errors_of("perf-short", lambda b: b["discriminators"].pop()))
        self.assertNotIn("matches no stored row", errors_of("perf-short2", lambda b: b["discriminators"].pop()))
        self.assertIn(
            "unknown key 'tensor_parallel' for profile perf-kernel",
            errors_of("perf-foreign", lambda b: b["discriminators"].append({"key": "tensor_parallel", "value": "1"})),
        )

    def test_generic_profile(self):
        def generic(keys):
            def change(b):
                b["profile"] = "generic"
                b["discriminators"] = [{"key": k, "value": "v"} for k in keys]

            return change

        self.assertEqual(block_errors("gen-ok", generic(["a", "b"])), "")
        self.assertEqual(block_errors("gen-empty", generic([])), "")
        self.assertIn("sorted prop names", errors_of("gen-unsorted", generic(["b", "a"])))
        self.assertIn("listed twice", errors_of("gen-dup", generic(["a", "a"])))
        self.assertIn("backend is the comparison axis", errors_of("gen-backend", generic(["backend"])))

    def test_example_profiles_resolve(self):
        cat = ingest_demo()
        by = {c["name"]: cat.ingest_inputs(c) for c in cat.cases}
        self.assertEqual(by["op_latency_fp16"]["disc_keys"], list(engine.ID_PROFILES["vllm"]))
        self.assertEqual(by["op_generic_add"]["disc_keys"], ["iterations", "op"])
        self.assertEqual({a["profile"] for a in by.values()}, {"perf-kernel", "vllm", "generic"})

    def test_id_token_follows_the_joined_string_the_ingest_hashes(self):
        d = copy_ingest_demo("collide")
        ops = d / "ops"
        for name, cfg, shp in (("op-softmax", "p,input_shapes=q", "r"), ("op-softmax2", "p", "q,input_shapes=r")):
            src = yaml.safe_load((X / "ingest-demo/ops/op-softmax.yaml").read_text())
            src["metadata"]["name"] = name
            src["spec"]["variants"] = [{"name": name.replace("-", "_") + "_fp16"}]
            src["spec"]["identity"]["name"] = "x"
            src["spec"]["identity"]["discriminators"][1] = {"key": "config_name", "value": cfg}
            src["spec"]["identity"]["discriminators"][2] = {"key": "input_shapes", "value": shp}
            (ops / f"{name}.yaml").write_text(yaml.safe_dump(src, sort_keys=False))
        self.assertIn("same benchmark_id as case", "\n".join(ingest_demo(d).errors))

    def test_id_token_normalises_tags_and_ignores_order(self):
        cat = ingest_demo()
        c = next(c for c in cat.cases if c["name"] == "op_generic_add")

        def token(tags):
            return cat.id_token({**c, "identity": {**c["identity"], "tags": tags}})

        self.assertEqual(token(["domain__a", "domain__b"]), token(["domain__b", "Domain__A", "domain__a"]))
        self.assertNotEqual(token(["domain__a"]), token(["domain__b"]))

    def test_component_override(self):
        cat = ingest_demo(edit_identity("comp", lambda b: b.update(component="other-comp")))
        self.assertEqual(
            {cat.ingest_inputs(c)["component"] for c in cat.cases if c["benchmark"] == "op-matmul"}, {"other-comp"}
        )


class Probe(unittest.TestCase):
    @staticmethod
    def derive_with(**mut):
        """The ingest rule re-implemented, with the named deviations switched on."""
        import unicodedata

        def val(v, strip=True, lower=True):
            if mut.get("bool_int") and isinstance(v, bool):
                v = int(v)
            if mut.get("float_int") and isinstance(v, float) and v.is_integer():
                v = int(v)
            v = "" if v is None else str(v)
            if strip:
                v = (
                    v.strip(" ")
                    if mut.get("strip_spaces_only")
                    else v.rstrip()
                    if mut.get("rstrip_only")
                    else v.strip()
                )
            v = v.lower() if lower else v
            if mut.get("casefold"):
                v = v.casefold()
            if mut.get("nfkc"):
                v = unicodedata.normalize("NFKC", v)
            if mut.get("nfc"):
                v = unicodedata.normalize("NFC", v)
            if mut.get("collapse"):
                v = " ".join(v.split())
            if mut.get("nl_replace"):
                v = v.replace("\n", " ")
            return v

        def derive(component, name, tags, disc=None, disc_keys=()):
            ts = [
                val(t, strip=not mut.get("tag_nostrip"), lower=not mut.get("tag_nolower"))
                for t in tags
                if not (mut.get("ignore_nonstr_tags") and not isinstance(t, str))
            ]
            if not mut.get("keep_empty"):
                ts = [t for t in ts if t]
            if mut.get("alias"):
                ts = [{"nightly": "cadence__nightly"}.get(t, t) for t in ts]
            if mut.get("drop_refcov"):
                ts = [t for t in ts if not t.startswith("refcoverage__")]
            if mut.get("rstrip_underscore"):
                ts = [t.rstrip("_") for t in ts]
            if mut.get("tag_nosort"):
                ts = list(dict.fromkeys(ts))
            elif mut.get("tag_nodedupe"):
                ts = sorted(ts)
            else:
                ts = sorted(set(ts))
            if mut.get("split"):
                ts = [t for t in ts if not t.startswith("cadence__")]
            keys = list(disc_keys)
            if mut.get("sort_keys"):
                keys = sorted(keys)
            if mut.get("dedupe_keys"):
                keys = list(dict.fromkeys(keys))
            if mut.get("skip_absent"):
                keys = [k for k in keys if k in disc]
            rawkey = (lambda k: k.strip()) if mut.get("key_strip") else (lambda k: k)
            rawkey = (lambda k: k.lower()) if mut.get("lower_keys") else rawkey
            vs = [val(disc.get(k)) for k in keys]
            if mut.get("trunc_value"):
                vs = [v[:256] for v in vs]
            dp = ",".join(f"{rawkey(k)}={v}" for k, v in zip(keys, vs))
            if mut.get("drop_empty_disc") and not any(vs):
                dp = ""
            nm = val(name)[:200] if mut.get("trunc_name") else val(name)
            comp = str(component) if mut.get("comp_nostrip") else val(component)
            return str(engine.uuid.uuid5(engine.ID_NAMESPACE, "|".join([comp.lower(), nm, ",".join(ts), dp])))

        return derive

    def load(self, derive, ns=engine.ID_NAMESPACE):
        from types import SimpleNamespace

        mod = SimpleNamespace(BenchmarkId=SimpleNamespace(derive=derive), ID_NAMESPACE=ns)
        ingest_adapter.reset()
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": mod}):
            return ingest_adapter._load()

    def tearDown(self):
        ingest_adapter.reset()

    def test_probe_id_is_the_engine_rule_too(self):
        for args, expected in ingest_adapter.PROBES:
            self.assertEqual(engine.benchmark_id(*args), expected)

    def test_faithful_derivation_passes_each_mutation_fails(self):
        self.assertTrue(callable(self.load(self.derive_with())))
        for mut in (
            "tag_nosort", "tag_nodedupe", "tag_nolower", "tag_nostrip", "keep_empty", "alias", "drop_refcov",
            "casefold", "nfkc", "nfc", "collapse", "lower_keys", "key_strip", "skip_absent", "sort_keys",
            "dedupe_keys", "comp_nostrip", "split", "strip_spaces_only", "bool_int", "float_int", "trunc_value",
            "trunc_name", "rstrip_only", "drop_empty_disc", "ignore_nonstr_tags", "nl_replace", "rstrip_underscore",
        ):  # fmt: skip
            with self.subTest(mut), self.assertRaises(ingest_adapter.IngestUnavailable) as cm:
                self.load(self.derive_with(**{mut: True}))
            self.assertIn("the ingest rule changed", str(cm.exception))

    def test_any_exception_from_the_derivation_is_a_rule_change_and_available_is_a_bool(self):
        for exc in (ValueError("x"), KeyError("k"), RuntimeError("r"), ZeroDivisionError()):

            def boom(*a, _e=exc, **k):
                raise _e

            with self.subTest(type(exc).__name__):
                with self.assertRaises(ingest_adapter.IngestUnavailable) as cm:
                    self.load(boom)
                self.assertIn(f"{type(exc).__name__}", str(cm.exception))
                self.assertIn("the ingest rule changed", str(cm.exception))
        from types import SimpleNamespace

        mod = SimpleNamespace(BenchmarkId=SimpleNamespace(derive=boom), ID_NAMESPACE=engine.ID_NAMESPACE)
        ingest_adapter.reset()
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": mod}):
            self.assertIs(ingest_adapter.available(), False)
            self.assertFalse(ingest_adapter.info()["available"])

    def test_signature_change_is_a_rule_change(self):
        with self.assertRaises(ingest_adapter.IngestUnavailable) as cm:
            self.load(lambda component, name: "x")
        self.assertIn("the ingest rule changed", str(cm.exception))

    def test_missing_attribute_is_not_importable(self):
        from types import SimpleNamespace

        ingest_adapter.reset()
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": SimpleNamespace(ID_NAMESPACE=1)}):
            with self.assertRaises(ingest_adapter.IngestUnavailable) as cm:
                ingest_adapter._load()
        self.assertIn("AttributeError", str(cm.exception))
        self.assertIn("uv pip install", str(cm.exception))
        self.assertNotIn("pip install 'spyre-bench[ingest]'", str(cm.exception))


class Review(unittest.TestCase):
    def test_est_hours_must_be_finite(self):
        d = copy_ingest_demo("hours")
        f = d / "ops" / "op-softmax.yaml"
        for bad in (".nan", ".inf"):
            f.write_text(
                (X / "ingest-demo/ops/op-softmax.yaml")
                .read_text()
                .replace("  type: perf", f"  est_hours: {bad}\n  type: perf", 1)
            )
            self.assertIn("est_hours", "\n".join(ingest_demo(d).errors), bad)

    def test_variant_without_ingest_reports_null_ids(self):
        cat = ingest_demo()
        with without_ingest():
            _, diff = cat.variant("op_softmax_fp16", ["workload.shape=[1,1024,4096]"])
        self.assertEqual(diff["benchmark_id"], (None, None))
        self.assertTrue(diff["identity_changed"])

    def test_ids_on_a_mixed_catalog_prints_engine_cases_and_fails_for_ingest_ones(self):
        d = copy_ingest_demo("mixed")
        f = d / "ops" / "op-softmax.yaml"
        doc = yaml.safe_load(f.read_text())
        del doc["spec"]["identity"]
        f.write_text(yaml.safe_dump(doc, sort_keys=False))
        with without_ingest():
            code, out, err = run_cli("ids", "--catalog", str(d), "--central", str(CENTRAL), "--json")
        rows = {r["case"]: r for r in json.loads(out)}
        self.assertEqual(code, 2)
        self.assertEqual(rows["op_softmax_fp16"]["mode"], "engine")
        self.assertTrue(rows["op_softmax_fp16"]["benchmark_id"])
        self.assertIn("error", rows["op_generic_add"])
        self.assertIn("uv pip install", err)

    def test_ids_filters_by_benchmark_or_case_name(self):
        args = ("--catalog", str(X / "ingest-demo"), "--central", str(CENTRAL), "--inputs", "--json")
        _, out, _ = run_cli("ids", "op-matmul", *args)
        self.assertEqual(len(json.loads(out)), 2)
        _, out, _ = run_cli("ids", "op_softmax_fp16", *args)
        self.assertEqual([r["case"] for r in json.loads(out)], ["op_softmax_fp16"])

    def test_ids_inputs_for_engine_cases_carry_tags_and_disc(self):
        _, out, _ = run_cli("ids", "--catalog", str(X / "pinning-demo"), "--inputs", "--json")
        row = json.loads(out)[0]
        self.assertEqual(set(row["inputs"]), {"component", "name", "tags", "disc"})
        self.assertTrue(row["inputs"]["disc"])

    def test_recompute_without_disc_keys_is_a_mismatch(self):
        result = {
            "status": {
                "case": {
                    "benchmark_id": "x",
                    "id_mode": "ingest",
                    "component": "c",
                    "name": "n",
                    "tags": [],
                    "disc": {},
                },
                "environment": {"fingerprint": {}, "env_id": "e"},
                "repeats": [],
                "series": {"labels": {}, "key": "k"},
            }
        }
        with without_ingest():
            self.assertIn("benchmark_id", engine.recompute(result))

    def test_base_dir_compares_ingest_inputs_not_the_engine_id(self):
        base = TMP / "base-dir2"
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir()
        shutil.copytree(X / "ingest-demo", base / "ingest-demo")
        cur = TMP / "cur-dir2"
        shutil.rmtree(cur, ignore_errors=True)
        cur.mkdir()
        shutil.copytree(X / "ingest-demo", cur / "ingest-demo")
        f = cur / "ingest-demo" / "ops" / "op-matmul.yaml"
        f.write_text(f.read_text().replace("precision: fp16", "precision: bf16"))
        a, b = ingest_demo(base / "ingest-demo"), ingest_demo(cur / "ingest-demo")
        self.assertNotEqual(a.engine_id(a.cases[0]), b.engine_id(b.cases[0]))
        code, out, _ = run_cli("validate", str(cur / "ingest-demo"), "--central", str(CENTRAL), "--base-dir", str(base))
        self.assertEqual(code, 0)
        self.assertNotIn("a new series starts", out)


def generic_keys(keys):
    def change(b):
        b["profile"] = "generic"
        b["discriminators"] = [{"key": k, "value": "v"} for k in keys]

    return change


def set_project_groupings(label, groupings):
    d = edit_identity(label, lambda b: b.update(tags=["domain__a"]))
    p = d / "project.yaml"
    docs = list(yaml.safe_load_all(p.read_text()))
    docs[0]["spec"]["groupings"] = groupings
    p.write_text(yaml.safe_dump_all(docs, sort_keys=False))
    return d


class RunLevelRules(unittest.TestCase):
    def test_generic_keys_refuse_backend_and_run_level_names(self):
        for key in ("Backend", " backend", "backend.x", "backend__x", "system.backend", "tier", "run.tier", "suite.ops",
                    "run.repeats", "tier__x", "tier=core", "Tier:x"):  # fmt: skip
            with self.subTest(key):
                got = block_errors(
                    "key-" + str(len(key)) + key.replace(" ", "_").replace(":", "-").replace("=", "-"),
                    generic_keys([key]),
                )
                self.assertRegex(got, "backend|run-level grouping")

    def test_generic_keys_that_are_not_run_level_pass(self):
        for key in ("run_mode", "system.name", "Zeta", "backends", "tiers", "input.shapes"):
            with self.subTest(key):
                self.assertEqual(block_errors("keyok-" + key, generic_keys([key])), "")

    def test_empty_key_and_empty_names(self):
        self.assertIn("key is empty", block_errors("empty-key", generic_keys([""])))
        self.assertIn("component is empty", block_errors("empty-comp", lambda b: b.update(component=" ")))
        self.assertIn("name is empty", block_errors("empty-name", lambda b: b.update(name="")))
        self.assertIn("does not match", block_errors("bad-name", lambda b: b.update(name="Has Space")))
        self.assertEqual(block_errors("underscore-name", lambda b: b.update(name="a_b")), "")

    def test_tags_that_reach_a_grouping_through_an_underscore_name(self):
        groupings = [
            {"name": "tier", "ordered": True, "cardinality": "one", "values": ["core", "sanity"]},
            {"name": "release_train", "ordered": False, "cardinality": "many"},
        ]
        for tag in ("release_train", "release_train__x", "release_train-x", "release_train.x", "release_train:x"):
            with self.subTest(tag):
                d = set_project_groupings("rt-" + tag.replace(":", "-"), groupings)
                f = d / "ops" / "op-matmul.yaml"
                docs = list(yaml.safe_load_all(f.read_text()))
                for doc in docs:
                    if doc["kind"] == "Benchmark":
                        doc["spec"]["identity"]["tags"] = [tag]
                f.write_text(yaml.safe_dump_all(docs, sort_keys=False))
                self.assertIn("starts with a run-level grouping", "\n".join(ingest_demo(d).errors))

    def test_near_misses_are_allowed(self):
        for tag in ("backends", "backend1", "be", "tiers", "release"):
            with self.subTest(tag):
                self.assertEqual(block_errors("near-" + tag, lambda b, t=tag: b.update(tags=[t])), "")


class AdapterGuards(unittest.TestCase):
    def tearDown(self):
        ingest_adapter.reset()

    def test_profile_keys_none_only_when_the_module_is_absent(self):
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.results": None}):
            self.assertIsNone(ingest_adapter.profile_keys("perf-kernel"))

    def test_renamed_constant_with_module_present_is_a_rule_change(self):
        from types import SimpleNamespace

        for profile, module in (("perf-kernel", "results"), ("vllm", "vllm"), ("vllm-sweep", "vllm")):
            with (
                self.subTest(profile),
                mock.patch.dict(sys.modules, {f"spyre_clickhouse_ingest.{module}": SimpleNamespace()}),
            ):
                with self.assertRaises(ingest_adapter.IngestUnavailable) as cm:
                    ingest_adapter.profile_keys(profile)
                self.assertIn("the ingest rule changed", str(cm.exception))
                with self.assertRaises(ingest_adapter.IngestUnavailable):
                    engine.check_profile(profile)

    def test_check_profile_reports_match_skipped_unverified_and_mismatch(self):
        status = ingest_adapter.profile_status
        with mock.patch.object(ingest_adapter, "profile_status", return_value=("keys", engine.ID_PROFILES["vllm"])):
            self.assertEqual(engine.check_profile("vllm"), "match")
        with mock.patch.object(ingest_adapter, "profile_status", return_value=("absent", "gone")):
            self.assertTrue(engine.check_profile("vllm").startswith("skipped"))
        with mock.patch.object(
            ingest_adapter, "profile_status", return_value=("unverified", "producer modules not importable: x")
        ):
            self.assertTrue(engine.check_profile("vllm").startswith("unverified (producer modules not importable"))
        self.assertTrue(engine.check_profile("generic").startswith("skipped"))
        with mock.patch.object(ingest_adapter, "profile_status", return_value=("keys", ("record_type",))):
            with self.assertRaises(ingest_adapter.IngestUnavailable):
                engine.check_profile("vllm-sweep")
        self.assertTrue(callable(status))

    def test_producer_modules_that_exist_but_cannot_import_are_unverified_not_skipped(self):
        if sibling_ingest()[0] is None:
            self.skipTest("needs the sibling ingest so the producer modules exist")
        err = ModuleNotFoundError("No module named 'clickhouse_connect'", name="clickhouse_connect")
        with mock.patch.object(ingest_adapter.importlib, "import_module", side_effect=err):
            kind, why = ingest_adapter.profile_status("perf-kernel")
        self.assertEqual(kind, "unverified")
        self.assertIn("producer modules not importable", why)
        self.assertIn("clickhouse_connect", why)
        with mock.patch.object(ingest_adapter.importlib, "import_module", side_effect=err):
            self.assertIsNone(ingest_adapter.profile_keys("perf-kernel"))

    def test_absent_producer_module_is_skipped(self):
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.results": None}):
            self.assertEqual(ingest_adapter.profile_status("perf-kernel")[0], "absent")

    def test_adapter_verb_reports_and_gates_on_the_comparison(self):
        with mock.patch.object(ingest_adapter, "profile_status", return_value=("absent", "x")):
            code, out, _ = run_cli("ids", "--adapter")
        self.assertTrue(all(v.startswith("skipped") for v in json.loads(out)["profile_comparison"].values()))
        un = ("unverified", "producer modules not importable: x")
        with (
            mock.patch.object(ingest_adapter, "profile_status", return_value=un),
            mock.patch.object(ingest_adapter, "available", return_value=True),
        ):
            code, out, _ = run_cli("ids", "--adapter")
            self.assertEqual(code, 2)
            self.assertTrue(all(v.startswith("unverified") for v in json.loads(out)["profile_comparison"].values()))
            code, _, _ = run_cli("ids", "--adapter", "--allow-unverified")
            self.assertEqual(code, 0)
        with (
            mock.patch.object(ingest_adapter, "profile_status", return_value=("keys", ("x",))),
            mock.patch.object(ingest_adapter, "available", return_value=True),
        ):
            code, out, _ = run_cli("ids", "--adapter", "--allow-unverified")
        self.assertEqual(code, 2)
        self.assertTrue(all(v.startswith("MISMATCH") for v in json.loads(out)["profile_comparison"].values()))

    def test_a_sweep_case_is_refused_when_the_ingest_sweep_keys_differ(self):
        cat = ingest_demo()
        c = next(c for c in cat.cases if c["name"] == "op_latency_fp16")
        keys = list(engine.ID_PROFILES["vllm-sweep"])
        sweep = {**c, "identity": {**c["identity"], "profile": "vllm-sweep",
                                   "discriminators": [{"key": k, "value": "1"} for k in keys]}}  # fmt: skip
        with mock.patch.object(ingest_adapter, "profile_status", return_value=("keys", tuple(keys[:-1]))):
            with self.assertRaises(ingest_adapter.IngestUnavailable) as cm:
                cat.ids(sweep)
        self.assertIn("the ingest rule changed", str(cm.exception))


class IdTokenAndCli(unittest.TestCase):
    def test_id_token_normalises_component_and_follows_key_order(self):
        cat = ingest_demo()
        c = next(c for c in cat.cases if c["name"] == "op_generic_add")

        def token(**over):
            return cat.id_token({**c, "identity": {**c["identity"], **over}})

        self.assertEqual(token(component="Ingest-Demo"), token(component="ingest-demo"))
        flipped = list(reversed(c["identity"]["discriminators"]))
        self.assertNotEqual(token(), token(discriminators=flipped))

    def test_ids_with_an_unknown_name_fails(self):
        code, _, err = run_cli(
            "ids", "no-such", "--catalog", str(X / "ingest-demo"), "--central", str(CENTRAL), "--inputs"
        )
        self.assertEqual(code, 2)
        self.assertIn("no case or Benchmark named", err)


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ingest, cls.why = sibling_ingest()

    def result(self, cat, case, name):
        fp = {k: "x" for k in sorted(cat.reg.fingerprint_keys())}
        rep = [{"run.repeat": 1, "placement": {"nodes": ["n1"], "devices": []}, "samples": {"kernel.time.mean": [1.0]}}]
        return engine.build_result(cat, case, name, {"artifact_id": "a"}, fp, {}, {}, rep)

    def test_results_of_an_overridden_ingest_name_are_selected_by_benchmark(self):
        if self.ingest is None:
            self.skipTest(self.why)
        from spyre_bench import analysis

        d = copy_ingest_demo("two-one-name")
        ops = d / "ops"
        src = yaml.safe_load((X / "ingest-demo/ops/op-softmax.yaml").read_text())
        src["metadata"]["name"] = "op-softmax2"
        src["spec"]["variants"] = [{"name": "op_softmax2_fp16"}]
        src["spec"]["identity"]["discriminators"][1] = {"key": "config_name", "value": "other"}
        (ops / "op-softmax2.yaml").write_text(yaml.safe_dump(src, sort_keys=False))
        cat = ingest_demo(d)
        self.assertEqual(cat.errors, [])
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": self.ingest}):
            ingest_adapter.reset()
            a = self.result(cat, next(c for c in cat.cases if c["name"] == "op_softmax_fp16"), "ra")
            b = self.result(cat, next(c for c in cat.cases if c["name"] == "op_softmax2_fp16"), "rb")
        ingest_adapter.reset()
        self.assertEqual((a["status"]["case"]["name"], b["status"]["case"]["name"]), ("softmax", "softmax"))
        self.assertEqual(
            (a["status"]["case"]["benchmark"], a["status"]["case"]["case"]), ("op-softmax", "op_softmax_fp16")
        )
        self.assertEqual(analysis.select_results([a, b], {"op-softmax"}), [a])
        self.assertEqual(analysis.select_results([a, b], {"op-softmax2"}), [b])
        self.assertEqual(analysis.select_results([a, b], {"softmax"}), [])
        analysis.analyze({"select": {"benchmarks": ["op-softmax"]}, "metric": "kernel.time.mean"}, [a, b])

    def test_engine_results_still_select_by_name(self):
        from spyre_bench import analysis

        r = {"status": {"case": {"name": "bench-a"}}}
        self.assertEqual(analysis.select_results([r], {"bench-a"}), [r])

    def test_recompute_accepts_a_generic_key_named_like_an_engine_prop(self):
        if self.ingest is None:
            self.skipTest(self.why)
        d = copy_ingest_demo("generic-system-name")
        f = d / "ops" / "op-generic.yaml"
        doc = yaml.safe_load(f.read_text())
        doc["spec"]["identity"]["discriminators"].append({"key": "system.name", "value": "x"})
        f.write_text(yaml.safe_dump(doc, sort_keys=False))
        cat = ingest_demo(d)
        self.assertEqual(cat.errors, [])
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": self.ingest}):
            ingest_adapter.reset()
            r = self.result(cat, next(c for c in cat.cases if c["name"] == "op_generic_add"), "r")
            self.assertEqual(engine.recompute(r), [])
        ingest_adapter.reset()


class Round4(unittest.TestCase):
    def test_keys_that_name_backend_or_a_grouping_in_any_form_are_refused(self):
        for key in ("backend_kind", "backend-x", "Backend_Kind", "tier_core", "suite_ops", "tier-core", "suite-x",
                    "run.x", "run__x", "run:x"):  # fmt: skip
            with self.subTest(key):
                got = block_errors("k4-" + key.replace(":", "-").replace(".", "-"), generic_keys([key]))
                self.assertRegex(got, "backend|run-level grouping")

    def test_legitimate_keys_pass(self):
        for key in ("run_mode", "system.name", "record_type", "config_name", "input_shapes", "kernel_name", "is_total",
                    "precision", "batch.size", "llama-3.1-8b", "model.v2-x", "backends", "tiers", "runtime"):  # fmt: skip
            with self.subTest(key):
                self.assertEqual(block_errors("k4ok-" + key.replace(".", "-"), generic_keys([key])), "")

    def test_names_and_tags_refuse_a_trailing_newline_and_upper_case(self):
        for field in ("name", "component"):
            for bad in ("foo\n", "Foo", " foo", "foo "):
                with self.subTest((field, bad)):
                    got = block_errors(
                        f"nl-{field}-{abs(len(bad))}{ord(bad[0])}", lambda b, f=field, v=bad: b.update({f: v})
                    )
                    self.assertIn("does not match", got)
        self.assertIn("does not match", block_errors("tag-nl", lambda b: b.update(tags=["foo\n"])))
        self.assertIn("does not match", block_errors("tag-upper", lambda b: b.update(tags=["Foo"])))

    def test_id_token_normalises_the_name(self):
        cat = ingest_demo()
        c = next(c for c in cat.cases if c["name"] == "op_generic_add")

        def token(name):
            return cat.id_token({**c, "identity": {**c["identity"], "name": name}})

        self.assertEqual(token(" Op-Generic "), token("op-generic"))

    def test_validate_reports_an_unresolvable_base_tree_instead_of_raising(self):
        base = TMP / "base-bad-from"
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir()
        shutil.copytree(X / "ingest-demo", base / "ingest-demo")
        f = base / "ingest-demo" / "ops" / "op-softmax.yaml"
        f.write_text(f.read_text().replace("from: case.dtype", "from: case.nope"))
        code, out, err = run_cli(
            "validate", str(X / "ingest-demo"), "--central", str(CENTRAL), "--base-dir", str(base), "--json"
        )
        self.assertEqual(code, 0, err)
        warnings = json.loads(out)[0]["warnings"]
        self.assertTrue(any("cannot compare identity (base tree)" in w for w in warnings), warnings)

    def test_from_sources_include_system_and_input_values(self):
        cat = ingest_demo()
        c = next(c for c in cat.cases if c["name"] == "op_softmax_fp16")
        got = cat.ingest_inputs({**c, "identity": {**c["identity"], "discriminators": [
            {"key": "config_name", "from": "system.precision"}, *c["identity"]["discriminators"][2:]]}})  # fmt: skip
        self.assertEqual(got["disc"]["config_name"], "fp16")
        blk = {"mode": "ingest", "discriminators": [{"key": k, "value": ""} for k in engine.INGEST_DISC_KEYS]}
        for src in ("system.precision", "input.shapes.asset", "input.shapes.select", "case.op"):
            blk["discriminators"][1] = {"key": "config_name", "from": src}
            self.assertEqual(cat._identity_errors(blk), [], src)
        blk["discriminators"][1] = {"key": "config_name", "from": "run.tier"}
        self.assertTrue(cat._identity_errors(blk))
        llm = load(X / "llm-demo", central=[CENTRAL])
        with_input = next(c for c in llm.cases if c["workload"].get("inputs"))
        n = with_input["workload"]["inputs"][0]["name"]
        blk = {"mode": "ingest", "discriminators": [{"key": k, "value": ""} for k in engine.INGEST_DISC_KEYS]}
        blk["discriminators"][1] = {"key": "config_name", "from": f"input.{n}.asset"}
        resolved = llm.ingest_inputs({**with_input, "identity": blk})
        self.assertTrue(resolved["disc"]["config_name"])


class Conformance(unittest.TestCase):
    """The table was produced by the real ingest; the engine's adapter and the example blocks must reproduce it."""

    @classmethod
    def setUpClass(cls):
        cls.ingest, why = sibling_ingest()
        cls.why = why
        ingest_adapter.reset()

    def need_ingest(self):
        if self.ingest is None:
            self.skipTest(self.why)

    def test_table_inputs_match_the_example_blocks(self):
        cat = ingest_demo()
        rows = [r for r in table() if r["example_case"]]
        self.assertEqual({r["example_case"].split("/", 1)[1] for r in rows}, INGEST_CASES)
        for r in rows:
            c = next(c for c in cat.cases if c["name"] == r["example_case"].split("/", 1)[1])
            a = cat.ingest_inputs(c)
            hashed = lambda disc: [engine.norm(disc.get(k)) for k in r["disc_keys"]]  # noqa: E731 - absent hashes empty
            self.assertEqual(a["disc_keys"], r["disc_keys"], r["example_case"])
            self.assertEqual(hashed(a["disc"]), hashed(r["disc"]), r["example_case"])
            self.assertEqual((a["component"], a["name"], a["tags"]), (r["component"], r["name"], r["tags"]))

    def test_table_ids_are_the_real_derivation(self):
        self.need_ingest()
        for r in table():
            real = self.ingest.BenchmarkId.derive(r["component"], r["name"], r["tags"], r["disc"], r["disc_keys"])
            self.assertEqual(real, r["benchmark_id"], r)

    def test_adapter_reproduces_the_table(self):
        self.need_ingest()
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": self.ingest}):
            ingest_adapter.reset()
            self.assertTrue(ingest_adapter.available())
            for r in table():
                got = ingest_adapter.benchmark_id(r["component"], r["name"], r["tags"], r["disc"], r["disc_keys"])
                self.assertEqual(got, r["benchmark_id"], r)

    def test_example_cases_derive_the_table_ids(self):
        self.need_ingest()
        cat = ingest_demo()
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": self.ingest}):
            ingest_adapter.reset()
            for r in (r for r in table() if r["example_case"]):
                c = next(c for c in cat.cases if c["name"] == r["example_case"].split("/", 1)[1])
                ids = cat.ids(c)
                self.assertEqual(ids["benchmark_id"], r["benchmark_id"], r["example_case"])
                self.assertEqual(ids["id_mode"], "ingest")
                self.assertNotEqual(ids["engine_id"], ids["benchmark_id"])
                self.assertEqual(ids["engine_id"], cat.engine_id(c))

    def test_table_rows_are_what_the_producers_write(self):
        self.need_ingest()
        try:
            import refresh_conformance

            with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": self.ingest}):
                text = refresh_conformance.build()
        except ImportError as x:
            self.skipTest(f"the ingest producers are not importable (needs clickhouse-connect): {x}")
        self.assertEqual(text, (GOLDEN / "ingest-ids.tsv").read_text())

    def test_engine_constants_equal_the_ingests(self):
        self.need_ingest()
        i = self.ingest
        self.assertEqual(engine.ID_NAMESPACE, i.ID_NAMESPACE)
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": i}):
            ingest_adapter.reset()
            self.assertEqual(ingest_adapter.info()["namespace"], str(engine.ID_NAMESPACE))
        ingest_adapter.reset()
        extras = ("RUN_CONTEXT_TAG_NAMESPACES", "RESULT_TAG_NAMESPACES", "LEGACY_TAG_ALIASES")
        if not all(hasattr(i, n) for n in extras):
            self.skipTest("a vendored identity module needs only BenchmarkId and ID_NAMESPACE")
        self.assertEqual(engine.RUN_CONTEXT_TAG_NAMESPACES, frozenset(i.RUN_CONTEXT_TAG_NAMESPACES))
        self.assertEqual(engine.RESULT_TAG_NAMESPACES, frozenset(i.RESULT_TAG_NAMESPACES))
        self.assertEqual(engine.LEGACY_TAG_ALIASES, dict(i.LEGACY_TAG_ALIASES))
        try:
            from spyre_clickhouse_ingest import results
        except Exception as x:  # noqa: BLE001 - results imports a client library the identity module does not
            self.skipTest(f"ingest results module not importable: {x}")
        self.assertEqual(engine.INGEST_DISC_KEYS, tuple(results._V2_BENCH_ID_KEYS))

    def test_profile_keys_equal_the_ingests(self):
        self.need_ingest()
        for profile, keys in engine.ID_PROFILES.items():
            theirs = ingest_adapter.profile_keys(profile)
            if theirs is None:
                self.skipTest(f"ingest constants for {profile} are not importable (needs clickhouse-connect)")
            self.assertEqual(tuple(theirs), keys, profile)

    def test_a_changed_profile_in_the_ingest_refuses_ids(self):
        self.need_ingest()
        cat = ingest_demo()
        c = next(c for c in cat.cases if c["name"] == "op_latency_fp16")
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": self.ingest}):
            ingest_adapter.reset()
            with mock.patch.object(ingest_adapter, "profile_status", return_value=("keys", ("record_type",))):
                with self.assertRaises(ingest_adapter.IngestUnavailable) as cm:
                    cat.ids(c)
        self.assertIn("the ingest rule changed", str(cm.exception))

    def test_result_round_trip_and_probe_guard(self):
        self.need_ingest()
        cat = ingest_demo()
        c = next(c for c in cat.cases if c["name"] == "op_softmax_fp16")
        fp = {k: "x" for k in sorted(cat.reg.fingerprint_keys())}
        with mock.patch.dict(sys.modules, {"spyre_clickhouse_ingest.identity": self.ingest}):
            ingest_adapter.reset()
            r = engine.build_result(
                cat,
                c,
                "r1",
                {"artifact_id": "a"},
                fp,
                {},
                {},
                [
                    {
                        "run.repeat": 1,
                        "placement": {"nodes": ["n1"], "devices": []},
                        "samples": {"kernel.time.mean": [1.0]},
                    }
                ],
            )
            case = r["status"]["case"]
            self.assertEqual((case["id_mode"], case["disc_keys"]), ("ingest", list(engine.INGEST_DISC_KEYS)))
            self.assertEqual(case["engine_id"], cat.engine_id(c))
            self.assertEqual(engine.shape_schema("result").errors(r), [])
            self.assertEqual(engine.recompute(r), [])
            case["disc"]["run_mode"] = "other"
            self.assertEqual(engine.recompute(r), ["benchmark_id"])
            ingest_adapter.reset()
            with mock.patch.object(ingest_adapter, "PROBES", ((ingest_adapter.PROBES[0][0], "0" * 36),)):
                with self.assertRaises(ingest_adapter.IngestUnavailable) as cm:
                    ingest_adapter.benchmark_id("c", "n", [], {}, [])
                self.assertIn("the ingest rule changed", str(cm.exception))
        ingest_adapter.reset()


if __name__ == "__main__":
    unittest.main()

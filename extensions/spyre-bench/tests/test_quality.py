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

"""Quality: validity gates, the quality ValidityPolicy, and a score compared with a reference.
python -m pytest -q -p no:cacheprovider tests/test_quality.py
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml
from test_run import NOW, SUBJECT, capture, clean_env, manifest

from spyre_bench import analysis, engine, gen_schemas, quality
from spyre_bench import plan as plan_cmd
from spyre_bench import run as run_cmd

EXT = Path(__file__).resolve().parent.parent
DEMO = EXT / "examples" / "quality-demo"
CENTRAL = EXT / "examples" / "central"
TMP = Path(tempfile.mkdtemp(prefix="spyre-bench-quality-tests-"))
DIGEST = "sha256:4f3a1c0de2b64a5f8c7e91d0b2a36f5e4d8c7b6a5f4e3d2c1b0a99887766554a"


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def good(**kw):
    return {
        "expect_records": 8,
        "expect_digest": DIGEST,
        "expect_temperature": 0,
        "n_samples": 8,
        "error_frac": 0.0,
        "truncated_frac": 0.0,
        "dataset_digest": DIGEST,
        "harness_version": "1.0",
        "protocol": "p-0001",
        "model_revision": "abc",
        "temperature": 0.0,
        "seed": None,
        **kw,
    }


class Gates(unittest.TestCase):
    def caps(self, **kw):
        return quality.gate_caps(sorted(quality.GATES), good(**kw))

    def test_complete_facts_pass_every_gate(self):
        self.assertEqual(self.caps(), [])

    def test_dataset_digest(self):
        self.assertEqual(self.caps(dataset_digest="sha256:" + "0" * 64), [("invalid", "dataset_mismatch")])
        self.assertEqual(self.caps(dataset_digest=None), [("invalid", "dataset_unreported")])
        self.assertEqual(self.caps(expect_digest=None), [("suspect", "dataset_unpinned")])
        self.assertEqual(self.caps(dataset_digest=DIGEST.removeprefix("sha256:")), [])

    def test_records(self):
        self.assertEqual(self.caps(n_samples=5), [("invalid", "records_incomplete")])
        self.assertEqual(self.caps(n_samples=None), [("invalid", "records_unreported")])
        self.assertEqual(self.caps(expect_records=None), [("suspect", "records_undeclared")])

    def test_nonnumeric_expected_records_do_not_raise(self):
        self.assertEqual(self.caps(expect_records="many"), [("invalid", "records_unreported")])

    def test_errors_and_truncation(self):
        self.assertEqual(self.caps(error_frac=0.1), [("invalid", "infrastructure")])
        self.assertEqual(self.caps(error_frac=None), [("suspect", "errors_unreported")])
        self.assertEqual(self.caps(truncated_frac=0.02), [("suspect", "truncation")])
        self.assertEqual(self.caps(truncated_frac=0.01), [])

    def test_pinning(self):
        self.assertEqual(self.caps(protocol=""), [("suspect", "protocol_unpinned")])
        self.assertEqual(self.caps(harness_version=None), [("suspect", "protocol_unpinned")])
        self.assertEqual(self.caps(model_revision=None), [("suspect", "model_unpinned")])

    def test_determinism(self):
        self.assertEqual(self.caps(temperature=0.7), [("suspect", "sampling_in_greedy_case")])
        self.assertEqual(self.caps(temperature=None), [("suspect", "seed_unpinned")])
        self.assertEqual(self.caps(expect_temperature=0.7, temperature=0.7), [("suspect", "seed_unpinned")])
        self.assertEqual(self.caps(expect_temperature=0.7, temperature=0.7, seed=3), [])


POLICY = {
    "purposes": {
        "nightly": {
            "floor": "shared_env",
            "requires": {"min_repeats": 1, "gates": ["llm.dataset_digest", "llm.records_complete", "llm.no_errors"]},
            "baseline": True,
            "reference": False,
        }
    }
}


class Validity(unittest.TestCase):
    def state(self, facts, **kw):
        return quality.validity_of(POLICY, "nightly", facts, **kw)

    def test_valid(self):
        self.assertEqual(self.state([good()]), ("valid", []))

    def test_worst_cap_wins_and_reasons_are_distinct(self):
        bad = good(n_samples=5, protocol="")  # protocol is not a listed gate: ignored
        self.assertEqual(self.state([bad, bad]), ("invalid", ["records_incomplete"]))

    def test_unlisted_gates_do_not_apply(self):
        self.assertEqual(self.state([good(model_revision=None)]), ("valid", []))

    def test_not_measured_and_external(self):
        self.assertEqual(self.state([], measured=False)[0], "invalid")
        self.assertEqual(self.state([good()], trust="external"), ("external", ["trust_external"]))


class Reasons(unittest.TestCase):
    def test_every_reason_a_gate_can_emit_is_documented(self):
        emitted = set()
        base = good()
        for gate in quality.GATES.values():
            for change in (
                {}, {"dataset_digest": "sha256:" + "0" * 64}, {"dataset_digest": None}, {"expect_digest": None},
                {"n_samples": 1}, {"n_samples": None}, {"expect_records": None}, {"error_frac": 1.0}, {"error_frac": None},
                {"truncated_frac": 1.0}, {"truncated_frac": None}, {"harness_version": ""}, {"model_revision": None},
                {"temperature": 0.7}, {"temperature": None}, {"expect_temperature": 0.7, "temperature": 0.7},
                {"expect_temperature": 0.7, "temperature": 0.5, "seed": 1},
            ):  # fmt: skip
                if cap := gate({**base, **change}):
                    emitted.add(cap[1])
        docs = (EXT / "docs" / "README.md").read_text()
        self.assertEqual([r for r in sorted(emitted) if r not in docs], [])

    def test_determinism_compares_the_declared_temperature(self):
        facts = good(expect_temperature=0.7, temperature=0.0, seed=1)
        self.assertEqual(quality.GATES["llm.determinism"](facts), ("suspect", "sampling_mismatch"))


class Cases(unittest.TestCase):
    def test_greedy_case_rejects_a_seed(self):
        case = {"type": "quality", "workload": {"eval": {"task": "t", "temperature": 0, "seed": 1}}}
        self.assertEqual(len(quality.case_errors(case)), 1)
        self.assertEqual(quality.case_errors({**case, "workload": {"eval": {"task": "t", "seed": 1}}}), [])
        self.assertEqual(quality.case_errors({"type": "perf", "workload": {}}), [])


class Judge(unittest.TestCase):
    def row(self, score, **kw):
        args = {"se": 0.01, "ref": 0.70, "ref_se": 0.01, "direction": "higher_is_better", "tol": {"stderr": 3}}
        args |= kw
        return quality.judge_against(score, args["se"], args["ref"], args["ref_se"], args["direction"], args["tol"],
                                     args.get("validity"), args.get("reasons", ()))  # fmt: skip

    def test_pass_within_tolerance(self):
        self.assertEqual(self.row(0.68)["status"], "pass")

    def test_strong_shortfall_is_regressed_single_run(self):
        r = self.row(0.60)
        self.assertEqual(r["status"], "regressed")
        self.assertGreaterEqual(r["z_worse"], quality.STRONG_Z)

    def test_shortfall_past_tolerance_but_not_strong_is_suspected(self):
        r = self.row(0.65, tol={"stderr": 2})
        self.assertEqual((r["status"], r["reason"]), ("suspected", "single_run"))

    def test_improvement(self):
        self.assertEqual(self.row(0.80)["status"], "improved")

    def test_lower_is_better_flips_the_sign(self):
        self.assertEqual(self.row(0.80, direction="lower_is_better")["status"], "regressed")

    def test_invalid_and_suspect_are_never_regressed(self):
        for validity, want in (("invalid", "invalid"), ("suspect", "suspected")):
            r = self.row(0.10, validity=validity, reasons=["records_incomplete"])
            self.assertEqual(r["status"], want)
            self.assertNotEqual(r["status"], "regressed")

    def test_missing_stderr_cannot_be_judged(self):
        self.assertEqual(self.row(0.1, se=None)["status"], "no_stderr")

    def test_underpowered_when_the_practical_floor_is_below_the_noise(self):
        r = self.row(0.69, se=0.05, ref_se=0.05, tol={"stderr": 3, "abs": 0.02})
        self.assertEqual((r["status"], r["reason"]), ("no_power", "underpowered"))

    def test_abs_floor_alone(self):
        self.assertEqual(self.row(0.69, se=0.001, ref_se=0.001, tol={"abs": 0.02})["status"], "pass")
        self.assertEqual(self.row(0.60, se=0.001, ref_se=0.001, tol={"abs": 0.02})["status"], "regressed")


def result(case, score, n=1319, stderr=None, validity=("valid", [])):
    m = {
        "llm.eval.exact_match": {"value": score, "unit": "1"},
        "llm.eval.n_samples": {"value": n, "unit": "{record}"},
    }
    if stderr is not None:
        m["llm.eval.exact_match_stderr"] = {"value": stderr, "unit": "1"}
    return {
        "status": {
            "case": {"name": "eval-questions", "case": case, "benchmark": "eval-questions"},
            "aggregate": {"metrics": m},
            "validity": {"state": validity[0], "reasons": validity[1]},
            "repeats": [],
        }
    }


SPEC = {
    "select": {"benchmarks": ["eval-questions"]},
    "metric": "llm.eval.exact_match",
    "reference": {"case": "ref", "tolerance": {"stderr": 3}},
}
METRICS = {"llm.eval.exact_match": {"direction": "higher_is_better", "category": "quality", "unit": "1"}}


class Report(unittest.TestCase):
    def rows(self, spec, results):
        return {r["case"]: r for r in analysis.analyze(spec, results, METRICS)["reference"]["rows"]}

    def test_case_reference_and_binomial_stderr(self):
        rows = self.rows(SPEC, [result("ref", 0.70), result("a", 0.55), result("b", 0.69)])
        self.assertEqual(set(rows), {"a", "b"})
        self.assertEqual((rows["a"]["status"], rows["b"]["status"]), ("regressed", "pass"))
        self.assertAlmostEqual(rows["a"]["stderr"], (0.55 * 0.45 / 1319) ** 0.5)

    def test_harness_stderr_is_preferred(self):
        rows = self.rows(SPEC, [result("ref", 0.70, stderr=0.2), result("a", 0.55, stderr=0.2)])
        self.assertEqual(rows["a"]["status"], "pass")

    def test_inline_reference(self):
        spec = {**SPEC, "reference": {"label": "cpu-fp32-ref", "value": 0.7, "n": 1319, "tolerance": {"stderr": 3}}}
        rep = analysis.analyze(spec, [result("a", 0.55)], METRICS)["reference"]
        self.assertEqual((rep["reference"]["label"], rep["rows"][0]["status"]), ("cpu-fp32-ref", "regressed"))

    def test_invalid_result_is_reported_not_judged(self):
        rows = self.rows(SPEC, [result("ref", 0.70), result("a", 0.10, validity=("invalid", ["records_incomplete"]))])
        self.assertEqual((rows["a"]["status"], rows["a"]["reason"]), ("invalid", "records_incomplete"))

    def test_unjudged_result_is_suspected(self):
        unjudged = result("a", 0.10)
        del unjudged["status"]["validity"]
        rows = self.rows(SPEC, [result("ref", 0.70), unjudged])
        self.assertEqual((rows["a"]["status"], rows["a"]["reason"]), ("suspected", "validity_unjudged"))

    def test_invalid_reference_taints_every_row(self):
        rows = self.rows(SPEC, [result("ref", 0.70, validity=("invalid", ["records_incomplete"])), result("a", 0.10)])
        self.assertEqual((rows["a"]["status"], rows["a"]["reason"]), ("suspected", "reference_not_valid"))

    def test_zero_noise_is_not_judged(self):
        spec = {**SPEC, "reference": {"value": 0.99, "stderr": 0.0, "tolerance": {"stderr": 3}}}
        r = result("a", 1.0, stderr=0.0)
        self.assertEqual(analysis.analyze(spec, [r], METRICS)["reference"]["rows"][0]["status"], "no_stderr")

    def test_perfect_score_keeps_a_nonzero_binomial_stderr(self):
        self.assertGreater(quality.binomial_stderr(1.0, 100), 0)
        self.assertGreater(quality.binomial_stderr(0.0, 100), 0)

    def test_metric_without_a_direction_is_not_judged(self):
        r = quality.judge_against(0.7, 0.01, 0.8, 0.01, "none", {"stderr": 3}, "valid", [])
        self.assertEqual((r["status"], r["reason"]), ("no_value", "no_direction"))

    def test_only_valid_data_can_be_regressed(self):
        for state in ("shared_env", "unqualified_env", "external"):
            r = quality.judge_against(0.1, 0.01, 0.9, 0.01, "higher_is_better", {"stderr": 3}, state, [])
            self.assertEqual(r["status"], "suspected", state)

    def test_missing_score_value_does_not_raise(self):
        r = result("a", None)
        r["status"]["aggregate"]["metrics"]["llm.eval.exact_match"]["value"] = None
        self.assertEqual(self.rows(SPEC, [result("ref", 0.7), r])["a"]["status"], "no_value")

    def test_two_reference_runs_are_refused(self):
        with self.assertRaises(analysis.AnalysisError):
            analysis.analyze(SPEC, [result("ref", 0.7), result("ref", 0.71), result("a", 0.6)], METRICS)


class Catalog(unittest.TestCase):
    def cat(self, path=DEMO):
        return engine.Catalog([path], [], [CENTRAL])

    def test_demo_validates_with_a_separate_quality_policy(self):
        cat = self.cat()
        self.assertEqual(cat.errors, [])
        self.assertEqual(cat.quality_policy.name, "demo-quality")
        self.assertEqual(cat.policy.name, "demo-default")
        pol = cat.policy_of(cat.cases[0])
        self.assertEqual(pol.name, "demo-quality")

    def test_schemas_on_disk_are_current(self):
        self.assertEqual(gen_schemas.main(["--check"]), 0)

    def test_policy_rules(self):
        def errs(spec):
            doc = type("D", (), {"spec": spec, "where": "p"})()
            return quality.policy_errors(doc, self.cat().reg)

        row = {"floor": "valid", "baseline": True, "reference": True}
        ok_gates = list(quality.REFERENCE_GATES)
        self.assertEqual(errs({"purposes": {"reference": {**row, "requires": {"gates": ok_gates}}}}), [])
        self.assertTrue(errs({"purposes": {"reference": {**row, "requires": {"gates": ok_gates[:2]}}}}))
        self.assertTrue(errs({"purposes": {"nightly": {**row, "requires": {"max_cv_pct": 3}}}}))
        self.assertTrue(errs({"purposes": {"nightly": {**row, "requires": {"clean": "enforced"}}}}))
        self.assertTrue(errs({"purposes": {"nightly": {**row, "requires": {"gates": ["llm.nope"]}}}}))
        self.assertTrue(errs({"purposes": {"pr": {**row, "requires": {"gates": ok_gates}}}}))

    def test_reference_tolerance_must_be_positive(self):
        spec = {"metric": "llm.eval.exact_match", "reference": {"value": 0.5, "n": 10, "tolerance": {"abs": 0}}}
        metrics = {"llm.eval.exact_match": {"category": "quality", "direction": "higher_is_better"}}
        self.assertTrue(any("above 0" in e for e in quality.reference_errors(spec, set(), metrics, "w")))

    def test_mixed_or_misplaced_policy_is_rejected(self):
        d = TMP / "bad-policy"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(DEMO, d)
        pol = yaml.safe_load_all((CENTRAL / "validity-policy.yaml").read_text())
        q = next(p for p in pol if p["metadata"]["name"] == "demo-quality")
        q["metadata"]["name"] = "mixed"
        q["spec"]["applies_to"] = ["perf", "quality"]
        (d / "mixed.yaml").write_text(yaml.safe_dump(q))
        self.assertTrue(any("mixed list" in e for e in engine.Catalog([d], [], [CENTRAL]).errors))

    def test_reference_block_is_checked(self):
        d = TMP / "bad-ref"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(DEMO, d)
        doc = yaml.safe_load((d / "analysis/quality-vs-reference.yaml").read_text())
        doc["spec"]["reference"] = {"case": "nope", "value": 0.5, "tolerance": {}}
        (d / "analysis/quality-vs-reference.yaml").write_text(yaml.safe_dump(doc))
        msg = "\n".join(engine.Catalog([d], [], [CENTRAL]).errors)
        self.assertIn("exactly one of case or value", msg)
        self.assertIn("is not a case", msg)
        self.assertIn("tolerance needs", msg)


def make_plan(path, purpose="nightly"):
    argv = [
        "--catalog", str(DEMO), "--central", str(CENTRAL), "--arch", "x86_64", "--reference-node", "--now", NOW,
        "--purpose", purpose, "--type", "quality", "--out", str(path), "--subject-id", SUBJECT,
    ]  # fmt: skip
    code, _, err = capture(plan_cmd.main, argv)
    assert code in (0, 4), (code, err)
    return path


class EndToEnd(unittest.TestCase):
    """plan then run the stub quality command: the Result carries the quality policy's verdict."""

    def go(self, flaw="", purpose="nightly", score=None, keep=None):
        n = len(list(TMP.glob("run-*")))
        out = TMP / f"run-{n}"
        env = {"STUB_FLAW": flaw, **({"STUB_SCORE": str(score)} if score is not None else {})}
        with clean_env(**env):
            plan = make_plan(TMP / f"plan-{n}.json", purpose)
            argv = ["--plan", str(plan), "--out", str(out), "--workdir", str(DEMO), "--catalog", str(DEMO),
                    "--central", str(CENTRAL), "--pass-env", "STUB_FLAW,STUB_SCORE"]  # fmt: skip
            code, _, err = capture(run_cmd.main, argv)
        self.assertEqual(code, 0, err)
        m = manifest(out)
        res = {c["name"]: json.loads((out / c["result"]).read_text()) for c in m["cases"]}
        if keep is not None:
            keep.extend(res.values())
        return res["eval_questions_demoa_fp16_l8"]["status"]

    def test_clean_run_is_valid_without_a_clean_node_or_qualification(self):
        st = self.go()
        self.assertEqual((st["validity"]["state"], st["validity"]["reasons"]), ("valid", []))
        self.assertEqual(st["run_props"]["run.purpose"], "nightly")

    def test_each_flaw_caps_validity_with_its_reason(self):
        for flaw, state, reason in (
            ("digest", "invalid", "dataset_mismatch"),
            ("short", "invalid", "records_incomplete"),
            ("errors", "invalid", "infrastructure"),
            ("truncated", "suspect", "truncation"),
            ("unpinned", "suspect", "protocol_unpinned"),
            ("sampled", "suspect", "sampling_in_greedy_case"),
        ):
            with self.subTest(flaw):
                v = self.go(flaw)["validity"]
                self.assertEqual((v["state"], reason in v["reasons"]), (state, True))

    def test_analyze_compares_the_two_cases_of_one_run(self):
        results = []
        self.go(keep=results)
        f = TMP / "results.json"
        f.write_text(json.dumps(results))
        code, out, err = capture(
            analysis.main,
            ["quality-vs-reference", "--catalog", str(DEMO), "--central", str(CENTRAL), "--results", str(f), "--json"],
        )
        self.assertEqual(code, 0, err)
        rep = json.loads(out)["reference"]
        self.assertEqual([r["case"] for r in rep["rows"]], ["eval_questions_demoa_fp16_l8"])
        self.assertEqual(rep["rows"][0]["status"], "no_power")  # 8 questions cannot resolve a 0.02 floor
        self.assertEqual(rep["reference"]["label"], "eval_questions_demoa_ref_l8")

    def test_a_purpose_the_quality_policy_lacks_is_refused_at_plan_time(self):
        argv = ["--catalog", str(DEMO), "--central", str(CENTRAL), "--arch", "x86_64", "--now", NOW,
                "--purpose", "qualification", "--type", "quality", "--subject-id", SUBJECT]  # fmt: skip
        with clean_env():
            code, _, err = capture(plan_cmd.main, argv)
        self.assertNotEqual(code, 0)
        self.assertIn("ValidityPolicy/demo-quality", err)

    def test_case_key_is_stored_for_quality_results_only(self):
        self.assertEqual(self.go()["case"]["case"], "eval_questions_demoa_fp16_l8")

    def test_pr_purpose_lists_fewer_gates(self):
        v = self.go("unpinned", purpose="pr")["validity"]
        self.assertEqual(v["state"], "valid")  # harness_pinned is not a pr gate

    def test_the_policy_digest_is_the_quality_policy(self):
        st = self.go()
        cat = engine.Catalog([DEMO], [], [CENTRAL])
        self.assertEqual(st["validity"]["policy_digest"], cat.digest("ValidityPolicy", "demo-quality"))


if __name__ == "__main__":
    unittest.main()

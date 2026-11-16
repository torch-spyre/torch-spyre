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

"""`vs=`: resolution, reuse-aware planning, trust rules, the preview and the request record.
python -m pytest -q -p no:cacheprovider tests/test_compare.py
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone

from spyre_bench import compare as cmp
from spyre_bench import ondemand as od
from spyre_bench.store import MemoryStore

T0 = datetime(2026, 11, 10, tzinfo=timezone.utc)
SUBJ = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
OLDER = "33333333-3333-4333-8333-333333333333"
REFA = "44444444-4444-4444-8444-444444444444"
ENV = "env-a"
LABELS = {"env.label.node.class": "pool-a"}
B = {n: f"00000000-0000-4000-8000-00000000000{i}" for i, n in enumerate("abcd", 1)}
GHE = od.POLICIES["ghe"]
KINDS = GHE.vs_kinds


def case(name, action="run", reason="", reused_from=None, labels=None):
    return {
        "name": name,
        "benchmark_id": B[name],
        "action": action,
        "reason": reason,
        "reused_from": reused_from,
        "est_min": 30.0,
        "accelerators": 2,
        "cache": "warm",
        "labels": LABELS if labels is None else labels,
        "repeats": 3,
        "environment": "env-x",
    }


def plan(cases=None, env_id=ENV, subject=SUBJ, arch="x86_64"):
    cases = cases or [case("a"), case("b"), case("c")]
    counts = {a: sum(1 for c in cases if c["action"] == a) for a in ("run", "reuse", "skip")}
    return {
        "purpose": "pr",
        "request": {"purpose": "pr", "subject": {"artifact_id": subject}, "selection": {"type": "perf"}},
        "leg": {"arch": arch, "env_id": env_id},
        "summary": {"cases": len(cases), **counts, "reasons": {}},
        "estimate": {"wall_min": 90.0, "device_hours": 3.0},
        "cases": cases,
    }


def row(
    name, artifact=OTHER, run="run-o", env=ENV, validity="valid", trust="internal", purpose="nightly", days=0, **kw
):
    return {
        "run_id": run,
        "benchmark_id": B[name],
        "subject_id": artifact,
        "env_id": env,
        "labels": LABELS,
        "validity": validity,
        "trust": trust,
        "purpose": purpose,
        "n": 5,
        "ts": T0 - timedelta(days=days),
        **kw,
    }


def target(artifact=OTHER, source="nightly", component="spyre-inference", arch="x86_64", days=0, **kw):
    return {
        "artifact_id": artifact,
        "component": component,
        "arch": arch,
        "ts": T0 - timedelta(days=days),
        "source": source,
        **kw,
    }


def run(vs, results, targets, p=None, host=GHE, **kw):
    st = MemoryStore(results=results, targets=targets, **{k: kw.pop(k) for k in ("events", "fail") if k in kw})
    return cmp.plan_compare(
        p or plan(), cmp.parse_vs(vs), st, product="spyre-inference", host_kinds=host.vs_kinds, **kw
    )


def by(doc):
    return {c["name"]: c for c in doc["cases"]}


class Grammar(unittest.TestCase):
    def test_forms(self):
        for text, kind, value in [
            ("main", "main", ""),
            ("nightly", "main", ""),
            ("tag:release-1", "tag", "release-1"),
            ("build:orch/nightly#12", "build", "orch/nightly#12"),
            (f"artifact:{OTHER}", "artifact", OTHER),
            ("ref:ref-a", "ref", "ref-a"),
        ]:
            vs = cmp.parse_vs(text)
            self.assertEqual((vs.kind, vs.value), (kind, value), text)
        self.assertEqual(cmp.parse_vs("nightly").text, "main")

    def test_rejected(self):
        for bad in [
            "",
            "tomorrow",
            "main:x",
            "tag:",
            "tag:a b",
            "tag:a;rm",
            "tag:$(id)",
            "tag:`x`",
            "ref:",
            "ref:../x",
            "artifact:abc",
            "artifact:0B7F1C52-3D1E-4A39-9C21-5F6A8E2D4B10",
            "build:a/../b#1",
            "build:a#",
            "build:#1",
            "build:a#1a",
            "tag:x\ny",
            "ref:[x](http://e)",
            "tag:" + "a" * 129,
        ]:
            with self.assertRaises(ValueError, msg=bad):
                cmp.parse_vs(bad)

    def test_request_roundtrip_and_policy(self):
        r = od.parse_flags({"bench": "sanity", "vs": "ref:ref-a"})
        self.assertEqual(od.OnDemandRequest.from_dict(r.to_dict()).vs, "ref:ref-a")
        self.assertEqual(od.check_policy(r, od.POLICIES["ghe"]), [])
        self.assertEqual(od.check_policy(r, od.POLICIES["github.com"]), ["vs=ref is not available here"])
        self.assertEqual(
            od.check_policy(od.parse_flags({"bench": "sanity", "vs": "main"}), od.POLICIES["github.com"]), []
        )
        with self.assertRaises(od.RequestError):
            od.OnDemandRequest.from_dict({**r.to_dict(), "vs": "tag:a;b"})
        self.assertNotIn("<", str(od.RequestError("x")))


class Resolution(unittest.TestCase):
    def test_main_takes_the_newest_target_and_reuses_what_is_stored(self):
        results = [row("a", OTHER), row("b", OTHER), row("a", OLDER, run="run-old", days=3)]
        doc = run("main", results, [target(OTHER), target(OLDER, days=3)])
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["target"]["artifact_id"], OTHER)
        self.assertEqual(
            (doc["counts"]["ready"], doc["counts"]["cannot_compare"], doc["counts"]["will_run_other"]), (2, 1, 0)
        )
        self.assertEqual(by(doc)["a"]["other"]["run_id"], "run-o")
        self.assertEqual(by(doc)["c"]["why"], "no_stored_result")
        self.assertEqual(doc["other_requests"], [], "main is never measured here")

    def test_main_ignores_non_main_artifacts(self):
        doc = run("main", [row("a")], [target(OTHER, source="pr")])
        self.assertEqual(doc["refusal"]["code"], "vs_not_found")

    def test_main_needs_a_main_purpose(self):
        doc = run("main", [row("a", purpose="pr")], [target(OTHER)])
        self.assertEqual(by(doc)["a"]["why"], "not_a_main_result")
        self.assertEqual(run("main", [row("a", purpose="pr")], [target(OTHER)], purposes=("pr",))["counts"]["ready"], 1)

    def test_tag_build_artifact(self):
        t = target(OTHER, tags=["release-1"], build="orch/nightly#12")
        for vs in ("tag:release-1", "build:orch/nightly#12", f"artifact:{OTHER}"):
            doc = run(vs, [row("a", purpose="pr")], [t])
            self.assertEqual(doc["target"]["artifact_id"], OTHER, vs)
            self.assertEqual(by(doc)["a"]["state"], "ready", vs)
        self.assertEqual(run("tag:nope", [], [t])["refusal"]["code"], "vs_not_found")

    def test_only_the_missing_cases_are_planned_on_the_other_side(self):
        doc = run("tag:t1", [row("a", purpose="pr")], [target(OTHER, tags=["t1"])])
        self.assertEqual(by(doc)["a"]["state"], "ready")
        self.assertEqual({k for k, c in by(doc).items() if c["state"] == "will_run_other"}, {"b", "c"})
        (req,) = doc["other_requests"]
        self.assertEqual(req["selection"]["cases"], ["b", "c"])
        self.assertEqual(req["subject"], {"artifact_id": OTHER})
        self.assertEqual(req["reuse"]["mode"], "never")
        self.assertEqual(doc["estimate"], {"subject_device_hours": 3.0, "other_device_hours": 2.0, "known": True})

    def test_a_reused_or_skipped_subject_case_costs_nothing_and_a_skipped_one_cannot_compare(self):
        p = plan(
            [case("a", "reuse", "same_subject+same_environment", "run-s"), case("b"), case("c", "skip", "capacity")]
        )
        doc = run("tag:t1", [row("a"), row("b"), row("c")], [target(OTHER, tags=["t1"])], p=p)
        self.assertEqual(doc["estimate"]["subject_device_hours"], 1.0)
        self.assertEqual(by(doc)["c"]["state"], "cannot_compare")
        self.assertIn("subject_skip", by(doc)["c"]["why"])
        self.assertEqual(cmp.compare_links(doc), [{"case": "a", "a": "run-s", "b": "run-o", "benchmark_id": B["a"]}])
        self.assertEqual(cmp.compare_links(doc, {"b": "run-new"})[1]["a"], "run-new")


class Trust(unittest.TestCase):
    def test_other_side_must_be_valid_internal_and_not_invalidated(self):
        t = [target(OTHER, tags=["t1"])]
        for bad, why in [
            (row("a", validity="suspect"), "validity_suspect"),
            (row("a", validity="unqualified_env"), "validity_unqualified_env"),
            (row("a", validity="bogus"), "validity_bogus"),
            (row("a", trust="external"), "trust_external"),
        ]:
            doc = run("tag:t1", [bad], t)
            self.assertEqual((by(doc)["a"]["state"], by(doc)["a"]["why"]), ("cannot_compare", why))
            self.assertNotIn(
                "a", [n for r in doc["other_requests"] for n in r["selection"]["cases"]], "never re-measured silently"
            )
        ev = [{"id": "e1", "state": "invalid", "scope_kind": "run", "run_id": "run-o"}]
        self.assertEqual(by(run("tag:t1", [row("a")], t, events=ev))["a"]["why"], "invalidated")
        ev2 = [
            {"id": "e1", "state": "invalid", "scope_kind": "run_benchmark", "run_id": "run-o", "benchmark_id": B["b"]}
        ]
        d2 = by(run("tag:t1", [row("a"), row("b")], t, events=ev2))
        self.assertEqual((d2["a"]["state"], d2["b"]["why"]), ("ready", "invalidated"))
        ev3 = ev + [{"id": "e2", "state": "restored", "restores": "e1", "scope_kind": "run", "run_id": "run-o"}]
        self.assertEqual(by(run("tag:t1", [row("a")], t, events=ev3))["a"]["state"], "ready")

    def test_a_better_older_result_is_not_used_to_hide_the_newest_being_bad(self):
        results = [row("a", run="run-new", validity="suspect"), row("a", run="run-old", days=5)]
        doc = run("tag:t1", results, [target(OTHER, tags=["t1"])])
        self.assertEqual(by(doc)["a"]["other"]["run_id"], "run-old")

    def test_a_pr_run_stays_capped_and_is_never_a_baseline(self):
        doc = run("tag:t1", [row("a")], [target(OTHER, tags=["t1"])])
        self.assertEqual(doc["cap"], {"verdict": "suspected", "purpose_not_baseline": True})

    def test_host_must_allow_the_kind(self):
        gh = od.POLICIES["github.com"]
        doc = run("tag:t1", [row("a")], [target(OTHER, tags=["t1"])], host=gh)
        self.assertEqual(doc["refusal"]["code"], "vs_not_allowed")
        self.assertEqual(run("main", [row("a")], [target(OTHER)], host=gh)["status"], "ok")


class Environment(unittest.TestCase):
    def test_different_environment_is_refused_for_main_and_remeasured_for_a_named_artifact(self):
        r = [row("a", env="env-other")]
        doc = run("main", r, [target(OTHER)])
        self.assertEqual((by(doc)["a"]["state"], by(doc)["a"]["why"]), ("cannot_compare", "different_environment"))
        doc = run("artifact:" + OTHER, r, [target(OTHER)])
        self.assertEqual((by(doc)["a"]["state"], by(doc)["a"]["why"]), ("will_run_other", "different_environment"))

    def test_unknown_subject_environment_refuses_instead_of_guessing(self):
        doc = run("main", [row("a")], [target(OTHER)], p=plan(env_id=None))
        self.assertEqual(by(doc)["a"]["why"], "environment_unknown")

    def test_pinning_must_match(self):
        pinned = {**LABELS, "env.label.cpu_pinning": "numa"}
        r = [{**row("a"), "labels": pinned}]
        self.assertEqual(by(run("main", r, [target(OTHER)]))["a"]["why"], "different_pinning")

    def test_arch_and_product_must_match(self):
        doc = run("main", [row("a")], [target(OTHER, arch="s390x")])
        self.assertEqual(doc["refusal"]["code"], "vs_unusable")
        doc = run("main", [row("a")], [target(OTHER, component="torch-spyre")])
        self.assertEqual(doc["refusal"]["code"], "vs_unusable")
        doc = run("main", [row("a")], [target(OTHER, arch="s390x"), target(OLDER, days=1)], p=plan())
        self.assertEqual(doc["target"]["artifact_id"], OLDER, "the next target of the right arch is taken")

    def test_relax_matches_labels_not_the_id(self):
        r = [row("a", env="env-other")]
        doc = run("main", r, [target(OTHER)], relax=("env.label.node.*",))
        self.assertEqual(by(doc)["a"]["state"], "ready")
        r2 = [{**row("a", env="env-other"), "labels": {"env.label.node.class": "pool-b"}}]
        self.assertEqual(
            by(run("main", r2, [target(OTHER)], relax=("env.label.node.*",)))["a"]["why"], "different_environment"
        )

    def test_comparing_with_itself_is_refused(self):
        doc = run("artifact:" + SUBJ, [], [target(SUBJ)])
        self.assertEqual(doc["refusal"]["code"], "vs_is_subject")


class References(unittest.TestCase):
    def ref(self, **kw):
        return target(
            REFA,
            **{
                "source": "reference",
                "component": "reference",
                "ref_key": "ref-a",
                "ref_state": "current",
                "product": "spyre-inference",
                **kw,
            },
        )

    def refrow(self, name):
        return row(name, REFA, run=REFA, env=None, purpose="reference", reference=True)

    def test_a_reference_is_a_target_never_a_baseline(self):
        doc = run("ref:ref-a", [self.refrow("a")], [self.ref()])
        self.assertEqual(doc["target"]["role"], "reference_target")
        self.assertIs(doc["target"]["baseline_eligible"], False)
        self.assertEqual(by(doc)["a"]["state"], "ready")
        self.assertEqual(doc["other_requests"], [], "a reference is never measured")
        self.assertEqual(by(doc)["b"]["state"], "cannot_compare")

    def test_a_reference_is_reachable_only_through_ref(self):
        for vs in ("main", "tag:t1", f"artifact:{REFA}", "build:j#1"):
            t = self.ref(tags=["t1"], build="j#1")
            doc = run(vs, [self.refrow("a")], [t])
            self.assertIn(doc["refusal"]["code"], ("reference_only_via_ref", "vs_not_found"), vs)
            self.assertEqual(doc["counts"]["ready"], 0, vs)
        self.assertEqual(
            run(f"artifact:{REFA}", [self.refrow("a")], [self.ref()])["refusal"]["code"], "reference_only_via_ref"
        )

    def test_ref_needs_the_current_approved_version(self):
        for state in ("retired", "superseded", "untagged", ""):
            doc = run("ref:ref-a", [self.refrow("a")], [self.ref(ref_state=state)])
            self.assertEqual(doc["refusal"]["code"], "vs_unusable", state)

    def test_ref_does_not_resolve_an_ordinary_artifact(self):
        doc = run("ref:ref-a", [row("a")], [target(OTHER, ref_key="ref-a")])
        self.assertEqual(doc["status"], "refused")

    def test_a_reference_row_cannot_be_main(self):
        doc = run("ref:ref-a", [self.refrow("a")], [self.ref()], purposes=("nightly",))
        self.assertEqual(by(doc)["a"]["state"], "ready", "ref: results carry no purpose gate")


class Failure(unittest.TestCase):
    def test_store_outage_is_unavailable_not_empty(self):
        doc = run("main", [], [], fail="down")
        self.assertEqual((doc["status"], doc["refusal"]["code"]), ("unavailable", "store_unavailable"))
        self.assertEqual(doc["counts"]["cannot_compare"], 3)

    def test_no_store_and_no_subject(self):
        from spyre_bench.store import NullStore

        d = cmp.plan_compare(plan(), cmp.parse_vs("main"), NullStore(), product="spyre-inference", host_kinds=KINDS)
        self.assertEqual(d["refusal"]["code"], "no_store")
        d = run("main", [], [], p=plan(subject=""))
        self.assertEqual(d["refusal"]["code"], "no_subject")

    def test_the_target_is_pinned_between_preview_and_run(self):
        t1, t2 = target(OTHER), target(OLDER, days=-1)
        first = run("main", [row("a")], [t1])
        self.assertEqual(first["target"]["artifact_id"], OTHER)
        again = run("main", [row("a"), row("a", OLDER, run="r2")], [t1, t2], expect_artifact=OTHER)
        self.assertEqual(again["refusal"]["code"], "target_moved")
        self.assertNotEqual(first["digest"], again["digest"])

    def test_digest_is_stable_and_follows_content(self):
        a = run("main", [row("a")], [target(OTHER)])
        self.assertEqual(a["digest"], run("main", [row("a")], [target(OTHER)])["digest"])
        self.assertNotEqual(a["digest"], run("main", [row("a"), row("b")], [target(OTHER)])["digest"])


class Preview(unittest.TestCase):
    def setUp(self):
        od.SETTINGS.update({"enabled": True, "enforce_budgets": False})
        self.addCleanup(lambda: od.SETTINGS.update(od._load_settings()))
        self.req = od.parse_flags({"bench": "sanity", "vs": "tag:t1"})

    def doc(self):
        return run("tag:t1", [row("a", purpose="pr")], [target(OTHER, tags=["t1"])])

    def test_cost_includes_the_other_side_and_the_digest_follows_it(self):
        d = self.doc()
        plain = od.build_preview(plan(), self.req, GHE)
        pv = od.build_preview(plan(), self.req, GHE, cmp=d)
        self.assertEqual(plain["estimate"]["device_hours"], 3.0)
        self.assertEqual(pv["estimate"]["device_hours"], 5.0, "the estimate the budget judges covers both sides")
        self.assertEqual(pv["compare"]["estimate"]["other_device_hours"], 2.0)
        self.assertNotIn("compare", plain)
        self.assertNotEqual(plain["digest"], pv["digest"])
        self.assertEqual(plain["digest"], od.build_preview(plan(), self.req, GHE)["digest"])
        self.assertNotIn("value", json.dumps(pv["compare"]).replace("other_validity", ""))
        a = next(c for c in pv["compare"]["cases"] if c["name"] == "a")
        self.assertEqual((a["benchmark_id"], a["other_run"], a["subject_run"]), (B["a"], "run-o", ""))

    def test_markdown_names_ids_internally_and_not_publicly(self):
        pv = od.build_preview(plan(), self.req, GHE, cmp=self.doc())
        internal = od.render_markdown(pv, public=False)
        public = od.render_markdown(pv, public=True)
        self.assertIn("Compared with tag:t1", internal)
        self.assertIn("other side stored", internal)
        self.assertIn(OTHER[:8], internal)
        self.assertNotIn(OTHER[:8], public)
        self.assertIn("never becomes a baseline", public)
        self.assertIn("other side runs", public)

    def test_hostile_text_is_inert_in_the_comment(self):
        d = self.doc()
        d["cases"][0]["name"] = "x|[l](http://e)@org <b>"
        d["cases"][0]["state"] = "cannot_compare"
        d["cases"][0]["why"] = "![i](http://e/p.png) @team"
        out = od.render_markdown(od.build_preview(plan(), self.req, GHE, cmp=d), public=True)
        for bad in ("](http", "![", "<b>", "@team", "@org"):
            self.assertNotIn(bad, out)

    def test_budget_counts_both_sides(self):
        od.SETTINGS.update({"enforce_budgets": True})
        big = run("tag:t1", [], [target(OTHER, tags=["t1"])], p=plan([case(n) for n in "abcd"]))
        big["estimate"]["other_device_hours"] = 100.0
        pv = od.build_preview(plan(), self.req, GHE, cmp=big)
        self.assertTrue(pv["budget"]["over"])

    def test_request_row_records_the_target(self):
        d = self.doc()
        row_ = od.ci_request_row(
            self.req,
            "previewed",
            "dev1",
            host="ghe",
            pr=7,
            repo="o/r",
            head_sha="a" * 40,
            plan=plan(),
            cmp=d,
            now="2026-11-16T10:00:00Z",
        )
        p = row_["props"]
        self.assertEqual((p["bench.vs"], p["bench.vs.kind"], p["bench.vs.artifact"]), ("tag:t1", "tag", OTHER))
        self.assertEqual(p["bench.vs.subject_artifact"], SUBJ)
        self.assertEqual(p["bench.vs.digest"], d["digest"])
        self.assertEqual(json.loads(row_["canonical"])["vs"], "tag:t1")
        bare = od.ci_request_row(self.req, "previewed", "dev1", host="ghe", pr=7, repo="o/r", head_sha="a" * 40)
        self.assertNotIn("bench.vs", bare["props"])


class Cli(unittest.TestCase):
    def test_compare_then_preview_end_to_end(self):
        from spyre_bench.cli import main as cli_main

        with tempfile.TemporaryDirectory() as d:

            def w(name, obj):
                path = os.path.join(d, name)
                with open(path, "w") as f:
                    json.dump(obj, f, default=lambda x: x.isoformat())
                return path

            r = od.parse_flags({"bench": "sanity", "vs": "tag:t1"})
            pp, rp = w("plan.json", plan()), w("req.json", r.to_dict())
            sp = w("store.json", {"results": [row("a", purpose="pr")], "targets": [target(OTHER, tags=["t1"])]})

            def call(*argv):
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = cli_main(["ondemand", *argv])
                return code, out.getvalue(), err.getvalue()

            base = [
                "compare",
                "--plan",
                pp,
                "--request",
                rp,
                "--product",
                "spyre-inference",
                "--host",
                "ghe",
                "--store-file",
                sp,
            ]
            code, out, _ = call(*base)
            self.assertEqual(code, 0)
            doc = json.loads(out)
            self.assertEqual(doc["counts"], {"cases": 3, "ready": 1, "will_run_other": 2, "cannot_compare": 0})
            code, _, err = call(*base, "--expect-digest", "0" * 16)
            self.assertNotEqual(code, 0)
            self.assertIn("changed since the preview", err)
            self.assertEqual(call(*base, "--expect-digest", doc["digest"])[0], 0)
            self.assertEqual(
                call(*base, "--expect-artifact", OLDER)[0], 2, "a moved target fails the run, not just the document"
            )
            moved = json.loads(call(*base, "--expect-artifact", OLDER)[1])
            self.assertEqual(moved["refusal"]["code"], "target_moved")
            code, _, err = call(*base, "--relax-labels", "env.label.*", "--host", "github.com")
            self.assertNotEqual(code, 0)
            self.assertIn("not available here", err)
            code, md, _ = call(
                "preview", "--plan", pp, "--request", rp, "--host", "ghe", "--compare", w("cmp.json", doc)
            )
            self.assertEqual(code, 0)
            self.assertIn("Compared with tag:t1", md)
            rec = call(
                "record",
                "--request",
                rp,
                "--plan",
                pp,
                "--state",
                "previewed",
                "--actor",
                "dev1",
                "--host",
                "ghe",
                "--pr",
                "7",
                "--repo",
                "o/r",
                "--head-sha",
                "a" * 40,
                "--compare",
                w("cmp.json", doc),
            )
            self.assertEqual(json.loads(rec[1])["props"]["bench.vs.artifact"], OTHER)

    def test_a_missing_store_says_so(self):
        from spyre_bench.cli import main as cli_main

        with tempfile.TemporaryDirectory() as d:
            pp, rp = os.path.join(d, "p.json"), os.path.join(d, "r.json")
            json.dump(plan(), open(pp, "w"))
            json.dump(od.parse_flags({"bench": "sanity", "vs": "main"}).to_dict(), open(rp, "w"))
            out = io.StringIO()
            with redirect_stdout(out):
                cli_main(["ondemand", "compare", "--plan", pp, "--request", rp, "--product", "spyre-inference"])
            self.assertEqual(json.loads(out.getvalue())["refusal"]["code"], "no_store")


class SqlShape(unittest.TestCase):
    """The ClickHouse reads, against a recording client: parameters bound, never interpolated."""

    def test_values_are_bound_not_interpolated(self):
        from spyre_bench.store import ClickHouseStore

        seen = []

        class Result:
            column_names = ["artifact_id", "component", "arch", "ts", "source", "ref_key", "ref_state"]
            result_rows = []

        class Client:
            def query(self, sql, parameters=None):
                seen.append((sql, parameters))
                return Result()

        st = ClickHouseStore(client=Client(), database="db1")
        hostile = "x'; DROP TABLE t; --"
        for kind in ("tag", "build", "ref", "main", "artifact"):
            st.resolve_vs(kind, hostile if kind != "artifact" else OTHER, "spyre-inference", ("nightly",))
        for sql, params in seen:
            self.assertNotIn("DROP", sql)
            self.assertIn("{v:", sql) if "{v:" in sql else self.assertIn("{c:", sql)
        self.assertEqual(seen[0][1]["v"], hostile)
        self.assertEqual(st.stored_results([], OTHER), [])


if __name__ == "__main__":
    unittest.main()


class ReviewFindings(unittest.TestCase):
    def test_main_falls_back_to_the_newest_artifact_that_has_something_usable(self):
        results = [row("a", OTHER, validity="suspect"), row("a", OLDER, run="run-old", days=3)]
        doc = run("main", results, [target(OTHER), target(OLDER, days=3)])
        self.assertEqual(doc["target"]["artifact_id"], OLDER)
        self.assertEqual(by(doc)["a"]["other"]["run_id"], "run-old")
        doc = run("main", [row("a", OTHER, validity="suspect")], [target(OTHER), target(OLDER, days=3)])
        self.assertEqual(
            doc["target"]["artifact_id"], OTHER, "nothing usable anywhere: the newest is shown with its reasons"
        )
        self.assertEqual(by(doc)["a"]["why"], "validity_suspect")

    def test_an_unknown_arch_on_either_side_is_refused(self):
        self.assertEqual(run("main", [row("a")], [target(OTHER, arch="")])["refusal"]["code"], "vs_unusable")
        self.assertEqual(run("main", [row("a")], [target(OTHER)], p=plan(arch=""))["refusal"]["code"], "vs_unusable")

    def test_a_reference_for_another_product_is_refused(self):
        r = References()
        for product in ("torch-spyre", None):
            t = r.ref(product=product) if product else {k: v for k, v in r.ref().items() if k != "product"}
            self.assertEqual(run("ref:ref-a", [r.refrow("a")], [t])["refusal"]["code"], "vs_unusable", product)

    def test_reference_values_are_refused_by_every_other_kind_and_ref_takes_only_reference_values(self):
        leaked = row("a", purpose="reference")
        for vs in ("tag:t1", f"artifact:{OTHER}", "main"):
            doc = run(vs, [leaked], [target(OTHER, tags=["t1"])], purposes=("nightly", "reference"))
            self.assertEqual(by(doc)["a"]["why"], "reference_only_via_ref", vs)
        r = References()
        doc = run("ref:ref-a", [row("a", REFA, run=REFA)], [r.ref()])
        self.assertEqual(by(doc)["a"]["why"], "not_a_reference_value")

    def test_a_reference_comparison_offers_no_run_to_open(self):
        r = References()
        doc = run(
            "ref:ref-a", [r.refrow("a")], [r.ref()], p=plan([case("a", "reuse", "x", "run-s"), case("b"), case("c")])
        )
        self.assertEqual(by(doc)["a"]["other"]["run_id"], "")
        self.assertEqual(cmp.compare_links(doc), [])

    def test_the_moved_message_and_public_comment_name_no_artifact(self):
        st = MemoryStore(results=[row("a")], targets=[target(OLDER)])
        doc = cmp.plan_compare(
            plan(), cmp.parse_vs("main"), st, product="spyre-inference", host_kinds=KINDS, expect_artifact=OTHER
        )
        self.assertEqual(doc["refusal"]["code"], "target_moved")
        self.assertNotIn(OLDER[:8], json.dumps(doc["refusal"]))
        self.assertNotIn(OTHER[:8], json.dumps(doc["refusal"]))
        od.SETTINGS.update({"enabled": True, "enforce_budgets": False})
        self.addCleanup(lambda: od.SETTINGS.update(od._load_settings()))
        req = od.parse_flags({"bench": "sanity", "vs": "main"})
        out = od.render_markdown(od.build_preview(plan(), req, GHE, cmp=doc), public=True)
        self.assertIn("the comparison target changed since the preview", out)

    def test_an_unusable_estimate_is_unknown_not_zero(self):
        p = plan([case("a"), {**case("b"), "est_min": None}, {k: v for k, v in case("c").items() if k != "est_min"}])
        doc = run("tag:t1", [], [target(OTHER, tags=["t1"])], p=p)
        self.assertIs(doc["estimate"]["known"], False)
        od.SETTINGS.update({"enabled": True, "enforce_budgets": False})
        self.addCleanup(lambda: od.SETTINGS.update(od._load_settings()))
        pv = od.build_preview(p, od.parse_flags({"bench": "sanity", "vs": "tag:t1"}), GHE, cmp=doc)
        self.assertFalse(pv["budget"]["estimate_known"])
        self.assertIs(run("tag:t1", [row("a")], [target(OTHER, tags=["t1"])])["estimate"]["known"], True)

    def test_amd64_and_x86_64_are_the_same_platform_and_main_skips_the_subject(self):
        doc = run("main", [row("a")], [target(OTHER, arch="amd64")])
        self.assertEqual(doc["status"], "ok")
        doc = run("main", [row("a", OLDER, run="r2", days=1)], [target(SUBJ), target(OLDER, days=1)])
        self.assertEqual(doc["target"]["artifact_id"], OLDER)
        self.assertEqual(run("artifact:" + SUBJ, [], [target(SUBJ)])["refusal"]["code"], "vs_is_subject")

    def test_the_compare_view_names_the_subject_artifact(self):
        d = run("tag:t1", [row("a")], [target(OTHER, tags=["t1"])])
        self.assertEqual(od.compare_view(d)["subject_artifact"], SUBJ)


class RealPlan(unittest.TestCase):
    """The compare step over a plan the real planner produced from the demo catalog."""

    def test_the_demo_plan_compared_with_a_stored_main_artifact(self):
        from test_run import SUBJECT as SUB
        from test_run import plan_doc

        doc = plan_doc("--purpose", "pr", "--subject-id", SUB, "--env-id", "env-1")
        cases = [c for c in doc["cases"] if c["action"] == "run"]
        self.assertGreaterEqual(len(cases), 2)
        stored = [
            {**row("a", OTHER, env="env-1"), "benchmark_id": cases[0]["benchmark_id"], "labels": cases[0]["labels"]},
        ]
        st = MemoryStore(results=stored, targets=[target(OTHER, component="run-demo", arch=doc["leg"]["arch"])])
        out = cmp.plan_compare(doc, cmp.parse_vs("main"), st, product="run-demo", host_kinds=KINDS)
        states = {c["name"]: c["state"] for c in out["cases"]}
        self.assertEqual(states[cases[0]["name"]], "ready")
        self.assertEqual(states[cases[1]["name"]], "cannot_compare")
        pv = od.build_preview(doc, od.parse_flags({"bench": "sanity", "vs": "main"}), GHE, cmp=out)
        self.assertEqual(pv["compare"]["counts"]["ready"], 1)

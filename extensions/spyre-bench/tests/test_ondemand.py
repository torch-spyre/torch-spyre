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

"""On-demand PR requests: grammar, policy, budgets, supersede, fork safety, preview.
python -m pytest -q -p no:cacheprovider tests/test_ondemand.py
"""

import json
import unittest

import yaml

from spyre_bench import ondemand as od
from spyre_bench.cli import main as cli_main
from test_run import SUBJECT, TMP, capture, plan_doc, plan_main

SHA = "a" * 40
SHA2 = "b" * 40
NOW = "2026-11-16T10:00:00Z"
WRITER = {"login": "dev1", "permission": "write", "association": "COLLABORATOR"}
READER = {"login": "rando", "permission": "read", "association": "NONE"}
GH, GHE = od.POLICIES["github.com"], od.POLICIES["ghe"]


def req(**flags):
    return od.parse_flags({"bench": "sanity", **flags})


def facts(request=None, host="github.com", actor=None, head_repo="o/r", sha=SHA, **kw):
    r = request or req()
    return {
        "host": host,
        "now": NOW,
        "actor": actor or WRITER,
        "pr": {"number": 7, "base_repo": "o/r", "head_repo": head_repo, "head_sha": sha},
        "request": r.to_dict(),
        **kw,
    }


class Grammar(unittest.TestCase):
    def sel(self, text):
        return od.parse_selection(text)

    def test_tier_default_product(self):
        self.assertEqual(self.sel("sanity"), (("spyre-inference", {"tier": "sanity"}, "sanity"),))

    def test_suites_with_tier(self):
        ((p, s, _),) = self.sel("hf-adapters:decoder+startup@core")
        self.assertEqual((p, s), ("hf-adapters", {"suites": ["decoder", "startup"], "tier": "core"}))

    def test_suite_without_tier_has_no_tier_key(self):
        ((_, s, _),) = self.sel("decoder+startup")
        self.assertNotIn("tier", s)

    def test_case_list_and_named_selection(self):
        self.assertEqual(self.sel("a_1,b-2")[0][1], {"cases": ["a_1", "b-2"]})
        self.assertEqual(self.sel("#pr-sanity")[0][1], {"cases": ["pr-sanity"]})

    def test_two_products(self):
        got = self.sel("spyre-inference:decoder@core;torch-spyre:ladder@core")
        self.assertEqual([g[0] for g in got], ["spyre-inference", "torch-spyre"])

    def test_errors(self):
        for text, frag in [
            ("", "needs a selection"),
            ("spyre-infernce:sanity", "did you mean 'spyre-inference'"),
            ("decoder@sanitty", "did you mean 'sanity'"),
            ("a,b@core", "not both"),
            ("torch-spyre:a;torch-spyre:b", "twice"),
            ("bad name", "letters, digits"),
            ("a$(id)", "letters, digits"),
            ("#", "selection name"),
        ]:
            with self.subTest(text), self.assertRaises(od.RequestError) as cm:
                self.sel(text)
            self.assertIn(frag, str(cm.exception))


class Flags(unittest.TestCase):
    def test_defaults(self):
        r = req()
        self.assertEqual(
            (r.reuse, r.profile, r.vs, r.clean, r.env, r.purpose), ("if_valid", "off", "nightly", False, "", "pr")
        )

    def test_prefix_optional(self):
        a, b = req(reuse="only"), req(**{"bench-reuse": "only"})
        self.assertEqual(a.reuse, b.reuse)

    def test_duplicate_after_prefix_strip(self):
        with self.assertRaises(od.RequestError):
            od.parse_flags({"bench": "sanity", "reuse": "only", "bench-reuse": "never"})

    def test_bad_values(self):
        for flags, frag in [
            ({"reuse": "alwys"}, "reuse"),
            ({"purpose": "nightly"}, "only ask for purpose 'pr'"),
            ({"profile": "heavy"}, "profile"),
            ({"vs": "tomorrow"}, "vs="),
            ({"clean": "maybe"}, "true or false"),
            ({"max-wait": "45m"}, "only applies with clean"),
            ({"clean": "true", "max-wait": "45"}, "duration"),
            ({"clean": "true", "max-wait": "0m"}, "duration"),
            ({"budget": "nan"}, "budget"),
            ({"budget": "-1"}, "budget"),
            ({"budget": "inf"}, "budget"),
            ({"budjet": "3"}, "did you mean 'budget'"),
            ({"force": "a", "reuse": "never"}, "no effect"),
            ({"force": "all,a"}, "cannot be combined"),
        ]:
            with self.subTest(flags), self.assertRaises(od.RequestError) as cm:
                req(**flags)
            self.assertIn(frag, str(cm.exception))

    def test_roundtrip(self):
        r = req(reuse="prefer", force="x,y", clean="true", **{"max-wait": "45m"}, budget="3")
        self.assertEqual(od.OnDemandRequest.from_dict(json.loads(json.dumps(r.to_dict()))), r)
        self.assertEqual(r.max_wait_s, 2700)

    def test_missing_bench(self):
        with self.assertRaises(od.RequestError):
            od.parse_flags({"reuse": "never"})


class Command(unittest.TestCase):
    def test_positional_and_flags(self):
        c = od.parse_command("hi\n/bench decoder@core reuse=never clean=true\nbye")
        self.assertEqual((c.kind, c.flags), ("run", {"bench": "decoder@core", "reuse": "never", "clean": "true"}))

    def test_confirm_cancel(self):
        self.assertEqual(od.parse_command("/bench confirm").kind, "confirm")
        self.assertEqual(od.parse_command("/bench cancel").kind, "cancel")

    def test_ignored_inside_fence_quote_indent_and_prose(self):
        for text in [
            "```\n/bench sanity\n```",
            "> /bench sanity",
            "  /bench sanity",
            "please /bench sanity",
            "/benchmark sanity",
            "",
        ]:
            with self.subTest(text):
                self.assertIsNone(od.parse_command(text))

    def test_after_closed_fence(self):
        self.assertIsNotNone(od.parse_command("```\nx\n```\n/bench sanity"))

    def test_stray_token_and_limits(self):
        with self.assertRaises(od.RequestError):
            od.parse_command("/bench sanity reuse")
        with self.assertRaises(od.RequestError):
            od.parse_command("/bench " + " ".join(f"a{i}=1" for i in range(50)))
        with self.assertRaises(od.RequestError):
            od.parse_command("/bench sanity bench=core")

    def test_crlf(self):
        self.assertEqual(od.parse_command("/bench sanity\r\nmore").flags, {"bench": "sanity"})

    def test_custom_command(self):
        self.assertEqual(od.parse_command("/spyre-test bench=sanity", "/spyre-test").flags, {"bench": "sanity"})


class Policy(unittest.TestCase):
    def violations(self, host, actor=None, **flags):
        b = flags.pop("bench", "sanity")
        return od.check_policy(od.parse_flags({"bench": b, **flags}), od.POLICIES[host], actor)

    def test_public_limits(self):
        self.assertEqual(self.violations("github.com"), [])
        for flags in [
            {"bench": "core"},
            {"bench": "extended"},
            {"reuse": "never"},
            {"force": "all"},
            {"profile": "light"},
            {"env": "e1"},
        ]:
            with self.subTest(flags):
                self.assertTrue(self.violations("github.com", **flags))

    def test_internal_allows_more_but_not_extended(self):
        self.assertEqual(self.violations("ghe", bench="core", force="all", profile="light", env="e1"), [])
        self.assertEqual(self.violations("ghe", reuse="never"), [])
        self.assertTrue(self.violations("ghe", bench="extended"))
        self.assertTrue(self.violations("ghe", profile="full"))

    def test_clean_needs_write_on_both_hosts(self):
        for host in od.HOSTS:
            self.assertTrue(self.violations(host, READER, clean="true"))
            self.assertEqual(self.violations(host, WRITER, clean="true"), [])

    def test_max_wait_cap(self):
        self.assertTrue(self.violations("ghe", WRITER, clean="true", **{"max-wait": "2h"}))
        self.assertEqual(self.violations("ghe", WRITER, clean="true", **{"max-wait": "60m"}), [])

    def test_explicit_budget_above_host_cap(self):
        self.assertTrue(self.violations("github.com", budget="50"))

    def test_invoker(self):
        self.assertTrue(od.invoker_allowed(WRITER, GH))
        self.assertFalse(od.invoker_allowed({"permission": "read", "association": "MEMBER"}, GH))
        self.assertTrue(od.invoker_allowed({"permission": "read", "association": "MEMBER"}, GHE))
        self.assertFalse(od.invoker_allowed({"permission": "read", "association": "NONE"}, GHE))
        self.assertFalse(od.invoker_allowed(None, GH))
        self.assertFalse(od.invoker_allowed({"permission": "triage"}, GH))


class Budget(unittest.TestCase):
    def test_limit_uses_largest_tier_and_explicit_budget_only_lowers(self):
        r = od.parse_flags({"bench": "a@sanity;torch-spyre:b@core"})
        self.assertEqual(od.budget_limit(r, GHE), 48.0)
        self.assertEqual(od.budget_limit(od.parse_flags({"bench": "core", "budget": "5"}), GHE), 5.0)
        self.assertEqual(od.budget_limit(od.parse_flags({"bench": "sanity", "budget": "500"}), GH), 4.0)

    def test_check(self):
        self.assertEqual(od.check_budget(1, 4, GH), (True, False))
        self.assertEqual(od.check_budget(2.01, 4, GH), (True, True))
        self.assertEqual(od.check_budget(2.0, 4, GH), (True, False))
        self.assertEqual(od.check_budget(4.0, 4, GH), (True, True))
        self.assertEqual(od.check_budget(4.01, 4, GH), (False, False))


class Approvals(unittest.TestCase):
    def test_make_requires_write_and_full_sha(self):
        with self.assertRaises(od.RequestError):
            od.make_approval(READER, SHA, req(), NOW)
        with self.assertRaises(od.RequestError):
            od.make_approval(WRITER, "abc123", req(), NOW)
        with self.assertRaises(od.RequestError):
            od.make_approval(WRITER, "A" * 40, req(), NOW)

    def test_valid_only_for_same_sha_request_and_fresh(self):
        a = od.make_approval(WRITER, SHA, req(), NOW)
        self.assertEqual(od.approval_valid(a, SHA, req(), NOW, GH), (True, ""))
        self.assertFalse(od.approval_valid(a, SHA2, req(), NOW, GH)[0])
        self.assertFalse(od.approval_valid(a, SHA, req(reuse="only"), NOW, GH)[0])
        self.assertFalse(od.approval_valid(a, SHA, req(), "2026-11-20T10:00:01Z", GH)[0])
        self.assertFalse(od.approval_valid(None, SHA, req(), NOW, GH)[0])
        forged = od.Approval(SHA, "x", "read", od.request_digest(req()), NOW)
        self.assertFalse(od.approval_valid(forged, SHA, req(), NOW, GH)[0])

    def test_digest_ignores_how_it_was_written(self):
        self.assertEqual(od.request_digest(req(reuse="only")), od.request_digest(req(**{"bench-reuse": "only"})))

    def test_sandbox_has_no_secrets(self):
        for fork in (True, False):
            p = od.sandbox_profile(fork)
            self.assertFalse(p["secrets"] or p["registry_write"] or p["store_credentials"] or p["baseline_eligible"])
            self.assertEqual(p["checkout"], "pinned_sha")
        self.assertEqual(od.sandbox_profile(True)["provenance"], "fork")


class Authorize(unittest.TestCase):
    def test_same_repo_admit(self):
        d = od.authorize(facts())
        self.assertEqual((d.state, d.pinned_sha, d.sandbox["provenance"]), ("admit", SHA, "branch"))

    def test_reader_denied(self):
        self.assertFalse(od.authorize(facts(actor=READER)).allowed)

    def test_fork_with_write_commenter_gets_pinned_approval(self):
        d = od.authorize(facts(head_repo="someone/r"))
        self.assertTrue(d.allowed)
        self.assertEqual(d.approval["sha"], SHA)
        self.assertEqual(d.sandbox["provenance"], "fork")

    def test_fork_member_without_write_denied_on_ghe(self):
        a = {"login": "m", "permission": "read", "association": "MEMBER"}
        d = od.authorize(facts(host="ghe", actor=a, head_repo="x/r"))
        self.assertEqual(d.state, "deny")
        self.assertIn("fork", d.reasons[0])

    def test_head_moved_after_request(self):
        d = od.authorize(facts(pinned_sha=SHA2))
        self.assertEqual(d.state, "deny")
        self.assertIn("head moved", d.reasons[0])

    def test_stale_approval_for_old_sha_cannot_run_new_head(self):
        old = od.make_approval(WRITER, SHA2, req(), NOW).to_dict()
        d = od.authorize(facts(head_repo="x/r", actor=READER, approval=old))
        self.assertEqual(d.state, "deny")

    def test_approval_by_another_writer_covers_a_reader_requester_on_ghe_member(self):
        ok = od.make_approval(WRITER, SHA, req(), NOW).to_dict()
        a = {"login": "m", "permission": "read", "association": "MEMBER"}
        self.assertTrue(od.authorize(facts(host="ghe", actor=a, head_repo="x/r", approval=ok)).allowed)

    def test_bad_sha(self):
        self.assertEqual(od.authorize(facts(sha="main")).state, "deny")

    def test_policy_violation_denies(self):
        self.assertEqual(od.authorize(facts(request=req(reuse="never"))).state, "deny")

    def test_estimate_over_limit_and_confirm_band(self):
        self.assertEqual(od.authorize(facts(estimate_device_hours=5)).state, "deny")
        self.assertEqual(od.authorize(facts(estimate_device_hours=3)).state, "needs_confirm")
        self.assertEqual(od.authorize(facts(estimate_device_hours=1)).state, "admit")

    def history(self, **kw):
        return [
            {
                "id": "r1",
                "pr": 7,
                "user": "dev1",
                "ts": "2026-11-16T09:00:00Z",
                "state": "active",
                "est": 3,
                "clean": False,
                **kw,
            }
        ]

    def test_new_request_supersedes_the_active_one_on_the_same_pr(self):
        d = od.authorize(facts(history=self.history(), estimate_device_hours=1))
        self.assertEqual((d.state, d.cancel), ("admit", ["r1"]))

    def test_other_pr_not_cancelled(self):
        d = od.authorize(facts(history=self.history(pr=8), estimate_device_hours=1))
        self.assertEqual(d.cancel, [])

    def test_finished_request_not_cancelled_but_still_counts_toward_window(self):
        h = self.history(state="done", est=7)
        d = od.authorize(facts(history=h, estimate_device_hours=3))
        self.assertEqual(d.cancel, [])
        self.assertEqual(d.state, "deny")
        self.assertIn("window", d.reasons[0])

    def test_cancelled_work_does_not_count(self):
        h = self.history(state="cancelled", est=7)
        self.assertTrue(od.authorize(facts(history=h, estimate_device_hours=3)).allowed)

    def test_superseded_estimate_is_freed(self):
        h = self.history(est=7)
        d = od.authorize(facts(history=h, estimate_device_hours=3))
        self.assertTrue(d.allowed)
        self.assertEqual(d.cancel, ["r1"])

    def test_window_expires(self):
        h = self.history(state="done", est=7, ts="2026-11-14T09:00:00Z")
        self.assertTrue(od.authorize(facts(history=h, estimate_device_hours=3)).allowed)

    def test_rate_limit(self):
        h = [
            {"id": f"r{i}", "pr": 100 + i, "user": "dev1", "ts": "2026-11-16T09:30:00Z", "state": "done", "est": 0}
            for i in range(6)
        ]
        d = od.authorize(facts(history=h))
        self.assertEqual(d.state, "deny")
        self.assertIn("rate", d.reasons[0])

    def test_clean_quotas(self):
        r = req(clean="true")
        busy = self.history(pr=9, clean=True)
        d = od.authorize(facts(request=r, history=busy))
        self.assertIn("concurrent", " ".join(d.reasons))
        day = [
            self.history(pr=20 + i, clean=True, state="done", ts="2026-11-16T01:00:00Z", id=f"d{i}")[0]
            for i in range(3)
        ]
        self.assertIn("per day", " ".join(od.authorize(facts(request=r, history=day)).reasons))
        cluster = [
            {
                "id": f"c{i}",
                "pr": 30 + i,
                "user": f"u{i}",
                "ts": "2026-11-16T09:30:00Z",
                "state": "active",
                "est": 1,
                "clean": True,
            }
            for i in range(2)
        ]
        self.assertIn("at once", " ".join(od.authorize(facts(request=r, history=cluster)).reasons))

    def test_clean_by_reader_denied(self):
        self.assertEqual(od.authorize(facts(request=req(clean="true"), actor=READER)).state, "deny")


class EngineMapping(unittest.TestCase):
    def test_engine_request_shape(self):
        r = od.parse_flags(
            {
                "bench": "spyre-inference:decoder@core",
                "reuse": "prefer",
                "force": "x",
                "budget": "3",
                "profile": "light",
            }
        )
        e = od.engine_request(r, "spyre-inference", {"artifact_id": SUBJECT}, "env-a")
        self.assertEqual(e["selection"], {"suites": ["decoder"], "tier": "core", "type": "perf"})
        self.assertEqual(e["reuse"], {"mode": "prefer", "override": True, "force": ["x"]})
        self.assertEqual(
            (e["budget"], e["overrides"], e["environment"]),
            ({"device_hours": 3.0}, {"profile_level": "light"}, "env-a"),
        )
        with self.assertRaises(KeyError):
            od.engine_request(r, "hf-adapters")

    def test_env_request_beats_default_environment(self):
        r = req(env="mine")
        self.assertEqual(od.engine_request(r, "spyre-inference", environment="dflt")["environment"], "mine")

    def test_bench_env(self):
        r = od.parse_flags({"bench": "a_1,b", "force": "all", "profile": "light"})
        env = od.bench_env(r, "spyre-inference", SUBJECT)
        self.assertEqual(env["BENCH_SELECTION"], "a_1,b")
        self.assertEqual(
            (env["BENCH_FORCE"], env["BENCH_SET"], env["BENCH_SUBJECT_ID"]), ("all", "run.profile_level=light", SUBJECT)
        )

    def test_the_real_planner_accepts_the_request(self):
        r = od.parse_flags({"bench": "op_add_fp16,op_mul_fp16", "budget": "1"})
        f = TMP / "ondemand-req.yaml"
        f.write_text(yaml.safe_dump(od.engine_request(r, "spyre-inference", {"artifact_id": SUBJECT})))
        doc = plan_doc("--request", str(f))
        self.assertEqual((doc["summary"]["run"], doc["purpose"]), (2, "pr"))
        self.assertEqual(doc["request"]["reuse"]["mode"], "if_valid")


class Preview(unittest.TestCase):
    def plan(self, *argv):
        return plan_doc("--purpose", "pr", "--subject-id", SUBJECT, *argv)

    def test_counts_estimate_and_order(self):
        pv = od.build_preview(self.plan(), req(), GH)
        self.assertEqual(pv["counts"], {"cases": 2, "run": 2, "reuse": 0, "skip": 0})
        self.assertEqual([c["name"] for c in pv["cases"]], ["op_add_fp16", "op_mul_fp16"])
        self.assertFalse(pv["budget"]["over"])

    def test_stable(self):
        a = od.render_markdown(od.build_preview(self.plan(), req(), GH))
        b = od.render_markdown(od.build_preview(self.plan(), req(), GH))
        self.assertEqual(a, b)
        self.assertTrue(a.startswith(od.PREVIEW_MARKER))

    def test_skips_show_in_diff_with_reasons(self):
        code, out, _ = plan_main("--json", "--purpose", "pr", "--subject-id", SUBJECT, "--devices", "spyre/pf=0")
        self.assertEqual(code, 4)
        pv = od.build_preview(json.loads(out), req(), GH)
        self.assertEqual(pv["counts"]["skip"], 2)
        self.assertTrue(any(d["field"] == "cases" and "skipped" in d["planned"] for d in pv["diff"]))
        self.assertTrue(pv["reasons"])

    def test_force_of_unknown_case_is_flagged(self):
        pv = od.build_preview(self.plan(), req(force="nope"), GH)
        self.assertTrue(any(d["field"] == "force" and d["asked"] == "nope" for d in pv["diff"]))

    def test_suite_without_tier_is_flagged(self):
        pv = od.build_preview(self.plan(), od.parse_flags({"bench": "ops+more"}), GH)
        self.assertTrue(any("tier" in d["field"] and d["planned"] == "sanity" for d in pv["diff"]))

    def test_reuse_asked_but_not_applied(self):
        pv = od.build_preview(self.plan(), req(reuse="prefer"), GH)
        self.assertTrue(any(d["field"] == "reuse" for d in pv["diff"]))

    def test_over_budget_and_confirm_text(self):
        plan = self.plan()
        plan["estimate"]["device_hours"] = 5.0
        self.assertIn("nothing will run", od.render_markdown(od.build_preview(plan, req(), GH)))
        plan["estimate"]["device_hours"] = 3.0
        text = od.render_markdown(od.build_preview(plan, req(), GH))
        self.assertIn("`/bench confirm`", text)
        self.assertIn("3 device-hours", text)

    def test_untrusted_text_cannot_inject(self):
        plan = self.plan()
        plan["cases"][0]["name"] = "x|y @team `<img src=x>`\n# h"
        plan["cases"][0]["action"] = "skip"
        plan["cases"][0]["reason"] = "see @org/team <b>"
        text = od.render_markdown(od.build_preview(plan, req(), GH))
        self.assertNotIn("@team", text)
        self.assertNotIn("<img", text)
        self.assertNotIn("<b>", text)
        self.assertEqual(text.count("\n# h"), 0)
        for line in text.splitlines():
            if line.startswith("| skip"):
                self.assertEqual(line.count("|"), 5)

    def test_long_plan_is_truncated(self):
        plan = self.plan()
        plan["cases"] = [dict(plan["cases"][0], name=f"c{i:03d}") for i in range(100)]
        plan["summary"]["cases"] = 100
        text = od.render_markdown(od.build_preview(plan, req(), GH))
        self.assertIn("…and 60 more", text)

    def test_public_comment_has_no_metric_fields(self):
        text = od.render_markdown(od.build_preview(self.plan(), req(), GH), public=True)
        for word in ("latency", "throughput", "tokens/s", "ms "):
            self.assertNotIn(word, text)

    def test_json_is_serialisable_and_versioned(self):
        pv = od.build_preview(
            self.plan(), req(), GH, queue={"queue": "ci-x1"}, dashboard_url="https://example.invalid/r/1"
        )
        self.assertEqual(json.loads(json.dumps(pv))["schema"], "bench.preview/1")
        self.assertIn("ci-x1", od.render_markdown(pv))


class Cli(unittest.TestCase):
    def test_parse_authorize_preview(self):
        c = TMP / "ondemand-comment.txt"
        c.write_text("run it\n/bench sanity reuse=prefer\n")
        code, out, _ = capture(cli_main, ["ondemand", "parse", "--comment-file", str(c)])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc["policy_violations"], [])
        rf, ff = TMP / "ondemand-r.json", TMP / "ondemand-f.json"
        rf.write_text(json.dumps(doc["request"]))
        ff.write_text(json.dumps(facts(request=od.OnDemandRequest.from_dict(doc["request"]))))
        code, out, _ = capture(cli_main, ["ondemand", "authorize", "--facts", str(ff)])
        self.assertEqual((code, json.loads(out)["state"]), (0, "admit"))
        pf = TMP / "ondemand-p.json"
        pf.write_text(json.dumps(plan_doc("--purpose", "pr", "--subject-id", SUBJECT)))
        code, out, _ = capture(cli_main, ["ondemand", "preview", "--plan", str(pf), "--request", str(rf)])
        self.assertEqual(code, 0)
        self.assertIn("Benchmark plan", out)

    def test_no_command_and_bad_command(self):
        c = TMP / "ondemand-none.txt"
        c.write_text("hello")
        self.assertEqual(capture(cli_main, ["ondemand", "parse", "--comment-file", str(c)])[0], 4)
        c.write_text("/bench sanity reuse=zzz")
        code, _, err = capture(cli_main, ["ondemand", "parse", "--comment-file", str(c)])
        self.assertEqual(code, 2)
        self.assertIn("reuse", err)

    def test_denied_exit_code(self):
        ff = TMP / "ondemand-f2.json"
        ff.write_text(json.dumps(facts(actor=READER)))
        self.assertEqual(capture(cli_main, ["ondemand", "authorize", "--facts", str(ff)])[0], 2)


if __name__ == "__main__":
    unittest.main()

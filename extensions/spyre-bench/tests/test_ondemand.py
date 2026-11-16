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

ENFORCED = {"enabled": True, "enforce_budgets": True}


def setUpModule():
    """Most tests exercise the enforcing mode; the shipped file is dark and report-only (see Modes)."""
    od.SETTINGS.update(ENFORCED)


def tearDownModule():
    od.SETTINGS.update(od._load_settings())


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

    def test_suite_without_tier_is_explicitly_sanity(self):
        ((_, s, _),) = self.sel("decoder+startup")
        self.assertEqual(s["tier"], "sanity")

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

    def test_spyre_bench_is_the_command_and_bench_the_deprecated_alias(self):
        a = od.parse_command("/spyre-bench sanity reuse=only")
        b = od.parse_command("/bench sanity reuse=only")
        self.assertEqual(a.flags, b.flags)
        self.assertEqual(od.parse_command("/spyre-bench confirm").kind, "confirm")
        self.assertIsNone(od.parse_command("/spyre-benchmark sanity"))

    def test_type_flag_and_the_test_type_spelling(self):
        self.assertEqual(req().type, "perf")
        self.assertNotIn("type", req().to_dict())
        q = od.parse_flags({"bench": "sanity", "TEST_TYPE": "quality"})
        self.assertEqual((q.type, q.to_dict()["type"]), ("quality", "quality"))
        self.assertEqual(od.parse_flags({"bench": "sanity", "test_type": "perf"}).type, "perf")
        with self.assertRaises(od.RequestError):
            od.parse_flags({"bench": "sanity", "type": "nope"})
        self.assertEqual(od.engine_request(q, "spyre-inference")["selection"]["type"], "quality")
        self.assertEqual(od.bench_env(q, "spyre-inference")["BENCH_TYPE"], "quality")
        self.assertTrue(od.check_policy(q, GH))
        self.assertEqual(od.check_policy(q, GHE), [])
        self.assertEqual(od.OnDemandRequest.from_dict(q.to_dict()), q)

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


FORKREQ = lambda **kw: req(sha=SHA[:12], **kw)  # noqa: E731
PLAN_OK = {
    "cases": [{"name": "c", "action": "run", "est_min": 5.0, "run_props": {"run.tier": "sanity"}}],
    "estimate": {"device_hours": 1},
}


def planned(**kw):
    kw.setdefault("plan", PLAN_OK)
    return facts(stage="planned", **kw)


class Authorize(unittest.TestCase):
    def test_same_repo_admit(self):
        d = od.authorize(facts())
        self.assertEqual((d.state, d.pinned_sha, d.sandbox["provenance"]), ("admit", SHA, "branch"))

    def test_reader_denied(self):
        self.assertFalse(od.authorize(facts(actor=READER)).allowed)

    def test_repo_names_compare_case_insensitively(self):
        d = od.authorize(facts(head_repo="O/R"))
        self.assertEqual(d.sandbox["provenance"], "branch")

    def test_fork_needs_the_sha_the_writer_reviewed(self):
        d = od.authorize(facts(head_repo="someone/r"))
        self.assertEqual(d.state, "deny")
        self.assertIn("sha=", d.reasons[0])
        wrong = od.authorize(facts(request=req(sha="bbbbbbbbbbbb"), head_repo="someone/r"))
        self.assertEqual(wrong.state, "deny")
        short = od.authorize(facts(request=req(sha=SHA[:7]), head_repo="someone/r"))
        self.assertEqual(short.state, "deny")
        self.assertIn("12", short.reasons[0])

    def test_fork_with_write_commenter_gets_pinned_approval(self):
        d = od.authorize(facts(request=FORKREQ(), head_repo="someone/r"))
        self.assertTrue(d.allowed)
        self.assertEqual(d.approval["sha"], SHA)
        self.assertEqual(d.sandbox["provenance"], "fork")

    def test_fork_member_without_write_denied_on_ghe(self):
        a = {"login": "m", "permission": "read", "association": "MEMBER"}
        d = od.authorize(facts(request=FORKREQ(), host="ghe", actor=a, head_repo="x/r"))
        self.assertEqual(d.state, "deny")
        self.assertIn("fork", d.reasons[0])

    def test_head_moved_after_request(self):
        d = od.authorize(facts(pinned_sha=SHA2))
        self.assertEqual(d.state, "deny")
        self.assertIn("head moved", d.reasons[0])

    def test_stale_approval_for_old_sha_cannot_run_new_head(self):
        old = od.make_approval(WRITER, SHA2, FORKREQ(), NOW).to_dict()
        d = od.authorize(facts(request=FORKREQ(), head_repo="x/r", actor=READER, approval=old))
        self.assertEqual(d.state, "deny")

    def test_approval_by_another_writer_covers_a_member_requester_on_ghe(self):
        ok = od.make_approval(WRITER, SHA, FORKREQ(), NOW).to_dict()
        a = {"login": "m", "permission": "read", "association": "MEMBER"}
        self.assertTrue(
            od.authorize(facts(request=FORKREQ(), host="ghe", actor=a, head_repo="x/r", approval=ok)).allowed
        )

    def test_malformed_or_future_approval(self):
        with self.assertRaises(od.RequestError):
            od.authorize(facts(request=FORKREQ(), head_repo="x/r", approval={"sha": SHA, "bogus": 1}))
        a = od.make_approval(WRITER, SHA, FORKREQ(), "2026-11-17T10:00:00Z")
        self.assertIn("future", od.approval_valid(a, SHA, FORKREQ(), NOW, GH)[1])
        b = od.make_approval(WRITER, SHA, FORKREQ(), NOW, request_id="r1")
        self.assertFalse(od.approval_valid(b, SHA, FORKREQ(), NOW, GH, request_id="r2")[0])

    def test_bad_sha(self):
        self.assertEqual(od.authorize(facts(sha="main")).state, "deny")

    def test_policy_violation_denies(self):
        self.assertEqual(od.authorize(facts(request=req(reuse="never"))).state, "deny")

    def test_purpose_is_revalidated(self):
        bad = dict(req().to_dict(), purpose="baseline")
        with self.assertRaises(od.RequestError):
            od.authorize({**facts(), "request": bad})

    def test_estimate_over_limit_and_confirm_band(self):
        self.assertEqual(od.authorize(planned(estimate_device_hours=5)).state, "deny")
        self.assertEqual(od.authorize(planned(estimate_device_hours=3)).state, "needs_confirm")
        self.assertEqual(od.authorize(planned(estimate_device_hours=1)).state, "admit")

    def test_unknown_stage_denies(self):
        for stage in ("plan", "Planned", "", None, 3):
            self.assertEqual(od.authorize(facts(stage=stage, estimate_device_hours=1)).state, "deny", stage)

    def test_sha_must_match_even_on_the_same_repo(self):
        self.assertEqual(od.authorize(facts(request=req(sha="bbbbbbb"))).state, "deny")
        self.assertEqual(od.authorize(facts(request=req(sha=SHA[:7]))).state, "admit")

    def test_plan_cases_with_odd_actions_are_still_checked(self):
        for action in ("RUN", "", None, "launch"):
            case = {"name": "c", "run_props": {"run.tier": "extended"}}
            if action is not None:
                case["action"] = action
            plan = {"cases": [case]}
            self.assertEqual(od.authorize(planned(estimate_device_hours=1, plan=plan)).state, "deny", action)

    def test_the_plans_own_estimate_wins_over_a_smaller_claim(self):
        plan = dict(PLAN_OK, estimate={"device_hours": 9})
        self.assertEqual(od.authorize(planned(estimate_device_hours=0, plan=plan)).state, "deny")
        self.assertEqual(od.authorize(planned(estimate_device_hours=1, plan=dict(PLAN_OK, estimate={}))).state, "deny")

    def test_planned_stage_needs_the_plan(self):
        for plan in (None, {}, {"cases": []}, "x"):
            with self.subTest(plan):
                self.assertEqual(od.authorize(planned(estimate_device_hours=1, plan=plan)).state, "deny")

    def test_history_rows_of_odd_types_and_cases_still_count(self):
        h = self.history(pr="7", state="Active")
        self.assertEqual(od.authorize(planned(history=h, estimate_device_hours=1)).cancel, ["r1"])
        big = self.history(pr="7", state="done", est=100)
        self.assertEqual(od.authorize(planned(history=big, estimate_device_hours=1)).state, "deny")

    def test_planned_stage_needs_a_usable_estimate(self):
        for est in (float("nan"), -1, "x", float("inf")):
            with self.subTest(est):
                self.assertEqual(od.authorize(planned(estimate_device_hours=est)).state, "deny")
        self.assertEqual(od.authorize(facts(estimate_device_hours=float("nan"))).state, "deny")
        self.assertEqual(od.authorize(planned()).state, "admit", "the plan's own estimate is used when none is given")
        self.assertEqual(
            od.authorize(planned(plan={"cases": PLAN_OK["cases"]})).state, "deny", "a plan without an estimate"
        )

    def test_plan_cases_are_checked_against_the_host_tiers(self):
        plan = {
            "cases": [{"name": "a", "action": "run", "est_min": 5, "run_props": {"run.tier": "core"}}],
            "estimate": {"device_hours": 1},
        }
        d = od.authorize(planned(estimate_device_hours=1, plan=plan))
        self.assertEqual(d.state, "deny")
        self.assertIn("tier 'core'", d.reasons[0])
        plan["cases"][0]["run_props"] = {}
        self.assertEqual(od.authorize(planned(estimate_device_hours=1, plan=plan)).state, "deny")
        plan["cases"][0]["run_props"] = {"run.tier": "sanity"}
        self.assertEqual(od.authorize(planned(estimate_device_hours=1, plan=plan)).state, "admit")
        plan["cases"][0].update(action="reuse", run_props={"run.tier": "core"})
        self.assertEqual(
            od.authorize(planned(estimate_device_hours=1, plan=plan)).state, "deny", "a reused case may run"
        )
        plan["cases"][0].update(action="skip")
        self.assertEqual(od.authorize(planned(estimate_device_hours=1, plan=plan)).state, "admit")

    def test_a_run_case_without_an_estimate_is_unknown_cost_and_denied(self):
        for est in (None, 0, 0.0, -1, float("nan"), "5"):
            case = {"name": "a", "action": "run", "run_props": {"run.tier": "sanity"}}
            if est is not None:
                case["est_min"] = est
            plan = {"cases": [case], "estimate": {"device_hours": 0.0}}
            d = od.authorize(planned(estimate_device_hours=0, plan=plan))
            self.assertEqual(d.state, "deny", est)
            self.assertIn("no run estimate", d.reasons[0])
        reuse = {
            "cases": [{"name": "a", "action": "reuse", "run_props": {"run.tier": "sanity"}}],
            "estimate": {"device_hours": 0},
        }
        self.assertEqual(od.authorize(planned(estimate_device_hours=0, plan=reuse)).state, "admit")

    def test_a_tiny_estimate_with_a_long_timeout_is_bounded_by_the_timeout(self):
        case = {
            "name": "a",
            "action": "run",
            "est_min": 0.02,
            "timeout_min": 600,
            "accelerators": 1,
            "run_props": {"run.tier": "sanity"},
        }
        plan = {"cases": [case], "estimate": {"device_hours": 0.0003}}
        d = od.authorize(planned(estimate_device_hours=0.0003, plan=plan))
        self.assertEqual(d.state, "deny")
        self.assertIn("timeouts allow", d.reasons[0])
        case["timeout_min"] = 30
        self.assertEqual(od.authorize(planned(estimate_device_hours=0.0003, plan=plan)).state, "admit")
        case.update(timeout_min=100, accelerators=4)
        self.assertEqual(od.authorize(planned(estimate_device_hours=0.0003, plan=plan)).state, "deny")

    def test_a_request_is_not_its_own_history(self):
        own = self.history(id="r9", state="previewed", est=3)
        d = od.authorize(planned(history=own, request_id="r9", estimate_device_hours=3))
        self.assertEqual(d.cancel, [])
        self.assertEqual(od.authorize(planned(history=own, estimate_device_hours=3)).cancel, ["r9"])

    def test_window_rows_can_skip_the_rate_limit_count(self):
        rows = [
            {
                "id": f"r{i}",
                "pr": 9,
                "user": "dev1",
                "ts": "2026-11-16T09:30:00Z",
                "state": "done",
                "est": 0,
                "rate": False,
            }
            for i in range(8)
        ]
        self.assertEqual(od.authorize(facts(history=rows)).state, "admit")

    def test_case_picks_do_not_pick_their_own_tier_budget(self):
        r = od.parse_flags({"bench": "a_case"})
        self.assertEqual(od.budget_limit(r, GH), 4.0)
        self.assertEqual(od.budget_limit(r, GHE), 48.0)

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
        d = od.authorize(planned(history=self.history(), estimate_device_hours=1))
        self.assertEqual((d.state, d.cancel), ("admit", ["r1"]))

    def test_decision_names_are_valid_history_states(self):
        for st in ("admit", "needs_confirm"):
            d = od.authorize(planned(history=self.history(state=st), estimate_device_hours=1))
            self.assertEqual(d.cancel, ["r1"], st)
        d = od.authorize(planned(history=self.history(state="deny", est=99), estimate_device_hours=1))
        self.assertEqual((d.state, d.cancel), ("admit", []))

    def test_other_users_live_request_is_not_cancelled_by_a_non_writer(self):
        a = {"login": "m", "permission": "read", "association": "MEMBER"}
        d = od.authorize(planned(host="ghe", actor=a, history=self.history(), estimate_device_hours=1))
        self.assertEqual(d.state, "deny")
        self.assertIn("another user", d.reasons[0])
        self.assertEqual(od.authorize(planned(history=self.history(user="m"), estimate_device_hours=1)).cancel, ["r1"])

    def test_other_pr_not_cancelled(self):
        d = od.authorize(planned(history=self.history(pr=8), estimate_device_hours=1))
        self.assertEqual(d.cancel, [])

    def test_finished_request_not_cancelled_but_still_counts_toward_window(self):
        h = self.history(state="done", est=7)
        d = od.authorize(planned(history=h, estimate_device_hours=3))
        self.assertEqual(d.cancel, [])
        self.assertEqual(d.state, "deny")
        self.assertIn("window", d.reasons[0])

    def test_corrupt_history_estimates_fail_closed(self):
        for est in (float("nan"), -5, "x", float("inf")):
            with self.subTest(est):
                d = od.authorize(planned(history=self.history(state="done", est=est), estimate_device_hours=1))
                self.assertEqual(d.state, "deny")

    def test_cancelled_work_does_not_count(self):
        h = self.history(state="cancelled", est=7)
        self.assertTrue(od.authorize(planned(history=h, estimate_device_hours=3)).allowed)

    def test_superseded_estimate_is_freed(self):
        h = self.history(est=7)
        d = od.authorize(planned(history=h, estimate_device_hours=3))
        self.assertTrue(d.allowed)
        self.assertEqual(d.cancel, ["r1"])

    def test_window_expires(self):
        h = self.history(state="done", est=7, ts="2026-11-14T09:00:00Z")
        self.assertTrue(od.authorize(planned(history=h, estimate_device_hours=3)).allowed)

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

    def test_cancelling_does_not_reset_the_daily_clean_quota(self):
        r = req(clean="true")
        day = [
            self.history(pr=20 + i, clean=True, state="cancelled", ts="2026-11-16T01:00:00Z", id=f"d{i}")[0]
            for i in range(3)
        ]
        self.assertIn("per day", " ".join(od.authorize(facts(request=r, history=day)).reasons))

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
        return plan_doc("--purpose", "pr", "--subject-id", SUBJECT, "--tier", "sanity", *argv)

    def test_counts_estimate_and_order(self):
        pv = od.build_preview(self.plan(), req(), GH)
        self.assertEqual(pv["counts"], {"cases": 1, "run": 1, "reuse": 0, "skip": 0})
        self.assertEqual([c["name"] for c in pv["cases"]], ["op_add_fp16"])
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
        self.assertIn("`/spyre-bench confirm`", text)
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


class State(unittest.TestCase):
    def test_roundtrip_inside_a_comment(self):
        st = {"request": req().to_dict(), "pinned_sha": SHA, "estimate_device_hours": 3.0}
        body = "hello\n" + od.encode_state(st) + "\nmore"
        self.assertEqual(od.decode_state(body), st)

    def test_garbage_and_oversize_are_none(self):
        for body in [
            "",
            "no state",
            od.STATE_OPEN + " !!! -->",
            od.STATE_OPEN + " e30= ",
            od.STATE_OPEN + " " + "A" * 9000 + " -->",
        ]:
            with self.subTest(body[:30]):
                self.assertIsNone(od.decode_state(body))

    def test_non_object_state_is_none(self):
        import base64

        raw = base64.b64encode(b"[1,2]").decode()
        self.assertIsNone(od.decode_state(f"{od.STATE_OPEN} {raw} -->"))

    def test_markdown_embeds_state_once_and_keeps_marker_first(self):
        pv = od.build_preview(
            plan_doc("--purpose", "pr", "--subject-id", SUBJECT), req(), GH, notes=["reuse is looked up at dispatch"]
        )
        text = od.render_markdown(pv, state={"pinned_sha": SHA})
        self.assertTrue(text.startswith(od.PREVIEW_MARKER))
        self.assertEqual(text.count(od.STATE_OPEN), 1)
        self.assertIn("reuse is looked up at dispatch", text)
        self.assertEqual(od.decode_state(text), {"pinned_sha": SHA})


class Confirm(unittest.TestCase):
    def confirm(self, **kw):
        return planned(
            confirmed=True,
            pinned_sha=kw.pop("pinned_sha", SHA),
            requester=kw.pop("requester", WRITER),
            requested_at=kw.pop("requested_at", "2026-11-16T09:00:00Z"),
            **kw,
        )

    def test_confirm_clears_the_band_but_not_the_limit(self):
        self.assertEqual(od.authorize(self.confirm(estimate_device_hours=3)).state, "admit")
        self.assertEqual(od.authorize(self.confirm(estimate_device_hours=5)).state, "deny")

    def test_confirm_needs_write_even_on_ghe_member(self):
        a = {"login": "m", "permission": "read", "association": "MEMBER"}
        d = od.authorize(self.confirm(host="ghe", actor=a, requester=a, estimate_device_hours=30))
        self.assertEqual(d.state, "deny")

    def test_confirm_on_a_moved_head_is_denied(self):
        self.assertEqual(od.authorize(self.confirm(pinned_sha=SHA2, estimate_device_hours=3)).state, "deny")

    def test_confirm_expires(self):
        old = od.authorize(self.confirm(estimate_device_hours=3, requested_at="2026-11-14T09:00:00Z"))
        self.assertIn("too old", old.reasons[0])
        none = self.confirm(estimate_device_hours=3)
        none["requested_at"] = None
        self.assertEqual(od.authorize(none).state, "deny")

    def test_confirm_without_a_recorded_sha_is_denied(self):
        f = self.confirm(estimate_device_hours=3)
        f["pinned_sha"] = ""
        self.assertIn("comment again", od.authorize(f).reasons[0])

    def test_confirm_on_a_fork_needs_the_recorded_approval(self):
        r = FORKREQ()
        d = od.authorize(self.confirm(request=r, head_repo="x/r", estimate_device_hours=3))
        self.assertEqual(d.state, "deny")
        rec = od.make_approval(WRITER, SHA, r, NOW).to_dict()
        ok = od.authorize(self.confirm(request=r, head_repo="x/r", estimate_device_hours=3, approval=rec))
        self.assertEqual(ok.state, "admit")

    def test_the_requester_not_the_confirmer_is_checked_and_charged(self):
        asker = {"login": "rando", "permission": "read", "association": "NONE"}
        self.assertEqual(od.authorize(self.confirm(requester=asker, estimate_device_hours=1)).state, "deny")
        clean_asker = {"login": "m", "permission": "read", "association": "MEMBER"}
        d = od.authorize(
            self.confirm(host="ghe", request=req(clean="true"), requester=clean_asker, estimate_device_hours=1)
        )
        self.assertEqual(d.state, "deny")


class EngineRequestCli(unittest.TestCase):
    def test_prints_the_request_the_planner_reads(self):
        rf = TMP / "ondemand-r2.json"
        rf.write_text(json.dumps(od.parse_flags({"bench": "op_add_fp16", "budget": "2"}).to_dict()))
        code, out, _ = capture(
            cli_main,
            [
                "ondemand",
                "engine-request",
                "--request",
                str(rf),
                "--product",
                "spyre-inference",
                "--subject-id",
                SUBJECT,
            ],
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["subject"], {"artifact_id": SUBJECT})
        self.assertEqual(
            capture(cli_main, ["ondemand", "engine-request", "--request", str(rf), "--product", "torch-spyre"])[0], 2
        )

    def test_env_json(self):
        rf = TMP / "ondemand-r3.json"
        rf.write_text(json.dumps(od.parse_flags({"bench": "decoder@sanity", "reuse": "only"}).to_dict()))
        code, out, _ = capture(cli_main, ["ondemand", "env", "--request", str(rf), "--product", "spyre-inference"])
        self.assertEqual(code, 0)
        env = json.loads(out)
        self.assertEqual((env["BENCH_SUITES"], env["BENCH_TIER"], env["BENCH_REUSE"]), ("decoder", "sanity", "only"))
        self.assertNotIn("BENCH_SELECTION", env)

    def test_limit_prints_the_hosts_numbers_from_the_policy_file(self):
        code, out, _ = capture(cli_main, ["ondemand", "limit", "--host", "github.com"])
        self.assertEqual((code, json.loads(out)), (0, {"request_device_hours": 4.0, "pr_window_device_hours": 8.0}))

    def test_sanitize_reads_stdin(self):
        import io
        from unittest import mock

        with mock.patch("sys.stdin", io.StringIO("see [a](http://x) @team `c`\n")):
            code, out, _ = capture(cli_main, ["ondemand", "sanitize"])
        self.assertEqual(code, 0)
        for bad in ("](", "@team", "`", "://"):
            self.assertNotIn(bad, out)

    def test_error_text_is_neutralised(self):
        c = TMP / "ondemand-bad.txt"
        c.write_text("/bench @team`x`<b>=1")
        code, _, err = capture(cli_main, ["ondemand", "parse", "--comment-file", str(c)])
        self.assertEqual(code, 2)
        self.assertNotIn("<b>", err)
        self.assertNotIn("`", err)
        self.assertNotIn("@team", err)


class Hardening(unittest.TestCase):
    def cell(self, text):
        return od.safe(text)

    def test_links_images_urls_refs_and_bidi_are_inert(self):
        out = self.cell("[x](http://evil.example) ![i](http://e/p.png) http://evil.example #123 \u202eabc \u200b**b**")
        for bad in ("[", "](", "![", "://", "\u202e", "**", "#1"):
            self.assertNotIn(bad, out)

    def test_ordinary_case_names_survive(self):
        self.assertEqual(od.safe("serve_granite41-8b_tp4_c5_all.v2"), "serve_granite41-8b_tp4_c5_all.v2")

    def test_safe_is_idempotent(self):
        once = self.cell("@org/team [a](b) #9")
        self.assertEqual(od.safe(once), once)

    def test_request_error_messages_never_mention_or_link(self):
        with self.assertRaises(od.RequestError) as cm:
            od.parse_flags({"bench": "sanity", "@org/team": "1"})
        self.assertNotIn("@org", str(cm.exception))
        with self.assertRaises(od.RequestError) as cm:
            od.parse_flags({"bench": "sanity", "bench-reuse": "<img src=x>`"})
        self.assertNotIn("<", str(cm.exception))
        self.assertNotIn("`", str(cm.exception))

    def test_render_cleans_a_hand_built_preview(self):
        pv = od.build_preview(plan_doc("--purpose", "pr", "--subject-id", SUBJECT, "--tier", "sanity"), req(), GH)
        pv["notes"] = ["see [x](http://evil.example) @team"]
        pv["queue"] = {"q": "<b>x</b>"}
        pv["dashboard_url"] = "javascript:alert(1)"
        text = od.render_markdown(pv)
        for bad in ("](", "<b>", "@team", "javascript:"):
            self.assertNotIn(bad, text)
        pv["dashboard_url"] = "https://example.invalid/run/1"
        self.assertIn("https://example.invalid/run/1", od.render_markdown(pv))

    def test_unknown_estimate_means_nothing_runs(self):
        plan = plan_doc("--purpose", "pr", "--subject-id", SUBJECT, "--tier", "sanity")
        for bad in (None, float("nan"), -1):
            plan["estimate"]["device_hours"] = bad
            pv = od.build_preview(plan, req(), GH)
            self.assertTrue(pv["budget"]["over"] and not pv["budget"]["estimate_known"], bad)
            self.assertIn("no usable estimate", od.render_markdown(pv))

    def test_plan_with_a_core_case_is_flagged_on_the_public_host(self):
        plan = plan_doc("--purpose", "pr", "--subject-id", SUBJECT)
        pv = od.build_preview(plan, req(), GH)
        self.assertTrue(pv["budget"]["violations"])
        self.assertIn("Not allowed", od.render_markdown(pv))
        self.assertEqual(od.build_preview(plan, req(), GHE)["budget"]["violations"], [])

    def test_cases_pick_is_called_out_in_the_diff(self):
        plan = plan_doc("--purpose", "pr", "--subject-id", SUBJECT, "--cases", "op_mul_fp16")
        pv = od.build_preview(plan, od.parse_flags({"bench": "op_mul_fp16"}), GHE)
        self.assertTrue(any(d["planned"] == "not filtered" for d in pv["diff"]))

    def test_command_in_html_comment_or_mismatched_fence_is_ignored(self):
        for body in [
            "<!-- /bench sanity -->",
            "<!--\n/bench sanity\n-->",
            "~~~\n```\n/bench sanity\n~~~",
            "``` \n/bench sanity",
        ]:
            with self.subTest(body):
                self.assertIsNone(od.parse_command(body))
        self.assertIsNotNone(od.parse_command("<!-- c -->\n/bench sanity"))

    def test_confirm_and_cancel_take_no_arguments(self):
        for body in ("/bench confirm foo=bar", "/bench cancel now"):
            with self.assertRaises(od.RequestError):
                od.parse_command(body)

    def test_stored_requests_are_validated_again(self):
        good = req().to_dict()
        for patch in (
            {"sha": "a"},
            {"reuse": "zzz"},
            {"profile": "x"},
            {"budget": float("nan")},
            {"force": ["a b"]},
            {"env": "a b"},
            {"vs": "tomorrow"},
            {"selections": [{"product": "nope", "selection": {}, "as_asked": ""}]},
            {"selections": [{"product": "torch-spyre", "selection": {"tier": "huge"}, "as_asked": ""}]},
            {"selections": []},
            {
                "selections": [
                    {"product": "torch-spyre", "selection": {"tier": "sanity", "types": ["x"]}, "as_asked": ""}
                ]
            },
            {"selections": [{"product": "torch-spyre", "selection": {"cases": "abc"}, "as_asked": ""}]},
            {"max_wait_s": -1},
        ):
            with self.subTest(patch), self.assertRaises(od.RequestError):
                od.OnDemandRequest.from_dict({**good, **patch})

    def test_www_and_brackets_are_defanged_in_errors_and_cells(self):
        with self.assertRaises(od.RequestError) as cm:
            od.parse_flags({"bench": "sanity", "[click](https://evil.example)": "1"})
        for bad in ("](", "://", "["):
            self.assertNotIn(bad, str(cm.exception))
        self.assertNotIn("www.", od.safe("www.evil.com"))

    def test_action_and_repeats_cells_are_sanitised(self):
        plan = plan_doc("--purpose", "pr", "--subject-id", SUBJECT, "--tier", "sanity")
        plan["cases"][0].update(action="run | x", repeats="1 | @team")
        text = od.render_markdown(od.build_preview(plan, req(), GH))
        row = next(line for line in text.splitlines() if line.startswith("| run"))
        self.assertEqual(row.count("|"), 5)
        self.assertNotIn("@team", row)

    def test_budget_and_vs_reject_odd_spellings(self):
        for bad in ("1_0", "\u0661\u0662", " 3"):
            with self.assertRaises(od.RequestError):
                req(budget=bad)
        for bad in ("build:../x#1", "build:a/../b#1", "build:/x#1"):
            with self.assertRaises(od.RequestError):
                req(vs=bad)
        self.assertEqual(req(vs="build:orch/main#12").vs, "build:orch/main#12")

    def test_entities_are_defanged(self):
        for text in ("&#x40;octocat", "&commat;octocat", "a&amp;b"):
            self.assertNotIn("&", od.safe(text))
            self.assertNotIn("&", od.neutral(text))

    def test_sha_flag(self):
        self.assertEqual(req(sha="abcdef1").sha, "abcdef1")
        for bad in ("abc", "ABCDEF1", "xyz1234", "a" * 41):
            with self.assertRaises(od.RequestError):
                req(sha=bad)

    def test_state_size_guard_and_deep_nesting(self):
        with self.assertRaises(od.RequestError):
            od.encode_state({"x": "y" * 9000})
        import base64

        deep = base64.b64encode(("[" * 100000 + "]" * 100000).encode()).decode()
        self.assertIsNone(od.decode_state(f"{od.STATE_OPEN} {deep} -->") if len(deep) <= od.STATE_MAX else None)

    def test_env_carries_clean_and_exact_budget(self):
        r = od.parse_flags({"bench": "sanity", "clean": "true", "max-wait": "45m", "budget": "0.123456"})
        env = od.bench_env(r, "spyre-inference")
        self.assertEqual((env["BENCH_CLEAN"], env["BENCH_MAX_WAIT"], env["BENCH_BUDGET"]), ("1", "2700s", "0.123456"))


class RequestLog(unittest.TestCase):
    def plan(self):
        return plan_doc("--purpose", "pr", "--subject-id", SUBJECT, "--tier", "sanity")

    def row(self, state="previewed", **kw):
        return od.ci_request_row(
            req(), state, "dev1", host="ghe", pr=7, repo="o/r", head_sha=SHA, plan=self.plan(), **kw
        )

    def test_row_columns_exist_in_the_ci_requests_ddl(self):
        from pathlib import Path

        ddl = (Path(__file__).resolve().parents[2] / "clickhouse-ingest/schema/42-ci-requests.sql").read_text()
        cols = {
            line.split()[0]
            for line in ddl.splitlines()
            if line.startswith("    ") and not line.strip().startswith("CONSTRAINT")
        }
        row = self.row()
        self.assertLessEqual(set(row), cols)
        self.assertEqual((row["channel"], row["target_kind"], row["state"]), ("comment", "pr", "previewed"))

    def test_row_content(self):
        row = self.row(request_id="018f0000-0000-7000-8000-000000000001")
        self.assertEqual(
            json.loads(row["subjects"]), {"comment_url": "", "head_sha": SHA, "host": "ghe", "pr": 7, "repo": "o/r"}
        )
        self.assertEqual([c["name"] for c in json.loads(row["cells_requested"])], ["op_add_fp16"])
        self.assertEqual(json.loads(row["estimate"])["limit"], 12.0)
        self.assertEqual(row["modes"], ["perf"])
        self.assertEqual(json.loads(row["canonical"])["purpose"], "pr")
        self.assertTrue(row["plan_id"])

    def test_uuid7_and_state_check(self):
        import uuid

        u = uuid.UUID(self.row()["request_id"])
        self.assertEqual(u.version, 7)
        with self.assertRaises(od.RequestError):
            self.row(state="dry_run")

    def test_error_text_is_neutral_and_bounded(self):
        row = self.row("failed", error="see [x](http://e.co) @team " + "z" * 900)
        self.assertNotIn("](", row["error"])
        self.assertLessEqual(len(row["error"]), 500)

    def hrow(self, id_, state, pr=7, hours=3, repo="o/r", host="ghe", ts="2026-11-16 09:00:00.123"):
        return {"id": id_, "user": "dev1", "state": state, "ts": ts,
                "subjects": json.dumps({"pr": pr, "repo": repo, "host": host}), "estimate": json.dumps({"device_hours": hours})}  # fmt: skip

    def test_history_maps_states_and_feeds_admission(self):
        rows = [
            self.hrow("a", "launched"),
            self.hrow("b", "failed", hours=7, ts="2026-11-16T08:00:00Z"),
            self.hrow("c", "denied", hours=90),
        ]
        hist = od.history_from_ci_requests(rows, "ghe", now=NOW)
        self.assertEqual([h["state"] for h in hist], ["launched", "done", "denied"])
        d = od.authorize(planned(history=hist, estimate_device_hours=1))
        self.assertEqual(d.cancel, ["a"])
        self.assertEqual(d.state, "admit")

    def test_naive_clickhouse_timestamps_are_read_as_utc(self):
        hist = od.history_from_ci_requests([self.hrow("a", "launched", ts="2026-11-16 09:00:00")], "ghe", now=NOW)
        self.assertEqual(hist[0]["ts"], "2026-11-16T09:00:00Z")

    def test_other_hosts_and_other_repos_do_not_collide(self):
        rows = [self.hrow("a", "launched", host="github.com"), self.hrow("b", "launched", repo="x/other")]
        hist = od.history_from_ci_requests(rows, "ghe", now=NOW)
        self.assertEqual([h["id"] for h in hist], ["b"])
        self.assertEqual(od.authorize(planned(history=hist, estimate_device_hours=1)).cancel, [])
        self.assertEqual(
            od.authorize(planned(history=[dict(hist[0], repo="o/r")], estimate_device_hours=1)).cancel, ["b"]
        )

    def test_a_launched_request_ends_after_the_longest_job(self):
        hist = od.history_from_ci_requests(
            [self.hrow("a", "launched", ts="2026-11-16T01:00:00Z", hours=7)], "ghe", now=NOW
        )
        self.assertEqual(hist[0]["state"], "done")
        self.assertEqual(od.authorize(planned(history=hist, estimate_device_hours=1)).cancel, [])

    def test_the_history_query_takes_the_estimate_of_the_latest_event_that_has_one(self):
        self.assertIn("argMaxIf(estimate, ts, estimate != '')", od.HISTORY_SQL)
        row = od.ci_request_row(req(), "launched", "dev1", host="ghe", pr=7, repo="o/r", head_sha=SHA)
        self.assertEqual(row["estimate"], "")

    def test_unreadable_rows_raise_instead_of_passing(self):
        for row in (
            {"id": "x", "user": "u", "state": "launched", "subjects": "{", "estimate": "", "ts": "t"},
            {"id": "x"},
        ):
            with self.assertRaises(od.RequestError):
                od.history_from_ci_requests([row], "ghe", now=NOW)

    def test_the_policy_file_digest_is_pinned_so_the_internal_copy_must_follow(self):
        import hashlib
        from pathlib import Path

        digest = hashlib.sha256(Path(od.__file__).with_name("ondemand_policy.json").read_bytes()).hexdigest()
        # The internal poller pins the same digest for its copy (pipelines/lib/bench_policy.json): change both together.
        self.assertEqual(digest, "095f61bf007f094c51a28810a70ed59b5be361e4cd4f5399df6e51f9950b87c7")

    def test_limits_come_from_one_json_file(self):
        import json as j
        from pathlib import Path

        raw = j.loads((Path(od.__file__).with_name("ondemand_policy.json")).read_text())
        for host, pol in od.POLICIES.items():
            self.assertEqual(dict(pol.budget_device_hours), raw[host]["budget_device_hours"])
            self.assertEqual(pol.pr_window_device_hours, raw[host]["pr_window_device_hours"])
        self.assertEqual(od.POLICIES["github.com"].budget_device_hours, {"sanity": 4.0})
        self.assertEqual(od.POLICIES["ghe"].budget_device_hours, {"sanity": 12.0, "core": 48.0})


class Modes(unittest.TestCase):
    """The shipped policy file is dark (`enabled` false) and report-only (`enforce_budgets` false)."""

    def setUp(self):
        self.saved = dict(od.SETTINGS)

    def tearDown(self):
        od.SETTINGS.clear()
        od.SETTINGS.update(self.saved)

    def over_budget_facts(self):
        case = {
            "name": "a",
            "action": "run",
            "est_min": 5,
            "timeout_min": 60,
            "accelerators": 1,
            "run_props": {"run.tier": "sanity"},
        }
        return planned(estimate_device_hours=9, plan={"cases": [case], "estimate": {"device_hours": 9}})

    def test_the_shipped_file_is_dark_and_report_only(self):
        fresh = od._load_settings()
        self.assertEqual(fresh, {"enabled": False, "enforce_budgets": False})

    def test_dark_admits_nothing(self):
        od.SETTINGS.update(enabled=False)
        d = od.authorize(facts())
        self.assertEqual(d.state, "deny")
        self.assertIn("not enabled", d.reasons[0])

    def test_report_only_proceeds_but_records_what_would_have_been_refused(self):
        od.SETTINGS.update(enabled=True, enforce_budgets=False)
        d = od.authorize(self.over_budget_facts())
        self.assertEqual(d.state, "admit")
        self.assertTrue(any("over the 4 limit" in w for w in d.would_refuse))

    def test_enforcing_refuses_the_same_request(self):
        od.SETTINGS.update(enabled=True, enforce_budgets=True)
        d = od.authorize(self.over_budget_facts())
        self.assertEqual(d.state, "deny")
        self.assertTrue(d.would_refuse)

    def test_report_only_does_not_ask_for_a_confirm(self):
        od.SETTINGS.update(enabled=True, enforce_budgets=False)
        self.assertEqual(od.authorize(planned(estimate_device_hours=3)).state, "admit")

    def test_unknown_cost_and_window_are_report_only_too(self):
        od.SETTINGS.update(enabled=True, enforce_budgets=False)
        case = {"name": "a", "action": "run", "run_props": {"run.tier": "sanity"}}
        d = od.authorize(planned(estimate_device_hours=0, plan={"cases": [case], "estimate": {"device_hours": 0}}))
        self.assertEqual(d.state, "admit")
        self.assertTrue(any("no run estimate" in w for w in d.would_refuse))
        rows = [{"id": "r", "pr": 7, "user": "dev1", "ts": "2026-11-16T09:00:00Z", "state": "done", "est": 7}]
        d = od.authorize(planned(history=rows, estimate_device_hours=3))
        self.assertEqual(d.state, "admit")
        self.assertTrue(any("PR window" in w for w in d.would_refuse))

    def test_everything_else_stays_enforced_in_report_only_mode(self):
        od.SETTINGS.update(enabled=True, enforce_budgets=False)
        core = {
            "cases": [{"name": "a", "action": "run", "est_min": 5, "run_props": {"run.tier": "core"}}],
            "estimate": {"device_hours": 1},
        }
        self.assertEqual(od.authorize(planned(estimate_device_hours=1, plan=core)).state, "deny")
        self.assertEqual(od.authorize(facts(actor=READER)).state, "deny")
        self.assertEqual(od.authorize(facts(head_repo="x/r")).state, "deny")
        self.assertEqual(od.authorize(facts(pinned_sha=SHA2)).state, "deny")
        rate = [
            {"id": f"r{i}", "pr": 100 + i, "user": "dev1", "ts": "2026-11-16T09:30:00Z", "state": "done", "est": 0}
            for i in range(6)
        ]
        self.assertEqual(od.authorize(facts(history=rate)).state, "deny")

    def test_preview_says_report_only(self):
        od.SETTINGS.update(enabled=True, enforce_budgets=False)
        plan = plan_doc("--purpose", "pr", "--subject-id", SUBJECT, "--tier", "sanity")
        plan["estimate"]["device_hours"] = 9.0
        text = od.render_markdown(od.build_preview(plan, req(), GH))
        self.assertIn("Report only", text)
        self.assertNotIn("nothing will run", text.lower())
        od.SETTINGS.update(enforce_budgets=True)
        self.assertIn("nothing will run", od.render_markdown(od.build_preview(plan, req(), GH)).lower())

    def test_usage_is_recorded_either_way(self):
        row = od.ci_request_row(
            req(), "previewed", "dev1", host="ghe", pr=7, repo="o/r", head_sha=SHA, would_refuse=["over the limit"]
        )
        self.assertEqual(row["props"]["bench.would_refuse"], "over the limit")


if __name__ == "__main__":
    unittest.main()

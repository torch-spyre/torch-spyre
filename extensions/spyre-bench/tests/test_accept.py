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

"""`accept` (propose a run as the new anchor of its series) and the anchored verdict beside the rolling one in `verdict`.
python -m pytest -q -p no:cacheprovider tests/test_accept.py
"""

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from spyre_bench import accept as accept_cmd
from spyre_bench import engine
from spyre_bench import store as stores
from spyre_bench import verdict as verdict_cmd
from test_run import RUN, FakeClient, capture, clean_env
from test_run import verdict_row as base_verdict_row

PROPS = {"dtype": "float16", "batch": "4"}


def case_row(name, component="demo", props=PROPS):
    bid = engine.benchmark_id(component, name, [], props, sorted(props))
    return {"benchmark_key": bid, "component": component, "name": name, "tags": [], "props": dict(props)}


B1, B2 = case_row("op_add")["benchmark_key"], case_row("op_b")["benchmark_key"]
CASES = [case_row("op_add"), case_row("op_b")]


def verdict_row(**kw):
    return base_verdict_row(**{"benchmark_id": B1, **kw})


def mem(**kw):
    return stores.MemoryStore(**{"cases": list(CASES), **kw})


WHEN = datetime(2026, 11, 10, 2, 0, tzinfo=timezone.utc)
TMP = Path(tempfile.mkdtemp(prefix="spyre-bench-accept-"))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def series(bid=B1, name="op_add", metric="kernel_mean_ms", samples=(12.0, 12.1, 11.9), **kw):
    row = {
        "run_key": RUN, "benchmark_key": bid, "name": name, "component": "demo", "backend": "spyre", "arch": "x86_64", "source_job": "nightly",
        "metric": metric, "samples": list(samples), "eligible": True, "off_purpose": False, "suspect": False, "run_ts": WHEN,
    }  # fmt: skip
    return {**row, **kw}


def anchor_row(bid=B2, metric="kernel_mean_ms", samples=(4.0, 4.1), version="3", **kw):
    row = {
        "benchmark_key": bid, "component": "demo", "arch": "x86_64", "backend": "spyre", "source_job": "nightly",
        "metric": metric, "samples": list(samples), "anchor_from": datetime(2026, 10, 1, tzinfo=timezone.utc), "version": version,
    }  # fmt: skip
    return {**row, **kw}


def anchored(**kw):
    row = {
        "run_key": RUN, "benchmark_key": B1, "name": "op_add", "backend": "spyre", "arch": "x86_64", "source_job": "nightly",
        "metric": "kernel_mean_ms", "status": "regressed",
        "reason": "anchor_regressed", "version": "3", "anchor_value": 10.0, "current_value": 12.0, "delta_pct": 20.0,
        "worse_pct": 20.0, "floor_pct": 3.0, "since_ts": datetime(2026, 11, 1, 2, tzinfo=timezone.utc), "n_exceeding_runs": 10,
        "days_exceeding": 9,
    }  # fmt: skip
    return {**row, **kw}


class AcceptTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.sink = accept_cmd.MemorySink()

    def call(self, *argv, series_rows=None, anchors=(), anchored_rows=None):
        st = mem(
            series={RUN: [series()] if series_rows is None else series_rows},
            anchors=list(anchors),
            anchored={RUN: anchored_rows or []},
        )
        return capture(accept_cmd.main, ["--run-id", RUN, "--accepted-by", "amy", *argv], st=st, sink=self.sink)

    def test_a_run_becomes_one_request_per_scope_with_its_reason_and_time(self):
        code, out, err = self.call("--reason", "new normal after the driver update", "--json")
        self.assertEqual(code, 0, err)
        (req,) = self.sink.requests
        self.assertEqual(
            (req["role"], req["origin"], req["class"], req["accepted_by"], req["reason"]),
            ("anchor", "generated", "perf", "amy", "new normal after the driver update"),
        )
        self.assertEqual(
            req["scope"], {"component": "demo", "arch": "x86_64", "backend": "spyre", "source_job": "nightly"}
        )
        self.assertEqual(
            req["anchor"], {"role": "anchor", "run_id": RUN, "from_run_id": RUN, "as_of": "2026-11-10T02:00:00Z"}
        )
        self.assertRegex(req["key"], r"^anchor-demo-x86-64-spyre-nightly-[0-9a-f]{8}$")
        (v,) = req["values"]
        self.assertEqual(
            (v["benchmark_id"], v["metric"], v["samples"], v["carried"]),
            (B1, "kernel_mean_ms", [12.0, 12.1, 11.9], False),
        )
        self.assertEqual(json.loads(out)["requests"][0]["result"]["state"], "recorded")

    def test_the_rest_of_the_scopes_anchor_is_carried_with_its_own_time(self):
        self.call("--reason", "r", series_rows=[series()], anchors=[anchor_row(), anchor_row(bid=B1, version="3")])
        (req,) = self.sink.requests
        carried = [v for v in req["values"] if v["carried"]]
        self.assertEqual(
            [(v["benchmark_id"], v["samples"], v["as_of"]) for v in carried], [(B2, [4.0, 4.1], "2026-10-01T00:00:00Z")]
        )
        self.assertEqual([v["benchmark_id"] for v in req["values"] if not v["carried"]], [B1])

    def test_an_anchor_of_another_scope_is_not_carried(self):
        self.call("--reason", "r", anchors=[anchor_row(backend="cpu"), anchor_row(source_job="weekly")])
        self.assertEqual([v for v in self.sink.requests[0]["values"] if v["carried"]], [])

    def test_the_replaced_version_is_recorded_per_value(self):
        self.call("--reason", "r", anchored_rows=[anchored()])
        self.assertEqual(self.sink.requests[0]["values"][0]["replaces"], "3")

    def test_a_suspect_or_off_purpose_run_is_refused_and_nothing_is_proposed(self):
        for kw, why in (
            ({"suspect": True}, "suspect run"),
            ({"eligible": False, "off_purpose": True}, "off-purpose run"),
        ):
            self.sink.requests.clear()
            code, out, err = self.call("--reason", "r", series_rows=[series(**{"eligible": False, **kw})])
            self.assertEqual(code, 1)
            self.assertIn(why, err)
            self.assertEqual(self.sink.requests, [])

    def test_a_reason_is_required(self):
        code, out, err = self.call()
        self.assertEqual(code, 2)
        self.assertIn("--reason", err)

    def test_filters_narrow_the_run_and_an_empty_match_is_an_error(self):
        rows = [series(), series(bid=B2, name="op_b", metric="total_duration_ms")]
        self.call("--reason", "r", "--benchmark", B2, series_rows=rows)
        self.assertEqual([v["benchmark_id"] for v in self.sink.requests[0]["values"]], [B2])
        code, out, err = self.call("--reason", "r", "--metric", "nope", series_rows=rows)
        self.assertEqual(code, 2)
        self.assertIn("no series", err)

    def test_series_of_several_scopes_make_one_request_each(self):
        rows = [series(), series(bid=B2, name="op_b", source_job="weekly")]
        self.call("--reason", "r", series_rows=rows)
        self.assertEqual(sorted(r["scope"]["source_job"] for r in self.sink.requests), ["nightly", "weekly"])
        self.assertEqual(len({r["key"] for r in self.sink.requests}), 2)

    def test_a_dry_run_writes_nothing_and_the_default_sink_writes_a_file(self):
        out = TMP / "files"
        st = mem(series={RUN: [series()]}, anchored={RUN: []})
        code, text, err = capture(
            accept_cmd.main, ["--run-id", RUN, "--reason", "r", "--dry-run", "--out", str(out)], st=st
        )
        self.assertEqual(code, 0, err)
        self.assertFalse(out.exists())
        self.assertIn("would propose 1", text)
        code, text, err = capture(
            accept_cmd.main, ["--run-id", RUN, "--reason", "r", "--out", str(out), "--json"], st=st
        )
        path = json.loads(text)["requests"][0]["result"]["path"]
        saved = json.loads(Path(path).read_text())
        self.assertEqual(saved["request_schema"], "bench.reference-request/1")
        code, text, err = capture(accept_cmd.main, ["--run-id", RUN, "--reason", "r", "--out", str(out)], st=st)
        self.assertEqual(code, 2)  # the same run and scope again never replaces the pending request
        self.assertIn("already proposed", err)
        self.assertEqual(json.loads(Path(path).read_text()), saved)

    def test_a_store_failure_is_exit_5(self):
        st = mem(fail="boom")
        code, out, err = capture(accept_cmd.main, ["--run-id", RUN, "--reason", "r"], st=st, sink=self.sink)
        self.assertEqual(code, 5)
        self.assertIn("boom", err)


class AcceptEdgeTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.sink = accept_cmd.MemorySink()

    def call(self, *argv, series_rows=None, anchors=(), anchored_rows=None):
        st = mem(series={RUN: series_rows or [series()]}, anchors=list(anchors), anchored={RUN: anchored_rows or []})
        return capture(accept_cmd.main, ["--run-id", RUN, "--accepted-by", "amy", *argv], st=st, sink=self.sink)

    def test_scopes_that_only_differ_in_punctuation_get_different_keys(self):
        keys = {
            accept_cmd.slug(*scope)
            for scope in (
                ("a_b", "x86_64", "spyre", "nightly"),
                ("a-b", "x86_64", "spyre", "nightly"),
                ("a", "x86_64", "spyre", ""),
                ("a", "x86_64", "spyre", "any"),
            )
        }
        self.assertEqual(len(keys), 4)

    def test_two_scopes_write_two_files(self):
        out = TMP / "two"
        rows = [series(component="a_b"), series(component="a-b", bid=B2)]
        st = mem(series={RUN: rows}, anchored={RUN: []})
        code, text, err = capture(accept_cmd.main, ["--run-id", RUN, "--reason", "r", "--out", str(out)], st=st)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(list(out.glob("reference-request-*.json"))), 2)

    def test_a_non_finite_or_empty_sample_is_refused(self):
        for bad in ([1.0, float("nan")], [float("inf")], []):
            code, out, err = self.call("--reason", "r", series_rows=[series(samples=bad)])
            self.assertEqual(code, 1, bad)
            self.assertIn("non-finite", err)
            self.assertEqual(self.sink.requests, [])

    def test_a_suspect_and_off_purpose_run_names_both(self):
        code, out, err = self.call(
            "--reason", "r", series_rows=[series(eligible=False, suspect=True, off_purpose=True)]
        )
        self.assertIn("off-purpose run and suspect run", err)

    def test_the_other_metric_of_an_accepted_benchmark_is_carried(self):
        self.call("--reason", "r", anchors=[anchor_row(bid=B1, metric="total_duration_ms")])
        carried = [v for v in self.sink.requests[0]["values"] if v["carried"]]
        self.assertEqual([(v["benchmark_id"], v["metric"]) for v in carried], [(B1, "total_duration_ms")])

    def test_replaces_is_read_per_series_not_per_benchmark(self):
        rows = [series(source_job="nightly"), series(source_job="weekly")]
        self.call(
            "--reason", "r", series_rows=rows,
            anchored_rows=[anchored(source_job="nightly", backend="spyre", arch="x86_64", version="7"), anchored(source_job="weekly", backend="spyre", arch="x86_64", version="9")],
        )  # fmt: skip
        got = {r["scope"]["source_job"]: r["values"][0]["replaces"] for r in self.sink.requests}
        self.assertEqual(got, {"nightly": "7", "weekly": "9"})

    def test_no_anchor_rows_replace_nothing(self):
        self.call("--reason", "r", anchored_rows=[anchored(status="no_anchor", version="")])
        self.assertIsNone(self.sink.requests[0]["values"][0]["replaces"])

    def test_carried_series_with_non_finite_samples_are_dropped_with_a_warning(self):
        self.call("--reason", "r", anchors=[anchor_row(samples=(1.0, float("nan")))])
        (req,) = self.sink.requests
        self.assertEqual([v for v in req["values"] if v["carried"]], [])
        self.assertIn("carried series dropped", req["warnings"][0])

    def test_a_file_collision_in_a_later_scope_writes_nothing(self):
        out = TMP / "preflight"
        shutil.rmtree(out, ignore_errors=True)
        rows = [series(source_job="nightly"), series(source_job="weekly")]
        st = mem(series={RUN: rows}, anchored={RUN: []})
        accept_cmd.main(["--run-id", RUN, "--reason", "r", "--out", str(out), "--json"], st=st)
        first = sorted(out.glob("reference-request-*.json"))
        for f in first[1:]:
            f.unlink()  # leave only the first scope's file
        code, text, err = capture(accept_cmd.main, ["--run-id", RUN, "--reason", "r", "--out", str(out)], st=st)
        self.assertEqual(code, 2)
        self.assertEqual(len(list(out.glob("reference-request-*.json"))), 1)
        self.assertIn("nothing written", err)

    def test_a_timezone_aware_time_is_converted_to_utc(self):
        from datetime import timedelta

        local = datetime(2026, 11, 10, 7, 0, tzinfo=timezone(timedelta(hours=5)))
        self.assertEqual(accept_cmd.iso(local), "2026-11-10T02:00:00Z")
        self.assertEqual(accept_cmd.iso(datetime(2026, 11, 10, 2, 0)), "2026-11-10T02:00:00Z")


class AcceptWindowTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.sink = accept_cmd.MemorySink()

    def history(self, n):
        key = (B1, "kernel_mean_ms", "spyre", "nightly")
        return [
            {
                "series": key,
                "run_key": f"r{i}",
                "run_ts": datetime(2026, 11, i + 1, tzinfo=timezone.utc),
                "samples": [100.0 + i],
            }
            for i in range(n)
        ]

    def call(self, *argv, hist=None):
        row = series(samples=(104.0,), run_key="r4", run_ts=datetime(2026, 11, 5, tzinfo=timezone.utc))
        st = mem(series={RUN: [row]}, anchored={RUN: []}, windows=self.history(5) if hist is None else hist)
        return capture(
            accept_cmd.main, ["--run-id", RUN, "--accepted-by", "amy", "--reason", "r", *argv], st=st, sink=self.sink
        )

    def test_a_single_sample_run_is_refused_with_a_hint(self):
        code, out, err = self.call()
        self.assertEqual(code, 1)
        self.assertIn("1 sample(s), the anchor needs 3 (use --window N)", err)
        self.assertEqual(self.sink.requests, [])

    def test_a_window_pools_the_newest_runs_up_to_the_accepted_one(self):
        code, out, err = self.call("--window", "3")
        self.assertEqual(code, 0, err)
        (v,) = self.sink.requests[0]["values"]
        self.assertEqual((v["samples"], v["window_run_ids"]), ([102.0, 103.0, 104.0], ["r2", "r3", "r4"]))

    def test_the_accepted_run_is_always_in_its_own_window(self):
        code, out, err = self.call("--window", "3", hist=self.history(4))  # the store's window lacks r4
        self.assertEqual(code, 0, err)
        (v,) = self.sink.requests[0]["values"]
        self.assertEqual(v["window_run_ids"], ["r2", "r3", "r4"])

    def test_a_window_is_capped(self):
        self.call("--window", "999")
        self.assertLessEqual(len(self.sink.requests[0]["values"][0]["window_run_ids"]), accept_cmd.MAX_WINDOW)


class AnchoredVerdictTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)

    def call(self, *argv, rows=None, anchored_rows=None):
        st = mem(verdicts={RUN: rows or [verdict_row()]}, anchored=anchored_rows)
        return capture(verdict_cmd.main, ["--run-id", RUN, *argv], st=st)

    def test_the_anchored_verdict_sits_beside_the_rolling_one(self):
        code, out, err = self.call("--json", anchored_rows={RUN: [anchored()]})
        doc = json.loads(out)
        (v,) = doc["verdicts"]
        self.assertEqual(v["status"], "pass")  # the rolling verdict absorbed the shift
        self.assertEqual(
            (
                v["anchored"]["status"],
                v["anchored"]["reason"],
                v["anchored"]["since_ts"],
                v["anchored"]["n_exceeding_runs"],
            ),
            ("regressed", "anchor_regressed", "2026-11-01T02:00:00Z", 10),
        )
        self.assertEqual(doc["summary"]["anchored"]["regressed"], 1)
        self.assertIn(
            "vs anchor v3: regressed, since 2026-11-01 (10 runs)", self.call(anchored_rows={RUN: [anchored()]})[1]
        )

    def test_a_metric_without_an_anchored_row_has_none(self):
        code, out, err = self.call("--json", anchored_rows={RUN: []})
        self.assertIsNone(json.loads(out)["verdicts"][0]["anchored"])
        self.assertEqual(json.loads(out)["summary"]["anchored"]["regressed"], 0)

    def test_a_store_without_the_view_omits_the_anchored_summary(self):
        code, out, err = self.call("--json", anchored_rows=None)
        doc = json.loads(out)
        self.assertNotIn("anchored", doc["summary"])
        self.assertIsNone(doc["verdicts"][0]["anchored"])

    def test_no_anchor_flag_and_quiet_and_unknown_status(self):
        code, out, err = self.call("--json", "--no-anchor", anchored_rows={RUN: [anchored()]})
        self.assertNotIn("anchored", json.loads(out)["summary"])
        code, out, err = self.call("--json", "--quiet", anchored_rows={RUN: [anchored()]})
        a = json.loads(out)["verdicts"][0]["anchored"]
        self.assertNotIn("worse_pct", a)
        self.assertEqual(a["status"], "regressed")
        code, out, err = self.call("--json", anchored_rows={RUN: [anchored(status="odd")]})
        self.assertEqual(code, 2)

    def test_several_runs_never_share_an_anchored_row(self):
        other = "8e1f7a52-1b8d-4a3e-9d7e-0a1b2c3d4e5f"
        st = mem(
            verdicts={RUN: [verdict_row(run_id=RUN)], other: [verdict_row(run_id=other)]},
            anchored={RUN: [anchored()], other: [anchored(run_key=other, status="improved", reason="anchor_improved")]},
        )
        rows = {r: [dict(v, run_key=r) for v in st.verdicts[r]] for r in (RUN, other)}
        st.verdicts = rows
        code, out, err = capture(verdict_cmd.main, ["--run-id", RUN, "--json"], st=st)
        self.assertEqual(json.loads(out)["verdicts"][0]["anchored"]["status"], "regressed")
        doc = verdict_cmd.build(
            rows[RUN] + rows[other],
            [RUN, other],
            None,
            "memory",
            False,
            [anchored(), anchored(run_key=other, status="improved")],
        )
        self.assertEqual(sorted(v["anchored"]["status"] for v in doc["verdicts"]), ["improved", "regressed"])

    def test_two_series_of_one_benchmark_each_get_their_own(self):
        rows = [verdict_row(source_job="nightly", run_key=RUN), verdict_row(source_job="weekly", run_key=RUN)]
        doc = verdict_cmd.build(
            rows,
            [RUN],
            None,
            "memory",
            False,
            [
                anchored(source_job="nightly", backend="spyre", arch="x86_64"),
                anchored(source_job="weekly", backend="spyre", arch="x86_64", status="improved"),
            ],
        )
        self.assertEqual(sorted(v["anchored"]["status"] for v in doc["verdicts"]), ["improved", "regressed"])

    def test_fail_on_anchored_needs_the_view(self):
        code, out, err = self.call("--fail-on-anchored", "regressed", anchored_rows=None)
        self.assertEqual(code, 2)
        self.assertIn("needs the anchored view", err)
        self.assertEqual(self.call("--fail-on-anchored", "regressed", "--no-anchor", anchored_rows={RUN: []})[0], 2)

    def test_fail_on_anchored_fails_when_a_metric_has_no_anchored_row(self):
        code, out, err = self.call("--fail-on-anchored", "regressed", anchored_rows={RUN: []})
        self.assertEqual(code, 2)
        self.assertIn("no anchored row", err)
        self.assertEqual(self.call(anchored_rows={RUN: []})[0], 0)  # without the gate an unmatched row is only a null

    def test_fail_on_anchored_gates_only_when_asked(self):
        rows = {RUN: [anchored()]}
        self.assertEqual(self.call(anchored_rows=rows)[0], 0)
        self.assertEqual(self.call("--fail-on-anchored", "regressed", anchored_rows=rows)[0], 1)
        self.assertEqual(self.call("--fail-on-anchored", "improved", anchored_rows=rows)[0], 0)
        self.assertEqual(self.call("--fail-on-anchored", "bogus", anchored_rows=rows)[0], 2)


class MissingObjectTests(unittest.TestCase):
    def test_only_the_named_object_missing_counts(self):
        view = "v_benchmark_anchored_verdicts"
        sql = f"SELECT * FROM db.{view} WHERE run_id = x"
        yes = (
            f"Code: 60. DB::Exception: Table db.{view} does not exist. (UNKNOWN_TABLE)",
            f"Code: 60. DB::Exception: Unknown table expression identifier '{view}' in scope {sql}",
            f"Table `db`.`{view}` doesn't exist",
        )
        no = (
            f"Code: 60. Table db.benchmark_baseline_purposes does not exist. (UNKNOWN_TABLE) while executing {sql}",
            f"Unknown table expression identifier 'v_bench_references' in scope {sql}",
            f"connection reset while running {sql}",
        )
        for text in yes:
            self.assertTrue(stores.missing_object(RuntimeError(text), view), text)
        for text in no:
            self.assertFalse(stores.missing_object(RuntimeError(text), view), text)


class ClickHouseAnchorTests(unittest.TestCase):
    def test_a_missing_anchored_view_is_none_not_an_error(self):
        class Boom(FakeClient):
            def query(self, sql, parameters=None):
                raise RuntimeError(
                    "Unknown table expression identifier 'v_benchmark_anchored_verdicts' (UNKNOWN_TABLE)"
                )

        st = stores.ClickHouseStore(client=Boom(rows=[], columns=[]), database="d")
        self.assertIsNone(st.anchored_rows(RUN))

    def test_another_failure_still_raises(self):
        class Boom(FakeClient):
            def query(self, sql, parameters=None):
                raise RuntimeError("connection reset")

        with self.assertRaises(stores.StoreError):
            stores.ClickHouseStore(client=Boom(rows=[], columns=[]), database="d").anchored_rows(RUN)

    def test_series_window_binds_the_series_and_orders_oldest_first(self):
        rows = [("r2", datetime(2026, 11, 3), [2.0]), ("r1", datetime(2026, 11, 2), [1.0])]
        client = FakeClient(rows=rows, columns=["run_key", "run_ts", "samples"])
        st = stores.ClickHouseStore(client=client, database="d")
        row = series(run_ts=datetime(2026, 11, 3))
        got = st.series_window(row, 5)
        self.assertEqual([g["run_key"] for g in got], ["r1", "r2"])
        ((sql, params),) = client.queries
        self.assertIn("v_benchmark_series_runs", sql)
        self.assertIn("(run_ts, run_id) <=", sql)
        self.assertNotIn("until", params)
        self.assertEqual(
            (params["b"], params["be"], params["j"], params["n"], params["run"]), (B1, "spyre", "nightly", 5, RUN)
        )

    def test_series_runs_and_anchor_rows_query_the_views_by_parameter(self):
        client = FakeClient(rows=[], columns=["benchmark_key"])
        st = stores.ClickHouseStore(client=client, database="d")
        st.series_runs(RUN)
        st.anchor_rows("demo", "x86_64", "spyre", "nightly")
        (q1, p1), (q2, p2) = client.queries
        self.assertIn("v_benchmark_series_runs", q1)
        self.assertEqual(p1, {"run": RUN})
        self.assertIn("v_benchmark_anchor", q2)
        self.assertEqual(p2, {"c": "demo", "a": "x86_64", "b": "spyre", "j": "nightly"})


if __name__ == "__main__":
    unittest.main()


class CaseBlockTests(unittest.TestCase):
    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.sink = accept_cmd.MemorySink()

    def call(self, *argv, st):
        return capture(accept_cmd.main, ["--run-id", RUN, "--reason", "r", *argv], st=st, sink=self.sink)

    def store(self, cases=None, **kw):
        return mem(series={RUN: [series()]}, anchored={RUN: []}, **({} if cases is None else {"cases": cases}), **kw)

    def test_every_value_carries_a_case_block_that_rebuilds_its_id(self):
        code, out, err = self.call(st=self.store(anchors=[anchor_row()]))
        self.assertEqual(code, 0, err)
        by_id = {v["benchmark_id"]: v for v in self.sink.requests[0]["values"]}
        self.assertEqual(set(by_id), {B1, B2})
        self.assertEqual(
            by_id[B1]["case"],
            {"component": "demo", "name": "op_add", "tags": [], "disc": PROPS, "disc_keys": ["batch", "dtype"]},
        )
        for bid, v in by_id.items():
            self.assertEqual(accept_cmd.rebuilds(v["case"]), bid)

    def test_a_case_that_is_not_registered_is_refused_and_nothing_is_proposed(self):
        code, out, err = self.call(st=self.store(cases=[case_row("op_b")]))
        self.assertEqual(code, 1)
        self.assertIn(B1, err)
        self.assertIn("no case data", err)
        self.assertEqual(self.sink.requests, [])

    def test_a_case_whose_data_derives_another_id_is_refused(self):
        wrong = {**case_row("op_add"), "props": {"dtype": "float32", "batch": "4"}}
        code, out, err = self.call(st=self.store(cases=[wrong]))
        self.assertEqual(code, 1)
        self.assertIn("derives", err)
        self.assertEqual(self.sink.requests, [])

    def test_a_carried_series_without_case_data_is_refused_too(self):
        code, out, err = self.call(st=self.store(cases=[case_row("op_add")], anchors=[anchor_row()]))
        self.assertEqual(code, 1)
        self.assertIn(B2, err)

    def test_the_run_dir_results_give_the_exact_disc_keys_and_profile(self):
        keys = ["dtype"]
        bid = engine.benchmark_id("demo", "op_add", [], PROPS, keys)
        d = TMP / "rd"
        (d / "c1").mkdir(parents=True, exist_ok=True)
        case = {"benchmark_id": bid, "component": "demo", "name": "op_add", "tags": [], "disc": PROPS,
                "disc_keys": keys, "id_profile": "generic", "props": {}}  # fmt: skip
        (d / "c1" / "result.json").write_text(json.dumps({"kind": "Result", "status": {"case": case}}))
        (d / "run.json").write_text(json.dumps({"stored_run_ids": [RUN], "cases": [{"result": "c1/result.json"}]}))
        st = mem(series={RUN: [series(bid=bid)]}, anchored={RUN: []}, cases=[])
        code, out, err = capture(accept_cmd.main, ["--run-dir", str(d), "--reason", "r"], st=st, sink=self.sink)
        self.assertEqual(code, 0, err)
        (v,) = self.sink.requests[0]["values"]
        self.assertEqual(v["case"]["disc_keys"], keys)
        self.assertEqual(v["case"]["id_profile"], "generic")
        self.assertNotIn("props", v["case"])


try:
    from spyre_bench import reference
except ImportError:  # the reference engine is a sibling branch
    reference = None


@unittest.skipIf(reference is None, "the reference engine is not on this branch")
class RoundTripTests(unittest.TestCase):
    """accept -> `reference create --from-request` -> validate."""

    def setUp(self):
        self._env = clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.dir = Path(tempfile.mkdtemp(dir=TMP))

    def propose(self):
        st = mem(series={RUN: [series()]}, anchored={RUN: []}, anchors=[anchor_row()])
        code, out, err = capture(
            accept_cmd.main, ["--run-id", RUN, "--reason", "r", "--out", str(self.dir / "req")], st=st
        )
        self.assertEqual(code, 0, err)
        (path,) = (self.dir / "req").glob("reference-request-*.json")
        return path

    def create(self, path, out):
        meta = self.dir / "meta.json"
        meta.write_text(json.dumps({
            "model": {"name": "demo"}, "dataset": {"name": "demo"}, "producer": "demo", "terms": "internal-generated",
            "comparability": {"caveats": "demo"}, "system": {"backend": "spyre"},
        }))  # fmt: skip
        st = stores.NullStore()
        st.registered = set  # every case counts as registered
        argv = ["create", "--from-request", str(path), "--out", str(out), "--meta", str(meta)]
        return capture(reference.main, argv, st=st)

    def test_the_request_becomes_a_valid_reference(self):
        out = self.dir / "ref"
        code, text, err = self.create(self.propose(), out)
        self.assertEqual(code, 0, err)
        doc = json.loads((out / "reference.json").read_text())
        self.assertEqual({v["case"]["benchmark_id"] for v in doc["values"]}, {B1, B2})
        self.assertEqual(doc["source_run"]["run_id"], RUN)
        code, text, err = capture(reference.main, ["validate", str(out)])
        self.assertEqual(code, 0, text + err)

    def test_a_tampered_case_block_is_refused(self):
        path = self.propose()
        req = json.loads(path.read_text())
        req["values"][0]["case"]["name"] = "something_else"
        path.write_text(json.dumps(req))
        code, text, err = self.create(path, self.dir / "ref2")
        self.assertNotEqual(code, 0)
        self.assertIn("derives", err)
        self.assertFalse((self.dir / "ref2").exists())

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


"""Tests for the sweep driver, its plan and the scaling fits: pure data, no device.

The plan checks read ``workloads.py`` with ``ast``, since importing it needs a built
``torch_spyre._C``.
"""

import ast
import importlib.util
import json
import os
import subprocess
import sys

import pytest

_TOOLS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "frontend_timing",
)
_PLAN = os.path.join(_TOOLS, "sweep_plan.json")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_TOOLS, filename))
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves its own module out of sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_sweep = _load("fts_run_sweep", "run_sweep.py")
scaling = _load("fts_scaling", "scaling.py")


def _write_plan(tmp_path, points):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"points": points}))
    return str(path)


class TestPlanLoading:
    def test_tier_selects_only_the_points_declaring_it(self, tmp_path):
        path = _write_plan(
            tmp_path,
            [
                {"workload": "mlp", "tiers": ["pr", "nightly"], "layers": 1},
                {"workload": "mlp", "tiers": ["nightly"], "layers": 2},
                {"workload": "mlp", "tiers": ["weekly"], "layers": 8},
            ],
        )
        assert len(run_sweep.load_plan(path, "pr")) == 1
        assert len(run_sweep.load_plan(path, "nightly")) == 2
        assert len(run_sweep.load_plan(path, "weekly")) == 1
        assert len(run_sweep.load_plan(path, None)) == 3

    def test_a_point_without_tiers_runs_in_every_tier(self, tmp_path):
        # Keeps a plan written before tiers existed behaving exactly as it did.
        path = _write_plan(tmp_path, [{"workload": "mlp", "layers": 2}])
        for tier in ("pr", "nightly", "weekly", None):
            assert len(run_sweep.load_plan(path, tier)) == 1

    def test_a_bare_list_is_still_a_plan(self, tmp_path):
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps([{"workload": "mlp", "layers": 2}]))
        (point,) = run_sweep.load_plan(str(path))
        assert point.params == {"layers": 2}

    def test_reserved_keys_never_reach_the_builder(self, tmp_path):
        path = _write_plan(
            tmp_path,
            [
                {
                    "workload": "mlp",
                    "tiers": ["pr"],
                    "env": {"SENCORES": "1"},
                    "comment": "why this point exists",
                    "layers": 2,
                }
            ],
        )
        (point,) = run_sweep.load_plan(path, "pr")
        # Anything left in params is passed as a builder keyword, so a leak here is a
        # TypeError in the child rather than a mistake anyone sees here.
        assert point.params == {"layers": 2}
        assert point.env == {"SENCORES": "1"}

    def test_a_missing_workload_is_rejected(self, tmp_path):
        path = _write_plan(tmp_path, [{"tiers": ["pr"], "layers": 2}])
        with pytest.raises(SystemExit):
            run_sweep.load_plan(path)


class TestArms:
    def test_an_arm_has_a_stable_label(self):
        point = run_sweep.Point(workload="mlp", env={"B": "2", "A": "1"})
        # Sorted, so the same arm produces the same label whatever order it was written.
        assert point.arm == "A=1,B=2"
        assert run_sweep.Point(workload="mlp").arm == ""

    def test_arms_of_one_point_get_different_record_names(self):
        params = {"S": 512}
        default = run_sweep.point_id("granite_layer", params, "")
        armed = run_sweep.point_id("granite_layer", params, "SENCORES=1")
        # Same filename would mean the second arm silently overwrites the first.
        assert default != armed
        assert "/" not in armed and "=" not in armed

    def test_the_child_gets_the_arm_environment(self, tmp_path):
        point = run_sweep.Point(workload="mlp", env={"SENCORES": "1"})
        env = run_sweep._child_env(str(tmp_path), "r.json", str(tmp_path), True, point)
        assert env["SENCORES"] == "1"
        assert env["TORCH_SPYRE_FRONTEND_ONLY"] == "1"
        assert env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] == "1"
        # One compile thread, or the per-kernel backend region runs in a pool worker
        # that does not share the recorder and the frontend absorbs it silently.
        assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "1"

    def test_an_arm_can_override_a_default_the_driver_set(self, tmp_path):
        point = run_sweep.Point(workload="mlp", env={"TORCH_SPYRE_FRONTEND_ONLY": "0"})
        env = run_sweep._child_env(str(tmp_path), "r.json", str(tmp_path), True, point)
        # The arm is applied last on purpose, so a point can measure the backend share.
        assert env["TORCH_SPYRE_FRONTEND_ONLY"] == "0"


class TestShippedPlan:
    """The plan in the tree has to agree with the builders in the tree."""

    @staticmethod
    def _builder_signatures():
        tree = ast.parse(open(os.path.join(_TOOLS, "workloads.py")).read())
        sigs = {
            node.name: {a.arg for a in node.args.kwonlyargs}
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("build_")
        }
        registry = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and getattr(node.target, "id", "") == "BUILDERS"
            ):
                for key, value in zip(node.value.keys, node.value.values):
                    registry[key.value] = value.id
        return registry, sigs

    @staticmethod
    def _points():
        return json.load(open(_PLAN))["points"]

    def test_every_point_names_a_registered_workload(self):
        registry, _ = self._builder_signatures()
        unknown = {p["workload"] for p in self._points()} - set(registry)
        assert not unknown, f"plan names workloads with no builder: {sorted(unknown)}"

    def test_every_point_parameter_is_a_builder_keyword(self):
        registry, sigs = self._builder_signatures()
        problems = []
        for point in self._points():
            accepted = sigs[registry[point["workload"]]]
            unknown = set(point) - run_sweep.RESERVED_PLAN_KEYS - accepted
            if unknown:
                problems.append(f"{point['workload']}: {sorted(unknown)}")
        assert not problems, f"plan params no builder accepts: {problems}"

    def test_every_point_declares_its_tiers(self):
        # An untiered point runs in *every* tier, so forgetting the key silently puts a
        # multi-minute compile in the per-PR lane.
        untiered = [p["workload"] for p in self._points() if not p.get("tiers")]
        assert not untiered, f"points with no tiers: {untiered}"

    def test_every_tier_has_points_and_pr_is_the_smallest(self):
        counts = {
            tier: len(run_sweep.load_plan(_PLAN, tier))
            for tier in ("pr", "nightly", "weekly")
        }
        assert all(counts.values()), f"an empty tier: {counts}"
        assert counts["pr"] < counts["nightly"] <= counts["weekly"], counts

    def test_every_registered_workload_is_swept(self):
        registry, _ = self._builder_signatures()
        swept = {p["workload"] for p in self._points()}
        missing = set(registry) - swept
        assert not missing, f"builders nothing sweeps: {sorted(missing)}"

    def test_depth_and_sequence_series_are_long_enough_to_fit(self):
        points = [p for p in self._points() if p["workload"] == "granite_layer"]
        plain = [p for p in points if not p.get("env")]
        depths = {p.get("layers", 1) for p in plain if p.get("S") == 512}
        seqs = {p["S"] for p in plain if p.get("layers", 1) == 1}
        # scaling.py refuses to quote an exponent below four points, and the 40-layer
        # Granite figure is an extrapolation off the depth series.
        assert len(depths) >= scaling.MIN_POINTS, sorted(depths)
        assert len(seqs) >= scaling.MIN_POINTS, sorted(seqs)


def _rows(tmp_path, points, schema="frontend-timing-rows/1"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"schema": schema, "meta": {}, "points": points}))
    return str(path)


def _point(workload, params, arm="", **metrics):
    return {
        "name": "p",
        "workload": workload,
        "params": params,
        "arm": arm,
        "samples": 1,
        "measurements": {k: [float(v)] for k, v in metrics.items()},
    }


class TestFits:
    def test_a_known_exponent_is_recovered(self):
        # y = 100 * x^1.5 exactly, so the fit has one right answer and R2 is 1.
        fit = scaling.loglog_fit([(x, 100.0 * x**1.5) for x in (1, 2, 4, 8)])
        assert fit.exponent == pytest.approx(1.5, abs=1e-9)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
        assert fit.points == 4
        assert fit.reliable

    def test_extrapolation_follows_the_fit(self):
        fit = scaling.loglog_fit([(x, 100.0 * x**1.5) for x in (1, 2, 4, 8)])
        assert fit.predict(40) == pytest.approx(100.0 * 40**1.5, rel=1e-9)

    def test_too_few_points_is_not_reliable(self):
        fit = scaling.loglog_fit([(x, float(x)) for x in (1, 2, 4)])
        # A perfect line through three points is still three points.
        assert fit.r_squared == pytest.approx(1.0)
        assert fit.points == 3
        assert not fit.reliable

    def test_a_poor_fit_is_not_reliable(self):
        fit = scaling.loglog_fit([(1, 1.0), (2, 100.0), (4, 2.0), (8, 300.0)])
        assert fit.r_squared < scaling.R_SQUARED_FLOOR
        assert not fit.reliable

    def test_non_positive_values_are_dropped_not_nudged(self):
        # A fudge to make log() defined would silently change the exponent reported.
        pairs = [(1, 0.0), (2, 4.0), (4, 16.0), (8, 64.0), (16, 256.0)]
        fit = scaling.loglog_fit(pairs)
        assert fit.points == 4
        assert fit.exponent == pytest.approx(2.0, abs=1e-9)

    def test_one_distinct_axis_value_has_no_slope(self):
        assert scaling.loglog_fit([(4, 1.0), (4, 2.0), (4, 3.0)]) is None

    def test_too_little_data_has_no_fit(self):
        assert scaling.loglog_fit([(4, 1.0)]) is None


class TestSeriesDiscovery:
    def test_a_series_is_found_along_the_moving_axis(self, tmp_path):
        points = [
            _point("granite_layer", {"S": 512, "layers": n}, frontend_ms=100.0 * n)
            for n in (1, 2, 4, 8)
        ]
        found = scaling.find_series(scaling.load_points([_rows(tmp_path, points)]))
        layers = [s for s in found if s.axis == "layers"]
        assert len(layers) == 1
        fit = scaling.fit_series(layers[0], "frontend_ms")
        assert fit.exponent == pytest.approx(1.0, abs=1e-9)
        assert "granite_layer[layers]" in layers[0].label

    def test_moving_two_axes_at_once_yields_no_series(self, tmp_path):
        # Otherwise the tool would happily fit a diagonal and call it an exponent.
        points = [
            _point("granite_layer", {"S": s, "layers": n}, frontend_ms=100.0)
            for s, n in ((128, 1), (512, 2), (1024, 4), (2048, 8))
        ]
        found = scaling.find_series(scaling.load_points([_rows(tmp_path, points)]))
        assert found == []

    def test_arms_are_separate_series(self, tmp_path):
        points = []
        for arm in ("", "SENCORES=1"):
            points += [
                _point("granite_layer", {"layers": n}, arm=arm, frontend_ms=100.0 * n)
                for n in (1, 2, 4, 8)
            ]
        found = scaling.find_series(scaling.load_points([_rows(tmp_path, points)]))
        # Two arms fitted together would average a control into its treatment.
        assert len(found) == 2
        assert {s.arm for s in found} == {"", "SENCORES=1"}

    def test_an_unknown_rows_schema_is_refused(self, tmp_path):
        path = _rows(tmp_path, [], schema="frontend-timing-rows/99")
        with pytest.raises(SystemExit):
            scaling.load_points([path])


class TestFitRendering:
    def _rendered(self, tmp_path, exponent, count):
        points = [
            _point("granite_layer", {"layers": n}, frontend_ms=100.0 * n**exponent)
            for n in [2**i for i in range(count)]
        ]
        series = scaling.find_series(scaling.load_points([_rows(tmp_path, points)]))
        return scaling.render(series, ["frontend_ms"], ("layers", 40.0))

    def test_a_reliable_fit_is_projected(self, tmp_path):
        table = self._rendered(tmp_path, 1.5, 4)
        assert "| ok |" in table
        assert "layers=40" in table
        assert "25298" in table

    def test_an_unreliable_fit_is_not_projected(self, tmp_path):
        table = self._rendered(tmp_path, 1.5, 3)
        assert "insufficient" in table
        # The dash is the point: no number is offered where none is earned.
        assert table.rstrip().endswith("| - |")


class TestDriverLoop:
    """run_point's launch sequence, with the child stubbed out."""

    @staticmethod
    def _fake_child(outcomes):
        """Record each launch and return the next outcome from ``outcomes``."""
        launches = []

        def fake(argv, env, timeout_s):
            launches.append({"argv": argv, "env": env, "timeout": timeout_s})
            code = outcomes[min(len(launches) - 1, len(outcomes) - 1)]
            if code == "timeout":
                raise subprocess.TimeoutExpired(argv, timeout_s)
            return subprocess.CompletedProcess(argv, code, "", "boom\n")

        return fake, launches

    def test_each_sample_plus_a_warmup_is_launched(self, tmp_path, monkeypatch):
        fake, launches = self._fake_child([0])
        monkeypatch.setattr(run_sweep, "_run_child", fake)
        failures = run_sweep.run_point(
            run_sweep.Point("mlp", {"layers": 2}), str(tmp_path), 3, True, 60
        )
        assert failures == []
        assert len(launches) == 4
        indexes = [
            launch["argv"][launch["argv"].index("--sample-index") + 1]
            for launch in launches
        ]
        assert indexes == ["0", "1", "2", "3"]

    def test_a_failed_warmup_skips_the_samples(self, tmp_path, monkeypatch):
        fake, launches = self._fake_child([1])
        monkeypatch.setattr(run_sweep, "_run_child", fake)
        failures = run_sweep.run_point(
            run_sweep.Point("mlp"), str(tmp_path), 3, True, 3600
        )
        # One launch, not four. At the default timeout, carrying on costs four hours
        # per hanging point and measures nothing either way.
        assert len(launches) == 1
        assert len(failures) == 1
        assert "warmup" in failures[0]

    def test_a_failed_sample_does_not_stop_the_others(self, tmp_path, monkeypatch):
        fake, launches = self._fake_child([0, 1, 0, 0])
        monkeypatch.setattr(run_sweep, "_run_child", fake)
        failures = run_sweep.run_point(
            run_sweep.Point("mlp"), str(tmp_path), 3, True, 60
        )
        assert len(launches) == 4
        assert len(failures) == 1

    def test_a_timeout_is_a_failure_not_a_crash(self, tmp_path, monkeypatch):
        fake, _ = self._fake_child(["timeout"])
        monkeypatch.setattr(run_sweep, "_run_child", fake)
        failures = run_sweep.run_point(
            run_sweep.Point("mlp"), str(tmp_path), 1, True, 42
        )
        assert len(failures) == 1
        assert "timed out after 42s" in failures[0]

    def test_the_arm_travels_on_argv(self, tmp_path, monkeypatch):
        fake, launches = self._fake_child([0])
        monkeypatch.setattr(run_sweep, "_run_child", fake)
        point = run_sweep.Point("mlp", env={"SENCORES": "1"})
        run_sweep.run_point(point, str(tmp_path), 1, True, 60)
        argv = launches[0]["argv"]
        assert argv[argv.index("--arm") + 1] == "SENCORES=1"

    def test_the_warmup_record_lands_where_the_summarizer_does_not_look(
        self, tmp_path, monkeypatch
    ):
        fake, launches = self._fake_child([0])
        monkeypatch.setattr(run_sweep, "_run_child", fake)
        run_sweep.run_point(run_sweep.Point("mlp"), str(tmp_path), 1, True, 60)
        destinations = [
            launch["env"]["TORCH_SPYRE_TIMING_OUT"] for launch in launches
        ]
        assert os.path.dirname(destinations[0]) == str(tmp_path / "warmup")
        assert os.path.dirname(destinations[1]) == str(tmp_path)


class TestMainGuards:
    """main's refusals, each of which used to be a silently wrong sweep."""

    def _plan(self, tmp_path):
        return _write_plan(tmp_path, [{"workload": "mlp", "tiers": ["pr"]}])

    def test_a_records_directory_that_already_holds_records_is_refused(
        self, tmp_path, monkeypatch
    ):
        out = tmp_path / "out"
        out.mkdir()
        (out / "mlp-run1.1234.json").write_text("{}")
        monkeypatch.setattr(run_sweep, "_run_child", self_fail)
        with pytest.raises(SystemExit) as caught:
            run_sweep.main(
                ["--workload", "mlp", "--out", str(out), "--samples", "1"]
            )
        # Records are never replaced, so a second sweep here is summarized with the
        # first -- two builds averaged into one point.
        assert "already holds" in str(caught.value)

    def test_append_allows_it_deliberately(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        (out / "mlp-run1.1234.json").write_text("{}")
        fake, launches = TestDriverLoop._fake_child([0])
        monkeypatch.setattr(run_sweep, "_run_child", fake)
        assert (
            run_sweep.main(
                ["--workload", "mlp", "--out", str(out), "--samples", "1", "--append"]
            )
            == 0
        )
        assert launches

    def test_param_with_a_plan_is_refused(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            run_sweep.main(["--plan", self._plan(tmp_path), "-p", "layers=2"])
        # Previously ignored in silence, so a mistyped single-point run swept the
        # whole plan instead.
        assert "--param applies to a single point" in str(caught.value)

    def test_env_with_a_plan_is_refused(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            run_sweep.main(["--plan", self._plan(tmp_path), "--env", "SENCORES=1"])
        assert "--env applies to a single point" in str(caught.value)

    def test_a_tier_with_no_points_is_refused(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            run_sweep.main(["--plan", self._plan(tmp_path), "--tier", "weekly"])
        assert "no plan points declare tier" in str(caught.value)

    def test_a_nonzero_exit_when_a_sample_failed(self, tmp_path, monkeypatch):
        fake, _ = TestDriverLoop._fake_child([1])
        monkeypatch.setattr(run_sweep, "_run_child", fake)
        out = tmp_path / "fresh"
        assert (
            run_sweep.main(
                ["--workload", "mlp", "--out", str(out), "--samples", "1"]
            )
            == 1
        )


def self_fail(argv, env, timeout_s):
    raise AssertionError("no child should be launched")


class TestEnvParsing:
    def test_values_are_not_coerced_to_int(self):
        # int() would make SENCORES=0007 into 7 and 1_000 into 1000, so the arm run
        # would not be the arm that was asked for.
        env = run_sweep.parse_env(["SENCORES=0007", "MAX=1_000", "FLAG=true"])
        assert env == {"SENCORES": "0007", "MAX": "1_000", "FLAG": "true"}

    def test_a_value_may_contain_equals(self):
        assert run_sweep.parse_env(["OPTS=a=b"]) == {"OPTS": "a=b"}

    def test_a_missing_value_is_refused(self):
        with pytest.raises(SystemExit):
            run_sweep.parse_env(["SENCORES"])

    def test_a_duplicate_point_across_rows_files_is_refused(self, tmp_path):
        points = [
            _point("granite_layer", {"layers": n}, frontend_ms=100.0 * n)
            for n in (1, 2, 4, 8)
        ]
        first = _rows(tmp_path, points)
        second = _rows(tmp_path / "second", points)
        # Silently keeping the last would make two builds one curve, and drop the
        # other build's points without saying so.
        with pytest.raises(SystemExit) as caught:
            scaling.find_series(scaling.load_points([first, second]))
        assert "duplicate point" in str(caught.value)

    def test_the_fit_uses_the_minimum_sample(self, tmp_path):
        points = []
        for n in (1, 2, 4, 8):
            point = _point("granite_layer", {"layers": n}, frontend_ms=100.0 * n)
            # A contended extra sample, which only ever runs slow.
            point["measurements"]["frontend_ms"].append(100.0 * n * 4)
            points.append(point)
        series = scaling.find_series(scaling.load_points([_rows(tmp_path, points)]))
        fit = scaling.fit_series(series[0], "frontend_ms")
        assert fit.exponent == pytest.approx(1.0, abs=1e-9)

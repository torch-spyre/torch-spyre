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


"""Tests for the frontend timing sweep driver, its plan, and the scaling fits.

Loaded by path for the same reason as the summarizer tests: these are scripts, not a
package, and they depend on nothing but the standard library. Nothing here touches the
Spyre device -- the driver's plan handling, the plan's own consistency, and the fits are
all pure data.

The plan-consistency tests read ``workloads.py`` with ``ast`` rather than importing it,
because importing it needs ``torch_spyre._C`` and therefore a built extension. A static
read still catches the drift that matters: a plan point naming a parameter its builder
does not accept fails in the child, minutes into a sweep, on a machine you are not
watching.
"""

import ast
import importlib.util
import json
import os
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

    def test_the_child_gets_the_arm_and_its_label(self, tmp_path):
        point = run_sweep.Point(workload="mlp", env={"SENCORES": "1"})
        env = run_sweep._child_env(str(tmp_path), "r.json", str(tmp_path), True, point)
        assert env["SENCORES"] == "1"
        assert env[run_sweep.ARM_ENV_VAR] == "SENCORES=1"
        assert env["TORCH_SPYRE_FRONTEND_ONLY"] == "1"
        assert env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] == "1"

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

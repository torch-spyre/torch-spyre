#!/usr/bin/env python3
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

"""Fit how a metric scales along one sweep axis, and say how well the fit holds.

    python3 tools/frontend_timing/scaling.py rows.json
    python3 tools/frontend_timing/scaling.py rows.json --extrapolate layers=40

A series is the set of points that agree on everything except one parameter. Each is
fitted as ``log(metric) = a * log(axis) + b`` and reported with its R-squared, because
a slope through a few noisy points always produces a number. ``--extrapolate`` projects
a reliable fit, e.g. Granite's depth series to its 40 layers, which no single graph
compiles.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Any

#: Fewer points than this and a fit is arithmetic rather than evidence. Four is the
#: smallest series that can disagree with a straight line and show it.
MIN_POINTS = 4

#: Below this, report the exponent as unreliable rather than quoting it.
R_SQUARED_FLOOR = 0.90


@dataclass
class Series:
    """Points that agree on everything but one axis."""

    workload: str
    arm: str
    axis: str
    fixed: dict[str, Any]
    #: axis value -> median metric value, per metric name.
    samples: dict[Any, dict[str, float]]

    @property
    def label(self) -> str:
        held = "_".join(f"{k}{self.fixed[k]}" for k in sorted(self.fixed))
        name = f"{self.workload}[{self.axis}]"
        if held:
            name += f" {held}"
        if self.arm:
            name += f" +{self.arm}"
        return name


@dataclass
class Fit:
    exponent: float
    intercept: float
    r_squared: float
    points: int

    @property
    def reliable(self) -> bool:
        return self.points >= MIN_POINTS and self.r_squared >= R_SQUARED_FLOOR

    def predict(self, axis_value: float) -> float:
        return math.exp(self.intercept + self.exponent * math.log(axis_value))


def loglog_fit(pairs: list[tuple[float, float]]) -> Fit | None:
    """Least-squares fit of log(y) on log(x). None when the data cannot carry one.

    Zero or negative values have no logarithm, so they are dropped rather than nudged --
    a fudge factor here would silently change the exponent reported. A decode point at
    ``S=1`` survives (log 1 is 0); a metric that measured 0 ms does not.
    """
    usable = [(x, y) for x, y in pairs if x > 0 and y > 0]
    xs = [math.log(x) for x, _ in usable]
    ys = [math.log(y) for _, y in usable]
    if len(set(xs)) < 2:
        # Fewer than two distinct axis values: a slope is not defined.
        return None
    slope, intercept = statistics.linear_regression(xs, ys)
    # correlation() rejects a constant input; a flat series is fitted exactly.
    r_squared = statistics.correlation(xs, ys) ** 2 if len(set(ys)) > 1 else 1.0
    return Fit(
        exponent=slope, intercept=intercept, r_squared=r_squared, points=len(usable)
    )


def load_points(paths: list[str]) -> list[dict[str, Any]]:
    """Read every rows file, refusing a schema this version does not know."""
    points: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as handle:
            payload = json.load(handle)
        schema = payload.get("schema")
        if schema != "frontend-timing-rows/1":
            raise SystemExit(f"{path}: unknown rows schema {schema!r}")
        points += payload.get("points", [])
    return points


def find_series(points: list[dict[str, Any]]) -> list[Series]:
    """Every (workload, arm, axis) series with at least two distinct axis values.

    A candidate axis is any numeric parameter; the series is the set of points holding
    every *other* parameter still. A sweep plan that moves two axes at once therefore
    yields no series for either, which is the intended answer rather than a fit through
    a diagonal.
    """
    grouped: dict[tuple, dict[Any, dict[str, float]]] = {}
    for point in points:
        params = point.get("params", {})
        for axis, value in params.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            fixed = {k: v for k, v in params.items() if k != axis}
            key = (
                point.get("workload", "unknown"),
                point.get("arm", ""),
                axis,
                tuple(sorted(fixed.items())),
            )
            medians = {
                name: statistics.median(values)
                for name, values in point.get("measurements", {}).items()
                if values
            }
            grouped.setdefault(key, {})[value] = medians

    series = []
    for (workload, arm, axis, fixed_items), samples in grouped.items():
        if len(samples) < 2:
            continue
        series.append(
            Series(
                workload=workload,
                arm=arm,
                axis=axis,
                fixed=dict(fixed_items),
                samples=samples,
            )
        )
    return sorted(series, key=lambda s: s.label)


def fit_series(series: Series, metric: str) -> Fit | None:
    pairs = [
        (float(axis_value), metrics[metric])
        for axis_value, metrics in series.samples.items()
        if metric in metrics
    ]
    return loglog_fit(pairs) if len(pairs) >= 2 else None


def render(
    series_list: list[Series],
    metrics: list[str],
    extrapolate: tuple[str, float] | None = None,
) -> str:
    header = ["| series | metric | n | exponent | R2 | verdict |"]
    header.append("|---|---|--:|--:|--:|---|")
    if extrapolate:
        axis, value = extrapolate
        header[0] += f" {axis}={value:g} |"
        header[1] += "--:|"

    lines = list(header)
    for series in series_list:
        for metric in metrics:
            fit = fit_series(series, metric)
            if fit is None:
                continue
            if fit.points < MIN_POINTS:
                verdict = f"insufficient (n<{MIN_POINTS})"
            elif fit.r_squared < R_SQUARED_FLOOR:
                verdict = f"unreliable (R2<{R_SQUARED_FLOOR})"
            else:
                verdict = "ok"
            row = (
                f"| {series.label} | {metric} | {fit.points} | "
                f"{fit.exponent:.2f} | {fit.r_squared:.3f} | {verdict} |"
            )
            if extrapolate:
                axis, value = extrapolate
                if series.axis == axis and fit.reliable:
                    row += f" {fit.predict(value):.0f} |"
                else:
                    row += " - |"
            lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", nargs="+", help="rows files from summarize.py --json")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="metric to fit; repeatable (default: frontend_ms and graph_operations)",
    )
    parser.add_argument(
        "--extrapolate",
        metavar="AXIS=VALUE",
        help="also project each reliable fit on this axis to this value",
    )
    args = parser.parse_args(argv)

    points = load_points(args.rows)
    if not points:
        print("no points in rows file(s)", file=sys.stderr)
        return 1

    metrics = args.metric or ["frontend_ms", "graph_operations"]
    extrapolate = None
    if args.extrapolate:
        axis, _, raw = args.extrapolate.partition("=")
        if not raw:
            raise SystemExit("--extrapolate expects AXIS=VALUE")
        extrapolate = (axis, float(raw))

    series_list = find_series(points)
    if not series_list:
        print(
            "no series found: every point differs in more than one parameter",
            file=sys.stderr,
        )
        return 1
    print(render(series_list, metrics, extrapolate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

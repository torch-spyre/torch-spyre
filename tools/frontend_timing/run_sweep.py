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


"""Drive cold frontend-compile samples and leave one timing record per sample.

    python3 tools/frontend_timing/run_sweep.py --plan .../sweep_plan.json
    python3 tools/frontend_timing/run_sweep.py --workload mlp -p seq_len=128 -p layers=2

ONE PROCESS PER SAMPLE, and that is the whole reason this is a driver rather than a
loop. ``TORCHINDUCTOR_CACHE_DIR`` is read at import, so no in-process cache reset gives
a sample a cache directory that never held this graph. A fresh child with a fresh
directory does.

Each point gets a discarded warmup plus ``--samples`` measured runs, run serially --
the Spyre device is exclusive per process, and a parallel sweep would measure
contention. Records land wherever ``--out`` says; this script commits nothing and knows
no repository path.

Backend compilation is skipped by default (``TORCH_SPYRE_FRONTEND_ONLY=1``) because it
dominates wall time and is not what this measures. Pass ``--with-backend`` for a point
where the backend share itself is the question.

A plan point may carry ``tiers`` and ``env``. ``--tier NAME`` runs only the points that
declare it, which is how one plan serves a per-PR lane, a nightly lane and a weekly lane
without three files drifting apart; a point with no ``tiers`` runs in every tier.
``env`` sets environment for that point's children only, which is the A/B facility: the
only way to measure two configurations against one tree. The CP-SAT evidence behind the
complexity audit was taken before that optimization became the default, so re-running
both arms matters.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Where records go when nothing says otherwise. Matches tools/cost_model/records.py:
#: an explicit path wins, then the environment, then a directory beside this file.
DEFAULT_RECORDS_ENV = "SPYRE_FRONTEND_TIMING_RECORDS"


def resolve_out_dir(explicit: str | None) -> str:
    if explicit:
        return explicit
    from_env = os.environ.get(DEFAULT_RECORDS_ENV)
    if from_env:
        return from_env
    return os.path.join(_HERE, "records")


#: Plan keys that configure the sweep rather than the workload. Everything else in a
#: point is a builder keyword, so a new key here must also be added to this set or the
#: child will reject it as an unexpected argument.
RESERVED_PLAN_KEYS = frozenset({"workload", "tiers", "env", "comment"})

#: Carries the resolved A/B arm to the child, which records it so a summary can tell two
#: arms of one point apart instead of averaging them together.
ARM_ENV_VAR = "SPYRE_FTS_ENV_ARM"


@dataclass
class Point:
    """One sweep point: what to build, with what, under what environment."""

    workload: str
    params: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    tiers: tuple[str, ...] = ()

    @property
    def arm(self) -> str:
        """A short stable label for the environment arm; empty when there is none."""
        return ",".join(f"{k}={self.env[k]}" for k in sorted(self.env))


def parse_params(pairs: list[str]) -> dict[str, Any]:
    """Turn ``key=value`` strings into typed parameters."""
    params: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            params[key] = int(raw)
        except ValueError:
            params[key] = raw
    return params


def point_id(workload: str, params: dict[str, Any], arm: str = "") -> str:
    """A filesystem-safe, order-independent name for one sweep point.

    The arm is part of the name: without it two arms of the same point write to the same
    record filename and the second silently overwrites the first.
    """
    name = workload
    if params:
        name += "-" + "_".join(f"{k}{params[k]}" for k in sorted(params))
    if arm:
        safe = arm.replace("=", "").replace(",", "-").replace("/", "_")
        name += f"+{safe}"
    return name


# ---------------------------------------------------------------------------
# Worker: one cold compile, one record.


def run_sample(workload: str, params: dict[str, Any], sample: int) -> int:
    import torch
    from torch_spyre._inductor import config, timing_recorder

    # Resolves because main() put the repository root on sys.path before calling here.
    from tools.frontend_timing import workloads

    built = workloads.build(workload, **params)
    timing_recorder.set_run_meta(
        workload=built.name,
        sample=sample,
        cold=True,
        cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        spyre_config=_config_snapshot(config),
        env_arm=os.environ.get(ARM_ENV_VAR, ""),
        **built.params,
    )

    # fullgraph: a graph break would split one measurement across two compiles and
    # quietly change what is being timed. The control-flow workload needs it outright --
    # without it Dynamo leaves the scan HOP and hits a data-dependent scalar.
    compiled = torch.compile(built.fn, fullgraph=True)
    started = time.perf_counter()
    try:
        compiled(*built.args)
    except RuntimeError as exc:
        # Frontend-only mode compiles and then refuses to launch; that is the point
        # being reached, not a failure. Anything else is real.
        if not config.frontend_only or "TORCH_SPYRE_FRONTEND_ONLY" not in str(exc):
            raise
    wall_ms = (time.perf_counter() - started) * 1000

    # Recorded after the compile, so they describe it. compile_wall_ms is deliberately
    # redundant with the recorder's own total: when the two disagree, the recorder is
    # missing a region, and that is worth knowing from the record itself.
    timing_recorder.set_run_meta(
        compile_wall_ms=wall_ms,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        kernels_skipped=len(
            timing_recorder.RECORDER.run_meta.get("backend_skipped_kernels", []) or []
        ),
    )
    return 0


def _config_snapshot(config: Any) -> dict[str, Any]:
    """Resolved Spyre config, so a record says what planning produced it.

    Taken whole rather than from a list of names: a curated list goes stale silently,
    and a record that names the wrong configuration is worse than one that names too
    much. ``dir()`` does not work here -- install_config_module hides the entries behind
    a wrapper, and iterating attributes returns nothing.
    """
    for api in ("get_config_copy", "shallow_copy_dict", "to_dict"):
        getter = getattr(config, api, None)
        if callable(getter):
            return dict(getter())
    return {}


# ---------------------------------------------------------------------------
# Driver.


def _child_env(
    out_dir: str,
    record_name: str,
    cache_dir: str,
    frontend_only: bool,
    point: Point,
):
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    env["TORCH_SPYRE_TIMING"] = "1"
    env["TORCH_SPYRE_TIMING_OUT"] = os.path.join(out_dir, record_name)
    if frontend_only:
        env["TORCH_SPYRE_FRONTEND_ONLY"] = "1"
    else:
        env.pop("TORCH_SPYRE_FRONTEND_ONLY", None)
    # The arm goes last so a point can deliberately override anything above it, and its
    # label travels separately so the child can record which arm produced the record.
    env.update(point.env)
    env[ARM_ENV_VAR] = point.arm
    return env


def run_point(
    point: Point,
    out_dir: str,
    samples: int,
    frontend_only: bool,
    timeout_s: int,
) -> list[str]:
    """Run one point's warmup plus samples. Returns a list of failure descriptions."""
    workload, params = point.workload, point.params
    name = point_id(workload, params, point.arm)
    warmup_dir = os.path.join(out_dir, "warmup")
    os.makedirs(warmup_dir, exist_ok=True)
    failures: list[str] = []

    # Sample 0 is the warmup: its record goes somewhere the summarizer does not read.
    for sample in range(0, samples + 1):
        is_warmup = sample == 0
        target = warmup_dir if is_warmup else out_dir
        record_name = f"{name}-run{sample}.json"
        cache_dir = tempfile.mkdtemp(prefix=f"fts-{name}-{sample}-")
        label = f"{name} {'warmup' if is_warmup else f'sample {sample}/{samples}'}"
        argv = [
            sys.executable,
            os.path.abspath(__file__),
            "--_sample",
            "--workload",
            workload,
            "--sample-index",
            str(sample),
        ]
        for key, value in params.items():
            argv += ["--param", f"{key}={value}"]

        started = time.perf_counter()
        try:
            proc = subprocess.run(
                argv,
                env=_child_env(target, record_name, cache_dir, frontend_only, point),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            elapsed = time.perf_counter() - started
            if proc.returncode != 0:
                failures.append(f"{label}: exit {proc.returncode}")
                tail = (proc.stderr or "").strip().splitlines()[-5:]
                print(f"  {label}: FAILED after {elapsed:.1f}s", file=sys.stderr)
                for line in tail:
                    print(f"    {line}", file=sys.stderr)
            else:
                print(f"  {label}: {elapsed:.1f}s")
        except subprocess.TimeoutExpired:
            failures.append(f"{label}: timed out after {timeout_s}s")
            print(f"  {label}: TIMED OUT after {timeout_s}s", file=sys.stderr)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    return failures


def load_plan(path: str, tier: str | None = None) -> list[Point]:
    """Load a plan, keeping the points that belong to ``tier``.

    A bare list is still accepted, and a point with no ``tiers`` runs in every tier, so
    a plan written before tiers existed behaves exactly as it did.
    """
    with open(path) as handle:
        plan = json.load(handle)
    entries = plan["points"] if isinstance(plan, dict) else plan

    points: list[Point] = []
    for entry in entries:
        if "workload" not in entry:
            raise SystemExit(f"plan entry missing 'workload': {entry}")
        tiers = tuple(entry.get("tiers", ()))
        if tier is not None and tiers and tier not in tiers:
            continue
        env = {str(k): str(v) for k, v in (entry.get("env") or {}).items()}
        points.append(
            Point(
                workload=entry["workload"],
                params={k: v for k, v in entry.items() if k not in RESERVED_PLAN_KEYS},
                env=env,
                tiers=tiers,
            )
        )
    return points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", help="sweep plan JSON; omit to run a single point")
    parser.add_argument("--workload", help="workload name for a single point")
    parser.add_argument(
        "--param",
        "-p",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="workload parameter; repeatable",
    )
    parser.add_argument(
        "--tier",
        help="run only plan points declaring this tier (e.g. pr, nightly, weekly)",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment for this point's children; repeatable (A/B arm)",
    )
    parser.add_argument("--out", help="directory for records")
    parser.add_argument("--samples", type=int, default=3, help="measured samples (3)")
    parser.add_argument(
        "--with-backend",
        action="store_true",
        help="run the backend compiler too (much slower)",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="per-sample timeout in seconds"
    )
    parser.add_argument("--_sample", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sample-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    # The worker re-enters this file as a script, so the repository root has to be
    # importable for `from tools.frontend_timing import workloads` to resolve.
    repo_root = os.path.dirname(os.path.dirname(_HERE))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if args._sample:
        if not args.workload:
            raise SystemExit("--_sample requires --workload")
        return run_sample(args.workload, parse_params(args.param), args.sample_index)

    if bool(args.plan) == bool(args.workload):
        raise SystemExit("pass exactly one of --plan or --workload")

    cli_env = {k: str(v) for k, v in parse_params(args.env).items()}
    if args.plan:
        points = load_plan(args.plan, args.tier)
        if cli_env:
            raise SystemExit("--env applies to a single point; a plan carries its own")
    else:
        points = [
            Point(
                workload=args.workload,
                params=parse_params(args.param),
                env=cli_env,
            )
        ]
    if not points:
        raise SystemExit(f"no plan points declare tier {args.tier!r}")

    out_dir = resolve_out_dir(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"records -> {out_dir}")
    tier_note = f", tier {args.tier}" if args.tier else ""
    print(
        f"{len(points)} point(s){tier_note}, {args.samples} sample(s) each, "
        "plus a warmup"
    )
    if not args.with_backend:
        print("backend compilation skipped (TORCH_SPYRE_FRONTEND_ONLY=1)")

    failures: list[str] = []
    for point in points:
        print(f"{point_id(point.workload, point.params, point.arm)}:")
        failures += run_point(
            point,
            out_dir,
            args.samples,
            not args.with_backend,
            args.timeout,
        )

    if failures:
        print(f"\n{len(failures)} sample(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nall samples completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

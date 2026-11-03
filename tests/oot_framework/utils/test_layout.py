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

"""Group short test configs into shared CI jobs, from measured runtimes.

Which job a config runs in is a CI scheduling concern, so it is decided here at matrix time and
never written into the test configs. Two files feed it:

  .github/test_layout_policy.json  the knobs, edited by people and reviewed in PRs
  .github/test_timings.json        per-config runtimes, regenerated from ClickHouse
                                   (tests/scripts/refresh_test_timings.py)

Each config the tier filter selected is either `alone` (its own job, as today) or in a `bucket`
with other short configs. A config too long to share a job under `target_job_minutes` simply ends
up alone. So does a config with fewer than `min_runs` recorded runs (a new one, say): it runs on
its own until its median is worth trusting, then the next refresh of the timings bins it. A config that shares a test
file with another config also runs alone: a bucket reports under the bucket's name, and the refresh
can only credit its cases back to a config through a test file that config alone lists. Members of
a bucket share a runner, directory, flags and `global:` block, and never two of them name the same
test file, so a bucket runs exactly the tests its members would run on their own.

No policy file means every config runs alone, which is today's behaviour, so this can never be the
reason a matrix fails to build. A malformed policy or timings file is an error instead, because it
would otherwise change what runs without anyone noticing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_POLICY: dict = {
    "version": 1,
    "target_job_minutes": 8.0,
    "max_members": 16,
    "min_runs": 5,
    "exclude": [],
}


class LayoutError(ValueError):
    """A policy or timings file that would silently change what runs."""


def load_policy(path: Optional[str]) -> Optional[dict]:
    """The policy merged over the defaults, or None when the file is absent (grouping off)."""
    if not path or not Path(path).is_file():
        return None
    try:
        raw = json.loads(Path(path).read_text())
    except ValueError as exc:
        raise LayoutError(f"{path}: not valid JSON ({exc})") from None
    if not isinstance(raw, dict):
        raise LayoutError(f"{path}: expected a JSON object")
    unknown = set(raw) - set(DEFAULT_POLICY) - {"_comment"}
    if unknown:
        raise LayoutError(f"{path}: unknown key(s) {sorted(unknown)}")
    policy = {**DEFAULT_POLICY, **{k: v for k, v in raw.items() if k != "_comment"}}
    v = policy["target_job_minutes"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise LayoutError(
            f"{path}: target_job_minutes must be a positive number, got {v!r}"
        )
    for name, low in (("max_members", 2), ("min_runs", 1)):
        v = policy[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < low:
            raise LayoutError(f"{path}: {name} must be an integer >= {low}, got {v!r}")
    if not isinstance(policy["exclude"], list) or not all(
        isinstance(e, str) for e in policy["exclude"]
    ):
        raise LayoutError(
            f"{path}: exclude must be a list of config paths or directory prefixes"
        )
    return policy


def load_timings(path: Optional[str]) -> Dict[str, dict]:
    """{config key: {"p50_min", "runs"}}; {} when absent (every config then runs alone)."""
    if not path or not Path(path).is_file():
        return {}
    try:
        raw = json.loads(Path(path).read_text())
    except ValueError as exc:
        raise LayoutError(f"{path}: not valid JSON ({exc})") from None
    configs = raw.get("configs") if isinstance(raw, dict) else None
    if not isinstance(configs, dict):
        raise LayoutError(f'{path}: expected a "configs" object')
    out = {}
    for key, rec in configs.items():
        if not isinstance(rec, dict):
            raise LayoutError(f"{path}: {key}: expected an object, got {rec!r}")
        t, n = rec.get("p50_min"), rec.get("runs")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or t < 0:
            raise LayoutError(
                f"{path}: {key}: p50_min must be a number >= 0, got {t!r}"
            )
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise LayoutError(f"{path}: {key}: runs must be an integer >= 0, got {n!r}")
        out[key] = {"p50_min": float(t), "runs": n}
    return out


def step_overhead(minutes: float) -> float:
    """Pytest step time beyond the summed case time: ~0.25 min small, up to ~2 min for big configs."""
    return 0.25 + min(2.0, 0.12 * minutes)


def plan(
    configs: List[dict], policy: Optional[dict], timings: Dict[str, dict]
) -> Dict[str, str]:
    """{config key: bucket name, or "" to run alone} for the configs one matrix will run.

    `configs` are the tier-filtered records: key, dir, runner, flags, global, files, and
    shared_file (some test file of it is also listed by another config on disk).
    """
    if policy is None:
        return {c["key"]: "" for c in configs}
    cap = float(policy["target_job_minutes"])
    out: Dict[str, str] = {}
    groups: Dict[tuple, List[dict]] = {}
    for c in configs:
        key = c["key"]
        rec = timings.get(key)
        excluded = any(
            key == e or (e.endswith("/") and key.startswith(e))
            for e in policy["exclude"]
        )
        # Too few runs to size it: alone, exactly as it would run without grouping.
        if (
            excluded
            or c.get("shared_file")
            or rec is None
            or rec["runs"] < policy["min_runs"]
        ):
            out[key] = ""
            continue
        t = rec["p50_min"]
        group = (
            c["runner"],
            c["dir"],
            json.dumps(c["flags"], sort_keys=True),
            json.dumps(c["global"], sort_keys=True),
        )
        groups.setdefault(group, []).append({**c, "t": t})
    # Number buckets per (runner, dir) across all their flag/global groups: groups that differ only
    # in flags or `global:` share a prefix, and a per-group counter would give two of them the same
    # name, which filter_configs.py would then fold into one bucket.
    counters: Dict[str, int] = {}
    for (runner, d, _flags, _glob), members in sorted(groups.items()):
        prefix = f"{d.replace('/', '_') or 'top'}_{runner}"
        for bucket in _pack(members, cap, int(policy["max_members"])):
            if len(bucket) < 2:
                for m in bucket:
                    out[m["key"]] = ""
                continue
            counters[prefix] = counters.get(prefix, 0) + 1
            for m in bucket:
                out[m["key"]] = f"{prefix}_{counters[prefix]}"
    return out


def _pack(members: List[dict], cap: float, max_members: int) -> List[List[dict]]:
    """First-fit decreasing: longest first onto the fullest bucket that stays under both caps."""
    bins: List[list] = []  # [minutes, members, test file stems]
    for m in sorted(members, key=lambda m: (-m["t"], m["key"])):
        stems = {Path(f).stem for f in m["files"]}
        best = None
        for b in bins:
            after = b[0] + m["t"]
            if (
                after + step_overhead(after) > cap
                or len(b[1]) >= max_members
                or (b[2] & stems)
            ):
                continue
            if best is None or b[0] > best[0]:
                best = b
        if best is None:
            bins.append([m["t"], [m], set(stems)])
        else:
            best[0] += m["t"]
            best[1].append(m)
            best[2] |= stems
    return [b[1] for b in bins]

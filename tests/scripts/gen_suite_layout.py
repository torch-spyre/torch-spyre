#!/usr/bin/env python3
"""Generate .github/suite_layout.yaml: which CI job each test config runs in.

The layout is a CI concern, so it lives next to runner_overrides.yaml instead of in the test
configs. It is generated from measured runtimes and checked in; filter_configs.py reads it with
--layout-map. Every config is one of:

  shards: N     a long config, split into N parallel jobs (utils/shard.py balances them)
  bucket: NAME  a short config, run in one job with the other configs of that bucket
  (absent)      runs as its own job, as before

Packing, per tier-independent config:
  * expected test step = recorded minutes + a per-job step overhead (case durations
    understate the pytest step: ~0.25 min for small configs, up to ~2 min for big ones)
  * a config whose step is over --cap is split into the fewest shards that each fit
  * the rest are packed first-fit-decreasing into buckets that stay under --cap, never
    mixing runners, directories, flags or `global:` blocks, and never putting two configs
    that name the same test file (or the same test file stem) in one bucket
  * configs that must stay alone: multi-card (distributed_tests/), anything with
    ${TORCH_ROOT}, and anything named in --keep-alone

Input: a TSV with columns config, config_dir, p50_min, p50_min_last3d (from ClickHouse
test_case_runs; see experiments 018). Missing runtimes count as 0.5 min.

Usage:
  python tests/scripts/gen_suite_layout.py --runtimes runtimes.tsv --cap 7 \
      --config-root tests/configs/torch_spyre_tests --out .github/suite_layout.yaml
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml

UNKNOWN_MIN = 0.5


def step_overhead(minutes: float) -> float:
    return 0.25 + min(2.0, 0.12 * minutes)


def load_runtimes(path: str) -> dict:
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            t = r.get("p50_min_last3d") or r.get("p50_min")
            if t:
                out[r["config"]] = float(t)
    return out


def load_map(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def config_facts(cfg: Path, root: Path) -> dict:
    raw = yaml.safe_load(cfg.read_text()) or {}
    tsc = raw.get("test_suite_config") or {}
    files = [str(f.get("path", "")) for f in tsc.get("files") or []]
    return {
        "rel": str(cfg.relative_to(root)),
        "dir": str(cfg.parent.relative_to(root)),
        "stem": cfg.stem,
        "global": tsc.get("global"),
        "files": files,
        "file_stems": {Path(f).stem for f in files},
        "torch_root": any("${TORCH_ROOT}" in f for f in files),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runtimes", required=True)
    ap.add_argument(
        "--cap", type=float, default=7.0, help="target minutes of test step per job"
    )
    ap.add_argument(
        "--max-members",
        type=int,
        default=16,
        help="most configs in one bucket: a flake reruns, and a wedged card fails, all of them",
    )
    ap.add_argument("--config-root", default="tests/configs/torch_spyre_tests")
    ap.add_argument("--runner-map", default=".github/runner_overrides.yaml")
    ap.add_argument("--flags-map", default=".github/suite_test_flags.yaml")
    ap.add_argument(
        "--keep-alone",
        default="",
        help="comma-separated config paths that must run alone",
    )
    ap.add_argument("--out", default=".github/suite_layout.yaml")
    args = ap.parse_args(argv)

    root = Path(args.config_root).resolve()
    runtimes = load_runtimes(args.runtimes)
    runner_map = load_map(args.runner_map)
    flags_map = load_map(args.flags_map)
    keep_alone = {k.strip() for k in args.keep_alone.split(",") if k.strip()}

    layout = {}
    small = defaultdict(list)
    for cfg in sorted(root.rglob("*.yaml")):
        f = config_facts(cfg, root)
        minutes = runtimes.get(cfg.stem, UNKNOWN_MIN)
        f["t"] = minutes
        f["runner"] = runner_map.get(f["rel"], "spyre_pf_x1")
        f["flags"] = flags_map.get(f["rel"], "")
        if (
            f["rel"].startswith("distributed_tests/")
            or f["torch_root"]
            or f["rel"] in keep_alone
        ):
            continue
        if minutes + step_overhead(minutes) > args.cap:
            n = 2
            while minutes / n + step_overhead(minutes / n) > args.cap:
                n += 1
            layout[f["rel"]] = {"shards": n}
            continue
        key = (f["runner"], f["dir"], repr(f["flags"]), repr(f["global"]))
        small[key].append(f)

    for (runner, d, _flags, _glob), members in sorted(small.items()):
        members.sort(key=lambda m: (-m["t"], m["rel"]))
        bins = []  # [minutes, [facts], set(file stems)]
        for m in members:
            best = None
            for b in bins:
                after = b[0] + m["t"]
                if after + step_overhead(after) > args.cap:
                    continue
                if len(b[1]) >= args.max_members:
                    continue
                if b[2] & m["file_stems"]:
                    continue  # never two configs on one test file in a job
                if best is None or b[0] > best[0]:
                    best = b
            if best is None:
                bins.append([m["t"], [m], set(m["file_stems"])])
            else:
                best[0] += m["t"]
                best[1].append(m)
                best[2] |= m["file_stems"]
        n = 0
        for b in bins:
            if len(b[1]) < 2:
                continue  # a bucket of one saves nothing
            n += 1
            name = f"{d.replace('/', '_') or 'top'}_{runner}_{n}"
            for m in b[1]:
                layout[m["rel"]] = {"bucket": name}

    header = (
        "# Generated by tests/scripts/gen_suite_layout.py from measured runtimes; do not edit by hand.\n"
        f"# cap={args.cap} min of test step per job, at most {args.max_members} configs per bucket.\n"
        "# Keys are config paths relative to\n"
        "# tests/configs/torch_spyre_tests/, like runner_overrides.yaml. Read by filter_configs.py\n"
        "# --layout-map: `shards: N` splits a config into N jobs; configs sharing a `bucket` run as one job.\n"
    )
    with open(args.out, "w") as fh:
        fh.write(header)
        yaml.safe_dump(
            dict(sorted(layout.items())), fh, default_flow_style=None, sort_keys=False
        )
    sharded = sum(1 for v in layout.values() if "shards" in v)
    buckets = len({v["bucket"] for v in layout.values() if "bucket" in v})
    print(
        f"wrote {args.out}: {sharded} sharded configs, {buckets} buckets "
        f"({sum(1 for v in layout.values() if 'bucket' in v)} configs)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

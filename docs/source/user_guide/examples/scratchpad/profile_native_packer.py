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

"""Paired A/B of the native C++ packer vs the Python packer, with dispersion.

Successor to ``benchmarks/profile_native_packer.py`` on the
``packer-native-benchmarks`` branch, which reported a ratio of *summed* per-size
totals over 4 seeds x 1 repeat -- no per-instance pairing and no interval. This
version keeps the same workload generator (so the numbers stay comparable) and
changes only the measurement:

* every (size, seed) instance is solved by BOTH packers, so ratios are paired;
* arm order alternates per repeat, so frequency/thermal drift cannot favour
  whichever arm runs first;
* both wall-clock (``perf_counter``) and CPU time (``process_time``) recorded;
* ``deepcopy`` of the buffer list is hoisted OUT of the timed region (the old
  script timed it, an additive constant on both arms that biases the ratio
  toward 1);
* the finalized addresses and quality are compared across arms per instance --
  the packers are bit-identical, so this doubles as a correctness check and
  establishes that only time is under test;
* per size: median paired ratio, IQR, bootstrap 95% CI, win count, sign test.

No Spyre hardware is touched: layout planning is single-threaded CPU work, and
the backend is kept unloaded via ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``.

Measured results, and the methodology in prose, are in
``docs/source/compiler/native_packer_performance.md``.

Reproducing the reported figures
--------------------------------
Build the ``_C`` extension WITHOUT ``TORCH_SPYRE_DEBUG`` (a ``-O0`` build inflates
every ratio), then, from anywhere::

    P=docs/source/user_guide/examples/scratchpad/profile_native_packer.py

    # representative capacity -- the headline number
    python $P --sizes 8 16 32 64 128 --seeds 15 --repeats 2 \\
        --cap-rule foot2 --min-steps 100 --json foot2.json

    # tighter capacity, for the pressure axis
    python $P ... --cap-rule foot4 ...

    # the rule the original measurement used, for comparability
    python $P ... --cap-rule 3xmax --repeats 3 ...

Ratios depend strongly on ``--cap-rule``: capacity pressure, not problem size, is
the dominant variable (see ``_capacity``). Quote the rule alongside any number.
The speedup is a large constant factor that erodes as ``n`` grows past ~128,
because the native packer recomputes placement from scratch per operation while
the Python one is incremental -- so ``--sizes`` matters too. Real captured graphs
run n~5-80.
"""

import os

# Pin BLAS/OMP threads *before* importing torch.
for _v in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")
# Keep the backend (and the accelerator) out of this entirely.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random as rnd  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402


def _repo_root():
    """Walk up from this file to the checkout root.

    The instance generator is imported from ``tests/`` (see below), so the repo
    root has to be importable no matter where this script is invoked from or
    where in the tree it ends up living.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "torch_spyre")) and os.path.isdir(
            os.path.join(d, "tests")
        ):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("cannot locate the torch-spyre checkout root from __file__")


sys.path.insert(0, _repo_root())

from torch_spyre._inductor import config  # noqa: E402
from torch_spyre._inductor.scratchpad.permutation_layout import (  # noqa: E402
    NativePermutationLayoutSolver,
    PermutationBasedLayoutSolver,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import (  # noqa: E402
    SimulatedAnnealingLayoutSolver,
)

# The canonical instance generator: the very one the native-vs-Python
# differential suite and the SA equivalence test drive. The old benchmark
# inlined its own copy, which can wire a *write-only* in-place parent -- a
# relationship current validation rejects outright (check_in_place_parent_is_read),
# so that copy no longer runs on this tree at all past n~8. Importing the test's
# version fixes that and makes the workload provenance exact.
from tests.inductor.test_perm_layout_solver import _random_buffers  # noqa: E402


def _capacity(buffers, rule):
    """Scratchpad capacity for one instance under the chosen rule.

    ``3xmax`` is what the archived benchmark used. It forces the SA search to
    iterate, but scales badly: capacity tracks the *largest buffer* while the
    footprint grows with ``n``, so cap/footprint falls from ~0.8 at n=8 to ~0.03
    at n=256 and all but ~13% of buffers end up evicted -- an over-subscribed
    regime unlike real LX planning. ``foot<K>`` sets capacity to
    ``footprint // K``, the convention the sibling co-optimizer benchmarks use,
    which holds the pressure constant as ``n`` grows.
    """
    if rule == "3xmax":
        return max(b.size for b in buffers) * 3
    return max(1, sum(b.size for b in buffers) // int(rule[4:]))


def _make_instances(sizes, seeds, rule):
    """Deterministic (n, seed, capacity, buffers) instances across sizes x seeds."""
    instances = []
    for n in sizes:
        for seed in seeds:
            rng = rnd.Random(seed * 1000 + n)
            buffers = _random_buffers(rng, n)
            instances.append((n, seed, _capacity(buffers, rule), buffers))
    return instances


# --- one timed solve -------------------------------------------------------- #
def _timed_solve(buffers, capacity, seed, native):
    """Solve one pre-copied instance under one packer.

    Returns (wall_seconds, cpu_seconds, addresses, quality). The buffer copy is
    the caller's job: only solver construction + solve + finalize are timed.
    """
    with config.patch(native_layout_packer=native):
        w0, c0 = time.perf_counter(), time.process_time()
        solver = SimulatedAnnealingLayoutSolver(
            buffers, capacity, 128, random=rnd.Random(seed)
        )
        # Guard against the knob silently selecting the wrong packer, which would
        # quietly turn this into a native-vs-native measurement.
        expect = (
            NativePermutationLayoutSolver if native else PermutationBasedLayoutSolver
        )
        if not isinstance(solver.plan, expect):
            raise RuntimeError(
                f"native={native} selected {type(solver.plan).__name__}, "
                f"expected {expect.__name__}"
            )
        out = solver.plan_layout()
        wall = time.perf_counter() - w0
        cpu = time.process_time() - c0
    quality = solver.plan.quality()
    # Annealing steps actually taken. An instance whose initial layout is already
    # optimal exits solve() immediately (steps == 0) and its timing reflects
    # construction overhead, not the packer -- reported separately below.
    steps = sum(len(q) for q in solver.quality_logs)
    allocated = solver.plan.count_allocated()
    return wall, cpu, tuple(b.address for b in out), quality, steps, allocated


def _copy_cost(buffers, reps=5):
    """Wall-clock of one deepcopy, to quantify the old script's additive bias."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        copy.deepcopy(buffers)
        best = min(best, time.perf_counter() - t0)
    return best


# --- statistics ------------------------------------------------------------- #
def _bootstrap_ci(xs, rng, iters=20000, alpha=0.05):
    """Percentile bootstrap CI for the median of ``xs``."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    meds = []
    k = len(xs)
    for _ in range(iters):
        meds.append(statistics.median(rng.choices(xs, k=k)))
    meds.sort()
    lo = meds[int(alpha / 2 * iters)]
    hi = meds[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def _iqr(xs):
    if len(xs) < 4:
        return (min(xs), max(xs))
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return (q[0], q[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="Buffer counts to sweep. Captured real graphs run n~5-80.",
    )
    ap.add_argument("--seeds", type=int, default=15, help="Instances per size.")
    ap.add_argument(
        "--repeats", type=int, default=3, help="Timed repeats of each (arm, instance)."
    )
    ap.add_argument(
        "--reduce",
        choices=("min", "median"),
        default="min",
        help="How to reduce repeats to one time per (arm, instance). Timing "
        "noise is one-sided, so min is the better estimate of true cost; "
        "median is reported alongside for comparison.",
    )
    ap.add_argument(
        "--cap-rule",
        default="3xmax",
        help="Capacity rule: '3xmax' (archived benchmark) or 'footK' "
        "(footprint // K, e.g. foot2), which holds pressure constant in n.",
    )
    ap.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Instances whose SA search took fewer steps than this are reported "
        "separately: their timing measures construction, not the search.",
    )
    ap.add_argument("--json", default=None, help="Write raw + summary JSON here.")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    instances = _make_instances(args.sizes, seeds, args.cap_rule)
    print(
        f"instances: {len(instances)} ({len(args.sizes)} sizes x {args.seeds} seeds), "
        f"repeats={args.repeats}, sizes={args.sizes}",
        flush=True,
    )
    print(
        f"host: {platform.node()} {platform.machine()} "
        f"cpus={os.cpu_count()} python={sys.version.split()[0]}",
        flush=True,
    )

    # raw[(n, seed)][arm] = {"wall": [...], "cpu": [...]}
    raw: dict = {}
    mismatches = []
    for n, seed, capacity, buffers in instances:
        key = (n, seed)
        footprint = sum(b.size for b in buffers)
        raw[key] = {
            "python": {"wall": [], "cpu": []},
            "native": {"wall": [], "cpu": []},
            "copy": _copy_cost(buffers),
            "capacity": capacity,
            "footprint": footprint,
            "cap_over_footprint": capacity / footprint,
            "steps": None,
            "allocated": None,
            "n_buffers": n,
        }
        results = {}
        for rep in range(args.repeats):
            # Alternate which arm goes first, so drift within a pair cannot
            # systematically favour one packer.
            arms = [("native", True), ("python", False)]
            if rep % 2:
                arms.reverse()
            for label, native in arms:
                work = copy.deepcopy(buffers)
                wall, cpu, addrs, quality, steps, alloc = _timed_solve(
                    work, capacity, seed, native
                )
                raw[key][label]["wall"].append(wall)
                raw[key][label]["cpu"].append(cpu)
                raw[key]["steps"] = steps
                raw[key]["allocated"] = alloc
                results.setdefault(label, (addrs, quality))
                if results[label] != (addrs, quality):
                    mismatches.append((n, seed, label, "nondeterministic"))
        if results["native"] != results["python"]:
            mismatches.append((n, seed, "cross-arm", "addresses/quality differ"))
        print(
            f"  n={n:4d} seed={seed:2d}  "
            f"native={min(raw[key]['native']['wall']) * 1e3:8.2f} ms  "
            f"python={min(raw[key]['python']['wall']) * 1e3:8.2f} ms  "
            f"ratio={min(raw[key]['python']['wall']) / min(raw[key]['native']['wall']):5.2f}x"
            f"  steps={raw[key]['steps']:5d} alloc={alloc:4d}/{n}"
            f" cap/foot={capacity / footprint:.3f}",
            flush=True,
        )

    reduce_fn = min if args.reduce == "min" else statistics.median
    rng = rnd.Random(12345)
    summary = {}

    trivial = [k for k, v in raw.items() if v["steps"] < args.min_steps]
    if trivial:
        print(
            f"\nexcluded {len(trivial)}/{len(raw)} instances with < {args.min_steps} "
            f"annealing steps (initial layout already optimal -> solve() returns "
            f"immediately, so the timing measures construction, not the packer):"
        )
        for n, seed in sorted(trivial):
            r = raw[(n, seed)]
            print(
                f"    n={n:4d} seed={seed:2d} steps={r['steps']:5d} "
                f"ratio={reduce_fn(r['python']['wall']) / reduce_fn(r['native']['wall']):5.2f}x "
                f"native={reduce_fn(r['native']['wall']) * 1e3:7.3f} ms"
            )

    print("\n=== paired per-instance ratios (python / native), search-exercising ===")
    hdr = (
        f"{'n':>5} {'inst':>5} {'steps':>6} {'c/f':>5} {'alloc':>6} "
        f"{'native ms':>10} {'python ms':>10} "
        f"{'median':>7} {'IQR':>15} {'95% CI':>15} {'wins':>7} {'sign p':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for axis in ("wall", "cpu"):
        if axis == "cpu":
            print(f"\n--- CPU time ({axis}) ---")
            print(hdr)
            print("-" * len(hdr))
        for n in args.sizes:
            ratios, nat, pyt = [], [], []
            for seed in seeds:
                r = raw[(n, seed)]
                if r["steps"] < args.min_steps:
                    continue
                a = reduce_fn(r["native"][axis])
                b = reduce_fn(r["python"][axis])
                nat.append(a)
                pyt.append(b)
                ratios.append(b / a)
            if not ratios:
                continue
            ref = next(
                raw[(n, s)] for s in seeds if raw[(n, s)]["steps"] >= args.min_steps
            )
            med = statistics.median(ratios)
            lo, hi = _bootstrap_ci(ratios, rng)
            q1, q3 = _iqr(ratios)
            wins = sum(1 for r in ratios if r > 1.0)
            # One-sided sign test against "the packers are equally fast".
            p = 0.5 ** len(ratios) if wins == len(ratios) else float("nan")
            summary[f"{axis}/n={n}"] = {
                "instances": len(ratios),
                "steps": ref["steps"],
                "cap_over_footprint": ref["cap_over_footprint"],
                "allocated": ref["allocated"],
                "native_ms_median": statistics.median(nat) * 1e3,
                "python_ms_median": statistics.median(pyt) * 1e3,
                "ratio_median": med,
                "ratio_iqr": [q1, q3],
                "ratio_ci95": [lo, hi],
                "wins": wins,
                "sign_p": p,
                "ratios": ratios,
            }
            print(
                f"{n:>5} {len(ratios):>5} {ref['steps']:>6} "
                f"{ref['cap_over_footprint']:>5.2f} {ref['allocated']:>3}/{n:<3} "
                f"{statistics.median(nat) * 1e3:>10.2f} "
                f"{statistics.median(pyt) * 1e3:>10.2f} "
                f"{med:>6.2f}x [{q1:>5.2f}, {q3:>5.2f}] "
                f"[{lo:>5.2f}, {hi:>5.2f}] {wins:>3}/{len(ratios):<3} "
                f"{p:>9.2e}"
            )

    # Pooled across all sizes, wall-clock, search-exercising instances only.
    pooled = [
        reduce_fn(raw[(n, s)]["python"]["wall"])
        / reduce_fn(raw[(n, s)]["native"]["wall"])
        for n in args.sizes
        for s in seeds
        if raw[(n, s)]["steps"] >= args.min_steps
    ]
    lo, hi = _bootstrap_ci(pooled, rng)
    print(
        f"\npooled (all sizes, wall, {args.reduce}-of-{args.repeats}): "
        f"median {statistics.median(pooled):.2f}x, "
        f"95% CI [{lo:.2f}, {hi:.2f}], range [{min(pooled):.2f}, {max(pooled):.2f}], "
        f"n={len(pooled)} instances"
    )

    copy_frac = statistics.median(
        [
            raw[(n, s)]["copy"] / reduce_fn(raw[(n, s)]["native"]["wall"])
            for n in args.sizes
            for s in seeds
        ]
    )
    print(
        f"deepcopy cost (excluded here, included by the old script): "
        f"median {copy_frac * 100:.1f}% of a native solve"
    )

    if mismatches:
        print(f"\n!! {len(mismatches)} MISMATCHES: {mismatches[:5]}")
    else:
        print(
            f"\ncross-arm check: addresses + quality identical on all "
            f"{len(instances)} instances (bit-exact, as the differential suite asserts)"
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "host": {
                        "node": platform.node(),
                        "machine": platform.machine(),
                        "cpus": os.cpu_count(),
                        "python": sys.version.split()[0],
                    },
                    "summary": summary,
                    "raw": {f"{n}/{s}": raw[(n, s)] for n in args.sizes for s in seeds},
                    "mismatches": mismatches,
                },
                f,
                indent=1,
            )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

# Copyright 2025 The Torch-Spyre Authors.
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

"""Offline cost-model accuracy evaluator -- score a model version WITHOUT hardware.

The measured device time (``kernel_us``) is version-INDEPENDENT and already stored in
``tools/cost_model/sweep_records.json``; only the model's PREDICTION changes when we edit the model.
So new accuracy is a pure computation: take each run's serialized feature vector, call
``cost_model.predict_ops`` with the current (or overridden) params, and compare to the
stored ``kernel_us``. No Spyre run, and the arithmetic is done by this program -- not by
hand -- so it is reliable at scale.

Feature source per record (in priority order):
  1. ``feats`` -- the exact serialized OpFeatures the harness now dumps (``MODEL FEATS``).
     Populated on every sweep run going forward (free -- one extra log line).
  2. ``io`` -- for OLDER rows without ``feats``, reconstruct OpFeatures from the stored
     device-I/O block. RECONSTRUCTION IS SELF-VALIDATED: we re-predict with the CURRENT
     default params and require it to reproduce the row's stored ``pred_us`` (which the
     current model produced) within a tolerance; rows that fail are reported and EXCLUDED,
     never silently trusted. Matmul is not reconstructed (split-derived features are
     unreliable from I/O alone) -> those rows need a ``feats`` re-run.

``cost_model.py`` imports only ``dataclasses``/``math`` (no torch), so this runs anywhere.

    python tools/cost_model/eval_model.py                         # is_current rows, current params
    python tools/cost_model/eval_model.py --all                   # every row (not just is_current)
    python tools/cost_model/eval_model.py --category matmul --op softmax_row_tiling
    python tools/cost_model/eval_model.py --params bw_peak_gbps=155
    python tools/cost_model/eval_model.py --verify                # feature-fidelity vs stored pred_us
    python tools/cost_model/eval_model.py --update                # write recomputed pred_us/err back
"""

import argparse
import collections
import importlib.util
import json
import math
import os
import sys

import regex as re

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from records import records_path  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(_HERE))  # tools/cost_model/ -> repo root


def _load_cost_model():
    """Import cost_model.py standalone (it has no torch dependency)."""
    path = os.path.join(_ROOT, "torch_spyre", "_inductor", "cost_model.py")
    spec = importlib.util.spec_from_file_location("cost_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load_cost_model()

# op -> reporting category
_CATEGORY = {}
for _c, _ops in {
    # add5/add6/add_indep2 belong here as much as add3/add4 do; leaving them in `other`
    # hid a real signal (the dependent-chain gap below) behind an unrelated bucket.
    "pointwise": "neg gelu relu sigmoid exp mul add add3 add4 add5 add6 add_indep2".split(),
    "reduction": "sumrow sumcol sumall amax mean read".split(),
    "transport": "transpose transpose_outer cat0 cat1".split(),
    # copy (x+1.0) is really a broadcast op -- an add with a resident broadcast constant.
    "broadcast": "copy bcast mulbcast bcastcol write".split(),
    "matmul": ["mm"],  # planner-chosen split (balanced) -- the no-regress baseline
    "matmul_split": ["mmwd"],  # FORCED split (incl. lopsided) -- the split-shape target
    "matmul_row": ["matmul_row_tiling"],
    "matmul_k": ["matmul_k_tiling"],
    "matmul_nested": ["mm_nested_m_k"],
    "bmm": ["bmm_k_tiling"],
    "bmm_3d2d": ["bmm_3d2d_k_tiling"],
    "bmm_nested": ["bmm_nested_b_k"],
    "bmm_split": ["bmm_wd", "bmm_wd_3d2d"],  # FORCED-split bmm (full + shared-weight)
    "softmax": ["softmax_row_tiling", "softmax_noexp_row_tiling"],
    "softmax_unrolled": ["softmax_unrolled"],
    "coarse_reduction": "ctsum ctamax ctamin chain".split(),
}.items():
    for _o in _ops:
        _CATEGORY[_o] = _c

# hbm_pattern the extractor assigns per bench op -- used ONLY for I/O reconstruction of
# old rows (feats, when present, carry the exact value). Cross-row (dim0) reductions --
# sumcol and the coarse ct* variants -- fold turnaround into the reduce_outer rate.
_HBM_PATTERN_BY_OP = {
    "transpose": "restickify",
    "cat0": "stick_scatter",
    "sumcol": "reduce_outer",
    "ctsum": "reduce_outer",
    "ctamax": "reduce_outer",
    "ctamin": "reduce_outer",
}
_NO_RECONSTRUCT = {
    "mm",
    "mmwd",
    "bmm_wd",
    "bmm_wd_3d2d",
    "matmul_row_tiling",
}  # matmul-type: need `feats` (IO reconstruction can't rebuild the compute term)


def category(op):
    return _CATEGORY.get(op, "other")


def make_params(overrides):
    """CostParams with ``k=v`` overrides (values parsed as float)."""
    p = cm.CostParams()
    for kv in overrides:
        k, _, v = kv.partition("=")
        k = k.strip()
        if not hasattr(p, k):
            raise SystemExit(f"unknown CostParams field: {k!r}")
        setattr(p, k, float(v))
    return p


def _shape(v):
    """Parse a stored shape ('[1024, 512]' or a list) into list[int]; [] when absent."""
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    if isinstance(v, str) and v.strip().startswith("["):
        body = v.strip()[1:-1].strip()
        if not body:
            return []
        try:
            return [int(x) for x in body.split(",")]
        except ValueError:
            return []
    return []


def reconstruct_from_io(rec):
    """Best-effort OpFeatures for an old row from its stored device-I/O block. Returns a
    list[OpFeatures] or None if unsupported (matmul / no I/O). SELF-VALIDATED by the
    caller against the stored pred_us before it is trusted."""
    io = rec.get("io")
    op_name = rec.get("op")
    if not io or op_name in _NO_RECONSTRUCT:
        return None
    try:  # old feats-less rows can carry cores='-'/None -> unscoreable, skip
        cores = int(rec.get("cores") or 1)
    except (ValueError, TypeError):
        return None
    tiles = int(rec.get("tiles") or 0)
    tiled = tiles >= 2
    rows = rec.get("rows")
    tile_rpc = rows / (cores * tiles) if (tiled and rows and cores) else 0.0
    pat = _HBM_PATTERN_BY_OP.get(op_name, "")
    ops = []
    for o in io:
        args, out_elems = [], 0
        for t in o.get("tensors", []):
            elems = int(t["elems"])
            args.append(
                cm.ArgTraffic(
                    name=t["name"],
                    role=t["role"],
                    is_lx=t["mem"].lower() == "lx",
                    elems=elems,
                    broadcast="broadcast" in (t.get("flags") or ""),
                    loop_factor=1,
                    # The I/O block records both shapes; carry them through. Without them the
                    # access-pattern paths that key on shape (`_transport_kind`, the bmm layout
                    # classifier) silently fall through to the default copy model, so a
                    # feats-less row would be scored against the WRONG model.
                    logical=_shape(t.get("logical")),
                    dims=_shape(t.get("device")),
                )
            )
            if t["role"] == "output":
                out_elems = elems
        is_red = o.get("kind") == "reduction"
        red_cores = (
            max(1, cores // out_elems) if (is_red and 0 < out_elems < cores) else 1
        )
        ops.append(
            cm.OpFeatures(
                name=o["op"],
                is_reduction=is_red,
                out_elems=out_elems,
                cores=cores,
                dtype_bytes=2,
                args=args,
                reduction_cores=red_cores,
                loop_trip=tiles if tiled else 1,
                tiles_output_dim=tiled,
                tile_rows_per_core=tile_rpc,
                hbm_pattern="" if is_red and pat != "reduce_outer" else pat,
            )
        )
    return ops


def features_for(rec, default_params, tol=0.02):
    """Return (ops, source) for a record, or (None, reason). Reconstruction is gated two
    ways: (1) the HBM byte totals (R, W) must match the stored ones -- a MODEL-INDEPENDENT
    check, so it holds even when the model is changed; and (2) for ops whose model term is
    UNCHANGED since the row was logged, the reconstructed prediction must also reproduce the
    stored ``pred_us`` -- this catches pattern/tile reconstruction errors that bytes alone
    miss. The exception is any op affected by a model term added THIS cycle (the broadcast
    rate): its prediction legitimately differs from the logged one, so it is byte-checked
    only. (When ``feats`` are present none of this applies -- they are exact.)"""
    if rec.get("feats"):
        return [cm.op_from_dict(d) for d in rec["feats"]], "feats"
    ops = reconstruct_from_io(rec)
    if ops is None:
        return None, "no-feats"
    m = rec.get("model") or {}
    R, W = m.get("R"), m.get("W")
    if R is None or W is None:
        return None, "no-oracle"
    r, w = cm._fused_hbm_bytes(ops)
    if abs(r - R) > 0.01 * max(R, 1) or abs(w - W) > 0.01 * max(W, 1):
        return None, f"io-mismatch(bytes R {r}v{R} W {w}v{W})"
    stored = rec.get("pred_us")
    if stored and not any(cm._is_broadcast_op(o) for o in ops):
        got = cm.predict_ops(ops, default_params) / 1000.0
        if abs(got - stored) / stored > tol:
            return None, f"io-mismatch(pred {got:.1f}v{stored:.1f})"
    return ops, "io"


# --------------------------------------------------------------------------- scope
# STANDING SCOPE DECISIONS (user, 2026-07-31). These are permanently OUT of the
# model's evaluation set -- not filtered ad hoc per analysis, so a number quoted
# anywhere always refers to the same population.
#
#   1. cores < 8. Never a target regime. The few points we had were also
#      hopelessly confounded (cores aliased with per-core area and split fanout,
#      n=2 per core count on a single shape), so they could only ever have been
#      fitted, not explained.
#   2. A fused reduction with fewer than 1024 columns. The reduced-axis length is
#      the dominant driver of the fused-reduction rate and the short-row corner is
#      a different regime; including it drags a shape term toward a corner we do
#      not care about.
#
# Everything else stays in, including the slow bmm layouts: those are the EVIDENCE
# for the additive layout model even though only the fast layout is a target.
_MIN_CORES = 8
_MIN_FUSED_REDUCTION_COLS = 1024
# 4. Extreme lopsided work-division splits (16x2 / 2x16) ON THE COARSE-TILING OPS ONLY.
#    Those splits are not chosen by us there -- the coarse-tiling hint makes the planner
#    pick them, and they land far outside the calibration point of the matmul/bmm compute
#    rate, which is fitted at ONE split (4x8) and cores=32.
#    Evidence it is the SPLIT, not the tiling: a PLAIN (non-coarse) bmm forced to 16x2
#    reads -53.2 %, essentially the same as coarse-tiled bmm at 16x2 (-54.1 %), while
#    plain bmm at 4x8 is -7.4 % over 213 rows.
#    DELIBERATELY NOT applied to plain mm/bmm: `mmwd` at 16x2 (n=26, 10.6 % RMS) and
#    2x16 (n=24, 10.7 %) is modelled WELL, and those rows are the evidence that the mm
#    rate generalises across split geometry. Excluding them would discard good data and
#    make matmul_split look worse (15.1 -> 15.8) purely by dropping easy points.
#    This scopes out a regime we do not run; it does NOT excuse the bmm rate, whose
#    split confound is tracked separately.
_EXCLUDED_SPLITS = {(16, 2), (2, 16)}
_COARSE_OPS = {
    "matmul_row_tiling",
    "matmul_k_tiling",
    "mm_nested_m_k",
    "bmm_k_tiling",
    "bmm_nested_b_k",
    "bmm_3d2d_k_tiling",
}
# 3. Logs written at these SHAs carry CORRUPT features, not merely stale ones. Proof:
#    the same measured config (bmm_k_tiling B=4 1024x2048x1024 cores=32, untiled) is
#    recorded at rows_per_core=4 with m_split/n_split=None here, versus rows_per_core=64
#    with m_split=16/n_split=2 in every later log -- a 16x error in M/m. The MEASUREMENTS
#    agree to 0.8 %, so only the features drifted; scoring them yields +20 % where the
#    correct features yield -51 % on an identical kernel. Rows like that cannot judge the
#    model in either direction.
_CORRUPT_FEATURE_SHAS = {"7f37527", "f321503"}
# 4. Rows extracted by the FIRST version of the per-level `loop_factor` code, whose
#    guard `if syms and not (syms & free)` skipped every level with NO tiled symbols.
#    An op that tiles NOTHING at a level is loop-INVARIANT there, so all its args
#    repeat -- the coarse_tile_fill / coarse_tile_combine case. The bug under-counted
#    one K-tiled bundle 552 MB -> 216 MB and moved the control op -6.3 % -> -64.2 %.
#    Only bundles CONTAINING a fill/combine op are affected, which is why
#    matmul_row_tiling (a single-op bundle) is unaffected and stays in scope.
_BUGGY_LOOP_FACTOR_LOG = "coarse_reextract_20260804_004110.log"


_DEFAULT_BMM_LAYOUT = "0,1,2"  # the order the compiler emits (config default)

#: `add3_sep` ... `add6_sep`: one chain measured as several kernels. See `in_scope`.
_SEP_CHAIN = re.compile(r"add\d+_sep")


def in_scope(rec) -> bool:
    """False for rows excluded by the standing scope decisions above."""
    try:
        cores = int(rec.get("cores"))
    except (TypeError, ValueError):
        cores = None
    if cores is not None and cores < _MIN_CORES:
        return False
    sha = rec.get("model_sha") or ""
    if sha in _CORRUPT_FEATURE_SHAS and rec.get("feats"):
        return False
    # NON-DEFAULT BMM OPERAND LAYOUT. A faster device tile order exists (all three
    # operands on [1,0,2], up to 8.3x at byte-identical traffic), but it is reachable by
    # opting into the alternative operand layout, which is not the default -- so nothing the compiler
    # emits uses it. The model prices only what ships, so runs that FORCED another
    # arrangement are out of scope. The measurement itself is kept: it is the evidence
    # for that 8.3x, and the database still holds every combination.
    la, lb = rec.get("layout_a"), rec.get("layout_b")
    if (la or lb) and not (la == _DEFAULT_BMM_LAYOUT and lb == _DEFAULT_BMM_LAYOUT):
        return False
    op = rec.get("op") or ""
    # The `write` OUTER PRODUCT (b[1,C] + c[R,1]) costs far more than its bytes, and the
    # fitted surface that used to charge for it was removed as an unexplained black box.
    # Nothing models it now, so scoring against it measures a deliberate gap rather than
    # the model: 41.1 % RMS at -32.3 % mean, against 3.0-6.6 % for every operand this
    # section does model. The runs stay in the database as the evidence.
    if op == "write":
        return False
    # `add{N}_sep` CANNOT BE SCORED AS RECORDED -- a harness limitation, not a model gap.
    # The workload is N-1 SEPARATE compiled kernels (profile_ops.py, `add{N}_sep`), the
    # measured time covers all of them, but the cost-model dump captures only the LAST
    # sub-kernel's features. So a one-kernel prediction is compared against an N-kernel
    # measurement, and the error is arithmetic: -55/-70/-80/-83 % for N-1 = 2/3/4/5.
    # Multiplying the prediction by the sub-kernel count collapses those to -10/-9/-19/
    # -17 %, the same band as the fused `add{N}` rows -- which is the check that this is
    # bookkeeping and not a real miss. Scoring them measures the harness. The runs stay
    # in the database: they are the fusion control, read as add{N} vs add{N}_sep.
    if _SEP_CHAIN.fullmatch(op):
        return False
    if op.startswith("softmax"):
        cols = rec.get("cols")
        if isinstance(cols, int) and cols < _MIN_FUSED_REDUCTION_COLS:
            return False
    # Ops whose recorded features PREDATE the corrected per-arg loop_factor and cannot be
    # repaired offline: their bundles contain coarse_tile_fill / _combine ops whose args
    # need loop_trip (matmul_k_tiling) or the K-split (mm_nested_m_k), and no recorded
    # field pins those. Scoring a model against a wrong byte count is meaningless, so
    # they are excluded until `run_coarse_final_reextract.sh` replaces them.
    # matmul_row_tiling is NOT here: single-op bundle, repaired deterministically and
    # validated 11/11 against fresh extractions.
    if op in ("matmul_k_tiling", "mm_nested_m_k"):
        for f in rec.get("feats") or []:
            if f.get("is_matmul") and (f.get("tiles_reduction_dim") is None):
                return False  # pre-fix extractor: field did not exist yet
    # Feature-corrupt: fill/combine args mis-counted by the first per-level implementation.
    # STALE PER-ARG loop_factor. A coarse-tiled MATMUL always has a loop-invariant
    # operand: row-tiling repeats B, K-tiling repeats the accumulator. So at
    # tiles > 1 SOME arg must carry loop_factor > 1. If none does, the row was extracted
    # before the per-arg fix and its byte count is wrong -- scoring any model against it
    # is meaningless. Detected structurally rather than by log name so a future stale
    # log cannot slip back in. NOT applied to the softmax ops: row-tiling a reduction
    # tiles every tensor alike, so flat factors are CORRECT there (all 147 rows are flat,
    # in every log, including the freshly re-extracted ones).
    if op in ("matmul_row_tiling", "matmul_k_tiling", "mm_nested_m_k"):
        try:
            _t = int(rec.get("tiles") or 1)
        except (TypeError, ValueError):
            _t = 1
        if _t > 1 and rec.get("feats"):
            _mx = max(
                (a.get("loop_factor") or 1)
                for f in rec["feats"]
                for a in (f.get("args") or [])
            )
            if _mx <= 1:
                return False
    if rec.get("log_file") == _BUGGY_LOOP_FACTOR_LOG:
        for f in rec.get("feats") or []:
            names = [a.get("name", "") for a in f.get("args") or []]
            if any(
                "coarse_tile_fill" in n or "coarse_tile_combine" in n for n in names
            ):
                return False
    if op in _COARSE_OPS:
        for f in rec.get("feats") or []:
            if not f.get("is_matmul"):
                continue
            m, n = f.get("matmul_m_split"), f.get("matmul_n_split")
            if (m, n) in _EXCLUDED_SPLITS:
                return False
    return True


def _select_build(rows, spec):
    """Restrict `rows` to one measurement build, and say which builds are present.

    A build here is the ``model_sha`` stamped on a sweep log, which is the git sha of the
    tree that ran it. Measured ``kernel_us`` belongs to that build and to no other:
    kernel performance moves as the compiler develops, and pooling builds turns a
    compiler change into an apparent model error. The database has always spanned many
    builds, so ``all`` stays the default rather than silently redefining every published
    accuracy figure -- but it prints the census, so the pooling is never invisible.

    ``latest`` picks the newest build by log date, which is what a fresh sweep produces.
    """
    census = collections.Counter(r.get("model_sha") or "(no sha)" for r in rows)
    if spec == "all":
        if len(census) > 1:
            newest = max(
                (r.get("log_date") or "", r.get("model_sha") or "")
                for r in rows
                if r.get("model_sha")
            )
            print(
                f"NOTE: pooling {len(rows)} rows from {len(census)} measurement builds "
                f"(newest {newest[1]}, {census[newest[1]]} rows). "
                "Use --build latest to score one."
            )
        return rows

    if spec == "latest":
        dated = [r for r in rows if r.get("model_sha") and r.get("log_date")]
        if not dated:
            sys.exit(
                "no row carries both a model_sha and a log_date; --build latest "
                "needs a sweep log with a `git:` header"
            )
        sha = max(dated, key=lambda r: r["log_date"])["model_sha"]
    else:
        matches = {s for s in census if s.startswith(spec)}
        if len(matches) != 1:
            sys.exit(
                f"--build {spec!r} matches {len(matches)} builds. Present: "
                + ", ".join(f"{s}({n})" for s, n in census.most_common())
            )
        sha = matches.pop()

    kept = [r for r in rows if r.get("model_sha") == sha]
    print(f"build {sha}: {len(kept)} of {len(rows)} rows")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="")
    ap.add_argument("--all", action="store_true", help="all rows (default: is_current)")
    ap.add_argument("--category", default="", help="filter to one reporting category")
    ap.add_argument("--op", default="", help="filter to one op")
    ap.add_argument(
        "--params", default="", help="comma list of CostParams k=v overrides"
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="report feature-fidelity (pred vs stored pred_us) and exit",
    )
    ap.add_argument(
        "--update",
        action="store_true",
        help="write recomputed pred_us/err_pct back to the records",
    )
    ap.add_argument(
        "--list-worst", type=int, default=0, help="print the N worst |err| rows"
    )
    ap.add_argument(
        "--build",
        default="all",
        help="score one measurement build: a git sha prefix, `latest` (the newest "
        "sweep in the database), or `all` (default -- every build pooled)",
    )
    args = ap.parse_args()

    args.records = records_path(args.records or None)
    with open(args.records, encoding="utf-8") as f:
        records = json.load(f)["records"]
    default_params = cm.CostParams()
    overrides = [s for s in args.params.split(",") if s.strip()]
    params = make_params(overrides)

    rows = [r for r in records if not r.get("failed") and r.get("kernel_us")]
    rows = [r for r in rows if in_scope(r)]
    rows = _select_build(rows, args.build)
    if not args.all:
        # Score every row that CAN be scored against the current model: any row carrying
        # `feats` (its exact input, recomputed with the current model regardless of which
        # version logged it) OR an `is_current` row (reconstructed from its I/O block).
        # This spans measurement runs at different SHAs (feats are version-independent).
        rows = [r for r in rows if r.get("feats") or r.get("is_current")]
    if args.category:
        rows = [r for r in rows if category(r.get("op")) == args.category]
    if args.op:
        rows = [r for r in rows if r.get("op") == args.op]

    evaluated, skipped = [], {}
    for r in rows:
        ops, src = features_for(r, default_params)
        if ops is None:
            skipped[src] = skipped.get(src, 0) + 1
            continue
        pred = cm.predict_ops(ops, params) / 1000.0
        meas = r["kernel_us"]
        err = (pred - meas) / meas * 100.0
        evaluated.append((r, pred, meas, err, src))

    if (
        args.verify
    ):  # fidelity check: default-param pred should reproduce stored pred_us
        print(f"feature fidelity ({len(evaluated)} rows):")
        by_src = {}
        for r, _, _, _, src in evaluated:
            by_src[src] = by_src.get(src, 0) + 1
        for s, n in sorted(by_src.items()):
            print(f"  {n:4d}  {s}")
        for s, n in sorted(skipped.items()):
            print(f"  {n:4d}  SKIPPED: {s}")
        # for feats rows, also confirm pred(default) == stored pred_us
        drift = []
        for r, _, _, _, src in evaluated:
            if src != "feats" or not r.get("pred_us"):
                continue
            pd = cm.predict_ops(
                [cm.op_from_dict(d) for d in r["feats"]], default_params
            )
            pd /= 1000.0
            if abs(pd - r["pred_us"]) / r["pred_us"] > 0.02:
                drift.append((r["op"], round(pd, 1), r["pred_us"]))
        if drift:
            print(
                f"  WARNING: {len(drift)} feats rows drift from stored pred_us "
                f"(model changed since the log): e.g. {drift[:3]}"
            )
        return

    if overrides:
        print(f"params overrides: {overrides}")
    print(
        f"scored {len(evaluated)} rows"
        + (f"  (skipped: {dict(skipped)})" if skipped else "")
    )
    print(f"\n{'category':>16} {'n':>4} {'RMS%':>7} {'mean%':>7} {'range':>14}")
    cats = {}
    for r, pred, meas, err, src in evaluated:
        cats.setdefault(category(r["op"]), []).append(err)
    allerr = []
    for c in sorted(cats):
        es = cats[c]
        allerr += es
        rms = math.sqrt(sum(e * e for e in es) / len(es))
        print(
            f"{c:>16} {len(es):>4} {rms:>7.1f} {sum(es) / len(es):>+7.1f}"
            f"  [{min(es):+.0f}..{max(es):+.0f}]"
        )
    if allerr:
        rms = math.sqrt(sum(e * e for e in allerr) / len(allerr))
        print(
            f"{'OVERALL':>16} {len(allerr):>4} {rms:>7.1f} "
            f"{sum(allerr) / len(allerr):>+7.1f}  "
            f"[{min(allerr):+.0f}..{max(allerr):+.0f}]"
        )

    if args.list_worst:
        print(f"\nworst {args.list_worst} |err|:")
        for r, pred, meas, err, src in sorted(evaluated, key=lambda x: -abs(x[3]))[
            : args.list_worst
        ]:
            print(
                f"  {err:+6.1f}%  {r['op']:20} "
                f"[{r.get('rows')},{r.get('cols')}] "
                f"pred={pred:.1f} meas={meas:.1f} ({src})"
            )

    if args.update:
        idx = {r["id"]: (pred, err) for r, pred, _, err, _ in evaluated}
        for r in records:
            if r["id"] in idx:
                pred, err = idx[r["id"]]
                r["pred_us"], r["err_pct"] = round(pred, 3), round(err, 1)
        with open(args.records, "w", encoding="utf-8") as f:
            json.dump({"records": records}, f, indent=2)
        print(f"\nupdated pred_us/err_pct for {len(idx)} rows -> {args.records}")


if __name__ == "__main__":
    main()

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

The measured device time (``kernel_us``) does not depend on the model version and is
already stored in ``tools/cost_model/sweep_records.json``; only the model's PREDICTION
changes when we edit the model. (It does depend on the build that ran and on how the
harness timed it: see ``_HOST_MARKER_TIMING_SINCE``.)
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

``cost_model.py`` is imported through its package (it has package-relative imports since
#3810), with the Spyre backend autoload off: this needs torch_spyre's Python sources and
its dependencies, but no Spyre card or device runtime.

    python tools/cost_model/eval_model.py                         # is_current rows, current params
    python tools/cost_model/eval_model.py --all                   # every row (not just is_current)
    python tools/cost_model/eval_model.py --category matmul --op softmax_row_tiling
    python tools/cost_model/eval_model.py --params bw_peak_gbps=155
    python tools/cost_model/eval_model.py --verify                # feature-fidelity vs stored pred_us
    python tools/cost_model/eval_model.py --update                # write recomputed pred_us/err back
    python tools/cost_model/eval_model.py --plot /tmp/scatter.png # predicted-vs-measured PNG + MSE
    python tools/cost_model/eval_model.py --relayout              # LX relayout law vs the swap runs
    python tools/cost_model/eval_model.py --relayout --fit        # ... and refit a, b, K from them
"""

import argparse
import collections
import dataclasses
import importlib
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
    """Import this checkout's ``torch_spyre._inductor.cost_model``.

    Through the package, because the module has package-relative imports; loading the
    file on its own fails at import."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    return importlib.import_module("torch_spyre._inductor.cost_model")


if __name__ == "__main__":
    # The CLI scores records on any host, so keep torch from loading the Spyre backend
    # (which needs a card and its runtime) when cost_model imports torch_spyre. Only
    # here: importing this module must not change the importer's environment, which
    # every later subprocess of, e.g., a pytest session would inherit.
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
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
    "relayout": ["relayout"],  # whole fused bundle; the shuffle itself: --relayout
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
# 5. Rows timed with the host markers. Since #3424 (merged 2026-08-08) the runtime
#    records `launch::Compute` / `launch::D2H` / `launch::H2D` as CPU events, and the
#    profiler credits each with its device twin's time. `profile_ops` counted them as
#    kernel time until it learned to skip CPU events, so its `kernel_us` read about
#    twice the kernel plus the output read-back. A SUMMARY from the fixed harness
#    carries `timing=device`. An unmarked row is scored only if its log is dated before
#    #3424, when the markers did not exist; the reference database is all from then.
#    A row with no date at all predates the sweep runner (#3572, 2026-08-14), whose
#    header always writes one, so it is kept.
_HOST_MARKER_TIMING_SINCE = "20260808"
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


def _host_marker_timed(rec) -> bool:
    """Whether ``rec``'s ``kernel_us`` may include the launch::* host markers (scope
    decision 5)."""
    return (
        rec.get("timing") != "device"
        and (rec.get("log_date") or "") >= _HOST_MARKER_TIMING_SINCE
    )


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
    if _host_marker_timed(rec):
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


_LBL_SENCORES = re.compile(r"\bSENCORES=(\d+)\b")


def _relayout_sencores(rec):
    """The SENCORES budget a `relayout` run had: its ``sencores`` field, else the run
    header's ``SENCORES=`` (logs from before the field), else None. Not ``cores``,
    which is the widest division's core count, not the budget."""
    if isinstance(rec.get("sencores"), int):
        return rec["sencores"]
    if m := _LBL_SENCORES.search(rec.get("label") or ""):
        return int(m[1])
    return None


def _relayout_copy(rec):
    """The one LX relayout copy in a `relayout` record's features, or None."""
    copies = [
        o
        for o in (cm.op_from_dict(d) for d in rec.get("feats") or [])
        if o.is_lx_relayout
    ]
    return copies[0] if len(copies) == 1 else None


def _relayout_priceable(o):
    """Whether copy ``o`` carries the features the law needs. A copy without them --
    e.g. dumped by a build that predates them, so ``op_from_dict`` defaulted them to
    0 -- is one ``relayout_ns`` prices at 0 and the fit cannot use."""
    return (
        o.relayout_run_elems > 0
        and o.relayout_split > 0
        and (o.relayout_cores or o.cores) > 0
        and o.dtype_bytes > 0
    )


def _relayout_regressors(o, params):
    """The law's three regressors for copy ``o``, so that ``relayout_ns`` is
    ``a*x_a + b*x_b + x_k/K`` with ``K`` in GB/s (bytes per ns): ``x_a`` the per-core
    run count, ``x_b`` that times ``log2(split)``, ``x_k`` the strided span per core.
    The split is clamped into ``params``' range exactly as ``relayout_ns`` clamps it,
    so the fit is of the law the solver evaluates."""
    cores = o.relayout_cores or o.cores
    per_core = o.out_elems * o.dtype_bytes / cores
    runs = per_core / (o.relayout_run_elems * o.dtype_bytes)
    split = min(
        params.relayout_split_max, max(params.relayout_split_min, o.relayout_split)
    )
    return runs, runs * math.log2(split), per_core * split


#: The standard error of the median of n normal draws, over sigma / sqrt(n).
_MEDIAN_SE = math.sqrt(math.pi / 2)


def _pool(records):
    """One configuration's runs as ``(time_us, se_us, processes)``.

    Each record is one process: one compile and ``reps`` timed repetitions, of which
    ``kernel_us`` is the median. Several processes give the mean of their medians and
    its standard error from their spread, which includes everything that differs
    between processes (the compile, the solver's plan, the device's state). One
    process gives only the standard error of its median from the spread of its
    repetitions, a lower bound: it cannot see a difference between processes."""
    times = [r["kernel_us"] for r in records]
    n = len(times)
    if n >= 2:
        mean = sum(times) / n
        sd = math.sqrt(sum((t - mean) ** 2 for t in times) / (n - 1))
        return mean, sd / math.sqrt(n), n
    rec = records[0]
    reps = max(1, int(rec.get("reps") or 1))
    return times[0], _MEDIAN_SE * (rec.get("kernel_us_std") or 0.0) / math.sqrt(reps), 1


def _decision(runs, on_us, off_key, params):
    """Measured and predicted ``T_off - T_on`` for one direction, or None. ``on_us``
    is the pooled time of the `on` run, ``runs[off_key]`` the off run's records."""
    offs = runs.get(off_key)
    if not offs:
        return None
    on_key = (off_key[0], off_key[1], "on")
    on_ops = [cm.op_from_dict(d) for d in runs[on_key][0]["feats"]]
    off_ops = [cm.op_from_dict(d) for d in offs[0]["feats"]]
    return (
        _pool(offs)[0] - on_us,
        (cm.predict_ops(off_ops, params) - cm.predict_ops(on_ops, params)) / 1000,
    )


def _one_copy(records, params):
    """The cross run's copy, or a reason it has none usable: every repeat must carry
    exactly one copy, and the same one (the solver may plan differently between
    processes, and then the repeats are not one measurement)."""
    copies = [_relayout_copy(r) for r in records]
    if any(c is None for c in copies):
        return None, "pair(s) dropped: a cross run without exactly one copy"
    first = copies[0]
    shape = (first.relayout_split, _relayout_regressors(first, params))
    if any(
        (c.relayout_split, _relayout_regressors(c, params)) != shape for c in copies
    ):
        return None, "pair(s) dropped: the copy differs between repeats"
    if not _relayout_priceable(first):
        return None, "pair(s) dropped: a copy without the law's features"
    return first, ""


def relayout_pairs(rows, params, notes=None):
    """Measured vs predicted LX shuffle, one entry per swap pair.

    The `relayout` sweep measures each division pair X != Y of one [rows, cols] tensor
    four ways (see ``profile_ops._relayout_workload``); the producer and consumer work
    cancel in ``(T[X->Y] + T[Y->X] - T[X->X] - T[Y->Y]) / 2``, which leaves the mean of
    the two shuffles. The prediction is the mean of ``relayout_ns`` over the two copies'
    stored features under ``params``.

    Runs pair only within one build, shape, ``lx`` and SENCORES budget. A configuration
    measured in several processes there is pooled (``_pool``), so its repeats set the
    pair's ``noise_us``: the four runs' standard errors in quadrature, halved.
    ``repeats`` is the fewest processes behind any of the four.

    Also returned per pair, when an RL_MODE=off run of either direction exists: the
    decision quantity, measured ``T_off - T_on`` (each against the `on` run of its own
    direction) against the model's same difference -- what the layout solver weighs
    when it picks the relayout over spilling the edge.

    ``notes`` (a ``collections.Counter``), when passed, counts the pooled
    configurations and each pair dropped, by reason.
    """
    notes = collections.Counter() if notes is None else notes
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("op") != "relayout" or r.get("feats") is None:
            continue
        key = (
            r.get("model_sha"),
            r.get("rows"),
            r.get("cols"),
            r.get("lx"),
            _relayout_sencores(r),
        )
        groups[key][(r["rl_src"], r["rl_dst"], r["rl_mode"])].append(r)
    out = []
    for (sha, nrows, ncols, _, _), runs in sorted(groups.items(), key=str):
        for records in runs.values():
            if len(records) > 1:
                notes["configuration(s) measured in several processes: pooled"] += 1
        divs = sorted({d for src, dst, _ in runs for d in (src, dst)})
        for x in divs:
            for y in divs:
                if x >= y:
                    continue
                need = [(x, y, "on"), (y, x, "on"), (x, x, "on"), (y, y, "on")]
                if not all(k in runs for k in need):
                    continue
                (cxy, why_xy), (cyx, why_yx) = (
                    _one_copy(runs[k], params) for k in need[:2]
                )
                if cxy is None or cyx is None:
                    notes[why_xy or why_yx] += 1
                    continue
                (txy, sxy, nxy), (tyx, syx, nyx), (txx, sxx, nxx), (tyy, syy, nyy) = (
                    _pool(runs[k]) for k in need
                )
                pred = (
                    cm.relayout_ns(cxy, params) + cm.relayout_ns(cyx, params)
                ) / 2000
                pair = {
                    "sha": sha,
                    "shape": (nrows, ncols),
                    "x": x,
                    "y": y,
                    "split": max(cxy.relayout_split, cyx.relayout_split),
                    "run_elems": cxy.relayout_run_elems,
                    "bytes_per_core": cxy.out_elems
                    * cxy.dtype_bytes
                    / (cxy.relayout_cores or cxy.cores),
                    "meas_us": (txy + tyx - txx - tyy) / 2,
                    "pred_us": pred,
                    "regressors": [
                        (u + v) / 2
                        for u, v in zip(
                            _relayout_regressors(cxy, params),
                            _relayout_regressors(cyx, params),
                        )
                    ],
                    # Four independent estimates, halved: their errors add in
                    # quadrature, not linearly.
                    "noise_us": math.sqrt(sxy**2 + syx**2 + sxx**2 + syy**2) / 2,
                    "repeats": min(nxy, nyx, nxx, nyy),
                }
                decision = _decision(runs, txy, (x, y, "off"), params) or _decision(
                    runs, tyx, (y, x, "off"), params
                )
                if decision is not None:
                    pair["decision_meas_us"], pair["decision_pred_us"] = decision
                out.append(pair)
    return out


def _physical(fitted, p):
    """Why ``fitted`` is not a usable law, or "". The per-run cost
    ``a + b*log2(split)`` must not fall with the split (``b >= 0``) nor go negative
    anywhere in ``p``'s accepted split range, and the walk rate must be positive."""
    a, b, k = (
        fitted["relayout_run_a_ns"],
        fitted["relayout_run_b_ns"],
        fitted["relayout_span_gbps"],
    )
    why = []
    if b < 0:
        why.append("b < 0: the per-run cost falls as the split grows")
    for split in (p.relayout_split_min, p.relayout_split_max):
        if a + b * math.log2(split) < 0:
            why.append(f"the per-run cost is negative at split {split}")
    if not k > 0:
        why.append("K <= 0")
    return "; ".join(why)


#: ``fit_relayout`` weightings: the pairs each uses, and each pair's weight.
_FIT_WEIGHTS = {
    # Relative error: every pair counts alike whatever its size. Right only if the
    # noise grows with the shuffle, which the four-run estimate's does not.
    "relative": (lambda p: p["meas_us"] > 0, lambda p: 1 / p["meas_us"]),
    # Inverse noise: each pair counts by how well it is measured. A pair whose
    # shuffle came out <= 0 is still a measurement here, not a missing point.
    "noise": (lambda p: p["noise_us"] > 0, lambda p: 1 / p["noise_us"]),
}


def fit_relayout(pairs, weight="noise"):
    """Weighted least-squares ``(a, b, K)`` for the law over ``pairs``, each residual
    multiplied by ``_FIT_WEIGHTS[weight]``: 1/noise (default), so the fit's chi^2 is
    the usual one, or 1/measured. The law is linear in ``(a, b, 1/K)``, so this is one
    ``lstsq``, not an iteration.

    Returns ``(fitted, stderr)``: the three ``CostParams`` fields and their standard
    errors, scaled by the residual variance per degree of freedom so that a fit worse
    than the noise widens them rather than hiding it. Nothing here judges whether the
    result is physical; see ``_physical``."""
    import torch

    usable, w = _FIT_WEIGHTS[weight]
    use = [p for p in pairs if usable(p)]
    splits = sorted({p["split"] for p in use})
    if len(splits) < 2:
        # At one split log2(split) is a constant, so `a` and `b` are collinear and a
        # least-squares answer is one of infinitely many.
        raise ValueError(
            f"a fit needs pairs at two or more governing splits (have {splits})"
        )
    design = torch.tensor(
        [[x * w(p) / 1000 for x in p["regressors"]] for p in use],
        dtype=torch.float64,
    )
    target = torch.tensor([[p["meas_us"] * w(p)] for p in use], dtype=torch.float64)
    if int(torch.linalg.matrix_rank(design)) < 3:
        raise ValueError("singular fit: the pairs do not separate a, b and K")
    coef = torch.linalg.lstsq(design, target, driver="gelsd").solution.squeeze(1)
    a, b, inv_k = (float(c) for c in coef)
    residual = target.squeeze(1) - design @ coef
    dof = max(1, len(use) - 3)
    cov = torch.linalg.inv(design.T @ design) * float(residual @ residual) / dof
    se_a, se_b, se_inv_k = (float(v) ** 0.5 for v in torch.diagonal(cov))
    k = 1 / inv_k if inv_k != 0 else float("inf")
    fitted = {
        "relayout_run_a_ns": a,
        "relayout_run_b_ns": b,
        "relayout_span_gbps": k,
    }
    stderr = {
        "relayout_run_a_ns": se_a,
        "relayout_run_b_ns": se_b,
        # K = 1/inv_k, so its standard error is se(inv_k) / inv_k^2.
        "relayout_span_gbps": se_inv_k / inv_k**2 if inv_k != 0 else float("inf"),
    }
    return fitted, stderr


def _chi2(pairs, fitted_params=0):
    """``(chi^2 per degree of freedom, pairs within 2 sigma, pairs used)`` of the
    prediction against the measurement, in units of each pair's noise. About 1 when
    the law is right and the noise is all there is; ``fitted_params`` is how many
    of the law's constants were fitted to these same pairs."""
    use = [p for p in pairs if p["noise_us"] > 0]
    if not use:
        return float("nan"), 0, 0
    z = [(p["pred_us"] - p["meas_us"]) / p["noise_us"] for p in use]
    dof = max(1, len(use) - fitted_params)
    return sum(v * v for v in z) / dof, sum(abs(v) <= 2 for v in z), len(use)


def _err_summary(errs):
    if not errs:
        return "n=0"
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    return f"RMS {rms:5.1f}%  mean {sum(errs) / len(errs):+5.1f}%  n={len(errs)}"


def _pair_errs(pairs):
    """Percent errors of the pairs whose estimated shuffle is positive. The rest are
    estimator noise, not a measurement of the law, and are reported apart."""
    return [
        (p["pred_us"] - p["meas_us"]) / p["meas_us"] * 100
        for p in pairs
        if p["meas_us"] > 0
    ]


def _noise_summary(pairs, fitted_params=0):
    chi2, within, n = _chi2(pairs, fitted_params)
    return f"chi2/dof {chi2:.2f}, {within} of {n} within 2 sigma"


def relayout_report(rows, params, fit=False):
    """Print the relayout validation: the shuffle per pair, by split, and the decision
    quantity; with ``fit``, the refit parameters under each weighting and the pairs
    under them."""
    notes = collections.Counter()
    pairs = relayout_pairs(rows, params, notes)
    for why, n in sorted(notes.items()):
        print(f"{n} {why}")
    if not pairs:
        print("no complete relayout swap groups (need X->Y, Y->X, X->X and Y->Y runs)")
        return
    print(
        "LX relayout shuffle: (T[X->Y] + T[Y->X] - T[X->X] - T[Y->Y]) / 2 vs relayout_ns"
    )
    print(
        f"{'shape':>12} {'X':>6} {'Y':>9} {'split':>5} {'run':>6} {'KiB/core':>8} "
        f"{'n':>2} {'meas_us':>8} {'+-us':>5} {'pred_us':>8} {'err%':>7}  "
        f"{'decision m/p us':>16}"
    )
    by_split = collections.defaultdict(list)
    for p in sorted(pairs, key=lambda p: (p["shape"], p["split"])):
        by_split[p["split"]].append(p)
        err = (
            f"{(p['pred_us'] - p['meas_us']) / p['meas_us'] * 100:>+7.1f}"
            if p["meas_us"] > 0
            else f"{'noise':>7}"
        )
        dec = (
            f"{p['decision_meas_us']:7.1f}/{p['decision_pred_us']:7.1f}"
            if "decision_meas_us" in p
            else ""
        )
        print(
            f"{p['shape'][0]:>5}x{p['shape'][1]:<6} {p['x']:>6} {p['y']:>9} "
            f"{p['split']:>5} {p['run_elems']:>6} {p['bytes_per_core'] / 1024:>8.0f} "
            f"{p['repeats']:>2} {p['meas_us']:>8.2f} {p['noise_us']:>5.2f} "
            f"{p['pred_us']:>8.2f} {err}  {dec:>16}"
        )
    noise = sum(p["meas_us"] <= 0 for p in pairs)
    print(f"\nall pairs: {_err_summary(_pair_errs(pairs))}")
    if noise:
        print(f"  ({noise} pair(s) left out: estimated shuffle <= 0, below the noise)")
    for split in sorted(by_split):
        print(f"  split {split:>2}: {_err_summary(_pair_errs(by_split[split]))}")
    rel = sorted(p["noise_us"] / p["meas_us"] * 100 for p in pairs if p["meas_us"] > 0)
    if rel:
        print(f"noise: median {rel[len(rel) // 2]:.1f}% of the shuffle")
    print(f"law vs noise: {_noise_summary(pairs)}")
    if single := sum(p["repeats"] < 2 for p in pairs):
        print(
            f"  ({single} pair(s) rest on a run measured in one process: their noise "
            "is the within-process spread only, a lower bound)"
        )
    dec = [p for p in pairs if "decision_meas_us" in p]
    if dec:
        derr = [
            (p["decision_pred_us"] - p["decision_meas_us"])
            / p["decision_meas_us"]
            * 100
            for p in dec
            if p["decision_meas_us"] > 0
        ]
        lost = len(dec) - len(derr)
        wrong = sum(
            (p["decision_pred_us"] > 0) != (p["decision_meas_us"] > 0) for p in dec
        )
        print(
            f"decision (off - on): {_err_summary(derr)}; "
            f"model picks the wrong side on {wrong} of {len(dec)}"
        )
        if lost:
            print(
                f"  ({lost} pair(s) where the relayout measured no better than the "
                "spill are left out of the error but counted above)"
            )
    if not fit:
        return
    for weight in _FIT_WEIGHTS:
        try:
            fitted, stderr = fit_relayout(pairs, weight)
        except ValueError as exc:
            print(f"\nno refit ({weight} weights): {exc}")
            continue
        print(f"\nrefit ({weight} weights):")
        for k, v in fitted.items():
            print(
                f"  {k} = {v:.4g} +- {stderr[k]:.2g}  (current {getattr(params, k):.4g})"
            )
        if why := _physical(fitted, params):
            print(f"  WARNING: not a physical law ({why}); do not adopt it as is")
        refit = relayout_pairs(rows, dataclasses.replace(params, **fitted))
        print(
            f"  refit pairs: {_err_summary(_pair_errs(refit))}; "
            f"{_noise_summary(refit, fitted_params=3)}"
        )


def plot_scatter(evaluated, out_path):
    """Scatter plot of predicted vs. measured ``kernel_us``, one point per scored row,
    colored by reporting category, with the 1:1 line and the overall MSE/RMSE (in time
    units, not RMS% of relative error) annotated.

    This is a diagnostic companion to the RMS%-of-relative-error table ``main()`` prints:
    RMS% treats a 50% error the same whether the kernel is 10 us or 1 ms, which is right
    for ranking candidate plans against each other but hides whether a `cost_model.py`
    tweak's absolute error is concentrated in a few slow kernels or spread evenly. Compare
    this plot before/after a tweak (e.g. a `--params` override, or editing
    `torch_spyre/_inductor/cost_model.py` directly) to see which failure mode the tweak
    moves: points moving toward the 1:1 line is an improvement, points moving away in a
    specific category is a regression confined to that category, and a cloud that
    barely shifts is a no-op change. Requires matplotlib, which is NOT a torch-spyre
    dependency (base or optional) -- ``pip install matplotlib`` separately before using
    ``--plot``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("--plot requires matplotlib: pip install matplotlib")

    by_cat = {}
    for r, pred, meas, _err, _src in evaluated:
        by_cat.setdefault(category(r["op"]), []).append((pred, meas))

    fig, ax = plt.subplots(figsize=(7, 7))
    for c in sorted(by_cat):
        pts = by_cat[c]
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            label=f"{c} (n={len(pts)})",
            alpha=0.6,
            s=20,
        )

    all_meas = [meas for _r, _pred, meas, _err, _src in evaluated]
    all_pred = [pred for _r, pred, _meas, _err, _src in evaluated]
    lo = min(all_meas + all_pred)
    hi = max(all_meas + all_pred)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="pred = meas (1:1)")

    mse = sum((p - m) ** 2 for m, p in zip(all_meas, all_pred)) / len(all_meas)
    rmse = math.sqrt(mse)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted kernel_us")
    ax.set_ylabel("measured kernel_us")
    ax.set_title(
        f"measured vs. predicted ({len(evaluated)} rows)\nMSE={mse:,.0f} us^2  RMSE={rmse:,.1f} us"
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}  (MSE={mse:,.0f} us^2, RMSE={rmse:,.1f} us)")


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
        "--plot",
        default="",
        metavar="PATH",
        help="write a predicted-vs-measured scatter plot (PNG) to PATH and print "
        "its MSE/RMSE in time units, alongside the RMS%% table. Requires matplotlib.",
    )
    ap.add_argument(
        "--build",
        default="all",
        help="score one measurement build: a git sha prefix, `latest` (the newest "
        "sweep in the database), or `all` (default -- every build pooled)",
    )
    ap.add_argument(
        "--relayout",
        action="store_true",
        help="validate the LX relayout law against the `relayout` swap runs instead of "
        "scoring whole kernels (see relayout_pairs)",
    )
    ap.add_argument(
        "--fit",
        action="store_true",
        help="with --relayout: also refit relayout_run_a_ns, relayout_run_b_ns and "
        "relayout_span_gbps from the pairs and re-score under them",
    )
    args = ap.parse_args()

    args.records = records_path(args.records or None)
    with open(args.records, encoding="utf-8") as f:
        records = json.load(f)["records"]
    default_params = cm.CostParams()
    overrides = [s for s in args.params.split(",") if s.strip()]
    params = make_params(overrides)

    rows = [r for r in records if not r.get("failed") and r.get("kernel_us")]
    if n := sum(_host_marker_timed(r) for r in rows):
        print(
            f"NOTE: {n} rows left out: timed by a harness that counted the launch::* "
            "host markers (re-measure them; see _HOST_MARKER_TIMING_SINCE)"
        )
    rows = [r for r in rows if in_scope(r)]
    rows = _select_build(rows, args.build)
    if args.relayout:
        relayout_report(rows, params, fit=args.fit)
        return
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

    if args.plot:
        plot_scatter(evaluated, args.plot)

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

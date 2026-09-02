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

"""A simple, high-level analytical cost model for Spyre kernels.

Goal: predict *relative* device latency from the "after-pre-scheduling"
LoopLevel IR to guide higher-level optimization. Deliberately NOT a simulator.

Model (per fused bundle / single-op kernel):

    T   = max(compute, mem) + split                        mem = HBM / (eff * s_lx)

    HBM = [ (R+W)/BW + alpha*min(R,W) ] + spill + write_extra
    s_lx = min(1, (512KB/ws)**0.15)   for a coarse-tiled kernel with ws > 512KB   (else 1)

  where R = HBM bytes READ (inputs), W = HBM bytes WRITTEN (outputs). LX-resident
  traffic is treated as ~free. ``eff`` (<=1) derates the MEMORY term for OUTPUT-dim
  (pointwise) coarse-tiling that shrinks each core's per-tile height; ``s_lx`` (<=1) further
  derates it when the per-core working set overflows LX (spilled traffic runs slower).
  ``compute`` is nonzero only for matmul (see below). A genuine-reduction cross-core ring term,
  and a per-iteration coarse-tiling loop overhead (c_loop*L), once lived here but are dropped
  (the ring is <=~5ns sub-noise; c_loop had no current op to validate it). For a normal untiled
  non-matmul kernel eff = 1, s_lx = 1 and compute = 0, so this reduces to the bandwidth model.

- ``fill`` ~= 0: the golden kernel has no fixed term (section-A intercept ~0; the old
  ~20us "fixed" was a separate non-deterministic Memset/host-setup bucket, not kernel).
- ``BW_PEAK`` ~150 GB/s (== bytes/ns) is the PEAK HBM bandwidth, reached when traffic is
  one-directional (read-only or write-only). HBM is a shared bus that must turn around
  between reading and writing, so a kernel doing BOTH pays a penalty on the overlap.
- ``alpha * min(R, W)`` is that read/write turnaround penalty. min(R,W) is the overlap:
  0 for pure read or pure write (no switching), maximal at a balanced 1R+1W. This gives
  the measured V-shaped effective BW -- ~150 read-only, dipping to ~105 at balanced,
  back up for write-only -- with one extra constant instead of a second bandwidth.
  Verified on the B-F profiler sweep: ~2% error on core pointwise + reductions, ~7%
  overall (turnaround) vs ~11% for an additive two-rate. Using a single aggregate
  BW_PEAK is the "shared HBM" assumption (rung-5: core-independent for >=2 cores).
- memory traffic counts each tensor-arg's bytes once, attributed to HBM or LX by
  its allocation. LX-placed tensors don't touch HBM, and their LX traffic is treated
  as ~free (the measured per-pass LX cost is below run-to-run noise). Broadcast inputs
  are loaded ONCE and reused across the broadcast dim, so they are counted at their own
  (one-row/-col) DEVICE size -- NOT scaled up to the output size (the rung-6 runs proved
  a core does not re-read the operand per output element), but NOT dropped to zero
  either (it is still a real one-time load). That one load is tiny vs the output, so the
  bcast/mulbcast runs still land ~on the 2-pass latency. They are flagged (``broadcast``
  in :class:`ArgTraffic`) for visibility. The "once, not per core" count is VERIFIED on
  device (rung-G reload probe, cores=32, R=64): bcast (b[1,C]) ~= bcastcol (b[R,1]) ~=
  30-33us, both far below the full 3-pass add (52us). A per-core reload would have added
  ~cores*C and pushed bcast up toward add; it did not -- so the operand costs a single
  load regardless of how the work splits across cores.

Byte counts use each arg's DEVICE layout (stick-padded ``device_size``), not the torch
logical shape -- so a reduction's reduced input is naturally full-sized and stick
rounding is captured. REDUCTIONS are a tiny WRITE + a full READ, so they run at a read
rate. That rate is NOT flat: it falls with ROWS (op-independent -- read/sum/amax/mean/
sumall collapse to one curve), ``reduction_read_bw = min(150, 114+61*exp(-ROWS/3700))``,
applied to a STANDALONE row-reduction (single op; a fused softmax stays on bw_peak so its
input dedup is not broken; ``sumcol`` uses reduce_outer). (A cross-core ring-combine term
for a split reduced axis was dropped as sub-noise -- provably <=~5ns on us kernels.)

MATMUL (reduction_type batchmatmul) is priced by one of two independent
implementations, switched on ``CostParams.use_bundled_cost_model``:

- UPSTREAM (``use_bundled_cost_model=False``): the compute/split-shape part of
  ``work_division._matmul_split_cost`` -- the same heuristic the work-division planner
  uses to choose a matmul's core split, called with ``include_hbm=False``. Its own
  HBM-traffic term is dropped because the bundle memory term below already charges the
  operand/output bytes, and does so LX-aware; charging both double-counts memory.

- BUNDLED (``use_bundled_cost_model=True``): the original device-calibrated model,
  kept alongside the above rather than deleted. Adds a compute term that OVERLAPS the
  HBM term: the engine streams operands while the array works, so the kernel takes the
  LONGER of the two, not their sum: ``T = max(compute, HBM)``, with
  ``compute = MACs / cores / (mac_peak * pt_eff)`` (MACs = M*N*K, mac_peak=1140
  SUSTAINED). The matmul HBM uses a SINGLE rate = the copy peak (150) plus an operand
  RE-READ "tile spill": ``(|A|+|B|)*f(area)`` at that rate, f a saturating log of the
  per-core output-tile area (M/m)*(N/n) past the on-chip capacity ~64K elems). A
  default-layout BMM (both rank-3 operands on the compiler-default device tile order)
  runs the systolic array at a slower calibrated rate; see ``_matmul_mac_peak``. See
  ``_matmul_ns_bundled`` and the individual ``CostParams`` fields (``mac_peak_per_core_ns``,
  ``mm_spill_*``, ``mm_split_*``, ``mm_bw_*``, ``bmm_*``) for the fitted constants.

COARSE-TILING (fused kernel, e.g. ``softmax_row_tiling``): a coarse-tiled op is ONE fused
kernel with intermediates kept in LX -- NOT a sum of per-op kernels. Two things follow.
(1) HBM = distinct EXTERNAL inputs ONCE + outputs once (``_fused_hbm_bytes``): softmax
reads ``arg0`` in both ``amax`` and ``sub``, but the fused kernel loads it from HBM once
and serves the 2nd read on-chip -- the naive per-op sum double-counts it (~+25% at the
floor; at the floor softmax runs at ~100 GB/s = the balanced-copy rate, i.e. 1 read +
1 write, confirming the once-count). (2) UNDERFILL: tiling an output dim cuts each core's
per-tile SIZE -- rpc = ROWS/(cores*tiles) rows of COLS -- and a small tile underfills the
streaming pipeline. ``coarse_underfill_eff = min(cap, (rpc/r_full)**exp *
(COLS/col_ref)**col_exp)`` (cap~1.08, r_full~7.9 at col_ref 2048, exp~0.50, col_exp~0.38),
SEPARATE from the matmul ``pt_eff``. Re-fit 2026-08-11 on 48 tiled softmax rows at
cores=32 spanning rpc 2..256 AND COLS 128..4096; the width term is that re-fit's finding
and REPLACES the earlier rows-only reading, whose cross-COLS control (2048 vs 4096) sat
entirely inside the plateau and so could not see a width dependence. It remains a
tile-SIZE effect, not a tile-COUNT one (four T=4..32 points at rpc=16 cost the same).
KNOWN residuals: at cores 8 and 16 the memory term has no core-count scaling at all, so a
coarse-tiled softmax there is UNDER-predicted by 30-50% (identified and controlled --
matched tile shape, tile count ruled out -- but deliberately unmodeled; see the report);
and when the per-core WORKING SET overflows the
practically available LX (~512 KB/core) the intermediates SPILL to HBM. The extractor already
counts those spilled bytes as HBM (they show up read+written), so it is NOT a byte miss -- but
that spilled traffic runs SLOWER than the modeled rate, so the effective bandwidth is DERATED
(``_lx_spill_bw_derate``: BW *= (cap/ws)**0.15 for ws>cap). Softmax-calibrated
(10.7%->6.0% RMS on softmax_row_tiling, 32 cores, tiles>=2, n=60),
gated to non-matmul coarse tiling. See report S14.

ACCESS-PATTERN effective-BW overrides (db_sweep; fold turnaround into one measured rate).
STICK-PLANE TRANSPORTS -- a `clone` reorganizing the stick layout, parallelized by splitting
the stick-plane dim sp = C/64 across the 32 cores. Effective BW falls with the per-row
strided stick gather -- more planes (sp) and a longer gather stride (more rows R) -- so cat0,
transpose_outer and cat1 SHARE one form, clamp(a - b*log2(sp) - d*log2(R), floor, bw_peak),
with per-op constants (``tx_*``): cat0 a steep untiled gather, transpose_outer a gentle tiled
block-transpose (calibrated at its best middle dim M=8; M!=8 flagged), cat1 nearly flat
(copies stored outermost -> contiguous). ``transpose`` is a hardware restickify (stick
swapped) -> flat ``bw_restickify_gbps`` ~116; "reduce_outer" sumcol (cross-row reduce) ~113.
cat0/transpose carry an ``OpFeatures.hbm_pattern`` from the extractor; cat1/transpose_outer
are detected structurally from the logical reshape (``_transport_kind``). (A dependent multi-op
chain like add3/add4 costs more than its byte count -- a read-after-write ACROSS op
boundaries: a program-level effect, not a single-op cost, so NOT modeled here.) Transports land
within ~6%. BROADCAST-operand ops (copy/bcast/bcastcol/mulbcast: a full input + a small
broadcast operand) run ~118 GB/s -- ``bw_broadcast_gbps`` (mechanism open). ``write``
(b[1,C]+c[R,1], BOTH operands broadcast -> outer-product) is slow + super-linear; modeled by an
KNOWN residual (not yet modeled): transpose_outer away from M=8 (peaks at M~8); reductions
~-15% at large ROWS (stick-inflated scattered output write).

CALIBRATION NOTE: the golden per-op measurement is the torch.profiler "Self SPYRE"
(sdsc_fused) KERNEL device time -- NOT our old SPYRE_PROFILE_SYNC min (which folded in a
non-deterministic Memset/host-setup bucket, the source of the obsolete ~20us fill).

Parameters live in :class:`CostParams`, calibrated from device measurements
(``examples/run_cost_model_plan.sh``).
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence

import sympy

from .work_division import _matmul_split_cost, min, max, log2
from . import config


@dataclasses.dataclass
class ArgTraffic:
    """Traffic for one tensor argument of an op."""

    name: str
    role: str  # "input" | "output"
    is_lx: bool
    elems: int  # device element count = prod(dims) (its own one-load size)
    broadcast: bool = False  # loaded once & reused across the broadcast dim
    # DEVICE (stick) shape, e.g. [4, 512, 64]
    dims: list = dataclasses.field(default_factory=list)
    # LOGICAL torch shape, e.g. [512, 1024] -- shown next to dims so the stickification
    # (a row of N rounds up to ceil(N/64)*64 sticks) is visible per tensor.
    logical: list = dataclasses.field(default_factory=list)
    # Coarse-tiling loop multiplier on this arg's bytes. 1 for a normal arg or an
    # ADVANCING tiled arg (it walks the full tensor once across the loop, so its full
    # device_size already covers all tiles). L (= loop trip count) for a FIXED arg held
    # at one address across the loop (a per-tile accumulator re-read/written each
    # iteration). LX-resident args are ~free regardless (excluded from read/write).
    loop_factor: int = 1

    @property
    def mem(self) -> str:
        if isinstance(self.is_lx, bool):
            return "lx" if self.is_lx else "hbm"
        raise ValueError("Symbolic is_lx not supported for this operation")


@dataclasses.dataclass
class OpFeatures:
    """Cost-relevant features of one LoopLevel-IR op."""

    name: str  # origin op name (e.g. "gelu", "mul", "sub")
    is_reduction: bool
    out_elems: int
    cores: int
    dtype_bytes: int
    args: list  # list[ArgTraffic]
    reduction_cores: int = 1  # cores splitting the REDUCED axis (1 = none → no combine)
    loop_trip: int = 1  # coarse-tiling loop trip count for this op (prod of loop_count)
    # OUTPUT-dim (pointwise) coarse-tiling: True when this op tiles an output dim, so
    # each core's per-tile height shrinks and the underfill derate applies. False for
    # reduction-dim tiling and for untiled ops.
    tiles_output_dim: bool = False
    # True when the coarse loop tiles the REDUCTION dim. Recorded because it is what
    # decides whether the extractor's raw `out_elems * k_size` is a per-iteration slice
    # or the whole-loop total; `matmul_macs` is now always the TOTAL, so nothing here
    # multiplies by `loop_trip`. Records written before that fix carry per-tile macs for
    # reduction-tiled ops and will under-count compute -- they also lack this field, so
    # `tiles_reduction_dim=False` on a coarse op with `loop_trip>1` marks a stale row.
    tiles_reduction_dim: bool = False
    # Per-core per-tile pass-row height (output-tiled ops only): the streamed tile's
    # "rows" / cores. Drives ``eff_underfill``; 0.0 = unknown / not applicable -> no
    # derate.
    tile_rows_per_core: float = 0.0
    # MATMUL (reduction_type batchmatmul): adds an ADDITIVE compute term. matmul_macs =
    # M*N*K (total multiply-accumulates); matmul_rows_per_core = M/m (per-core M tile,
    # drives pt_eff). K-split k is carried in ``reduction_cores`` (-> the combine/PSUM
    # term). All zero/False for non-matmul ops.
    is_matmul: bool = False
    matmul_macs: int = 0
    matmul_rows_per_core: float = 0.0  # M/m (per-core A tile height -> A re-read)
    matmul_cols_per_core: float = 0.0  # N/n (per-core B tile width  -> B re-read)
    matmul_a_bytes: int = 0  # |A| = M*K device bytes (re-read scales with M/m)
    matmul_b_bytes: int = 0  # |B| = K*N device bytes (re-read scales with N/n)
    # Per-dim core-split counts (m along M, n along N). Recorded so a split-shape term
    # can key on the raw fanout / arrangement, not just the area (M/m)*(N/n). 1 =
    # unsplit.
    matmul_m_split: int = 1
    matmul_n_split: int = 1
    # Access-pattern HBM effective-BW override (from the LoopLevel IR index/layout):
    # "restickify" (transpose: write-stick var read with coeff!=1), "stick_scatter"
    # (cat on a partition dim -> a device dim <64 just inside the stick), "reduce_outer"
    # (cross-row reduction: reduced var read with coeff!=1). "" -> default
    # 150+turnaround.
    hbm_pattern: str = ""
    # LX RELAYOUT (PR #3439): an identity copy the scratchpad planner inserts when a
    # producer and its consumers own an LX buffer under different per-core divisions.
    # Its traffic is entirely LX, which this model charges at zero -- calibrated for
    # ops whose LX traffic rides beside a larger HBM stream, and exactly wrong for the
    # one op that is nothing but LX traffic. Measured on 53 configurations it costs
    # 2.3-24.2 us at 2.1 MB / 8 cores depending on slice geometry alone, so it gets an
    # additive term (see ``relayout_ns`` in predict_ops) keyed on these features.
    # ``is_lx_relayout`` derives from the planner's materialization registry, never
    # inferred; the other two are 0 when it is False. The bytes moved are NOT a
    # separate feature: a relayout copy is an identity clone, so they are exactly
    # ``out_elems * dtype_bytes`` (grouped gathers (#3440) will multiply by
    # ``cores / owners`` via a future ``relayout_owners`` feature, not by a bytes
    # field).
    is_lx_relayout: bool = False
    # Per-core contiguous run, in ELEMENTS, of the finer (governing) of the two
    # PerCoreViews: (device_size[d] // split[d]) * prod(device_size[d+1:]) for the
    # innermost split dim d. The 10x-at-fixed-bytes variable.
    relayout_run_elems: int = 0
    # Split factor of the governing side's innermost split dim. Sets the walked-span
    # multiplier: the engine streams the whole strided span at ~K GB/s per core and a
    # core keeps 1/split of it (BW*split measured constant at 588/540/549).
    relayout_split: int = 0

    def read_bytes(self) -> int:
        """HBM bytes READ (input args). Each HBM arg is counted at its own device size,
        scaled by ``loop_factor`` (L for a per-tile accumulator re-read every iteration,
        1 for an advancing tiled arg or a normal arg). A broadcast operand carries its
        real (one-row/-col) ``elems`` -- loaded once, NOT scaled to the output.
        """
        return (
            sum(
                a.elems * a.loop_factor * (1 - a.is_lx)
                for a in self.args
                if a.role == "input"
            )
            * self.dtype_bytes
        )

    def write_bytes(self) -> int:
        """HBM bytes WRITTEN (output args), scaled by ``loop_factor``."""
        return (
            sum(
                a.elems * a.loop_factor * (1 - a.is_lx)
                for a in self.args
                if a.role == "output"
            )
            * self.dtype_bytes
        )

    def hbm_bytes(self) -> int:
        """Total HBM traffic = read + write (kept for the dump / LAST_IO totals)."""
        return self.read_bytes() + self.write_bytes()

    def lx_bytes(self) -> int:
        return sum(a.elems * a.is_lx for a in self.args) * self.dtype_bytes


def op_to_dict(op: "OpFeatures") -> dict:
    """Serialize one OpFeatures (incl. its ArgTraffic list) to a plain JSON-able dict.

    This is the model's INPUT feature vector. Dumped next to the measured kernel time so
    a NEW model version can be scored OFFLINE (predict_ops on the stored features) without
    re-running on hardware -- the measurement is version-independent, only the prediction
    changes. See tools/cost_model/eval_model.py.
    """
    return dataclasses.asdict(op)


def op_from_dict(d: dict) -> "OpFeatures":
    """Rebuild an OpFeatures from :func:`op_to_dict` output. Robust to schema drift:
    unknown keys are ignored and missing ones fall back to the dataclass defaults, so an
    old dataset still loads against a newer OpFeatures/ArgTraffic definition."""
    afields = {f.name for f in dataclasses.fields(ArgTraffic)}
    args = []
    for a in d.get("args", []):
        kw = {k: v for k, v in a.items() if k in afields}
        # Datasets captured before `mem` became the derived `is_lx` carry the string.
        if "is_lx" not in kw and "mem" in a:
            kw["is_lx"] = str(a["mem"]).lower() == "lx"
        args.append(ArgTraffic(**kw))
    ofields = {f.name for f in dataclasses.fields(OpFeatures)}
    kw = {k: v for k, v in d.items() if k in ofields and k != "args"}
    return OpFeatures(args=args, **kw)


def _jsonable(o):
    """Fallback encoder: coerce non-native leaves (sympy ``Integer``/``Float`` from the
    inductor size expressions, numpy scalars, …) to plain int/float so ``json`` accepts
    them; anything else falls back to ``str``."""
    try:
        f = float(o)
    except (TypeError, ValueError):
        return str(o)
    return int(f) if f.is_integer() else f


def ops_to_json(ops: list) -> str:
    """Serialize a fused bundle (list of OpFeatures) to a single JSON string. Sizes coming
    off the IR are often sympy ``Integer``, so a ``default`` coercer is required."""
    import json

    return json.dumps(
        [op_to_dict(o) for o in ops], separators=(",", ":"), default=_jsonable
    )


def ops_from_json(s: str) -> list:
    """Deserialize a bundle serialized by :func:`ops_to_json`."""
    import json

    return [op_from_dict(d) for d in json.loads(s)]


@dataclasses.dataclass
class CostParams:
    """Fittable parameters for ``T = fill + (R+W)/BW_PEAK + alpha*min(R,W)``.

    Predicts the GOLDEN per-kernel device time (torch.profiler "Self SPYRE"). Fitted on
    the B-F profiler sweep (examples/run_profile_sweep.sh, fp16):
    - fill ~0       -- no fixed kernel term (section A intercept ~0).
    - BW_PEAK ~150 GB/s (== bytes/ns) -- the one-directional peak; read-only reductions
      and read probes land at ~145-155.
    - alpha ~0.00574 ns/byte -- the read/write turnaround penalty, calibrated so a
      balanced 1R+1W neg (R=W) lands at its measured ~105 GB/s effective:
      2/(2/BW_PEAK + alpha) = 105. min(R,W) is the read/write overlap (0 for
      one-directional traffic, maximal at balanced) -> reproduces the V-shaped
      effective BW. ~2% error on core ops, ~7% overall (see module docstring biases).
    LX traffic ~FREE (rung-4 below noise). Verified: arithmetic-free (gelu/exp == neg);
    broadcast operand loaded ONCE (rung-G, not per core); HBM BW shared / core-
    independent >=2 cores (rung 5).
    """

    fill_ns: float = 0.0  # golden kernel has ~no fixed term (section A: intercept ~0)
    bw_peak_gbps: float = 150.0  # one-directional peak HBM BW (read-only / write-only)
    # Read/write turnaround penalty (ns per overlapping byte). HBM is a shared bus that
    # must switch between read and write; the cost falls on the overlap min(R,W). Solved
    # from balanced neg (eff 105): alpha = 2/105 - 2/BW_PEAK.
    rw_turnaround_ns_per_byte: float = 0.00574
    # Genuine-reduction cross-core ring combine: (k-1) hops each touching every output
    # element. Fires ONLY for real reductions (NOT matmul -- gated in predict_ops), and
    # only when out_elems<cores, so it is bounded by ~cores*psum (<=~4.5ns) --
    # effectively inert (kept for structure). The matmul K-split PSUM ring is
    # deliberately NOT modeled: the planner keeps K whole (WD_K=1), and forcing WD_K>1
    # made this term explode (+489%). (A per-iteration coarse-tiling loop overhead
    # c_loop*L was removed: it was calibrated on the dropped chain/ctsum reduction-dim
    # sweeps and no current op exercises it.) Pipeline-fill (underfill) derate for
    # OUTPUT-dim (pointwise) coarse-tiling: eff = min(1, (rows_per_core / (pass_rows *
    # target_passes)) ** exponent). Same FORM as the matmul pt_eff (work_division.py);
    # the 8-row pass is the shared hardware constant, target_passes differs by op
    # structure. PROVISIONAL -- guessed from the chain K-sweep (flat to ~16 rows/core,
    # cliff at 8; data hints exponent ~0.4). To be calibrated by the untiled-small-ROWS
    # underfill-confirm runs. Only wired up for the BUNDLED matmul pt_eff
    # (``use_bundled_cost_model``, see the module docstring) -- it was never used for
    # anything else.
    underfill_pass_rows: float = 8.0  # PT / stream pass granularity (matmul _PT_ROWS)
    underfill_target_passes_pointwise: float = 2.0  # pointwise full-fill ~2 pass (=16)
    # Falloff exponent. CALIBRATED 0.35 from the Section-B chain sweep (rc 16->2): eff
    # rc8=0.72 rc4=0.63 rc2=0.50 (sqrt over-derated rc2 to 0.35). rc4/rc2 imply ~0.335;
    # rc8 is slightly concave (~9% residual). rc1 is NOT underfill -- there the planner
    # splits COLUMNS not rows, so row_split=1 -> full row tile (handled by the extractor
    # keying tile_rows_per_core on the row-dim split, not total cores).
    underfill_exponent: float = 0.35
    # COARSE-TILING (fused pointwise / softmax) underfill -- SEPARATE from the matmul
    # pt_eff above. RE-FIT 2026-08-11 as a TWO-variable surface on 48 tiled
    # softmax_row_tiling rows at cores=32 (h = ROWS/(cores*T) = 2..256, COLS =
    # 128..4096), backing the derate out of the measurement with the LX-spill term also
    # disabled so the two are not entangled:
    #
    #   eff = min(cap, (h/r_full)**exp * (COLS/col_ref)**col_exp)
    #
    # WHY IT GAINED A WIDTH. The previous single-variable fit keyed on ROWS alone, on
    # the strength of a cross-COLS control at COLS 2048 vs 4096. That control was blind,
    # not wrong: at both those widths the required efficiency is already at the plateau
    # for every h it sampled, so it could not see a width term. Sweeping down to COLS
    # 128 shows the required efficiency rising steadily with width at fixed h (at h=4:
    # 0.18 -> 0.32 -> 0.49 -> 0.68 for COLS 128/256/512/2048), i.e. a narrow tensor
    # needs a much taller tile to fill the pipeline. The fitted exponents (0.50 on h,
    # 0.38 on COLS) say the governing quantity is close to the per-core TILE SIZE h*COLS
    # -- the bytes each core streams per tile -- with a weak residual preference for
    # taller tiles. That is the opposite of the retired conclusion, and it is the
    # reading the mechanism (a stream too short to amortise pipeline fill) predicted all
    # along.
    #
    # RMS 9.0 % in log units over the 48 fitted rows, 4.8 % over the 29 with COLS >=
    # 1024 (the in-scope width band -- see eval_model.in_scope). The width evidence
    # itself comes from COLS 128..512, which the standing scope decision puts OUT of the
    # scored population: those rows fix the shape but are not part of any accuracy
    # claim.
    coarse_underfill_rfull: float = 7.9  # rows/core at full fill, AT col_ref
    coarse_underfill_exp: float = 0.50  # tile-HEIGHT exponent
    coarse_underfill_col_ref: float = 2048.0  # width r_full is quoted at
    coarse_underfill_col_exp: float = 0.38  # tensor-WIDTH exponent
    # PLATEAU. 1.08, not 1.0: it is a fitted RATIO, not a physical efficiency. On this
    # build a coarse-tiled fused softmax with a large per-core tile runs ~8 % FASTER
    # than the modeled memory term, so the surface has to be able to speed the model up.
    # The offset is fused-softmax-specific (pointwise, which sets the base rate, runs -6
    # % mean the other way), so it is absorbed here rather than in bw_peak_gbps -- but
    # it is DOUBLE DUTY, and the deciding experiment is an untiled COLS sweep at
    # cores=32: the three untiled in-scope rows we have disagree (+22 %, -14 %, -14 %),
    # too few to separate a base-rate offset from a tiling one. Weakly identified:
    # 1.06..1.11 span 0.1 pp of RMS.
    coarse_underfill_cap: float = 1.08
    # MATMUL keeps the PRE-2026-08-11 rows-only curve, frozen. Every row behind the new
    # surface is a softmax row; a coarse-tiled matmul was always priced by extrapolating
    # the softmax fit onto it, and nothing in the re-sweep says the new shape transfers.
    # The one piece of evidence available says it does not: swapping tiled matmul onto
    # the new surface moves matmul_row from 38.3 to 41.7 % RMS (mean -36.4 -> -39.8),
    # because the shallower height curve stops derating its short tiles. That is weak
    # evidence -- matmul_row under-predicts by a third for reasons unrelated to this
    # term -- but it points the wrong way, so the matmul branch does not move. Re-fit it
    # when a coarse matmul tile-count sweep exists. See `coarse_underfill_eff_matmul`.
    coarse_underfill_rfull_matmul: float = 13.0
    coarse_underfill_exp_matmul: float = 0.68
    coarse_underfill_cap_matmul: float = 0.95
    #: Smallest per-core tile height the MATMUL curve was fitted at (see `_eff0` sibling).
    coarse_underfill_h0_matmul: float = 2.0
    # LX-SPILL bandwidth derate. When a coarse-tiled kernel's per-core working set (its
    # live intermediate tiles, ~2 of them) overflows the practically available LX (~512
    # KB/core), those intermediates spill to HBM. The extractor ALREADY counts the
    # spilled bytes as HBM (they show up read+written), so this is NOT a byte miss --
    # but that spilled traffic runs SLOWER than the modeled rate, so the effective
    # bandwidth is derated: BW *= min(1, (lx_spill_cap / ws_per_core) ** lx_spill_exp),
    # ws > cap only. NOTE 512 KB is NOT a documented capacity -- this repo documents 2
    # MB physical and ~1.6 MB allocatable per core. The value is fitted; do not describe
    # it as matching the hardware.
    #
    # EXPONENT RE-FIT 2026-08-11, 0.15 -> 0.06, JOINTLY with the coarse-underfill
    # surface above. The two act on the same rows and both fall off with the per-core
    # tile, so fitting either alone hands it the other's error: the 0.15 fit was taken
    # with the underfill surface keyed on rows only, and on the current build it
    # over-derated the large-tile end by up to 24 % where the measurement wants ~7 %.
    # The decline itself is real and survives the joint fit -- D falls from 1.13 at ws
    # 256 KB to 0.98 at 3 MB -- it is just much milder than the old value. Weakly
    # identified (0.03..0.09 spans 0.07 pp of RMS); the CAP is barely identified at all
    # (256 KB..512 KB spans 0.02 pp), so it is left where it was.
    lx_spill_cap_bytes: float = 524288.0  # ~512 KB/core practically available LX
    lx_spill_exp: float = 0.06
    # Matmul-specific spill calibration (see _lx_spill_bw_derate). Defaults make the
    # term fire only on the largest matmul tiles (it acts on 19 in-scope rows). BUNDLED
    # matmul path only (see the module docstring).
    # The cap is only weakly identified: 1 MB..infinity spans 0.44 pp of RMS.
    # 2 MB = the FULL per-core LX. softmax's practical limit is lower (512 KB) because a
    # fused reduction holds several live intermediates at once, whereas a coarse matmul
    # tile is a single working set and can use the whole scratchpad before it spills.
    # The EXPONENT is unchanged from the softmax fit (0.15) -- same physical form, only
    # the capacity threshold differs.
    mm_spill_ws_cap_bytes: float = 2097152.0
    mm_spill_ws_exp: float = 0.15
    # Switch between the two matmul cost implementations (see the module docstring).
    # True (default) -- matmul uses the original device-calibrated
    # compute/HBM/spill/split-shape model below (``_matmul_ns_bundled``).
    # False -- delegates matmul entirely to ``work_division._matmul_split_cost``.
    use_bundled_cost_model: bool = True
    # MATMUL compute term. T_matmul = max(compute, HBM), where
    # compute = MACs/cores/(mac_peak*pt_eff). mac_peak=1140 (sustained) fit on the
    # compute-DOMINANT low-core runs (cores 4-8, compute 80-90% of the kernel; the old
    # 1536 datasheet was ~33% optimistic). A single peak over-predicts cores=32 -- the
    # RMS 1.7% across cores 4->32. pt_eff reuses the underfill derate (~1 for M/m>=64).
    # BUNDLED matmul path only.
    mac_peak_per_core_ns: float = 1140.0  # MAC/ns/core (sustained; compute-isolate fit)
    underfill_target_passes_matmul: float = 8.0  # matmul full-fill ~8 passes (=64 rows)
    # Scale on the loop-invariant re-read. 1.0 = a full HBM pass per iteration. Fitted
    # JOINTLY with the overlap shape because the two are confounded (r = -0.90 between
    # rho and the re-read share): an adversarial review showed 26/125 rows demanding an
    # overlap fraction > 1.0, which is physically impossible and means the re-read is
    # over-charged rather than the overlap under-modelled.
    loop_reread_scale: float = 0.85
    overlap_gamma: float = 1.0  # compute/HBM overlap: min(compute,HBM) partly hidden
    # LX RELAYOUT (shuffle) term: per-core serial descriptor cost plus a stride-limited
    # walk. Fitted 2026-08-17 on 21 direct per-kernel rows (main @ 65508a02, fp16,
    # BUNDLE_SYMBOLIC_ARGS=0 -- a method #3741 has since removed; re-measuring needs
    # the fused swap estimator at ~+/-8%): mean +2.5%, RMS 10.0% over bytes 0.52-4.19
    # MB, cores 4-32, run 128-4096 B, split 2-8.
    #
    #   relayout_ns = runs_per_core * (a + b*log2(split)) + span_per_core / K
    #     runs_per_core = bytes/cores / run_bytes
    #     span_per_core = bytes/cores * split   (the engine walks the whole strided
    #                                            span; a core keeps 1/split of it --
    #                                            BW*split measured 588/540/549, so K
    #                                            is per-core and split-invariant)
    #
    # The log2 in the per-run cost is FITTED, no mechanism behind it. VALID FOR SPLITS
    # 2..8 ONLY: past 8 the law over-predicts 12-40% (measured at split 16), so the
    # term clamps split into [2, 8] rather than extrapolate. Carries a +/-25% error
    # bar from the non-governing side of the view pair -- reproducible scatter with no
    # consistent direction (destination sweep, 10 pairs x 15 reps), so it is an error
    # bar and not a term. Fanout was measured NOT to be a term (2/4/8 at fixed
    # geometry, no trend). NO loop_trip factor: a relayout inside a coarse-tiling loop
    # is structurally impossible today (the planner rejects coarse_tile_copy
    # consumers).
    relayout_run_a_ns: float = -1.14  # per-run cost intercept
    relayout_run_b_ns: float = 3.92  # per-run cost log2(split) slope
    relayout_span_gbps: float = 547.0  # per-core stride-limited walk rate
    # Matmul operand RE-READ (tile spill): the per-core OUTPUT-accumulator tile has area
    # (M/m)*(N/n); once it exceeds the on-chip capacity (~64K fp16 elems/core) it no
    # longer stays resident, so the operands are re-streamed from HBM. The re-read
    # magnitude is the operand bytes; the fraction grows with how far the tile
    # overflows:
    #   reread = (|A| + |B|) * f(area),  f(area) = min(cap, slope*log2(area/area0)).
    # Fit on the decouple + re-read sweeps (area spill, K-split never used so psum ring
    # = 0).
    mm_spill_area0: float = (
        65536.0  # per-core output-tile area (elems) below which no spill
    )
    mm_spill_slope: float = 0.45
    mm_spill_cap: float = 1.50
    # SPLIT-SHAPE re-read: the area spill above is symmetric in m<->n and so is
    # blind to how the output is split. A forced-split sweep shows a large per-core tile
    # that is ALSO split many ways costs extra the area term misses -- an INTERACTION of
    # tile size and how far each output dimension is split. It is SIZE-GATED (bites
    # only once the tile is ~2x the area-spill knee, so a small lopsided tile stays
    # accurate) and TWO-SIDED: splitting the LONGER output dim into many cores is
    # penalized sooner (knee 8, the planner's own _COHORT_LIMIT) and harder than
    # splitting the SHORTER dim (knee 16, empirical). This asymmetry -- e.g. at M>>N,
    # a 32x1 split (fan the long M) costs ~2x a 1x32 (fan the short N) at equal area
    # -- is real (leave-one-shape-out RMS 337->249 vs the one-sided form) and a
    # symmetric term cannot express it.
    # split = cL*max(0,area-area0)*max(0,log2(fan_long/8)) +
    # cS*max(0,area-area0)*max(0,log2(fan_short/16)) area = (M/m)*(N/n);  fan_long = m
    # if M>=N else n;  fan_short = the other split count Zero for balanced splits (both
    # fanouts small) and for small tiles. Fitted offline with `eval_model.py`.
    mm_split_reread_us_per_elem: float = 2.62e-3  # cL: long-dim split coefficient
    mm_split_short_us_per_elem: float = 2.88e-3  # cS: short-dim split coefficient
    mm_split_area0: float = 131072.0  # per-core tile elems below which no split re-read
    mm_split_long_knee: float = (
        8.0  # long-dim cores before the re-read (planner _COHORT_LIMIT)
    )
    mm_split_short_knee: float = (
        16.0  # short dim tolerated to a higher fanout (empirical)
    )
    # Matmul HBM: a SINGLE effective rate = the pointwise copy peak (150). The earlier
    # two-rate fit (143 read / 156 write) is retired: 156 > 150 is unphysical (a write
    # cannot beat the copy peak) and was a compute-free-fit artifact absorbing the
    # overlap term. On the planner-realistic envelope a single 150 + turnaround +
    # overlap scores better than the old two-rate (RMS ~5.8% with the area-spill term)
    # -- equal-or-better AND physical. Read/write are not separately identifiable from
    # these data.
    mm_bw_read_gbps: float = 150.0
    mm_bw_write_gbps: float = 150.0
    # DEFAULT-LAYOUT BMM slow compute rate (cat 4). A batched matmul whose BOTH rank-3
    # operands carry the COMPILER-DEFAULT [0,1,2] device tile order -- the batch dim B
    # sits just inside the stick (device pos -2) -- runs the systolic array at a much
    # SLOWER sustained rate than a plain 2D matmul: on clean reps=7 data (bmm_wd +
    # bmm_layout both-default, cores=32, B>=4) the effective rate is a stable ~160
    # MAC/ns/core (us/GMAC ~214, mac_peak 145-147 raw; 160 the overlap-corrected fit),
    # vs 1140 for a 2D matmul. The [1,0,2] "fast" layout (B outermost) sustains
    # ~460-490; the compiler emits the slow [0,1,2] for every real bmm, so the model
    # must charge the slow rate. It is COMPUTE-bound (us/GMAC is shape-flat; the
    # effective HBM BW only appears to vary because io_hbm/MACs varies), so the fix is a
    # per-op mac_peak override, NOT a bandwidth change. Detector:
    # ``_default_layout_bmm_batch`` (both batched operands default at dev pos -2). GATED
    # to B >= ``bmm_default_min_batch``: at B=2 (batch << the 32-way M*N split) the slow
    # penalty halves (us/GMAC ~108, rate ~290) -- a distinct small-batch corner left on
    # the plain rate. Also GATED to cores >= ``bmm_default_min_cores``: the slow rate is
    # a many-core contention effect (implied per-core peak 407@c1, 241@c2, 168@c4 ->
    # ~160 only at c>=8; the clean fit is all cores=32), so low-core bmm keeps the plain
    # peak. Gold-safe: never fires on plain 2D ``mmwd`` (0/343) or the 3d-2d projection
    # bmm.
    bmm_default_mac_peak_per_core_ns: float = 160.0
    bmm_default_min_batch: int = 4
    bmm_default_min_cores: int = 8
    # LAYOUT IS NOT MODELLED, deliberately. A faster device tile order exists: with all
    # three operands on [1,0,2] (batch outermost) a bmm runs up to 8.3x faster at
    # byte-identical traffic, and the OUTPUT layout only pays off once both inputs are
    # already batch-outer. It is reachable only behind an opt-in layout preference that
    # the compiler does not enable by default, so nothing it emits today uses it.
    # Pricing all four operand combinations cost three constants and an additive form
    # for configurations that never ship; that was removed in favour of the single
    # default-layout rate above. The measurement stands and is kept in the database; the
    # non-default runs are excluded from scoring (`eval_model.in_scope`). A 3d-2d
    # PROJECTION bmm (one rank-3 operand, one shared 2D operand) runs far faster than a
    # full both-batched bmm. Two rates, keyed on batch size. MECHANISM: OPEN. The
    # obvious story -- "the 2D operand loads once and is reused, so it escapes the
    # per-batch re-gather" -- is WRONG, or at least already paid for: that operand is
    # tagged `broadcast=True` with `loop_factor=1`, so the byte count ALREADY charges it
    # once, and charging it again as a rate would double-book. Two further facts
    # contradict a pure amortisation story: (a) all 43 measured rank-3 operands are in
    # the SLOW default order, for which the layout-additivity term's own rate is 235
    # MAC/ns/core -- 2.6x below what 3d2d actually sustains, so the two shipped
    # mechanisms disagree about the same physical configuration; and (b) amortising a
    # once-loaded operand should get BETTER with more batches, whereas the measured rate
    # STEPS DOWN above B=4. These are empirical rates with the mechanism unresolved, and
    # they are labelled as such deliberately. THE B STEP IS REAL, not a shape confound
    # (an earlier flat-rate version of this term claimed otherwise and was refuted).
    # Per-GMAC cost normalised to B=4, within shape and at an identical 4x8 split:
    # 1024x1024x1024 -> 1.136/1.000/1.352/1.384 and 1024x2048x1024 ->
    # 1.145/1.000/1.425/1.415 at B=2/4/8/16, i.e. three shapes carry more than one B and
    # two are fully repeat-backed ladders. It is NOT capacity: the per-core working set
    # at the step (768->1024 KB, 1024->1536 KB) stays under the 1638 KB LX budget, and
    # two configs with IDENTICAL 1024 KB working sets run at 57.6 vs 80.1 us/GMAC.
    # Leave-one-SHAPE-out (holding out a whole B ladder) prefers the step over a flat
    # rate. Calibrated on repeat-backed PLAIN 3d2d only (n=19, B in {2,4,8,16});
    # `bmm_3d2d_k_tiling` is EXCLUDED because its `matmul_macs` is per-tile (up to 16x
    # under-counted), so it cannot calibrate a rate. Calibration RMS 16.5 -> 5.6 %, mean
    # -2.2 -> -0.2 %. RESIDUAL, disclosed: the smallest shape `512x2048x512` at B=4
    # still reads +10.4 % (it wants ~965, 1.6x the fitted rate) -- the small-shape
    # corner is not priced by either rate. It is the ONE repeat-backed record anywhere
    # in the database that this term makes worse (4.3 -> 10.4 %); the flat-rate version
    # it replaces left it at +21.9 %. NO CORES GATE, unlike the sibling layout term, and
    # deliberately: every measured 3d2d row is already cores >= 8 (44 at 32, two each at
    # 8/16), so a gate is a no-op on current data, and below it the fallback would be
    # the plain 1140 -- a rate the sibling's own low-core data shows is 2.5-7x too fast
    # for a batched matmul. Extrapolating the measured 3d2d rate is the lesser error.
    # The cores=8/16 rows are single-shot and remain the worst in the set (-68.8 ->
    # -52.9 %, -66.5 -> -51.1 %); they are improved but not resolved.
    bmm_3d2d_mac_peak_lo_ns: float = 705.0  # B <= bmm_3d2d_batch_knee
    bmm_3d2d_mac_peak_hi_ns: float = 470.0  # B >  bmm_3d2d_batch_knee
    bmm_3d2d_batch_knee: int = 4

    # Access-pattern effective HBM BW (GB/s) for non-matmul ops whose stick layout is
    # reorganized -- these fold turnaround into the single rate (measured io/kernel on
    # the db_sweep). Keyed by OpFeatures.hbm_pattern; default ops keep
    # bw_peak+turnaround.
    bw_restickify_gbps: float = (
        116.0  # transpose: stick swapped, LESS turnaround (faster)
    )
    # Stick-plane transports (cat0, transpose_outer, cat1): a `clone` that reorganizes
    # the stick layout. The 32 cores split the stick-plane dim (sp = C/64); each core
    # does the per-row stick work. Effective BW falls with the per-row strided stick
    # gather -- more planes (sp) and a longer gather stride (more rows R) -- so all
    # three share ONE form, effBW = clamp(a - b*log2(sp) - d*log2(R), floor, bw_peak)
    # (report S6), differing only in calibration. cat0 = a STEEP untiled gather (its two
    # cat copies sit just inside the stick, so each output row re-gathers sp scattered
    # input sticks). transpose_outer = a GENTLE tiled block-transpose, calibrated at its
    # best middle dim M=8 (effBW peaks at M~8 and falls either side -- M!=8 is a flagged
    # residual, not yet modeled). cat1 = nearly FLAT (its cat copies sit outermost ->
    # contiguous read+write). transpose itself is a hardware restickify (stick swapped)
    # -> flat bw_restickify_gbps.
    tx_cat0_a: float = 144.0
    tx_cat0_b: float = 9.6
    tx_cat0_d: float = 2.4
    tx_cat0_floor_gbps: float = 44.0
    tx_touter_a: float = 140.0
    tx_touter_b: float = 6.8
    tx_touter_d: float = 1.2
    tx_touter_floor_gbps: float = 83.0
    # transpose_outer only: the surface above is fit at the common M=8 (M = the
    # outer/swapped dim, `TO_MID`). M is the output's CONTIGUOUS STICK-RUN length in the
    # device layout [R, sp, M, 64], so M<8 means sub-1 KB writes and the rate falls
    # further -- measured (all repeat-backed, cv<1%): M=8 +3.7%, M=4 -16.0%, M=2 -26.8%
    # error, i.e. monotone in log2(M). Charged as a per-halving BW loss BELOW M=8,
    # applied AFTER the surface clamp because it is a separate effect from the (R,C)
    # surface whose floor is calibrated at M=8. Fit: 13 GB/s/halving takes
    # transpose_outer RMS 9.6% -> 7.1% and the worst M<8 point from -29% to <10%. The
    # M>8 side is deliberately NOT modelled: it is weaker, R-dependent (-0%/-4% at R=512
    # vs -9%/-22% at R=2048) and confounded with a planner split-shape change at large
    # sizes ((1,32,1)->(2,2,8)->(4,4,2)); it needs the TAILS sweep.
    tx_touter_m_ref: float = 8.0
    tx_touter_m_penalty_gbps: float = 13.0
    tx_touter_m_floor_gbps: float = 40.0
    tx_cat1_a: float = 110.0
    tx_cat1_b: float = 0.8
    tx_cat1_d: float = 0.0
    tx_cat1_floor_gbps: float = 90.0
    #: Smallest value the coarse-underfill surface was fitted at -- its support corner
    #: (COLS 128, h 4). Below it the fit is not evaluated; the curve is anchored here and
    #: continued with `gamma`. Was a bound on h alone; now on the surface VALUE, because
    #: with a width term the support boundary is no longer a single tile height.
    coarse_underfill_eff0: float = 0.248
    #: Decay exponent BELOW the anchor. Not fitted -- see `coarse_underfill_eff`. Small
    #: enough to bound the derate, non-zero so the tile still orders configurations.
    coarse_underfill_gamma: float = 0.1
    bw_reduce_outer_gbps: float = 113.0  # cross-row (dim0) reduction (sumcol)
    # Row-reduction READ rate falls with ROWS (the read pipeline degrades as each core
    # streams more rows), op-independent, saturating. Fit on the reduction-rows sweep
    # (read/sumrow/amax/mean/sumall collapse to one curve): effBW = floor +
    # amp*exp(-ROWS/ scale), clamped to peak. sumcol (reduce_outer) is exempt --
    # different access pattern.
    red_read_bw_floor_gbps: float = 114.0
    red_read_bw_amp_gbps: float = 61.0
    red_read_bw_scale_rows: float = 3700.0
    # A reduction streams a full-tensor READ over the shared HBM bus; with FEWER active
    # cores fewer parallel LX request streams are in flight, so a smaller fraction of
    # peak HBM bandwidth is realized. reduction_read_bw() above is the cores=32
    # (full-bus) calibration; this table derates it below 32 cores.
    # g(cores)=BW(cores)/BW(32), the mean over the clean plain reductions
    # (read/amax/sumrow/mean) at their non-underfilled anchor shapes. It is
    # SUB-linear/saturating (a single core drives ~11% of the bus, not 1/32=3%);
    # g=cores/32 is falsified 3.6x. g(32)=1.0 EXACTLY -> the cores=32 gold path is
    # untouched. c8/c16 are single-shot (no reps); the measured plateau ~0.54 is shipped
    # as-is (a repeated low-core reduction sweep is the deciding experiment). Applies
    # ONLY on the standalone-reduction branch below; fused softmax (len>1) is a separate
    # io_hbm effect.
    red_bw_cores_g: dict = dataclasses.field(
        default_factory=lambda: {1: 0.11, 2: 0.22, 4: 0.43, 8: 0.54, 16: 0.54, 32: 1.0}
    )
    # Ops that stream a FULL input plus a small BROADCAST operand (loaded once) -- copy
    # (x+const), bcast, bcastcol, mulbcast -- run FASTER than a plain 1R:1W op (~118 vs
    # ~105 GB/s; mechanism open). NOT `write` (both operands broadcast, no full input).
    bw_broadcast_gbps: float = (
        118.0  # fallback rate (used only if the logical shape is absent)
    )
    # ALL FOUR broadcast ops share the SAME shape of effective-BW surface (a dense R×C
    # sweep): a well-filled regime (ROWS >= min_rows) where the rate declines gently
    # with both COLS and ROWS, and a short-tensor regime (ROWS < min_rows) that is a
    # V-valley with its minimum at ROWS = COLS/64 (the output stick-plane count). The
    # small-ROWS collapse and the COLS/ROWS dependence are GENERAL broadcast-kernel
    # effects -- they are NOT specific to the b[1,C] operand (copy, a scalar broadcast,
    # shows the same collapse). The only operand-specific difference is a small rate
    # lift: the ROW-broadcast ops (bcast/mulbcast, b[1,C]) run a few GB/s faster than
    # the scalar/column ops (copy/bcastcol), so each FAMILY gets its own fit.
    bcast_bw_min_rows: float = (
        1024.0  # split between the well-filled and short-tensor regimes
    )
    bcast_v_cols_split: float = (
        4096.0  # short-tensor: COLS<=split -> quadratic, else -> V
    )
    # -- ROW-broadcast family (bcast, mulbcast: operand b[1,C]) --
    bcast_bw_a: float = (
        183.5  # well-filled: effBW = a - b*log2(COLS) - d*log2(ROWS)  (1.3% RMS)
    )
    bcast_bw_b: float = 2.8
    bcast_bw_d: float = 2.6
    bcast_q_a: float = (
        -350.0
    )  # short COLS<=4k: a + b*lr + c*lr^2 + e*log2(COLS)  (lr=log2 ROWS)
    bcast_q_b: float = 105.0
    bcast_q_c: float = -5.5
    bcast_q_e: float = -2.0
    bcast_v_plateau: float = 128.0  # short COLS>=8k V: min at ROWS=COLS/64
    bcast_v_floor: float = 92.0
    bcast_v_bl: float = 32.0
    bcast_v_br: float = 10.0
    # -- SCALAR/COLUMN family (copy: scalar; bcastcol: b[R,1]) -- same shape, slightly
    # slower --
    cbc_bw_a: float = 162.0  # well-filled surface (5.1% RMS)
    cbc_bw_b: float = 1.8
    cbc_bw_d: float = 2.2
    cbc_q_a: float = -270.0  # short COLS<=4k quadratic (6.0% RMS)
    cbc_q_b: float = 70.0
    cbc_q_c: float = -3.5
    cbc_q_e: float = 3.0
    cbc_v_plateau: float = 120.0  # short COLS>=8k V (10.2% RMS)
    cbc_v_floor: float = 98.0
    cbc_v_bl: float = 15.0
    cbc_v_br: float = 8.0


def underfill_eff(
    rows_per_core: float,
    params: CostParams | None = None,
    target_passes: float | None = None,
) -> float:
    """Pipeline-fill efficiency (<=1) for a per-core tile of ``rows_per_core`` rows.

    The streaming / PT pipeline processes a core's tile in passes of
    ``underfill_pass_rows`` (8) rows; a tile shorter than ``pass_rows * target_passes``
    cannot amortise pipeline fill/drain, so effective throughput derates as
    ``(rows / r_full) ** exponent``, capped at 1. ``target_passes`` defaults to the
    pointwise value (coarse-tiling); pass ``underfill_target_passes_matmul`` for the
    matmul compute term (same FORM, deeper pipeline). ``rows_per_core <= 0`` (unknown)
    -> 1.0 (no derate). BUNDLED matmul path only (see the module docstring).
    """
    p = params or CostParams()
    if rows_per_core <= 0:
        return 1.0
    tp = p.underfill_target_passes_pointwise if target_passes is None else target_passes
    r_full = p.underfill_pass_rows * tp
    if r_full <= 1.0:
        # when r_full <= 1.0, rows_per_core / r_full >=1.0, resulting in 1.0
        return 1.0
    return min(1.0, (rows_per_core / r_full) ** p.underfill_exponent)


def _op_cols(o) -> float:
    """The op's row WIDTH: the longest last logical dim over its operands. This is the
    length of one contiguous run in the device layout, so it is what sets how much a core
    streams per row -- used by both the underfill surface and the LX working set."""
    return max((a.logical[-1] for a in o.args if a.logical), default=0)


def coarse_underfill_eff(
    rpc: float,
    cols: float,
    params: CostParams | None = None,
    cap: float | None = None,
) -> float:
    """Pipeline-fill efficiency for a COARSE-tiled (fused pointwise / softmax) kernel whose
    per-core tile is ``rpc`` rows tall (rpc = ROWS/(cores*tiles)) and ``cols`` wide.
    DISTINCT from the matmul pt_eff (``underfill_eff``). ``rpc<=0`` (untiled) or
    ``cols<=0`` (width unknown) -> 1.0 (no derate).

        eff = min(cap, (rpc/r_full)**exp * (cols/col_ref)**col_exp)

    BOTH variables matter, and the width one is the 2026-08-11 correction: the surface
    used to key on ``rpc`` alone, and over-predicted a wide tiled softmax by 26-54 % while
    under-predicting a narrow one. What the tile really has to be is BIG -- the fitted
    exponents (0.50, 0.38) put the governing quantity close to ``rpc*cols``, the elements
    each core streams per tile -- so ``r_full`` is now the full-fill height AT ``col_ref``,
    and a tensor half as wide needs ~1.7x the height. See CostParams for the fit.

    SOFTMAX / fused-pointwise ONLY. A coarse-tiled MATMUL goes through
    ``coarse_underfill_eff_matmul``, which is the pre-re-fit rows-only curve held frozen:
    every row behind this surface is a softmax row, and the one check available says the
    new shape does not transfer to matmul. Same split as the sibling
    ``_lx_spill_bw_derate``, which already carries a separate cap/exponent pair for matmul.
    """
    p = params or CostParams()
    if rpc <= 0 or cols <= 0:
        return 1.0
    raw = (rpc / p.coarse_underfill_rfull) ** p.coarse_underfill_exp * (
        cols / p.coarse_underfill_col_ref
    ) ** p.coarse_underfill_col_exp
    # OUTSIDE THE FIT, DO NOT EXTRAPOLATE THE FIT. The surface is calibrated over a
    # (rpc, cols) box whose smallest corner -- cols 128, rpc 4 -- puts it at
    # `coarse_underfill_eff0`. Below that it was being evaluated anyway, and being an
    # unbounded power law it kept diverging: a recorded flash bundle carried rpc 0.0078,
    # giving eff 0.0065 -- a 155x derate that inflated its prediction 42x.
    #
    # So the curve is ANCHORED at the smallest VALUE it was fitted at and continues
    # below with a much weaker exponent. The anchor is on the surface value rather than
    # on rpc because with two variables the edge of the support is no longer one tile
    # height. Two properties matter: * inside the fit nothing changes at all, so no
    # calibrated point moves; * below it the tile still ORDERS configurations -- a plain
    # floor does not, and measurably collapsed two distinct flash tilings onto one
    # prediction.
    #
    # `coarse_underfill_gamma` is a FORM choice, not a fitted constant: the only data
    # below the anchor is six flash bundles, far too few to fit an exponent, and no more
    # can be collected while coarse tiling does not compile. Re-fit it when tiled
    # measurements exist there again.
    ceil_ = p.coarse_underfill_cap if cap is None else cap
    e0 = p.coarse_underfill_eff0
    if raw >= e0:
        return min(ceil_, raw)
    return min(ceil_, e0) * (raw / e0) ** p.coarse_underfill_gamma


def coarse_underfill_eff_matmul(rpc: float, params: CostParams | None = None) -> float:
    """Coarse-tiling underfill for a tiled MATMUL: ``min(0.95, (rpc/13)**0.68)``, anchored
    at ``rpc = 2`` and continued below with ``gamma``. BUNDLED matmul path only (see the
    module docstring).

    This is the surface as it stood before the 2026-08-11 two-variable re-fit, kept for
    matmul alone and deliberately FROZEN -- see ``coarse_underfill_rfull_matmul``. It is
    an extrapolation from softmax either way (no matmul measurement calibrates it), but it
    is the extrapolation the tiled-matmul accuracy figures were taken under, and the new
    one scores worse there. ``rpc<=0`` (untiled/unknown) -> 1.0.
    """
    p = params or CostParams()
    if rpc <= 0:
        return 1.0
    h0, ceil_ = p.coarse_underfill_h0_matmul, p.coarse_underfill_cap_matmul
    r_full, exp = p.coarse_underfill_rfull_matmul, p.coarse_underfill_exp_matmul
    if rpc >= h0:
        return min(ceil_, (rpc / r_full) ** exp)
    return min(ceil_, (h0 / r_full) ** exp) * (rpc / h0) ** p.coarse_underfill_gamma


def _lx_spill_working_set(ops: list) -> float:
    """Per-core LX working set (bytes) of a coarse-tiled bundle: ~2 live intermediate tiles,
    each ``tile_rows_per_core * cols`` elements. 0.0 if nothing is output-tiled."""
    ws = 0.0
    for o in ops:
        if o.tiles_output_dim and o.tile_rows_per_core > 0:
            ws = max(ws, 2.0 * o.tile_rows_per_core * _op_cols(o) * o.dtype_bytes)
    return ws


def _lx_spill_bw_derate(ops: list, params: CostParams | None = None) -> float:
    """Bandwidth derate when a coarse-tiled kernel's per-core working set overflows LX. The
    spilled intermediates are already counted as HBM bytes, but that traffic runs slower, so
    ``BW *= (lx_spill_cap / ws)**lx_spill_exp`` for ``ws > cap``. Gated to non-matmul coarse
    tiling (softmax-calibrated); 1.0 when it does not apply."""
    p = params or CostParams()
    ws = _lx_spill_working_set(ops)
    _cap, _exp = p.lx_spill_cap_bytes, p.lx_spill_exp
    if any(getattr(o, "is_matmul", False) for o in ops):
        _cap, _exp = p.mm_spill_ws_cap_bytes, p.mm_spill_ws_exp
    if ws <= _cap:
        return 1.0
    return (_cap / ws) ** _exp


def _bmm_layout_pair(o) -> tuple:
    """Classify a batched matmul's two operands by device tile order. BUNDLED matmul path
    only (see the module docstring).

    Returns ``(B, a_is_default, b_is_default)`` for a batched matmul (two rank-3 operands,
    B>=2), else ``(0, False, False)``. The compiler-default ``[0, 1, 2]`` order places the
    batch dim just INSIDE the stick -- device position -2 (logical ``[B, X, Y]`` -> device
    ``[X, Y/64, B, 64]``); the alternate ``[1, 0, 2]`` puts B outermost. Each operand is
    classified INDEPENDENTLY because the two penalties are separate and unequal (cat 4).

    Operand ORDER is load-bearing here (operand A's penalty is the larger one), and is safe:
    the ``o.args`` input order matches the recorded ``layout_a``/``layout_b`` on 54/54 swept
    rows. NEVER matches a plain 2D matmul (no rank-3 input) or a 3d-2d projection bmm (only
    one rank-3 operand)."""
    if not getattr(o, "is_matmul", False):
        return (0, False, False)
    batched = [
        a
        for a in o.args
        if a.role == "input" and len(a.logical) == 3 and a.logical[0] >= 2
    ]
    if len(batched) < 2:
        return (0, False, False)

    def _is_default(a) -> bool:
        # default [0,1,2] <=> batch dim B lands at device pos -2 (just before the stick)
        return len(a.dims) >= 2 and a.dims[-2] == a.logical[0]

    return (batched[0].logical[0], _is_default(batched[0]), _is_default(batched[1]))


def _bmm_3d2d_batch(o) -> int:
    """Batch size B iff this is a 3d-2d PROJECTION bmm (exactly ONE rank-3 operand), else 0.
    BUNDLED matmul path only (see the module docstring).

    A full bmm has TWO rank-3 inputs and is handled by `_bmm_layout_pair`; a plain 2D matmul
    has none -- the two classifiers are mutually exclusive by construction and 0 of 2068 swept
    records trip both. Fires on 48 records by leading op (62 bundles once every op in the
    bundle is considered), all `bmm_wd_3d2d` or `bmm_3d2d_k_tiling`, and on NO other op:
    0/573 plain 2D matmul, 0/175 full bmm, 0 rank-4 (flash)."""
    if not getattr(o, "is_matmul", False):
        return 0
    batched = [
        a
        for a in o.args
        if a.role == "input" and len(a.logical) == 3 and a.logical[0] >= 2
    ]
    return batched[0].logical[0] if len(batched) == 1 else 0


def _default_layout_bmm_batch(o) -> int:
    """Back-compat shim: batch size B iff BOTH operands are default-layout, else 0. BUNDLED
    matmul path only (see the module docstring)."""
    b, a_def, b_def = _bmm_layout_pair(o)
    return b if (a_def and b_def) else 0


def _matmul_mac_peak(o, params: "CostParams") -> float:
    """Per-core sustained MAC/ns for a matmul op. BUNDLED matmul path only (see the module
    docstring).

    The device tile order of a batched matmul sets the sustained rate the systolic array
    reaches, and the two operands' penalties are **additive in time to within ~2 %** -- measured
    on 11 matched quads (same B/M/K/N/split, byte-identical, copy-free): the ratio
    ``(A-slow + B-slow) / (both-slow + both-fast)`` is 0.983 (0.953-1.018 across rep
    aggregations). It is *not* 1.0 within noise -- see the parameter comment for the residual
    ~1.7 % super-additivity that this form knowingly drops. So the reciprocal rates add::

        1/peak = 1/peak_fast + [A default]/peak_a_default + [B default]/peak_b_default

    Three constants reproduce all four measured combos to a few percent (both-fast ~650,
    A-default-only ~236, B-default-only ~287, both-default 161.5 MAC/ns/core), where a single
    constant could only reproduce the both-default sum -- which is all production needs today,
    since every emitted bmm is both-default; the other three rates price a layout *change*.
    Gated to B >= ``bmm_default_min_batch``
    and cores >= ``bmm_default_min_cores``: small-batch and low-core bmm are genuinely
    different regimes (implied peak 407/241/168 at cores 1/2/4; B=2 runs ~2x faster) whose
    data cannot yet support a rate. Everything else -- plain 2D matmul, 3d-2d projection,
    gated-out bmm -- keeps the plain ``mac_peak_per_core_ns``, so the gold path is untouched."""
    b3 = _bmm_3d2d_batch(o)
    if b3:
        # 3d-2d projection: empirical rate, stepping down above B=4. Mechanism open --
        # see the parameter comment for the three facts that refute the obvious
        # amortisation story.
        return (
            params.bmm_3d2d_mac_peak_lo_ns
            if b3 <= params.bmm_3d2d_batch_knee
            else params.bmm_3d2d_mac_peak_hi_ns
        )
    b, a_def, b_def = _bmm_layout_pair(o)
    if isinstance(o.cores, sympy.Basic):
        # TODO: make symbolic
        return params.mac_peak_per_core_ns
    if b < params.bmm_default_min_batch or o.cores < params.bmm_default_min_cores:
        return params.mac_peak_per_core_ns
    if a_def and b_def:
        # The ONLY batched arrangement the compiler emits: the layout preference that
        # would change it is opt-in and off by default, so both rank-3 operands carry
        # the default [0,1,2] order. One measured rate covers it.
        return params.bmm_default_mac_peak_per_core_ns
    # A faster order exists -- up to 8.3x with all three operands batch-outer -- but it
    # is reachable only behind an opt-in layout preference that no released build
    # enables, so it is deliberately NOT modelled and those runs are out of scope.
    # Modelling it cost three constants and an additive form for a configuration
    # nothing emits.
    return params.mac_peak_per_core_ns


def mm_spill_frac(tile_area: float, params: CostParams | None = None) -> float:
    """Operand RE-READ fraction for a matmul: once the per-core output-accumulator tile
    of area ``(M/m)*(N/n)`` exceeds ``mm_spill_area0`` (the on-chip capacity) the operands
    no longer stay resident and are re-streamed from HBM. Saturating log growth
    ``min(cap, slope*log2(area/area0))``; 0 at/below area0. BUNDLED matmul path only (see
    the module docstring)."""
    p = params or CostParams()
    if tile_area <= 0:
        return 0.0
    return min(
        p.mm_spill_cap,
        p.mm_spill_slope * log2(max(1.0, tile_area / p.mm_spill_area0)),
    )


def relayout_ns(o: "OpFeatures", params: "CostParams | None" = None) -> float:
    """Additive cost of one LX relayout (shuffle) op; 0 for everything else.

    A permutation across cores: every core moves its own per-core bytes, paying a
    fixed cost per contiguous run plus a stride-limited walk over the whole span
    (see the ``relayout_*`` CostParams for the fitted law, its derivation and its
    validity range). Kept as a standalone function on purpose: the solver-objective
    path (#3810) cannot lower ``log2`` or division by a decision variable, so it
    evaluates THIS function per candidate division and folds the result into an
    AddElement table -- one source of truth for both paths.

    The split is clamped into [2, 8], the fitted range. Below 2 cannot occur for a
    real ownership change (and would turn the fitted intercept negative); above 8
    the law over-predicts by 12-40% (measured at split 16), so the clamp bounds the
    error instead of extrapolating an unmeasured log2.
    """
    p = params or CostParams()
    if not o.is_lx_relayout or o.out_elems <= 0:
        return 0.0
    if o.relayout_run_elems <= 0 or o.cores <= 0:
        return 0.0
    # Bytes moved == the copy's device bytes: a relayout is an identity clone.
    per_core = o.out_elems * o.dtype_bytes / o.cores
    run_bytes = o.relayout_run_elems * o.dtype_bytes
    split = min(8, max(2, o.relayout_split))
    runs = per_core / run_bytes
    return (
        runs * (p.relayout_run_a_ns + p.relayout_run_b_ns * math.log2(split))
        + per_core * split / p.relayout_span_gbps
    )


def _fused_hbm_bytes(ops: list) -> tuple:
    """(read, write) HBM bytes for a FUSED bundle, counting each distinct EXTERNAL graph
    input (name starts ``arg``) ONCE even if several fused ops read it -- a fused kernel
    loads it from HBM once and serves the re-reads on-chip/LX (softmax reads ``arg0`` in
    both ``amax`` and ``sub``; the naive per-op sum double-counts it, ~+25% at the floor).
    Internal-buffer traffic is taken as the IR reports it: LX buffers are ~free (excluded),
    and a buffer that SPILLED to HBM and is re-read stays counted (the spill is exactly why
    it can't be reused on-chip). Outputs summed as-is (distinct per op)."""
    r = w = 0
    ext_in: dict = {}  # external input name -> its one-load HBM bytes (dedup across ops)
    for o in ops:
        for a in o.args:
            b = a.elems * a.loop_factor * o.dtype_bytes * (1 - a.is_lx)
            if a.role == "input" and a.name.startswith("arg"):
                if a.name in ext_in:
                    ext_in[a.name] = max(ext_in[a.name], b)
                else:
                    ext_in[a.name] = b
            elif a.role == "input":
                r += b
            else:
                w += b
    r += sum(ext_in.values())
    return r, w


def _loop_reread_bytes(ops: list) -> float:
    """HBM bytes re-read because an operand is LOOP-INVARIANT under coarse tiling.

    A coarse-tiled kernel is one op containing an L-iteration loop. An operand whose
    index expression does not involve the tiled loop symbol is re-entered at the SAME
    address every iteration -- so it is transferred L times, not once. The extractor
    marks this per arg as ``loop_factor > 1``; this returns only the EXCESS over the
    first pass, i.e. ``elems * (loop_factor - 1) * dtype``.

    Established for ``matmul_row_tiling`` (tiles M) two independent ways:

    * IR: ``inner_fn`` loads A at ``r0_0 + 2048*i0`` (contains the tiled symbol ``i0``)
      and B at ``i1 + 2048*r0_0`` (does not), with ``loop_tiled_dims=[[0]]`` and
      ``DimHint(dim_names=['M'], loop_var=d0)``.
    * The recorded features already show the asymmetry: ``matmul_b_bytes`` is the FULL
      ``K*N*2`` at every L (ratio 1.00), while ``matmul_a_bytes`` is exactly ``M*K*2/L``.

    B is NOT served from LX. The scratchpad allocator only pins a graph input by cloning
    it, and ``_input_residency_reason``'s first gate rejects any input whose
    ``_read_count`` (which excludes the unavoidable first read) is 0. The L iterations
    live INSIDE one op, so the allocator sees a single use and declines -- confirmed on
    every record we hold: coarse-matmul input args are 0 LX (several hundred
    reads, all HBM; the exact count moves as the database grows), while
    ``softmax_row_tiling`` gets 1420 LX args, so the planner is working and simply does
    not apply here.

    SCOPE -- deliberately narrow. Counts only an **input** of a **matmul** whose loop
    tiles an **output** dim. That is exactly the case established above, and the gate
    matters: ``loop_factor > 1`` ALREADY occurs today on the REDUCTION-tiled ops
    (``matmul_k_tiling``, ``bmm_k_tiling``, ``bmm_3d2d_k_tiling``, ``ctsum``/``ctamax``/
    ``ctamin``), where it is set on the output accumulator AND the inputs alike. Those
    factors mean something different -- under K-tiling each iteration consumes a fresh
    K-slice, so the inputs ADVANCE and only the accumulator is re-touched -- and they are
    part of the known per-iteration-vs-per-loop inconsistency in the extractor, not a
    verified quantity. Charging them here made ``matmul_k`` 7.9 -> 32.4 % and
    ``bmm_3d2d`` 18.0 -> 38.8 %, so they are excluded until that inconsistency is fixed.

    0.0 for every op the extractor emits today (output-dim-tiled matmuls carry
    ``loop_factor = 1`` on all args), so this term is INERT until the extractor sets the
    invariant input's factor -- verified by the gold gate.
    """
    extra = 0.0
    for o in ops:
        if not (getattr(o, "is_matmul", False) and o.tiles_output_dim):
            continue
        for a in o.args:
            if a.mem != "hbm" or a.role != "input":
                continue
            lf = getattr(a, "loop_factor", 1) or 1
            if lf > 1:
                extra += a.elems * (lf - 1) * o.dtype_bytes
    return extra


def _is_broadcast_op(o) -> bool:
    """True for an op that streams a FULL HBM input AND a small BROADCAST operand (loaded
    once): copy (x+const), bcast, bcastcol, mulbcast. These run at ``bw_broadcast_gbps``,
    faster than a plain 1R:1W op. Excludes matmul/reduction and ``write`` (both operands
    broadcast -> no full input, and a different, super-linear regime)."""
    if getattr(o, "is_matmul", False) or o.is_reduction:
        return False
    ins = [a for a in o.args if a.role == "input" and a.mem == "hbm"]
    return any(a.broadcast for a in ins) and any(not a.broadcast for a in ins)


def _logical_rc(o):
    """(rows, cols) from the op's output logical [.., R, C], or None."""
    out = next(
        (
            a
            for a in o.args
            if a.role == "output" and a.mem == "hbm" and len(a.logical) >= 2
        ),
        None,
    )
    return (out.logical[-2], out.logical[-1]) if out else None


def _has_row_broadcast_operand(o) -> bool:
    """True if a HBM input is a ROW-broadcast operand ``b[1, C]`` (bcast/mulbcast). copy
    (scalar const) and bcastcol (``b[R, 1]``) do NOT qualify -- no wide COLS-growing operand."""
    return any(
        a.role == "input"
        and a.mem == "hbm"
        and a.broadcast
        and len(a.logical) >= 2
        and a.logical[-2] == 1
        and a.logical[-1] > 1
        for a in o.args
    )


def broadcast_bw(o, p):
    """Effective HBM BW for a broadcast-operand op. All four ops share one surface SHAPE
    (well-filled COLS/ROWS decline + a short-tensor V-valley whose min is at ROWS=COLS/64);
    the ROW-broadcast ops (bcast/mulbcast, b[1,C]) run a few GB/s faster than the scalar/
    column ops (copy/bcastcol), so each family has its own constants."""
    rc = _logical_rc(o)
    if rc is None:
        return p.bw_broadcast_gbps
    rows, cols = rc
    if cols <= 0 or rows <= 0:
        return p.bw_broadcast_gbps
    if _has_row_broadcast_operand(o):  # bcast / mulbcast
        sa, sb, sd = p.bcast_bw_a, p.bcast_bw_b, p.bcast_bw_d
        qa, qb, qc, qe = p.bcast_q_a, p.bcast_q_b, p.bcast_q_c, p.bcast_q_e
        vp, vf, vbl, vbr = (
            p.bcast_v_plateau,
            p.bcast_v_floor,
            p.bcast_v_bl,
            p.bcast_v_br,
        )
        s_lo, s_hi, q_hi = 100.0, 135.0, 140.0
    else:  # copy (scalar) / bcastcol (b[R,1])
        sa, sb, sd = p.cbc_bw_a, p.cbc_bw_b, p.cbc_bw_d
        qa, qb, qc, qe = p.cbc_q_a, p.cbc_q_b, p.cbc_q_c, p.cbc_q_e
        vp, vf, vbl, vbr = p.cbc_v_plateau, p.cbc_v_floor, p.cbc_v_bl, p.cbc_v_br
        s_lo, s_hi, q_hi = 95.0, 130.0, 120.0
    lr, lc = log2(rows), log2(cols)
    if rows >= p.bcast_bw_min_rows:  # well-filled: gentle decline with both dims
        return max(s_lo, min(s_hi, sa - sb * lc - sd * lr))
    if (
        cols <= p.bcast_v_cols_split
    ):  # short + narrow: only the V's rising side (a quadratic)
        return max(45.0, min(q_hi, qa + qb * lr + qc * lr * lr + qe * lc))
    dip = lc - 6.0  # short + wide: the full V-valley, minimum at ROWS = COLS/64
    return max(40.0, min(vp, vf + vbl * max(0.0, dip - lr) + vbr * max(0.0, lr - dip)))


# Per-op (a, b, d, floor) for the shared stick-plane-transport BW form (see CostParams).
_TX_PARAM_ATTRS = {
    "cat0": ("tx_cat0_a", "tx_cat0_b", "tx_cat0_d", "tx_cat0_floor_gbps"),
    "transpose_outer": (
        "tx_touter_a",
        "tx_touter_b",
        "tx_touter_d",
        "tx_touter_floor_gbps",
    ),
    "cat1": ("tx_cat1_a", "tx_cat1_b", "tx_cat1_d", "tx_cat1_floor_gbps"),
}


def _transport_kind(o) -> str:
    """Classify a stick-plane transport from its logical shapes. cat0 and transpose carry an
    ``hbm_pattern`` already; cat1 and transpose_outer do NOT (they ride the default), so they
    are detected here from the input->output logical reshape. Returns "" for anything else."""
    if getattr(o, "is_matmul", False) or o.is_reduction:
        return ""
    outs = [a for a in o.args if a.role == "output" and a.mem == "hbm" and a.logical]
    ins = [a for a in o.args if a.role == "input" and a.mem == "hbm" and a.logical]
    if not outs or not ins:
        return ""
    ol, il = outs[0].logical, ins[0].logical
    # transpose_outer: [R, M, C] -> [M, R, C] (rank-3 outer-dim swap; stick dim C kept).
    if (
        len(ol) == 3
        and len(il) == 3
        and ol[0] == il[1]
        and ol[1] == il[0]
        and ol[2] == il[2]
        and ol[0] > 1
        and ol[1] > 1
    ):
        return "transpose_outer"
    # cat1: a 2-D input [R, C] reused across k copies -> [R, k, C] (concat on the stick
    # axis).
    if (
        len(ol) == 3
        and len(il) == 2
        and ol[1] >= 2
        and ol[0] == il[0]
        and ol[2] == il[1]
    ):
        return "cat1"
    return ""


def _transport_rc(o, kind):
    """(rows R, cols C) for a stick-plane transport. C is always the stick dim (logical[-1]);
    R is the big non-stick dim -- logical[-2] EXCEPT cat1, whose output logical is [R, k, C]
    (the copy count k sits at [-2]), so R is the outer dim there."""
    outs = [
        a
        for a in o.args
        if a.role == "output" and a.mem == "hbm" and len(a.logical) >= 2
    ]
    if not outs:
        return None
    ol = outs[0].logical
    rows = ol[0] if kind == "cat1" else ol[-2]
    return rows, ol[-1]


def _transport_outer_m(o):
    """transpose_outer's swapped OUTER dim M: the output logical is [M, R, C] (from an input
    [R, M, C]), so M is logical[0]. It is the output's contiguous stick-run length on device
    ([R, sp, M, 64]). Returns None when the output is not rank-3."""
    for a in o.args:
        if a.role == "output" and a.mem == "hbm" and len(a.logical) == 3:
            return a.logical[0]
    return None


def transport_bw(o, p, kind):
    """Shared stick-plane-transport effective BW: clamp(a - b*log2(sp) - d*log2(R), floor,
    peak), sp = C/64. One form for cat0 / transpose_outer / cat1; per-op constants.

    ``transpose_outer`` additionally loses bandwidth when its swapped outer dim M drops below
    the M=8 the surface is fit at: M is the output's contiguous stick-run length, so M<8 means
    sub-1 KB writes. Charged after the clamp -- see ``tx_touter_m_penalty_gbps``."""
    a, b, d, fl = (getattr(p, n) for n in _TX_PARAM_ATTRS[kind])
    rc = _transport_rc(o, kind)
    if rc is None:
        return fl
    rows, cols = rc
    if rows <= 0 or cols <= 0:
        return fl
    sp = max(1.0, cols / 64.0)
    bw = a - b * log2(sp) - d * log2(max(2, rows))
    bw = min(p.bw_peak_gbps, max(fl, bw))
    if kind == "transpose_outer":
        m = _transport_outer_m(o)
        if m and m < p.tx_touter_m_ref:
            bw -= p.tx_touter_m_penalty_gbps * log2(p.tx_touter_m_ref / m)
            bw = max(p.tx_touter_m_floor_gbps, bw)
    return bw


def _reduction_rows(o):
    """ROWS of a reduction's input (governs its read rate), from the largest HBM input."""
    ins = [
        a
        for a in o.args
        if a.role == "input" and a.mem == "hbm" and len(a.logical) >= 2
    ]
    return max(ins, key=lambda a: a.elems).logical[-2] if ins else 0


def reduction_read_bw(rows, p):
    """Row-reduction read rate: peak at small ROWS, falling+saturating as ROWS grows."""
    return min(
        p.bw_peak_gbps,
        p.red_read_bw_floor_gbps
        + p.red_read_bw_amp_gbps * math.exp(-rows / p.red_read_bw_scale_rows),
    )


def _reduction_bw_cores_factor(cores, p):
    """g(cores)=BW(cores)/BW(32): the fraction of full-bus reduction bandwidth realized
    with `cores` active cores. Piecewise-linear in log2(cores) over the measured anchor
    table; 1.0 at cores>=32 (or unknown) so the cores=32 gold path is unchanged."""
    g = p.red_bw_cores_g
    if isinstance(cores, sympy.Basic):
        # TODO: make symbolic
        return 1.0
    if cores is None or cores >= 32:
        return 1.0
    if cores <= 1:
        return g[1]
    ks = sorted(g)
    lc = log2(cores)
    for a, b in zip(ks, ks[1:]):
        if a <= cores <= b:
            la, lb = log2(a), log2(b)
            return g[a] + (g[b] - g[a]) * (lc - la) / (lb - la)
    return 1.0


def _matmul_axes_for_split_cost(o) -> tuple | None:
    """Recover the ``(B,b),(M,m),(N,n),(K,k)`` axis pairs, the ``shared_weight`` flag,
    and the cores actually used -- everything ``work_division._matmul_split_cost``
    needs -- from one matmul :class:`OpFeatures` record.

    Returns ``None`` when matmul_a_bytes and matmul_b_bytes are not given
    """
    m_split = max(1, o.matmul_m_split)
    n_split = max(1, o.matmul_n_split)
    k_split = max(1, o.reduction_cores)
    M = o.matmul_rows_per_core * m_split
    N = o.matmul_cols_per_core * n_split
    if o.matmul_a_bytes:
        K = o.matmul_a_bytes / (M * o.dtype_bytes)
    elif o.matmul_b_bytes:
        K = o.matmul_b_bytes / (N * o.dtype_bytes)
    else:
        return None
    B_total = max(1.0, o.out_elems / (M * N))
    b_split = o.cores // (m_split * n_split * k_split)
    shared_weight = any(a.role == "input" and a.broadcast for a in o.args)
    return (
        (round(B_total), b_split),
        (round(M), m_split),
        (round(N), n_split),
        (round(K), k_split),
        shared_weight,
    )


def _matmul_ns_upstream(ops: list, p: CostParams) -> float:
    """Compute-side (ns) of a bundle containing a matmul, using the UPSTREAM
    (``CostParams.use_bundled_cost_model=False``) matmul model.

    HBM traffic is EXCLUDED (``include_hbm=False``) and left to ``predict_ops``' shared
    memory term, exactly as for ``_matmul_ns_bundled`` -- the two terms count the same
    operand/output bytes, so charging both double-counts memory (and the split-cost
    version is blind to LX residency, which is what the co-optimizing planner steers).
    """
    total_us = 0.0
    for o in ops:
        if not getattr(o, "is_matmul", False):
            continue
        axes = _matmul_axes_for_split_cost(o)
        if axes is None:
            raise RuntimeError(
                f"matmul op {o.name!r} has unresolvable axes (missing "
                "matmul_a_bytes/matmul_b_bytes) -- the UPSTREAM matmul cost model "
                "cannot price it"
            )
        b_axis, m_axis, n_axis, k_axis, shared_weight = axes
        us = _matmul_split_cost(
            b_axis,
            m_axis,
            n_axis,
            k_axis,
            config.sencores,
            shared_weight=shared_weight,
            include_hbm=False,
        )
        if us == float("inf"):
            cores_used = b_axis[1] * m_axis[1] * n_axis[1] * k_axis[1]
            raise RuntimeError(
                f"matmul op {o.name!r} has an infeasible core split "
                f"(b_axis={b_axis}, m_axis={m_axis}, n_axis={n_axis}, "
                f"k_axis={k_axis}, cores_used={cores_used})"
            )
        total_us += us
    return total_us * 1000.0  # us -> ns


def _matmul_ns_bundled(ops: list, p: CostParams) -> float:
    """Bundled matmul predicted device latency (ns)"" """
    # MATMUL compute = MACs/cores derated by pt_eff (PT-array fill).
    compute = 0.0
    for o in ops:
        if o.is_matmul:
            # A coarse-tiled matmul appears to underfill the array MORE per tile than a
            # standalone one, but the current data is too weak to fit -- so tiled
            # matmuls take pt_eff=1; standalone matmuls use the array-fill derate.
            if o.tiles_output_dim:
                pt_eff = 1.0
            else:
                pt_eff = underfill_eff(
                    o.matmul_rows_per_core, p, p.underfill_target_passes_matmul
                )
            # A DEFAULT-LAYOUT bmm (both operands on the slow [0,1,2] tile order,
            # B>=gate) runs the array at the slow rate; every other matmul keeps the
            # plain peak.
            mac_peak = _matmul_mac_peak(o, p)
            compute += o.matmul_macs / o.cores / (mac_peak * pt_eff)
    # SPLIT-SHAPE re-read: a large per-core output tile that is ALSO split many ways
    # re-reads operands beyond what the symmetric area spill counts. Two-sided:
    # splitting the LONGER output dim (knee 8) is penalized sooner/harder than the
    # SHORTER (knee 16). An INTERACTION of tile size and split; 0 for balanced splits
    # and small tiles.
    split_ns = 0.0
    for o in ops:
        if o.is_matmul:
            m_dev = o.matmul_rows_per_core * o.matmul_m_split
            n_dev = o.matmul_cols_per_core * o.matmul_n_split
            if isinstance(m_dev, sympy.Basic) or isinstance(n_dev, sympy.Basic):
                # TODO: make symbolic
                continue
            if m_dev >= n_dev:  # M is the longer output dim
                long_fan, short_fan = o.matmul_m_split, o.matmul_n_split
            else:
                long_fan, short_fan = o.matmul_n_split, o.matmul_m_split
            area_exc = max(
                0.0, o.matmul_rows_per_core * o.matmul_cols_per_core - p.mm_split_area0
            )
            split_us = area_exc * (
                p.mm_split_reread_us_per_elem
                * max(0.0, log2(long_fan) - log2(p.mm_split_long_knee))
                + p.mm_split_short_us_per_elem
                * max(0.0, log2(short_fan) - log2(p.mm_split_short_knee))
            )
            split_ns += split_us * 1000.0  # us -> ns
    return compute + split_ns


def predict_ops(ops: list, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a bundle of ops (one fused kernel).

    A matmul in the bundle adds a compute term from ``_matmul_ns_upstream`` (defers to
    ``work_division._matmul_split_cost``) or, when ``CostParams.use_bundled_cost_model``
    is set, ``_matmul_ns_bundled`` (the original device-calibrated compute/spill/
    split-shape model); see the module docstring. Either way the operand/output HBM
    bytes are charged ONCE, by the memory term below -- neither matmul model carries an
    HBM term of its own. Every bundle uses the device-calibrated bandwidth/turnaround/
    underfill model below:

    ``T = fill + [(R+W)/BW_PEAK + alpha*min(R,W)] / eff_underfill`` where R/W are the
    bundle's HBM read/write bytes (LX ~free), already
    loop-scaled per arg (see ArgTraffic.loop_factor). R/W come from ``_fused_hbm_bytes``:
    a fused kernel loads each distinct EXTERNAL input from HBM ONCE (re-reads served
    on-chip), so a shared input is not double-counted; reads and writes are summed over
    the bundle before the turnaround term (shared bus). ``eff_underfill`` derates the
    bandwidth term when OUTPUT-dim (coarse) tiling shrinks each core's per-tile stream
    (``coarse_underfill_eff``, keyed on per-core rows per tile AND the row width).
    """
    p = params or CostParams()
    r, w = _fused_hbm_bytes(ops)
    # HBM. Pointwise/reduction/transport keep the single-BW turnaround model.
    _pat_bw = {
        "restickify": p.bw_restickify_gbps,
        "reduce_outer": p.bw_reduce_outer_gbps,
    }

    def _eff_bw(o):  # per-op effective-BW override, or None -> default turnaround
        if any(isinstance(a.is_lx, sympy.Basic) for a in o.args):
            # TODO: make symbolic
            return None
        pat = getattr(o, "hbm_pattern", "")
        if pat == "stick_scatter":  # cat0: strided stick-plane gather (tagged)
            return transport_bw(o, p, "cat0")
        if pat in _pat_bw:  # restickify (transpose), reduce_outer (sumcol)
            return _pat_bw[pat]
        if _is_broadcast_op(o):
            return broadcast_bw(o, p)
        kind = _transport_kind(
            o
        )  # cat1 / transpose_outer -- untagged, detected structurally
        if kind:
            return transport_bw(o, p, kind)
        return None

    # Only the fused-reduction branch below raises this; every other path leaves it at 0
    # so the `max()` at `mem_t` is a no-op for them.
    if any(_eff_bw(o) is not None for o in ops):
        # Per-op effective BW (access-pattern transports OR a broadcast operand); these
        # fold turnaround into the rate. Ops without an override keep the default
        # single-BW + turnaround.
        mem = 0.0
        for o in ops:
            ro, wo = o.read_bytes(), o.write_bytes()
            bw = _eff_bw(o)
            if bw:
                mem += (ro + wo) / bw
            else:
                mem += (ro + wo) / p.bw_peak_gbps + p.rw_turnaround_ns_per_byte * min(
                    ro, wo
                )
    elif (
        len(ops) == 1
        and ops[0].is_reduction
        and not getattr(ops[0], "is_matmul", False)
        and not ops[0].tiles_output_dim
    ):
        # A STANDALONE row-reduction (sum/amax/mean/read over the last axis, or sumall)
        # reads at a rate that FALLS with ROWS. The rate is fit as (R+W)/time, so it
        # already includes the read/write turnaround -- do NOT add it again. sumcol
        # takes the reduce_outer path above; a FUSED coarse kernel (len>1, e.g. softmax)
        # stays on bw_peak below so its input dedup is not broken. The ROWS rate is the
        # cores=32 calibration; below 32 cores fewer streams drive the HBM bus, so
        # derate by g(cores) (g(32)=1 -> gold untouched). Only read/amax/sumrow/mean
        # reach here at cores<32; ctsum/ctamax/ctamin/sumall structurally qualify too
        # but only ever run at cores=32.
        _bw = reduction_read_bw(
            _reduction_rows(ops[0]), p
        ) * _reduction_bw_cores_factor(ops[0].cores, p)
        mem = (r + w) / _bw
    elif (
        len(ops) > 1
        and any(o.is_reduction for o in ops)
        and not any(getattr(o, "is_matmul", False) for o in ops)
    ):
        # A FUSED reduction bundle (softmax = amax -> sub -> exp -> sum -> div). This
        # path had no core-count term at all, which made low-core softmax the model's
        # worst category (median -82 % at cores<32; `softmax_unrolled` runs at cores=1
        # BY DESIGN, so every one of its points sat near -92 %). The binding constraint
        # there is PER-CORE ELEMENT that separate the two. Charged as a floor, so it
        # only ever raises a prediction and never binds at cores=32 (0/89 records) ->
        # the cores=32 path is byte-identical. FLAGGED, deliberately NOT modelled: the
        # floor alone leaves a systematic residual at cores=8/16 (median -17 % / -41 %),
        # where the throughput bound hands back to the memory term and that term is
        # itself too optimistic. The true form is a roofline whose OTHER side also
        # saturates; fitting that here would need the bus term re-derived at the same
        # time. Independently, the memory side still tracks the reduced-axis length COLS
        # at cores=32 (median error -50.9 / -38.7 / -17.5 / +0.4 % at cols 128/256/512/
        # 2048), which no core-count term can reach. Deciding experiment: a COLS x ROWS
        # cores ladder with repeats -- the present cells are one shape, one log, n=1 per
        # config. NOTE the floor is applied to the FINAL memory time below, not here:
        # `mem` is still divided by the underfill/spill derates further down, which
        # would inflate a floor imposed at this point by 1/(eff*spill_derate).
        mem = (r + w) / p.bw_peak_gbps + p.rw_turnaround_ns_per_byte * min(r, w)
    else:
        mem = (r + w) / p.bw_peak_gbps + p.rw_turnaround_ns_per_byte * min(r, w)
        # NOTE: a multi-op dependent chain (e.g. add3/add4 = chained binary adds) runs
        # slower than its byte count because the intermediate is written then read back
        # through HBM -- a READ-AFTER-WRITE dependency ACROSS op boundaries. That
        # is a program-level / coarse-tiling effect, NOT a single-op cost, so it is
        # deliberately NOT modeled here. `add_n` is not a native op; the single-op model
        # stays pure.
        #
        # SIZE OF THE GAP, measured on one build (2026-08-07, 78 pointwise rows): the
        # fused chains under-predict by -9.3 % on average, rising with chain depth --
        # add -5 %, add3 -10 %, add4 -15 %, add6 -16 %. `add_indep2` is the control that
        # identifies it: two INDEPENDENT adds, more bytes than add3 and the same op
        # count, predicted to -0.9 %. Two alternative readings are ruled out by the same
        # data -- op count (add_indep2 has two ops) and the read/write ratio (add, add5
        # and add6 all run at R:W = 2:1 and err -2 %, -15 %, -15 %). What remains is the
        # dependency itself. A byte-keyed read-after-write term, unified with the coarse
        # LX-spill derate below, is the natural next step.
    # OUTPUT-dim (pointwise) coarse-tiling underfill: a short per-core tile underfills
    # the streaming pipeline, derating the bandwidth term. The smallest tile in the
    # bundle governs (worst underfill). 1.0 (no derate) when nothing is output-tiled.
    # (This is the non-matmul, softmax-calibrated surface; a matmul bundle reaches it
    # too, and the matmul-calibrated curve `coarse_underfill_eff_matmul` is currently
    # used only by the bundled explain path.)
    eff = 1.0
    for o in ops:
        if o.loop_trip > 1 and o.tiles_output_dim and o.tile_rows_per_core > 0:
            eff = min(eff, coarse_underfill_eff(o.tile_rows_per_core, _op_cols(o), p))
    # LX-SPILL bandwidth derate: a coarse-tiled kernel whose per-core working set (~2
    # live intermediate tiles) overflows LX spills to HBM, and that spilled traffic runs
    # slower than the modeled rate. Bytes are already counted as HBM; here we derate the
    # BW.
    spill_derate = _lx_spill_bw_derate(ops, p)
    # A fused reduction bundle is floored by per-core element throughput (see
    mem_t = p.fill_ns + mem / eff / spill_derate
    # LOOP-INVARIANT OPERAND RE-READ, charged AFTER the derates and at the plain peak
    # rate. Placement is the mechanism, not a convenience: `eff` models a SHORT PER-TILE
    # stream underfilling the pipeline, but a re-read of a loop-invariant operand is one
    # large CONTIGUOUS pass over the whole operand -- it is not tile-shaped and must not
    # be inflated by 1/eff (the same reason `_fused_floor_ns` is applied here and not
    # inside `mem`). Measured: at cores=32 each extra iteration costs 0.90-1.24x a full
    # HBM pass over B at 150 GB/s (B = 2/4/8/16 MB -> 1.24/0.99/0.97/0.90), while the
    # marginal cost itself spans 5.8x -- so it scales with the operand, and is not a
    # fixed per-iteration overhead. 0 until the extractor sets per-arg `loop_factor`.
    mem_t += p.loop_reread_scale * _loop_reread_bytes(ops) / p.mm_bw_read_gbps

    if p.use_bundled_cost_model:
        compute = _matmul_ns_bundled(ops, p)
    else:
        compute = _matmul_ns_upstream(ops, p)
    # compute/HBM OVERLAP: the engine streams operands while the systolic array works,
    # so a kernel takes the LONGER of the two rather than their sum. For a non-matmul
    # bundle compute=0 -> t = mem_t (unchanged). The split re-read is charged AFTER the
    # overlap: it is not hidden by compute -- it is why the lopsided kernel runs long.
    #
    # A PARTIAL-overlap form was fitted and deliberately dropped; see the module
    # docstring. What remains below is the GATE, stated honestly. `loop_trip > 1`
    # currently decides ZERO rows (it is implied by tiles_output_dim on every record we
    # hold), so its "a loop can pipeline across iterations" rationale is UNTESTED -- it
    # is kept only as a guard for future non-looped output-tiled ops.
    # `tiles_output_dim` binds on just 8 rows (bmm_3d2d_k_tiling), which already
    # under-predict ~17 %, so those rows cannot
    # distinguish "reduction-tiled iterations are dependent" from "that op's memory term
    # is too small". What IS established is only that SOME gate is needed: removing both
    # regresses mmwd 15.1 -> 17.6 and bmm_layout 20.2 -> 25.5. Compute and memory
    # OVERLAP: the engine streams operands while the array works, so a kernel takes the
    # LONGER of the two rather than their sum.
    # LX RELAYOUT (shuffle): an ownership-change identity copy is its own DSC inside
    # the bundle, moves no HBM bytes, and DSCs in a bundle execute serially (the
    # allocator's half-tick lifetime scheme depends on exactly that) -- so it ADDS to
    # the kernel time rather than overlapping, and sits OUTSIDE the compute/memory
    # overlap term below. Measured directly: fusing never recovers it (fused total
    # == unfused sum +/- 2%), and it costs its full time beside a 223 us bmm exactly
    # as beside a 25 us relu, so it does not hide behind PT-array work either. See
    # relayout_ns() and the CostParams note for the fitted law and its validity range.
    rel_ns = sum(relayout_ns(o, p) for o in ops)
    t = compute + mem_t - p.overlap_gamma * min(compute, mem_t) + rel_ns
    # (A genuine-reduction cross-core ring-combine term once lived here; it is provably
    # bounded by ~cores * a tiny per-elem cost <= ~5 ns -- below run-to-run noise --
    # so it is dropped as inert. K is never split for matmul, so there is no matmul
    # analogue. A per-iteration coarse-tiling LOOP overhead (c_loop*L) also once lived
    # here, calibrated on the now-dropped chain/ctsum reduction-dim sweeps; no current
    # op exercises it, so it is removed rather than carried unvalidated.)
    return t


def predict_op(op: OpFeatures, params: CostParams | None = None) -> float:
    """Predicted device latency (ns) for a single op (as its own kernel)."""
    return predict_ops([op], params)


def _explain_matmul_bundled(lines: list, ops: list, p: CostParams) -> str:
    """BUNDLED (``CostParams.use_bundled_cost_model=True``) counterpart of the
    matmul branch in :func:`explain` -- pairs with ``_matmul_ns_bundled``. ``lines``
    already carries the per-op traffic breakdown; this appends the base/turnaround/
    underfill/compute/split-shape breakdown and returns the joined string.
    """
    R, W = _fused_hbm_bytes(ops)  # external input counted once (fused kernel)
    base = R / p.mm_bw_read_gbps + W / p.mm_bw_write_gbps
    turn = p.rw_turnaround_ns_per_byte * min(R, W)
    # Underfill derate (output-dim tiling): smallest per-core tile governs.
    eff, eff_rows = 1.0, 0.0
    for o in ops:
        if o.loop_trip > 1 and o.tiles_output_dim and o.tile_rows_per_core > 0:
            e = coarse_underfill_eff_matmul(o.tile_rows_per_core, p)
            if e < eff:
                eff, eff_rows = e, o.tile_rows_per_core
    # Matmul compute (additive): sum the per-op compute term for any matmul ops.
    mm_us, mm_lines = 0.0, []
    for o in ops:
        if o.is_matmul:
            pe = (
                1.0
                if o.tiles_output_dim
                else underfill_eff(
                    o.matmul_rows_per_core, p, p.underfill_target_passes_matmul
                )
            )
            mac_peak = _matmul_mac_peak(o, p)
            c_ns = o.matmul_macs / o.cores / (mac_peak * pe)
            mm_us += c_ns / 1000
            slow = (
                " [default-layout bmm slow rate]"
                if mac_peak != p.mac_peak_per_core_ns
                else ""
            )
            mm_lines.append(
                f"     compute = MACs/cores/(mac_peak*pt_eff) = {o.matmul_macs}/"
                f"{o.cores}/({mac_peak:.0f}*{pe:.3f}) = {c_ns / 1000:.2f}"
                f" us  (M/m={o.matmul_rows_per_core:.0f}, pt_eff={pe:.3f}){slow}"
            )
    t = predict_ops(ops, p)
    parts = "(R+W)/BW_PEAK + a*min(R,W)"
    if eff < 1.0:
        parts = f"[{parts}] / eff_underfill"
    if mm_us > 0:
        parts = f"compute + {parts}"
    lines.append(f"  -- prediction (turnaround, bundled matmul model): T = {parts} --")
    lines.append(f"     R={R}B (read)   W={W}B (write)")
    lines.extend(mm_lines)
    blab = f"R/{p.mm_bw_read_gbps:.0f} + W/{p.mm_bw_write_gbps:.0f}"
    lines.append(f"     base = {blab} = {base / 1000:.2f} us")
    lines.append(
        f"     turn = a*min(R,W) = {p.rw_turnaround_ns_per_byte}*{min(R, W)} "
        f"= {turn / 1000:.2f} us"
    )
    if eff < 1.0:
        shape = (
            f"({eff_rows:.1f}/{p.coarse_underfill_rfull_matmul:g})"
            f"**{p.coarse_underfill_exp_matmul}"
        )
        ceil_ = p.coarse_underfill_cap_matmul
        lines.append(
            f"     eff_underfill = min({ceil_}, {shape}) = {eff:.3f}  "
            f"-> (base+turn)/eff = {(base + turn) / eff / 1000:.2f} us"
        )
    split_us = 0.0
    for o in ops:
        if getattr(o, "is_matmul", False):
            m_dev = o.matmul_rows_per_core * o.matmul_m_split
            n_dev = o.matmul_cols_per_core * o.matmul_n_split
            lf, sf = (
                (o.matmul_m_split, o.matmul_n_split)
                if m_dev >= n_dev
                else (o.matmul_n_split, o.matmul_m_split)
            )
            area_exc = max(
                0.0, o.matmul_rows_per_core * o.matmul_cols_per_core - p.mm_split_area0
            )
            split_us += area_exc * (
                p.mm_split_reread_us_per_elem
                * max(0.0, log2(lf) - log2(p.mm_split_long_knee))
                + p.mm_split_short_us_per_elem
                * max(0.0, log2(sf) - log2(p.mm_split_short_knee))
            )
    if split_us > 0:
        lines.append(
            f"     split_reread = max(0,area-{p.mm_split_area0:.0f})*"
            f"[cL*log2(fan_long/{p.mm_split_long_knee:.0f}) + "
            f"cS*log2(fan_short/{p.mm_split_short_knee:.0f})] = {split_us:.2f} us"
        )
    lines.append(f"     => T_model = {t / 1000:.2f} us")
    return "\n".join(lines)


def group_features_by_bundle(
    operations: Sequence, features_by_buffer: Mapping[str, OpFeatures]
) -> list[list[OpFeatures]]:
    """Group caller-supplied per-op features into the bundles ``operations`` will become."""
    from .fusion import estimate_bundles

    bundles: list[list[OpFeatures]] = []
    for group in estimate_bundles(operations):
        feats: list[OpFeatures] = []
        for op in group:
            # ``getattr``: an op the caller could not model may be an extern or
            # fallback node carrying no buffer name at all. Either way -- no name,
            # or a name with no features -- it is skipped.
            name = getattr(op, "name", None)
            if name is not None and name in features_by_buffer:
                feats.append(features_by_buffer[name])
        if feats:  # a group with no modellable ops is not a bundle
            bundles.append(feats)
    return bundles


def predict_by_bundle(
    operations: Sequence,
    features_by_buffer: Mapping[str, OpFeatures],
    params: CostParams | None = None,
) -> float:
    """Predicted latency (ns) for ``operations``, scored one bundle at a time."""
    return sum(
        predict_ops(bundle, params)
        for bundle in group_features_by_bundle(operations, features_by_buffer)
    )


def explain(ops: list, params: CostParams | None = None) -> str:
    """Human-readable breakdown of the prediction for a bundle of ops."""
    p = params or CostParams()
    lines = []
    for o in ops:
        r, w, lx = o.read_bytes(), o.write_bytes(), o.lx_bytes()
        loop = f" loop_trip={o.loop_trip}" if o.loop_trip > 1 else ""
        pat = f" [{o.hbm_pattern}]" if getattr(o, "hbm_pattern", "") else ""
        lines.append(f"  {o.name:<12} read={r}B write={w}B lx={lx}B{loop}{pat}")
        for a in o.args:
            bc = " broadcast (loaded once)" if a.broadcast else ""
            lf = f" xL={a.loop_factor}" if a.loop_factor > 1 else ""
            counted = a.elems * a.loop_factor * o.dtype_bytes * (1 - a.is_lx)
            dev = a.dims if a.dims else [a.elems]
            log = f"torch {a.logical} -> " if a.logical else ""
            try:
                mem_repr = a.mem.upper()
            except ValueError:
                mem_repr = str(a.is_lx)
            # One line per DEVICE-LAYOUT tensor: name, role, logical->device dims,
            # residency, byte calc, the HBM bytes the model counts, and the loop factor.
            lines.append(
                f"      {a.role:<6} {a.name:<22} {log}device {dev} in {mem_repr}"
                f"  | {a.elems} elems x {o.dtype_bytes}B = {a.elems * o.dtype_bytes} B"
                f" (hbm counted: {counted} B){lf}{bc}"
            )
    if any(getattr(o, "is_matmul", False) for o in ops) and p.use_bundled_cost_model:
        return _explain_matmul_bundled(lines, ops, p)
    if any(getattr(o, "is_matmul", False) for o in ops):
        # Matmul compute comes from work_division._matmul_split_cost (HBM excluded --
        # see the module docstring), and the bundle memory term supplies the traffic.
        # Report the reconstructed axes each matmul op was priced with, then R/W.
        lines.append(
            "  -- prediction (matmul, via work_division._matmul_split_cost) --"
        )
        compute_ns = 0.0
        for o in ops:
            if not getattr(o, "is_matmul", False):
                continue
            axes = _matmul_axes_for_split_cost(o)
            if axes is None:
                raise RuntimeError(
                    f"matmul op {o.name!r} has unresolvable axes (missing "
                    "matmul_a_bytes/matmul_b_bytes) -- the UPSTREAM matmul cost model "
                    "cannot price it"
                )
            (B, b), (M, m), (N, n), (K, k), shared_weight = axes
            us = _matmul_split_cost(
                (B, b),
                (M, m),
                (N, n),
                (K, k),
                config.sencores,
                shared_weight=shared_weight,
                include_hbm=False,
            )
            cores_used = b * m * n * k
            if us == float("inf"):
                raise RuntimeError(
                    f"matmul op {o.name!r} has an infeasible core split "
                    f"(B={B}/{b}, M={M}/{m}, N={N}/{n}, K={K}/{k}, "
                    f"cores_used={cores_used})"
                )
            lines.append(
                f"     {o.name}: B={B}(/{b}) M={M:.0f}(/{m}) N={N:.0f}(/{n}) "
                f"K={K:.0f}(/{k}) shared_weight={shared_weight} "
                f"cores={cores_used} -> compute {us:.2f} us"
            )
            compute_ns += us * 1000.0
        R, W = _fused_hbm_bytes(ops)  # external input counted once (fused kernel)
        lines.append(
            f"     compute = {compute_ns / 1000:.2f} us   R={R}B (read) W={W}B (write)"
        )
        lines.append(f"     => T_model = {predict_ops(ops, p) / 1000:.2f} us")
        return "\n".join(lines)
    # Prediction with the rough calculation spelled out, so SPYRE_DUMP_COST shows the
    # same step-by-step breakdown (base + turnaround, then the underfill derate for
    # pointwise tiling).
    R, W = _fused_hbm_bytes(ops)  # external input counted once (fused kernel)
    base = (R + W) / p.bw_peak_gbps
    turn = p.rw_turnaround_ns_per_byte * min(R, W)
    # Underfill derate (output-dim tiling): smallest per-core tile governs.
    eff, eff_rows, eff_cols = 1.0, 0.0, 0.0
    for o in ops:
        if o.loop_trip > 1 and o.tiles_output_dim and o.tile_rows_per_core > 0:
            e = coarse_underfill_eff(o.tile_rows_per_core, _op_cols(o), p)
            if e < eff:
                eff, eff_rows, eff_cols = e, o.tile_rows_per_core, _op_cols(o)
    t = predict_ops(ops, p)
    parts = "(R+W)/BW_PEAK + a*min(R,W)"
    if eff < 1.0:
        parts = f"[{parts}] / eff_underfill"
    lines.append(f"  -- prediction (turnaround): T = {parts} --")
    lines.append(f"     R={R}B (read)   W={W}B (write)")
    lines.append(f"     base = (R+W)/{p.bw_peak_gbps:.0f} = {base / 1000:.2f} us")
    lines.append(
        f"     turn = a*min(R,W) = {p.rw_turnaround_ns_per_byte}*{min(R, W)} "
        f"= {turn / 1000:.2f} us"
    )
    if eff < 1.0:
        shape = (
            f"({eff_rows:.1f}/{p.coarse_underfill_rfull:g})"
            f"**{p.coarse_underfill_exp}"
            f" * ({eff_cols:.0f}/{p.coarse_underfill_col_ref:.0f})"
            f"**{p.coarse_underfill_col_exp}"
        )
        lines.append(
            f"     eff_underfill = min({p.coarse_underfill_cap}, {shape}) = {eff:.3f}"
            f"  -> (base+turn)/eff = {(base + turn) / eff / 1000:.2f} us"
        )
    lines.append(f"     => T_model = {t / 1000:.2f} us")
    return "\n".join(lines)

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

"""Predicted-runtime reporting over the pre-scheduling loop-level IR.

Runs immediately after ``CustomPreSchedulingPasses`` has finished, where the
loop-level IR is final: layouts resolved, restickify inserted, coarse-tiling applied,
work division committed, scratchpad placed. It partitions the graph into the kernels
the backend will build, prices each with the analytical model in ``cost_model.py``,
and returns a :class:`CostReport` carrying the program total plus a breakdown.

The report is a *value*, not just a printout, so another pass or an external tool can
compare two candidate plans without compiling and running either. That is the point:
``report_a.total_us`` vs ``report_b.total_us`` is a cheap plan comparison.

NOT related to ``work_division.cost_model_matmul_division``, which is a separate model
used to choose a matmul work division. This module is the analytical whole-program
runtime model documented in ``docs/source/compiler/cost_model.md``.

Disabled by default. When the flag is off this module does no work at all -- it returns
before touching the graph -- so leaving it off costs one attribute read per compilation.

HOW OPS ARE GROUPED, AND WHY IT MATTERS
---------------------------------------
``predict_ops`` is defined over ONE FUSED KERNEL and is deliberately not additive over
its ops: it de-duplicates external inputs shared inside a bundle and its turnaround and
overlap terms are ``min``/``max`` reductions over bundle totals. Pricing ops separately
and adding is therefore simply a different quantity -- measured across 521 recorded
multi-op bundles the difference ranges from -94 % to +33 %. It is zero on 32 % of them
(no shared external input, so the de-dup never fires) and, where it differs at all, an
UNDER-count about seven times in ten. So the grouping has to match what the backend
actually fuses.

``spyre_fuse_nodes`` (``fusion.py``) accumulates every *contiguous run* of Spyre nodes
into one bundle, with no size limit; only a non-Spyre node breaks the run. This module
mirrors that: contiguous modellable ops form one group. An earlier version made every
untiled op its own group, which under-predicted a 5-op softmax by 45 % because the real
kernel is one bundle, not five.

Coarse-tiling ``loop_group_id`` is still read, but for *labelling* the loop structure
inside a group -- not for deciding group boundaries.
"""

from __future__ import annotations

import dataclasses
import threading

from torch._inductor.ir import ComputedBuffer

from . import config
from .constants import DEVICE_NAME
from .cost_model import CostParams, _loop_reread_bytes, predict_ops
from .dump_cost_model import extract_op_features
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("cost_model_pass")

#: Truthy spellings, matching ``dump_cost_model.cost_dump_enabled`` so that "0" means
#: off. Gating on plain truthiness would enable the pass for SPYRE_DUMP_COST=0.
_ON = {"1", "true", "yes", "on"}

#: Report from the most recent run, for tools that want it without threading a return
#: value through the pipeline. Mirrors ``dump_cost_model.LAST_IO``, the existing
#: convention for handing extraction results out of a pass.
#:
#: PER-THREAD on purpose. ``torch.compile`` can be driven from several threads at once,
#: and each one runs this pipeline over its own graph. A plain module global would hand
#: a caller whichever graph happened to finish last -- silently, and most often under
#: load, which is exactly when a predicted-runtime number gets looked at. Each thread
#: reads back only the report for the graph it compiled itself.
#:
#: The RETURN VALUE of :func:`cost_model_pass` is the authoritative path and is
#: unaffected by any of this; ``LAST_REPORT`` is the convenience one.
_STATE = threading.local()

#: Declared for type checkers only -- deliberately unbound, so attribute access falls
#: through to ``__getattr__`` below and reaches the per-thread storage.
LAST_REPORT: CostReport | None


def __getattr__(name: str) -> "CostReport | None":
    # PEP 562. Keeps the documented ``cost_model_pass.LAST_REPORT`` spelling working
    # while the storage underneath is per-thread.
    if name == "LAST_REPORT":
        return getattr(_STATE, "report", None)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _enabled() -> bool:
    """True when the cost model should run. Anything unrecognised means off."""
    return str(config.cost_model or "").strip().lower() in _ON


@dataclasses.dataclass
class OpCost:
    """One operation's share of its group's predicted time.

    ``predicted_us`` is an ATTRIBUTION of the group total, not a standalone prediction
    for this op -- pricing ops separately would give a different number entirely.
    """

    name: str
    loop_group_id: tuple[int, ...] | None
    hbm_bytes: int
    #: On-chip traffic. An LX buffer is allocated per-tile, so inside a loop this is
    #: already ONE iteration's working set, not the whole loop's.
    lx_bytes: int
    predicted_us: float
    #: Iterations this op's loop runs (1 when it is not inside one).
    trip: int = 1
    #: Main-memory bytes ONE iteration moves. ``hbm_bytes`` is the whole loop; this is
    #: the per-pass working set, which is the figure re-tiling actually changes.
    hbm_per_iter: int = 0
    #: Of ``hbm_bytes``, the excess over a single pass -- input bytes fetched again on
    #: later iterations. See ``_per_iteration`` on why this is not always charged.
    reread_bytes: int = 0


@dataclasses.dataclass
class GroupCost:
    """One kernel: a contiguous run of ops the backend will fuse into one bundle."""

    index: int
    op_names: list[str]
    #: Distinct coarse-tiling loop groups represented in this kernel, for labelling.
    loop_group_ids: list[tuple[int, ...]]
    loop_trip: int
    predicted_us: float
    ops: list[OpCost]

    @property
    def has_loop(self) -> bool:
        return bool(self.loop_group_ids)


@dataclasses.dataclass
class CostReport:
    """Predicted runtime for one compiled graph.

    ``total_us`` is the headline: the sum over kernels. It is the number another
    component should compare between two candidate plans.
    """

    total_us: float
    groups: list[GroupCost]

    @property
    def n_modelled_ops(self) -> int:
        return sum(len(g.op_names) for g in self.groups)

    def format(self) -> str:
        lines = [
            f"predicted total: {self.total_us:10.1f} us over "
            f"{len(self.groups)} kernel(s), {self.n_modelled_ops} op(s)",
            "",
            f"  {'kernel':>10}  {'predicted us':>13}  ops",
        ]
        for g in sorted(self.groups, key=lambda x: -x.predicted_us):
            lines.append(f"  {g.index:>10}  {g.predicted_us:>13.1f}  {len(g.op_names)}")
            # Size the name column to this kernel: real names run from "mm" to
            # "constant_pad_nd_default", and a fixed width breaks the alignment.
            width = max((len(o.name) for o in g.ops), default=0)
            # One block per loop, so an iteration count is stated once for exactly the
            # ops it governs. A fill op sits OUTSIDE the loop and runs once beside ops
            # that run `trip` times, so there is no kernel-wide count to put in a
            # column.
            blocks = _by_loop(g.ops)
            show_headers = len(blocks) > 1 or any(o.trip > 1 for o in g.ops)
            for header, ops in blocks:
                indent = 2 if show_headers else 0
                if show_headers:
                    subtotal = sum(o.predicted_us for o in ops)
                    lines.append(f"  {'':>10}  {subtotal:>13.1f}    {header}")
                for o in sorted(ops, key=lambda x: -x.predicted_us):
                    lines.append(
                        f"  {'':>10}  {o.predicted_us:>13.1f}    "
                        f"{' ' * indent}{_traffic(o, width)}"
                    )
        lines += [
            "",
            "  Operations are grouped by the loop they run in; the count on each group",
            "  header is that loop's iterations.  HBM = main-memory traffic for the whole",
            "  loop, and '= T x B' breaks it into T iterations of B bytes.  LX = on-chip",
            "  scratchpad traffic, already ONE iteration (an LX buffer is allocated per",
            "  tile).  re-fetch = the part of HBM that is the same bytes read again on a",
            "  later iteration.",
            "",
            "  Each kernel is priced ONCE, as a bundle. The per-op column splits that",
            "  total by each op's share of the HBM bytes: an attribution, not a set of",
            "  independent predictions. It carries no compute term, so it misattributes a",
            "  compute-bound kernel, and because the weights are per-op while the total",
            "  de-duplicates shared inputs, a share can be off by up to a third of itself.",
            "  An op with no HBM traffic shows 0.0 -- its output stays in LX, so it adds",
            "  no main-memory traffic of its own.",
        ]
        return "\n".join(lines)


def _is_modellable(op) -> bool:
    """True when ``op`` belongs in a Spyre bundle.

    Both halves matter. ``spyre_fuse_nodes`` breaks a bundle at any node that is not
    on the Spyre device, and a CPU op's buffer IS a ``ComputedBuffer`` -- so testing
    the type alone neither breaks the run nor stops the op being priced as Spyre
    traffic. ``SpyreConstantFallback``/``SpyreEmptyFallback`` are ExternKernels and
    are excluded by the type test.
    """
    if not isinstance(op, ComputedBuffer):
        return False
    try:
        device = op.get_device()
    except Exception:  # noqa: BLE001 - best-effort; treat as a boundary
        return False
    return device is not None and device.type == DEVICE_NAME


def _by_loop(ops: list) -> list:
    """Split a kernel's ops into blocks, one per loop, in first-appearance order.

    A kernel has no single iteration count -- a ``coarse_tile_fill`` is inserted outside
    the loop and runs once beside ops that run ``trip`` times -- so the count belongs on
    a block of ops rather than in a column beside every op in the kernel. Ops are keyed
    by (loop id, trip): the id alone is absent on replayed features, and the trip alone
    would merge two distinct loops that happen to share a depth.
    """
    blocks: dict = {}
    for o in ops:
        blocks.setdefault((o.loop_group_id, o.trip), []).append(o)
    out = []
    for (gid, trip), members in blocks.items():
        if trip > 1:
            where = f"loop {'/'.join(str(i) for i in gid)}" if gid else "loop"
            out.append((f"{where}, runs {trip} times", members))
        else:
            out.append(("not in a loop, runs once", members))
    # Costliest first, matching how the ops inside each block are ordered.
    out.sort(key=lambda b: -sum(o.predicted_us for o in b[1]))
    return out


def _bytes(n: int) -> str:
    """Bytes at a scale that stays readable. MB rounds real 8 KB buffers to `0.0`."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MB"
    if n >= 1000:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"


def _traffic(o: "OpCost", width: int = 12) -> str:
    """One op's traffic: whole-loop total, one iteration's share, and the re-fetch.

    The whole-loop total alone is misleading in both directions -- it looks huge beside
    an untiled op, and it hides how much of itself is the same bytes fetched again.
    """
    hbm = f"HBM {_bytes(o.hbm_bytes):>9}" if o.hbm_bytes else f"HBM {'-':>9}"
    # An LX buffer is per-tile, so this is already one iteration either way.
    lx = f"   LX {_bytes(o.lx_bytes):>9}" if o.lx_bytes else ""
    per = (
        f" = {o.trip} x {_bytes(o.hbm_per_iter)}" if o.trip > 1 and o.hbm_bytes else ""
    )
    refetch = f"   re-fetch {_bytes(o.reread_bytes)}" if o.reread_bytes else ""
    return f"{o.name:<{width}} {hbm}{per}{lx}{refetch}".rstrip()


def _loop_group_id(op) -> tuple[int, ...] | None:
    gid = getattr(getattr(op, "loop_info", None), "loop_group_id", None)
    try:
        return tuple(gid) if gid else None
    except TypeError:  # a malformed id (e.g. a bare int) must not sink the report
        return None


def _per_iteration(f) -> dict:
    """Split one op's traffic into what a single loop iteration moves.

    The two memories report ``elems`` on different bases, so they split differently:

    * An **HBM** arg reports the UNTILED count, and ``hbm_bytes()`` already multiplies it
      by ``loop_factor``, so that total is whole-loop traffic and one iteration is simply
      ``hbm_bytes / trip``. This holds for an operand that advances (factor 1, whole
      tensor spread over the loop) and one the loop holds fixed (factor = trip, whole
      tensor fetched every pass) alike, and for the nested case in between -- ``mm`` under
      ``mm_nested_m_k`` really does carry ``1 < factor < trip``, so a rule that branched on
      "advancing or fixed" would be wrong there by up to 2.8x.
    * An **LX** arg is allocated per-tile, so ``lx_bytes()`` is ALREADY one iteration's
      on-chip working set and must not be divided again. See the comment at
      ``dump_cost_model._loop_features`` on HBM full-buffer vs LX per-tile extents.

    ``reread_bytes`` defers to ``cost_model._loop_reread_bytes`` rather than re-deriving
    the excess from ``loop_factor``, so the figure shown can never drift from the one
    charged. That function's scope is deliberately narrow -- an input of a matmul whose
    loop tiles an OUTPUT dim -- because under REDUCTION tiling ``loop_factor > 1`` means
    the opposite thing: each iteration consumes a fresh K-slice, so those inputs advance
    and it is only the accumulator that is re-touched. Reporting those as re-fetched
    would be wrong, not merely uncharged.
    """
    trip = max(1, int(getattr(f, "loop_trip", 1) or 1))
    return {
        "trip": trip,
        "hbm_per_iter": max(0, f.hbm_bytes()) // trip,
        "reread_bytes": int(_loop_reread_bytes([f])),
    }


def _price(feats, index, params) -> GroupCost | None:
    """Price one kernel. Returns None if the model cannot price it."""
    try:
        predicted_us = predict_ops(feats, params) / 1000.0
    except Exception:  # noqa: BLE001 - a bad group must not lose the whole report
        logger.warning("cost model could not price kernel %d; skipping", index)
        return None

    # Attribute by main-memory bytes. The weights are each op's OWN byte count while
    # the total de-duplicates inputs shared across the kernel, so the split is not
    # self-consistent: measured on recorded softmax bundles an op's share can be off by
    # up to a third of itself (e.g. 33.3 % where a consistent split gives 25.0 %), and
    # the error runs both ways -- de-dup shifts weight off the shared-input readers and
    # onto the op that owns the write. Kept because a consistent split would need the
    # reader set per input, which OpFeatures does not carry; flagged in format().
    # The price above is the answer; the attribution is only presentation, so a failure
    # here costs the breakdown for this kernel and nothing else.
    ops = []
    try:
        weights = [max(0, f.hbm_bytes()) for f in feats]
        total = sum(weights)
        for f, w in zip(feats, weights):
            ops.append(
                OpCost(
                    name=f.name,
                    loop_group_id=getattr(f, "_loop_group_id", None),
                    hbm_bytes=w,
                    lx_bytes=max(0, f.lx_bytes()),
                    predicted_us=predicted_us
                    * (w / total if total > 0 else 1.0 / max(1, len(feats))),
                    **_per_iteration(f),
                )
            )
    except Exception:  # noqa: BLE001 - keep the price, lose only the breakdown
        logger.warning("cost model could not attribute kernel %d; price kept", index)
        ops = []
    gids: list[tuple[int, ...]] = []
    for f in feats:
        gid = getattr(f, "_loop_group_id", None)
        if gid is not None and gid not in gids:
            gids.append(gid)
    return GroupCost(
        index=index,
        op_names=[f.name for f in feats],
        loop_group_ids=gids,
        loop_trip=max((getattr(f, "loop_trip", 1) or 1) for f in feats),
        predicted_us=predicted_us,
        ops=ops,
    )


def build_report(operations: list, params: CostParams | None = None) -> CostReport:
    """Price ``operations`` kernel by kernel, mirroring how the backend fuses.

    Split out from the pass so it can be unit-tested without a GraphLowering.
    """
    params = params or CostParams()

    # Contiguous runs of modellable ops become one kernel each, which is what
    # spyre_fuse_nodes does. Anything it cannot model breaks the run, exactly as a
    # non-Spyre node breaks a real bundle.
    runs: list[list] = []
    current: list = []
    # When symbolic bundle args are off the backend does not fuse at all
    # (fusion.py returns the node list untouched), so every op is its own kernel.
    fuses = bool(getattr(config, "bundle_symbolic_args", True))

    for op in operations:
        feats = None
        if _is_modellable(op):
            try:
                feats = extract_op_features(op)
                feats._loop_group_id = _loop_group_id(op)
            except Exception:  # noqa: BLE001 - skip ops the extractor cannot model
                feats = None
        if feats is None:
            if current:
                runs.append(current)
                current = []
            continue
        current.append(feats)
        if not fuses:  # no fusion -> one kernel per op
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    groups = [g for i, r in enumerate(runs) if (g := _price(r, i, params)) is not None]
    return CostReport(total_us=sum(g.predicted_us for g in groups), groups=groups)


def cost_model_pass(graph) -> CostReport | None:
    """Predict this graph's runtime; ``None`` when the model is disabled.

    Read-only: the graph is never mutated. Called after the pre-scheduling pipeline,
    where the loop-level IR is final.

    Returns the report so a caller can use it -- ``CustomPreSchedulingPasses`` exposes
    it as ``last_cost_report`` for exactly that reason.
    """
    _STATE.report = None
    if not _enabled():
        return None

    try:
        report = build_report(getattr(graph, "operations", None) or [])
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        # A sibling dump hook once raised ImportError here and broke every compile.
        # Reporting must never be able to do that.
        logger.warning("cost model pass failed: %r", exc)
        return None

    try:
        from .dump_common import banner, emit

        emit(
            banner("Cost model: predicted runtime (after pre-scheduling)")
            + "\n"
            + report.format()
            + "\n"
        )
    except Exception as exc:  # noqa: BLE001 - printing must not raise either
        logger.warning("cost model report could not be emitted: %r", exc)

    _STATE.report = report
    return report

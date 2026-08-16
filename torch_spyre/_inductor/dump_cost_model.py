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

"""Extract cost-model features from the after-pre-scheduling LoopLevel IR.

Walks ``graph.operations`` and builds :class:`cost_model.OpFeatures` per op
(per-core cores, per-tensor-arg bytes + HBM/LX residency + broadcast flags),
then a dump hook (``SPYRE_DUMP_COST=1``) prints the features and the predicted
device latency so it can be compared against the measured value on hardware.

Extraction is best-effort and defensive: anything it can't resolve falls back to
a safe default and never raises into compilation. The numbers must be validated
against device measurements (``examples/bench_*``); the model is only as good as
this extraction.
"""

import os

from torch._inductor.ir import ComputedBuffer

from .constants import BATCH_MATMUL_OP
from .cost_model import ArgTraffic, OpFeatures, explain
from .pass_utils import apply_splits_from_index_coeff, iteration_space_from_op


def cost_dump_enabled() -> bool:
    return os.environ.get("SPYRE_DUMP_COST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _int(x, default: int = 1) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _prod_ints(seq) -> int:
    n = 1
    for s in seq:
        n *= _int(s, 1)
    return n


def _op_name(op) -> str:
    data = getattr(op, "data", None)
    node = getattr(data, "origin_node", None)
    if node is not None:
        return getattr(node, "name", None) or str(getattr(node, "target", node))
    rtype = getattr(data, "reduction_type", None)
    if rtype:
        return str(rtype)
    return type(data).__name__ if data is not None else op.get_operation_name()


def _cores(op) -> int:
    splits = getattr(op, "op_it_space_splits", None)
    if not splits:
        return 1
    try:
        rw = op.get_read_writes()
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        it_space = iteration_space_from_op(op)
        readable = apply_splits_from_index_coeff(
            splits, write_index, read_index, it_space
        )
        return _prod_ints(readable.values())
    except Exception:  # noqa: BLE001 - best-effort feature extraction
        return 1


def _mem_of_layout(layout) -> str:
    alloc = getattr(layout, "allocation", None)
    if isinstance(alloc, dict) and "lx" in alloc:
        return "lx"
    return "hbm"


def _device_dims(layout):
    """Stick-padded DEVICE dims (e.g. [4, 512, 64]) from a committed FixedTiledLayout
    -- the TRUE shape that moves (sticks are 64 fp16 elems; a row of N rounds up to
    ceil(N/64)*64). None when the device layout isn't available (use logical instead).
    """
    dl = getattr(layout, "device_layout", None)
    ds = getattr(dl, "device_size", None) if dl is not None else None
    if not ds:
        return None
    try:
        return [_int(x, 1) for x in ds]
    except Exception:  # noqa: BLE001 - symbolic/unresolved
        return None


def _input_traffic(name: str):
    """(mem, dims, elems, logical) for a read buffer -- ``dims`` is the device (stick)
    shape, ``logical`` the torch shape, ``elems`` the device product (logical fallback
    if the device layout isn't committed). So a reduction's reduced input is naturally
    full-sized, with no reduction_size scaling. Returns (None, None, None, None) if the
    buffer can't be resolved (caller falls back)."""
    try:
        from torch._inductor.virtualized import V

        buf = V.graph.get_buffer(name)
        if buf is not None:
            layout = buf.get_layout()
            dims = _device_dims(layout)
            logical = [_int(x, 1) for x in buf.get_size()]
            elems = _prod_ints(dims) if dims else _prod_ints(logical)
            return _mem_of_layout(layout), (dims if dims else logical), elems, logical
    except Exception:  # noqa: BLE001 - graph inputs / unresolved
        pass
    return None, None, None, None


def _loop_features(op):
    """(loop_trip, tiles_reduction_dim, tiles_output_dim) from the coarse-tiling
    ``loop_info`` (loop_info.py / coarse_tile.py). ``loop_trip`` = product of
    loop_count (1 if not tiled). tiles_reduction_dim = loop_tiled_reduction_dims is
    non-empty (reduction-dim tiling); tiles_output_dim = loop_tiled_dims is non-empty
    (output / pointwise-dim tiling). NOTE the fill/combine ops carry the same loop_info
    but tile NEITHER (both lists empty), so their accumulators stay fixed (factor L); a
    genuinely tiled op's args advance (factor 1)."""
    li = getattr(op, "loop_info", None)
    if li is None:
        return 1, False, False
    trip = 1
    for c in getattr(li, "loop_count", None) or []:
        trip *= _int(c, 1)
    red_dims = getattr(li, "loop_tiled_reduction_dims", None) or []
    out_dims = getattr(li, "loop_tiled_dims", None) or []
    return (
        max(1, trip),
        any(bool(level) for level in red_dims),
        any(bool(level) for level in out_dims),
    )


def _tiled_symbols_per_level(op):
    """Per NESTING LEVEL, the set of loop symbols that level tiles.

    ``CoarseTileInfo`` stores the tiled dims as HOST-RANGE indices, one list per level
    (``loop_tiled_dims`` for output dims, ``loop_tiled_reduction_dims`` for reduction
    dims), while the index expressions are written in iteration-space symbols. The
    iteration space SKIPS unit-size ranges, so a host index must be mapped through the
    non-unit ranges to reach the right symbol -- the same ``host_to_it`` correction
    ``spyre_kernel.py`` applies when it builds ``tiled_syms``.

    Returns ``[(trip, {symbols}), ...]`` outermost-first, or ``[]`` when the op is not
    coarse-tiled.

    IR-verified on ``mm_nested_m_k`` (M outer, K inner)::

        loop_count               = [2, 4]
        loop_tiled_dims          = [[0], []]      # level 0 tiles output dim 0 -> i0
        loop_tiled_reduction_dims= [[],  [0]]     # level 1 tiles reduction dim 0 -> r0_0
    """
    li = getattr(op, "loop_info", None)
    if li is None:
        return []
    counts = list(getattr(li, "loop_count", None) or [])
    out_lv = list(getattr(li, "loop_tiled_dims", None) or [])
    red_lv = list(getattr(li, "loop_tiled_reduction_dims", None) or [])
    n_levels = max(len(out_lv), len(red_lv))
    if not n_levels:
        return []
    try:
        data = op.data
        ranges = list(getattr(data, "ranges", []) or [])
        rranges = list(getattr(data, "reduction_ranges", []) or [])
        it_syms = list(iteration_space_from_op(op).keys())
    except Exception:  # noqa: BLE001 - best-effort feature extraction
        return []

    # host-range index -> iteration-space position, skipping unit-size ranges.
    def _host_to_it(rs, offset):
        m, pos = {}, offset
        for host_idx, r in enumerate(rs):
            if _int(r, 1) != 1:
                m[host_idx] = pos
                pos += 1
        return m, pos

    out_map, n_out = _host_to_it(ranges, 0)
    red_map, _ = _host_to_it(rranges, n_out)

    levels = []
    for lv in range(n_levels):
        syms = set()
        declared = 0
        for h in out_lv[lv] if lv < len(out_lv) else []:
            declared += 1
            p = out_map.get(_int(h, -1))
            if p is not None and p < len(it_syms):
                syms.add(it_syms[p])
        for h in red_lv[lv] if lv < len(red_lv) else []:
            declared += 1
            p = red_map.get(_int(h, -1))
            if p is not None and p < len(it_syms):
                syms.add(it_syms[p])
        trip = max(1, _int(counts[lv], 1) if lv < len(counts) else 1)
        # ``declared`` is kept separate from ``syms`` so the two ways a level can end up
        # with no symbols are not conflated: an op that tiles NOTHING at this level
        # (declared == 0) is loop-invariant there and every arg repeats, whereas a level
        # whose declared dims could not be resolved to symbols is unknown and must not
        # be guessed. See _loop_factor_for_index.
        levels.append((trip, syms, declared))
    return levels


def _loop_factor_for_index(index, levels) -> int:
    """How many times traffic at ``index`` is transferred over the whole loop nest.

    An operand is re-transferred at a level whose tiled symbols do NOT appear in its
    index (it is re-entered at the same address each iteration of that level), and is
    walked -- transferred once in total -- at a level whose tiled symbol it does carry.
    So the multiplier is the PRODUCT over levels::

        factor = prod( trip[L] if index has no tiled symbol of level L else 1 )

    A single per-op scalar cannot express this: ``mm_nested_m_k``'s OUTPUT advances at
    level 0 (its index has ``i0``) and repeats at level 1 (no ``r0_0``), giving 1*4 = 4,
    while its B operand does the opposite, giving 2*1 = 2. IR-verified factors for that
    op at t=4 are out=4, A=1, B=2 -- the extractor previously emitted 1/1/1.
    """
    if not levels:
        return 1
    try:
        free = set(getattr(index, "free_symbols", None) or ())
    except Exception:  # noqa: BLE001
        return 1
    factor = 1
    for trip, syms, _declared in levels:
        # An arg REPEATS at a level whenever that level's tiled symbols are absent from
        # its index -- for EITHER reason:
        # * the level tiles nothing this op has (`coarse_tile_fill` / `_combine`, whose
        # loop_info names a dim they do not iterate), or * the level tiles a dim this
        # arg's address does not depend on (matmul B under M-tiling). Both mean the same
        # physical thing: the op re-enters the same address each iteration of that
        # level. An earlier version guarded this with `if syms`, which silently dropped
        # the first case and under-counted one K-tiled bundle 552 MB -> 216 MB, moving
        # the control op from -6.3 % to -64.2 %.
        if not (syms & free):
            factor *= trip
    return factor


def _row_split(op, default: int) -> int:
    """Core split of the ROW (partition) device dim = the output var with the largest
    write-index coefficient (the outer/row dim; the stick dim has the smallest). Used so
    ``tile_rows_per_core`` divides by the cores actually on the rows, not total cores --
    they differ once the planner splits columns instead (extreme tiling). ``default``
    (usually total cores) on any failure -> the prior all-cores-on-rows behavior.
    """
    try:
        splits = getattr(op, "op_it_space_splits", None)
        if not splits:
            return default
        rw = op.get_read_writes()
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        it_space = iteration_space_from_op(op)
        readable = apply_splits_from_index_coeff(
            splits, write_index, read_index, it_space
        )
        out_vars = [
            (abs(int(write_index.coeff(s))), s)
            for s in it_space
            if write_index.coeff(s) != 0
        ]
        if not out_vars:
            return default
        out_vars.sort(key=lambda t: t[0])  # largest coeff = row (outer) dim
        return max(1, int(readable.get(out_vars[-1][1], default)))
    except Exception:  # noqa: BLE001 - best-effort feature extraction
        return default


def _matmul_features(
    op,
    out_elems: int,
    dtype_bytes: int,
    loop_trip: int = 1,
    tiles_red_dim: bool = False,
):
    """(macs, rows_per_core, cols_per_core, a_bytes, b_bytes, k_split, m_split, n_split).

    ``macs`` = the TOTAL multiply-accumulates the op performs across the WHOLE coarse
    loop, never a per-iteration slice. That distinction used to leak into the feature
    file: ``out_elems`` comes from the committed device layout and ``k_size`` from
    ``reduction_ranges``, and when the coarse loop tiles the REDUCTION dim each iteration
    sees only ``K/loop_trip``, so the raw product came out ``TOTAL/loop_trip``. When the
    loop tiles only an OUTPUT dim the output buffer is full-extent and the raw product is
    already the total. The consumer (cost_model.predict_ops) multiplies nothing by
    ``loop_trip``, so the reduction-tiled ops were under-counting compute by up to 16x.
    ``tiles_reduction_dim`` is exactly the discriminator -- it predicts the convention on
    all six coarse ops measured (row_tiling total; k_tiling / nested / bmm_k / bmm_nested
    / bmm_3d2d per-tile) -- so the factor is applied here, once, at the source. ``rows_per_core`` = M/m (drives pt_eff + A re-read),
    ``cols_per_core`` = N/n (drives B re-read). ``a_bytes`` = |A| = M*K, ``b_bytes`` =
    |B| = K*N (device dtype). ``k_split``/``m_split``/``n_split`` = the K/M/N core splits.
    M/N/K + splits are recovered from the iteration space: reduction (K) vars have coeff 0
    in the write index. Among the OUTPUT vars the batch is EXCLUDED -- a 3D [B,M,N] bmm
    output puts the batch at the LARGEST write-index coeff, so the old "largest coeff = M"
    mis-picked batch as M for B>=2 (rows_per_core came out as the batch size). M/N are taken
    from the named-dim map when present (work_div-hinted runs) else from the two smallest
    coeffs (M the larger, N the stick/inner). Falls back to zeros/1 on any failure -> the
    model drops the spill (safe for the validated balanced regime).
    """
    data = getattr(op, "data", None)
    k_size = _prod_ints(getattr(data, "reduction_ranges", None) or [])
    # Scale a reduction-tiled slice back up to the whole-loop total (see docstring).
    macs = out_elems * k_size * (loop_trip if tiles_red_dim else 1)
    rows_per_core = cols_per_core = 0.0
    a_bytes = b_bytes = 0
    k_split = m_split = n_split = 1
    try:
        splits = getattr(op, "op_it_space_splits", None)
        if splits:
            rw = op.get_read_writes()
            write_index = next(iter(rw.writes)).index
            read_index = next((d.index for d in rw.reads), write_index)
            it_space = iteration_space_from_op(op)
            readable = apply_splits_from_index_coeff(
                splits, write_index, read_index, it_space
            )
            out_vars = []
            for s in it_space:
                wc = write_index.coeff(s)
                if wc != 0:
                    out_vars.append((abs(int(wc)), s))
                else:  # reduction (K) dim -> contributes to the K-split
                    k_split *= max(1, int(readable.get(s, 1)))
            if out_vars:
                # Identify M (row/outer) and N (stick/inner), EXCLUDING batch. Prefer
                # the exact named-dim map (present on work_div-hinted runs); else drop
                # the largest-coeff var(s) as batch and take M/N from the two smallest
                # coeffs.
                m_sym = n_sym = None
                wdli = getattr(op, "work_div_loop_info", None)
                if wdli:
                    for _, s in out_vars:
                        names = wdli.get(s, ())
                        if m_sym is None and "M" in names:
                            m_sym = s
                        elif n_sym is None and "N" in names:
                            n_sym = s
                if m_sym is None or n_sym is None:
                    ordered = sorted(out_vars, key=lambda t: t[0])  # ascending by coeff
                    mn = ordered[
                        :2
                    ]  # two smallest = (N, M); larger-coeff vars are batch
                    m_sym = mn[-1][1]
                    n_sym = mn[0][1] if len(mn) >= 2 else None
                m_size = _int(it_space[m_sym], 1)
                n_size = _int(it_space[n_sym], 1) if n_sym is not None else 1
                m_split = max(1, int(readable.get(m_sym, 1)))
                n_split = (
                    max(1, int(readable.get(n_sym, 1))) if n_sym is not None else 1
                )
                if m_size > 1:
                    rows_per_core = m_size / m_split
                if n_size > 1:
                    cols_per_core = n_size / n_split
                a_bytes = m_size * k_size * dtype_bytes
                b_bytes = k_size * n_size * dtype_bytes
    except Exception:  # noqa: BLE001 - best-effort feature extraction
        rows_per_core = cols_per_core = 0.0
        a_bytes = b_bytes = 0
        k_split = m_split = n_split = 1
    return (
        macs,
        rows_per_core,
        cols_per_core,
        a_bytes,
        b_bytes,
        k_split,
        m_split,
        n_split,
    )


def _hbm_pattern(op, is_reduction: bool, out_dims) -> str:
    """Access-pattern effective-BW tag, read straight from the LoopLevel IR.

    Reuses the same "a var's coefficient in the write vs read index" decode ``_cores``
    uses for the matmul K-dim (stick var = coeff 1; reduced var = coeff 0 in the write):
      "stick_scatter": a device dim <64 sits just INSIDE the 64-stick -- a cat on a
          partition dim (cat0 device_size [...,2,64]) -> fine sub-stick interleave (slow).
      "restickify"   : the WRITE stick var is READ with coeff != 1 -> the stick dim is
          remapped (transpose) -- less turnaround, faster.
      "reduce_outer" : a REDUCED var is READ with coeff != 1 -> the reduction runs across
          rows/outer, not within the stick (sumcol).
    "" -> ordinary contiguous access; the default bw_peak + turnaround applies.
    """
    try:
        rw = op.get_read_writes()
        write_index = next(iter(rw.writes)).index
        it_space = iteration_space_from_op(op)

        def _c(idx, s) -> int:
            try:
                return int(idx.coeff(s))
            except Exception:  # noqa: BLE001
                return 0

        read_syms: set = set()
        for dep in rw.reads:
            ri = getattr(dep, "index", None)
            if ri is not None:
                read_syms |= getattr(ri, "free_symbols", None) or set()
        out_vars = [s for s in it_space if _c(write_index, s) != 0]
        stick = [s for s in it_space if _c(write_index, s) == 1]  # kept inner/stick var
        reduced = [s for s in it_space if _c(write_index, s) == 0]  # reduced-away vars
        # A CONCAT copies its input into an output dim absent from the read index (the
        # concat "which-copy" var, read-coeff 0). cat0 (concat on a PARTITION dim)
        # wedges a small (<64) device dim just inside the 64-stick -> fine sub-stick
        # interleave. (Gated on the concat dim so a mere permutation like
        # transpose_outer -- whose small outer dim also lands at [-2] -- is NOT mistaken
        # for it.)
        concat = any(s not in read_syms for s in out_vars)
        if (
            concat
            and out_dims
            and len(out_dims) > 3
            and 0 < _int(out_dims[-2], 64) < 64
        ):
            return "stick_scatter"
        for dep in rw.reads:
            ri = getattr(dep, "index", None)
            syms = getattr(ri, "free_symbols", None) or set()
            if ri is None:
                continue
            # reduce_outer: a REDUCED var read with coeff != 1 (across rows/outer) WHILE
            # a stick dim is kept in the output (sumcol). A full reduction to a scalar
            # (sumall) keeps no stick -> stays default (it is fast, not cross-row).
            if is_reduction:
                if stick and any(s in syms and abs(_c(ri, s)) > 1 for s in reduced):
                    return "reduce_outer"
            # restickify: the WRITE stick var is READ with coeff != 1 (transpose).
            elif any(s in syms and _c(ri, s) not in (0, 1) for s in stick):
                return "restickify"
        return ""
    except Exception:  # noqa: BLE001 - best-effort feature extraction
        return ""


def extract_op_features(op) -> OpFeatures:
    """Build OpFeatures for one ComputedBuffer op (best-effort)."""
    data = getattr(op, "data", None)
    is_reduction = getattr(data, "reduction_type", None) is not None
    loop_trip, tiles_red_dim, tiles_out_dim = _loop_features(op)
    # An arg ADVANCES (factor 1, walks the full tensor once across tiles) when this op
    # tiles a dim the arg traverses: an OUTPUT (pointwise) dim -> all args advance; a
    # REDUCTION dim -> only the reduced input advances. An arg is FIXED (factor L,
    # re-accessed each iteration) when this op tiles neither but shares the loop -- a
    # combine's accumulator / a per-tile partial. (fill/combine: loop_tiled_dims and
    # loop_tiled_reduction_dims are both empty, so out/red are False -> factor L.)
    is_tiled_red = is_reduction and tiles_red_dim
    dtype_bytes = _int(getattr(op.get_dtype(), "itemsize", 2), 2)
    out_size = list(op.get_size())
    # TRUE I/O sizes come from the committed DEVICE layout (sticks), not the torch
    # logical shape -- a row of N fp16 rounds up to ceil(N/64)*64, and reduction/
    # broadcast operands carry their own device size.
    out_dims = _device_dims(op.get_layout()) or out_size
    out_elems = _prod_ints(out_dims)

    cores = _cores(op)

    # Cross-core ring combine: work division splits OUTPUT dims first, then the reduced
    # axis with leftover cores -> the reduced axis is split only when out_elems < cores.
    # Approx k as the cores not absorbed by the output (refine if rung 11 needs it).
    reduction_cores = 1
    if is_reduction and out_elems < cores:
        reduction_cores = max(1, cores // max(1, out_elems))

    out_mem = _mem_of_layout(op.get_layout())

    # Matmul (batchmatmul reduction): compute-bound -> extra additive compute term. Pull
    # MACs (M*N*K), the per-core M tile (pt_eff), and the K-split k (-> reduction_cores,
    # so the existing combine term becomes the PSUM ring). Non-matmul ops keep is_matmul
    # False and the generic reduction_cores above.
    is_matmul = getattr(data, "reduction_type", None) == BATCH_MATMUL_OP
    matmul_macs, matmul_rows_per_core, matmul_cols_per_core = 0, 0.0, 0.0
    matmul_a_bytes = matmul_b_bytes = 0
    matmul_m_split = matmul_n_split = 1
    if is_matmul:
        (
            matmul_macs,
            matmul_rows_per_core,
            matmul_cols_per_core,
            matmul_a_bytes,
            matmul_b_bytes,
            k_split,
            matmul_m_split,
            matmul_n_split,
        ) = _matmul_features(op, out_elems, dtype_bytes, loop_trip, is_tiled_red)
        reduction_cores = k_split

    # Per-core per-tile pass-row height for the UNDERFILL derate -- only for OUTPUT-dim
    # (pointwise) tiling (a reduction's tiny output has no pass-row height). The "rows"
    # is the partition device dim (out_dims[-2]); an HBM full-buffer output reports the
    # UNTILED height, so divide by loop_trip to recover the per-tile slice, whereas an
    # LX intermediate is already allocated per-tile. Then divide by the ROW-dim core
    # split -- NOT total cores: at extreme tiling the planner may split COLUMNS instead
    # (rows/tile < col-sticks), leaving each core a full row tile (no underfill). 0.0 =
    # N/A -> no derate.
    tile_rows_per_core = 0.0
    if tiles_out_dim and loop_trip > 1 and cores > 0 and len(out_dims) >= 2:
        # Row extent from the LOGICAL shape, not the device shape. ``out_dims[-2]`` is
        # the row count only for a rank-2 tensor, whose device layout is rank-3. A
        # rank-3 or rank-4 tensor has a rank-4/5 device layout in which [-2] is a
        # degenerate or batch axis: a rank-4 flash tensor [1,4,1024,128] lays out as
        # [4,1024,2,1,64], so [-2] is 1, and a rank-3 bmm output [2,1024,1024] lays out
        # as [1024,16,2,64], so [-2] is the batch (2). Both then divide by loop_trip and
        # the core split, producing sub-unity "rows per core" -- 0.008 on flash -- which
        # drove coarse_underfill_eff to ~0.007 and inflated the memory term 60-248x.
        # ``logical[-2]`` is the row extent at every rank. Verified equal to the old
        # value on all 1177 recorded rank-2 tiled ops, so this changes nothing that was
        # previously modelled; it only repairs rank>=3. Same class of mistake, and the
        # same fix, as _matmul_features' batch-dim exclusion above.
        rows = (out_size[-2] if len(out_size) >= 2 else 0) or out_dims[-2]
        if out_mem != "lx":  # full-buffer alloc: per-tile slice is rows / loop_trip
            rows = rows / loop_trip
        # `loop_trip > 1` is guaranteed by the branch condition; `_row_split` can in
        # principle return 0 if a split map ever records one, and this term is a
        # diagnostic -- a ZeroDivisionError here would take down a compile for a number
        # nothing depends on. Guard locally rather than rely on the caller's condition.
        split = _row_split(op, cores) or 1
        tile_rows_per_core = rows / split

    # PER-ARG, PER-LEVEL loop factors. An operand is re-transferred at a nesting level
    # whose tiled symbol its index does NOT contain, and walked (transferred once) at a
    # level whose symbol it does; the multiplier is the product over levels. See
    # `_loop_factor_for_index`.
    #
    # This replaces two PER-OP scalars that could not express the nested case:
    #     out_factor = 1 if tiles_out_dim else loop_trip
    #     in_factor  = 1 if (tiles_out_dim or is_tiled_red) else loop_trip
    # IR-verified consequences at 4096x2048x2048, t=4 (out / A / B):
    #     matmul_k_tiling    4 / 1 / 1   -- old rule already correct (and it is the
    #                                      best-scoring coarse op, 7.9 % RMS)
    #     matmul_row_tiling  1 / 1 / 4   -- old rule gave B=1; B is invariant in M
    #     mm_nested_m_k      4 / 1 / 2   -- old rule gave 1/1/1. The OUTPUT advances at
    #                                      level 0 (index has i0) and repeats at level 1
    #                                      (no r0_0) => 1*4; B does the opposite => 2*1.
    _levels = _tiled_symbols_per_level(op)
    try:
        _rw = op.get_read_writes()
        _write_index = next(iter(_rw.writes)).index
    except Exception:  # noqa: BLE001 - best-effort feature extraction
        _write_index = None
    if _levels and _write_index is not None:
        out_factor = _loop_factor_for_index(_write_index, _levels)
    else:  # no loop_info (or unreadable index) -> the pre-existing behaviour
        out_factor = 1 if tiles_out_dim else loop_trip
    in_factor = 1 if (tiles_out_dim or is_tiled_red) else loop_trip

    args: list = []
    # Output arg (device-sized).
    args.append(
        ArgTraffic(
            name=op.get_operation_name(),
            role="output",
            mem=out_mem,
            elems=out_elems,
            dims=list(out_dims),
            logical=list(out_size),
            loop_factor=out_factor,
        )
    )
    # Input args, from the op's reads. Each read is sized by ITS OWN buffer's device
    # layout -- so a reduction's reduced input is naturally full-sized (no separate
    # reduction scaling), and a broadcast operand carries its real (one-row) size.
    try:
        reads = op.get_read_writes().reads
    except Exception:  # noqa: BLE001
        reads = []
    n_out_vars = len(out_size)
    for dep in reads:
        name = getattr(dep, "name", "?")
        index = getattr(dep, "index", None)
        # Broadcast heuristic: the read index references fewer loop variables than
        # the output rank -> it is loaded ONCE and reused across the broadcast dim, so
        # it is counted at its own (small) device size, not the output size. This
        # INCLUDES scalars/constants (0 loop vars, e.g. the `1.0` in `x + 1.0`): a
        # scalar is the maximally-broadcast input -- its one-load size is ~1 stick, so
        # it costs ~nothing, but it is no longer forced to exactly zero.
        broadcast = False
        try:
            n_index_vars = len(getattr(index, "free_symbols", []) or [])
            broadcast = n_index_vars < n_out_vars
        except Exception:  # noqa: BLE001
            broadcast = False
        mem, dims, in_elems, in_logical = _input_traffic(name)
        if in_elems is None:  # unresolved buffer -> fallback
            # A broadcast operand with no resolvable buffer (e.g. a scalar constant)
            # is loaded once and is at most ~1 element -- do NOT inflate it to the
            # output size. Only a NON-broadcast unresolved read is conservatively
            # sized at the full output.
            if broadcast:
                mem, dims, in_elems, in_logical = "hbm", [1], 1, [1]
            else:
                mem, dims, in_elems, in_logical = "hbm", list(out_dims), out_elems, []
        args.append(
            ArgTraffic(
                name=name,
                role="input",
                mem=mem,
                elems=in_elems,
                broadcast=broadcast,
                dims=list(dims),
                logical=list(in_logical) if in_logical else [],
                # Per-arg: this read's OWN index decides which levels it repeats at.
                loop_factor=(
                    _loop_factor_for_index(index, _levels)
                    if (_levels and index is not None)
                    else in_factor
                ),
            )
        )

    return OpFeatures(
        name=_op_name(op),
        is_reduction=is_reduction,
        out_elems=out_elems,
        cores=cores,
        dtype_bytes=dtype_bytes,
        args=args,
        reduction_cores=reduction_cores,
        loop_trip=loop_trip,
        tiles_output_dim=tiles_out_dim,
        tiles_reduction_dim=is_tiled_red,
        tile_rows_per_core=tile_rows_per_core,
        is_matmul=is_matmul,
        matmul_macs=matmul_macs,
        matmul_rows_per_core=matmul_rows_per_core,
        matmul_cols_per_core=matmul_cols_per_core,
        matmul_a_bytes=matmul_a_bytes,
        matmul_b_bytes=matmul_b_bytes,
        matmul_m_split=matmul_m_split,
        matmul_n_split=matmul_n_split,
        hbm_pattern="" if is_matmul else _hbm_pattern(op, is_reduction, out_dims),
    )


def extract_features(operations: list) -> list:
    """Build OpFeatures for every ComputedBuffer op in the graph."""
    feats = []
    for op in operations:
        if isinstance(op, ComputedBuffer):
            try:
                feats.append(extract_op_features(op))
            except Exception:  # noqa: BLE001 - skip ops we can't model
                continue
    return feats


# Totals + per-arg detail from the most recent extraction, using the DEVICE-layout
# byte accounting. Tools (e.g. examples/profile_ops.py) read this to get the model's
# I/O size and verify BW = hbm_bytes / kernel_time, without re-parsing the printed dump.
# LAST_FEATS holds the raw OpFeatures so a tool can call cost_model.predict_ops() to get
# the model's estimated kernel time.
LAST_IO: dict = {}
LAST_FEATS: list = []


def _record_last_io(feats: list) -> None:
    global LAST_IO, LAST_FEATS
    LAST_FEATS = list(feats)
    ops = []
    for o in feats:
        args = []
        for a in o.args:
            bs = a.elems * o.dtype_bytes
            # Every HBM arg counts at its own size x loop_factor (L for a per-tile
            # accumulator re-accessed each loop iteration, 1 otherwise); broadcast
            # operands carry their small one-load size (counted, not zeroed). LX ~free.
            counted = bs * a.loop_factor if a.mem == "hbm" else 0
            args.append(
                {
                    "name": a.name,
                    "role": a.role,
                    "mem": a.mem,
                    "dims": list(a.dims) if a.dims else [a.elems],
                    "logical": list(a.logical),
                    "elems": a.elems,
                    "loop_factor": a.loop_factor,
                    "bytes": bs,
                    "hbm_counted": counted,
                    "broadcast": a.broadcast,
                }
            )
        ops.append({"name": o.name, "is_reduction": o.is_reduction, "args": args})
    LAST_IO = {
        "hbm_bytes": sum(o.hbm_bytes() for o in feats),
        "lx_bytes": sum(o.lx_bytes() for o in feats),
        "ops": ops,
    }


def dump_cost_model(operations: list) -> None:
    """Print per-op cost features + predicted latency; no-op unless SPYRE_DUMP_COST.

    Treats the whole op list as one bundle (matching full fusion, e.g. softmax);
    for single-op example programs this is just that op.
    """
    if not cost_dump_enabled():
        return
    from .dump_common import banner, emit

    try:
        feats = extract_features(operations)
        _record_last_io(feats)
        bar = banner("Cost model features + prediction (after pre-scheduling)")
        emit(f"{bar}\n{explain(feats)}\n")
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        emit(f"[SPYRE_DUMP_COST] failed: {exc!r}")

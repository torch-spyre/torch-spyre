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


import dataclasses
from typing import Any

import regex as re
import sympy

import torch
import torch.compiler
import torch.fx.traceback
from torch._dynamo.symbolic_convert import InstructionTranslator
from torch._inductor.ir import Operation

from .logging_utils import get_inductor_logger
from .patches import OBSERVER_HOOKS_KEY

logger = get_inductor_logger("propagate_hints")


@dataclasses.dataclass
class DimHint:
    dim_names: list[str]  # e.g. ["A"]
    split_count: int  # tile count; 1 for a for_each_tile level (see loop_var_range)
    loop_var: "sympy.Symbol | None"  # the loop variable (e.g. c0, c1) for this dim;
    # None when op is broadcast w.r.t. this hint scope
    is_reduction: bool
    hint_id: int = 0  # the _hint_N counter value identifying the scope
    loop_var_range: "sympy.Expr | int | None" = None
    # Valid range (trip count) of loop_var, when loop_var is an unbacked
    # scalar symbol that is NOT a key of any MemoryDep.ranges (e.g. the
    # WhileLoop-splice per-iteration symbol synthesized by
    # for_each_tile_lowering.py's _synthesize_dim_hints_for_group). Ordinary
    # spyre_hint()-scope loop vars are real Inductor iteration-range
    # variables already present in dep.ranges, so this stays None for them
    # -- host_coordinates/compute_coordinates already know their range.
    # op_out_coords uses this to give the WhileLoop symbol a real coordinate
    # term instead of silently dropping it (see coarse_tile.py's
    # _loop_var_to_ranges_pos, which requires the symbol to actually appear
    # in the op's output coordinates).


# op.dim_hints: list[DimHint]
#
# One entry per tiled dimension, ordered outermost loop level first.  Built by
# for_each_tile's lowering (one per nesting level, with loop_var_range set) and
# by the span-overflow planner.  Sorting by hint ID gives outermost-first
# ordering.
#
# Example — a two-level span-overflow tiling of one op:
#
# dim_hints = [DimHint(dim_names=["A"], split_count=2, loop_var=c0, ...),
#              DimHint(dim_names=["B"], split_count=4, loop_var=c1, ...)]


_HINT_RE = re.compile(r"^_hint_(\d+)$")

# meta["custom"] key under which collect_spyre_hints stamps each scan node
# with its position in _scan_combine_subgraphs's encounter order, so the
# while_loop that decompose_scan_to_while_loop replaces the scan with can be
# paired back to that snapshot: the stamp rides through the replacement
# because meta["custom"] is a torch.fx.proxy._COPY_META_FIELDS entry, which
# Match.replace_by_example transfers from the matched node onto every
# replacement node. Deliberately does not match _HINT_RE, so no spyre_hint
# consumer (get_op_hints and everything built on it) can mistake it for a
# hint scope annotation.
_SCAN_SNAPSHOT_SEQ_KEY = "spyre_scan_snapshot_seq"

# One snapshot entry per call_function node: (target, custom meta, node name).
# The name is None when the snapshotted graph is replaced by a retrace rather
# than edited in place, since no node of the replacement is the snapshotted one.
_SnapshotEntry = tuple[Any, dict | None, str | None]


@torch.compiler.allow_in_graph
def get_id():
    """Returns a new hint ID each time it is called

    The IDs are generated per compilation session
    and always start on 0 and are incremented by one.
    The dependency to a global counter is hidden by
    `allow_in_graph` and this call is removed by
    dead code elimination because the counter is not
    used for any computation. The counter is thread
    local.
    """

    tx = InstructionTranslator.current_tx()
    assert tx is not None

    counter = getattr(tx, "__spyre_hint_counter", 0)
    setattr(tx, "__spyre_hint_counter", counter + 1)

    # returning a tensor is required for allow_in_graph
    return counter, torch.empty(0, device="cpu")


# The keys a spyre_hint scope can carry.  Tiling is expressed with for_each_tile,
# so the former tiling keys are rejected rather than silently ignored: an old
# annotation would otherwise compile to an untiled graph with no signal.
_VALID_HINT_KEYS = frozenset(
    {"work_div", "named_dims", "expected_named_dims", "expected_reduction_dims"}
)
_REMOVED_TILING_KEYS = frozenset({"num_tiles_per_dim", "tiles", "slices"})


def _validate_hint_keys(kwargs: dict[str, Any]) -> None:
    for key in kwargs:
        if key in _REMOVED_TILING_KEYS:
            raise ValueError(
                f"spyre_hint({key}=...) has been removed; express tiled loops with "
                f"for_each_tile (torch_spyre._inductor.wsr.for_each_tile) instead"
            )
        if key not in _VALID_HINT_KEYS:
            raise ValueError(
                f"spyre_hint() got an unrecognized key {key!r}; accepted keys: "
                f"{sorted(_VALID_HINT_KEYS)}"
            )


def spyre_hint(**kwargs: Any):
    """
    Attach a hint and a unique hint id to every FX node in scope.

    Accepted keys: ``work_div`` (per-dimension core split), ``named_dims`` (dimension
    names for the ops in scope) and the assertions ``expected_named_dims`` /
    ``expected_reduction_dims``.  Anything else raises ``ValueError``.
    """
    _validate_hint_keys(kwargs)
    if torch.compiler.is_compiling():
        _id, _ = get_id()
    else:
        _id = 0
    return torch.fx.traceback.annotate({f"_hint_{_id}": kwargs})


def get_op_hints(op: Operation) -> dict[int, dict[str, Any]]:
    """
    Return all hints for an Operation keyed by hint id.
    """
    custom = None
    for fx_node in getattr(op, "origins", ()):
        c = (fx_node.meta or {}).get("custom") or {}
        if c:
            custom = c
            break
    if not custom:
        return {}

    hints: dict[int, dict[str, Any]] = {}
    for k, v in custom.items():
        m = _HINT_RE.match(k)
        if m:
            hints[int(m.group(1))] = v
    return hints


def log_new_nodes(node: torch.fx.Node):
    if (
        node.graph.owning_module is not None
        and (meta := node.graph.owning_module.meta.get(OBSERVER_HOOKS_KEY)) is not None
    ):
        # `meta["pass"]`/`meta["subsystem"]` are only present while an
        # `apply_graph_pass` call is in flight (patches.py pops them in its
        # `finally`); the dict itself is never removed from gm.meta, so a
        # node created outside that window -- e.g. by an `apply_gm_pass`
        # call such as `decompose_scan_to_while_loop`, which never sets
        # these keys at all -- must not KeyError here.
        _pass = meta.get("pass", "unknown")
        subsystem = meta.get("subsystem", "unknown")
    else:
        _pass = "unknown"
        subsystem = "unknown"

    logger.debug(
        "Post-grad insertion of node %s with target %s detected after"
        " pass %s, subsystem %s. Spyre hints could be invalidated.",
        node.name,
        node.target,
        _pass,
        subsystem,
    )


def _scan_combine_subgraphs(graph: torch.fx.Graph) -> list[torch.fx.GraphModule]:
    """``GraphModule``s referenced as a ``scan`` call's combine_fn in ``graph``, in
    encounter order.

    Matches on the ``scan`` HOP specifically (``torch.ops.higher_order.scan``), not on
    "any get_attr whose target is a GraphModule": a graph can carry other GraphModule
    subgraphs that are not scan combine functions and have no corresponding
    ``while_loop_body_graph`` for recovery to pair against (e.g. a ``torch.cond``
    branch left in the graph at collection time). Collecting those too would inflate
    collect_spyre_hints's count past what _recover_hop_subgraph_hints's
    while_loop_body_graph-only count can ever match, reintroducing the
    mismatch-abandons-everything failure this guards against.

    Only ``graph``'s own direct call_function nodes are considered -- a nested
    for_each_tile's inner scan, reached through this scan's own combine_fn subgraph,
    is not walked into here. Empirically (verified via a real nested-for_each_tile
    compile), by the time an outer scan's combine_fn is snapshotted, Inductor has
    already fully decomposed any inner scan reached through it into its own
    while_loop -- so there is no pending inner scan subgraph left for this
    function to miss. The inner level was compiled as its own unit first, and
    *that* compile's collect_spyre_hints snapshotted its combine; those snapshots
    ride along, unrecovered, on the module embedded as this scan's combine-fn,
    and _recover_orphaned_hints -- run at this, the next level's collect --
    recovers them before this compile's own decomposition retrace.

    Each scan node is also stamped with its position in the returned list
    (see ``_SCAN_SNAPSHOT_SEQ_KEY``): decompose_scan_to_while_loop later
    replaces each scan with a while_loop and transfers ``meta["custom"]``
    onto the replacement, and that stamp is what recovery pairs the two on
    -- see _recover_hop_subgraph_hints.
    """
    assert graph.owning_module is not None
    subgraphs: list[torch.fx.GraphModule] = []
    for node in graph.nodes:
        if node.op != "call_function" or node.target is not torch.ops.higher_order.scan:
            continue
        combine_fn_node = node.args[0]
        assert isinstance(combine_fn_node, torch.fx.Node) and combine_fn_node.op == (
            "get_attr"
        ), f"{node}: scan's first arg must be a get_attr node, got {combine_fn_node!r}"
        node.meta.setdefault("custom", {})[_SCAN_SNAPSHOT_SEQ_KEY] = len(subgraphs)
        sub_gm = getattr(graph.owning_module, combine_fn_node.target)
        subgraphs.append(sub_gm)
    return subgraphs


def _snapshot_subgraph(sub_gm: torch.fx.GraphModule) -> list[_SnapshotEntry]:
    # decompose_scan_to_while_loop retraces the combine-fn into a new body graph,
    # whose node names say nothing about these nodes, so none are recorded.
    return [
        (n.target, n.meta.get("custom"), None)
        for n in sub_gm.graph.nodes
        if n.op == "call_function"
    ]


def collect_spyre_hints(graph: torch.fx.Graph) -> None:
    """
    Snapshot call_function nodes' (target, custom-meta, name) by topological
    position. Pairs with recover_spyre_hints to survive AOT re-tracing.

    Targets are stored alongside the meta so recovery can re-align even when the
    node sequence changes between the two passes (e.g. ``x + x`` materializes the
    shared producer as two identical nodes -> ``add(mm, mm_default)``). A node
    created after this snapshot gets a fresh name (mm_default), so its name says
    nothing, but its ``target`` OpOverload does. A node that survives until
    recovery keeps its name, so recovery anchors on it.

    Names identify nodes only within this ``Graph`` object, where FX never
    reissues one, even after its node is erased. The passes up to
    recover_spyre_hints edit this graph in place (decompose_auto_functionalized
    splices its make_fx trace in through replace_by_example, so those nodes get
    fresh names here). A pass that replaced ``gm.graph`` with a retrace would
    restart the name counters. Where no name matches, recovery falls back to
    the unbounded target scan, but a retraced node that happens to share a
    snapshotted node's name and target anchors on that entry and can take a
    hint the target scan would not have given it.

    Also snapshots every scan-combine-fn subgraph in encounter order (see
    ``_scan_combine_subgraphs``): for_each_tile's body traces into its own
    ``GraphModule``, invisible to the flat walk above. ``decompose_scan_to_
    while_loop`` (later in post_grad_passes) replaces each ``scan`` with a
    freshly retraced ``while_loop`` whose body starts with no hint metadata --
    the old combine subgraph is left behind, dead but not yet DCE'd, so we
    snapshot it here rather than rely on it surviving to recovery time. See
    recover_spyre_hints for how these snapshots get matched back up.

    Before taking this compile's own snapshots, first recovers any orphaned
    snapshots that modules reachable from this graph still carry from earlier,
    unscheduled compiles (see ``_recover_orphaned_hints``): a nested
    for_each_tile's inner levels run collect but never recover on their own,
    and this is the one point where their recovery is both possible and
    still useful.
    """
    assert graph.owning_module is not None

    # post_grad_custom_pre_pass is called twice
    if graph.owning_module.meta.get("__spyre_dim_hints") is None:
        graph.owning_module._register_create_node_hook(log_new_nodes)

        # Before this compile's own snapshot writes: recover snapshots left
        # behind by embedded, never-scheduled compiles. Must precede the meta
        # stores below -- the walk starts at this module, so once its own
        # snapshot is stored the walk would consume it here instead of leaving
        # it for recover_spyre_hints at scheduler time.
        _recover_orphaned_hints(graph.owning_module)
        snapshot = [
            (node.target, node.meta.get("custom"), node.name)
            for node in graph.nodes
            if node.op == "call_function"
        ]
        graph.owning_module.meta["__spyre_dim_hints"] = snapshot

        scan_subgraphs = _scan_combine_subgraphs(graph)
        subgraph_snapshots = [_snapshot_subgraph(sub_gm) for sub_gm in scan_subgraphs]
        graph.owning_module.meta["__spyre_dim_hints_subgraphs"] = subgraph_snapshots


def _tile_dim_marker_op():
    """``torch.ops.spyre.tile_dim_marker.default``, looked up lazily since this
    module loads before that op is registered.

    for_each_tile.combine_fn tiles every tiled operand (index_select/movedim +
    this marker) before calling the user's body once. That prologue's op count
    can drift across decompose_scan_to_while_loop's retrace (e.g. index_select
    retracing to a different number of select/squeeze ops), but the marker
    count can't -- it's fixed by combine_fn's own zip(operands, specs), one per
    tiled operand, regardless of how any single _tile() call happens to lower.
    So markers are a sync point a plain target scan can use to skip the
    unstable prologue instead of matching through it.
    """
    return torch.ops.spyre.tile_dim_marker.default


def _split_on_markers(items, target_of) -> list[list]:
    """Split ``items`` into segments at each tile_dim_marker (which starts the
    next segment). See _apply_snapshot for why.
    """
    marker_op = _tile_dim_marker_op()
    segments: list[list] = [[]]
    for item in items:
        if target_of(item) == marker_op:
            segments.append([])
        segments[-1].append(item)
    return segments


def _apply_snapshot(
    nodes: list[torch.fx.Node], snapshot: list[_SnapshotEntry]
) -> tuple[int, int]:
    """Align ``snapshot`` (from ``_snapshot_subgraph``/collect_spyre_hints's
    top-level scan) against ``nodes`` by ``target``, writing back recovered
    ``custom``. Returns (matched, hinted).

    Segments both sides on tile_dim_marker first and aligns each segment
    independently: without this, a for_each_tile prologue's retrace-unstable
    op count can shift the cursor past a repeated target (e.g. two
    ``select.int``s) before it reaches the hinted ops after it. A subgraph
    with no markers (the outer-graph call) is just one segment.
    """
    node_segments = _split_on_markers(nodes, lambda n: n.target)
    snapshot_segments = _split_on_markers(snapshot, lambda s: s[0])
    if len(node_segments) != len(snapshot_segments):
        # Mismatched tiled-operand count: no safe segment pairing, fall back
        # to one flat scan over everything.
        node_segments = [nodes]
        snapshot_segments = [snapshot]

    matched = 0
    hinted = 0
    for seg_nodes, seg_snapshot in zip(node_segments, snapshot_segments):
        seg_matched, seg_hinted = _apply_snapshot_flat(seg_nodes, seg_snapshot)
        matched += seg_matched
        hinted += seg_hinted
    return matched, hinted


def _apply_snapshot_flat(
    nodes: list[torch.fx.Node], snapshot: list[_SnapshotEntry]
) -> tuple[int, int]:
    """Forward-scan alignment over one already-synchronized segment.

    Inserted nodes (in ``nodes``, not in ``snapshot``) are left untouched;
    deleted ones (in ``snapshot``, not in ``nodes``) are skipped past. A node
    whose target repeats the last consumed entry is a duplicate (e.g. the
    second mm in add(mm, mm)) and inherits the same hint.

    A node that still carries a snapshotted name and target is that entry, so
    it anchors the alignment: the target scan for any other node stops at the
    next anchor's entry. Without that bound, an inserted node whose target
    recurs later in the snapshot (mm_to_bmm's unsqueeze/expand) would jump the
    cursor past every entry in between and take the hint of an unrelated,
    distant scope.

    The bound includes the anchor's own entry, so a same-target node inserted
    directly before a surviving hinted node still takes that node's hint. This
    is deliberate: it is how a copy of a hinted node placed just before the
    original keeps the hint, and target alone cannot tell the two apart. Anchors
    stop a hint from travelling past a surviving node; they do not make an
    insertion next to one safe.
    """
    position = {name: i for i, (_, _, name) in enumerate(snapshot) if name}

    def anchor_of(node: torch.fx.Node) -> int | None:
        i = position.get(node.name)
        return i if i is not None and snapshot[i][0] == node.target else None

    anchors = sorted(i for n in nodes if (i := anchor_of(n)) is not None)
    next_anchor = 0
    cursor = 0
    last_target = None
    last_custom = None
    matched = 0
    for node in nodes:
        custom = None
        found_at = anchor_of(node)
        if found_at is None:
            while next_anchor < len(anchors) and anchors[next_anchor] < cursor:
                next_anchor += 1
            stop = (
                anchors[next_anchor] + 1
                if next_anchor < len(anchors)
                else len(snapshot)
            )
            for i in range(cursor, stop):
                if snapshot[i][0] == node.target:
                    found_at = i
                    break
        if found_at is not None:
            # Advance cursor past all skipped (deleted) entries and this one.
            cursor = max(cursor, found_at + 1)
            last_target, last_custom, _ = snapshot[found_at]
            custom = last_custom
            matched += 1
        elif node.target == last_target:
            # Duplicate of the just-consumed snapshot node (same computation,
            # e.g. the second mm in add(mm, mm)); reuse its hint.
            custom = last_custom

        if not custom:
            continue
        if node.meta.get("custom") is None:
            node.meta["custom"] = {}
        node.meta["custom"].update(custom)

    hinted = sum(1 for _, c, _ in snapshot if c)
    return matched, hinted


def _recover_hop_subgraph_hints(
    graph: torch.fx.Graph, subgraph_snapshots: list[list[_SnapshotEntry]]
) -> None:
    """Recover collect_spyre_hints's per-scan-combine-fn snapshots into the
    while_loop bodies decompose_scan_to_while_loop replaced them with.

    That upstream pass retraces each scan's combine_fn into a brand-new body
    GraphModule via a fresh make_fx trace, not a copy of the old subgraph's
    nodes -- so op targets aren't reliably stable across it (e.g. `reshape`
    can retrace to `view`), and the old combine subgraph is left behind dead
    but not necessarily DCE'd. Both mean we can't identify the right pairing
    by content or rely on the old node surviving. Instead each snapshot is
    paired with its replacement through the ``_SCAN_SNAPSHOT_SEQ_KEY`` stamp
    collect_spyre_hints left on the scan node: replace_by_example transfers
    ``meta["custom"]`` (a ``torch.fx.proxy._COPY_META_FIELDS`` entry) from
    the matched scan node onto the replacement while_loop node, so the stamp
    survives the retrace even though nothing else about the replacement
    links back to the scan it came from.

    A while_loop without the stamp is not scan-derived and is ignored,
    wherever it sits relative to the scans -- most importantly a plain,
    user-authored ``torch.while_loop``, which traces to exactly this same
    ``higher_order.while_loop`` call shape. (An earlier version paired the
    Nth while_loop in graph.nodes order with the Nth snapshot; any such
    extra loop inflated the count and made recovery abandon every scan in
    the graph, not just itself.)

    A stamped while_loop is still shape-checked before use:
    decompose_scan_to_while_loop always builds its replacement as
    while_loop(cond_fn, body_fn, carried_inputs) with body_fn a literal
    get_attr node (torch/_inductor/fx_passes/post_grad.py), so a stamped
    call without that shape means upstream's construction changed. Skip it
    -- the coverage check below then abandons recovery -- rather than
    assert: crashing the whole compile on a merely-unusual while_loop call
    would be worse than this function's declared failure mode (skip subgraph
    hint recovery, log, keep compiling) for every other case this pass
    handles.

    The stamped seqs must exactly cover the snapshots: a missing, extra or
    duplicated seq -- including zero stamped loops at all, e.g. if a future
    torch version stops transferring "custom" through replace_by_example --
    skips recovery with a debug log, the same declared failure mode as a
    count mismatch. collect_spyre_hints snapshots and stamps every scan,
    hinted or not, so seq i is always the i-th scan encountered; filtering
    unhinted scans out anywhere would desync the numbering against
    ``subgraph_snapshots``.
    """
    if not subgraph_snapshots:
        return
    assert graph.owning_module is not None

    stamped_bodies: list[tuple[int, torch.fx.Node]] = []
    for node in graph.nodes:
        if (
            node.op != "call_function"
            or node.target is not torch.ops.higher_order.while_loop
        ):
            continue
        custom = node.meta.get("custom")
        if not isinstance(custom, dict) or _SCAN_SNAPSHOT_SEQ_KEY not in custom:
            # Not scan-derived: the stamp is transferred onto every
            # replacement decompose_scan_to_while_loop builds, so a
            # while_loop without one -- e.g. a plain, user-authored
            # torch.while_loop, which traces to this same call shape --
            # did not come from a scan and must not consume any scan's
            # snapshot.
            continue
        seq = custom[_SCAN_SNAPSHOT_SEQ_KEY]
        # type-is-int, not isinstance: bool is an int subclass, and a stray
        # True would alias seq 1 past the coverage check below.
        if type(seq) is not int:
            continue
        if len(node.args) < 2:
            continue
        body_fn_node = node.args[1]
        if not isinstance(body_fn_node, torch.fx.Node) or body_fn_node.op != "get_attr":
            continue
        stamped_bodies.append((seq, body_fn_node))

    expected = list(range(len(subgraph_snapshots)))
    if sorted(seq for seq, _ in stamped_bodies) != expected:
        logger.debug(
            "recover_spyre_hints: found %d stamped while_loop body subgraph(s) "
            "but collected %d scan-combine-fn snapshot(s); skipping subgraph "
            "hint recovery (mismatched count).",
            len(stamped_bodies),
            len(subgraph_snapshots),
        )
        return

    for seq, body_node in stamped_bodies:
        snapshot = subgraph_snapshots[seq]
        if not any(custom for _, custom, _ in snapshot):
            continue
        sub_gm = getattr(graph.owning_module, body_node.target, None)
        if not isinstance(sub_gm, torch.fx.GraphModule):
            logger.debug(
                "recover_spyre_hints: while_loop body attribute %s did not "
                "resolve to a GraphModule; skipping its hint recovery.",
                body_node.target,
            )
            continue
        nodes = [n for n in sub_gm.graph.nodes if n.op == "call_function"]
        matched, hinted = _apply_snapshot(nodes, snapshot)
        if matched < hinted:
            logger.debug(
                "recover_spyre_hints: in while_loop body %s, matched %d/%d "
                "hinted snapshot entries from its scan-combine-fn.",
                body_node.target,
                matched,
                hinted,
            )


def _recover_collected_hints(gm: torch.fx.GraphModule) -> bool:
    """Apply ``gm``'s own collect-time snapshots (flat + per-scan-combine) to
    its current graph, popping both meta keys. Returns False, touching
    nothing, if ``gm`` never ran collect_spyre_hints.

    Extracted from recover_spyre_hints so the orphan walk below can run the
    exact same recovery on modules it did not reach through a scheduler.
    """
    if "__spyre_dim_hints" not in gm.meta:
        return False
    _dim_hints = gm.meta.pop("__spyre_dim_hints")
    subgraph_snapshots = gm.meta.pop("__spyre_dim_hints_subgraphs", [])
    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    matched, hinted = _apply_snapshot(nodes, _dim_hints)
    target_of = {n.name: n.target for n in nodes}
    if hinted and not any(target_of.get(name) == t for t, _, name in _dim_hints):
        logger.debug(
            "_recover_collected_hints: on %s, no snapshot entry kept its node "
            "name and target; hints were aligned on target alone, unbounded.",
            gm._get_name(),
        )
    if matched < hinted:
        logger.debug(
            "_recover_collected_hints: on %s, matched %d/%d hinted snapshot "
            "entries; %d entries were for nodes removed by post-grad passes.",
            gm._get_name(),
            matched,
            hinted,
            hinted - matched,
        )
    _recover_hop_subgraph_hints(gm.graph, subgraph_snapshots)
    return True


def _recover_orphaned_hints(root_gm: torch.fx.GraphModule) -> None:
    """Recover snapshots left behind by embedded, never-scheduled compiles.

    A nested for_each_tile's inner levels are compiled as their own units and
    then embedded as FX submodules of the next level's combine-fn; only the
    outermost compile ever reaches ``GraphLowering._update_scheduler``, where
    recover_spyre_hints runs. Every inner level's collect-time snapshot is
    therefore taken and then orphaned -- still carried, unpopped, on the
    embedded module itself.

    This runs at the *next* level's collect, the one window where recovery is
    both possible (the inner compile has fully finished, so the stamped
    while_loop replacements its snapshots pair against already exist) and
    useful (this compile's own decompose_scan_to_while_loop retrace, which
    copies custom meta into the replacement bodies, has not run yet). The
    unpopped ``__spyre_dim_hints`` key is the marker: a compile that did run
    recover -- any normally-scheduled one -- has already popped it, so this
    walk is idempotent and never double-applies.

    Pairing stays scoped to each orphaned module's own graph and snapshot
    list, so scan-stamp sequence numbers -- numbered per compile and hence
    colliding across levels -- never mix.
    """
    seen: set[int] = set()
    pending = [root_gm]
    while pending:
        gm = pending.pop()
        if id(gm) in seen:
            continue
        seen.add(id(gm))
        _recover_collected_hints(gm)
        for node in gm.graph.nodes:
            if node.op != "get_attr":
                continue
            child = getattr(gm, node.target, None)
            if isinstance(child, torch.fx.GraphModule):
                pending.append(child)


def recover_spyre_hints(graph: torch.fx.Graph) -> None:
    """
    Restore custom meta on AOT-renamed call_function nodes by aligning the
    snapshot from collect_spyre_hints against the current graph on ``target``.

    Passes running between collect and recover can both insert nodes (e.g.
    decompose_auto_functionalized inserts copy_forced_default) and delete/replace
    nodes (e.g. auto_functionalized_v2 is replaced). The algorithm must handle
    both cases:

    - Inserted nodes (in graph, not in snapshot): skip the node, leave it
      with whatever meta["custom"] it already has.
    - Deleted nodes (in snapshot, not in graph): skip past the snapshot entry
      to stay aligned with the graph.

    We implement this as a forward scan: a graph node that kept its snapshotted
    name is matched to its own entry. Any other node scans forward in the
    snapshot (from the current cursor, up to the next surviving node's entry)
    looking for a matching target. If found, consume all snapshot entries up to
    and including it. If not found, the node was inserted after the snapshot —
    leave it untouched. A node whose target repeats the last consumed entry is a
    duplicate (e.g. the second mm in add(mm, mm)) and inherits the same hint.

    Also recovers the per-scan-combine-fn snapshots into their corresponding
    while_loop bodies -- see _recover_hop_subgraph_hints.
    """

    assert graph.owning_module is not None

    if log_new_nodes in graph.owning_module._create_node_hooks:
        graph.owning_module._unregister_create_node_hook(log_new_nodes)

    _recover_collected_hints(graph.owning_module)

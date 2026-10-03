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
    split_count: int  # from slices={"A": 4}, e.g. 4
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
# One entry per hinted dimension, ordered outermost hint scope first.
# Outer hint IDs are smaller than inner hint IDs (guaranteed by spyre_hint
# counter order), so sorting by hint ID gives outermost-first ordering.
#
# Example — two nested hints on one op:
#   with spyre_hint(slices={"A": 2}):      # outer scope → smaller hint ID
#       with spyre_hint(slices={"B": 4}):  # inner scope → larger hint ID
#           y = a + b
#
# dim_hints = [DimHint(dim_names=["A"], split_count=2, loop_var=c0, ...),
#                 DimHint(dim_names=["B"], split_count=4, loop_var=c1, ...)]


_HINT_RE = re.compile(r"^_hint_(\d+)$")


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


def spyre_hint(**kwargs: Any):
    """
    Attach a hint and a unique hint id to every FX node in scope.
    """
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


def _is_hop_subgraph_getattr(node: torch.fx.Node) -> bool:
    """Whether ``node`` is a ``get_attr`` referencing a traced subgraph, e.g.
    ``scan``'s ``combine_fn`` (for_each_tile's HOP). ``while_loop``/``map``
    bodies hinted directly are a future extension, not handled yet.
    """
    if node.op != "get_attr":
        return False
    target = getattr(node.graph.owning_module, node.target, None)
    return isinstance(target, torch.fx.GraphModule)


def _snapshot_subgraph(sub_gm: torch.fx.GraphModule) -> list[tuple[Any, dict | None]]:
    return [
        (n.target, n.meta.get("custom"))
        for n in sub_gm.graph.nodes
        if n.op == "call_function"
    ]


def collect_spyre_hints(graph: torch.fx.Graph) -> None:
    """
    Snapshot call_function nodes' (target, custom-meta) by topological position.
    Pairs with recover_spyre_hints to survive AOT re-tracing.

    Targets are stored alongside the meta so recovery can re-align even when the
    node sequence changes between the two passes (e.g. ``x + x`` materializes the
    shared producer as two identical nodes -> ``add(mm, mm_default)``). The node
    *name* is renamed by AOT re-tracing (mm -> mm_default) and so is unstable, but
    the ``target`` OpOverload is preserved and is what we align on.

    Also snapshots every scan-combine-fn subgraph in encounter order (see
    ``_is_hop_subgraph_getattr``): for_each_tile's body traces into its own
    ``GraphModule``, invisible to the flat walk above. ``decompose_scan_to_
    while_loop`` (later in post_grad_passes) replaces each ``scan`` with a
    freshly retraced ``while_loop`` whose body starts with no hint metadata --
    the old combine subgraph is left behind, dead but not yet DCE'd, so we
    snapshot it here rather than rely on it surviving to recovery time. See
    recover_spyre_hints for how these snapshots get matched back up.
    """
    assert graph.owning_module is not None

    # post_grad_custom_pre_pass is called twice
    if graph.owning_module.meta.get("__spyre_dim_hints") is None:
        graph.owning_module._register_create_node_hook(log_new_nodes)

        snapshot = [
            (node.target, node.meta.get("custom"))
            for node in graph.nodes
            if node.op == "call_function"
        ]
        graph.owning_module.meta["__spyre_dim_hints"] = snapshot

        subgraph_snapshots = []
        for node in graph.nodes:
            if not _is_hop_subgraph_getattr(node):
                continue
            sub_gm = getattr(graph.owning_module, node.target)
            sub_snapshot = _snapshot_subgraph(sub_gm)
            if any(custom for _, custom in sub_snapshot):
                subgraph_snapshots.append(sub_snapshot)
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
    nodes: list[torch.fx.Node], snapshot: list[tuple[Any, dict | None]]
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
    nodes: list[torch.fx.Node], snapshot: list[tuple[Any, dict | None]]
) -> tuple[int, int]:
    """Forward-scan alignment over one already-synchronized segment.

    Inserted nodes (in ``nodes``, not in ``snapshot``) are left untouched;
    deleted ones (in ``snapshot``, not in ``nodes``) are skipped past. A node
    whose target repeats the last consumed entry is a duplicate (e.g. the
    second mm in add(mm, mm)) and inherits the same hint.
    """
    cursor = 0
    last_target = None
    last_custom = None
    matched = 0
    for node in nodes:
        custom = None
        # Scan snapshot forward to find a matching entry for this node.
        found_at = None
        for i in range(cursor, len(snapshot)):
            if snapshot[i][0] == node.target:
                found_at = i
                break
        if found_at is not None:
            # Advance cursor past all skipped (deleted) entries and this one.
            cursor = found_at + 1
            last_target, last_custom = snapshot[found_at]
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

    hinted = sum(1 for _, c in snapshot if c)
    return matched, hinted


def _recover_hop_subgraph_hints(
    graph: torch.fx.Graph, subgraph_snapshots: list[list[tuple[Any, dict | None]]]
) -> None:
    """Recover collect_spyre_hints's per-scan-combine-fn snapshots into the
    while_loop bodies decompose_scan_to_while_loop replaced them with.

    That upstream pass retraces each scan's combine_fn into a brand-new body
    GraphModule via a fresh make_fx trace, not a copy of the old subgraph's
    nodes -- so op targets aren't reliably stable across it (e.g. `reshape`
    can retrace to `view`), and the old combine subgraph is left behind dead
    but not necessarily DCE'd. Both mean we can't identify the right pairing
    by content or rely on the old node surviving; instead the pairing is
    positional: decompose_scan_to_while_loop replaces each matched `scan` node
    in place without reordering matches relative to each other, so the Nth
    get_attr node (in graph.nodes order) whose target is a while_loop body
    subgraph corresponds to the Nth scan-combine-fn subgraph collect_spyre_hints
    snapshotted (verified empirically across 1-3 scan sites, including with
    the two name orders scrambled relative to each other).
    """
    if not subgraph_snapshots:
        return
    assert graph.owning_module is not None

    body_nodes: list[torch.fx.Node] = []
    for node in graph.nodes:
        if node.op != "get_attr":
            continue
        target = getattr(graph.owning_module, node.target, None)
        if not isinstance(target, torch.fx.GraphModule):
            continue
        # decompose_scan_to_while_loop always names a scan's replacement body
        # "while_loop_body_graph_*" -- excludes the sibling cond graph and the
        # old, now-orphaned scan_combine_graph_* node.
        if "while_loop_body_graph" in node.target:
            body_nodes.append(node)

    if len(body_nodes) != len(subgraph_snapshots):
        logger.debug(
            "recover_spyre_hints: found %d while_loop body subgraph(s) but "
            "collected %d hinted scan-combine-fn snapshot(s); skipping "
            "subgraph hint recovery (mismatched count).",
            len(body_nodes),
            len(subgraph_snapshots),
        )
        return

    for body_node, snapshot in zip(body_nodes, subgraph_snapshots):
        sub_gm = getattr(graph.owning_module, body_node.target)
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

    We implement this as a forward scan: for each graph node, scan forward in
    the snapshot (from the current cursor) looking for a matching target. If
    found, consume all snapshot entries up to and including it. If not found,
    the node was inserted after the snapshot — leave it untouched. A node
    whose target repeats the last consumed entry is a duplicate (e.g. the
    second mm in add(mm, mm)) and inherits the same hint.

    Also recovers the per-scan-combine-fn snapshots into their corresponding
    while_loop bodies -- see _recover_hop_subgraph_hints.
    """

    assert graph.owning_module is not None

    if log_new_nodes in graph.owning_module._create_node_hooks:
        graph.owning_module._unregister_create_node_hook(log_new_nodes)

    _dim_hints = graph.owning_module.meta.pop("__spyre_dim_hints")
    subgraph_snapshots = graph.owning_module.meta.pop("__spyre_dim_hints_subgraphs", [])
    nodes = [n for n in graph.nodes if n.op == "call_function"]

    matched, hinted = _apply_snapshot(nodes, _dim_hints)
    if matched < hinted:
        logger.debug(
            "recover_spyre_hints: matched %d/%d hinted snapshot entries; "
            "%d entries were for nodes removed by post-grad passes.",
            matched,
            hinted,
            hinted - matched,
        )

    _recover_hop_subgraph_hints(graph, subgraph_snapshots)

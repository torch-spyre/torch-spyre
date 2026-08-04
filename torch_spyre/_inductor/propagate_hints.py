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
        _pass = meta["pass"]
        subsystem = meta["subsystem"]
    else:
        _pass = "unknown"
        subsystem = "unknown"

    logger.warning(
        "Post-grad insertion of node %s with target %s detected after"
        " pass %s, subsystem %s. Spyre hints could be invalidated.",
        node.name,
        node.target,
        _pass,
        subsystem,
    )


def collect_spyre_hints(graph: torch.fx.Graph) -> None:
    """
    Snapshot call_function nodes' (target, custom-meta) by topological position.
    Pairs with recover_spyre_hints to survive AOT re-tracing.

    Targets are stored alongside the meta so recovery can re-align even when the
    node sequence changes between the two passes (e.g. ``x + x`` materializes the
    shared producer as two identical nodes -> ``add(mm, mm_default)``). The node
    *name* is renamed by AOT re-tracing (mm -> mm_default) and so is unstable, but
    the ``target`` OpOverload is preserved and is what we align on.
    """
    assert graph.owning_module is not None

    # post_grad_custom_pre_pass is called twice
    if graph.owning_module.meta.get("__spyre_dim_hints") is None:
        graph.owning_module._register_create_node_hook(log_new_nodes)

        graph.owning_module.meta["__spyre_dim_hints"] = [
            (node.target, node.meta.get("custom"))
            for node in graph.nodes
            if node.op == "call_function"
        ]


def recover_spyre_hints(graph: torch.fx.Graph) -> None:
    """
    Restore custom meta on AOT-renamed call_function nodes by aligning the
    snapshot from collect_spyre_hints against the current graph on ``target``.

    A simple positional zip is not robust: passes running between collect and
    recover can insert nodes, most commonly by un-sharing a producer feeding a
    pointwise op (``add(mm, mm)`` becomes two distinct ``aten.mm.default`` nodes).
    We instead walk both sequences with a single cursor: a node whose target
    matches the next snapshot entry consumes it; a node whose target repeats the
    last consumed entry is treated as a duplicate of that computation and inherits
    the same hint. This keeps alignment intact across such insertions, where the
    old count check would bail and silently drop every hint.
    """

    assert graph.owning_module is not None

    graph.owning_module._unregister_create_node_hook(log_new_nodes)

    _dim_hints = graph.owning_module.meta.pop("__spyre_dim_hints")
    nodes = [n for n in graph.nodes if n.op == "call_function"]

    cursor = 0
    last_target = None
    last_custom = None
    for node in nodes:
        custom = None
        if cursor < len(_dim_hints) and _dim_hints[cursor][0] == node.target:
            last_target, last_custom = _dim_hints[cursor]
            custom = last_custom
            cursor += 1
        elif node.target == last_target:
            # Duplicate of the just-consumed snapshot node (same computation,
            # e.g. the second mm in add(mm, mm)); reuse its hint.
            custom = last_custom

        if not custom:
            continue
        if node.meta.get("custom") is None:
            node.meta["custom"] = {}
        node.meta["custom"].update(custom)

    if cursor != len(_dim_hints):
        logger.warning(
            f"Warning: unable to recover spyre hints "
            f"(matched {cursor}/{len(_dim_hints)} snapshot entries)"
        )

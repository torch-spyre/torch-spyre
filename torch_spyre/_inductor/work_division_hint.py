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


from __future__ import annotations

import logging
import regex as re
from contextlib import contextmanager

import torch.fx
import torch.fx.traceback

HINT_KEY = "work_division_hint"

logger = logging.getLogger(__name__)

# Registry populated by collect_work_division_hints (pre-grad pass).
# Maps graph_id -> {node_name -> hint} so that propagate_work_division_hints
# (post-grad pass) can recover hints lost during ATen decomposition.
_hint_registry: dict[int, dict[str, list[int]]] = {}

# Populated by collect_pre_pass_hints (post-grad early pass).
# Maps node_name -> hint.  Used by propagate_post_pass_hints to recover
# hints on replacement nodes at CustomPostPasses time.
_pre_pass_hint_by_name: dict[str, list[int]] = {}

# ATen overload suffixes appended to node names during re-tracing.
_ATEN_OVERLOAD_SUFFIXES = re.compile(
    r"_(default|tensor|scalar|Tensor|Scalar|int|float|out)$"
)


def collect_work_division_hints(graph: torch.fx.Graph) -> None:
    """Pre-grad pass: record every hinted node into _hint_registry."""
    graph_id = id(graph)
    table: dict[str, list[int]] = {}
    for node in graph.nodes:
        custom = node.meta.get("custom") or {}
        hint = custom.get(HINT_KEY)
        if hint is not None:
            table[node.name] = list(hint)
    if table:
        _hint_registry[graph_id] = table


def propagate_work_division_hints(graph: torch.fx.Graph) -> None:
    """Post-grad pass: restore hints on ATen nodes whose Dynamo origin had one.

    AOT Autograd decomposition drops node.meta["custom"] on newly created ATen
    nodes. We recover it by following each node's from_node provenance chain
    back to the pre-grad node name, then looking up the hint in _hint_registry.
    A hint is applied only when all from_node ancestors agree on the same value.
    """
    for node in graph.nodes:
        if (node.meta.get("custom") or {}).get(HINT_KEY) is not None:
            continue  # already annotated

        from_nodes = node.meta.get("from_node") or []
        if not from_nodes:
            continue

        hints: set[tuple] = set()
        for source in from_nodes:
            graph_id = getattr(source, "graph_id", None)
            name = getattr(source, "name", None)
            if graph_id is None or name is None:
                continue
            table = _hint_registry.get(graph_id, {})
            hint = table.get(name)
            if hint is not None:
                hints.add(tuple(hint))

        if len(hints) == 1:
            if "custom" not in node.meta or node.meta["custom"] is None:
                node.meta["custom"] = {}
            node.meta["custom"][HINT_KEY] = list(hints.pop())


def _strip_aten_overload(name: str) -> str:
    """Strip ATen overload suffix from a node name.

    During AOT re-tracing, FX nodes are renamed with overload suffixes:
    ``mm`` → ``mm_default``, ``add`` → ``add_tensor``.  Stripping the suffix
    recovers the original base name for matching against the pre-pass registry.
    A trailing ``_N`` counter (e.g. ``add_tensor_1``) is preserved in the base
    so it still matches ``add_1``.
    """
    # Handle names like add_tensor_1 → strip _tensor → add_1
    # and mm_default → strip _default → mm
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        # Name ends with _N counter: strip overload from the prefix part
        base = _ATEN_OVERLOAD_SUFFIXES.sub("", parts[0])
        return f"{base}_{parts[1]}"
    return _ATEN_OVERLOAD_SUFFIXES.sub("", name)


def collect_pre_pass_hints(graph: torch.fx.Graph) -> None:
    """Snapshot hinted nodes at CustomPrePasses time for later recovery.

    Between CustomPrePasses and CustomPostPasses, AOT re-tracing creates new
    nodes (e.g. mm_default, add_tensor) that lose all custom metadata and
    from_node provenance. This function records hints keyed by node name so
    that propagate_post_pass_hints can match replacement nodes by stripping
    ATen overload suffixes from their names.
    """
    _pre_pass_hint_by_name.clear()
    for node in graph.nodes:
        custom = node.meta.get("custom") or {}
        hint = custom.get(HINT_KEY)
        if hint is not None:
            _pre_pass_hint_by_name[node.name] = list(hint)


def propagate_post_pass_hints(graph: torch.fx.Graph) -> None:
    """Recover lost hints at CustomPostPasses time via name-based matching.

    Nodes created by AOT re-tracing (e.g. mm_default, add_tensor) retain the
    same base name as their CustomPrePasses predecessors (mm, add) but with an
    ATen overload suffix appended.  We strip the suffix and look up the hint
    in _pre_pass_hint_by_name.
    """
    if not _pre_pass_hint_by_name:
        return

    for node in graph.nodes:
        if (node.meta.get("custom") or {}).get(HINT_KEY) is not None:
            continue

        base_name = _strip_aten_overload(node.name)
        hint = _pre_pass_hint_by_name.get(base_name)
        if hint is not None:
            if "custom" not in node.meta or node.meta["custom"] is None:
                node.meta["custom"] = {}
            node.meta["custom"][HINT_KEY] = list(hint)
            logger.debug(
                "Recovered hint %s on %s (matched %s)",
                hint,
                node.name,
                base_name,
            )


@contextmanager
def work_division_hint(splits: list[int]):
    """Override core division splits for operations within this block.

    Args:
        splits: Split factors in iteration-space order — output dims first,
            then reduction dims. For a 2D matmul out = x @ y with x:(M,K)
            and y:(K,N), pass [M_split, N_split, K_split].

    Example::

        from torch_spyre._inductor.work_division_hint import work_division_hint

        @torch.compile
        def model(x, y):
            with work_division_hint([2, 1, 2]):
                out = x @ y  # M split by 2, N unsplit, K split by 2
            return out

    Note:
        Different hint values for the same compiled function may hit Dynamo's
        graph cache. Call ``torch._dynamo.reset()`` between experiments.
    """
    with torch.fx.traceback.annotate({HINT_KEY: list(splits)}):
        yield

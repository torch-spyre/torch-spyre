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


from typing import Any

import regex as re

import torch
import torch.fx.traceback
from torch._inductor.ir import Operation

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("propagate_hints")


_HINT_RE = re.compile(r"^_hint_(\d+)$")
_hint_counter = 0

# Snapshot of FX node `custom` meta taken at CustomPrePasses time, indexed
# by call_function node position. Used by recover_spyre_hints to restore
# meta on nodes renamed by AOT re-tracing (e.g. mm -> mm_default), which
# drops node.meta["custom"].
_spyre_hints: list[dict[str, Any] | None] = []


def spyre_hint(**kwargs: Any):
    """
    Attach a hint and a unique hint id to every FX node in scope.
    """
    global _hint_counter

    _hint_counter += 1
    return torch.fx.traceback.annotate({f"_hint_{_hint_counter}": kwargs})


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


def collect_spyre_hints(graph: torch.fx.Graph) -> None:
    """
    Snapshot call_function nodes' custom meta by topological position.
    Pairs with recover_spyre_hints to survive AOT re-tracing.
    """
    global _spyre_hints

    _spyre_hints = [
        node.meta.get("custom") for node in graph.nodes if node.op == "call_function"
    ]


def recover_spyre_hints(graph: torch.fx.Graph) -> None:
    """
    Restore custom meta on AOT-renamed call_function nodes, matching the
    snapshot taken by collect_spyre_hints by topological position.
    """
    nodes = [n for n in graph.nodes if n.op == "call_function"]
    if len(nodes) != len(_spyre_hints):
        logger.warning("Warning: unable to recover spyre hints")
        return
    for node, custom in zip(nodes, _spyre_hints):
        if not custom:
            continue
        if node.meta.get("custom") is None:
            node.meta["custom"] = {}
        node.meta["custom"].update(custom)

# Copyright 2026 The IBM Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Give explicitly row-partitioned dense subviews independent compact storage."""

import math
import operator
import copy

import torch
from torch._inductor.fx_utils import get_node_storage
from torch.utils._pytree import tree_leaves

from . import config
from .constants import DEVICE_NAME
from .pass_utils import copy_fx_custom_meta


def _storages(value):
    return {
        storage
        for node in tree_leaves(value)
        if isinstance(node, torch.fx.Node)
        for tensor in tree_leaves(node.meta.get("val"))
        if isinstance(tensor, torch.Tensor)
        if (storage := tensor.untyped_storage()._cdata) is not None
    }


def _read_only_internal_storage(graph, storage):
    """Never break an observable alias or move a read across a parent mutation."""
    for node in graph.nodes:
        if storage not in _storages((node.args, node.kwargs)):
            continue
        if node.op == "output":
            return False
        if node.op != "call_function":
            return False
        if node.target is operator.getitem:
            continue
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            return False
        for i, argument in enumerate(schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            value = (
                node.args[i] if i < len(node.args) else node.kwargs.get(argument.name)
            )
            if storage in _storages(value):
                return False
    return True


def _row_hint(node, row_axis):
    value = node.meta["val"]
    for key, hint in (node.meta.get("custom") or {}).items():
        if not key.startswith("_hint_") or not isinstance(hint, dict):
            continue
        names = hint.get("named_dims")
        division = hint.get("work_div")
        if not isinstance(names, (list, tuple)) or len(names) != value.ndim:
            continue
        if not isinstance(division, dict) or not division:
            continue
        row_name = names[row_axis]
        split = division.get(row_name)
        if type(split) is not int or not 1 < split <= config.sencores:
            continue
        if value.shape[row_axis] % split:
            continue
        if any(name != row_name and part != 1 for name, part in division.items()):
            continue
        return hint
    return None


def _compact_column_view(node, parent):
    value, full = node.meta.get("val"), parent.meta.get("val")
    if not isinstance(value, torch.Tensor) or not isinstance(full, torch.Tensor):
        return False
    if value.dtype != torch.float16 or value.device.type != DEVICE_NAME:
        return False
    if value.dtype != full.dtype or value.device != full.device:
        return False
    if full.ndim != 2 or not full.is_contiguous() or full.storage_offset() != 0:
        return False
    if any(type(x) is not int for x in (*value.shape, *value.stride(), *full.shape)):
        return False
    rows, features = full.shape
    if rows <= 1 or features % 64 or type(value.storage_offset()) is not int:
        return False
    # Leading singleton batch dimensions do not change the row coordinate.
    row_axis = next((i for i, size in enumerate(value.shape) if size != 1), None)
    if row_axis is None or value.shape[row_axis] != rows:
        return False
    if value.stride(row_axis) != features or not _row_hint(node, row_axis):
        return False
    width = math.prod(value.shape[row_axis + 1 :])
    origin = value.storage_offset()
    if not 0 < width < features or width % 64 or origin % 64:
        return False
    if origin < 0 or origin + width > features:
        return False
    stride = 1
    for size, actual in reversed(
        list(zip(value.shape[row_axis + 1 :], value.stride()[row_axis + 1 :]))
    ):
        if size != 1 and actual != stride:
            return False
        stride *= size
    return True


def materialize_hinted_views(graph):
    """Lower an explicit view ownership request through existing clone planning.

    Only fresh matmul column views with a row-only division are supported. The
    ordinary allocator still decides residency and prices complete-parent LX
    relayouts. No compact cross-core subrange transfer is introduced here.
    """
    if not (
        config.lx_planning
        and config.co_optimizing_lx_planning
        and config.lx_planner_relayout
        and config.lx_extended_ownership
    ):
        return
    parents = {
        get_node_storage(node): node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.mm.default
        if get_node_storage(node) is not None
    }
    safe = {}
    for node in list(graph.nodes):
        # Skip clone users: a clone already materializes. A second run is a no-op.
        if node.op != "call_function" or node.target not in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
        ):
            continue
        storage = get_node_storage(node)
        parent = parents.get(storage)
        if node.meta.get("spyre_compact_reader"):
            continue
        if parent is None or not _compact_column_view(node, parent):
            continue
        users = [u for u in node.users if u.target != torch.ops.aten.clone.default]
        if not users:
            continue
        if storage not in safe:
            safe[storage] = _read_only_internal_storage(graph, storage)
        if not safe[storage]:
            continue
        value = node.meta["val"]
        rows = parent.meta["val"].shape[0]
        width = value.numel() // rows
        row_axis = next(i for i, size in enumerate(value.shape) if size != 1)
        hint = _row_hint(node, row_axis)
        # Clone as a 2-D matrix, as ordinary dense clones do, then restore the
        # user's head geometry with a view.
        with graph.inserting_before(users[0]):
            flat = graph.call_function(
                torch.ops.aten.view.default, (node, [rows, width])
            )
            compact = graph.call_function(
                torch.ops.aten.clone.default,
                (flat,),
                {"memory_format": torch.contiguous_format},
            )
            restored = graph.call_function(
                torch.ops.aten.view.default, (compact, list(value.shape))
            )
        flat.meta["val"] = value.view(rows, width)
        compact.meta = node.meta.copy()
        compact.meta["val"] = flat.meta["val"].clone(
            memory_format=torch.contiguous_format
        )
        copy_fx_custom_meta(node, compact)
        compact.meta["custom"] = copy.deepcopy(compact.meta["custom"])
        for value_hint in compact.meta["custom"].values():
            if value_hint == hint:
                value_hint["named_dims"] = [
                    hint["named_dims"][row_axis],
                    f"_untracked_compact_width_{width}",
                ]
        restored.meta["val"] = compact.meta["val"].view(value.shape)
        for user in users:
            user.replace_input_with(node, restored)
        node.meta["spyre_compact_reader"] = True
    graph.lint()

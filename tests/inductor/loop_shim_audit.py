# Copyright 2026 The Torch-Spyre Authors.
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

"""Inventory of the readers that can see a loop shim op (a helper, not a test).

Run as a pytest plugin over the real-graph suites::

    AUDIT_OUT=/tmp/out.json PYTHONPATH=tests/inductor python3 -m pytest <files> \
        -p no:spyre_inference_test -p loop_shim_audit

Every op the span-overflow / allocator transform stamps gets a
``RecordingLoopShim`` in place of its ``CoarseTileInfo``. The shim logs, with the
reader's file and line, every attribute read that is NOT one of its four fields (the
advance lists the real shim does not have) and answers ``[]``, which is what a
reader written as ``getattr(info, name, None) or []`` would see. An
``isinstance(x, CoarseTileInfo)`` check cannot be logged this way; those are found by
an AST scan of ``torch_spyre/`` (``scan_isinstance_sites`` below).
"""

import ast
import atexit
import collections
import json
import os
import pathlib
import sys

import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from torch_spyre._inductor.loop_info import LoopShim

_SHIM_FIELDS = {
    "loop_group_id",
    "loop_count",
    "loop_tiled_dims",
    "loop_tiled_reduction_dims",
}
ACCESSES: collections.Counter = collections.Counter()


class RecordingLoopShim(LoopShim):
    """A ``LoopShim`` that logs each read of an attribute it does not have."""

    def __getattr__(self, name):
        if name.startswith("__") or name in _SHIM_FIELDS:
            raise AttributeError(name)
        frame = sys._getframe(1)
        where = f"{frame.f_code.co_filename.split('torch_spyre/')[-1]}:{frame.f_lineno}"
        ACCESSES[(where, name)] += 1
        return []


def scan_isinstance_sites(root="torch_spyre"):
    """``(file, line)`` of every ``isinstance(..., CoarseTileInfo)`` call under ``root``."""
    sites = []
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance"
                and len(node.args) == 2
                and "CoarseTileInfo" in ast.unparse(node.args[1])
            ):
                sites.append((str(path), node.lineno))
    return sites


def pytest_configure(config):
    from torch_spyre._inductor import passes
    from torch_spyre._inductor.loop_node import all_ops
    from torch_spyre._inductor.scratchpad import coarse_tiling
    from torch_spyre._inductor.wsr import coarse_tile

    original = coarse_tile.coarse_tile_post_stickify

    def entry(graph, groups, group_idx_offset=0):
        before = {
            id(op): id(getattr(op, "loop_info", None))
            for op in all_ops(graph.operations)
        }
        result = original(graph, groups=groups, group_idx_offset=group_idx_offset)
        for op in all_ops(graph.operations):
            info = getattr(op, "loop_info", None)
            if info is None or before.get(id(op)) == id(info):
                continue
            op.loop_info = RecordingLoopShim(
                loop_group_id=tuple(info.loop_group_id),
                loop_count=list(info.loop_count),
                loop_tiled_dims=[list(d) for d in info.loop_tiled_dims],
                loop_tiled_reduction_dims=[
                    list(d) for d in info.loop_tiled_reduction_dims
                ],
            )
        return result

    coarse_tile.coarse_tile_post_stickify = entry
    passes.coarse_tile_post_stickify = entry
    coarse_tiling.coarse_tile_post_stickify = entry

    def dump():
        out = os.environ.get("AUDIT_OUT")
        if out:
            json.dump(
                {f"{w}  .{a}": n for (w, a), n in sorted(ACCESSES.items())},
                open(out, "w"),
                indent=1,
            )

    atexit.register(dump)

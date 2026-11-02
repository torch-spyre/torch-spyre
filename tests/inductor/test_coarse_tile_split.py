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

"""Tests for the coarse_tile.py / loop_tiling_common.py split."""

import ast
import pathlib

import pytest
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

MOVED_NAMES = [
    "_LoopVarRebaseHandler",
    "_NameSwapHandler",
    "_divide_ranges",
    "_loop_var_hinted_ranges",
    "_loop_var_to_ranges_pos",
    "_patch_retiled_load_indexes",
    "_rebase_point_splice_reads",
    "_rescale_index",
    "_splice_loop_vars",
    "reduction_loop_vars",
    "_stick_host_dim",
    "_loop_var_to_reduction_ranges_pos",
    "_raw_to_squeezed_pos",
    "_LOOPS_FREE_SYMS_KEY",
    "_LOOPS_INNER_FN_STR_KEY",
    "_LOOPS_INNER_FN_OPCOUNT_KEY",
    "_REDUCTION_FREE_SYMS_KEY",
    "_LAYOUT_FREE_SYMS_KEY",
    "_COMPUTED_BUF_FREE_SYMS_KEY",
    "_COMPUTED_BUF_SIZES_KEY",
    "logger",
]


@pytest.mark.parametrize("name", MOVED_NAMES)
def test_moved_names_importable_from_both_modules(name):
    from torch_spyre._inductor.wsr import coarse_tile, loop_tiling_common

    assert getattr(coarse_tile, name) is getattr(loop_tiling_common, name)


def test_for_each_tile_lowering_does_not_import_from_coarse_tile():
    from torch_spyre._inductor.wsr import for_each_tile_lowering

    tree = ast.parse(pathlib.Path(for_each_tile_lowering.__file__).read_text())
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.endswith("coarse_tile")
    ]
    assert not offenders, f"imports from coarse_tile at lines {offenders}"


def test_logger_name_is_stable():
    from torch_spyre._inductor.wsr import loop_tiling_common

    assert loop_tiling_common.logger.name == "spyre.inductor.wsr.coarse_tile"

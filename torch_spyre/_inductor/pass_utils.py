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

from typing import NamedTuple

from sympy import Expr, Symbol

import sympy
from torch._inductor.ir import FixedLayout
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V

from .ir import FixedTiledLayout


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: FixedTiledLayout


def get_mem_deps(n: SchedulerNode) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in n.read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")
            res.append(SchedNodeArg(arg, layout))
    return res


def wildcard_symbol(dim) -> Symbol:
    return sympy.Symbol(f"*_{dim}")


def map_dims_to_vars(layout: FixedLayout, index: Expr) -> dict[int, Symbol]:
    """
    Construct a mapping from the dimensions of layout
    to the free variables of index that correspond to them.
    Dimensions of size 1 are mapped to a wild_card_symbol of `*`

    This works by reversing the algorithm used by torch._inductor.ir. _fixed_indexer to build index.
    """
    result = {}
    for sym in index.free_symbols:
        stride_val = sympy_subs(index, {sym: 1}) - sympy_subs(index, {sym: 0})
        if stride_val in layout.stride:
            idx = layout.stride.index(stride_val)
            result[idx] = sym

    for d in range(len(layout.size)):
        if d not in result:
            assert layout.size[d] == 1, "non-trivial dim missing from index exper"
            result[d] = wildcard_symbol(d)

    return result

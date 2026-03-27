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


import sympy
from torch._inductor.ir import FixedLayout, Pointwise, Reduction
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep
from torch._inductor.virtualized import V
from torch_spyre._inductor.errors import Unsupported

from .ir import FixedTiledLayout
from .views import compute_coordinates


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


def host_coordinates(layout: FixedLayout, dep: MemoryDep) -> list[sympy.Expr]:
    return compute_coordinates(layout.size, layout.stride, dep.ranges, dep.index)


def device_coordinates(layout: FixedTiledLayout, dep: MemoryDep) -> list[sympy.Expr]:
    return compute_coordinates(
        layout.device_layout.device_size,
        layout.device_layout.stride_map,
        dep.ranges,
        dep.index,
    )


def iteration_space(n: SchedulerNode) -> dict[sympy.Symbol, sympy.Expr]:
    if isinstance(n.node.data, Pointwise):
        # The iteration space of a Pointwise is that of its output
        return next(iter(n.read_writes.writes)).ranges.copy()
    elif isinstance(n.node.data, Reduction):
        # The iteration space of a Reduction is that of its input
        return next(iter(n.read_writes.reads)).ranges.copy()
    else:
        raise Unsupported("Unexpected node type")

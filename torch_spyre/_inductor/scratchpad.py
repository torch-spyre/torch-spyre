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

import math
import os
from torch._inductor.ir import (
    ComputedBuffer,
)
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
)
from torch._inductor.virtualized import V


OPS_GOOD_FOR_LX_REUSE = {"input": {"sub", "div"}, "output": {"max", "sum"}}


class ScratchPadAllocator:
    """LX manager simplified version"""

    def __init__(self, size: int = -1):
        # scratch pad is 2MB = 2<<20 bytes in total. preserve total * DXP_LX_FRAC_AVAIL
        # for backend usage unless specified otherwise
        if size == -1:
            size = int(
                (2 << 20) * (1.0 - float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2")))
            )
        self.limit = size
        self.usage: dict = {}  # each record will be tensor_name:{"addr": yy, "size": zz}
        self.lx_usage_hist: list = []

    def get_lowest_addr_in_use(self):
        if len(self.usage) > 0:
            return min([rec["addr"] for rec in self.usage.values()])
        return None

    def get_highest_addr_in_use(self):
        if len(self.usage) > 0:
            return max([rec["addr"] + rec["size"] for rec in self.usage.values()])
        return None

    def find_free_block(self, size_needed: int):
        # cannot perform defragmentation yet, will add more cases in the future
        curr_lo = self.get_lowest_addr_in_use()
        curr_hi = self.get_highest_addr_in_use()
        if len(self.usage) == 0 or curr_lo >= size_needed:
            # completely free or enough room at addr0
            return 0
        elif curr_hi + size_needed < self.limit:
            # enough room at higher addr, return next 128-multiple
            return math.ceil(curr_hi / 128) * 128
        elif len(self.usage) > 1:
            # find a "hole" between lowest and highest (assume a block was dealloc'ed)
            rec_only = list(self.usage.values())  # simply drop tensor names, not needed
            sorted_rec = sorted(rec_only, key=lambda rec: rec["addr"])
            for i in range(len(sorted_rec) - 1):
                frag_st = sorted_rec[i]["addr"] + sorted_rec[i]["size"]
                frag_end = sorted_rec[i + 1]["addr"]
                if frag_end - frag_st >= size_needed:
                    return frag_st
            return -1
        else:
            # cannot find any free blocks
            return -1

    def try_allocate(
        self, mem_usage: dict, idx: int, org_op_name: str, is_last_node: bool
    ):
        """
        Allocate based on needed mem_usage of the node and then:
         1. keep a record in self.usage.
         2. add lx info to corresponding buffer.layout
        NOTE: 1. assume compiler always allocates inputs before output. Allocate inputs
                 first then output tensors. Dealloc at then end if inputs are not needed.
              2. Some unresolved issues still prevent the reuse of main input tensors.
                 But still need to alloc it on LX first, so that output tensors will not
                 overlap at lower addr where inputs will reside (entirely or partial).
              3. LX reuse strategy could vary for the same buffer on different Op. e.g.
                 arg0 at op MAX is the 1st time this buffer is used => can not be found
                 on LX and has to be loaded from HBM. But the following op SUB may be
                 able to reuse it from LX without reaching HBM again. => include Node Idx
                 (sequence in nodes) and verify it when generating sdsc

        TODO: may need to utilize info from previous Op's sdsc.out.out.out.json
        """
        lx_alloc_to_del = []
        for tensor_name, needed in mem_usage.items():
            # find the current LX usage of this tensor name, if exists
            lx_rec = self.usage.get(tensor_name, {})

            if lx_rec and lx_rec["size"] == needed["size"]:
                # same tensor name and size is on scratchpad already, reuse it
                addr = lx_rec["addr"]
            else:
                # new allocation or overwrite the existing one
                addr = self.find_free_block(needed["size"])
                if addr == -1:
                    # no further action if allocation failed
                    continue

            self.usage[tensor_name] = {"addr": addr, "size": needed["size"]}

            # Decide whether to reuse. For now, only allow ops we've tested successfully.
            # TODO may be able to generalize this decision in buf end-of-life analysis
            in_or_out = "input" if needed["is_input"] else "output"
            can_reuse = any(
                op in org_op_name for op in OPS_GOOD_FOR_LX_REUSE[in_or_out]
            )
            # Special cases check:
            # 1) tensor-to-be-reused is input but not on LX, or
            # 2) output can reuse but is last node (make sure it'll go back to HBM)
            if (needed["is_input"] and lx_rec == {}) or (
                not needed["is_input"] and is_last_node
            ):
                can_reuse = False

            # Directly add the lx info into V.graph.buffers.layout for later codegen use.
            force_pinning = False  # DEBUG use only, e.g. idx==1
            if can_reuse or force_pinning:
                buf = V.graph.get_buffer(tensor_name)
                layout = buf.get_layout()
                layout.allocation[f"lx:{idx}"] = addr  # see doctring Note 3
                # Record usage history for debugging
                self.lx_usage_hist.append(
                    {
                        "node_idx": idx,
                        "op_name": org_op_name,
                        "tensor_name": tensor_name,
                        "addr": addr,
                        "size": needed["size"],
                    }
                )
            else:
                lx_alloc_to_del.append(tensor_name)
        # see docstring NOTE 2
        self.deallocate(lx_alloc_to_del)

    def deallocate(self, bufs: list[str]):
        """Try to deallocate each of the buffers in a list, if exists."""
        if isinstance(bufs, str):
            bufs = [bufs]

        for buf in bufs:
            if buf in self.usage:
                del self.usage[buf]

    # TODO add dealloc and defrag mechanism to allocator later


def mem_usage_by_node(n: SchedulerNode):
    """Get a summary of memory usage of the input node"""
    mem_usage = {}
    for r_or_w, buf_memDeps in enumerate([n.read_writes.reads, n.read_writes.writes]):
        for buf_memDep in buf_memDeps:
            buf = V.graph.get_buffer(buf_memDep.name)
            dev_layout = buf.layout.device_layout  # this is device layout
            dev_size = (
                math.prod(dev_layout.device_size[:-1]) * 128
            )  # num_sticks * bytes_per_stick
            mem_usage[buf_memDep.name] = {
                "is_input": r_or_w == 0,
                "size": dev_size,
            }

    return mem_usage


def consider_for_scratchpad(
    n: SchedulerNode,
    alloc: ScratchPadAllocator,
    idx: int,
    is_last_node: bool,
):
    # 1. summarize both inputs and output sizes used by this node.
    mem_usage = mem_usage_by_node(n)

    # 2. if alloc successful, lx info will be added to corresponding FixedTiledLayout,
    # which will be used in generate_sdsc() later.
    org_op_name = n.node.origin_node.name
    alloc.try_allocate(mem_usage, idx, org_op_name, is_last_node)


def buf_end_of_life_analysis(nodes: list[BaseSchedulerNode]):
    """
    First, find out the last time each buffer was used. {buf1: idx_last_used, ...}
    Turn it into {idx_last_used+1:[buf1, ], ...}, ie. buffers to be deleted at given idx
    """
    last_used: dict = {}
    for idx, n in enumerate(nodes):
        for buf in n.used_buffer_names():  # just buf names
            last_used[buf] = idx

    bufs_to_dealloc_at_idx: dict = {}
    for buf, idx in last_used.items():
        # if last used at idx => del at idx+1
        if idx + 1 in bufs_to_dealloc_at_idx:
            bufs_to_dealloc_at_idx[idx + 1].append(buf)
        else:
            bufs_to_dealloc_at_idx[idx + 1] = [buf]

    return bufs_to_dealloc_at_idx


def scratchpad_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guarenteed by caller).
    # Work division has already been done.
    # Stickification has already been done (therefore all ComputedBeffers have FixedTiledLayouts)

    alloc = ScratchPadAllocator()

    node_idx_to_dealloc_bufs = buf_end_of_life_analysis(nodes)

    for idx, n in enumerate(nodes):
        # release unneeded LX allocations before actual planning
        alloc.deallocate(node_idx_to_dealloc_bufs.get(idx, []))
        is_last_node = idx == len(nodes) - 1

        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            consider_for_scratchpad(n, alloc, idx, is_last_node)
    # print(alloc.lx_usage_hist)
    return nodes

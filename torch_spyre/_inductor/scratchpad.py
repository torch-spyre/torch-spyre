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
from typing import Callable, Any, Optional, override

from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
)
from torch._inductor.lowering import lowerings, clone as clone_lowering
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V
from torch._inductor.graph import GraphLowering
from .logging_utils import get_inductor_logger
from .ir import FixedTiledLayout, TensorBox
from . import config

OP_OUTPUT_GOOD_FOR_LX_REUSE = [
    "max",
    "sum",
    # "clone",
    "exp",
    "sub",
    # "mul",
]

OP_GOOD_FOR_LX_INPLACE = [
    "exp",
    "sub",
]

logger = get_inductor_logger("LX_PLANNING")


class ScratchPadAllocator:
    """LX manager simplified version"""

    def __init__(self, size: int = -1):
        # scratch pad is 2MB = 2<<20 bytes in total. preserve total * DXP_LX_FRAC_AVAIL
        # for backend usage unless specified otherwise
        if size == -1:
            size = int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))
        self.limit = size
        self.usage: dict = {}  # each record will be tensor_name:{"addr": yy, "size": zz}
        self.lx_usage_hist: list = []
        self.graph_lowering = V.graph

    def get_lowest_addr_in_use(self):
        if len(self.usage) > 0:
            return min([rec["addr"] for rec in self.usage.values()])
        return None

    def get_highest_addr_in_use(self):
        if len(self.usage) > 0:
            return max([rec["addr"] + rec["size"] for rec in self.usage.values()])
        return None

    def get_available_total(self):
        total_avail = self.limit
        for rec in self.usage.values():
            total_avail -= rec["size"]
        return total_avail

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
            return None
        else:
            # cannot find any free blocks
            return None

    def get_output_names(self) -> list[str]:
        return V.graph.get_output_names()

    def is_graph_input(self, buffer: str) -> bool:
        return buffer not in V.graph.name_to_buffer

    def find_inplace_address(
        self, tensor_name: str, mem_usage: dict, needed_size: int
    ) -> Optional[int]:
        found_matched_input = False
        ten_dev_lay = self.graph_lowering.get_buffer(tensor_name).layout.device_layout
        for inp_i in mem_usage["all_inputs"]:
            inp_i_dev_lay = self.graph_lowering.get_buffer(inp_i).layout.device_layout
            inp_i_on_lx = inp_i in self.usage
            inp_i_size_match = needed_size == mem_usage[inp_i]["size"]
            inp_i_lay_match = ten_dev_lay == inp_i_dev_lay
            inp_i_eol = mem_usage[inp_i]["last_usage"]
            if inp_i_on_lx and inp_i_size_match and inp_i_lay_match and inp_i_eol:
                found_matched_input = True
                break  # see TODO
        if found_matched_input:
            return self.usage[inp_i]["addr"]
        else:
            # NOTE allow_inplace also implies allow_output_to_lx
            return None

    def try_allocate(self, mem_usage: dict, idx: int, org_op_name: str):
        """
        Simple reuse rule:
        1. for an "input" tensor, find a matched tensor (name and size) on LX
        2. for output of an "inplace Op", find out which input (must be on LX) can be
            used to store output, use that input's LX addr
        3. for an output tensor, if this op is on the "white list" => prep for pinning
            => alloc a new LX block for the "output" of the op
        If can_reuse => add lx info to corresponding buffer.layout
        TODO if more than 1 matched input for inplace Op, is it good enough to always
             use the first one? e.g. C=A+B, A and B have same size, both on LX
        NOTE: 1. if an op, e.g. max, occurs multiple times on graph, output buffers will
                 have different names -> end-of-life analysis will take care of dealloc
              2. prev Op's sdsc.out.out.out.json may have useful info, not needed yet
              3. may be able to generalize this decision in buf end-of-life analysis
              4. greedy alloc may cause fragments, can further improve
        """
        graph_output_buf_name = self.get_output_names()
        for tensor_name in mem_usage["all_buf_used"]:
            tensor_info = mem_usage[tensor_name]
            is_graph_input = self.is_graph_input(tensor_name)
            is_graph_output = tensor_name in graph_output_buf_name
            needed_size = tensor_info["size"]
            is_input = tensor_info["is_input"]
            core_div_mismatch = (not is_input) and tensor_info["core_div_mismatch"]
            if is_graph_input or is_graph_output or core_div_mismatch:
                # graph input itself cannot be pinned (see try_insert_clone_())
                # graph output has to go back to HBM
                # if buf users have diff core-splits -> cause cross-core LX read/write
                continue

            # Decide whether to reuse/pin.
            addr = None
            tensor_on_lx = self.usage.get(tensor_name, {})
            size_match = tensor_on_lx.get("size", 0) == needed_size
            allow_output_to_lx = self.op_output_good_for_lx_reuse(org_op_name)
            allow_inplace = self.op_good_for_lx_inplace(org_op_name)

            if is_input and tensor_on_lx and size_match:
                addr = self.usage[tensor_name]["addr"]
            elif not is_input and allow_inplace:
                addr = self.find_inplace_address(tensor_name, mem_usage, needed_size)
                if (
                    addr is None
                ):  # NOTE 0 is a legitimate address, so we can't test "if not addr:"
                    addr = self.find_free_block(needed_size)
            elif not is_input and allow_output_to_lx:
                addr = self.find_free_block(needed_size)

            # add lx info into V.graph.buffers.layout for later codegen use.
            if addr is not None:
                self.usage[tensor_name] = {"addr": addr, "size": needed_size}

                self.allocate(tensor_name, addr)
                # Record usage history for debugging
                self.lx_usage_hist.append(
                    {
                        "node_idx": idx,
                        "op_name": org_op_name,
                        "tensor_name": tensor_name,
                        "addr": addr,
                        "size": needed_size,
                    }
                )

    def allocate(self, tensor_name: str, addr: int):
        buf = V.graph.get_buffer(tensor_name)
        layout = buf.get_layout()
        layout.allocation["lx"] = addr
        # NOTE assume same addr for same buf, no realloc needed/allowed

    def deallocate(self, bufs: list[str]):
        """Try to deallocate each of the buffers in a list, if exists."""
        if isinstance(bufs, str):
            bufs = [bufs]

        for buf in bufs:
            if buf in self.usage:
                del self.usage[buf]

    # TODO add dealloc and defrag mechanism to allocator later

    def op_output_good_for_lx_reuse(self, org_op_name: str) -> bool:
        return any(op in org_op_name for op in OP_OUTPUT_GOOD_FOR_LX_REUSE)

    def op_good_for_lx_inplace(self, org_op_name: str) -> bool:
        return any(op in org_op_name for op in OP_GOOD_FOR_LX_INPLACE)

    def mem_usage_by_op(
        self,
        op: ComputedBuffer,
        core_div_mismatch: dict[str, bool] = {},
        release_next: list = [],
    ) -> dict[str, dict[str, bool | int | str] | list[str]]:
        """
        Get a summary of memory usage of the given operation. Two types of info can be found
        1. Name lists, e.g. mem_usage["all_inputs"], or "all_outputs", "all_buf_used"
        2. Detailed info of individual buf, e.g. mem_usage[<buf_name>], which has
            "is_input", "size", "core_div_mismatch", "last_usage" fields
        NOTE:
        if a buf is not in core_div_mismatch => it has no users => graph output
        if a buf is on release_next => it's the last time it'll be used => allow inplace
        """
        rw = op.get_read_writes()
        mem_usage: dict[str, Any] = {
            "all_inputs": [],
            "all_outputs": [],
        }

        for is_input, deps in [(True, rw.reads), (False, rw.writes)]:
            for dep in deps:
                buf = V.graph.get_buffer(dep.name)
                dev_layout = buf.layout.device_layout
                dev_size = (
                    math.prod(dev_layout.device_size[:-1]) * 128
                )  # num_sticks * bytes_per_stick
                mem_usage[dep.name] = {
                    "is_input": is_input,
                    "size": dev_size,
                    "core_div_mismatch": core_div_mismatch.get(dep.name, False),
                    "last_usage": dep.name in release_next,
                }

            if is_input:
                mem_usage["all_inputs"].append(dep.name)
            else:
                mem_usage["all_outputs"].append(dep.name)

        mem_usage["all_buf_used"] = mem_usage["all_inputs"] + mem_usage["all_outputs"]

        return mem_usage


class AllocationStrategy:
    def plan_allocation(self, operations: list[Operation]):
        raise NotImplementedError("This is an abstract base class.")


class GreedyAllocationStrategy(AllocationStrategy):
    def __init__(
        self,
        alloc: Optional[ScratchPadAllocator] = None,
        graph_lowering: Optional[GraphLowering] = None,
    ):
        self.alloc = alloc if alloc else ScratchPadAllocator()
        self.graph_lowering = graph_lowering if graph_lowering else V.graph

    def should_consider_op(self, op: Operation) -> bool:
        return isinstance(op, ComputedBuffer) and not isinstance(
            op.layout, MutationLayoutSHOULDREMOVE
        )

    def consider_for_scratchpad(
        self,
        op: ComputedBuffer,
        idx: int,
        core_div_mismatch: dict[str, bool] = {},
        release_next: list = [],
    ):
        """
        If core_div_mismatch is not provided, we will consider LX pinning without taking
        core division into account (previous behavior), may result in slices of a LX tensor
        scattered over different core's scratchpad, which may result in unusable tensor and
        incorrect results.
        """
        # 1. summarize both inputs and output sizes used by this node, also merge core_div
        #    info into the table.
        mem_usage = self.alloc.mem_usage_by_op(op, core_div_mismatch, release_next)

        # 2. Try to allocate as many buffers on LX as we can. If successful, lx info (addr)
        #    will be added to buffer.FixedTiledLayout and used in generate_sdsc() later.
        org_op_name = op.origin_node.target._opname
        self.alloc.try_allocate(mem_usage, idx, org_op_name)

    def buf_analysis(self, operations: list[Operation]):
        """
        First, find out the last time each buffer was used. {buf1: idx_last_used, ...}
        Turn it into {idx_last_used+1:[buf1, ], ...}, ie. buffers to be deleted at given idx
        Then check core division -> If any of the operations on a given buffer has different
        core division => should not pin this buffer to LX
        NOTE Because each core can only write to its own scratchpad. For example, if a
              buffer is sliced 8 ways (stored on 8 LX) but next Op is 4-cores -> each core
              in next op has to read from 2 different scratchpads...
        TODO looking for options to broadcast to or all_reduce from multiple scratchpad
        """
        last_used: dict = {}
        buf_read_counts: dict[str, int] = {}
        buf_write_counts: dict[str, int] = {}
        buf_users: dict[str, Operation] = {}
        buf_users_read_and_write: dict[str, list[Operation]] = {}
        core_div_mismatch: dict[str, bool] = {}

        for idx, op in enumerate(operations):
            rw = op.get_read_writes()
            read_names = op.get_read_names()
            for dep in rw.reads | rw.writes:  # union of the OrderedSets
                buf = dep.name  # buffer name, i.e. a str
                last_used[buf] = idx
                if buf in read_names:
                    buf_read_counts[buf] = buf_read_counts.get(buf, 0) + 1
                    buf_users[buf] = buf_users.get(buf, []) + [op]
                else:
                    buf_write_counts[buf] = buf_write_counts.get(buf, 0) + 1
                buf_users_read_and_write[buf] = buf_users_read_and_write.get(
                    buf, []
                ) + [op]

        bufs_to_dealloc_at_idx: dict = {}
        for buf, idx in last_used.items():
            # if last used at idx => del at idx+1
            if idx + 1 in bufs_to_dealloc_at_idx:
                bufs_to_dealloc_at_idx[idx + 1].append(buf)
            else:
                bufs_to_dealloc_at_idx[idx + 1] = [buf]

        using_multicore = config.sencores > 1
        for buf_name, users_rw in buf_users_read_and_write.items():
            # this dict includes graph input and output
            same_core_div = True
            if using_multicore and len(users_rw) > 1:
                # graph input and output can have only 1 read or 1 write user.
                u0_split = users_rw[0].op_it_space_splits  # a list like [16, 1]
                same_core_div = all(
                    u0_split == u.op_it_space_splits for u in users_rw[1:]
                )
            core_div_mismatch[buf_name] = not same_core_div

        return bufs_to_dealloc_at_idx, buf_users, core_div_mismatch

    class NameSwapHandler(WrapperHandler):
        def __init__(self, inner, name_map: dict[str, str]):
            super().__init__(inner)
            self._name_map = name_map

        def load(self, name, index):
            return super().load(self._name_map.get(name, name), index)

    def create_Loop_hack_inner_fn(self, old_Loop, name_map):
        """Use ops_handler to swap the name of buffers"""

        def new_inner_fn(*args):
            # Pointwise has 1 pos arg index while Reduction has 2, i.e. (index, rindex)
            with V.set_ops_handler(self.NameSwapHandler(V.ops, name_map)):
                return old_Loop.inner_fn(*args)

        # old_Loop could be a Pointwise or Reduction.
        kwargs = {k: getattr(old_Loop, k) for k in old_Loop.__dataclass_fields__.keys()}
        kwargs["inner_fn"] = new_inner_fn
        new_Loop = old_Loop.__class__(**kwargs)
        # Additional attr that are not included in dataclass_fields. NOTE it relies on a
        # special method to force reset attrs of a frozen dataclass, see ir.Loops.create()
        new_Loop._post_init_setattr("origins", old_Loop.origins)
        new_Loop._post_init_setattr("origin_node", old_Loop.origin_node)
        new_Loop._post_init_setattr("traceback", old_Loop.traceback)
        # .get_stack_traces() get info from "origins", no need to manually set anything
        # LoopBody will be created later when we call CompBuf.recompute()

        return new_Loop

    def insert_op_after(
        self,
        buf: TensorBox,
        lowering_func: Callable,
        buf_users: dict,
        operations: list[Operation],
    ) -> None:
        """
        Insert an operation using the provided lowering function (e.g. clone_lowering) in
        GraphLowering.operations list after the given op (buf, a TensorBox representing a
        ComputedBuffer). Will update GraphLowering FX graph and the operations list.
        For example, original ops list looks like:
            buf0 -> buf1 -> buf2
        insert a clone of buf0 and let buf1 read from it, will become
            buf0 ->(clone) buf3 -> buf1 ->buf2

        NOTE:
        - Simplified flow, everything is done before Scheduler. Only take care of FX and
          GraphLowering. list operations will be updated inplace, no need to return.
        - Even though it is not a necessary condition, we assume FX graph and Operations are
          fully consistent and we will try to maintain it that way.
        - To update existing users of the old buffer -> hack the inner_fn then refresh LoopIR
        """
        fx_graph = self.graph_lowering.graph

        # Step 1: Add a new FX node for clone and update dependencies
        buf_name = buf.data.data.name  # buf is a TensorBox
        buf_fx = list(buf.origins)[0]  # .origin_node may not exist
        old_users = list(buf_fx.users.keys())
        # make sure the user-provided lowering_func is legit
        assert lowering_func in lowerings.values(), (
            f"The provided lowering function {lowering_func} is not properly registered."
        )
        LUTlower_func_to_op = {func: aten_op for aten_op, func in lowerings.items()}
        user_aten_op = LUTlower_func_to_op[lowering_func]
        # TODO this is a large dict, move it to upper scope so we only need to do it once
        # aten_op is the overloaded version, e.g. ops.aten.clone.*out* instead of .default
        fx_graph.inserting_after(buf_fx)
        new_fx_node = fx_graph.create_node("call_function", user_aten_op, (buf_fx,))
        for user in old_users:
            user.args = tuple(new_fx_node if ar is buf_fx else ar for ar in user.args)
        self.graph_lowering.orig_gm.recompile()

        # Step 2: Create a new ComBuf of a Pointwise IR (need to support Reduction?)
        pw_ir_tb = lowering_func(buf)  # a TensorBox wrapping a PointwiseIR
        new_com_buf = ComputedBuffer(
            name=None,
            layout=FixedTiledLayout(
                buf.layout.device,
                buf.layout.dtype,
                buf.layout.size,
                buf.layout.stride,
                buf.layout.device_layout,
            ),  # create a new copy of FixedTiledLayout from buf's layout
            data=pw_ir_tb.data.data,
        )
        new_com_buf.origins.add(new_fx_node)
        new_com_buf.origin_node = new_fx_node
        # TODO why arg0 ComputedBuffer doesn't have this attr?
        new_com_buf.name = self.graph_lowering.register_buffer(new_com_buf)
        self.graph_lowering.register_operation(new_com_buf)
        new_buf_name = new_com_buf.name

        # Step 3: Update graph_lowering.name_to_users (a list of TensorBox), eg, existing
        # users of arg0, other than InpBuf and new_buf, should become users of new_buf.
        users_of_inp, users_of_new_buf = [], []
        for tb in self.graph_lowering.name_to_users[buf_name]:
            if tb.data.data.name in [buf_name, new_buf_name]:
                users_of_inp.append(tb)
            else:
                users_of_new_buf.append(tb)
        self.graph_lowering.name_to_users[buf_name] = users_of_inp
        self.graph_lowering.name_to_users[new_buf_name] = users_of_new_buf

        # Step 4: Hack user nodes' inner_fn
        for old_com_buf in buf_users[buf_name]:
            # hack inner_fn with a nameSwapper ops handler and make a new LoopIR
            new_Loop = self.create_Loop_hack_inner_fn(
                old_com_buf.data, name_map={buf_name: new_buf_name}
            )
            old_com_buf.data = new_Loop

        # NOTE: operations is a reference to graph_lowering.operations, which is already
        # updated when we call graph_lowering.register_operation() earlier. But the new Op
        # was appended at the end of the list, need to insert at the correct position.
        first_user = buf_users[buf_name][0]
        idx_to_first_user = operations.index(first_user)
        operations.remove(new_com_buf)
        operations.insert(idx_to_first_user, new_com_buf)

    def try_insert_clone_op_for_inputs(
        self,
        operations: list[Operation],
        lx_free_total: int,
        buf_users: dict[str, Operation],
        core_div_mismatch: dict[str, bool],
    ) -> None:
        """
        Check if any input tensors can fit onto scratchpad and needed more than once =>
        Add corresponding "clone operation" to copy it to scratchpad and reduce HBM read.
        """
        for inp_name in self.graph_lowering.graph_input_names:
            buf = self.graph_lowering.get_buffer(inp_name)  # this is a TensorBox
            dev_layout = buf.layout.device_layout
            dev_size = math.prod(dev_layout.device_size[:-1]) * 128
            is_on_lx = buf.layout.allocation != {}
            used_only_once = len(buf_users[inp_name]) == 1
            if (
                used_only_once
                or dev_size > lx_free_total
                or is_on_lx
                or core_div_mismatch[inp_name]
            ):
                continue

            self.insert_op_after(buf, clone_lowering, buf_users, operations)

            lx_free_total -= dev_size

    @override
    def plan_allocation(self, operations: list[Operation]):
        idx_to_dealloc_bufs, buf_users, core_div_mismatch = self.buf_analysis(
            operations
        )

        if "clone" in OP_OUTPUT_GOOD_FOR_LX_REUSE:
            num_ops_before = len(operations)
            self.try_insert_clone_op_for_inputs(
                operations,
                self.alloc.get_available_total(),
                buf_users,
                core_div_mismatch,
            )

            # refresh LUTs -- insertion may not happen, e.g. input tensor is used only once
            if len(operations) > num_ops_before:
                idx_to_dealloc_bufs, buf_users, core_div_mismatch = self.buf_analysis(
                    operations
                )

        for idx, op in enumerate(operations):
            # release unneeded LX allocations before actual planning
            release_now = idx_to_dealloc_bufs.get(idx, [])
            release_next = idx_to_dealloc_bufs.get(idx + 1, [])
            self.alloc.deallocate(release_now)

            if self.should_consider_op(op):
                self.consider_for_scratchpad(op, idx, core_div_mismatch, release_next)
        # logger.info(alloc.lx_usage_hist)


def scratchpad_planning(
    operations: list[Operation],
    strategy: Optional[AllocationStrategy] = None,
) -> None:
    # Operations are in topological order (guaranteed by GraphLowering).
    # Core division has already been done.
    # Stickification has already been done (therefore all ComputedBuffers have FixedTiledLayouts).
    if not strategy:
        strategy = GreedyAllocationStrategy()
    strategy.plan_allocation(operations)

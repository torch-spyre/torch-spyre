from collections import defaultdict
import copy
from dataclasses import dataclass
import itertools
from typing import Callable, Optional, Iterable, override
from unittest import TestCase, expectedFailure
from enum import Enum
import os
from functools import wraps

from torch.utils._ordered_set import OrderedSet

from torch_spyre._inductor.scratchpad import (
    scratchpad_planning,
    ScratchPadAllocator,
    GreedyAllocationStrategy,
)
from torch_spyre._inductor import config

# From scratchpad.py
AVAILABLE_LX_SIZE = int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))

if os.environ.get("SCRATCHPAD_PATTERN_BYPASS_XFAIL", "0") == "1":
    # Define usuallyExpectedFailure as a no-op. This should show failures indicating that the
    # current allocation uses more HBM than the good allocation, not anything else.
    def usuallyExpectedFailure(test_item: Callable) -> Callable:
        @wraps(test_item)
        def wrapper(*args, **kwargs):
            return test_item(*args, **kwargs)

        return wrapper
else:
    usuallyExpectedFailure = expectedFailure


class BufferDeviceLayout:
    """This class mimics the FixedTiledLayout.device_layout field."""

    def __init__(self, size: int):
        self.device_size = [(size + 127) // 128, 128]


class BufferLayout:
    """This class mimics the TensorBox.layout field (a FixedTiledLayout)."""

    def __init__(self, size: int):
        self.device_layout = BufferDeviceLayout(size)
        self.size = size
        self.allocation = {}


class Buffer:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.layout = BufferLayout(size)
        self.data = self  # This helps 'scratchpad'


def make_buffer_registry(names_sizes: dict[str, int]) -> dict[str, Buffer]:
    return {name: Buffer(name=name, size=size) for (name, size) in names_sizes.items()}


@dataclass
class ReadWrites:
    reads: OrderedSet[Buffer]
    writes: OrderedSet[Buffer]


@dataclass
class Operation:
    name: str
    inputs: list[str]
    outputs: list[str]
    _buffer_registry: dict[str, "Buffer"]

    # To make scratchpad.py work, we add origin_node and target fields that point to the op itself,
    # a field _opname that is the same as name, and a field op_it_space_splits that is used in core
    # division. (If the value of op_it_space_splits is different for operations in a sequence, that
    # blocks LX allocation, so we make sure it is always the same.)
    op_it_space_splits = None
    origin_node = None
    target = None
    _opname = None

    def __post_init__(self):
        self.op_it_space_splits = []
        self.origin_node = self
        self.target = self
        self._opname = self.name

    def get_read_writes(self) -> ReadWrites:
        # Returns a list of (buffer_name, "read" or "write") for all buffers used by this operation.
        reads = OrderedSet(
            self._buffer_registry[buffer_name] for buffer_name in self.inputs
        )
        writes = OrderedSet(
            self._buffer_registry[buffer_name] for buffer_name in self.outputs
        )
        return ReadWrites(reads=reads, writes=writes)

    def get_read_names(self):
        return self.inputs


def make_operations(
    names_inputs_outputs: Iterable[tuple[str, str | list[str], str | list[str]]],
    buffers: dict[str, Buffer],
) -> list[Operation]:
    result = []
    for name, ins, outs in names_inputs_outputs:
        if isinstance(ins, str):
            ins = [ins]
        if isinstance(outs, str):
            outs = [outs]
        result.append(Operation(name, ins, outs, buffers))
    return result


class Component(Enum):
    LX = "LX"
    HBM = "HBM"


@dataclass
class Allocation:
    buffer: str
    component: Component = Component.LX
    # If the component is LX, then the address must be an integer. If the component is HBM, we don't
    # care about the address; this is encoded by the address being None. (This is enforced in
    # TestExamplePattern.verify_pattern.)
    address: Optional[int] = None


# A type alias for the result of an allocation. The ith entry in the list is the state during
# the ith operation. It maps each allocated buffer to the scratch pad address where it is
# allocated at that point in time.
AllocationResult = list[dict[str, Allocation]]


def make_nonevicting_allocation_result(
    buffers: dict[str, Buffer], addresses: dict[str, int], ops: list[Operation]
) -> AllocationResult:
    """Simple way to create an allocation result if buffers don't move around and stay in memory
    from their first to their last op."""
    allocations = {}
    for buffer_name in buffers:
        if buffer_name in addresses:
            allocations[buffer_name] = Allocation(
                buffer=buffer_name, address=addresses[buffer_name]
            )
        else:
            allocations[buffer_name] = Allocation(
                buffer=buffer_name, component=Component.HBM
            )

    first_use = {}
    last_use = {}
    for i, op in enumerate(ops):
        for buffer in op.inputs + op.outputs:
            if buffer not in first_use:
                first_use[buffer] = i
            last_use[buffer] = i

    return [
        {
            buffer_name: alloc
            for buffer_name, alloc in allocations.items()
            if first_use[buffer_name] <= i <= last_use[buffer_name]
        }
        for i in range(len(ops))
    ]


def make_general_allocation_result(lists: list[list[Allocation]]) -> AllocationResult:
    """Fully general way to create an allocation result, when make_nonevicting_allocation_result is
    not appropriate."""
    return [{alloc.buffer: alloc for alloc in lst} for lst in lists]


@dataclass
class Pattern:
    buffers: dict[str, Buffer]
    operations: list[Operation]
    # A "good" allocation pattern that we want to compare to. The test verifies that this pattern
    # is valid and that the current result is at least as good -- that is, the HBM usage of the
    # current result is no more than that of the good pattern.
    good_allocation: AllocationResult

    def determine_inputs_outputs(self) -> tuple[list[str], list[str]]:
        # A buffer is an input if it is read before it is written. A buffer is an output if it is
        # only written to.
        bufs_written_to = set()
        bufs_read_from = set()
        inputs = set()

        for op in self.operations:
            bufs_read_from.update(op.inputs)
            for buf in op.inputs:
                if buf not in bufs_written_to:
                    inputs.add(buf)
            bufs_written_to.update(op.outputs)

        outputs = list(bufs_written_to.difference(bufs_read_from))
        return (list(inputs), outputs)


class InstrumentedAllocator(ScratchPadAllocator):
    def __init__(self, pattern: Pattern, lowering: "MockGraphLowering"):
        super().__init__()
        self.allocations: dict[str, int] = {}
        # This overwrites the value set in the superclass constructor:
        self.graph_lowering = lowering
        self.inputs, self.outputs = pattern.determine_inputs_outputs()

    @override
    def op_output_good_for_lx_reuse(self, org_op_name: str) -> bool:
        return True

    @override
    def op_good_for_lx_inplace(self, org_op_name: str) -> bool:
        return True

    @override
    def allocate(self, tensor_name: str, addr: int):
        if tensor_name in self.allocations:
            # TODO: At this point we don't know where we are in terms of time / operations, so we
            # can't record at what point in time the allocation happens. This is okay as long as we
            # every buffer name can be uniquely allocated with a single address. In order to change
            # this, we need to store allocations differently, and then modify the logic for
            # measuring HBM usage in TestExamplePattern.hbm_usage_for_actual_run to account for
            # this. Also update TestExamplePattern.verify_actual_run to account for this.
            assert self.allocations[tensor_name] == addr, (
                f"Buffer {tensor_name} was already allocated at address "
                f"{self.allocations[tensor_name]}, but is being allocated again at address {addr}."
                f" That is probably a good improvement, but it means this test needs to be "
                f"adjusted."
            )
        self.allocations[tensor_name] = addr

    @override
    def mem_usage_by_op(
        self,
        op: Operation,
        core_div_mismatch: dict[str, bool] = {},
        release_next: list = [],
    ) -> dict[str, dict[str, bool | int | str] | list[str]]:
        # Returns a dict mapping each buffer name to a dict with keys "is_input" and "size".
        # is_input is True if the buffer is an input to the op, and False otherwise. size is the
        # size of the buffer.
        result = {}
        for tensor_name, is_input in itertools.chain(
            ((tensor_name, True) for tensor_name in op.inputs),
            ((tensor_name, False) for tensor_name in op.outputs),
        ):
            result[tensor_name] = {
                "is_input": is_input,
                "size": op._buffer_registry[tensor_name].size,
                "core_div_mismatch": False,
                "last_usage": tensor_name in release_next,
            }

        result["all_inputs"] = op.inputs
        result["all_outputs"] = op.outputs
        result["all_buf_used"] = op.inputs + op.outputs

        return result

    @override
    def get_output_names(self) -> list[str]:
        return self.outputs

    @override
    def is_graph_input(self, buffer: str) -> bool:
        return buffer in self.inputs


class MockGraphLowering:
    """This class impersonates V.graph."""

    def __init__(self, pattern: Pattern):
        self.graph_input_names = pattern.determine_inputs_outputs()[0]
        self.buffers = pattern.buffers

    def get_buffer(self, buf: str) -> Buffer:
        return self.buffers[buf]


class InstrumentedGreedyAllocationStrategy(GreedyAllocationStrategy):
    def __init__(
        self,
        pattern: Pattern,
        alloc: InstrumentedAllocator,
        lowering: MockGraphLowering,
    ):
        super().__init__(alloc, lowering)
        self.buffers = pattern.buffers
        self.operations = pattern.operations

    @override
    def should_consider_op(self, op: Operation) -> bool:
        return True

    def new_name(self, prefix: str, current_names: set[str]) -> str:
        candidate = prefix
        i = 0
        while candidate in current_names:
            candidate = f"{prefix}_{i}"
            i += 1
        return candidate

    @override
    def insert_op_after(
        self,
        buf: Buffer,
        lowering_func: Callable,
        buf_users: dict,
        operations: list[Operation],
    ) -> None:
        buf_index = [i for i, op in enumerate(operations) if buf.name in op.inputs]
        if not buf_index:
            raise ValueError(
                f"Was asked to insert after {buf.name}, but couldn't find it"
            )

        buffer_name = self.new_name("copy_buf", {buf for buf in self.buffers})
        self.buffers[buffer_name] = Buffer(buffer_name, buf.size)

        op_name = self.new_name("copy_op", {op.name for op in self.operations})
        new_op = Operation(
            op_name,
            inputs=[buf.name],
            outputs=[buffer_name],
            _buffer_registry=self.buffers,
        )

        # The expected order in the list of operations is actually *before* the first operation
        # that uses buf.
        self.operations.insert(buf_index[0], new_op)

        for op in self.operations[buf_index[0] + 1 :]:
            op.inputs = [
                buffer_name if buf.name == input else input for input in op.inputs
            ]
            op.outputs = [
                buffer_name if buf.name == output else output for output in op.outputs
            ]


class TestExamplePattern(TestCase):
    def map_buffers(
        self,
        operations: list[Operation],
        allocations: AllocationResult,
        *,
        see_later: Optional[Callable[[Operation, Allocation, str], None]] = None,
        see_first: Optional[Callable[[Operation, Allocation, str], None]] = None,
    ):
        """Returns the set of buffers that are used only once in the list of operations. see_first
        is called the first time any buffer is seen, and see_later is called any other time any
        buffer is seen."""
        seen_buffers = set()
        for op, alloc in zip(operations, allocations, strict=True):
            for buffer_name in op.inputs + op.outputs:
                if buffer_name in seen_buffers:
                    if see_later is not None:
                        see_later(op, alloc[buffer_name], buffer_name)
                else:
                    if see_first is not None:
                        see_first(op, alloc[buffer_name], buffer_name)
                    seen_buffers.add(buffer_name)

    def verify_pattern(self, pattern: Pattern, *, inplace: bool = False):
        allocation = pattern.good_allocation
        operations = pattern.operations
        self.assertEqual(
            len(allocation),
            len(operations),
            f"Good allocation should have the same number of entries as the number of operations, "
            f"but found {len(allocation)} allocations and {len(operations)} operations.",
        )
        for alloc in allocation:
            for a in alloc.values():
                self.assertEqual(
                    a.address is not None,
                    a.component == Component.LX,
                    f"Buffers should have an address iff they are allocated in LX, but found {a}.",
                )

        # Verify that we didn't write any operations that write to a buffer, except possibly the
        # first time we see that buffer, unless this pattern is marked as inplace.
        def no_output(op: Operation, _: Allocation, buffer_name: str):
            self.assertNotIn(
                buffer_name,
                op.outputs,
                f"Buffer {buffer_name} is written to in operation {op.name}, but accessed before "
                f"that operation. However, this test is case is not marked as in-place, so we "
                f"avoid in-place operations.",
            )

        # Verify that the first access to any buffer in LX is a write access.
        def is_hbm_or_write(op: Operation, alloc: Allocation, buffer_name: str):
            self.assertTrue(
                alloc.component == Component.HBM or buffer_name in op.outputs,
                f"Buffer {buffer_name} is read from LX in operation {op.name} without first being "
                f"written into it",
            )

        self.map_buffers(
            operations,
            allocation,
            see_first=is_hbm_or_write,
            see_later=None if inplace else no_output,
        )

        for i, op in enumerate(operations):
            # Check that each buffer that is used is allocated (either in LX or HBM).
            for buffer_name in op.inputs + op.outputs:
                self.assertTrue(
                    any(
                        alloc.buffer == buffer_name for alloc in allocation[i].values()
                    ),
                    f"Buffer {buffer_name} used by operation {op.name} is not allocated at "
                    f"this point in the good allocation pattern, but it is used more than once.",
                )

            # Check that there is at least one output.
            self.assertGreater(
                len(op.outputs),
                0,
                f"Operation {op.name} should have at least one output.",
            )

            # Check that allocated buffers do not overlap.
            allocated_buffers = [
                alloc for alloc in allocation[i].values() if alloc.address is not None
            ]
            if allocated_buffers:
                # Sort by address:
                sorted_allocations = sorted(
                    list(allocated_buffers),
                    key=lambda x: x.address,  # pyright: ignore[reportCallIssue, reportArgumentType]
                )
                for j in range(len(sorted_allocations) - 1):
                    buffer_name_j = sorted_allocations[j].buffer
                    addr_j = sorted_allocations[j].address
                    buffer_name_next = sorted_allocations[j + 1].buffer
                    addr_next = sorted_allocations[j + 1].address
                    size_j = op._buffer_registry[buffer_name_j].size
                    self.assertLessEqual(
                        addr_j + size_j,
                        addr_next,
                        f"Buffers {buffer_name_j} and {buffer_name_next} overlap during operation "
                        f"{op.name}",
                    )

                self.assertLessEqual(
                    sorted_allocations[-1].address
                    + op._buffer_registry[sorted_allocations[-1].buffer].size,
                    AVAILABLE_LX_SIZE,
                    f"Buffer {sorted_allocations[-1].buffer} exceeds scratch pad size during "
                    f"operation {op.name}",
                )

    def verify_actual_run(self, pattern: Pattern, alloc: InstrumentedAllocator):
        # Verify that the actual run's allocation is valid. We assume that any allocation is "live"
        # during the entire liveness of the corresponding buffer.
        liveness_start = {}
        liveness_end = {}
        for i, op in enumerate(pattern.operations):
            for buffer_name in op.inputs + op.outputs:
                if buffer_name not in liveness_start:
                    liveness_start[buffer_name] = i
                liveness_end[buffer_name] = i

        # Sanity check -- every buffer should have a start and an end to its liveness.
        self.assertTrue(set(liveness_start.keys()) == set(liveness_end.keys()))

        allocate_at = defaultdict(list)
        deallocate_at = defaultdict(list)
        for buffer_name in liveness_start:
            if buffer_name in alloc.allocations:
                allocate_at[liveness_start[buffer_name]].append(buffer_name)
                deallocate_at[liveness_end[buffer_name] + 1].append(buffer_name)

        live_buffers = set()
        for i, op in enumerate(pattern.operations):
            live_buffers.update(allocate_at[i])
            for buffer_name in op.inputs + op.outputs:
                if buffer_name not in alloc.allocations:
                    # This buffer resides in HBM.
                    continue

                # Verify that buffer_name does not overlap with any allocated buffers at this point.
                addr = alloc.allocations[buffer_name]
                size = op._buffer_registry[buffer_name].size
                self.assertLessEqual(
                    addr + size,
                    AVAILABLE_LX_SIZE,
                    f"Buffer {buffer_name} exceeds scratch pad size during operation {op.name}",
                )
                for other_buffer_name in live_buffers:
                    if (
                        other_buffer_name == buffer_name
                        or other_buffer_name not in alloc.allocations
                    ):
                        continue
                    other_addr = alloc.allocations[other_buffer_name]
                    other_size = op._buffer_registry[other_buffer_name].size
                    if addr <= other_addr:
                        self.assertLessEqual(
                            addr + size,
                            other_addr,
                            f"Buffers {buffer_name} and {other_buffer_name} overlap during "
                            f"operation {op.name}",
                        )
                    else:
                        self.assertLessEqual(
                            other_addr + other_size,
                            addr,
                            f"Buffers {buffer_name} and {other_buffer_name} overlap during "
                            f"operation {op.name}",
                        )
            live_buffers.difference_update(deallocate_at[i + 1])

    def hbm_usage_for_good_allocation(
        self, allocation: AllocationResult, operations: list[Operation]
    ) -> int:
        hbm_usage = 0

        for op, alloc in zip(operations, allocation, strict=True):
            for buffer_name in op.inputs + op.outputs:
                if alloc[buffer_name].component == Component.HBM:
                    hbm_usage += op._buffer_registry[buffer_name].size

        return hbm_usage

    def hbm_usage_for_actual_run(
        self, operations: list[Operation], alloc: InstrumentedAllocator
    ) -> int:
        hbm_usage = 0

        # Count all usage for buffers not allocated in the scratchpad.
        for op in operations:
            for buffer_name in op.inputs + op.outputs:
                if buffer_name not in alloc.allocations:
                    # This buffer is not allocated in the scratch pad before this operation, so it
                    # must be loaded from HBM.
                    hbm_usage += op._buffer_registry[buffer_name].size

        return hbm_usage

    def run_pattern(self, pattern: Pattern):
        # The scratchpad_planning operation may modify the pattern (adding operations), and then
        # examining the "good" allocation will run into trouble.
        pattern_copy = copy.deepcopy(pattern)
        lowering = MockGraphLowering(pattern_copy)
        alloc = InstrumentedAllocator(pattern_copy, lowering)
        strategy = InstrumentedGreedyAllocationStrategy(pattern_copy, alloc, lowering)

        scratchpad_planning(pattern_copy.operations, strategy)

        # Verify that the currently implemented allocation is indeed valid
        self.verify_actual_run(pattern_copy, alloc)

        # Verify that the currently implemented allocation is at least as good as the "good
        # allocation" in terms of HBM usage.
        current_hbm_usage = self.hbm_usage_for_actual_run(
            pattern_copy.operations, alloc
        )
        good_hbm_usage = self.hbm_usage_for_good_allocation(
            pattern.good_allocation, pattern.operations
        )
        self.assertLessEqual(
            current_hbm_usage,
            good_hbm_usage,
            f"Current allocation uses more HBM ({current_hbm_usage} bytes) than the good allocation ({good_hbm_usage} bytes). ",
        )

    def make_simple_fragmentation_pattern(self) -> Pattern:
        """Allocate two buffers A and B that are each a third of the available scratchpad size,
        where A can be freed after the second operation. Then allocate a third buffer C
        that is two thirds of the scratchpad size. This can only fit if B was allocated at the start
        or end of the scratchpad, leaving a contiguous region for C."""
        third_scratchpad_size = AVAILABLE_LX_SIZE // 3
        third_scratchpad_size = (
            third_scratchpad_size // 128
        ) * 128  # round down to a multiple of the stick size
        buffers = make_buffer_registry(
            {
                "A": third_scratchpad_size,
                "A_LX": third_scratchpad_size,
                "B": third_scratchpad_size,
                "C": 2 * third_scratchpad_size,
                "D": third_scratchpad_size,
                "E": third_scratchpad_size,
            }
        )

        ops = make_operations(
            [
                ("op0", "A", "A_LX"),
                ("op1", "A_LX", "B"),
                ("op2", ["A_LX", "B"], "D"),
                ("op3", "B", "C"),
                ("op4", ["B", "C"], "E"),
            ],
            buffers,
        )

        # A_LX is used only during op1 and op2, so we allocate it after B. This way we can
        # evict it after op2 and have enough space for C during op3.
        good_allocation = make_nonevicting_allocation_result(
            buffers,
            {"A_LX": third_scratchpad_size, "B": 0, "C": third_scratchpad_size},
            ops,
        )
        return Pattern(buffers, ops, good_allocation=good_allocation)

    def test_verify_simple_fragmentation_pattern(self):
        self.verify_pattern(self.make_simple_fragmentation_pattern())

    @usuallyExpectedFailure
    def test_simple_fragmentation_pattern(self):
        self.run_pattern(self.make_simple_fragmentation_pattern())

    def make_staircase_pattern(self) -> Pattern:
        """Allocate N*2 buffers of sizes k, k, 2*k, 2*k, 3*k, 3*k, ..., N*k, N*k. After an
        even-numbered buffer is allocated, free the previous odd-numbered buffer. This creates a
        "staircase" pattern of allocations that can only be fit if the allocator is smart about
        fragmentation. In that case, the maximum scratchpad usage is
        k + 2*k + ... + N*k + N*k = k * N * (N + 1) / 2 + N * k = k * N * (N + 3) / 2, so we choose
        k such that this is just less than the available scratchpad size.

        The greedy allocator will always allocate the next buffer just after all other buffers,
        because no gap is big enough for the current size. So it uses
        2 * (k + 2*k + ... + N*k) = k * N * (N + 1) or roughly 2/3 times more."""
        N = 7
        k = (2 * AVAILABLE_LX_SIZE) // (N * (N + 3))
        k = (k // 128) * 128  # round down to a multiple of the stick size

        # This only works if the greedy allocator uses more than fits in the scratchpad, so we
        # assert that here.
        self.assertGreater(k * N * (N + 1), AVAILABLE_LX_SIZE)

        buffers = make_buffer_registry(
            {f"{letter}{i}": i * k for i in range(1, N + 1) for letter in ["A", "B"]}
            | {f"A{i}_HBM": i * k for i in range(1, N + 1)}
            | {f"C{i}": k for i in range(1, N + 2)}
        )

        def op_tuples(i: int) -> list[tuple[str, str | list[str], str]]:
            return [
                (f"op{i}_load", f"A{i}_HBM", f"A{i}"),
                (f"op{i}_0", f"A{i}", f"B{i}"),
                (f"op{i}_1", [f"A{i}", f"B{i}"], f"C{i}"),
            ]

        ops = make_operations(
            [op for i in range(1, N + 1) for op in op_tuples(i)]
            + [("op_final", [f"B{i}" for i in range(1, N + 1)], f"C{N + 1}")],
            buffers,
        )

        good_allocation = make_nonevicting_allocation_result(
            buffers,
            {f"A{i}": 0 for i in range(1, N + 1)}
            | {f"B{i}": (N + i * (i - 1) // 2) * k for i in range(1, N + 1)},
            ops,
        )

        pattern = Pattern(buffers, ops, good_allocation=good_allocation)
        return pattern

    def test_verify_staircase_pattern(self):
        self.verify_pattern(self.make_staircase_pattern())

    @usuallyExpectedFailure
    def test_staircase_pattern(self):
        self.run_pattern(self.make_staircase_pattern())

    def make_downward_staircase_pattern(self) -> Pattern:
        """Allocate 1+N*2 buffers of sizes k, N*k, N*k, (N-1)*k, (N-1)*k, ..., 2*k, 2*k, k, k.
        After an odd-numbered buffer (>1) is allocated, free the previous even-numbered buffer.
        This creates an easier "staircase" pattern of allocations than in
        `make_staircase_pattern`. Still, the greedy allocator will prefer to allocate
        buffers at the end if it can't allocate them at address 0. So we first allocate one small
        buffer at the start which will block address 0. In the optimal case, the maximum scratchpad
        usage is k + N*k + (N-1)*k + ... + 2*k + k + k = k * (4 + N * (N + 1)) / 2, so we choose k
        such that this is just less than the available scratchpad size.

        The greedy allocator will always allocate the next buffer just after all other buffers,
        up until the point where it reaches the top of available memory and starts looking for gaps.
        The total usage is less clear to analyze."""
        N = 5
        k = (2 * AVAILABLE_LX_SIZE) // (4 + N * (N + 1))
        k = (k // 128) * 128  # round down to a multiple of the stick size

        buffers = make_buffer_registry(
            {
                f"{letter}{i}": (N + 1 - i) * k
                for i in range(1, N + 1)
                for letter in ["A", "B"]
            }
            | {f"A{i}_HBM": (N + 1 - i) * k for i in range(1, N + 1)}
            | {"Z": k}
            | {"Z_HBM": k}
            | {f"C{i}": k for i in range(N + 2)}
        )

        def op_tuples(i: int) -> list[tuple[str, str | list[str], str]]:
            return [
                (f"op{i}_load", f"A{i}_HBM", f"A{i}"),
                (f"op{i}_0", f"A{i}", f"B{i}"),
                (f"op{i}_1", [f"A{i}", f"B{i}"], f"C{i}"),
            ]

        ops = make_operations(
            [
                ("op_start_load", "Z_HBM", "Z"),
                ("op_start", "Z", "C0"),
            ]
            + [op for i in range(1, N + 1) for op in op_tuples(i)]
            + [("op_final", ["Z"] + [f"B{i}" for i in range(1, N + 1)], f"C{N + 1}")],
            buffers,
        )

        good_allocation = make_nonevicting_allocation_result(
            buffers,
            {"Z": 0}
            | {f"A{i}": k for i in range(1, N + 1)}
            | {f"B{i}": ((N - i) * (N - i + 1) // 2 + 2) * k for i in range(1, N + 1)},
            ops,
        )

        pattern = Pattern(buffers, ops, good_allocation=good_allocation)
        return pattern

    def test_verify_downward_staircase_pattern(self):
        self.verify_pattern(self.make_downward_staircase_pattern())

    @usuallyExpectedFailure
    def test_downward_staircase_pattern(self):
        self.run_pattern(self.make_downward_staircase_pattern())

    def make_simple_eviction_pattern(self) -> Pattern:
        """This pattern requires allocating a buffer, evicting it, and then reallocating it later.

        We use two buffers A and B that are each exactly the available LX size. We have six
        operations. The first two use A, the next two use B, and the last two use A again. Optimal
        use would allocate A and B for two ops each at alternate times."""
        buffers = make_buffer_registry(
            {
                buf: AVAILABLE_LX_SIZE
                for buf in ["A", "B", "A_HBM", "B_HBM"] + [f"C{i}" for i in range(1, 7)]
            }
        )
        ops = make_operations(
            [
                ("loadA_0", "A_HBM", "A"),
                ("op1", "A", "C1"),
                ("op2", "A", "C2"),
                ("loadB", "B_HBM", "B"),
                ("op3", "B", "C3"),
                ("op4", "B", "C4"),
                ("loadA_1", "A_HBM", "A"),
                ("op5", "A", "C5"),
                ("op6", "A", "C6"),
            ],
            buffers,
        )

        good_allocation = [
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="A_HBM", component=Component.HBM),
            ],
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="C1", component=Component.HBM),
            ],
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="C2", component=Component.HBM),
            ],
            [
                Allocation(buffer="B", address=0),
                Allocation(buffer="B_HBM", component=Component.HBM),
            ],
            [
                Allocation(buffer="B", address=0),
                Allocation(buffer="C3", component=Component.HBM),
            ],
            [
                Allocation(buffer="B", address=0),
                Allocation(buffer="C4", component=Component.HBM),
            ],
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="A_HBM", component=Component.HBM),
            ],
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="C5", component=Component.HBM),
            ],
            [
                Allocation(buffer="A", address=0),
                Allocation(buffer="C6", component=Component.HBM),
            ],
        ]

        pattern = Pattern(
            buffers,
            ops,
            good_allocation=make_general_allocation_result(good_allocation),
        )
        return pattern

    def test_verify_simple_eviction_pattern(self):
        self.verify_pattern(self.make_simple_eviction_pattern(), inplace=True)

    @usuallyExpectedFailure
    def test_simple_eviction_pattern(self):
        self.run_pattern(self.make_simple_eviction_pattern())

    def make_eviction_reallocation_pattern(self) -> Pattern:
        """This pattern requires allocating a buffer, evicting it, and then reallocating it later
        at a different address to achieve optimality.

        We use four buffers total: A0, A1, A2 of size 1/3 the available size, and B of size twice
        that. We first ensure that A0, A1, and A2 must be allocated together, then A0 and B, then
        A1 and B, and finally A2 and B. Because B can fit only with one of the A buffers at the top
        or the bottom, whichever one was allocated in the middle must be moved.

        We ensure that any set is allocated together in an optimal allocation by using four ops
        in a row that use them all as input. This means that, whatever was in the scratchpad before
        and whatever is in it after, we can complete that phase with one full scratchpad worth of
        HBM transfers. On the other hand, if not everything is allocated on the scratchpad, then we
        have to stream at least one buffer four times, which entails at least 4/3 of the scratchpad
        size in HBM transfers."""
        A_size = AVAILABLE_LX_SIZE // 3
        A_size = (A_size // 128) * 128  # round down to a multiple of the stick size
        B_size = 2 * A_size

        # This will work if 4 * A_size > AVAILABLE_LX_SIZE.
        self.assertGreater(4 * A_size, AVAILABLE_LX_SIZE)

        pattern = [["A0", "A1", "A2"], ["A0", "B"], ["A1", "B"], ["A2", "B"]]

        buffers = make_buffer_registry(
            {f"S{i}_HBM": A_size for i in range(len(pattern))}
            | {f"A{i}": A_size for i in range(3)}
            | {"B": B_size}
            | {f"C{i}_{j}": A_size for i in range(4) for j in range(4)}
        )

        op_spec = [
            [
                (f"op{i}_load", f"S{i}_HBM", group),
                *[(f"op{i}_{j}", group, f"C{i}_{j}") for j in range(4)],
            ]
            for i, group in enumerate(pattern)
        ]
        ops = make_operations(
            [tupl for tup_lst in op_spec for tupl in tup_lst], buffers
        )

        addresses_per_group = [
            {"A0": 0, "A1": A_size, "A2": 2 * A_size},
            {"A0": 0, "B": A_size},
            {"A1": 0, "B": A_size},
            {"A2": 0, "B": A_size},
        ]

        good_allocations = []
        for i, group in enumerate(pattern):
            good_allocations.append(
                [Allocation(buffer=f"S{i}_HBM", component=Component.HBM)]
                + [
                    Allocation(buffer=buffer, address=addresses_per_group[i][buffer])
                    for buffer in group
                ]
            )

            for j in range(4):
                good_allocations.append(
                    [
                        Allocation(
                            buffer=buffer, address=addresses_per_group[i][buffer]
                        )
                        for buffer in group
                    ]
                    + [Allocation(buffer=f"C{i}_{j}", component=Component.HBM)]
                )

        pattern = Pattern(
            buffers,
            ops,
            good_allocation=make_general_allocation_result(good_allocations),
        )
        return pattern

    def test_verify_eviction_reallocation_pattern(self):
        self.verify_pattern(self.make_eviction_reallocation_pattern(), inplace=True)

    @usuallyExpectedFailure
    def test_eviction_reallocation_pattern(self):
        self.run_pattern(self.make_eviction_reallocation_pattern())

    def make_gqattention_pattern(self) -> Pattern:
        """We describe the GQA attention pattern. The "input" are three tensors, Q, K, and V. The
        dimensions of Q are typically B x Hq x S x D; the dimensions of K and V are B x Hkv x S x D.
        Here B is the batch size, Hq is the number of query heads, Hkv is the number of key/value
        heads (typically 1/4 or 1/8 of Hq); S is the sequence length; and D is the head dimension.

        The algorithm is essentially as follows, expanding the softmax to its constituent
        operations, fused to reductions / pointwise operations / matmuls with scaling and
        transposition and listing two broadcasts explicitly:

        K_broadcast = broadcast(K, Hq // Hkv, dim=1)   # dim: B x Hq x S x D, though typically the
                                                       # B x Hkv x S x D in memory
        Q_K = Q @ K_broadcast.transpose(-2, -1) / scalar   # dim: B x Hq x S x S
        m = max(Q_K, dim=-1)                           # dim: B x Hq x S
        numerators = exp(Q_K - m)                      # dim: B x Hq x S x S
        denominators = sum(numerators, dim=-1)         # dim: B x Hq x S
        scores = numerators / denominators             # dim: B x Hq x S x S
        V_broadcast = broadcast(V, Hq // Hkv, dim=1)   # dim: B x Hq x S x D, though typically the
                                                       # B x Hkv x S x D in memory
        output = scores @ V                            # dim: B x Hq x S x D

        The scalar is sqrt(Hq).

        Let's write G = Hq // Hkv (usually 4 or 8, as mentioned above) and write N = B x Hkv x S.
        Then the buffer sizes in memory are:

        N x G x D  (Q, output);
        N x D      (K, V, K_broadcast, V_broadcast);
        N x D x S  (Q_K, numerators, scores);
        N x G      (m, denominators).

        During the first matmul, we need buffers of total size N x D x (G + 1 + S); then when
        computing numerators, we need buffers of total size N x (2 x D x S + G); same when computing
        scores; and to compute output, we need N x D x (G + 1 + S) again. We choose the parameters
        so that both N x D x (G + 1 + S) and N x (2 x D x S + G) fit into LX, but only just.

        In the most general version, both 'scores' and 'output' are returned to the caller."""

        G = 8
        D = 64
        S = 16
        self.assertGreater(2 * D * S + G, G + 1 + S, "test is written assuming this")
        N = AVAILABLE_LX_SIZE // (2 * D * S + G)

        NGD, ND, NDS, NG = tuple(
            (x // 128) * 128 for x in [N * G * D, N * D, N * D * S, N * G]
        )

        buffers = make_buffer_registry(
            {
                "Q_HBM": NGD,
                "Q": NGD,
                "K_HBM": ND,
                "K": ND,
                "Q_K": NDS,
                "m": NG,
                "numerators": NDS,
                "denominators": NG,
                "scores": NDS,
                "scores_HBM": NDS,
                "V_HBM": ND,
                "V": ND,
                "output": NGD,
                "output_HBM": NGD,
            }
        )

        ops = make_operations(
            [
                ("load_Q", "Q_HBM", "Q"),
                ("load_K", "K_HBM", "K"),
                ("matmul_t", ["Q", "K"], "Q_K"),
                ("max", "Q_K", "m"),
                ("exp_sub", ["Q_K", "m"], "numerators"),
                ("sum", "numerators", "denominators"),
                ("div", ["numerators", "denominators"], "scores"),
                ("save_scores", "scores", "scores_HBM"),
                ("load_V", "V_HBM", "V"),
                ("matmul", ["scores", "V"], "output"),
                ("save_output", "output", "output_HBM"),
            ],
            buffers,
        )

        good_allocation = make_nonevicting_allocation_result(
            buffers,
            {
                "Q": NDS,
                "K": NDS + NGD,
                "Q_K": 0,
                "m": NDS,
                "numerators": NDS + NG,
                "denominators": NDS,
                "scores": 0,
                "V": NDS,
                "output": NDS + ND,
            },
            ops,
        )

        pattern = Pattern(buffers, ops, good_allocation=good_allocation)
        return pattern

    def test_verify_gqattention_pattern(self):
        self.verify_pattern(self.make_gqattention_pattern())

    @usuallyExpectedFailure
    def test_gqattention_pattern(self):
        self.run_pattern(self.make_gqattention_pattern())


if __name__ == "__main__":
    import unittest

    unittest.main()

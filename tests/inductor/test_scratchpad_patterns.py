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


from collections import defaultdict
import copy
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Iterable
from unittest import TestCase, expectedFailure
from enum import Enum
import os

import torch

from torch_spyre._inductor.scratchpad.allocator import (
    LifetimeBoundBuffer,
    _lx_planning_size,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import (
    SimulatedAnnealingLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    MemoryPlanSolver,
    GreedyLayoutSolver,
    LifetimeBoundBuffer as Buffer,
)

AVAILABLE_LX_SIZE = _lx_planning_size()

BYPASS_XFAIL = os.environ.get("SCRATCHPAD_PATTERN_BYPASS_XFAIL", "0") == "1"


def make_buffer_registry(names_sizes: dict[str, int]) -> dict[str, Buffer]:
    return {
        name: Buffer(name=name, size=size, uses=[])
        for (name, size) in names_sizes.items()
    }


@dataclass
class Operation:
    name: str
    inputs: list[str]
    output: str
    _buffer_registry: dict[str, Buffer]

    @property
    def outputs(self):
        return [self.output]


def make_operations(
    names_inputs_outputs: Iterable[tuple[str, str | list[str], str]],
    buffers: dict[str, Buffer],
) -> list[Operation]:
    result = []
    for name, ins, out in names_inputs_outputs:
        if isinstance(ins, str):
            ins = [ins]
        assert isinstance(out, str)
        result.append(Operation(name, ins, out, buffers))
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
    # PatternTests.verify_pattern.)
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
    inplace: bool = False

    def __post_init__(self):
        for i, op in enumerate(self.operations):
            for buffer_name in op.inputs + op.outputs:
                buffer = self.buffers[buffer_name]
                if i not in buffer.uses:
                    buffer.uses.append(i)

        self.inputs, self.outputs = self.determine_inputs_outputs()

        for input_name in self.inputs:
            self.buffers[input_name].first_use_is_read = True

        for i, op in enumerate(self.operations):
            output_buffer = self.buffers[op.outputs[0]]
            for buffer_name in op.inputs:
                buffer = self.buffers[buffer_name]
                if (
                    buffer_name not in self.inputs + self.outputs
                    and buffer.uses
                    and buffer.uses[-1] == i
                    and buffer.size == output_buffer.size
                ):
                    output_buffer.in_place_parents.append(buffer_name)

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


class PatternTests:
    """Mixin providing pattern definitions and test infrastructure.

    Concrete subclasses are created with a ``role`` keyword:
    - ``role="verify"``  — generates ``test_verify_{name}_pattern`` methods that validate the
      good allocation defined in the pattern
    - ``role="solver"``  — generates ``test_{name}_pattern`` methods that run the solver and
      check it meets the good allocation; names in ``expected_failures`` are wrapped with
      ``@expectedFailure``
    """

    PATTERNS: ClassVar[list[str]] = [
        "simple_fragmentation",
        "staircase",
        "downward_staircase",
        "simple_eviction",
        "eviction_reallocation",
        "gqattention",
        "moe_mlp",
    ]

    def __init_subclass__(cls, role: str = "", **kwargs: object) -> None:
        # This creates tests like test_verify_moe_mlp_pattern
        # and test_moe_mlp_pattern
        super().__init_subclass__(**kwargs)  # type: ignore[misc]
        if role == "verify":
            for name in PatternTests.PATTERNS:

                def _test_verify(self, _n: str = name) -> None:
                    self.verify_pattern(getattr(self, f"make_{_n}_pattern")())

                setattr(cls, f"test_verify_{name}_pattern", _test_verify)
        elif role == "solver":
            xfails: frozenset[str] = getattr(cls, "expected_failures", frozenset())
            for name in PatternTests.PATTERNS:

                def _test_solve(self, _n: str = name) -> None:
                    self.run_pattern(
                        self.solver_type, getattr(self, f"make_{_n}_pattern")()
                    )

                if name in xfails and not BYPASS_XFAIL:
                    _test_solve = expectedFailure(_test_solve)
                setattr(cls, f"test_{name}_pattern", _test_solve)

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0xAFFE)

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

    def verify_pattern(self, pattern: Pattern):
        allocation = pattern.good_allocation
        operations = pattern.operations
        inplace = pattern.inplace
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

    def verify_actual_run(
        self, pattern: Pattern, planned_buffers: dict[str, LifetimeBoundBuffer]
    ):
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
            if buffer_name not in planned_buffers:
                # This buffer is a graph input or output.
                continue
            addr = planned_buffers[buffer_name].address
            if addr is None:
                # This buffer resides in HBM.
                continue
            allocate_at[liveness_start[buffer_name]].append(buffer_name)
            deallocate_at[liveness_end[buffer_name] + 1].append(buffer_name)

        live_buffers = set()
        for i, op in enumerate(pattern.operations):
            live_buffers.update(allocate_at[i])
            for buffer_name in op.inputs + op.outputs:
                # Verify that buffer_name does not overlap with any allocated buffers at this point.
                if buffer_name not in planned_buffers:
                    # This buffer is a graph input or output.
                    continue
                addr = planned_buffers[buffer_name].address
                if addr is None:
                    # This buffer resides in HBM.
                    continue
                size = planned_buffers[buffer_name].size
                self.assertLessEqual(
                    addr + size,
                    AVAILABLE_LX_SIZE,
                    f"Buffer {buffer_name} exceeds scratch pad size during operation {op.name}",
                )
                for other_buffer_name in live_buffers:
                    other_addr = planned_buffers[other_buffer_name].address
                    if other_buffer_name == buffer_name or other_addr is None:
                        continue
                    other_size = planned_buffers[other_buffer_name].size
                    if addr == other_addr:
                        self.assertTrue(
                            other_buffer_name
                            in planned_buffers[buffer_name].in_place_parents
                            or buffer_name
                            in planned_buffers[other_buffer_name].in_place_parents,
                            f"Buffers {buffer_name} and {other_buffer_name} overlap during operation {op.name}",
                        )
                    elif addr <= other_addr:
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
        self, pattern: Pattern, planned_buffers: dict[str, LifetimeBoundBuffer]
    ) -> int:
        hbm_usage = 0
        allocation = pattern.good_allocation

        for op, alloc in zip(pattern.operations, allocation, strict=True):
            for buffer_name in op.inputs + op.outputs:
                if alloc[buffer_name].component == Component.HBM:
                    hbm_usage += pattern.buffers[buffer_name].size

        return hbm_usage

    def hbm_usage_for_actual_run(
        self, pattern: Pattern, planned_buffers: dict[str, LifetimeBoundBuffer]
    ) -> int:
        hbm_usage = 0

        # Count all usage for buffers not allocated in the scratchpad.
        for op in pattern.operations:
            for buffer_name in op.inputs + op.outputs:
                if (
                    buffer_name not in planned_buffers
                    or planned_buffers[buffer_name].address is None
                ):
                    # This buffer is not allocated in the scratch pad, so it
                    # must be loaded from HBM.
                    hbm_usage += pattern.buffers[buffer_name].size

        return hbm_usage

    def run_pattern(self, solver_type: type[MemoryPlanSolver], pattern: Pattern):
        solver = solver_type(AVAILABLE_LX_SIZE)

        buffers_to_plan = [
            copy.deepcopy(buf)
            for buf in pattern.buffers.values()
            if buf.name not in pattern.inputs + pattern.outputs
        ]
        planned_buffers = solver.plan_layout(buffers_to_plan)
        planned_buffers = {buf.name: buf for buf in planned_buffers}
        # Verify that the currently implemented allocation is indeed valid
        self.verify_actual_run(pattern, planned_buffers)

        # Verify that the currently implemented allocation is at least as good as the "good
        # allocation" in terms of HBM usage.
        current_hbm_usage = self.hbm_usage_for_actual_run(pattern, planned_buffers)
        good_hbm_usage = self.hbm_usage_for_good_allocation(pattern, planned_buffers)
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

    def make_staircase_pattern(self) -> Pattern:
        """Allocate N*2 buffers of sizes k, k, 2*k, 2*k, 3*k, 3*k, ..., N*k, N*k. After an
        even-numbered buffer is allocated, free the previous odd-numbered buffer. This creates a
        "staircase" pattern of allocations that can only be fit if the allocator is smart about
        fragmentation. In that case, the maximum scratchpad usage is
        k + 2*k + ... + N*k + N*k = k * N * (N + 1) / 2 + N * k = k * N * (N + 3) / 2, so we choose
        k such that this is just less than the available scratchpad size.

        The greedy allocator will always allocate the next buffer just after all other buffers,
        because no gap is big enough for the current size. So it uses
        2 * (k + 2*k + ... + N*k) = k * N * (N + 1) or roughly 2 times more."""
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
                (f"A{i}", f"A{i}_HBM", f"A{i}"),
                (f"B{i}", f"A{i}", f"B{i}"),
                (f"C{i}", [f"A{i}", f"B{i}"], f"C{i}"),
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
            inplace=True,
        )
        return pattern

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
                *[
                    (f"op{i}_{j}_load", f"S{i}_HBM", group[j])
                    for j in range(len(group))
                ],
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
            input_buffer = []
            for buffer in group:
                input_buffer.append(
                    Allocation(buffer=buffer, address=addresses_per_group[i][buffer])
                )
                good_allocations.append(
                    [Allocation(buffer=f"S{i}_HBM", component=Component.HBM)]
                    + input_buffer
                )

            for j in range(4):
                good_allocations.append(
                    input_buffer
                    + [Allocation(buffer=f"C{i}_{j}", component=Component.HBM)]
                )

        pattern = Pattern(
            buffers,
            ops,
            good_allocation=make_general_allocation_result(good_allocations),
            inplace=True,
        )
        return pattern

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

    def make_moe_mlp_pattern(self) -> Pattern:
        """This pattern is a (simplified) mixture of experts multi-layer perceptron.

        We start with a hidden state, of dimension batch x seq_len x hidden_size. We matmul with a
        selector matrix (of dimension hidden_size x num_experts) to obtain qualities of dimension
        batch x seq_len x num_experts, indicating the quality of each expert for each token. We find
        the top k indices for each subtensor in the last dimension (an integer tensor of dimension
        batch x seq_len x k). We create a masked version of quality of the same dimension (replacing
        the non-selected entries by -inf). We apply softmax to this to get the weights - we pretend
        here that softmax is simply a pointwise operation.

        We now determine the tokens for each expert from the top k indices. This is an integer
        tensor of dimension num_experts x (batch * seq_len * 2), giving for each expert the batch
        and seq_len index for each token that they need to process; there are at most batch *
        seq_len such indices for any one expert.

        Now, for each expert, we collect the n_tokens x hidden_size matrix of selected_tokens from
        the hidden state (where n_tokens <= batch * seq_len), compute up = selected_tokens @ M1 +
        bias_1 and gate = selected_tokens @ M2 + bias_2 where M1 and M2 are (expert-dependent)
        matrices of dimension hidden_size x mlp_size and bias_1 and bias_2 are (expert-dependent)
        vectors of dimension mlp_size. We define out = up * silu(gate), and down = out @ M3 +
        bias_3, where M3 is an (expert-dependent) mlp_size x hidden_size matrix and bias_3 is an
        (expert-dependent) hidden_size vector. Now we add the entries of out to the appropriate
        entries of hidden state, scaled by the appropriate weight. After all experts are done, we
        save the hidden state to HBM.

        We assume that both integers and floats are 16 bits. This means that there are at most 65536
        experts and that batch_size and seq_len are at most 32768, assuming we want to use -1 as a
        special value in tokens_per_expert.
        """
        B = 4  # batch size
        S = 96  # sequence length
        H = 512  # hidden size
        N = 5  # number of experts
        K = 2  # number of experts *selected*
        M = 256  # "internal" size of the mlp

        T = [200, 150, 350, 25]  # selected tokens per expert
        T.append(B * S * K - sum(T))
        self.assertGreaterEqual(
            T[-1],
            0,
            "increase B * S * K, or reduce the token counts per expert",
        )
        assert N == len(T)

        # Abbreviations. The factor 2 is bytes per entry.
        BSH2 = B * S * H * 2
        HN2 = H * N * 2
        BSN2 = B * S * N * 2
        BSK2 = B * S * K * 2
        HM2 = H * M * 2
        H2 = H * 2
        M2 = M * 2

        buffers = make_buffer_registry(
            {
                "hidden_HBM": BSH2,
                "hidden": BSH2,
                "selector_HBM": HN2,
                "selector": HN2,
                "qualities": BSN2,
                "top_k_idx": BSK2,
                "masked_qualities": BSN2,
                "weights": BSN2,
                "tok_per_expert": BSN2 * 2,
                "result_HBM": BSH2,
            }
            | {
                f"{key}_{i}": value
                for i in range(N)
                for key, value in {
                    "selected_tokens": T[i] * H2,
                    "M1_HBM": HM2,
                    "bias1_HBM": M2,
                    "M2_HBM": HM2,
                    "bias2_HBM": M2,
                    "up": T[i] * M2,
                    "gate": T[i] * M2,
                    "out": T[i] * M2,
                    "M3_HBM": HM2,
                    "bias3_HBM": H2,
                    "down": T[i] * H2,
                }.items()
            }
        )

        ops = make_operations(
            [
                ("load_hidden", "hidden_HBM", "hidden"),
                ("load_selector", "selector_HBM", "selector"),
                ("matmul_0", ["hidden", "selector"], "qualities"),
                ("topk", "qualities", "top_k_idx"),
                ("mask_qualities", ["qualities", "top_k_idx"], "masked_qualities"),
                ("softmax", "masked_qualities", "weights"),
                ("select_tokens", "top_k_idx", "tok_per_expert"),
            ]
            + [
                (f"{tupl[0]}_{i}", tupl[1], tupl[2])
                for i in range(N)
                for tupl in [
                    ("gather", ["hidden", "tok_per_expert"], f"selected_tokens_{i}"),
                    (
                        "matmul_add_1",
                        [f"selected_tokens_{i}", f"M1_HBM_{i}", f"bias1_HBM_{i}"],
                        f"up_{i}",
                    ),
                    (
                        "matmul_add_2",
                        [f"selected_tokens_{i}", f"M2_HBM_{i}", f"bias2_HBM_{i}"],
                        f"gate_{i}",
                    ),
                    ("swiglu", [f"up_{i}", f"gate_{i}"], f"out_{i}"),
                    (
                        "matmul_add_3",
                        [f"out_{i}", f"M3_HBM_{i}", f"bias3_HBM_{i}"],
                        f"down_{i}",
                    ),
                    (
                        "scatter",
                        [f"down_{i}", "weights", "hidden", "tok_per_expert"],
                        "hidden",
                    ),
                ]
            ]
            + [("save_result", "hidden", "result_HBM")],
            buffers,
        )

        # The allocation works as follows, where every address and size is in 2-byte words.
        #
        # | buffer           | address                     | size   | on top of                  |
        # |------------------|-----------------------------|--------|----------------------------|
        # | hidden           | 0                           | BSH    | -                          |
        # | qualities        | BSH                         | BSN    | hidden                     |
        # | selector         | BSH+BSN                     | HN     | qualities                  |
        # | masked_qualities | BSH+BSN                     | BSN    | qualities                  |
        # | top_k_idx        | BSH+BSN+BSN*2               | BSK    | selector, masked_qualities |
        # | weights          | BSH                         | BSN    | hidden                     |
        # | tok_per_expert   | BSH+BSN                     | BSN*2  | weights                    |
        # | selected_tokens  | BSH+BSN+BSN*2               | T[i]*H | tok_per_expert             |
        # | up               | BSH+BSN+BSN*2+T[i]*H        | T[i]*M | selected_tokens            |
        # | gate             | BSH+BSN+BSN*2+T[i]*H+T[i]*M | T[i]*M | up                         |
        # | out              | BSH+BSN+BSN*2               | T[i]*M | tok_per_expert             |
        # | down             | BSH+BSN+BSN*2+T[i]*M        | T[i]*H | out                        |

        assert 2 * B * S >= H, (
            "this is an assumption in the allocation below, for ensuring that "
            "top_k_idx doesn't overlap with selector"
        )

        good_allocation = make_nonevicting_allocation_result(
            buffers,
            {
                "hidden": 0,
                "selector": BSH2 + BSN2,
                "qualities": BSH2,
                "top_k_idx": BSH2 + BSN2 * 3,
                "masked_qualities": BSH2 + BSN2,
                "weights": BSH2,
                "tok_per_expert": BSH2 + BSN2,
            }
            | {
                f"{key}_{i}": value
                for i in range(N)
                for key, value in {
                    "selected_tokens": BSH2 + BSN2 * 3,
                    "up": BSH2 + BSN2 * 3 + T[i] * H2,
                    "gate": BSH2 + BSN2 * 3 + T[i] * H2 + T[i] * M2,
                    "out": BSH2 + BSN2 * 3,
                    "down": BSH2 + BSN2 * 3 + T[i] * M2,
                }.items()
            },
            ops,
        )

        pattern = Pattern(buffers, ops, good_allocation=good_allocation, inplace=True)
        return pattern


class TestVerifyPatterns(PatternTests, TestCase, role="verify"):
    pass


class TestGreedyPatterns(PatternTests, TestCase, role="solver"):
    solver_type = GreedyLayoutSolver
    expected_failures: ClassVar[frozenset[str]] = frozenset(
        {
            "simple_fragmentation",
            "staircase",
            "downward_staircase",
            "simple_eviction",
            "eviction_reallocation",
        }
    )


class TestBestFitPatterns(PatternTests, TestCase, role="solver"):
    solver_type = BestFitLayoutSolver
    expected_failures: ClassVar[frozenset[str]] = frozenset(
        {"eviction_reallocation", "simple_eviction"}
    )


class TestFirstFitPatterns(PatternTests, TestCase, role="solver"):
    solver_type = FirstFitLayoutSolver
    expected_failures: ClassVar[frozenset[str]] = frozenset(
        {"eviction_reallocation", "simple_eviction"}
    )


class TestSimulatedAnnealingPatterns(PatternTests, TestCase, role="solver"):
    solver_type = SimulatedAnnealingLayoutSolver
    # Simulated annealing is a permutation-based layout solver: it assigns each buffer a
    # single fixed address for its whole lifetime and never evicts or reallocates.
    # The two eviction patterns require moving a buffer to HBM and back (possibly
    # at a different address), which this abstraction cannot express -- the same
    # limitation shared by best-fit and first-fit.
    expected_failures: ClassVar[frozenset[str]] = frozenset(
        {"eviction_reallocation", "simple_eviction"}
    )


if __name__ == "__main__":
    import unittest

    unittest.main()

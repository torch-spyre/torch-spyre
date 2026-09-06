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

"""Unit tests for dedup_and_promote_constants.

Three test classes, each installing a different
CustomPreSchedulingPasses hook:

  - TestDedupConstants -- runs the full pre-scheduling pipeline and
    inspects the resulting operations list. Structural checks after
    dedup completes.
  - TestDedupConstantsPassLevel -- stops the pipeline just before
    dedup, hands control to a per-test callback that constructs a
    deterministic condition and invokes dedup_and_promote_constants
    directly. Used for edge cases the compilation pipeline may not
    naturally produce.
  - TestBuildReverseConsumerIndex -- standalone unit tests for
    _build_reverse_consumer_index that use lightweight mocks and do
    not require the Spyre device.
"""

from typing import Any, Callable, Optional, TypeVarTuple, override

import unittest
from unittest.mock import patch

import torch
from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation
from torch._inductor.virtualized import V

from torch_spyre._C import get_elem_in_stick
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes
from torch_spyre._inductor.dedup_constants import dedup_and_promote_constants
from torch_spyre._inductor.ir import SpyreConstantFallback
from torch_spyre._inductor.pass_utils import NameSwapHandler
from torch_spyre._inductor.passes import CustomPreSchedulingPasses


Ts = TypeVarTuple("Ts")


# ---------------------------------------------------------------------------
# Sentinel + hook classes
# ---------------------------------------------------------------------------


class _TestStopSignal(Exception):
    """Raised by a pass-level test callback to terminate the surrounding
    ``torch.compile`` without also swallowing arbitrary downstream errors.

    Pass-level tests intercept the pipeline immediately before
    ``dedup_and_promote_constants``, mutate ``graph.operations`` into a
    deterministic condition, invoke dedup by hand, assert, then raise
    this sentinel to skip the remainder of pre-scheduling and
    subsequent codegen. Any other exception observed during the compile
    -- including ones raised inside the callback's own assertions -- is
    a real failure and propagates.
    """


class _CapturingPasses(CustomPreSchedulingPasses):
    """Hook that runs the full pre-scheduling pipeline and captures the
    resulting operations list on the test instance."""

    test_instance: Optional["TestDedupConstants"] = None

    @classmethod
    def initialize(cls, test_instance: "TestDedupConstants") -> None:
        cls.test_instance = test_instance

    @override
    def __call__(self, graph: GraphLowering) -> None:
        assert self.test_instance is not None
        super().__call__(graph)
        self.test_instance.captured_operations = list(graph.operations)


class _StopBeforeDedupPasses(CustomPreSchedulingPasses):
    """Hook that runs pre-scheduling passes up to (but not including)
    dedup_and_promote_constants, then hands the graph to a per-test
    callback.

    The callback is stored on the class as ``test_callback``. It is
    invoked with the ``GraphLowering`` inside ``V.set_graph_handler``,
    the same context every pass runs under. The callback drives the
    dedup pass itself (typically by calling
    ``dedup_and_promote_constants(graph)`` directly), asserts on the
    resulting state, and finishes by raising ``_TestStopSignal`` to
    stop the pipeline before downstream passes see the deliberately
    mutated graph.
    """

    test_callback: Optional[Callable[[GraphLowering], None]] = None

    @classmethod
    def install(cls, cb: Callable[[GraphLowering], None]) -> None:
        cls.test_callback = cb

    @override
    def __call__(self, graph: GraphLowering) -> None:
        # Import here so we don't shadow module-level names.
        from torch_spyre._inductor.passes import _operations_have_spyre_device

        if not _operations_have_spyre_device(graph.operations):
            return

        assert self.test_callback is not None, "test_callback not installed"

        # Find the dedup step's index in the pass list; run everything
        # strictly before it.
        pass_list = list(self.passes)
        try:
            dedup_idx = next(
                i
                for i, p in enumerate(pass_list)
                if getattr(p, "__name__", "") == "dedup_and_promote_constants"
            )
        except StopIteration:
            raise AssertionError("dedup_and_promote_constants missing from pipeline")

        # Run everything before dedup -- verbatim, no extra observers.
        for pass_fn in pass_list[:dedup_idx]:
            pass_fn(graph)

        # Hand off to the test. Access via type() so Python does not
        # bind self as the first positional arg to a function stored
        # as a class attribute. The callback is expected to raise
        # _TestStopSignal to terminate the compile after its assertions.
        cb = type(self).test_callback
        assert cb is not None
        cb(graph)


# ---------------------------------------------------------------------------
# Base class -- shared setUp/tearDown for both compile-driven test classes.
# ---------------------------------------------------------------------------


class _DedupTestBase(unittest.TestCase):
    """Shared config patches for the compile-driven dedup tests."""

    def setUp(self) -> None:
        torch.manual_seed(0xBEEF)
        self.patchers: list[Any] = []
        self.patchers.append(t_inductor_config.patch("force_disable_caches", True))
        self.patchers.append(ts_inductor_config.patch("sencores", 1))
        # Subclasses install their own CustomPreSchedulingPasses replacement.
        for p in self.patchers:
            p.__enter__()
        torch.compiler.reset()

    def tearDown(self) -> None:
        for p in self.patchers:
            p.__exit__(None, None, None)
        torch.compiler.reset()

    @staticmethod
    def _constants(ops: list[Operation]) -> list[SpyreConstantFallback]:
        return [op for op in ops if isinstance(op, SpyreConstantFallback)]

    @staticmethod
    def _non_constants(ops: list[Operation]) -> list[Operation]:
        return [op for op in ops if not isinstance(op, SpyreConstantFallback)]


# ===========================================================================
# TestDedupConstants -- structural tests over the full pipeline.
# ===========================================================================


class TestDedupConstants(_DedupTestBase):
    """Structural tests for dedup_and_promote_constants."""

    captured_operations: list[Operation] = []

    def setUp(self) -> None:
        super().setUp()
        _CapturingPasses.initialize(self)
        p = patch.object(passes, "CustomPreSchedulingPasses", _CapturingPasses)
        p.__enter__()
        self.patchers.append(p)  # so tearDown exits it

    def _compile(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
    ) -> list[Operation]:
        self.captured_operations = []
        torch.compile(fn, fullgraph=True)(*args)
        return self.captured_operations

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_constants_at_front(self) -> None:
        """Every SpyreConstantFallback precedes every non-constant op after dedup."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        # Unaligned K forces padding → constant creation.
        k = stick_size + 1
        x = torch.randn(4, k, dtype=dtype, device="spyre")
        w = torch.randn(k, 32, dtype=dtype, device="spyre")

        def fn(x, w):
            return torch.mm(x, w)

        ops = self._compile(fn, (x, w))
        constants = self._constants(ops)
        if not constants:
            self.skipTest("No constants produced — K aligned or no pad needed")
        non_constants = self._non_constants(ops)
        last_const_idx = max(ops.index(c) for c in constants)
        if non_constants:
            first_non_const_idx = min(ops.index(nc) for nc in non_constants)
            self.assertLess(
                last_const_idx,
                first_non_const_idx,
                "Some SpyreConstantFallback op appears after a non-constant op",
            )

    def test_dedup_across_same_dtype_pad_sequences(self) -> None:
        """Multiple pad sequences with the same fill_value and dtype yield one constant.

        Two bmm calls both pad x and y with fill=0.0 at float16, producing four
        SpyreConstantFallback nodes before dedup.  After dedup, exactly one survives.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1  # unaligned → forces padding on both matmuls
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        ops = self._compile(fn, (x, w1, w2))
        constants = self._constants(ops)
        self.assertEqual(
            len(constants),
            1,
            f"Expected 1 SpyreConstantFallback after dedup, got {len(constants)}",
        )

    def test_different_dtype_constants_not_merged(self) -> None:
        """Constants with the same scalar value but different dtypes are not merged.

        x (fp16) + 1.0 and y (fp32) + 1.0 each produce a spyre.constant with
        different dtype, so the dedup key differs and both constants survive.
        """
        x = torch.randn(4, 32, dtype=torch.float16, device="spyre")
        y = torch.randn(4, 32, dtype=torch.float32, device="spyre")

        def fn(x, y):
            # Both arithmetic nodes have scalar arg 1.0.
            # Constant materialization emits one py_const per consumer.
            return x + 1.0, y + 1.0

        ops = self._compile(fn, (x, y))
        constants = self._constants(ops)
        # Two distinct dtypes → two distinct constants must survive.
        self.assertEqual(
            len(constants),
            2,
            f"Expected 2 SpyreConstantFallback (one per dtype), got {len(constants)}",
        )

    def test_no_orphans_in_name_to_buffer(self) -> None:
        """After dedup, name_to_buffer contains no key for removed constants.

        When a duplicate is dropped, its entry in name_to_buffer must be cleaned
        up so that subsequent passes don't observe stale buffer references.
        """
        from torch._inductor.virtualized import V  # noqa: F401

        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        ops = self._compile(fn, (x, w1, w2))
        surviving_constant_names = {op.get_name() for op in self._constants(ops)}
        for op in ops:
            if isinstance(op, SpyreConstantFallback):
                self.assertIn(
                    op.get_name(),
                    surviving_constant_names,
                    f"Unexpected constant {op.get_name()} in operations",
                )

    def test_surviving_constant_at_index_zero(self) -> None:
        """After dedup, the first operation is a SpyreConstantFallback when any exist."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(4, k, dtype=dtype, device="spyre")
        w = torch.randn(k, 32, dtype=dtype, device="spyre")

        def fn(x, w):
            return torch.mm(x, w)

        ops = self._compile(fn, (x, w))
        constants = self._constants(ops)
        if not constants:
            self.skipTest("No constants produced")
        self.assertIsInstance(
            ops[0],
            SpyreConstantFallback,
            f"Expected operations[0] to be SpyreConstantFallback, got {type(ops[0]).__name__}",
        )


# ===========================================================================
# TestDedupConstantsPassLevel -- pass-level edge-case coverage.
# ===========================================================================


class TestDedupConstantsPassLevel(_DedupTestBase):
    """Deterministic pass-level tests.

    Each test runs the real pre-scheduling pipeline up to
    insert_bmm_padding, then hands the GraphLowering to a callback
    which:

      1. Optionally mutates state into the exact condition the test
         needs (drop consumers, install additional live reads, ...).
      2. Invokes ``dedup_and_promote_constants(graph)`` directly.
      3. Asserts on the resulting state.
      4. Raises ``_TestStopSignal`` to stop the compile before
         downstream passes see the deliberately mutated graph.

    Fixture-shape assertions in these tests fail loudly with
    ``PRECONDITION`` in the message when the pipeline no longer
    produces the required incidental state (e.g. no duplicate group).
    Those failures indicate the fixture no longer exercises the
    intended dedup condition, NOT a dedup semantic failure. Do not
    silence them with ``skipTest`` -- the loud failure is what keeps
    the coverage honest.
    """

    def setUp(self) -> None:
        super().setUp()
        p = patch.object(passes, "CustomPreSchedulingPasses", _StopBeforeDedupPasses)
        p.__enter__()
        self.patchers.append(p)

    def _drive(
        self,
        cb: Callable[[GraphLowering], None],
        fn: Callable[..., Any],
        args: tuple[Any, ...],
    ) -> None:
        """Install ``cb`` as the pass callback and run ``fn(*args)`` under
        ``torch.compile``. The callback must raise ``_TestStopSignal`` to
        stop the compile after its assertions. Any other exception is
        re-raised.
        """
        _StopBeforeDedupPasses.install(cb)

        @torch.compile(fullgraph=True)
        def compiled_fn(*a):
            return fn(*a)

        # Only _TestStopSignal is expected. Anything else -- including an
        # AssertionError from inside the callback -- surfaces normally.
        # torch.compile / dynamo will wrap the sentinel in an InductorError;
        # walk the __cause__ chain to detect our marker.
        try:
            compiled_fn(*args)
        except Exception as e:
            cur: Optional[BaseException] = e
            while cur is not None:
                if isinstance(cur, _TestStopSignal):
                    return
                cur = cur.__cause__ or cur.__context__
            raise

    # ------------------------------------------------------------------
    # test_zero_consumer_duplicate
    # ------------------------------------------------------------------

    def test_zero_consumer_duplicate(self) -> None:
        """A duplicate constant that has no live readers still gets
        cleanly removed and its bookkeeping cleaned.

        Deterministic construction: after the real pipeline runs up
        to dedup, artificially drop the live consumer(s) of one dup
        so it has zero readers going into dedup.

        Asserts:
          - duplicate op removed from operations
          - duplicate buffer name in removed_buffers
          - duplicate buffer name absent from name_to_buffer
          - duplicate operation name absent from name_to_op
          - duplicate buffer name absent from name_to_users
          - canonical survives
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        def cb(graph: GraphLowering) -> None:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants(graph.operations)
            self.assertGreaterEqual(
                len(constants),
                2,
                "PRECONDITION: workload did not produce a duplicate group "
                "before dedup; the pipeline shape changed and this test's "
                "fixture no longer exercises the zero-consumer case. Not a "
                "dedup failure.",
            )
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertTrue(
                multi,
                "PRECONDITION: no multi-constant group; the pipeline shape "
                "changed and this test's fixture no longer exercises the "
                "zero-consumer case. Not a dedup failure.",
            )
            group = multi[0]
            canonical, chosen_dup = group[0], group[1]
            D = chosen_dup.get_name()

            # Discover live consumers of chosen_dup and drop them so
            # chosen_dup has zero readers going into dedup.
            live_consumers = [
                op
                for op in graph.operations
                if op is not chosen_dup
                and op is not canonical
                and any(dep.name == D for dep in op.get_read_writes().reads)
            ]
            for op in live_consumers:
                graph.operations.remove(op)

            chosen_dup_op_name = chosen_dup.get_operation_name()

            dedup_and_promote_constants(graph)

            self.assertNotIn(
                chosen_dup,
                graph.operations,
                "chosen_dup should be removed from graph.operations",
            )
            self.assertIn(canonical, graph.operations, "canonical should survive")
            self.assertIn(
                D,
                graph.removed_buffers,
                f"chosen_dup buffer {D} should be in removed_buffers",
            )
            self.assertNotIn(
                D,
                graph.name_to_buffer,
                f"chosen_dup buffer {D} should be absent from name_to_buffer",
            )
            self.assertNotIn(
                chosen_dup_op_name,
                graph.name_to_op,
                f"chosen_dup op {chosen_dup_op_name} should be absent from name_to_op",
            )
            self.assertNotIn(
                D,
                graph.name_to_users,
                f"chosen_dup buffer {D} should be absent from name_to_users",
            )
            raise _TestStopSignal()

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        self._drive(cb, fn, (x, w1, w2))

    # ------------------------------------------------------------------
    # test_one_duplicate_many_consumers
    # ------------------------------------------------------------------

    def test_one_duplicate_many_consumers(self) -> None:
        """A single duplicate name D is read by two or more distinct
        live ComputedBuffers before dedup. Every one of them gets
        redirected to the canonical.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        def cb(graph: GraphLowering) -> None:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants(graph.operations)
            self.assertGreaterEqual(
                len(constants),
                2,
                "PRECONDITION: no duplicate group; fixture no longer "
                "exercises the multi-consumer condition. Not a dedup failure.",
            )
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertTrue(
                multi,
                "PRECONDITION: no multi-constant group. Not a dedup failure.",
            )
            group = multi[0]
            canonical, dup = group[0], group[1]
            C, D = canonical.get_name(), dup.get_name()

            reader_of_dup = next(
                (
                    op
                    for op in graph.operations
                    if op is not dup
                    and op is not canonical
                    and any(dep.name == D for dep in op.get_read_writes().reads)
                ),
                None,
            )
            reader_of_canonical = next(
                (
                    op
                    for op in graph.operations
                    if op is not dup
                    and op is not canonical
                    and any(dep.name == C for dep in op.get_read_writes().reads)
                ),
                None,
            )
            self.assertIsNotNone(
                reader_of_dup,
                "PRECONDITION: no live reader of dup found. Fixture shape changed.",
            )
            self.assertIsNotNone(
                reader_of_canonical,
                "PRECONDITION: no live reader of canonical found. Fixture "
                "shape changed.",
            )
            self.assertIsInstance(reader_of_canonical, ComputedBuffer)
            self.assertIsNot(reader_of_dup, reader_of_canonical)

            # Re-wire reader_of_canonical to also read D via a
            # NameSwapHandler({C: D}) wrapper -- the same mechanism the
            # pass uses.
            orig_inner = reader_of_canonical.data.inner_fn

            def _new_inner(*args, _map={C: D}, _orig=orig_inner):
                with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
                    return _orig(*args)

            object.__setattr__(reader_of_canonical.data, "inner_fn", _new_inner)
            ComputedBuffer.get_default_sizes_body.clear_cache(reader_of_canonical)

            self.assertIn(
                D, {dep.name for dep in reader_of_dup.get_read_writes().reads}
            )
            self.assertIn(
                D,
                {dep.name for dep in reader_of_canonical.get_read_writes().reads},
            )

            dedup_and_promote_constants(graph)

            reads_1 = {dep.name for dep in reader_of_dup.get_read_writes().reads}
            reads_2 = {dep.name for dep in reader_of_canonical.get_read_writes().reads}
            self.assertNotIn(D, reads_1, f"{reader_of_dup} still reads D={D}")
            self.assertNotIn(D, reads_2, f"{reader_of_canonical} still reads D={D}")
            self.assertIn(
                C,
                reads_1,
                f"{reader_of_dup} should now read canonical C={C}",
            )
            self.assertIn(
                C,
                reads_2,
                f"{reader_of_canonical} should now read canonical C={C}",
            )
            self.assertNotIn(dup, graph.operations)
            self.assertIn(canonical, graph.operations)
            raise _TestStopSignal()

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        self._drive(cb, fn, (x, w1, w2))

    # ------------------------------------------------------------------
    # test_one_consumer_reads_two_duplicates_same_group  (B2)
    # ------------------------------------------------------------------

    def test_one_consumer_reads_two_duplicates_same_group(self) -> None:
        """A single live ComputedBuffer reads BOTH duplicates in the
        same dedup group. Both collapse to the canonical.

        This is the behavioral case where the snapshot-based reverse
        index and a live-rescan implementation could plausibly diverge:
        after the first D1 -> C redirect wraps the consumer's inner_fn
        with a ``NameSwapHandler({D1: C})``, its live
        ``get_read_writes()`` no longer contains D1. A live-rescan
        implementation would still find and patch it for D2. The
        snapshot implementation used here has the consumer listed
        under both ``consumers_by_name[D1]`` and
        ``consumers_by_name[D2]`` from the pre-redirect scan, so it
        gets patched twice and the two ``NameSwapHandler`` layers
        compose (each translates its own key). After dedup both
        duplicate names must be absent from the consumer's live reads
        and only the canonical must remain.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        # Three unaligned bmms -> three padding constants (same fill
        # value, same dtype, same device) in the SAME dedup group.
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w3 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        def cb(graph: GraphLowering) -> None:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants(graph.operations)
            self.assertGreaterEqual(
                len(constants),
                3,
                "PRECONDITION: expected at least three padding constants in "
                "one dedup group (three bmms with unaligned K); got "
                f"{len(constants)}. Not a dedup failure -- fixture shape "
                "changed.",
            )
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) >= 3]
            self.assertTrue(
                multi,
                "PRECONDITION: no dedup group with >=3 constants. Not a dedup failure.",
            )
            group = multi[0]
            canonical = group[0]
            dup1, dup2 = group[1], group[2]
            C = canonical.get_name()
            D1 = dup1.get_name()
            D2 = dup2.get_name()

            # Pick a live ComputedBuffer that currently reads D1 (the
            # natural fill Pointwise for dup1) and rewrite its
            # ``inner_fn`` so it reads BOTH D1 and D2 via a
            # NameSwapHandler({placeholder: D2}). We source the extra
            # read by wrapping a load of the CANONICAL name and
            # translating it to D2 via NameSwapHandler; the effect is
            # that this op's live get_read_writes now reports {D1, D2}.
            #
            # We identify the target by:
            #   - it is a ComputedBuffer,
            #   - it is not any constant in the group,
            #   - its live reads currently contain D1.
            target = next(
                (
                    op
                    for op in graph.operations
                    if isinstance(op, ComputedBuffer)
                    and op is not canonical
                    and op is not dup1
                    and op is not dup2
                    and D1 in {dep.name for dep in op.get_read_writes().reads}
                ),
                None,
            )
            self.assertIsNotNone(
                target,
                "PRECONDITION: no live ComputedBuffer reader of dup1 found. "
                "Fixture shape changed.",
            )

            # Wrap target's inner_fn with a NameSwapHandler({C: D2}) so
            # that if it ever loads C it will end up loading D2. Then
            # further wrap it to explicitly load C once in addition to
            # its original reads. Composed together, target's live
            # reads now include both D1 (its original read) and D2
            # (introduced via C -> D2 swap).
            orig_inner = target.data.inner_fn

            def _new_inner(*args, _c=C, _d2=D2, _orig=orig_inner):
                # Wrap the original inner_fn with a NameSwapHandler
                # that rewrites any load of C to a load of D2.
                with V.set_ops_handler(NameSwapHandler(V.ops, {_c: _d2})):
                    result = _orig(*args)
                    # Additionally emit a load of C -- which the swap
                    # handler will redirect to D2 -- so the op's live
                    # reads include D2 in addition to whatever _orig
                    # emitted.
                    _ = V.ops.load(_c, args[0][0])
                return result

            object.__setattr__(target.data, "inner_fn", _new_inner)
            ComputedBuffer.get_default_sizes_body.clear_cache(target)

            reads_before = {dep.name for dep in target.get_read_writes().reads}
            self.assertIn(
                D1,
                reads_before,
                f"PRECONDITION: target should still read D1={D1} before dedup; "
                f"live reads = {sorted(reads_before)}. Not a dedup failure.",
            )
            self.assertIn(
                D2,
                reads_before,
                f"PRECONDITION: target's synthetic wrapping should have "
                f"introduced D2={D2} into its live reads; live reads = "
                f"{sorted(reads_before)}. Not a dedup failure.",
            )

            dedup_and_promote_constants(graph)

            reads_after = {dep.name for dep in target.get_read_writes().reads}
            self.assertNotIn(
                D1,
                reads_after,
                f"target should not read D1={D1} after dedup; got "
                f"{sorted(reads_after)}",
            )
            self.assertNotIn(
                D2,
                reads_after,
                f"target should not read D2={D2} after dedup; got "
                f"{sorted(reads_after)}",
            )
            self.assertIn(
                C,
                reads_after,
                f"target should read canonical C={C} after dedup; got "
                f"{sorted(reads_after)}",
            )
            self.assertNotIn(dup1, graph.operations)
            self.assertNotIn(dup2, graph.operations)
            self.assertIn(canonical, graph.operations)
            raise _TestStopSignal()

        def fn(x, w1, w2, w3):
            return torch.bmm(x, w1) + torch.bmm(x, w2) + torch.bmm(x, w3)

        self._drive(cb, fn, (x, w1, w2, w3))

    # ------------------------------------------------------------------
    # test_name_to_users_fold_exact
    # ------------------------------------------------------------------

    def test_name_to_users_fold_exact(self) -> None:
        """After dedup, name_to_users[canonical] is exactly the
        identity-preserving concatenation of pre-dedup canonical
        entries plus each duplicate's pre-dedup entries. Each
        duplicate key is absent from post-dedup name_to_users.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        def cb(graph: GraphLowering) -> None:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants(graph.operations)
            self.assertGreaterEqual(
                len(constants),
                2,
                "PRECONDITION: no duplicate group produced. Not a dedup failure.",
            )
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertTrue(multi, "PRECONDITION: no multi-constant group")
            group = multi[0]
            canonical = group[0]
            dups = group[1:]
            C = canonical.get_name()
            dup_names = [d.get_name() for d in dups]

            pre_C = list(graph.name_to_users.get(C, []))
            pre_D_entries: dict[str, list] = {
                D: list(graph.name_to_users.get(D, [])) for D in dup_names
            }
            expected_C_after = pre_C + [
                entry for D in dup_names for entry in pre_D_entries[D]
            ]

            dedup_and_promote_constants(graph)

            post_C = list(graph.name_to_users.get(C, []))
            self.assertEqual(
                [id(x) for x in post_C],
                [id(x) for x in expected_C_after],
                f"name_to_users[{C!r}] after dedup is not the exact "
                f"identity-preserving concatenation of pre-dedup canonical "
                f"+ duplicate entries",
            )
            for D in dup_names:
                self.assertNotIn(
                    D,
                    graph.name_to_users,
                    f"name_to_users still has key for duplicate {D!r}",
                )
            raise _TestStopSignal()

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        self._drive(cb, fn, (x, w1, w2))

    # ------------------------------------------------------------------
    # test_provenance_transform_appended
    # ------------------------------------------------------------------

    def test_provenance_transform_appended(self) -> None:
        """merge_provenance appends exactly one ProvenanceTransform to
        the canonical constant with pass_name
        ``dedup_and_promote_constants`` per absorbed duplicate.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        def cb(graph: GraphLowering) -> None:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants(graph.operations)
            self.assertGreaterEqual(
                len(constants),
                2,
                "PRECONDITION: no duplicate group produced. Not a dedup failure.",
            )
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertTrue(multi, "PRECONDITION: no multi-constant group")
            canonical = multi[0][0]
            n_dups_in_group = len(multi[0]) - 1

            pre_history_len = len(getattr(canonical, "_spyre_prov_history", ()) or ())
            dedup_and_promote_constants(graph)
            post_history = getattr(canonical, "_spyre_prov_history", ())
            self.assertIsNotNone(
                post_history,
                "canonical should carry _spyre_prov_history after merge_provenance",
            )
            new_entries = post_history[pre_history_len:]
            dedup_transforms = [
                t
                for t in new_entries
                if getattr(t, "pass_name", "") == "dedup_and_promote_constants"
            ]
            self.assertEqual(
                len(dedup_transforms),
                n_dups_in_group,
                f"expected {n_dups_in_group} new dedup ProvenanceTransform(s), "
                f"got {len(dedup_transforms)}",
            )
            for t in dedup_transforms:
                self.assertEqual(getattr(t, "kind", None), "fusion")
                self.assertEqual(getattr(t, "reason", None), "duplicate constant")
            raise _TestStopSignal()

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        self._drive(cb, fn, (x, w1, w2))

    # ------------------------------------------------------------------
    # test_no_duplicates_fast_path
    # ------------------------------------------------------------------

    def test_no_duplicates_fast_path(self) -> None:
        """When there are no duplicate groups, dedup makes ZERO
        ComputedBuffer.get_read_writes calls. Guards the fast-path
        invariant the reverse-index optimization must preserve.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        # Aligned K -> no padding constant is generated.
        k_aligned = stick_size * 2
        x = torch.randn(2, 8, k_aligned, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k_aligned, 32, dtype=dtype, device="spyre")

        counter = {"n": 0}
        orig_grw = ComputedBuffer.get_read_writes

        def counted_grw(self):
            counter["n"] += 1
            return orig_grw(self)

        def cb(graph: GraphLowering) -> None:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants(graph.operations)
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertFalse(
                multi,
                "PRECONDITION: expected zero multi-constant groups (aligned "
                f"K workload); found {len(multi)}. Not a dedup failure -- "
                "fixture shape changed.",
            )

            with patch.object(ComputedBuffer, "get_read_writes", counted_grw):
                counter["n"] = 0
                dedup_and_promote_constants(graph)
                calls = counter["n"]

            self.assertEqual(
                calls,
                0,
                f"no-duplicate fast path violated: dedup made {calls} "
                "ComputedBuffer.get_read_writes calls when there were no "
                "duplicate groups",
            )
            raise _TestStopSignal()

        def fn(x, w1):
            return torch.bmm(x, w1)

        self._drive(cb, fn, (x, w1))

    # ------------------------------------------------------------------
    # test_reverse_index_scales_with_N_not_D  (B1 regression guard)
    # ------------------------------------------------------------------

    def test_reverse_index_scales_with_N_not_D(self) -> None:
        """The optimization guard: on a duplicate-bearing graph,
        ``ComputedBuffer.get_read_writes`` is called exactly once per
        candidate op during dedup -- not D times.

        This is the invariant the PR turns from an O(N*D) scan into an
        O(N) single sweep. If a regression re-introduced a per-duplicate
        rebuild of the reverse index, this test would observe a call
        count of roughly N*D and fail. The call count MUST be equal to
        the number of non-``SpyreConstantFallback``/non-canonical
        candidate ops seen during the single sweep, not scaled by D.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        # Three unaligned bmms → duplicate group of size 3 → D=2.
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w3 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        counter = {"n": 0}
        orig_grw = ComputedBuffer.get_read_writes

        def counted_grw(self):
            counter["n"] += 1
            return orig_grw(self)

        def cb(graph: GraphLowering) -> None:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants(graph.operations)
            self.assertGreaterEqual(
                len(constants),
                3,
                "PRECONDITION: three unaligned bmms should have produced at "
                f"least three padding constants; got {len(constants)}. Not "
                "a dedup failure -- fixture shape changed.",
            )
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) >= 3]
            self.assertTrue(
                multi,
                "PRECONDITION: no dedup group with >=3 constants (D>=2). "
                "Not a dedup failure.",
            )
            n_ops_at_entry = len(graph.operations)

            with patch.object(ComputedBuffer, "get_read_writes", counted_grw):
                counter["n"] = 0
                dedup_and_promote_constants(graph)
                calls = counter["n"]

            # In the single-sweep implementation, each ComputedBuffer in
            # graph.operations at pass entry is visited exactly once,
            # so the call count is at most n_ops_at_entry.
            #
            # A regression that rebuilds the reverse index inside the
            # per-duplicate loop would call get_read_writes on each op
            # once per duplicate; with D >= 2 in the largest group the
            # call count would be at least 2 * (number of ComputedBuffer
            # candidates) > n_ops_at_entry.
            #
            # The tight invariant this guard enforces:
            #
            #     get_read_writes calls <= n_ops_at_entry.
            #
            # Any per-duplicate rebuild would blow past this bound.
            self.assertLessEqual(
                calls,
                n_ops_at_entry,
                f"regression guard: dedup called "
                f"ComputedBuffer.get_read_writes {calls} times on a graph "
                f"with {n_ops_at_entry} ops at pass entry. A single-sweep "
                "reverse-index build should make at most one call per op. "
                "A count materially larger than N suggests the index is "
                "being rebuilt per duplicate (regression).",
            )
            # Also assert a lower bound so a broken impl that made zero
            # get_read_writes calls (and never built the index) does not
            # silently pass. Duplicates exist, so the index build must
            # have run.
            self.assertGreater(
                calls,
                0,
                "regression guard: duplicates exist but dedup made zero "
                "get_read_writes calls -- the reverse index was never "
                "built.",
            )
            raise _TestStopSignal()

        def fn(x, w1, w2, w3):
            return torch.bmm(x, w1) + torch.bmm(x, w2) + torch.bmm(x, w3)

        self._drive(cb, fn, (x, w1, w2, w3))

    # ------------------------------------------------------------------
    # test_all_output_name_duplicates_still_dropped
    # ------------------------------------------------------------------

    def test_all_output_name_duplicates_still_dropped(self) -> None:
        """When every duplicate is a graph output, Step 2 must still
        execute ``_drop_constant`` on those duplicates. Only the
        consumer-index scope is filtered by output names; the "does
        Step 2 run at all" gate is separate.

        This is the regression case Will identified in his second
        review: the output-name filter was correctly applied to
        ``duplicate_names`` (the reverse-index scope) but the same set
        was also used to gate whether Step 2 ran. When every duplicate
        was a graph output the set was empty, Step 2 was skipped
        entirely, and pristine's ``_drop_constant`` never ran --
        contradicting the pristine semantics where
        ``_redirect_consumers`` skipped the redirect for output-name
        duplicates but ``_drop_constant`` still cleaned them up.

        Construction: after ``insert_bmm_padding`` produces at least
        two padding constants in one dedup group, monkey-patch
        ``graph.get_output_names`` to return the set of duplicate
        buffer names in that group. That marks every duplicate as
        "output-name" for the pass, so the reverse-index scope
        collapses to the empty set. Assertions:

          - dedup_and_promote_constants makes ZERO
            ComputedBuffer.get_read_writes calls (the index scan is
            skipped because the scope is empty and there are no
            other non-output duplicates)
          - every duplicate op is removed from graph.operations
          - every duplicate buffer name is in removed_buffers
          - every duplicate buffer name is absent from name_to_buffer
          - every duplicate operation name is absent from name_to_op
          - canonical survives

        This test specifically fails on any implementation that gates
        Step 2 on the same output-filtered set used for index scope.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        k = stick_size + 1
        x = torch.randn(2, 8, k, dtype=dtype, device="spyre")
        w1 = torch.randn(2, k, 32, dtype=dtype, device="spyre")
        w2 = torch.randn(2, k, 32, dtype=dtype, device="spyre")

        counter = {"n": 0}
        orig_grw = ComputedBuffer.get_read_writes

        def counted_grw(self):
            counter["n"] += 1
            return orig_grw(self)

        def cb(graph: GraphLowering) -> None:
            from torch_spyre._inductor.dedup_constants import _constant_key

            constants = self._constants(graph.operations)
            self.assertGreaterEqual(
                len(constants),
                2,
                "PRECONDITION: no duplicate group produced. Not a dedup failure.",
            )
            groups: dict[tuple, list[SpyreConstantFallback]] = {}
            for c in constants:
                groups.setdefault(_constant_key(c), []).append(c)
            multi = [g for g in groups.values() if len(g) > 1]
            self.assertTrue(
                multi,
                "PRECONDITION: no multi-constant group. Not a dedup failure.",
            )
            group = multi[0]
            canonical = group[0]
            dups = group[1:]
            dup_names = [d.get_name() for d in dups]
            dup_op_names = [d.get_operation_name() for d in dups]

            # Snapshot pre-dedup state that the assertions consume.
            self.assertIn(canonical, graph.operations)
            for d in dups:
                self.assertIn(d, graph.operations)

            # Mark every duplicate name in the group as a graph output.
            # This collapses the reverse-index scope to empty while
            # preserving has_duplicates=True, which is exactly the
            # condition that triggered the regression.
            orig_get_output_names = graph.get_output_names
            output_set = set(dup_names)

            def patched_get_output_names(_o=orig_get_output_names, _s=output_set):
                return _s | set(_o())

            # patch on the instance for the duration of the pass.
            object.__setattr__(graph, "get_output_names", patched_get_output_names)

            try:
                with patch.object(ComputedBuffer, "get_read_writes", counted_grw):
                    counter["n"] = 0
                    dedup_and_promote_constants(graph)
                    calls = counter["n"]
            finally:
                # Restore -- best-effort; the test raises _TestStopSignal
                # right after this so the graph will not be used.
                try:
                    del graph.get_output_names  # type: ignore[misc]
                except Exception:
                    pass

            # Reverse-index scope is empty (all dup names filtered out),
            # so no candidate ops are scanned via get_read_writes.
            self.assertEqual(
                calls,
                0,
                f"regression guard: when every duplicate name is a graph "
                f"output the reverse-index scope should be empty and no "
                f"ComputedBuffer.get_read_writes calls should be made from "
                f"the index build; got {calls} calls.",
            )

            # ``_drop_constant`` must still have run on every duplicate.
            # This is exactly the assertion the v2 regression violated.
            for d, dname, opname in zip(dups, dup_names, dup_op_names):
                self.assertNotIn(
                    d,
                    graph.operations,
                    f"regression guard: duplicate op for buffer {dname} "
                    "should be removed from graph.operations even when it "
                    "is a graph output (_drop_constant must still run).",
                )
                self.assertIn(
                    dname,
                    graph.removed_buffers,
                    f"regression guard: duplicate buffer name {dname} "
                    "should be in removed_buffers even when it is a graph "
                    "output.",
                )
                self.assertNotIn(
                    dname,
                    graph.name_to_buffer,
                    f"regression guard: duplicate buffer name {dname} "
                    "should be absent from name_to_buffer.",
                )
                self.assertNotIn(
                    opname,
                    graph.name_to_op,
                    f"regression guard: duplicate op name {opname} should "
                    "be absent from name_to_op.",
                )
            self.assertIn(
                canonical,
                graph.operations,
                "canonical should survive dedup on the all-outputs path",
            )
            raise _TestStopSignal()

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        self._drive(cb, fn, (x, w1, w2))


# ===========================================================================
# TestBuildReverseConsumerIndex -- standalone unit tests over
# _build_reverse_consumer_index. No Spyre device required.
# ===========================================================================


class TestBuildReverseConsumerIndex(unittest.TestCase):
    """Guardrail for the E-only reverse-index construction.

    The pristine ``_redirect_consumers`` patches a matching op at most
    once per duplicate constant, regardless of how many separate
    dependency objects in ``op.get_read_writes().reads`` happen to
    share the same buffer name. The E-only index must preserve that:
    each op appears at most once in ``consumers_by_name[name]``.
    """

    def _fake_dep(self, name: str) -> Any:
        """A minimal Dep-like object with just ``.name``."""
        from types import SimpleNamespace

        return SimpleNamespace(name=name)

    def _fake_op(self, deps: list[Any]) -> Any:
        """A minimal Operation-like object whose ``get_read_writes`` returns
        an object with ``.reads == deps``.
        """
        from types import SimpleNamespace

        rw = SimpleNamespace(reads=deps)
        return SimpleNamespace(get_read_writes=lambda: rw)

    def test_op_with_two_deps_same_name_appears_once(self) -> None:
        """An op whose reads contain two distinct dep objects with the
        same name D appears exactly once in ``consumers_by_name[D]``."""
        from torch_spyre._inductor.dedup_constants import (
            _build_reverse_consumer_index,
        )

        op = self._fake_op([self._fake_dep("bufD"), self._fake_dep("bufD")])
        idx = _build_reverse_consumer_index([op], {"bufD"})
        self.assertEqual(len(idx["bufD"]), 1)
        self.assertIs(idx["bufD"][0], op)

    def test_op_with_two_deps_different_names(self) -> None:
        """An op whose reads contain two distinct duplicate names D1
        and D2 appears once in each of ``consumers_by_name[D1]`` and
        ``consumers_by_name[D2]``."""
        from torch_spyre._inductor.dedup_constants import (
            _build_reverse_consumer_index,
        )

        op = self._fake_op([self._fake_dep("bufD1"), self._fake_dep("bufD2")])
        idx = _build_reverse_consumer_index([op], {"bufD1", "bufD2"})
        self.assertEqual(len(idx["bufD1"]), 1)
        self.assertEqual(len(idx["bufD2"]), 1)
        self.assertIs(idx["bufD1"][0], op)
        self.assertIs(idx["bufD2"][0], op)

    def test_op_with_no_duplicate_reads_absent_from_index(self) -> None:
        """An op that reads only non-duplicate names does not appear
        in the index at all."""
        from torch_spyre._inductor.dedup_constants import (
            _build_reverse_consumer_index,
        )

        op = self._fake_op([self._fake_dep("bufX"), self._fake_dep("bufY")])
        idx = _build_reverse_consumer_index([op], {"bufD"})
        self.assertNotIn("bufD", idx)
        self.assertNotIn("bufX", idx)
        self.assertNotIn("bufY", idx)

    def test_multiple_ops_deterministic_order(self) -> None:
        """Ops appear in ``graph.operations`` order, preserving
        determinism for later passes and for debugging."""
        from torch_spyre._inductor.dedup_constants import (
            _build_reverse_consumer_index,
        )

        op1 = self._fake_op([self._fake_dep("bufD")])
        op2 = self._fake_op([self._fake_dep("bufD"), self._fake_dep("bufD")])
        op3 = self._fake_op([self._fake_dep("bufOther")])
        op4 = self._fake_op([self._fake_dep("bufD")])
        idx = _build_reverse_consumer_index([op1, op2, op3, op4], {"bufD"})
        self.assertEqual(idx["bufD"], [op1, op2, op4])

    def test_returned_mapping_is_plain_dict_not_defaultdict(self) -> None:
        """The reverse index returned by ``_build_reverse_consumer_index``
        must be a plain ``dict`` -- not a ``defaultdict(list)`` -- so
        that a lookup of an absent key does not silently insert an
        empty list into the mapping.
        """
        from collections import defaultdict

        from torch_spyre._inductor.dedup_constants import (
            _build_reverse_consumer_index,
        )

        op = self._fake_op([self._fake_dep("bufD")])
        idx = _build_reverse_consumer_index([op], {"bufD"})
        self.assertNotIsInstance(
            idx,
            defaultdict,
            "_build_reverse_consumer_index leaked its defaultdict backing "
            "store; return a plain dict so absent-key lookups do not mutate "
            "the mapping.",
        )
        # Prove the plain-dict semantics: accessing an absent key via [] must
        # raise KeyError, and .get() must NOT install a new key.
        with self.assertRaises(KeyError):
            _ = idx["absent"]
        _ = idx.get("absent")
        self.assertNotIn("absent", idx)


if __name__ == "__main__":
    unittest.main()

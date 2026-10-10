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


"""Tests for the frontend analysis-call counters."""

from typing import cast
from unittest.mock import patch

import pytest
from torch._inductor.ir import ComputedBuffer, Operation

from torch_spyre._inductor import pass_counters
from torch_spyre._inductor.pass_counters import (
    COUNTERS,
    READ_WRITES_EXTRACTIONS,
    READ_WRITES_MISSES,
    READ_WRITES_REQUESTS,
    PassCounters,
)
from torch_spyre._inductor.pass_utils import op_read_writes


@pytest.fixture(autouse=True)
def _clean_counters():
    """The store is process-wide; keep counts from leaking between tests."""
    COUNTERS.enabled = False
    COUNTERS.reset()
    yield
    COUNTERS.enabled = False
    COUNTERS.reset()


class _FakeOp:
    """Enough of an Operation for op_read_writes: a __dict__ and the method."""

    def __init__(self) -> None:
        self.extractions = 0

    def get_read_writes(self) -> str:
        self.extractions += 1
        return f"rw{self.extractions}"


class TestStore:
    """The counter store, independent of what is installed."""

    def test_bump_is_a_noop_while_disabled(self) -> None:
        counters = PassCounters()
        counters.bump("x", 5)
        assert counters.counts == {}

    def test_bump_accumulates_while_enabled(self) -> None:
        counters = PassCounters()
        counters.enabled = True
        counters.bump("x")
        counters.bump("x", 4)
        counters.bump("y")
        assert counters.counts == {"x": 5, "y": 1}

    def test_since_omits_counters_that_did_not_move(self) -> None:
        counters = PassCounters()
        counters.enabled = True
        counters.bump("x", 2)
        before = counters.snapshot()
        counters.bump("y", 3)
        # "x" is in the store but unchanged, so it must not appear -- otherwise
        # every pass event carries a key for every counter in the process.
        assert counters.since(before) == {"y": 3}

    def test_snapshot_and_since_are_empty_while_disabled(self) -> None:
        counters = PassCounters()
        counters.counts = {"x": 7}
        assert counters.snapshot() == {}
        assert counters.since({}) == {}

    def test_deltas_nest(self) -> None:
        """An inner delta is a subset of the outer one, like inclusive_ns."""
        counters = PassCounters()
        counters.enabled = True
        outer = counters.snapshot()
        counters.bump("x", 1)
        inner = counters.snapshot()
        counters.bump("x", 2)
        assert counters.since(inner) == {"x": 2}
        assert counters.since(outer) == {"x": 3}


class TestCountingRegion:
    """Installing and restoring the wrapper on upstream's method."""

    def test_install_and_restore(self) -> None:
        original = ComputedBuffer.get_read_writes
        with pass_counters.counting():
            assert ComputedBuffer.get_read_writes is not original
            assert COUNTERS.enabled
        assert ComputedBuffer.get_read_writes is original
        assert not COUNTERS.enabled

    def test_restores_after_an_exception(self) -> None:
        original = ComputedBuffer.get_read_writes
        with pytest.raises(RuntimeError):
            with pass_counters.counting():
                raise RuntimeError("boom")
        assert ComputedBuffer.get_read_writes is original
        assert not COUNTERS.enabled

    def test_disabled_installs_nothing(self) -> None:
        original = ComputedBuffer.get_read_writes
        with pass_counters.counting(install=False):
            assert ComputedBuffer.get_read_writes is original
            assert not COUNTERS.enabled
            COUNTERS.bump(READ_WRITES_REQUESTS)
        assert COUNTERS.counts == {}

    def test_counts_extractions_through_the_wrapper(self) -> None:
        # Stub the real method out: the point is that the wrapper counts and
        # delegates, not what dependency extraction returns.
        with patch.object(ComputedBuffer, "get_read_writes", lambda self: "rw"):
            with pass_counters.counting():
                assert ComputedBuffer.get_read_writes(object()) == "rw"
                ComputedBuffer.get_read_writes(object())
                assert COUNTERS.counts[READ_WRITES_EXTRACTIONS] == 2

    def test_nesting_does_not_double_count(self) -> None:
        with patch.object(ComputedBuffer, "get_read_writes", lambda self: "rw"):
            with pass_counters.counting():
                installed = ComputedBuffer.get_read_writes
                with pass_counters.counting():
                    # The inner region must leave the wrapper alone; wrapping a
                    # wrapper would count every call once per nesting level.
                    assert ComputedBuffer.get_read_writes is installed
                    ComputedBuffer.get_read_writes(object())
                assert COUNTERS.counts[READ_WRITES_EXTRACTIONS] == 1
                # ...and the inner exit must not have uninstalled it.
                assert COUNTERS.enabled
                assert ComputedBuffer.get_read_writes is installed

    def test_the_outermost_region_clears_earlier_totals(self) -> None:
        COUNTERS.enabled = True
        COUNTERS.bump(READ_WRITES_REQUESTS, 99)
        COUNTERS.enabled = False
        with pass_counters.counting():
            assert COUNTERS.counts == {}

    def test_counts_do_not_outlive_the_region(self) -> None:
        # Counts exist only inside a region, so a disabled store is
        # unambiguously empty rather than holding the last compile's totals.
        with pass_counters.counting():
            COUNTERS.bump(READ_WRITES_REQUESTS, 3)
        assert COUNTERS.counts == {}


class TestReadWritesCounters:
    """op_read_writes reports requests and memo misses separately."""

    def test_requests_count_every_call_misses_only_the_first(self) -> None:
        op = _FakeOp()
        with pass_counters.counting():
            assert op_read_writes(cast(Operation, op)) == "rw1"
            for _ in range(4):
                op_read_writes(cast(Operation, op))
            # Five asks, one extraction: the gap is what the memo absorbed, and
            # is the difference between a rescan and real work.
            assert COUNTERS.counts[READ_WRITES_REQUESTS] == 5
            assert COUNTERS.counts[READ_WRITES_MISSES] == 1
        assert op.extractions == 1

    def test_each_op_misses_once(self) -> None:
        ops = [_FakeOp() for _ in range(3)]
        with pass_counters.counting():
            for op in ops:
                op_read_writes(cast(Operation, op))
                op_read_writes(cast(Operation, op))
            assert COUNTERS.counts[READ_WRITES_REQUESTS] == 6
            assert COUNTERS.counts[READ_WRITES_MISSES] == 3

    def test_nothing_is_counted_while_disabled(self) -> None:
        op = _FakeOp()
        op_read_writes(cast(Operation, op))
        assert COUNTERS.counts == {}

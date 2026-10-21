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

from types import SimpleNamespace
from unittest import TestCase
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
    CpSatLayoutSolver,
    _Handoff,
    _PlacementUnit,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
    CoreDivision,
    RelayoutCopyBuffer,
    relayout_copy_name,
)


def _unit(name, start, end, footprint, offset, members=None):
    return _PlacementUnit(
        members=members or [name],
        footprint=footprint,
        start_time=start,
        end_time=end,
        original_offset=offset,
    )


def _bound_at(units, offsets, tick, handoffs=()):
    """The bound the program at ``tick`` states on the final graph: the highest"""
    dead = {(h.source, h.consumer_tick) for h in handoffs}
    return max(
        (
            offsets[u.members[0]] + footprint
            for u in units
            for name, start, end, footprint in u.spans()
            if start <= tick < end and (name, tick) not in dead
        ),
        default=0,
    )


def _qkv_units():
    """The measured shape, in 128 KiB units (Granite 3.3 8B at TP1, the fused"""
    return [
        _unit("residual", 3, 34, 1, 9),
        _unit("convert", 9, 11, 1, 0),
        _unit("norm_out", 10, 12, 1, 1),
        _unit("qkv_input", 11, 12, 8, 2),
    ]


_QKV_HANDOFF = _Handoff(source="norm_out", copy="qkv_input", consumer_tick=11)


class JustifyGivesMatmulsTheRoomOfADeadSourceTest(TestCase):
    def test_the_copy_goes_below_its_dying_source_at_a_matmul(self):
        """Longest-first packs the norm's outputs first, so the 1 MiB input sat"""
        units = _qkv_units()
        plain = CpSatLayoutSolver._justify(units, 12)
        room = CpSatLayoutSolver._justify(units, 12, frozenset({11}), [_QKV_HANDOFF])

        self.assertEqual(_bound_at(units, plain, 11, [_QKV_HANDOFF]), 11)
        self.assertEqual(_bound_at(units, room, 11, [_QKV_HANDOFF]), 1 + 8)
        self.assertLess(room["qkv_input"], room["norm_out"])
        # The hole moves to the norm's last elementwise program (tick 10),
        # which reads each input once whatever its room.
        self.assertEqual(_bound_at(units, plain, 10), 3)
        self.assertEqual(_bound_at(units, room, 10), 10)

    def test_the_gain_is_read_with_the_source_dead_at_the_consumer(self):
        """A source as large as the hole it leaves: counted live at the Q/K/V"""
        units = [
            _unit("residual", 3, 34, 1, 9),
            _unit("norm_out", 10, 12, 2, 1),
            _unit("qkv_input", 11, 12, 8, 3),
        ]
        plain = CpSatLayoutSolver._justify(units, 12)
        room = CpSatLayoutSolver._justify(units, 12, frozenset({11}), [_QKV_HANDOFF])
        self.assertEqual(_bound_at(units, plain, 11, [_QKV_HANDOFF]), 11)
        self.assertEqual(_bound_at(units, room, 11, [_QKV_HANDOFF]), 9)
        self.assertEqual(_bound_at(units, room, 11), 11)

    def test_a_move_that_buys_no_matmul_room_is_not_made(self):
        """With the only matmul elsewhere (tick 30, where just the residual is"""
        units = _qkv_units()
        self.assertEqual(
            CpSatLayoutSolver._justify(units, 12, frozenset({30}), [_QKV_HANDOFF]),
            CpSatLayoutSolver._justify(units, 12),
        )

    def test_the_last_packing_tried_is_the_one_returned(self):
        units = _qkv_units()
        room = CpSatLayoutSolver._justify(units, 12, frozenset({11}), [_QKV_HANDOFF])
        for u in units:
            self.assertEqual(u.justified_offset, room[u.members[0]])

    def test_without_matmul_ticks_the_packing_is_longest_first(self):
        units = _qkv_units()
        self.assertEqual(
            CpSatLayoutSolver._justify(units, 12, frozenset(), [_QKV_HANDOFF]),
            CpSatLayoutSolver._justify(units, 12),
        )

    def test_a_matmul_that_writes_the_source_keeps_it_low(self):
        """The O projection (a matmul, tick 5) writes the source and a pointwise"""
        units = [
            _unit("residual", 0, 20, 1, 0),
            _unit("o_out", 5, 7, 2, 1),
            _unit("o_copy", 6, 7, 2, 3),
        ]
        handoff = [_Handoff(source="o_out", copy="o_copy", consumer_tick=6)]
        plain = CpSatLayoutSolver._justify(units, 8)
        for matmuls in ({5}, {5, 6}):
            with self.subTest(matmuls=matmuls):
                self.assertEqual(
                    CpSatLayoutSolver._justify(units, 8, frozenset(matmuls), handoff),
                    plain,
                )
        self.assertLess(plain["o_out"], plain["o_copy"])

    def test_only_a_copy_whose_source_ends_on_its_one_tick_is_a_hand_off(self):
        """The copy op is inserted before the copy's first consumer. The source"""

        def relayout(parent, uses):
            return RelayoutCopyBuffer(
                name=relayout_copy_name(parent, 0),
                size=64,
                uses=list(uses),
                core_divisions=[CoreDivision(splits={"relayout_copy": 1})],
                relayout_parent=parent,
                group=0,
                candidates=(SimpleNamespace(consumer="consumer"),),
            )

        def handoffs(source_uses, copy_uses, spilled=()):
            source = LifetimeBoundBuffer("src", 64, list(source_uses))
            copy = relayout("src", copy_uses)
            return CpSatLayoutSolver._relayout_handoffs(
                {"src": source, copy.name: copy}, set(spilled)
            )

        copy_name = relayout_copy_name("src", 0)
        self.assertEqual(
            handoffs([10, 11], [11]),
            [_Handoff(source="src", copy=copy_name, consumer_tick=11)],
        )
        # The source is read again after the copy's consumer.
        self.assertEqual(handoffs([10, 11, 13], [11]), [])
        # Two consumers: the copy op may be listed before either one.
        self.assertEqual(handoffs([10, 11, 12], [11, 12]), [])
        # Nothing to pack when either side is spilled.
        self.assertEqual(handoffs([10, 11], [11], spilled=["src"]), [])
        self.assertEqual(handoffs([10, 11], [11], spilled=[copy_name]), [])

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

import torch

from torch_spyre._inductor.operation_scheduling import schedule_loop_body_for_liveness
from torch_spyre._inductor.patches import enable_spyre_context


class _FakeOperation:
    def __init__(self, name, reads=(), size=1, loop_path=(0,)):
        self.name = name
        self._reads = set(reads)
        self._size = [size]
        self.loop_info = SimpleNamespace(loop_group_id=loop_path)

    def get_name(self):
        return self.name

    def get_read_names(self):
        return self._reads

    def get_mutation_names(self):
        return ()

    def get_size(self):
        return self._size

    def get_dtype(self):
        return torch.float16


def test_loop_body_operation_order_is_preserved():
    # Even independent recurrence branches must remain in their input order;
    # scratchpad planning and the emitted LoopSpec use these same lifetimes.
    ops = [
        _FakeOperation("scores0", size=512),
        _FakeOperation("max0", ("scores0",), size=8),
        _FakeOperation("correction0", ("max0",), size=8),
        _FakeOperation("weighted0", ("scores0",), size=128),
        _FakeOperation("output0", ("weighted0", "correction0"), size=128),
        _FakeOperation("denominator0", ("scores0", "correction0"), size=8),
        _FakeOperation("scores1", ("max0",), size=512),
        _FakeOperation("max1", ("scores1",), size=8),
        _FakeOperation("correction1", ("max1",), size=8),
        _FakeOperation("weighted1", ("scores1",), size=128),
        _FakeOperation("output1", ("output0", "weighted1", "correction1"), size=128),
        _FakeOperation(
            "denominator1", ("denominator0", "scores1", "correction1"), size=8
        ),
    ]
    graph = SimpleNamespace(operations=ops)

    schedule_loop_body_for_liveness(graph)

    assert graph.operations == ops
    assert [op._spyre_preschedule_order for op in graph.operations] == list(
        range(len(ops))
    )


def test_loop_preheader_is_moved_before_the_atomic_loop_unit():
    first = _FakeOperation("first")
    copy = _FakeOperation("copy", loop_path=())
    consumer = _FakeOperation("consumer", ("copy",))
    graph = SimpleNamespace(operations=[first, copy, consumer])

    schedule_loop_body_for_liveness(graph)

    assert [op.name for op in graph.operations] == ["copy", "first", "consumer"]


def test_spyre_preserves_fx_program_order_before_loop_discovery():
    original = torch._inductor.config.reorder_for_locality
    with enable_spyre_context([]):
        assert not torch._inductor.config.reorder_for_locality
    assert torch._inductor.config.reorder_for_locality == original

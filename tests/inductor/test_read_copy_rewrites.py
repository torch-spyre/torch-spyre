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

"""Saved candidates undergo the same non-weight transforms as live consumers."""

from types import SimpleNamespace

import pytest
import sympy
import torch
from torch._inductor.ir import Pointwise
from torch._inductor.virtualized import V

from torch_spyre._inductor.loop_info import ReadCopyElisionRecord
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor
from torch_spyre._inductor.wsr.coarse_tile import (
    _RetiledBufferInfo,
    _retile_elision_record,
)


ROW, COL = sympy.symbols("i j", integer=True, nonnegative=True)


class Loads:
    def load(self, name, index):
        return name, index


def direct(index):
    i, j = index
    return V.ops.load("activation", 32 * i + j), V.ops.load("bank", 64 * i + j)


def staged(index):
    i, j = index
    return V.ops.load("activation", 32 * i + j), V.ops.load("copy", 64 * i + j)


def record():
    return ReadCopyElisionRecord("matmul", "copy", "bank", direct)


def read(fn):
    with V.set_ops_handler(Loads()):
        return fn([ROW, COL])


def info():
    return _RetiledBufferInfo(
        old_size=(sympy.Integer(8), sympy.Integer(32)),
        new_size=(sympy.Integer(8), sympy.Integer(8)),
        old_stride=(sympy.Integer(32), sympy.Integer(1)),
        new_stride=(sympy.Integer(8), sympy.Integer(1)),
    )


def test_retile_candidate_updates_activation_but_not_bank():
    original = record()
    updated = _retile_elision_record(original, {"activation": info()})
    assert read(updated.direct_inner_fn) == (
        ("activation", 8 * ROW + COL),
        ("bank", 64 * ROW + COL),
    )
    assert read(original.direct_inner_fn)[0] == ("activation", 32 * ROW + COL)
    assert (updated.consumer_name, updated.copy_name, updated.source_name) == (
        "matmul",
        "copy",
        "bank",
    )


@pytest.mark.parametrize("name", ["copy", "bank"])
def test_retile_of_saved_weight_contract_invalidates_candidate(name):
    assert _retile_elision_record(record(), {name: info()}) is None


@pytest.mark.parametrize("unknown", [None, object()])
def test_retile_does_not_invent_a_candidate(unknown):
    assert _retile_elision_record(unknown, {"activation": info()}) is None


def operation(saved=True):
    result = SimpleNamespace(
        data=Pointwise(
            device=torch.device("cpu"),
            dtype=torch.float16,
            inner_fn=staged,
            ranges=[8, 32],
        )
    )
    if saved:
        result._read_copy_elision_record = record()
    return result


def editor():
    # This unit exercises the actual graph rewrite without building an FX graph.
    return object.__new__(GraphEditor)


def test_relayout_renames_activation_in_live_and_saved_forms():
    op = operation()
    op._ts_cached_read_writes = object()
    editor()._replace_loop_input(op, "activation", "moved_activation")
    assert read(op.data.inner_fn) == (
        ("moved_activation", 32 * ROW + COL),
        ("copy", 64 * ROW + COL),
    )
    assert read(op._read_copy_elision_record.direct_inner_fn) == (
        ("moved_activation", 32 * ROW + COL),
        ("bank", 64 * ROW + COL),
    )
    assert not hasattr(op, "_ts_cached_read_writes")


@pytest.mark.parametrize("name", ["copy", "bank"])
def test_relayout_of_saved_weight_contract_invalidates_candidate(name):
    op = operation()
    editor()._replace_loop_input(op, name, "different_weight")
    assert not hasattr(op, "_read_copy_elision_record")


def test_relayout_does_not_invent_a_candidate():
    op = operation(saved=False)
    editor()._replace_loop_input(op, "activation", "moved_activation")
    assert not hasattr(op, "_read_copy_elision_record")


def test_retile_then_two_relayouts_compose_without_rewriting_bank():
    op = operation()
    op._read_copy_elision_record = _retile_elision_record(
        record(), {"activation": info()}
    )
    rewrite = editor()
    rewrite._replace_loop_input(op, "activation", "first_move")
    rewrite._replace_loop_input(op, "first_move", "second_move")
    assert read(op._read_copy_elision_record.direct_inner_fn) == (
        ("second_move", 8 * ROW + COL),
        ("bank", 64 * ROW + COL),
    )

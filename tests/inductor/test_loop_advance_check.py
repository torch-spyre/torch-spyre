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

"""The kernel's check that an access's stored advance equals its index's (stage 7.2).

``SpyreKernel._checked_tile_advance`` builds the advance from the stored record and
from the index, and raises ``LoopAdvanceMismatch`` when they differ. These tests pin
its decision logic on a kernel whose two builders are replaced.

The check also runs on every compile, so the compiled tests elsewhere are its guard on
real graphs: it raised on four of them when a pass moved a loop term out of an index
and left the step in the record (``test_sdpa_tiling.py::
test_fused_qkv_views_under_query_tiling``; in ``test_building_blocks.py``
``test_granite_gqa_prefill_noncontiguous_kv_prefix``,
``test_noncontiguous_kv_prefix_keeps_one_tile_fallback`` and
``test_siglip_multicrop_attention_span``).
"""

from types import SimpleNamespace

import pytest
import sympy
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from torch_spyre._inductor import spyre_kernel
from torch_spyre._inductor.loop_checker import LoopAdvanceMismatch

SYM = sympy.Symbol("_tile_adv_op_lvl0")
DEP = SimpleNamespace(index=sympy.Symbol("u0") * 8192)


def _kernel(recorded, from_index, *, consumed_after=1, consumed_before=0):
    """A kernel whose two builders return fixed values.

    ``consumed_*`` is the ``_general_tile_advance_seen`` count for the buffer before
    and after the record-based builder runs, which is how the check tells a repeat
    load (nothing consumed) from a first read.
    """
    kernel = spyre_kernel.SpyreKernel.__new__(spyre_kernel.SpyreKernel)
    kernel._general_tile_advance_seen = {"buf": consumed_before}

    def record_based(tensor, is_input, name):
        kernel._general_tile_advance_seen["buf"] = consumed_after
        return recorded

    kernel._general_tile_advance = record_based
    kernel._index_tile_advance = lambda tensor, is_input, name: from_index
    kernel.current_node = SimpleNamespace(
        node=SimpleNamespace(get_operation_name=lambda: "op3")
    )
    return kernel


def test_equal_advances_return_the_index_based_one():
    """The index is the source (stage 7.2c): the two builders can write the same
    value differently, and the one that is returned is the index's."""
    recorded = sympy.floor(8192 * SYM)
    # The same value written differently: sympy does not fold the "+ 0".
    from_index = sympy.Add(recorded, sympy.Integer(0), evaluate=False)
    assert recorded != from_index
    assert sympy.simplify(recorded - from_index) == 0
    kernel = _kernel(recorded, (from_index, DEP))
    assert kernel._checked_tile_advance(None, True, "buf") is from_index


def test_an_index_advance_of_zero_is_no_advance_not_a_zero():
    """Consumers test ``device_tile_advance_expr is not None``, so "no advance" must
    stay None; a zero expression would make the access look like an advancing one."""
    kernel = _kernel(None, (sympy.S.Zero, DEP))
    assert kernel._checked_tile_advance(None, True, "buf") is None


def test_a_stored_zero_against_a_zero_index_advance_is_no_advance():
    kernel = _kernel(sympy.S.Zero, (sympy.S.Zero, DEP))
    assert kernel._checked_tile_advance(None, True, "buf") is None


def test_an_op_with_no_index_based_value_keeps_the_stored_advance():
    """An op with no hinted loop variable (span-overflow, the allocator) has no
    index-based value; its stored advance is returned as it is."""
    expr = sympy.floor(8192 * SYM)
    kernel = _kernel(expr, None)
    assert kernel._checked_tile_advance(None, True, "buf") is expr


def test_a_differing_advance_raises_and_names_both_values():
    kernel = _kernel(sympy.floor(8192 * SYM), (sympy.S.Zero, DEP))
    with pytest.raises(LoopAdvanceMismatch) as caught:
        kernel._checked_tile_advance(None, True, "buf")
    message = str(caught.value)
    assert "op3 reads buf" in message
    assert "8192" in message and "index 8192*u0" in message


def test_no_stored_advance_against_an_advancing_index_raises():
    """The record builder answers None for "no advance"; that is a zero, not a skip."""
    kernel = _kernel(None, (sympy.floor(4096 * SYM), DEP))
    with pytest.raises(LoopAdvanceMismatch):
        kernel._checked_tile_advance(None, True, "buf")


def test_a_repeat_load_consumes_no_match_and_is_not_compared():
    """``x + x`` deduplicates to one dependency; the second load gets None from the
    record builder and must not be compared with the index (#6093-era bug)."""
    kernel = _kernel(None, (sympy.floor(4096 * SYM), DEP), consumed_before=1)
    assert kernel._checked_tile_advance(None, True, "buf") is None


def test_a_write_is_compared_even_though_it_consumes_no_read_match():
    """The "nothing consumed" skip is for reads; a write has no consumption counter."""
    kernel = _kernel(
        sympy.floor(8192 * SYM),
        (sympy.S.Zero, DEP),
        consumed_before=0,
        consumed_after=0,
    )
    with pytest.raises(LoopAdvanceMismatch):
        kernel._checked_tile_advance(None, False, "buf")


def test_the_error_is_not_unsupported():
    """A compiler inconsistency, not a user error (see LoopStructureError)."""
    from torch_spyre._inductor.errors import Unsupported

    assert not issubclass(LoopAdvanceMismatch, Unsupported)
    assert issubclass(LoopAdvanceMismatch, RuntimeError)

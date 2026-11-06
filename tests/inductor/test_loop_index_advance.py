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

"""Can a read's per-trip advance be taken from its index alone? (stage 7.0)

The loop plan (epic #5863, stage 7) makes the index the single source of an
access's advance: the coefficient of the loop's induction variable in
``dep.index``, projected through ``tiling_expr_to_device_expr``. Today the
kernel combines the index with the stored per-read record.

This file measures the premise on compiled ``for_each_tile`` graphs. At the
point where the kernel builds each ``TensorArg`` it computes the advance from the
index alone and compares it with the expression the kernel emits, with the
kernel's own level symbol standing in for the induction variable. That compares
VALUES, not just whether the variable is present.

``test_loop_advance.py`` is the gate on the record-based accessors; this is a
different question over the same corpus, so it is a separate file.

What it pins:

* every access in the corpus agrees, except the block-table reads of the
  ``paged_gather*`` fixtures. Those are POINT reads whose index
  ``_rebase_point_splice_reads`` rewrites to ``0`` on purpose, because layout
  propagation rejects the loop variable. The allow-list is exact: a fifth
  disagreement fails.
* a padded weight read (K = 67, tiled weight) does NOT agree today: padding drops
  the loop term from the index and the record covers for it (#6049). That test is
  a strict xfail and passes when the gap is fixed (stage 7.1).

What it does not cover: a read that advances through a non-affine index term
(``floor(u/2)``, ``Mod(u, 2)``). ``coeff`` reads those as 0, and no corpus graph
produces one.
"""

import contextlib
from unittest import mock

import pytest
import sympy
import torch
from torch._inductor.dependencies import MemoryDep

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor import loop_node
from torch_spyre._inductor import passes as spyre_passes
from torch_spyre._inductor.spyre_kernel import SpyreKernel
from torch_spyre._inductor.views import tiling_expr_to_device_expr
from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

from test_loop_advance import CORPUS
from utils_inductor import strict_xfail


class _Probe:
    """Records, per access, the kernel's advance and the index-derived one."""

    def __init__(self):
        self.inductions = {}
        self.rows = []  # (op, kind, buffer, index, kernel_expr, from_index, equal)

    def hand_off(self, original):
        def wrapper(graph):
            operations = getattr(graph.operations, "real", graph.operations)
            self.inductions.update(loop_node._inductions_of(operations))
            return original(graph)

        return wrapper

    def wrap(self, original):
        def wrapper(kernel, tensor, is_input, name):
            result = original(kernel, tensor, is_input, name)
            self._record(kernel, tensor, is_input, name, result)
            return result

        return wrapper

    def _dep(self, kernel, node, is_input, name):
        reads_writes = node.get_read_writes()
        if not is_input:
            writes = [d for d in reads_writes.writes if isinstance(d, MemoryDep)]
            return writes[0] if writes else None
        matching = [
            d for d in reads_writes.reads if isinstance(d, MemoryDep) and d.name == name
        ]
        # The kernel has already consumed one match for this call.
        consumed = kernel._general_tile_advance_seen.get(name, 0)
        if 0 < consumed <= len(matching):
            return matching[consumed - 1]
        return matching[0] if matching else None

    def _record(self, kernel, tensor, is_input, name, result):
        node = kernel.current_node.node
        info = getattr(node, "loop_info", None)
        if info is None:
            return
        dep = self._dep(kernel, node, is_input, name)
        if dep is None:
            return
        layout = tensor.layout.device_layout
        from_index = sympy.S.Zero
        for level, group in enumerate(info.loop_group_id):
            induction = self.inductions.get(group)
            if induction is None:
                continue
            coefficient = dep.index.coeff(induction)
            if coefficient == 0:
                continue
            symbol = kernel._tile_advance_symbols.get(level) or sympy.Symbol(
                f"_tile_adv_{node.get_operation_name()}_lvl{level}"
            )
            from_index += tiling_expr_to_device_expr(
                layout.device_size, layout.stride_map, coefficient * symbol
            )
        theirs = sympy.S.Zero if result is None else result
        self.rows.append(
            (
                node.get_operation_name(),
                "read" if is_input else "write",
                name,
                str(dep.index),
                theirs,
                from_index,
                sympy.simplify(theirs - from_index) == 0,
            )
        )


def _compile_and_probe(fn, args, ctx=None):
    probe = _Probe()
    torch._dynamo.reset()
    with (
        mock.patch.object(
            SpyreKernel,
            "_general_tile_advance",
            autospec=True,
            side_effect=probe.wrap(SpyreKernel._general_tile_advance),
        ),
        mock.patch.object(
            spyre_passes,
            "_hand_off_loop_nodes",
            probe.hand_off(spyre_passes._hand_off_loop_nodes),
        ),
        torch._inductor.config.patch(force_disable_caches=True),
        ctx if ctx is not None else contextlib.nullcontext(),
    ):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled(*(a.to(DEVICE_NAME) for a in args))
    return probe


# The block-table reads of the paged-gather fixtures: a point read has no
# iteration variable, its index is ``32*u0`` at the splice and is rebased to
# ``0`` afterwards, so the advance survives only in the record.
POINT_READ_FIXTURES = {"paged_gather", "paged_gather_kv", "paged_gather_nested"}


@pytest.mark.parametrize("case", sorted(CORPUS))
def test_the_index_gives_the_kernels_advance(case):
    fn, args, ctx = CORPUS[case]()
    probe = _compile_and_probe(fn, args, ctx)
    assert probe.rows, f"{case}: the kernel built no TensorArg with a loop_info"

    differing = [r for r in probe.rows if not r[6]]
    if case in POINT_READ_FIXTURES:
        # Exactly the point reads differ, and for the stated reason: the index
        # is the constant 0 while the kernel advances.
        assert differing, f"{case}: expected the point read to differ"
        for op, kind, name, index, kernel_expr, from_index, _ in differing:
            assert kind == "read" and index == "0", (op, kind, name, index)
            assert kernel_expr != 0 and from_index == 0, (op, kernel_expr, from_index)
    else:
        assert not differing, [
            (r[0], r[1], r[2], r[3], str(r[4]), str(r[5])) for r in differing
        ]

    advancing = [r for r in probe.rows if r[4] != 0]
    assert advancing, f"{case}: nothing advanced, so the comparison proves nothing"


def _padded_weight_case(tile):
    """K = 67 is not a multiple of 64, so insert_bmm_padding pads the weight;
    tiling N makes the padded weight read advance every trip."""
    m, k, n = 256, 67, 256
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(m, k, generator=g) * 0.1).half()
    y = (torch.randn(k, n, generator=g) * 0.1).half()

    def fn(a, b):
        def body(_, ops):
            (y_tile,) = ops
            return None, a @ y_tile

        _, out = for_each_tile(body, (b,), dims=(1,), tile_size=tile, out_dim=1)
        return out

    return fn, (x, y)


@pytest.mark.parametrize("tile", [64, 128])
def test_a_padded_weight_read_keeps_its_loop_term_in_the_index(request, tile):
    strict_xfail(
        request,
        AssertionError,
        "padding drops the loop term from a padded weight read's index and the "
        "stored record covers for it; the kernel advance is floor(8192*s) at "
        "tile 64 and floor(16384*s) at tile 128, the index gives 0 "
        "(#6049, fixed by stage 7.1)",
    )
    fn, args = _padded_weight_case(tile)
    probe = _compile_and_probe(fn, args)
    differing = [r for r in probe.rows if not r[6]]
    assert not differing, [
        (r[0], r[1], r[2], r[3], str(r[4]), str(r[5])) for r in differing
    ]
    assert any(r[4] != 0 for r in probe.rows)


def test_an_invariant_padded_weight_has_nothing_to_advance():
    """Control for the xfail above: with the weight NOT tiled (tile M), the
    padded weight read correctly has no advance, and the two sources agree."""
    m, k, n = 256, 67, 64
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(m, k, generator=g) * 0.1).half()
    y = (torch.randn(k, n, generator=g) * 0.1).half()

    def fn(a, b):
        def body(_, ops):
            (x_tile,) = ops
            return None, x_tile @ b

        _, out = for_each_tile(body, (a,), dims=(0,), tile_size=64, out_dim=0)
        return out

    probe = _compile_and_probe(fn, (x, y))
    assert probe.rows
    assert all(r[6] for r in probe.rows), [
        (r[0], r[1], r[2], r[3], str(r[4]), str(r[5])) for r in probe.rows if not r[6]
    ]


def test_coeff_reads_a_non_affine_use_of_the_variable_as_no_advance():
    """The rule's blind spot, stated so it cannot be forgotten: the variable can
    be in an index while ``coeff`` is 0. ``floor(u/2)`` moves every second trip
    and ``Mod(u, 2)`` alternates; ``coeff`` reports neither. No corpus graph
    builds one, so this is not an observed failure, only a limit of the rule."""
    u, d0 = sympy.symbols("u0 d0", integer=True)
    for index in (sympy.floor(u / 2) + d0, sympy.Mod(u, 2) * 64 + d0):
        assert u in index.free_symbols
        assert index.coeff(u) == 0
    assert (64 * u + d0).coeff(u) == 64

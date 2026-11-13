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

* every access in the corpus agrees, including the block-table reads of the
  ``paged_gather*`` fixtures. Those are POINT reads (no iteration variable) whose
  index used to be rewritten to ``0`` by ``_rebase_point_splice_reads`` because
  layout propagation was thought to reject the loop variable, so their advance
  survived only in the stored record. Stage 7.2 removed that rewrite (layout
  propagation accepts the variable: the three fixtures compile and give the same
  result), so the index carries the term and the comparison now holds for them too.
* a padded weight read (K = 67) agrees. Padding used to drop the loop term from
  its index and let the stored record cover for it (#6049); it now keeps the term
  at the padded buffer's stride. Three weights are covered because the stride a
  padded read needs differs: a ``[K, N]`` weight tiled along N (padding K leaves
  the step as it is), and per-expert ``[E, K, N]`` and ``[E, N, K]`` weights
  (padding K changes the per-expert stride). The existing padding tests check
  numerics, which the record keeps right, so they cannot see a wrong stride in the
  index; this comparison can. The transposed ``[E, N, K]`` weight needed a second
  fix: ``insert_restickify`` inserts a relayout copy after padding and used to move
  the advance onto it in the record only (#6051). It now moves the loop term in the
  index as well.

What it does not cover: a read that advances through a non-affine index term
(``floor(u/2)``, ``Mod(u, 2)``). ``coeff`` reads those as 0, and no corpus graph
produces one.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_index_symbol

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor import loop_node
from torch_spyre._inductor import passes as spyre_passes
from torch_spyre._inductor.spyre_kernel import SpyreKernel
from torch_spyre._inductor.views import tiling_expr_to_device_expr
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.padding import restride_loop_step
from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

from test_loop_advance import CORPUS


d1, d2 = (sympy_index_symbol(f"d{i}") for i in (1, 2))


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


@pytest.mark.parametrize("case", sorted(CORPUS))
def test_the_index_gives_the_kernels_advance(case):
    fn, args, ctx = CORPUS[case]()
    probe = _compile_and_probe(fn, args, ctx)
    assert probe.rows, f"{case}: the kernel built no TensorArg with a loop_info"

    differing = [r for r in probe.rows if not r[6]]
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


def _relative_error(fn, args, reference):
    torch._dynamo.reset()
    with torch._inductor.config.patch(force_disable_caches=True):
        out = torch.compile(fn, backend="inductor", fullgraph=True)(
            *(a.to(DEVICE_NAME) for a in args)
        )
    out = out.cpu().float()
    assert not torch.isnan(out).any()
    return ((out - reference).abs().max() / reference.abs().max()).item()


@pytest.mark.parametrize("tile", [64, 128])
def test_a_padded_weight_gives_the_right_numbers(tile):
    """The comparison above checks the index against the kernel, not the result.
    fp16 inputs scaled to 0.1 and a matmul over K = 67 keep the fp32 reference
    within about 3e-3 relative error; 2e-2 leaves headroom without hiding the
    NaN / wrong-tile failures this guards against."""
    fn, (x, y) = _padded_weight_case(tile)
    reference = x.float() @ y.float()
    assert _relative_error(fn, (x, y), reference) < 2e-2


@pytest.mark.parametrize("transposed", [False, True], ids=["E_K_N", "E_N_K"])
def test_a_padded_expert_weight_gives_the_right_numbers(transposed):
    """#5573 returned NaN for about 70% of the transposed result."""
    fn, (x, w) = _expert_weight_case(transposed)
    weight = w.float().transpose(-1, -2) if transposed else w.float()
    reference = torch.matmul(x.float(), weight)
    assert _relative_error(fn, (x, w), reference) < 2e-2


@pytest.mark.parametrize("tile", [64, 128])
def test_a_padded_weight_read_keeps_its_loop_term_in_the_index(tile):
    fn, args = _padded_weight_case(tile)
    probe = _compile_and_probe(fn, args)
    differing = [r for r in probe.rows if not r[6]]
    assert not differing, [
        (r[0], r[1], r[2], r[3], str(r[4]), str(r[5])) for r in differing
    ]
    # The comparison must not be vacuous: the padded weight read advances.
    padded = [r for r in probe.rows if r[1] == "read" and r[4] != 0]
    assert padded, "no advancing read: the padded weight read was not exercised"


def _expert_weight_case(transposed):
    """Per-expert weight with K = 67, tiled one expert per trip.

    Padding K grows the per-expert stride (67 * 128 to 128 * 128), so the loop
    term has to be re-strided: keeping the unpadded step walks the padded buffer
    by the wrong amount (#5573)."""
    e, m, k, n = 4, 64, 67, 128
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(e, m, k, generator=g) * 0.1).half()
    w_shape = (e, n, k) if transposed else (e, k, n)
    w = (torch.randn(*w_shape, generator=g) * 0.1).half()

    def fn(a, b):
        def body(_, ops):
            weight = ops[1].transpose(-1, -2) if transposed else ops[1]
            return None, torch.matmul(ops[0], weight)

        return for_each_tile(body, (a, b), dims=(0, 0), tile_size=1, out_dim=0)[1]

    return fn, (x, w)


@pytest.mark.parametrize("transposed", [False, True], ids=["E_K_N", "E_N_K"])
def test_a_padded_expert_weight_read_is_restrided_in_the_index(transposed):
    fn, args = _expert_weight_case(transposed)
    probe = _compile_and_probe(fn, args)
    differing = [r for r in probe.rows if not r[6]]
    assert not differing, [
        (r[0], r[1], r[2], r[3], str(r[4]), str(r[5])) for r in differing
    ]
    assert any(r[1] == "read" and r[4] != 0 for r in probe.rows)


def test_an_invariant_padded_weight_has_nothing_to_advance():
    """Control for the padded-weight tests above: with the weight NOT tiled (tile
    M), the padded weight read correctly has no advance, and the two sources agree."""
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


def _strides(size):
    out = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        out[i] = out[i + 1] * size[i + 1]
    return out


@pytest.mark.parametrize(
    "step, size, padded_size, expected",
    [
        # padding K leaves a [K, N] weight's row stride alone: a tile along N keeps its step
        (64, [67, 256], [128, 256], 64),
        (128, [67, 256], [128, 256], 128),
        # a per-expert stride includes K: 67 * 128 -> 128 * 128, and two experts per trip
        (8576, [4, 67, 128], [4, 128, 128], 16384),
        (17152, [4, 67, 128], [4, 128, 128], 32768),
        # [E, N, K]: the per-expert stride 64 * 67 -> 64 * 128
        (4288, [4, 64, 67], [4, 64, 128], 8192),
        # a step no dim owns is returned unchanged
        (7, [4, 67, 128], [4, 128, 128], 7),
    ],
)
def test_restride_loop_step(step, size, padded_size, expected):
    result = restride_loop_step(
        sympy.Integer(step), size, _strides(size), _strides(padded_size), "w"
    )
    assert result == expected


@pytest.mark.parametrize(
    "step, size, stride",
    [
        # two dims of size above one share the stride
        (64, [64, 64], [64, 64]),
        # a non-contiguous layout where one step reads as one position along
        # dim 0 or two along dim 1: re-striding the wrong one reads the wrong place
        (128, [4, 4], [128, 64]),
    ],
)
def test_restride_loop_step_rejects_a_step_two_dims_could_own(step, size, stride):
    with pytest.raises(Unsupported, match="cannot tell which dim"):
        restride_loop_step(sympy.Integer(step), size, stride, stride, "w")


def test_a_contiguous_layout_never_gives_a_step_two_readings():
    """Why the rule can raise on ambiguity without refusing real graphs: in a
    contiguous row-major buffer, no whole-position step is owned by two dims."""
    for size in ([4, 67, 128], [4, 128, 67], [67, 256], [2, 3, 4], [8, 8], [3, 5, 7]):
        stride = _strides(size)
        for dim, extent in enumerate(size):
            for count in range(1, extent):
                step = count * stride[dim]
                restride_loop_step(sympy.Integer(step), size, stride, stride, "w")


def test_loop_induction_symbols_is_the_unbacked_non_iteration_non_indirect_set():
    """``concretize_index`` keeps a loop variable symbolic because it is an unbacked
    symbol that is not an iteration variable; ``loop_induction_symbols`` must name
    the same set, so this pins the three-way split between a loop variable, a size
    symbol and an indirect symbol."""
    from torch._inductor.utils import sympy_index_symbol
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    from torch_spyre._inductor.pass_utils import loop_induction_symbols

    d0 = sympy_index_symbol("d0")
    u0 = sympy.Symbol("u0", integer=True)
    s0 = sympy.Symbol("s0", integer=True, positive=True)
    tmp0 = sympy_index_symbol("tmp0")
    index = 64 * d0 + 128 * u0 + s0 + tmp0
    assert free_unbacked_symbols(index) == {u0}
    assert loop_induction_symbols(index, {d0}) == [u0]
    # an iteration variable is never a loop variable, even if it is unbacked
    assert loop_induction_symbols(index, {d0, u0}) == []
    # not a sympy expression: nothing to find
    assert loop_induction_symbols(0, set()) == []


class _ReadOp:
    """A consumer with one memory read, enough for ``_moved_loop_term``."""

    def __init__(self, index):
        self.loop_info = SimpleNamespace()
        self._dep = MemoryDep("buf", index, (), ())

    def get_read_writes(self):
        return SimpleNamespace(reads=[self._dep])


def _moved_term(index):
    from torch_spyre._inductor import insert_restickify as ir

    transferred = SimpleNamespace(
        tiled_dims_per_read=[[[]]], squeezed_advance_per_read=[[[(16384, 1)]]]
    )
    copy = SimpleNamespace(loop_info=transferred)
    info = SimpleNamespace(arg_name="buf", dep_index=index, occurrence=0)
    return ir._moved_loop_term(_ReadOp(index), info, copy, 1, 1)


def test_a_linear_loop_term_moves_to_the_restickify_copy():
    u0 = sympy.Symbol("u0", integer=True)
    assert _moved_term(128 * d1 + d2 + 16384 * u0) == 16384 * u0


def test_a_loop_variable_used_non_linearly_cannot_move():
    """``coeff`` reads only the linear part. floor(u/2) would stay in the consumer's
    index while the copy took a term of 0, so it must be refused, not moved."""
    u0 = sympy.Symbol("u0", integer=True)
    for index in (d1 + sympy.floor(u0 / 2), d1 + 64 * sympy.Mod(u0, 2)):
        with pytest.raises(Unsupported, match="non-linearly"):
            _moved_term(index)


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

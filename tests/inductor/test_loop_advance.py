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

"""Tests for the per-read / output advance accessors in ``loop_facts``.

Two layers.  Unit tests pin the semantics on stub operations.  The corpus
test compiles the ``for_each_tile`` graphs and, at the point where the kernel
builds each ``TensorArg``, compares the accessor against the
``device_tile_advance_expr`` the kernel emits, and the ``*_anywhere``
accessors against the record-based readers in ``scratchpad/utils.py``.  It
compares against the existing readers, so it cannot pass by agreeing with
itself.
"""

import contextlib
import dataclasses
import sys
from types import SimpleNamespace
from unittest import mock

import torch
import pytest
import sympy
import torch.nn.functional as F
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import V

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor.loop_facts import (
    output_advance,
    output_advances_anywhere,
    read_advance,
    read_advances_anywhere,
)
from torch_spyre._inductor.loop_info import CoarseTileInfo
from torch_spyre._inductor.scratchpad.utils import (
    _is_read_advancing_anywhere,
    _is_tiled_advancing,
)
from torch_spyre._inductor.spyre_kernel import SpyreKernel
from torch_spyre._inductor.views import tiling_expr_to_device_expr

from for_each_tile_fixtures import (
    B,
    D,
    K,
    LK,
    LQ,
    M,
    N,
    PAGE_HS,
    PAGE_LQ,
    PAGE_POOL,
    PAGE_SIZE,
    STICK_COLS,
    STICK_ROWS,
    batched_online_softmax_fn,
    matmul_x_yt_nested_fn,
    moe_tile_experts_fn,
    nested_add_outer_row_inner_col_fn,
    nested_online_softmax_fn,
    nested_split_m_then_k_fn,
    nested_two_inner_loops_shared_init_fn,
    online_softmax_fn,
    paged_gather_fn,
    paged_gather_inputs,
    paged_gather_kv_fn,
    paged_gather_kv_inputs,
    paged_gather_nested_fn,
    shaped_randn,
    softmax_row_tiled_fn,
    split_k_caller_init_fn,
    split_k_fn,
    split_k_transposed_caller_init_fn,
    split_m_fn,
    triple_nested_stardep_multilevel_fn,
    unsqueeze_broadcast_tiled_fn,
    view_transpose_head_tiled_fn,
)

d0, d1, d2 = (sympy_index_symbol(f"d{i}") for i in range(3))


# ---------------------------------------------------------------------------
# Unit tests on stub operations
# ---------------------------------------------------------------------------


def _dep(name, index, nvars=2):
    var_names = tuple(sympy_index_symbol(f"d{i}") for i in range(nvars))
    return MemoryDep(name, index, var_names, tuple([sympy.Integer(8)] * nvars))


def _op(reads=(), writes=(), info=None, ranges=(8, 128)):
    rw = SimpleNamespace(reads=list(reads), writes=list(writes))
    op = SimpleNamespace(
        get_read_writes=lambda: rw,
        data=SimpleNamespace(ranges=[sympy.Integer(r) for r in ranges]),
    )
    if info is not None:
        op.loop_info = info
    return op


def _info(depth, per_read=(), out=(), sq_read=(), sq_out=()):
    return CoarseTileInfo(
        loop_group_id=tuple(range(depth)),
        loop_count=[sympy.Integer(2)] * depth,
        loop_tiled_dims=[[] for _ in range(depth)],
        loop_tiled_reduction_dims=[[] for _ in range(depth)],
        tiled_dims_per_read=[list(r) for r in per_read],
        output_tiled_dims=list(out),
        squeezed_advance_per_read=[list(r) for r in sq_read],
        squeezed_advance_output=list(sq_out),
    )


X = _dep("x", 128 * d0 + d1)
Y = _dep("y", d1)
OUT = _dep("out", 128 * d0 + d1)


def test_no_loop_info():
    op = _op(reads=[X], writes=[OUT])
    assert read_advance(op, X) == ()
    assert output_advance(op) == ()
    assert not read_advances_anywhere(op, X)
    assert not output_advances_anywhere(op)


def test_dep_not_a_read():
    op = _op(reads=[X], writes=[OUT], info=_info(1, per_read=[[[(0, 4)]]]))
    assert read_advance(op, Y) == ()
    assert read_advance(op, OUT) == ()
    assert not read_advances_anywhere(op, Y)


def test_one_tiled_dim():
    op = _op(reads=[X, Y], writes=[OUT], info=_info(1, per_read=[[[(0, 4)]], [[]]]))
    assert read_advance(op, X) == (512,)
    assert read_advance(op, Y) == (sympy.S.Zero,)
    assert read_advances_anywhere(op, X)
    assert not read_advances_anywhere(op, Y)


def test_two_levels_tile_the_same_dim():
    # Outer extent = inner extent (16) * inner count (4).
    info = _info(2, per_read=[[[(0, 64)], [(0, 16)]]])
    op = _op(reads=[X], writes=[OUT], info=info)
    assert read_advance(op, X) == (128 * 64, 128 * 16)


def test_constant_offset_is_not_an_advance():
    # A slice offset (+32) is not part of the per-trip step.
    dep = _dep("x", 128 * d0 + d1 + 32)
    op = _op(reads=[dep], writes=[OUT], info=_info(1, per_read=[[[(0, 4)]]]))
    assert read_advance(op, dep) == (512,)


def test_squeezed_only_read():
    op = _op(
        reads=[X],
        writes=[OUT],
        info=_info(1, per_read=[[[]]], sq_read=[[[(sympy.Integer(8192), 1)]]]),
    )
    assert read_advance(op, X) == (8192,)
    assert read_advances_anywhere(op, X)


def test_caller_supplied_reads_are_used_in_place_of_get_read_writes():
    # scratchpad passes the memoized op_read_writes reads; the record lookup
    # must use them, not re-trace the op.
    info = _info(1, per_read=[[[]], [[(0, 4)]]])
    op = _op(reads=[X, Y], writes=[OUT], info=info)
    op.get_read_writes = None  # would raise if called
    assert read_advances_anywhere(op, Y, reads=[X, Y])
    assert not read_advances_anywhere(op, X, reads=[X, Y])
    assert not read_advances_anywhere(op, OUT, reads=[X, Y])


def test_same_buffer_read_twice_takes_positional_record():
    a = _dep("x", 128 * d0 + d1)
    b = _dep("x", d1)
    info = _info(1, per_read=[[[(0, 2)]], [[(1, 64)]]])
    op = _op(reads=[a, b], writes=[OUT], info=info)
    assert read_advance(op, a) == (256,)
    assert read_advance(op, b) == (64,)


def test_empty_level_is_zero():
    info = _info(3, per_read=[[[], [(0, 2)], []]])
    op = _op(reads=[X], writes=[OUT], info=info)
    assert read_advance(op, X) == (sympy.S.Zero, 256, sympy.S.Zero)


def test_record_shorter_than_depth_pads_with_zero():
    op = _op(reads=[X, Y], writes=[OUT], info=_info(2, per_read=[[[(0, 2)]]]))
    assert read_advance(op, X) == (256, sympy.S.Zero)
    assert read_advance(op, Y) == (sympy.S.Zero, sympy.S.Zero)


def test_unit_dim_is_squeezed_out_of_symbol_numbering():
    # ranges (1, 8, 128): host dim 1 is d0 in dep.index.
    op = _op(
        reads=[X],
        writes=[OUT],
        info=_info(1, per_read=[[[(1, 2)]]]),
        ranges=(1, 8, 128),
    )
    assert read_advance(op, X) == (256,)


def test_output():
    info = _info(
        2, out=[[(1, 64)], []], sq_out=[[], [(sympy.Integer(1024), sympy.Integer(1))]]
    )
    op = _op(reads=[X], writes=[OUT, _dep("other", d0)], info=info)
    assert output_advance(op) == (64, 1024)
    assert output_advances_anywhere(op)


def test_output_with_no_memory_write():
    # No term without a write; the record reader still answers from the
    # record, as _is_tiled_advancing does.
    op = _op(reads=[X], writes=[], info=_info(1, out=[[(0, 2)]]))
    assert output_advance(op) == ()
    assert output_advances_anywhere(op)


def test_does_not_mutate():
    info = _info(1, per_read=[[[(0, 4)]]], out=[[(0, 4)]])
    before = dataclasses.replace(info)
    op = _op(reads=[X], writes=[OUT], info=info)
    read_advance(op, X)
    output_advance(op)
    assert op.loop_info == before
    assert X.index == 128 * d0 + d1


# ---------------------------------------------------------------------------
# Corpus comparison against what the kernel emits today
# ---------------------------------------------------------------------------


def _xavier(*shape, seed=0):
    out = torch.empty(shape, dtype=torch.float16)
    torch.nn.init.xavier_uniform_(out, generator=torch.Generator().manual_seed(seed))
    return out


def _sdpa_complete_gqa_tile_nest():
    """test_sdpa_tiling.test_complete_gqa_tile_nest's forced B/Hkv/G/Lq/Lk nest."""
    decomp = sys.modules["torch_spyre._inductor.decompositions"]
    select = decomp._select_sdpa_tiling

    def force_every_loop(**kwargs):
        return dataclasses.replace(
            select(**kwargs),
            strategy="work_divided_tiled",
            kv_block_size=128,
            num_kv_blocks=2,
            num_q_tiles=2,
            q_tile_size=16,
            num_batch_tiles=2,
            num_head_tiles=2,
            num_group_tiles=2,
            kv_blocks_per_loop_group=2,
        )

    def sdpa(q, k, v, b):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=b, dropout_p=0.0, scale=128**-0.5, enable_gqa=True
        )

    g = torch.Generator().manual_seed(0)
    args = (
        torch.randn(2, 4, 32, 128, dtype=torch.float16, generator=g),
        torch.randn(2, 2, 256, 128, dtype=torch.float16, generator=g),
        torch.randn(2, 2, 256, 128, dtype=torch.float16, generator=g),
        torch.randn(2, 4, 32, 256, dtype=torch.float16, generator=g) * 0.01,
    )
    ctx = mock.patch.object(decomp, "_select_sdpa_tiling", side_effect=force_every_loop)
    return sdpa, args, ctx


def _split_k_operands(acc_shape):
    return (shaped_randn(M, K), shaped_randn(K, N, seed=1), shaped_randn(*acc_shape))


def _plain(fn, *args):
    return fn, args, None


CORPUS = {
    "split_m": lambda: _plain(split_m_fn, shaped_randn(M, K), shaped_randn(K, N)),
    "split_k": lambda: _plain(split_k_fn, shaped_randn(M, K), shaped_randn(K, N)),
    "split_k_caller_init": lambda: _plain(
        split_k_caller_init_fn, *_split_k_operands((M, N))
    ),
    "split_k_transposed_caller_init": lambda: _plain(
        split_k_transposed_caller_init_fn, *_split_k_operands((N, M))
    ),
    "nested_split_m_then_k": lambda: _plain(
        nested_split_m_then_k_fn, _xavier(256, 256), _xavier(256, 64, seed=1)
    ),
    "nested_two_inner_loops_shared_init": lambda: _plain(
        nested_two_inner_loops_shared_init_fn,
        shaped_randn(M, K),
        shaped_randn(K, N, seed=1),
    ),
    "triple_nested_stardep_multilevel": lambda: _plain(
        triple_nested_stardep_multilevel_fn,
        _xavier(2, 256, 256),
        _xavier(2, 256, 64, seed=1),
    ),
    "online_softmax": lambda: _plain(
        online_softmax_fn,
        _xavier(LQ, D),
        _xavier(LK, D, seed=1),
        _xavier(LK, D, seed=2),
    ),
    "nested_online_softmax": lambda: _plain(
        nested_online_softmax_fn,
        _xavier(LQ, D),
        _xavier(LK, D, seed=1),
        _xavier(LK, D, seed=2),
    ),
    "batched_online_softmax": lambda: _plain(
        batched_online_softmax_fn,
        _xavier(B, 64, 128),
        _xavier(B, 256, 128, seed=1),
        _xavier(B, 256, 128, seed=2),
    ),
    "sdpa_complete_gqa_tile_nest": _sdpa_complete_gqa_tile_nest,
    "paged_gather": lambda: _plain(
        paged_gather_fn,
        _xavier(PAGE_POOL, PAGE_SIZE, PAGE_HS),
        paged_gather_inputs()[1],
        _xavier(PAGE_LQ, PAGE_HS, seed=1),
    ),
    "paged_gather_kv": lambda: _plain(
        paged_gather_kv_fn,
        _xavier(PAGE_POOL, PAGE_SIZE, PAGE_HS),
        _xavier(PAGE_POOL, PAGE_SIZE, PAGE_HS, seed=1),
        paged_gather_kv_inputs()[2],
        _xavier(PAGE_LQ, PAGE_HS, seed=2),
    ),
    "paged_gather_nested": lambda: _plain(
        paged_gather_nested_fn,
        _xavier(PAGE_POOL, PAGE_SIZE, PAGE_HS),
        paged_gather_inputs()[1],
        _xavier(PAGE_LQ, PAGE_HS, seed=1),
    ),
    "moe_tile_experts": lambda: _plain(
        moe_tile_experts_fn,
        shaped_randn(1, 64, 128),
        shaped_randn(4, 128, 64, seed=1),
    ),
    "view_transpose_head_tiled": lambda: _plain(
        lambda x, y: view_transpose_head_tiled_fn(x, y, 4, 64),
        shaped_randn(1, 128, 256),
        shaped_randn(1, 4, 128, 64, seed=1),
    ),
    "unsqueeze_broadcast_tiled_leading": lambda: _plain(
        lambda a, b: unsqueeze_broadcast_tiled_fn(a, b, True),
        shaped_randn(256, 128),
        shaped_randn(4, 256, 128, seed=1),
    ),
    "matmul_x_yt_nested": lambda: _plain(
        matmul_x_yt_nested_fn, shaped_randn(128, 256), shaped_randn(128, 256, seed=1)
    ),
    "nested_add_outer_row_inner_col": lambda: _plain(
        lambda a, b: nested_add_outer_row_inner_col_fn(a, b, 128, 128),
        shaped_randn(STICK_ROWS, STICK_COLS),
        shaped_randn(STICK_ROWS, STICK_COLS, seed=1),
    ),
    "softmax_row_tiled": lambda: _plain(
        lambda x: softmax_row_tiled_fn(x, 128), shaped_randn(STICK_ROWS, STICK_COLS)
    ),
}

# Fixtures that must yield at least one non-zero kernel advance; a fixture
# that silently stopped stamping an advancing record would otherwise make the
# comparison vacuous.
MUST_ADVANCE = set(CORPUS)


def _kernel_read_dep(kernel, ir_node, name):
    """The dep ``_general_tile_advance`` is about to use for read ``name``.

    The kernel's own rule (name resolved through ``mutation_real_name``, then
    the next unconsumed match), read from its state before the call.
    """
    reads = [d for d in ir_node.get_read_writes().reads if isinstance(d, MemoryDep)]
    scheduler = getattr(V.graph, "scheduler", None)
    real = scheduler.mutation_real_name if scheduler is not None else {}
    matching = [d for d in reads if real.get(d.name, d.name) == name]
    consumed = kernel._general_tile_advance_seen.get(name, 0)
    return matching[consumed] if consumed < len(matching) else None


class _Recorder:
    def __init__(self):
        self.compared = []  # (op, kind, name, kernel_expr, accessor)
        self.mismatches = []
        self.anywhere_mismatches = []
        # Records the record-based *_anywhere calls advancing although every
        # per-level term is zero (or the reverse).  Reported, not asserted:
        # the accessors follow the existing record-based readers.
        self.term_vs_record = []

    def check(self, kernel, tensor, is_input, name, result):
        ir_node = kernel.current_node.node
        op_name = ir_node.get_operation_name()
        if getattr(ir_node, "loop_info", None) is None:
            assert result is None
            dep = _kernel_read_dep(kernel, ir_node, name) if is_input else None
            assert (
                read_advance(ir_node, dep) if dep else output_advance(ir_node)
            ) == ()
            return
        if is_input:
            dep = self._pending_dep
            if dep is None:
                assert result is None
                return
            adv = read_advance(ir_node, dep)
            ours_any = read_advances_anywhere(ir_node, dep)
            # The reader memoizes read_writes on the op; do not leave that
            # behind on an op the kernel is still generating code for.
            had_cache = "_ts_cached_read_writes" in ir_node.__dict__
            theirs_any = _is_read_advancing_anywhere(
                dep.name, {dep.name: [(ir_node, dep)]}
            )
            if not had_cache:
                ir_node.__dict__.pop("_ts_cached_read_writes", None)
        else:
            adv = output_advance(ir_node)
            ours_any = output_advances_anywhere(ir_node)
            theirs_any = _is_tiled_advancing(ir_node)
        kind = "read" if is_input else "write"
        # Project the accessor's per-level host advances with the kernel's own
        # level symbols; a level the kernel skipped never minted a symbol, so
        # mint the name the kernel would have used.
        syms = [
            kernel._tile_advance_symbols.get(
                lvl, sympy.Symbol(f"_tile_adv_{op_name}_lvl{lvl}")
            )
            for lvl in range(len(adv))
        ]
        host = sum((a * s for a, s in zip(adv, syms)), sympy.S.Zero)
        layout = tensor.layout.device_layout
        ours = tiling_expr_to_device_expr(
            layout.device_size, layout.stride_map, sympy.sympify(host)
        )
        theirs = sympy.S.Zero if result is None else result
        record = (op_name, kind, name, theirs, adv)
        self.compared.append(record)
        if sympy.simplify(ours - theirs) != 0:
            self.mismatches.append((*record, ours))
        if ours_any != theirs_any:
            self.anywhere_mismatches.append((op_name, kind, name, ours_any, theirs_any))
        if any(a != 0 for a in adv) != ours_any:
            self.term_vs_record.append((op_name, kind, name, adv, ours_any))


def _compile_and_compare(case):
    fn, args, ctx = CORPUS[case]()
    rec = _Recorder()
    original = SpyreKernel._general_tile_advance

    def wrapper(kernel, tensor, is_input, name):
        if is_input:
            rec._pending_dep = _kernel_read_dep(kernel, kernel.current_node.node, name)
        result = original(kernel, tensor, is_input, name)
        rec.check(kernel, tensor, is_input, name, result)
        return result

    torch._dynamo.reset()
    with (
        mock.patch.object(
            SpyreKernel, "_general_tile_advance", autospec=True, side_effect=wrapper
        ),
        torch._inductor.config.patch(force_disable_caches=True),
        ctx if ctx is not None else contextlib.nullcontext(),
    ):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled(*(a.to(DEVICE_NAME) for a in args))
    return rec


@pytest.mark.parametrize("case", sorted(CORPUS))
def test_corpus_matches_kernel(case):
    rec = _compile_and_compare(case)
    advancing = [r for r in rec.compared if r[3] != 0]
    n_reads = sum(1 for r in rec.compared if r[1] == "read")
    n_writes = sum(1 for r in rec.compared if r[1] == "write")
    print(
        f"\nCORPUS {case}: compared {n_reads} reads + {n_writes} writes, "
        f"{len(advancing)} advancing (kernel expr != 0)"
    )
    for r in advancing:
        print(f"  {r[0]} {r[1]} {r[2]}: kernel={r[3]} accessor={r[4]}")
    for r in rec.term_vs_record:
        print(f"  TERM-VS-RECORD {r[0]} {r[1]} {r[2]}: terms={r[3]} record_any={r[4]}")
    assert rec.compared, f"{case}: the kernel built no TensorArg with a loop_info"
    if case in MUST_ADVANCE:
        assert advancing, f"{case}: no advancing record; comparison is vacuous"
    assert not rec.mismatches, "accessor != kernel:\n" + "\n".join(
        f"  {m[0]} {m[1]} {m[2]}: kernel={m[3]} accessor={m[4]} projected={m[5]}"
        for m in rec.mismatches
    )
    assert not rec.anywhere_mismatches, "anywhere != record reader:\n" + "\n".join(
        f"  {m[0]} {m[1]} {m[2]}: accessor={m[3]} existing={m[4]}"
        for m in rec.anywhere_mismatches
    )

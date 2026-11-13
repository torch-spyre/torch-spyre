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

"""Loop terms in the index of span-overflow groups (stage 7a.2).

The unit tests cover the step rule and the loop-variable minting. The device tests
add the term to the legacy lowering's result and check that the kernel reads the
same advance from the index as the legacy record holds (``LoopAdvanceMismatch``
otherwise), with the same numerics and the same kernel text.
"""

import pytest
import sympy
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.wsr.span_overflow_loop_terms import (
    _access_term,
    _plan_carries,
    _taken_symbols,
    access_step,
    add_loop_terms,
    mint_induction,
)

d0, d1 = sympy.symbols("d0 d1")
u0, u1, u2, u3 = (sympy.Symbol(f"u{n}") for n in range(4))


def test_mint_skips_taken_u_names():
    assert mint_induction([u0, u2]).name == "u1"
    assert mint_induction([u0, u1]).name == "u2"
    assert mint_induction([]).name == "u0"


def test_mint_counts_a_symbol_inside_an_expression():
    assert mint_induction([3 * u1 + d0, 7 * u0]).name == "u2"


def test_mint_ignores_names_that_are_not_loop_variables():
    assert mint_induction([sympy.Symbol("u"), sympy.Symbol("us0"), d0]).name == "u0"


def test_mint_is_integer_nonnegative():
    symbol = mint_induction([])
    assert symbol.is_integer and symbol.is_nonnegative
    assert sympy.floor(8 * symbol) == 8 * symbol


def test_step_uses_the_index_coefficient():
    assert access_step(1024 * d0 + d1, d1, sympy.Integer(512), 99) == 512
    assert access_step(1024 * d0 + d1, d0, sympy.Integer(2048), 99) == 2048 * 1024


def test_step_uses_the_stride_when_the_dim_is_squeezed():
    assert access_step(512 * d1, None, sympy.Integer(1), 32768) == 32768
    # a symbol that is not in the index is the same case
    assert access_step(512 * d1, d0, sympy.Integer(1), 32768) == 32768


def test_a_non_affine_index_is_unsupported():
    # floor(d0/2) holds d0 but has no coefficient for it; reading 0 would drop the term
    with pytest.raises(Unsupported, match="not affine"):
        access_step(sympy.floor(d0 / 2) + d1, d0, sympy.Integer(4), 1)


def test_a_symbolic_coefficient_is_unsupported():
    # coeff(d0) of d0*d1 is d1: not zero, but not a number either
    with pytest.raises(Unsupported, match="numeric coefficient"):
        access_step(d0 * d1, d0, sympy.Integer(4), 1)
    with pytest.raises(Unsupported, match="numeric coefficient"):
        access_step(d0 * d1 + d0, d0, sympy.Integer(4), 1)


def _graph_with(*, var_to_range=(), inputs=(), operations=()):
    from types import SimpleNamespace

    return SimpleNamespace(
        sizevars=SimpleNamespace(
            shape_env=SimpleNamespace(var_to_range=dict.fromkeys(var_to_range, 1))
        ),
        graph_inputs={
            f"arg{i}": SimpleNamespace(get_size=lambda s=s: s)
            for i, s in enumerate(inputs)
        },
        operations=list(operations),
    )


def test_mint_avoids_every_symbol_the_shape_env_allocated():
    # an unbacked scalar that appears in no index (a data-dependent size, say)
    graph = _graph_with(var_to_range=[u0, u1])
    assert mint_induction(_taken_symbols(graph, [])).name == "u2"


def test_taken_symbols_reads_graph_input_and_op_sizes():
    from types import SimpleNamespace

    rw = SimpleNamespace(reads=[], writes=[])
    op = SimpleNamespace(
        get_size=lambda: [u1, 8],
        get_read_writes=lambda: rw,
        get_operation_name=lambda: "op0",
    )
    graph = _graph_with(inputs=[[u0, 4]], operations=[op])
    names = {
        s.name
        for item in _taken_symbols(graph, [op])
        for s in getattr(item, "free_symbols", ())
    }
    assert {"u0", "u1"} <= names


def test_unreadable_reads_and_writes_convert_nothing():
    """A name that was not seen could be handed out twice, so when an op's reads and
    writes cannot be read the pass converts no group at all."""
    from types import SimpleNamespace

    def broken():
        raise RuntimeError("no read/write view")

    op = SimpleNamespace(get_read_writes=broken, get_operation_name=lambda: "op0")
    graph = _graph_with(operations=[op])
    with pytest.raises(Unsupported, match="op0"):
        _taken_symbols(graph, [op])
    assert add_loop_terms(graph, [([op], [(0, 2)])], 0) == {}


def test_a_pair_whose_symbol_is_not_in_the_index_does_not_advance():
    """A read that does not depend on the tiled dim (a broadcast) advances by
    nothing along it, as the stored record does; a buffer stride would be wrong
    whenever the op's dims are not the buffer's."""
    from types import SimpleNamespace

    from torch._inductor.utils import sympy_index_symbol

    i0, i1 = sympy_index_symbol("d0"), sympy_index_symbol("d1")
    op = SimpleNamespace(
        data=SimpleNamespace(ranges=[sympy.Integer(2048), sympy.Integer(512)])
    )
    graph = SimpleNamespace()  # never asked for a buffer: no stride is needed
    broadcast = SimpleNamespace(index=i1)  # no d0
    assert _access_term(graph, op, broadcast, "b", [(0, 2048)], []) == 0
    plain = SimpleNamespace(index=512 * i0 + i1)
    assert _access_term(graph, op, plain, "a", [(0, 2048)], []) == 512 * 2048


# --- the carry declaration, on stub ops (no device) ----------------------------


def _carry_member(
    name, data_cls, *, target=None, reads=(), writes=(), kind="reduction"
):
    """A group member with just what ``_plan_carries`` reads."""
    from types import SimpleNamespace

    from torch._inductor.ir import MutationLayoutSHOULDREMOVE

    layout = object()
    if target is not None:
        layout = mock.MagicMock(spec=MutationLayoutSHOULDREMOVE)
        layout.get_buffer.return_value = target
    return SimpleNamespace(
        data=mock.MagicMock(spec=data_cls),
        layout=layout,
        loop_info=SimpleNamespace(propagation=SimpleNamespace(kind=kind)),
        get_read_writes=lambda: SimpleNamespace(
            reads=[SimpleNamespace(name=n) for n in reads],
            writes=[SimpleNamespace(name=n) for n in writes],
        ),
        get_name=lambda: name,
        get_operation_name=lambda: name,
    )


def _carry_storage(name, **attrs):
    from types import SimpleNamespace

    return SimpleNamespace(get_name=lambda: name, **attrs)


def _carry_graph(*ops):
    from types import SimpleNamespace

    return SimpleNamespace(operations=list(ops))


def _k_tiled_group(acc="buf1", suffix=""):
    from torch._inductor.ir import Pointwise, Reduction

    storage = _carry_storage(acc)
    partial = _carry_member(
        f"op0{suffix}", Reduction, reads=["a", "b"], writes=[f"scratch{suffix}"]
    )
    combine = _carry_member(
        f"combine{suffix}",
        Pointwise,
        target=storage,
        reads=[f"scratch{suffix}", acc],
        writes=[f"combine{suffix}"],
    )
    return storage, partial, combine


def test_a_reduction_group_declares_one_carry_naming_its_storage_and_update():
    storage, partial, combine = _k_tiled_group()
    (edit,) = _plan_carries(_carry_graph(storage, partial, combine), [partial, combine])
    assert edit.update is combine and edit.storage is storage
    assert (edit.record.storage_name, edit.record.update_name) == ("buf1", "combine")
    assert edit.record.loop_origin is None


def test_a_group_that_is_not_a_reduction_declares_nothing():
    storage, partial, combine = _k_tiled_group()
    for op in (partial, combine):
        op.loop_info.propagation.kind = "copy_out"
    assert (
        _plan_carries(_carry_graph(storage, partial, combine), [partial, combine]) == []
    )


def test_two_reductions_in_one_group_declare_two_distinct_carries():
    s1, p1, c1 = _k_tiled_group("buf1", "_a")
    s2, p2, c2 = _k_tiled_group("buf2", "_b")
    edits = _plan_carries(_carry_graph(s1, s2, p1, c1, p2, c2), [p1, c1, p2, c2])
    assert [(e.record.storage_name, e.record.update_name) for e in edits] == [
        ("buf1", "combine_a"),
        ("buf2", "combine_b"),
    ]
    assert edits[0].record is not edits[1].record


@pytest.mark.parametrize(
    "case, match",
    [
        ("partial_without_a_combine", "partial reductions and"),
        ("another_op_touches_the_accumulator", "also touch the accumulator"),
        ("storage_already_declares_a_carry", "already declares a carry"),
        ("storage_is_not_in_the_graph", "is not an operation of the graph"),
    ],
)
def test_a_reduction_group_that_cannot_declare_its_carry_is_left_whole(case, match):
    from torch._inductor.ir import Pointwise

    storage, partial, combine = _k_tiled_group()
    members = [partial, combine]
    graph = _carry_graph(storage, partial, combine)
    if case == "partial_without_a_combine":
        members = [partial]
    elif case == "another_op_touches_the_accumulator":
        extra = _carry_member(
            "reader", Pointwise, reads=["buf1"], writes=["out"], kind="loop_internal"
        )
        members = [partial, combine, extra]
    elif case == "storage_already_declares_a_carry":
        storage._loop_carry_record = object()
    elif case == "storage_is_not_in_the_graph":
        graph = _carry_graph(partial, combine)
    with pytest.raises(Unsupported, match=match):
        _plan_carries(graph, members)


def test_a_storage_that_is_a_loop_member_is_left_whole():
    from types import SimpleNamespace

    storage, partial, combine = _k_tiled_group()
    storage.loop_info = SimpleNamespace(loop_group_id=(3,), loop_count=[4])
    # a ComputedBuffer in a counted loop has a loop path; the stub stands for one
    with mock.patch(
        "torch_spyre._inductor.wsr.span_overflow_loop_terms.counted_loop_path",
        lambda op: (3,) if op is storage else (),
    ):
        with pytest.raises(Unsupported, match="member of the loop"):
            _plan_carries(_carry_graph(storage, partial, combine), [partial, combine])


def test_a_symbolic_extent_is_unsupported():
    with pytest.raises(Unsupported, match="not an integer"):
        access_step(d0, d0, sympy.Symbol("s0"), 1)


# ---------------------------------------------------------------------------
# Device tests: the term is in the index and the kernel agrees with the record.
# ---------------------------------------------------------------------------

from unittest import mock  # noqa: E402

import regex  # noqa: E402
from torch._inductor.dependencies import MemoryDep  # noqa: E402
from torch._inductor.ir import Pointwise  # noqa: E402

from torch_spyre._inductor import config, passes  # noqa: E402
from torch_spyre._inductor.wsr import coarse_tile  # noqa: E402
from torch_spyre._inductor.wsr import span_overflow_loop_terms as slt  # noqa: E402

from test_span_overflow_gate import (  # noqa: E402
    CASES,
    _PERMUTE_SHAPE,
    _forced_one_level,
    _permuted,
    _relative_error,
    _shaped,
    compile_span_overflow,
    normalise,
)
from utils_inductor import expects_raise  # noqa: E402

# The gate's cases that write each tile through a copy-out, and all of them (the
# K-tiled matmul carries an accumulator instead, stage 7a.3).
COPY_OUT_CASES = sorted(set(CASES) - {"matmul_k_tiled"})
ALL_CASES = sorted(CASES)


class _Looped:
    """What the lowering left on one looped op, read right after the transform."""

    def __init__(self, op):
        self.name = op.get_operation_name()
        reads_writes = op.get_read_writes()
        self.reads = {
            d.name: str(d.index) for d in reads_writes.reads if isinstance(d, MemoryDep)
        }
        self.writes = {
            d.name: str(d.index)
            for d in reads_writes.writes
            if isinstance(d, MemoryDep)
        }
        self.loop_vars = [
            h.loop_var
            for h in getattr(op, "dim_hints", None) or ()
            if h.loop_var_range is not None
        ]
        self.group = op.loop_info.loop_group_id[0]


def _recording(run, *, terms=True):
    """Run ``run()`` (a compile) with a spy on the transform. Returns
    ``(result, [_Looped])``; ``terms=False`` switches the new pass off, which gives
    the legacy lowering to compare against."""
    seen: list[_Looped] = []
    original = coarse_tile.coarse_tile_post_stickify

    def spy(graph, groups, group_idx_offset=0):
        result = original(graph, groups=groups, group_idx_offset=group_idx_offset)
        seen.extend(
            _Looped(op)
            for op in graph.operations
            if getattr(op, "loop_info", None) is not None
            and hasattr(op, "data")
            and op.loop_info.loop_group_id[0] >= group_idx_offset
        )
        return result

    patches = [
        mock.patch.object(coarse_tile, "coarse_tile_post_stickify", spy),
        mock.patch.object(passes, "coarse_tile_post_stickify", spy),
    ]
    if not terms:
        patches.append(
            mock.patch.object(coarse_tile, "add_loop_terms", lambda *a, **k: {})
        )
    with contextlib_exit_stack(patches):
        return run(), seen


def contextlib_exit_stack(patches):
    import contextlib

    stack = contextlib.ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


def _compile_case(case, **kwargs):
    fn, shapes, limit, *_ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    compiled, looped = _recording(
        lambda: compile_span_overflow(fn, args, limit), **kwargs
    )
    return compiled, looped, fn, args


_STEP_SYMBOLS = (
    (r"_tile_adv_\w+?_lvl\d+", "SYM"),
    (r"\bu\d+\b", "SYM"),
    (r"floor\(([^()]*SYM)\)", r"\1"),
    # A comment in the generated file: the fused kernel's list of fx placeholders. The
    # full output buffer the copy-out mutates (``empty_default``) is listed as an input
    # of the flat loop, and not of the LoopSchedulerNode, which orders itself after the
    # allocation through a dependency it carries itself. No kernel code differs.
    (
        r"(?m)^#\s+%empty_default\w* : Tensor .*PlaceHolder\[target=empty_default\w*\]\n",
        "",
    ),
)


def _normalised(code):
    code = normalise(code)
    for pattern, replacement in _STEP_SYMBOLS:
        code = regex.sub(pattern, replacement, code)
    return code


@pytest.mark.parametrize("case", COPY_OUT_CASES)
def test_every_looped_op_lists_the_loop_variable_and_some_access_carries_it(case):
    compiled, looped, fn, args = _compile_case(case)
    assert looped, "the transform formed no loop"
    for op in looped:
        assert len(op.loop_vars) == 1, (op.name, op.loop_vars)
    variables = {op.loop_vars[0] for op in looped if op.group == looped[0].group}
    assert len(variables) == 1
    name = next(iter(variables)).name
    carrying = [
        op.name
        for op in looped
        if any(
            regex.search(rf"\b{name}\b", index)
            for index in (*op.reads.values(), *op.writes.values())
        )
    ]
    assert carrying, (case, [(o.name, o.reads, o.writes) for o in looped])
    assert _relative_error(compiled.outputs, fn(*[a.float() for a in args])) < 2e-2


@pytest.mark.parametrize(
    "case, read_step, write_step",
    [
        ("pointwise_copy_out", 512, 512),
        ("matmul_output_tiled_m", 128, 128),
        ("matmul_output_tiled_n", 4096, 4096),
    ],
)
def test_the_step_in_the_index_is_the_tile_advance(case, read_step, write_step):
    _, looped, _, _ = _compile_case(case)
    reads = [i for op in looped for i in op.reads.values()]
    writes = [i for op in looped for i in op.writes.values()]
    assert any(regex.search(rf"\b{read_step}\*u\d+\b", i) for i in reads), reads
    assert any(regex.search(rf"\b{write_step}\*u\d+\b", i) for i in writes), writes


@pytest.mark.parametrize("case", ALL_CASES)
def test_the_kernel_text_equals_the_legacy_lowerings_modulo_names(case):
    with_terms, _, _, _ = _compile_case(case)
    legacy, _, _, _ = _compile_case(case, terms=False)
    assert _normalised(with_terms.code) == _normalised(legacy.code)
    assert with_terms.groups == legacy.groups
    assert with_terms.added == legacy.added


@pytest.mark.parametrize("dim, step", [(0, 32768), (1, 262144)])
def test_a_permuted_buffer_tiled_eight_ways_uses_its_own_stride(dim, step):
    """Dim 0 is squeezed to extent one, so no symbol survives in the index: the
    step is the buffer's stride (32768), not the product of the sizes to its right
    (262144), which is what #6093 wrote."""
    (outcome, looped) = _recording(
        lambda: _forced_one_level(_PERMUTE_SHAPE, [(dim, 8)], _permuted)
    )
    wrong, loops, counts = outcome
    assert (loops, counts) == (1, [8]), (loops, counts)
    assert wrong == 0.0, wrong
    indexes = [i for op in looped for i in (*op.reads.values(), *op.writes.values())]
    assert any(regex.search(rf"\b{step}\*u\d+\b", i) for i in indexes), indexes


def test_a_k_tiled_matmul_carries_the_loop_in_the_index():
    """Both reads advance along the REDUCTION dim (the K tile), by the coefficient
    times the tile extent; the combine and the scratch partial do not advance, so
    there is no write term."""
    compiled, looped, _, _ = _compile_case("matmul_k_tiled")
    assert compiled.groups == [(1, ["Reduction"], [8])]
    assert looped and all(len(op.loop_vars) == 1 for op in looped), looped
    reads = {n: i for op in looped for n, i in op.reads.items() if n.startswith("arg")}
    assert regex.search(r"\b256\*u\d+\b", reads["arg0_1"]), reads
    assert regex.search(r"\b65536\*u\d+\b", reads["arg1_1"]), reads
    for op in looped:
        assert not any(regex.search(r"\bu\d+\b", i) for i in op.writes.values()), (
            op.name
        )


@pytest.fixture
def legacy_lowering():
    """These tests exercise the legacy transform's own internals (``add_loop_terms``,
    the group left whole): they run under the legacy lowering, and go with it."""
    with config.patch({"loop_tiling_lowering": "legacy"}):
        yield


@expects_raise
@pytest.mark.usefixtures("legacy_lowering")
def test_a_later_pass_that_rebuilds_the_op_data_fails_loudly():
    """The term lives in wrappers on ``op.data``. A pass that rebuilt the data from
    ``inner_fn`` and ``ranges`` would lose the write's wrapper and compile a kernel
    that writes every tile to tile zero. The kernel's check must raise."""
    original = coarse_tile.add_loop_terms

    def add_then_drop_the_write_wrapper(graph, groups, group_idx_offset=0):
        result = original(graph, groups, group_idx_offset)
        for op in graph.operations:
            if op.get_operation_name().startswith("coarse_tile_copy_"):
                op.data = Pointwise(
                    device=op.data.get_device(),
                    dtype=op.data.get_dtype(),
                    inner_fn=op.data.inner_fn,
                    ranges=list(op.data.ranges),
                )
        return result

    fn, shapes, limit, *_ = CASES["pointwise_copy_out"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with (
        mock.patch.object(
            coarse_tile, "add_loop_terms", add_then_drop_the_write_wrapper
        ),
        pytest.raises(Exception, match="LoopAdvanceMismatch"),
    ):
        compile_span_overflow(fn, args, limit)


@pytest.mark.usefixtures("legacy_lowering")
def test_the_memoized_read_writes_see_the_term():
    """Passes ask for reads and writes through ``pass_utils.op_read_writes``, which
    memoizes per op and is never refreshed by itself. The memo is already filled by
    the time the term is added, so the pass must invalidate it: a pass that kept
    reading the old index would not see the loop."""
    from torch_spyre._inductor.pass_utils import op_read_writes

    original = coarse_tile.add_loop_terms
    memoized: dict[str, str] = {}

    def fill_the_memo_then_add(graph, groups, group_idx_offset=0):
        for op in graph.operations:
            if getattr(op, "loop_info", None) is not None and hasattr(op, "data"):
                op_read_writes(op)
        result = original(graph, groups, group_idx_offset)
        for op in graph.operations:
            if getattr(op, "loop_info", None) is not None and hasattr(op, "data"):
                reads_writes = op_read_writes(op)
                memoized[op.get_operation_name()] = " ".join(
                    str(d.index)
                    for d in (*reads_writes.reads, *reads_writes.writes)
                    if isinstance(d, MemoryDep)
                )
        return result

    fn, shapes, limit, *_ = CASES["pointwise_copy_out"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with mock.patch.object(coarse_tile, "add_loop_terms", fill_the_memo_then_add):
        compile_span_overflow(fn, args, limit)
    assert memoized
    assert any(regex.search(r"\bu\d+\b", text) for text in memoized.values()), memoized


def test_a_for_each_tile_loop_and_a_span_overflow_group_have_distinct_variables():
    """Both families are named ``u<n>``: the for_each_tile loop's is an unbacked
    scalar Inductor allocated, the span-overflow one is minted here. A graph with
    both must not give two loops one name."""
    from for_each_tile_fixtures import split_m_fn

    def fn(x, y, a, b):
        return split_m_fn(x, y), a + b

    shapes = [(256, 256), (256, 64), (4096, 1024), (4096, 1024)]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    seen: dict[int, set[str]] = {}
    original = coarse_tile.coarse_tile_post_stickify

    def spy(graph, groups, group_idx_offset=0):
        result = original(graph, groups=groups, group_idx_offset=group_idx_offset)
        for op in graph.operations:
            info = getattr(op, "loop_info", None)
            if info is None:
                continue
            for hint in getattr(op, "dim_hints", None) or ():
                if hint.loop_var_range is not None:
                    seen.setdefault(info.loop_group_id[0], set()).add(
                        hint.loop_var.name
                    )
        return result

    with (
        mock.patch.object(coarse_tile, "coarse_tile_post_stickify", spy),
        mock.patch.object(passes, "coarse_tile_post_stickify", spy),
    ):
        compiled = compile_span_overflow(fn, args, 4 * 1024 * 1024)
    assert compiled.groups, "no span-overflow group formed"
    assert len(seen) >= 2, seen  # the for_each_tile loop and the span-overflow group
    assert all(len(names) == 1 for names in seen.values()), seen
    names = [next(iter(v)) for v in seen.values()]
    assert len(set(names)) == len(names), seen


@pytest.mark.parametrize("sencores", [1, 4, 32])
@pytest.mark.parametrize(
    "case",
    ["pointwise_copy_out", "matmul_output_tiled_n", "matmul_pointwise_consumer"],
)
def test_the_term_survives_every_core_split(case, sencores):
    """Work division may split the tiled dim across cores. The loop variable is
    pinned to zero in the base coordinates, so a split must neither drop the term
    nor apply it twice: the result matches fp32 and the kernel's check is silent."""
    from torch_spyre._inductor import config

    fn, shapes, limit, *_ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with config.patch({"sencores": sencores}):
        compiled, looped = _recording(lambda: compile_span_overflow(fn, args, limit))
    assert looped
    assert all(len(op.loop_vars) == 1 for op in looped)
    # work division really split the op across cores when there are several, and not
    # when there is one, so the pinned loop variable is exercised under a split
    assert ("(core_id" in compiled.code) == (sencores > 1), sencores
    assert _relative_error(compiled.outputs, fn(*[a.float() for a in args])) < 2e-2


# ---------------------------------------------------------------------------
# The converted groups become LoopNodes (stage 7a.2, Task 5).
# ---------------------------------------------------------------------------


def _nodes_of(operations):
    from torch_spyre._inductor.loop_node import LoopNode

    for op in operations:
        if isinstance(op, LoopNode):
            yield op
            yield from _nodes_of(op.body)


def _compile_seeing_nodes(fn, args, limit):
    """Compile, and return ``(compiled, nodes, violations)`` as the pass loop leaves
    the operations just before the scheduler: every ``LoopNode`` (nested ones
    included) with the names of its leaf ops, and what the loop checker finds."""
    from torch_spyre._inductor.loop_checker import find_violations
    from torch_spyre._inductor.loop_node import all_ops

    seen: dict = {}
    original = passes._hand_off_loop_nodes

    def spy(graph):
        seen["nodes"] = [
            (
                node.get_name(),
                node.induction,
                node.trip_count,
                [o.get_operation_name() for o in all_ops(node.body)],
            )
            for node in _nodes_of(graph.operations)
        ]
        seen["violations"] = find_violations(list(graph.operations))
        return original(graph)

    with mock.patch.object(passes, "_hand_off_loop_nodes", spy):
        compiled = compile_span_overflow(fn, args, limit)
    return compiled, seen["nodes"], seen["violations"]


@pytest.mark.parametrize("case", ALL_CASES)
def test_a_span_overflow_group_is_one_loop_node(case):
    fn, shapes, limit, groups, _ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    compiled, nodes, violations = _compile_seeing_nodes(fn, args, limit)
    assert len(nodes) == len(groups) == 1, nodes
    name, induction, trip, members = nodes[0]
    assert induction is not None and induction.name.startswith("u"), induction
    assert trip == groups[0][2][0], (trip, groups)
    # the tile ops and the copy-outs (or the combines) the transform added all sit in
    # the node
    assert any(
        m.startswith(("coarse_tile_copy_", "coarse_tile_combine_")) for m in members
    ), members
    assert len(members) >= groups[0][0] + 1, members
    assert violations == [], violations
    assert _relative_error(compiled.outputs, fn(*[a.float() for a in args])) < 2e-2


def _compile_seeing_carries(fn, args, limit):
    """Compile, and return ``(compiled, [(node name, [(storage, update, storage is a
    carry storage to the allocator, storage op has the record, update has it)])])``
    as the pass loop leaves the nodes just before the scheduler."""
    from torch_spyre._inductor.loop_node import all_ops
    from torch_spyre._inductor.scratchpad.allocator import _is_loop_carry_storage

    seen: dict = {}
    original = passes._hand_off_loop_nodes

    def spy(graph):
        live = {o.get_name(): o for o in all_ops(graph.operations)}
        seen["carries"] = [
            (
                node.get_name(),
                [
                    (
                        r.storage_name,
                        r.update_name,
                        _is_loop_carry_storage(live[r.storage_name]),
                        getattr(live[r.storage_name], "_loop_carry_record", None) is r,
                        getattr(live[r.update_name], "_loop_carry_record", None) is r,
                        r.loop_origin,
                    )
                    for r in node.carries
                ],
            )
            for node in _nodes_of(graph.operations)
        ]
        return original(graph)

    with mock.patch.object(passes, "_hand_off_loop_nodes", spy):
        compiled = compile_span_overflow(fn, args, limit)
    return compiled, seen["carries"]


def test_a_k_tiled_matmul_is_a_node_with_its_accumulator_declared_as_a_carry():
    fn, shapes, limit, *_ = CASES["matmul_k_tiled"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    compiled, carries = _compile_seeing_carries(fn, args, limit)
    assert len(carries) == 1, carries
    _, records = carries[0]
    assert len(records) == 1, records
    storage, update, is_storage, on_storage, on_update, origin = records[0]
    assert update.startswith("coarse_tile_combine_"), update
    assert is_storage and on_storage and on_update, records
    # a span-overflow loop has no FX while_loop: the post-loop drain plan declines
    # and the accumulator keeps its HBM behaviour
    assert origin is None, origin
    assert _relative_error(compiled.outputs, fn(*[a.float() for a in args])) < 2e-2


@pytest.mark.parametrize(
    "fn, name",
    [(lambda x: x.sum(dim=1), "sum"), (lambda x: x.amax(dim=1), "amax")],
    ids=["sum", "amax"],
)
def test_a_reduction_over_a_tiled_dim_declares_its_accumulator(fn, name):
    args = [_shaped((256, 16384), 0)]
    compiled, carries = _compile_seeing_carries(fn, args, 1024 * 1024)
    assert compiled.groups == [(1, ["Reduction"], [8])], compiled.groups
    assert len(carries) == 1 and len(carries[0][1]) == 1, carries
    assert carries[0][1][0][2:5] == (True, True, True), carries
    assert _relative_error(compiled.outputs, fn(args[0].float())) < 2e-2


@pytest.mark.parametrize(
    "case, fn, shapes",
    [
        ("k_tiled_matmul", CASES["matmul_k_tiled"][0], CASES["matmul_k_tiled"][1]),
        (
            "small_accumulator_read_later",
            lambda a, b, c: (a @ b) + c,
            [(256, 16384), (16384, 64), (256, 64)],
        ),
    ],
    ids=["k_tiled_matmul", "small_accumulator_read_later"],
)
def test_declaring_the_carry_does_not_change_the_kernel(case, fn, shapes):
    """The record is for what plans after the lowering; at these sizes the allocator
    makes the same choices with and without it, so the generated kernel is the same."""
    limit = CASES["matmul_k_tiled"][2] if case == "k_tiled_matmul" else 2 * 1024 * 1024
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with_record = compile_span_overflow(fn, args, limit)
    with mock.patch.object(slt, "_apply_carries", lambda edits: None):
        without = compile_span_overflow(fn, args, limit)
    assert _normalised(with_record.code) == _normalised(without.code)


def test_a_for_each_tile_node_and_a_span_overflow_node_in_one_graph():
    """The graph already holds a node when the span-overflow pass runs, so the pass
    sees the flat view of it; the new node is built after the pass, beside the
    existing one, with its own name and its own variable."""
    from for_each_tile_fixtures import split_m_fn

    def fn(x, y, a, b):
        return split_m_fn(x, y), a + b

    shapes = [(256, 256), (256, 64), (4096, 1024), (4096, 1024)]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    compiled, nodes, violations = _compile_seeing_nodes(fn, args, 4 * 1024 * 1024)
    assert len(nodes) == 2, nodes
    names = [n[0] for n in nodes]
    inductions = [n[1] for n in nodes]
    assert len(set(names)) == 2, names
    assert len({i for i in inductions if i is not None}) == 2, inductions
    assert violations == [], violations
    x, y, a, b = (t.float() for t in args)
    assert _relative_error(compiled.outputs, (x @ y, a + b)) < 2e-2


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#6243: span-overflow tiling returns wrong values for a transposed "
    "consumer of a tiled pointwise producer (about 60% of elements)",
)
def test_a_transposed_consumer_of_a_tiled_producer_is_right():
    fn = lambda a, b: (a + b).t() * 2  # noqa: E731
    args = [_shaped((4096, 1024), seed) for seed in range(2)]
    compiled = compile_span_overflow(fn, args, 4 * 1024 * 1024)
    reference = fn(*[a.float() for a in args])
    wrong = ((compiled.outputs[0] - reference).abs() > 0.05).float().mean().item()
    assert wrong == 0.0, wrong


def test_a_k_tiled_group_and_a_copy_out_group_become_two_nodes():
    """Two groups in one graph, each its own node with its own loop variable; only
    the K-tiled one declares a carry."""

    def fn(a, b, c, d):
        return a @ b, c + d

    shapes = [(2048, 2048), (2048, 256), (2048, 512), (2048, 512)]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    compiled, nodes, violations = _compile_seeing_nodes(fn, args, 1024 * 1024)
    assert compiled.groups == [(1, ["Reduction"], [8]), (1, ["Pointwise"], [2])]
    assert len(nodes) == 2, nodes
    assert sorted(n[2] for n in nodes) == [2, 8], nodes
    assert len({n[1] for n in nodes}) == 2 and None not in {n[1] for n in nodes}, nodes
    assert violations == [], violations
    a, b, c, d = (t.float() for t in args)
    assert _relative_error(compiled.outputs, (a @ b, c + d)) < 2e-2


def _compile_forced_view(view, levels):
    """Compile ``abs(view(x)) + 1`` with the span-overflow plan forced to ``levels``
    (over the view's shape), x a contiguous (64, 8, 512) fp16 input. Returns
    ``(fraction of elements wrong, [_Looped])``."""
    from test_span_overflow_hint_analysis import _forced_span_plan
    from torch._inductor.utils import run_and_get_code

    from torch_spyre._inductor import config
    from torch_spyre.constants import DEVICE_NAME

    torch.manual_seed(0xAFFE)
    x = torch.randn(64, 8, 512, dtype=torch.float16)
    fn = lambda x: torch.abs(view(x)) + 1  # noqa: E731
    reference = fn(x.float())
    shape = tuple(view(x).shape)

    def run():
        torch._dynamo.reset()
        with (
            mock.patch(
                "torch_spyre._inductor.wsr.coarse_tile_span_overflow."
                "plan_span_overflow_tile",
                _forced_span_plan(shape, levels),
            ),
            config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
            torch._inductor.config.patch(force_disable_caches=True),
        ):
            out, _ = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), x.to(DEVICE_NAME)
            )
        return ((out.cpu().float() - reference).abs() > 0.05).float().mean().item()

    return _recording(run)


@pytest.mark.usefixtures("legacy_lowering")
def test_a_read_through_a_permuted_view_keeps_the_legacy_lowering():
    """The op iterates (8, 64, 512) over an input stored (64, 8, 512): its dim 0 is
    the buffer's dim 1. Tiled 8 ways on dim 0 that dim is squeezed to extent one, so
    its step cannot come from the index and the buffer's stride for 'dim 0' would be
    the wrong dim's (4096, not 512). The pass must see that and leave the group whole:
    it still compiles, still right, with no loop variable and no term on any op."""
    wrong, looped = _compile_forced_view(lambda t: t.permute(1, 0, 2), [(0, 8)])
    assert wrong == 0.0, wrong
    assert looped, "the forced plan formed no loop"
    for op in looped:
        assert op.loop_vars == [], (op.name, op.loop_vars)
        assert not any(
            regex.search(r"\bu\d+\b", i)
            for i in (*op.reads.values(), *op.writes.values())
        ), (op.name, op.reads, op.writes)


def test_a_read_through_a_plain_view_is_still_converted():
    """Control for the test above: the same forced plan on the input as stored
    (same dim order, a squeezed dim 0) is converted."""
    wrong, looped = _compile_forced_view(lambda t: t, [(0, 64)])
    assert wrong == 0.0, wrong
    assert looped and all(len(op.loop_vars) == 1 for op in looped), looped


@expects_raise
def test_an_unsupported_release_refuses_a_span_overflow_graph_with_a_clear_error():
    """A converted group is a LoopNode, so its graph needs a release the loop
    scheduler supports, as a for_each_tile graph does. On any other the failure must
    be the one clear Unsupported naming the release, not a late error in the
    generated wrapper.

    TODO: if span-overflow loops are ever lowered without a LoopNode on an
    unsupported release, this becomes a positive test."""
    from torch._inductor.exc import InductorError

    from torch_spyre._inductor import loop_scheduler

    fn, shapes, limit, *_ = CASES["pointwise_copy_out"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with (
        mock.patch.object(loop_scheduler, "loop_scheduler_supported", lambda: False),
        pytest.raises(
            InductorError, match=r"LoopNode scheduler patch supports only torch"
        ),
    ):
        compile_span_overflow(fn, args, limit)

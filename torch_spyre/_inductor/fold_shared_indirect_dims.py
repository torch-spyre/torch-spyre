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

"""Fold a loop dim shared by an index and its gathered tensor into the index (#4840).

An indirect load ``x[..., IndirectAccess(idx[b]), ..., b, ...]`` uses the loop
dim ``b`` twice: to read the index and to address ``x`` directly. That is what
``logits.gather(1, pos.view(-1, 1, 1).expand(-1, 1, V))`` lowers to, and any
other op producing the same load hits the same problem. An SDSC layout lists
each dim once, so ``b``'s direct role is dropped and the wrong rows are read.

When ``x``'s address is ``S * s + c * b + rest`` with ``c`` a multiple of ``S``,
the ``b`` term can move into the index: ``x[S * (s + (c // S) * b) + rest]``.
This pass, run before stickification so new buffers get layouts like any
user-written op, lowers

    offsets = arange(0, (c // S) * R, c // S)      # R = extent of b
    rows    = idx + offsets                        # broadcast along b's index dim

and wraps the op's ``inner_fn`` so it reads ``rows`` instead of ``idx`` and drops
``c * b`` from ``x``'s address. The result is a plain row gather whose index and
value dims are disjoint. Spyre cannot generate ``offsets`` on the device
(no ``index_expr``), so ``arange`` lowers to its CPU fallback.

Only Pointwise ops with exactly one indirect load are rewritten, and only one
shared dim is folded; anything else is left unchanged.
"""

import dataclasses

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation, Pointwise
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V

from .logging_utils import get_inductor_logger
from .pass_utils import (
    _find_indirect_index_bufs,
    concretize_expr,
    find_fx_node,
    patch_env,
    replace_computed_buffer_body,
)

logger = get_inductor_logger("fold_shared_indirect_dims")

aten = torch.ops.aten


@dataclasses.dataclass(frozen=True)
class _FoldPlan:
    """Numbers for one fold; coefficients, never symbols (those go stale)."""

    index_name: str  # index buffer read by the indirect load
    value_name: str  # gathered buffer
    index_coeff: int  # coefficient of the shared dim in the index load
    value_coeff: int  # coefficient of the shared dim in the value load
    step: int  # value_coeff // (coefficient of the indirect symbol)
    extent: int  # range of the shared dim
    index_dim: int  # dim of the index buffer the shared dim walks


def _linear_coeff(expr: sympy.Expr, sym: sympy.Symbol) -> int | None:
    """Integer coefficient of ``sym`` if ``expr`` is linear in it, else None."""
    coeff = expr.coeff(sym)
    if not coeff.is_Integer or (expr - coeff * sym).has(sym):
        return None
    return int(coeff)


def _dense_strides(size: list[int]) -> list[int]:
    strides, running = [0] * len(size), 1
    for d in range(len(size) - 1, -1, -1):
        strides[d] = running
        running *= size[d]
    return strides


def _plan_fold(graph: GraphLowering, op: ComputedBuffer) -> _FoldPlan | None:
    """Return a fold plan when ``op`` reads ``x[IA(idx[b]), b]``, else None."""
    if not isinstance(op.data, Pointwise):
        return None
    reads = [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]
    if not any(d.is_indirect() for d in reads):
        return None
    index_by_value, _ = _find_indirect_index_bufs(op)
    if len(index_by_value) != 1:
        return None
    ((value_name, index_name),) = index_by_value.items()
    value_deps = [d for d in reads if d.name == value_name]
    index_deps = [d for d in reads if d.name == index_name]
    if len(value_deps) != 1 or len(index_deps) != 1:
        return None
    value_dep, index_dep = value_deps[0], index_deps[0]

    indirect_syms = [
        s for s in value_dep.index.free_symbols if s not in value_dep.ranges
    ]
    if len(indirect_syms) != 1:
        return None
    indirect_coeff = _linear_coeff(value_dep.index, indirect_syms[0])
    if not indirect_coeff:
        return None

    # Fold only when each index step jumps whole rows of the value's last dim.
    # A smaller step picks elements inside a row, i.e. inside a stick (#4327):
    # folding does not make that a row gather, and its index add cannot be laid
    # out.
    last = concretize_expr(graph.get_buffer(value_name).get_size()[-1])
    if indirect_coeff % last:
        return None
    index_loop_syms = [s for s in index_dep.index.free_symbols if s in index_dep.ranges]
    shared = [s for s in index_loop_syms if value_dep.index.has(s)]
    if len(shared) != 1:
        return None
    sym = shared[0]
    value_coeff = _linear_coeff(value_dep.index, sym)
    index_coeff = _linear_coeff(index_dep.index, sym)
    if not value_coeff or not index_coeff or value_coeff % indirect_coeff:
        return None

    index_buf = graph.get_buffer(index_name)
    if index_buf.get_dtype() not in (torch.int32, torch.int64):
        return None
    size = [concretize_expr(s) for s in index_buf.get_size()]
    stride = [concretize_expr(s) for s in index_buf.get_stride()]
    # rows is a fresh dense buffer read at idx's own index, so idx must be dense.
    if stride != _dense_strides(size):
        return None
    extent = concretize_expr(index_dep.ranges[sym])
    dims = [d for d, st in enumerate(stride) if st == index_coeff and size[d] == extent]
    if len(dims) != 1:
        return None
    return _FoldPlan(
        index_name=index_name,
        value_name=value_name,
        index_coeff=index_coeff,
        value_coeff=value_coeff,
        step=value_coeff // indirect_coeff,
        extent=extent,
        index_dim=dims[0],
    )


def _index_fx_node(graph: GraphLowering, name: str) -> torch.fx.Node | None:
    """FX node whose lowered value is buffer ``name``, restoring env if needed.

    Inductor drops env entries after their last use, so refill it the way other
    passes do (``patch_env``). A graph input is not covered by that, so fall back
    to its placeholder node, whose target is the input's buffer name.
    """
    patch_env(graph)
    node = find_fx_node(name, graph)
    if node is None and name in graph.graph_inputs:
        node = next(
            (
                n
                for n in graph.graph.nodes
                if n.op == "placeholder" and n.target == name
            ),
            None,
        )
        if node is not None:
            graph.env[node] = graph.graph_inputs[name]
    return node


def _lower_rows(
    graph: GraphLowering, op: ComputedBuffer, plan: _FoldPlan
) -> tuple[str, list[Operation]] | None:
    """Lower ``rows = idx + arange(0, step * extent, step)`` before ``op``.

    Built from FX nodes and ``run_node``, like ``lower_pad_sequence``, so the
    ops lower exactly as a user-written ``arange`` and ``add`` would (``arange``
    takes Spyre's CPU fallback). Returns (rows buffer name, new operations).
    """
    index_fx = _index_fx_node(graph, plan.index_name)
    anchor = op.get_origin_node() or (next(iter(op.origins)) if op.origins else None)
    if index_fx is None or anchor is None or anchor.graph is not graph.graph:
        return None
    index_buf = graph.get_buffer(plan.index_name)
    dtype, device = index_buf.get_dtype(), index_buf.get_device()
    size = [concretize_expr(s) for s in index_buf.get_size()]
    offsets_shape = [1] * len(size)
    offsets_shape[plan.index_dim] = plan.extent

    fx_graph = graph.graph
    ops_before = len(graph.operations)
    with fx_graph.inserting_before(anchor):
        offsets_fx = fx_graph.create_node(
            "call_function",
            aten.arange.start_step,
            args=(0, plan.step * plan.extent, plan.step),
            kwargs={"dtype": dtype, "device": device},
        )
        offsets_fx.meta["val"] = torch.empty([plan.extent], dtype=dtype, device=device)
        view_fx = fx_graph.create_node(
            "call_function", aten.view.default, args=(offsets_fx, offsets_shape)
        )
        view_fx.meta["val"] = torch.empty(offsets_shape, dtype=dtype, device=device)
        rows_fx = fx_graph.create_node(
            "call_function", aten.add.Tensor, args=(index_fx, view_fx)
        )
        rows_fx.meta["val"] = torch.empty(size, dtype=dtype, device=device)
    for node in (offsets_fx, view_fx, rows_fx):
        graph.env[node] = graph.run_node(node)
    rows_tb = graph.env[rows_fx]
    rows_tb.realize()
    return rows_tb.get_name(), graph.operations[ops_before:]


class _FoldSharedDimHandler(WrapperHandler):
    """Read ``rows`` for ``idx`` and drop the shared dim from the value address.

    Wraps the live ``inner_fn`` instead of rebuilding it, since rebuilt index
    expressions go stale (#2797). The shared dim's symbol is taken from the
    index load as it happens, and only its term (a fixed coefficient) is
    subtracted. The index load always precedes the value load, since the value
    load needs ``indirect_indexing`` of the index.
    """

    def __init__(self, inner, plan: _FoldPlan, rows_name: str):
        super().__init__(inner)
        self._plan = plan
        self._rows_name = rows_name
        self._shared_sym = None
        self._rows_pending = False

    def load(self, name, index):
        plan = self._plan
        if name == plan.index_name:
            index = sympy.sympify(index)
            matches = [
                s
                for s in index.free_symbols
                if _linear_coeff(index, s) == plan.index_coeff
            ]
            self._shared_sym = matches[0] if len(matches) == 1 else None
            self._rows_pending = self._shared_sym is not None
            if self._rows_pending:
                name = self._rows_name
        elif name == plan.value_name and self._shared_sym is not None:
            index = index - plan.value_coeff * self._shared_sym
        return super().load(name, index)

    def indirect_indexing(self, index_var, size, check=True, wrap_neg=True):
        # The folded row index spans every block, not one block's rows.
        if self._rows_pending:
            size = size + self._plan.step * (self._plan.extent - 1)
            self._rows_pending = False
        return super().indirect_indexing(index_var, size, check, wrap_neg)


def _move_ops_before(
    operations: list[Operation], new_ops: list[Operation], anchor: Operation
) -> None:
    for o in new_ops:
        operations.remove(o)
    idx = operations.index(anchor)
    for i, o in enumerate(new_ops):
        operations.insert(idx + i, o)


def fold_shared_indirect_dims(graph: GraphLowering) -> None:
    """Fold each shared index/value loop dim into the index (see module doc)."""
    for op in list(graph.operations):
        if not isinstance(op, ComputedBuffer):
            continue
        plan = _plan_fold(graph, op)
        if plan is None:
            continue
        lowered = _lower_rows(graph, op, plan)
        if lowered is None:
            logger.debug("fold_shared_indirect_dims: cannot lower rows for %s", op)
            continue
        rows_name, new_ops = lowered
        _move_ops_before(graph.operations, new_ops, op)

        orig_inner = op.data.inner_fn

        def new_inner_fn(*args, _orig=orig_inner, _plan=plan, _rows=rows_name):
            with V.set_ops_handler(_FoldSharedDimHandler(V.ops, _plan, _rows)):
                return _orig(*args)

        object.__setattr__(op.data, "inner_fn", new_inner_fn)
        replace_computed_buffer_body(
            op,
            op.data,
            graph.operations,
            pass_name="fold_shared_indirect_dims",
            reason=f"fold shared dim into index {plan.index_name} -> {rows_name}",
        )
        logger.info(
            "fold_shared_indirect_dims: %s reads %s via %s (step %d, extent %d)",
            op.get_name(),
            plan.value_name,
            rows_name,
            plan.step,
            plan.extent,
        )

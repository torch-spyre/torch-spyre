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

"""Physical width selection for static dense, bias-free inference chains.

Logical tensor shapes, strides, parameter objects, and inner_fn expressions
are unchanged. Load-time Linear allocations provide certified zero regions;
this pass may select a larger iteration extent for a closed producer ->
pointwise -> contraction chain. Unsupported graphs keep logical iteration.

Projection padding is *not* declared zero merely because its weight is zero:
the supported pointwise operations use the existing SAMV input mask to seed
their tail operands to zero. This gives the final contraction a neutral tail
under the same backend mask contract already used for ordinary stick padding.
"""

from dataclasses import dataclass
import math

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    Reduction,
)
from torch._inductor.virtualized import V

from torch_spyre._C import ElementArrangement, SpyreTensorLayout

from . import config
from .constants import BATCH_MATMUL_OP
from .errors import Unsupported
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("dense_padding")

# Hardware units, not model widths: expose a four-way output-stick split.
_STICK_GROUP = 4
ZERO_PRESERVING_POINTWISE = frozenset({"silu", "mul", "add", "sub", "neg", "abs"})
ZERO_MASK_INFO_KEY = "dense_padding_zero_mask"


def padded_linear_layout(layout, host_size, dtype):
    """Enlarge only whole-stick, standard 2-D Linear allocations.

    Return None if the existing allocation is already suitable or unsupported.
    The returned geometry carries no zero certificate until fill + DMA finish.
    """
    if dtype not in (torch.float16, torch.bfloat16) or len(host_size) != 2:
        return None
    if layout.element_arrangement != ElementArrangement.STANDARD:
        return None
    out_size, in_size = map(int, host_size)
    eps = layout.elems_per_stick()
    if min(out_size, in_size) <= 0 or out_size % eps or in_size % eps:
        return None
    if list(layout.device_size) != [out_size // eps, in_size, eps]:
        return None
    quantum = _STICK_GROUP * eps
    padded_out = (out_size + quantum - 1) // quantum * quantum
    padded_in = (in_size + quantum - 1) // quantum * quantum
    dims = [padded_out // eps, padded_in, eps]
    if dims == list(layout.device_size):
        return None
    return SpyreTensorLayout(
        dims,
        list(layout.stride_map),
        layout.device_dtype,
        layout.element_arrangement,
    )


def _layout(op):
    layout = op.get_layout()
    # Mutation aliases are supported only for compiler-created identity
    # dump/restore copies after selection, never during chain discovery.
    if hasattr(layout, "real_layout"):
        layout = layout.real_layout()
    return layout


def _padding_extents(op):
    """Read explicit compiler annotations without probing a tensor layout.

    Extern operations can have MultiOutputLayout, whose get_layout() raises.
    Unannotated operations retain the ordinary logical iteration path.
    """
    layout = getattr(op, "layout", None)
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    return (
        getattr(layout, "__dict__", {}).get("dense_padding"),
        op.__dict__.get("dense_reduction_padding"),
    )


def has_dense_padding(op):
    return any(padding is not None for padding in _padding_extents(op))


def physical_iteration_space(op, rw, logical_space):
    """Apply explicit physical extents after logical address decomposition.

    Resolve symbols by the live dependency's coefficient and expected extent,
    so scheduler symbol renaming is harmless. A transformed/flattened domain
    that no longer proves the selected axis fails closed.
    """
    result = dict(logical_space)
    output_padding, reduction_padding = _padding_extents(op)
    if output_padding is None and reduction_padding is None:
        return result
    writes = [dep for dep in rw.writes if isinstance(dep, MemoryDep)]
    if len(writes) != 1:
        raise Unsupported("physical dense padding requires one tensor write")
    write = writes[0]
    for padding, reduction in ((output_padding, False), (reduction_padding, True)):
        if padding is None:
            continue
        logical, physical = padding
        symbols = [
            sym
            for sym, extent in logical_space.items()
            if extent == logical
            and (
                (sym not in write.index.free_symbols)
                if reduction
                else write.index.coeff(sym) == 1
            )
        ]
        if len(symbols) != 1:
            raise Unsupported("dense padding axis lost during graph transformation")
        result[symbols[0]] = sympy.Integer(physical)
    return result


def zero_mask_for_op(op, rw, logical_space):
    """Codegen-only tail bounds for the certified output-stick axis."""
    if op.__dict__.get("dense_padding_zero_mask") is not True:
        return {}
    physical = physical_iteration_space(op, rw, logical_space)
    changes = [
        (int(logical_space[s]), int(e))
        for s, e in physical.items()
        if e != logical_space[s]
    ]
    if len(changes) != 1:
        raise Unsupported("dense zero mask must have one physical output axis")
    logical, extent = changes[0]
    return {"logical": logical, "physical": extent}


def copy_padding_layout(source, destination):
    """Identity clones must copy the whole chosen physical domain."""
    destination.dense_padding = source.dense_padding


@dataclass(frozen=True)
class DenseMatmul:
    op: ComputedBuffer
    x_name: str
    weight_name: str
    m: int
    n: int
    k: int
    m_symbol: sympy.Symbol
    n_symbol: sympy.Symbol
    k_symbol: sympy.Symbol


def _dense_shape(op):
    layout = _layout(op)
    if not isinstance(layout, FixedTiledLayout) or layout.offset != 0:
        return None
    if layout.dtype not in (torch.float16, torch.bfloat16):
        return None
    if len(layout.size) < 2 or any(
        not sympy.sympify(x).is_Integer for x in layout.size
    ):
        return None
    full_size = tuple(map(int, layout.size))
    size = full_size[-2:]
    if (
        any(x != 1 for x in full_size[:-2])
        or min(size) <= 1
        or list(layout.stride[-2:]) != [size[1], 1]
    ):
        return None
    if layout.device_layout.element_arrangement != ElementArrangement.STANDARD:
        return None
    if getattr(op, "loop_info", None) is not None:
        return None
    if getattr(op, "_input_layout_overrides", None):
        return None
    return size


def _matmul(op):
    from .pass_utils import op_read_writes

    if not isinstance(op, ComputedBuffer) or not isinstance(op.data, Reduction):
        return None
    if op.data.reduction_type != BATCH_MATMUL_OP or len(op.data.reduction_ranges) != 1:
        return None
    shape = _dense_shape(op)
    if (
        shape is None
        or tuple(op.data.ranges[-2:]) != shape
        or any(x != 1 for x in op.data.ranges[:-2])
    ):
        return None
    if not sympy.sympify(op.data.reduction_ranges[0]).is_Integer:
        return None
    m, n = shape
    k = int(op.data.reduction_ranges[0])
    rw = op_read_writes(op)
    reads = [d for d in rw.reads if isinstance(d, MemoryDep)]
    writes = [d for d in rw.writes if isinstance(d, MemoryDep)]
    if len(reads) != 2 or len(writes) != 1:
        return None
    out = writes[0]
    m_syms = [s for s in out.ranges if out.index.coeff(s) == n and out.ranges[s] == m]
    n_syms = [s for s in out.ranges if out.index.coeff(s) == 1 and out.ranges[s] == n]
    if len(m_syms) != 1 or len(n_syms) != 1:
        return None
    ms, ns = m_syms[0], n_syms[0]
    ks = (reads[0].index.free_symbols | reads[1].index.free_symbols) - {ms, ns}
    if len(ks) != 1:
        return None
    (ks,) = ks
    pairs = [
        (x, w)
        for x, w in (reads, reads[::-1])
        if x.index == k * ms + ks and w.index == k * ns + ks
    ]
    if out.index != n * ms + ns or len(pairs) != 1:
        return None
    x, weight = pairs[0]
    if x.ranges.get(ks) != k:
        return None
    if _dense_shape(V.graph.get_buffer(x.name)) != (m, k):
        return None
    weight_buf = V.graph.get_buffer(weight.name)
    if _dense_shape(weight_buf) != (n, k):
        return None
    return DenseMatmul(op, x.name, weight.name, m, n, k, ms, ns, ks)


def _certified_weight(name, n, k):
    if name not in V.graph.graph_input_names:
        return None
    layout = _layout(V.graph.get_buffer(name)).device_layout
    eps = layout.elems_per_stick()
    if n % eps or k % eps:
        return None
    valid = list(getattr(layout, "zero_padding_valid_size", ()))
    if valid != [n // eps, k, eps]:
        return None
    return layout


def _pointwise_name(op, shape):
    from .pass_utils import op_read_writes
    from .split_multi_ops import _trace_inner_fn, _get_compute_ops

    if not isinstance(op, ComputedBuffer) or not isinstance(op.data, Pointwise):
        return None
    if (
        _dense_shape(op) != shape
        or tuple(op.data.ranges[-2:]) != shape
        or any(x != 1 for x in op.data.ranges[:-2])
    ):
        return None
    trace = _trace_inner_fn(op)
    if trace is None:
        return None
    compute = _get_compute_ops(trace)
    is_identity = len(trace) == 1 and trace[0][0] == "load"
    if not is_identity and (
        len(compute) != 1 or compute[0][0] not in ZERO_PRESERVING_POINTWISE
    ):
        return None
    # No scalar/broadcast, indirect access, conversion or fused expression:
    # every operand is one full, equally indexed dense chain value.
    name = "identity" if is_identity else compute[0][0]
    if any(entry[0] not in {"load", name} for entry in trace):
        return None
    rw = op_read_writes(op)
    writes = [d for d in rw.writes if isinstance(d, MemoryDep)]
    reads = [d for d in rw.reads if isinstance(d, MemoryDep)]
    if len(writes) != 1 or not reads:
        return None
    if any(d.index != writes[0].index for d in reads):
        return None
    return name


def _grow_output(op, logical, physical):
    layout = _layout(op)
    stl = layout.device_layout
    eps = stl.elems_per_stick()
    m, n = map(int, layout.size[-2:])
    dims = list(stl.device_size)
    axes = [
        i
        for i in range(len(dims) - 1)
        if dims[i] == logical // eps and stl.stride_map[i] == eps
    ]
    if (
        n != logical
        or stl.stride_map[-1] != 1
        or len(axes) != 1
        or math.prod(dims) != m * logical
    ):
        raise Unsupported("dense padding needs the standard output-stick layout")
    dims[axes[0]] = physical // eps
    layout.device_layout = SpyreTensorLayout(
        dims,
        list(stl.stride_map),
        stl.device_dtype,
        stl.element_arrangement,
    )
    layout.dense_padding = (logical, physical)


def _best_cost(op, matmul):
    from .work_division import (
        enumerate_work_division_candidates,
        _matmul_execution_cost,
    )
    from .pass_utils import iteration_space_from_op

    candidates = enumerate_work_division_candidates(op, config.sencores)
    if not candidates:
        return math.inf
    space = iteration_space_from_op(op)
    if matmul is not None:
        mm = matmul
        return min(
            _matmul_execution_cost(
                (1, 1),
                (int(space[mm.m_symbol]), c[mm.m_symbol]),
                (int(space[mm.n_symbol]), c[mm.n_symbol]),
                (int(space[mm.k_symbol]), c[mm.k_symbol]),
                config.sencores,
                shared_weight=True,
            )
            for c in candidates
        )
    # Reuse the production model for pointwise work; it sees the physical
    # allocation and candidate split. No model/TP/sequence cutoff is used.
    from .cost_model import predict_op
    from .dump_cost_model import extract_op_features

    return min(predict_op(extract_op_features(op, c)) / 1000 for c in candidates)


def select_dense_padding(graph):
    """Choose one physical extent per closed dense chain by estimated cost."""
    if not config.compiler_dense_padding or config.ktir_emitter:
        return
    from .pass_utils import op_read_writes
    from .work_division import has_work_div_hint

    consumers = {}
    mutated = set(getattr(graph, "mutated_inputs", ()))
    for op in graph.operations:
        layout = getattr(op, "layout", None)
        if isinstance(layout, MutationLayoutSHOULDREMOVE):
            mutated.add(layout.target.get_name())
        mutated.update(getattr(op, "get_mutation_names", lambda: ())())
        for dep in op.get_read_writes().reads:
            # Extern/fallback reads can be StarDep rather than MemoryDep.
            # Any outside reader breaks the closed-chain proof.
            consumers.setdefault(dep.name, set()).add(op.get_name())
    outputs = set(graph.get_output_names())
    used = set()
    candidates_seen = 0
    chains_selected = 0
    for down in graph.operations:
        terminal = _matmul(down)
        if terminal is None or down.get_name() in used:
            continue
        weight = _certified_weight(terminal.weight_name, terminal.n, terminal.k)
        if weight is None or terminal.weight_name in mutated:
            continue
        physical = int(weight.device_size[1])
        logical = terminal.k
        if physical <= logical:
            continue
        shape = (terminal.m, logical)
        chain = {}
        matmuls = {}
        pointwise = set()
        zero_tail = {}

        def visit(name):
            if name in chain:
                return True
            op = graph.get_buffer(name)
            if name in used or name in outputs or name in mutated:
                return False
            name_of_op = _pointwise_name(op, shape)
            if name_of_op is not None:
                reads = [
                    d for d in op_read_writes(op).reads if isinstance(d, MemoryDep)
                ]
                if not all(visit(d.name) for d in reads):
                    return False
                if name_of_op != "identity":
                    pointwise.add(name)
                    zero_tail[name] = True
                else:
                    zero_tail[name] = all(zero_tail[d.name] for d in reads)
            else:
                mm = _matmul(op)
                if mm is None or (mm.m, mm.n) != shape:
                    return False
                w = _certified_weight(mm.weight_name, mm.n, mm.k)
                if (
                    w is None
                    or mm.weight_name in mutated
                    or int(w.device_size[0]) * w.elems_per_stick() != physical
                ):
                    return False
                matmuls[name] = mm
                zero_tail[name] = False
            chain[name] = op
            return True

        if not visit(terminal.x_name) or not zero_tail[terminal.x_name]:
            continue
        names = set(chain) | {down.get_name()}
        if any(consumers.get(name, set()) - names for name in chain):
            continue
        ops = [*chain.values(), down]
        if any(has_work_div_hint(op) for op in ops):
            continue
        if any(
            getattr(op, "iteration_space_ownership", None) is not None for op in ops
        ):
            continue
        candidates_seen += 1
        matmuls[down.get_name()] = terminal
        old = {op.get_name(): _layout(op).device_layout for op in chain.values()}
        try:
            logical_cost = sum(_best_cost(op, matmuls.get(op.get_name())) for op in ops)
            for op in chain.values():
                _grow_output(op, logical, physical)
                op.dense_padding_zero_mask = op.get_name() in pointwise
            down.dense_reduction_padding = (logical, physical)
            padded_cost = sum(_best_cost(op, matmuls.get(op.get_name())) for op in ops)
        except (Unsupported, ValueError, AssertionError) as error:
            logger.info(
                "dense chain %s: rejected unsupported candidate: %s",
                down.get_name(),
                error,
            )
            padded_cost = math.inf
            logical_cost = 0.0
        if not padded_cost < logical_cost:
            logger.info(
                "dense chain %s: retain logical %d (candidate %d), estimated %.3f -> %.3f us",
                down.get_name(),
                logical,
                physical,
                logical_cost,
                padded_cost,
            )
            for op in chain.values():
                _layout(op).device_layout = old[op.get_name()]
                _layout(op).dense_padding = None
                op.dense_padding_zero_mask = False
            down.dense_reduction_padding = None
            continue
        used.update(names)
        chains_selected += 1
        logger.info(
            "dense chain %s: logical %d -> physical %d, estimated %.3f -> %.3f us",
            down.get_name(),
            logical,
            physical,
            logical_cost,
            padded_cost,
        )
    logger.info(
        "dense padding: %d eligible closed chains, %d selected",
        candidates_seen,
        chains_selected,
    )

# Copyright 2025 The Torch-Spyre Authors.
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

"""Carry a dynamic dim's (min, max, granularity) from one kernel to the next.

A dynamic dim is declared once, with ``x.to("spyre", dynamic=...)``, and stored
on that tensor as ``reserved_dims``. Every ``torch.compile`` region has its own
ShapeEnv, so a later kernel would see the dim as a new symbol with no range.
This module has both halves of the fix:

* ``apply_reserved_dims_contract`` reads a graph input's ``reserved_dims`` into
  the ShapeEnv. ``min``/``max`` go into ``shape_env.var_to_range``, intersected
  with the range already there, so a contract can only tighten a range. The full
  triple goes into ``graph._spyre_dim_contracts``, keyed by the symbol, because
  the ShapeEnv has no notion of granularity.
* ``output_reserved_dims`` works out the contract of a graph output from its
  symbolic size, for the wrapper to attach to the allocated tensor.
"""

import itertools
import math
from typing import Optional

import sympy
import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import FixedLayout, InputBuffer, StorageBox, TensorBox
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges

from torch_spyre._C import get_reserved_dims

from .errors import Unsupported
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("reserved_dims_contract")


def get_dim_contracts(graph: GraphLowering) -> dict:
    """``{symbol: (min, max, granularity)}`` recorded for this compile."""
    return getattr(graph, "_spyre_dim_contracts", {})


# Upper limit on the number of legal input-size combinations that
# output_granularity will enumerate. Beyond it the answer is the always-safe 1.
_MAX_ENUMERATED_POINTS = 4096


def output_granularity(expr: sympy.Expr, contracts: dict) -> int:
    """Granularity of the values a symbolic output size can take.

    ``expr`` is the output's size expression, ``contracts`` is
    ``{symbol: (min, max, granularity)}`` for the inputs (see
    ``get_dim_contracts``). Returns G such that every legal output size is a
    multiple of G, namely the gcd of the sizes at every legal input size.

    Exact, not estimated: each input takes only the sizes ``{G, 2G, ..., max}``
    that are ``>= min``, so we evaluate ``expr`` at all of them. Returns 1
    (always safe: "any size") whenever that cannot be done:
    a symbol with no contract, a size count above ``_MAX_ENUMERATED_POINTS``
    (e.g. granularity 1 over a wide range), or a non-integer result.
    """
    if not isinstance(expr, sympy.Basic) or not expr.free_symbols:
        return max(abs(int(expr)), 1) if expr else 1
    symbols = sorted(expr.free_symbols, key=str)
    if any(symbol not in contracts for symbol in symbols):
        return 1

    legal_sizes = []
    num_points = 1
    for symbol in symbols:
        dim_min, dim_max, granularity = contracts[symbol]
        first_size = -(-dim_min // granularity) * granularity
        sizes = list(range(first_size, dim_max + 1, granularity))
        if not sizes:
            return 1
        num_points *= len(sizes)
        if num_points > _MAX_ENUMERATED_POINTS:
            return 1
        legal_sizes.append(sizes)

    result = 0
    for point in itertools.product(*legal_sizes):
        value = expr.xreplace(
            {symbol: sympy.Integer(size) for symbol, size in zip(symbols, point)}
        )
        if not isinstance(value, sympy.Integer):
            return 1
        result = math.gcd(result, int(value))
    return result if result > 0 else 1


def output_reserved_dims(graph: GraphLowering, sizes) -> Optional[dict]:
    """``{dim: (min, max, granularity)}`` to stamp on a graph output, or None.

    ``sizes`` is the output buffer's (possibly symbolic) size. min/max come from
    the ShapeEnv's bounds on the size expression, which already carry an input
    range through arithmetic; granularity from ``output_granularity``.

    Returns None, meaning "unknown" and never "static", when no dim is symbolic
    or when the ShapeEnv has no finite upper bound (the compiler would then be
    sizing from the optimization hint).

    Raises ``Unsupported`` for more than one symbolic dim: reservations cover a
    single dim today (see ``spyre_empty_reserved`` and the ``resize_`` guard).
    """
    symbolic_dims = [
        dim
        for dim, size in enumerate(sizes)
        if isinstance(size, sympy.Basic) and size.free_symbols
    ]
    if not symbolic_dims:
        return None
    if len(symbolic_dims) > 1:
        raise Unsupported(
            f"graph output has {len(symbolic_dims)} symbolic dims {symbolic_dims} "
            f"(sizes {tuple(sizes)}); more than one symbolic dim on an output "
            f"is not yet supported"
        )
    dim = symbolic_dims[0]
    expr = sizes[dim]
    bounds = graph.sizevars.shape_env.bound_sympy(expr)
    if not (isinstance(bounds.upper, sympy.Integer) and int(bounds.upper) > 0):
        return None
    lower = int(bounds.lower) if isinstance(bounds.lower, sympy.Integer) else 0
    return {
        dim: (
            max(lower, 0),
            int(bounds.upper),
            output_granularity(expr, get_dim_contracts(graph)),
        )
    }


def apply_reserved_dims_contract(graph: GraphLowering) -> None:
    """Apply each graph input's ``reserved_dims`` to the ShapeEnv."""
    contracts: dict = {}
    graph._spyre_dim_contracts = contracts  # type: ignore[attr-defined]
    if not graph.graph_input_names:
        return

    shape_env = graph.sizevars.shape_env
    for name, real_input in zip(graph.graph_input_names, V.get_real_inputs()):
        if not isinstance(real_input, torch.Tensor):
            continue
        reserved = get_reserved_dims(real_input)
        if not reserved:
            continue

        tb = graph.graph_inputs[name]
        if (
            not isinstance(tb, TensorBox)
            or not isinstance(tb.data, StorageBox)
            or not isinstance(tb.data.data, InputBuffer)
            or not isinstance(tb.data.data.layout, FixedLayout)
        ):
            continue
        layout = tb.data.data.layout

        for dim, info in reserved.items():
            if dim >= len(layout.size):
                continue
            sym = layout.size[dim]
            # A static dim has no symbol to constrain: nothing to record.
            if not isinstance(sym, sympy.Symbol):
                continue
            dim_min = info["min"]
            dim_max = info["max"]
            granularity = info["granularity"]
            current = shape_env.var_to_range.get(sym)
            declared = ValueRanges(dim_min, dim_max)
            shape_env.var_to_range[sym] = (
                declared if current is None else current & declared
            )
            contracts[sym] = (dim_min, dim_max, granularity)
            logger.debug(
                "input %s dim %d: %s in [%d, %d], granularity %d",
                name,
                dim,
                sym,
                dim_min,
                dim_max,
                granularity,
            )

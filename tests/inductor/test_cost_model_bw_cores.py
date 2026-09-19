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

"""The bandwidth-vs-cores penalty: what a memory-bound op's core division costs.

The bundle memory term charges HBM bytes at the shared peak rate and never divides
by the core count, so a memory-bound op's division is invisible to the objective and
the co-optimizing solver has no reason to prefer a parallel split (issue #4655).

Every test here fixes one thing the penalty must get right and the motivating
benchmark could not have shown: that the bytes charged are the bytes that actually
cross the bus under THIS plan's placement, that a plan costs the same whether it is
being chosen or has been committed, and that the caller's ``CostParams`` govern.

No Spyre device or backend compiler is required; features are built directly.
"""

import pytest
import sympy

from torch_spyre._inductor import config, cost_model
from torch_spyre._inductor.cost_model import (
    ArgTraffic,
    BwCoresPenalty,
    CostParams,
    OpFeatures,
    bw_cores_anchors,
    bw_cores_g,
)

ELEMS, DTYPE = 1024, 2
BYTES = ELEMS * DTYPE
SPLIT_A, SPLIT_B = sympy.symbols("split_buf0_d0 split_buf0_d1", positive=True)


@pytest.fixture(autouse=True)
def _derate_on():
    with config.patch({"cost_model_bw_cores_derate": True}):
        yield


def _op(*, cores, out_lx=False, in_lx=False, indirect_store=False, inputs=1):
    """A pointwise op writing ``buf0`` and reading ``inputs`` interior buffers."""
    args = [
        ArgTraffic("buf0", "output", out_lx, ELEMS, is_boundary=False),
    ]
    args += [
        ArgTraffic(f"buf{i + 1}", "input", in_lx, ELEMS, is_boundary=False)
        for i in range(inputs)
    ]
    op = OpFeatures(
        name="mul",
        is_reduction=False,
        out_elems=ELEMS,
        cores=cores,
        dtype_bytes=DTYPE,
        args=args,
    )
    if indirect_store:
        # #4668's feature; absent on this branch, so the guard reads it defensively.
        op.is_indirect_store = True
    return op


def _penalty(op, params=None):
    return cost_model._bw_cores_penalty_ns([op], params or CostParams())


def _expected(byte_count, cores, p=None):
    p = p or CostParams()
    g = bw_cores_g(cores, bw_cores_anchors(p))
    if g >= 1.0:
        return 0.0
    return byte_count / p.bw_peak_gbps * (1.0 / g - 1.0)


# --------------------------------------------------------------------------- #
# The bytes charged are the bytes that cross the bus.
# --------------------------------------------------------------------------- #


def test_only_the_arguments_that_touch_hbm_are_charged():
    """An LX-resident output does not stop an HBM input from being loaded.

    The first version weighed every argument and gated the whole sum on the
    OUTPUT's residency, so this op was charged nothing at all.
    """
    both_hbm = _penalty(_op(cores=8))
    out_resident = _penalty(_op(cores=8, out_lx=True))
    in_resident = _penalty(_op(cores=8, in_lx=True))
    assert both_hbm == pytest.approx(_expected(2 * BYTES, 8))
    assert out_resident == pytest.approx(_expected(BYTES, 8)), "the input still loads"
    assert in_resident == pytest.approx(_expected(BYTES, 8)), "the output still stores"


def test_a_fully_resident_op_has_no_penalty():
    """Nothing crosses the bus, so the core division cannot cost anything."""
    assert _penalty(_op(cores=1, out_lx=True, in_lx=True)) == 0.0


def test_a_pinned_resident_buffer_is_not_charged_as_spilled():
    """``is_lx=True`` is a decided residency, not an undecided one.

    The first version tested ``isinstance(is_lx, sympy.Basic)`` to mean "symbolic"
    and fell back to a gate of 0 -- which reads as NOT resident -- so a pinned
    buffer was charged at full weight.
    """
    pinned = _op(cores=1, out_lx=True, in_lx=True)
    assert _penalty(pinned) == 0.0
    # One pinned, one spilled: exactly one argument's bytes.
    assert _penalty(_op(cores=1, out_lx=True)) == pytest.approx(_expected(BYTES, 1))


def test_full_occupancy_is_free():
    """``g(32) = 1``, so nothing on the 32-core path moves by a single byte."""
    assert _penalty(_op(cores=32)) == 0.0


def test_the_penalty_grows_as_the_division_narrows():
    wide = _penalty(_op(cores=16))
    narrow = _penalty(_op(cores=1))
    assert 0.0 < wide < narrow


# --------------------------------------------------------------------------- #
# One plan, one price.
# --------------------------------------------------------------------------- #


def test_symbolic_and_committed_prices_agree():
    """The cost used to CHOOSE a plan must equal the cost REPORTED for it.

    The first version returned 0.0 whenever the core count was concrete, so the
    solver minimized one number and the cost dump printed another for the same
    plan.
    """
    symbolic = _penalty(_op(cores=SPLIT_A * SPLIT_B))
    committed = _penalty(_op(cores=8))
    assert symbolic.free_symbols == {SPLIT_A, SPLIT_B}
    bound = float(symbolic.subs({SPLIT_A: 4, SPLIT_B: 2}))
    assert bound == pytest.approx(committed)


def test_an_undecided_residency_prices_both_outcomes():
    is_lx = sympy.Symbol("is_lx_buf0", positive=True)
    op = _op(cores=8)
    op.args[0].is_lx = is_lx
    term = _penalty(op)
    spilled = float(term.subs({is_lx: 0}))
    resident = float(term.subs({is_lx: 1}))
    assert spilled == pytest.approx(_expected(2 * BYTES, 8))
    assert resident == pytest.approx(_expected(BYTES, 8)), "only the input remains"


def test_lambdify_evaluates_the_term_the_annealer_way():
    """``sa_cooptimizer`` scores plans with ``sympy.lambdify``, so the node's
    ``_imp_`` has to produce the same number ``subs`` does."""
    term = _penalty(_op(cores=SPLIT_A * SPLIT_B))
    syms = sorted(term.free_symbols, key=str)
    fn = sympy.lambdify(syms, term, modules="math")
    assert fn(4, 2) == pytest.approx(float(term.subs({SPLIT_A: 4, SPLIT_B: 2})))


def test_srepr_round_trips_the_node():
    """The objective dump stores terms as ``srepr``; a reader must restore them."""
    term = _penalty(_op(cores=SPLIT_A * SPLIT_B))
    back = sympy.sympify(sympy.srepr(term), locals={"BwCoresPenalty": BwCoresPenalty})
    assert back == term


# --------------------------------------------------------------------------- #
# The caller's parameters govern.
# --------------------------------------------------------------------------- #


def test_a_bandwidth_override_scales_the_penalty():
    """The first version built a default ``CostParams`` inside the node, so an
    override moved the base term and left the penalty behind."""
    slow = CostParams(bw_peak_gbps=75.0)
    assert _penalty(_op(cores=8), slow) == pytest.approx(
        2 * _penalty(_op(cores=8)), rel=1e-9
    )


def test_a_curve_override_governs_the_penalty():
    flat = CostParams(red_bw_cores_g={1: 1.0, 32: 1.0})
    assert _penalty(_op(cores=1), flat) == 0.0
    assert _penalty(_op(cores=1)) > 0.0


def test_the_node_carries_its_curve_rather_than_rebuilding_one():
    """Every parameter the penalty needs is folded into the node at construction,
    so no consumer -- printer, ``lambdify``, ``evalf`` -- has to know which
    ``CostParams`` built the objective."""
    slow = CostParams(bw_peak_gbps=75.0)
    term = _penalty(_op(cores=SPLIT_A * SPLIT_B), slow)
    node = next(a for a in term.atoms(BwCoresPenalty))
    anchors = node.args[2]
    assert tuple(anchors) == tuple(bw_cores_anchors(slow))
    assert float(node.args[0]) == pytest.approx(BYTES / slow.bw_peak_gbps)


# --------------------------------------------------------------------------- #
# Ownership: #4668 prices indirect stores.
# --------------------------------------------------------------------------- #


def test_an_indirect_stores_write_is_left_to_the_store_model():
    """PR #4668 charges the writing cores of a supported indexed row store. Its
    write must not also be charged here, while its reads still are."""
    ordinary = _penalty(_op(cores=8, inputs=1))
    indexed = _penalty(_op(cores=8, inputs=1, indirect_store=True))
    assert ordinary == pytest.approx(_expected(2 * BYTES, 8))
    assert indexed == pytest.approx(_expected(BYTES, 8)), "the read is still ours"


# --------------------------------------------------------------------------- #
# Wiring.
# --------------------------------------------------------------------------- #


def test_the_flag_turns_the_whole_term_off():
    with config.patch({"cost_model_bw_cores_derate": False}):
        assert _penalty(_op(cores=1)) == 0.0


def test_predict_ops_is_unchanged_at_full_occupancy():
    """End to end: the penalty is additive over the untouched bandwidth term, so a
    32-core prediction is byte-identical to one with the feature off."""
    op = _op(cores=32)
    with config.patch({"cost_model_bw_cores_derate": False}):
        before = cost_model.predict_ops([op], CostParams())
    after = cost_model.predict_ops([op], CostParams())
    assert after == pytest.approx(before)


def test_predict_ops_charges_the_penalty_below_full_occupancy():
    op = _op(cores=4)
    with config.patch({"cost_model_bw_cores_derate": False}):
        before = cost_model.predict_ops([op], CostParams())
    after = cost_model.predict_ops([op], CostParams())
    assert after - before == pytest.approx(_expected(2 * BYTES, 4))


def test_a_shared_graph_input_is_charged_once_for_the_bundle():
    """``_fused_hbm_bytes`` counts a fused bundle's external input once however
    many ops read it; the penalty follows the same accounting or it would price
    a softmax's ``arg0`` twice."""
    shared = [
        OpFeatures(
            name=name,
            is_reduction=False,
            out_elems=ELEMS,
            cores=8,
            dtype_bytes=DTYPE,
            args=[
                ArgTraffic(f"buf{i}", "output", True, ELEMS, is_boundary=False),
                ArgTraffic("arg0_1", "input", False, ELEMS, is_boundary=True),
            ],
        )
        for i, name in enumerate(("amax", "sub"))
    ]
    assert cost_model._bw_cores_penalty_ns(shared, CostParams()) == pytest.approx(
        _expected(BYTES, 8)
    )

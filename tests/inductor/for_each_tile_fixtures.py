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

"""Minimal for_each_tile fixtures for while_loop-splice tests.

A pure map, a pure carry, and an online-softmax multi-leaf carry -- give
test_for_each_tile_lowering.py and test_for_each_tile_e2e.py real
`while_loop` FX nodes to drive via `for_each_tile`
(torch_spyre._inductor.wsr.for_each_tile, landed in #4136).

Not collected by pytest directly (no `test_` prefix, no CI config entry) --
see tests/inductor/utils_inductor.py for the same pattern.
"""

import contextlib
from collections.abc import Callable

import torch
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor.wsr import for_each_tile


M, K, N = 8, 12, 6


def matmul_inputs() -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    torch.manual_seed(0)
    X = torch.randn(M, K)
    Y = torch.randn(K, N)
    return (X, Y), X @ Y


def split_m_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Case A: tile M as a map. Y is invariant; the result tile lays along dim 0."""

    def body(_, ops):
        x_tile, y_whole = ops
        return None, x_tile @ y_whole

    _, out = for_each_tile(body, (X, Y), dims=(0, None), tile_size=2, out_dim=0)
    return out


def split_m_elementwise_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Case A variant: an elementwise op reads the marker-tagged tile first.

    Unlike split_m_fn's direct ``x_tile @ y_whole`` (a matmul, which lowers
    to an aten-fallback ExternKernelOut -- a StarDep-shaped consumer even
    on a device-less CPU fixture), the intervening ``x_tile * 2.0``
    lowers to a genuine Pointwise ComputedBuffer on CPU too, giving a
    consumer that reaches for_each_tile_lowering.py's
    ``_inline_marker_into_consumer``/``_InlineMarkerHandler`` (the
    MemoryDep/inner_fn-backed branch of ``_consume_tile_dim_markers``)
    without needing a real Spyre device. split_m_fn's own StarDep-shaped
    consumer never exercises that branch at all.
    """

    def body(_, ops):
        x_tile, y_whole = ops
        scaled = x_tile * 2.0
        return None, scaled @ y_whole

    _, out = for_each_tile(body, (X, Y), dims=(0, None), tile_size=2, out_dim=0)
    return out


def split_k_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Case C: co-indexed split-K matmul; carry accumulates the partial product."""

    def body(acc, ops):
        x_tile, y_tile = ops
        return acc + x_tile @ y_tile, None

    final, _ = for_each_tile(
        body,
        (X, Y),
        dims=(-1, 0),
        tile_size=3,
        init=torch.zeros(M, N, device=X.device, dtype=X.dtype),
    )
    return final


def nested_split_m_then_k_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Nested case: outer for_each_tile maps M; inner for_each_tile carries K.

    Each outer M-tile computes its own row-block of X @ Y via an inner
    split-K accumulation -- two tile_dim_marker-tagged reads at two nesting
    levels (outer's M-tile of X, inner's K-tile of X and Y), the exact
    ambiguity shape (two markers on two different reads inside one nested
    body) the tile-dim-marker consumption design targets.
    """

    def outer_body(_, ops):
        x_tile, y_whole = ops

        def inner_body(acc, inner_ops):
            x_inner_tile, y_inner_tile = inner_ops
            return acc + x_inner_tile @ y_inner_tile, None

        m_tile = x_tile.shape[0]
        final, _ = for_each_tile(
            inner_body,
            (x_tile, y_whole),
            dims=(-1, 0),
            tile_size=3,
            init=torch.zeros(m_tile, N, device=X.device, dtype=X.dtype),
        )
        return None, final

    _, out = for_each_tile(outer_body, (X, Y), dims=(0, None), tile_size=2, out_dim=0)
    return out


def nested_split_m_then_k_reference(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Eager Python nesting of the same tiling, for value assertions."""
    m_tile_size, k_tile_size = 2, 3
    rows = []
    for m_start in range(0, X.shape[0], m_tile_size):
        x_m_tile = X[m_start : m_start + m_tile_size]
        acc = torch.zeros(x_m_tile.shape[0], N, device=X.device, dtype=X.dtype)
        for k_start in range(0, X.shape[1], k_tile_size):
            x_k_tile = x_m_tile[:, k_start : k_start + k_tile_size]
            y_k_tile = Y[k_start : k_start + k_tile_size]
            acc = acc + x_k_tile @ y_k_tile
        rows.append(acc)
    return torch.cat(rows, dim=0)


B = 2


def _batched_matmul_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    X = torch.randn(B, M, K)
    Y = torch.randn(B, K, N)
    return X, Y


def triple_nested_stardep_outer_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Three levels deep (outer batch B, middle M, inner K); StarDep at outer only.

    The outer level's tiles (x_b_tile, y_b_tile) are consumed directly by
    the middle for_each_tile call, which recursively contains matmul
    reductions, so the outer markers are STAR_DEP_KEPT (2 markers). The
    middle and inner levels both route their tiles through elementwise ops
    first (`x_m_tile * 1.0`, `x_k_tile * 1.0`), causing their markers to
    be INLINE_ERASED and removed from the graph. Exercises marker_resolution
    propagating through two more splice levels above STAR_DEP_KEPT markers.

    Note: .squeeze(0) on outer tiles reconciles the outer loop's tile_size=1
    singleton batch dimension with the middle loop's 2-D input expectation.
    """

    def outer_body(_, ops):
        x_b_tile, y_b_tile = ops
        x_b = x_b_tile.squeeze(0)
        y_b = y_b_tile.squeeze(0)

        def middle_body(_, mid_ops):
            x_m_tile, y_m_whole = mid_ops
            x_m_tile = x_m_tile * 1.0
            m_tile = x_m_tile.shape[0]

            def inner_body(acc, inner_ops):
                x_k_tile, y_k_tile = inner_ops
                x_k_tile = x_k_tile * 1.0
                return acc + x_k_tile @ y_k_tile, None

            final, _ = for_each_tile(
                inner_body,
                (x_m_tile, y_m_whole),
                dims=(-1, 0),
                tile_size=3,
                init=torch.zeros(m_tile, N, device=X.device, dtype=X.dtype),
            )
            return None, final

        _, mid_out = for_each_tile(
            middle_body, (x_b, y_b), dims=(0, None), tile_size=2, out_dim=0
        )
        return None, mid_out

    _, out = for_each_tile(outer_body, (X, Y), dims=(0, 0), tile_size=1, out_dim=0)
    return out


def triple_nested_stardep_outer_reference(
    X: torch.Tensor, Y: torch.Tensor
) -> torch.Tensor:
    """Eager Python nesting of the same tiling, for value assertions."""
    b_tile_size, m_tile_size, k_tile_size = 1, 2, 3
    batches = []
    for b_start in range(0, X.shape[0], b_tile_size):
        x_b_tile = X[b_start : b_start + b_tile_size]
        y_b_tile = Y[b_start : b_start + b_tile_size]
        for b_idx in range(x_b_tile.shape[0]):
            x_b = x_b_tile[b_idx]
            y_b = y_b_tile[b_idx]
            m_rows = []
            for m_start in range(0, x_b.shape[0], m_tile_size):
                x_m_tile = x_b[m_start : m_start + m_tile_size] * 1.0
                acc = torch.zeros(x_m_tile.shape[0], N, device=X.device, dtype=X.dtype)
                for k_start in range(0, x_m_tile.shape[1], k_tile_size):
                    x_k_tile = x_m_tile[:, k_start : k_start + k_tile_size] * 1.0
                    y_k_tile = y_b[k_start : k_start + k_tile_size]
                    acc = acc + x_k_tile @ y_k_tile
                m_rows.append(acc)
            batches.append(torch.cat(m_rows, dim=0))
    return torch.cat(batches, dim=0)


def triple_nested_stardep_middle_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Three levels deep (outer B, middle M, inner K); StarDep at outer AND middle.

    CORRECTED (final whole-branch review): empirical measurement
    (instrumenting `_consume_tile_dim_markers` per splice call) shows this
    fixture actually resolves to 1 STAR_DEP_KEPT marker at the outer level
    (from `y_b_tile`, which is never routed through an elementwise op --
    only `x_b_tile` gets `* 1.0`) plus 1 STAR_DEP_KEPT marker at the middle
    level, not "2 markers at middle only" as originally documented. The
    inner level's x_k_tile is routed through elementwise (`* 1.0`), causing
    its marker to be INLINE_ERASED, as originally documented. Exercises
    propagation through splice levels above two independently-resolved
    STAR_DEP_KEPT markers, one level apart.

    Note: .squeeze(0) on outer tiles reconciles the outer loop's tile_size=1
    singleton batch dimension with the middle loop's 2-D input expectation.
    """

    def outer_body(_, ops):
        x_b_tile, y_b_tile = ops
        x_b_tile = x_b_tile * 1.0
        x_b = x_b_tile.squeeze(0)
        y_b = y_b_tile.squeeze(0)

        def middle_body(_, mid_ops):
            x_m_tile, y_m_whole = mid_ops
            m_tile = x_m_tile.shape[0]

            def inner_body(acc, inner_ops):
                x_k_tile, y_k_tile = inner_ops
                x_k_tile = x_k_tile * 1.0
                return acc + x_k_tile @ y_k_tile, None

            final, _ = for_each_tile(
                inner_body,
                (x_m_tile, y_m_whole),
                dims=(-1, 0),
                tile_size=3,
                init=torch.zeros(m_tile, N, device=X.device, dtype=X.dtype),
            )
            return None, final

        _, mid_out = for_each_tile(
            middle_body, (x_b, y_b), dims=(0, None), tile_size=2, out_dim=0
        )
        return None, mid_out

    _, out = for_each_tile(outer_body, (X, Y), dims=(0, 0), tile_size=1, out_dim=0)
    return out


def triple_nested_stardep_middle_reference(
    X: torch.Tensor, Y: torch.Tensor
) -> torch.Tensor:
    """Eager Python nesting of the same tiling, for value assertions."""
    b_tile_size, m_tile_size, k_tile_size = 1, 2, 3
    batches = []
    for b_start in range(0, X.shape[0], b_tile_size):
        x_b_tile = X[b_start : b_start + b_tile_size] * 1.0
        y_b_tile = Y[b_start : b_start + b_tile_size]
        for b_idx in range(x_b_tile.shape[0]):
            x_b = x_b_tile[b_idx]
            y_b = y_b_tile[b_idx]
            m_rows = []
            for m_start in range(0, x_b.shape[0], m_tile_size):
                x_m_tile = x_b[m_start : m_start + m_tile_size]
                acc = torch.zeros(x_m_tile.shape[0], N, device=X.device, dtype=X.dtype)
                for k_start in range(0, x_m_tile.shape[1], k_tile_size):
                    x_k_tile = x_m_tile[:, k_start : k_start + k_tile_size] * 1.0
                    y_k_tile = y_b[k_start : k_start + k_tile_size]
                    acc = acc + x_k_tile @ y_k_tile
                m_rows.append(acc)
            batches.append(torch.cat(m_rows, dim=0))
    return torch.cat(batches, dim=0)


def triple_nested_stardep_inner_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Three levels deep (outer B, middle M, inner K).

    CORRECTED (final whole-branch review): this fixture's name/original
    docstring claimed "StarDep at inner only," but empirical measurement
    (instrumenting `_consume_tile_dim_markers` per splice call) shows the
    opposite: the inner level's marker is INLINE_ERASED (matmul consumes it
    directly, no StarDep involved), and the surviving STAR_DEP_KEPT marker
    is at the OUTER level instead -- because `y_b_tile` is never routed
    through an elementwise op (only `x_b_tile` gets `* 1.0`), so `y_b_tile`'s
    marker has no fusable consumer and stays live. This fixture does not
    currently exercise an inner-level STAR_DEP_KEPT marker at all; it
    exercises the same outer-level-only configuration as
    triple_nested_stardep_outer_fn/multilevel_fn, just with a different
    middle-level structure. Left in place (removing it would lose the
    middle-level elementwise-routing structural variation) but the name is
    now misleading -- see follow-up note in task-6 final-review ledger
    entry re: genuinely isolating each level would need `* 1.0` on
    `y_b_tile` too, which is a fixture-body change, not a docstring fix.

    Note: .squeeze(0) on outer tiles reconciles the outer loop's tile_size=1
    singleton batch dimension with the middle loop's 2-D input expectation.
    """

    def outer_body(_, ops):
        x_b_tile, y_b_tile = ops
        x_b_tile = x_b_tile * 1.0
        x_b = x_b_tile.squeeze(0)
        y_b = y_b_tile.squeeze(0)

        def middle_body(_, mid_ops):
            x_m_tile, y_m_whole = mid_ops
            x_m_tile = x_m_tile * 1.0
            m_tile = x_m_tile.shape[0]

            def inner_body(acc, inner_ops):
                x_k_tile, y_k_tile = inner_ops
                return acc + x_k_tile @ y_k_tile, None

            final, _ = for_each_tile(
                inner_body,
                (x_m_tile, y_m_whole),
                dims=(-1, 0),
                tile_size=3,
                init=torch.zeros(m_tile, N, device=X.device, dtype=X.dtype),
            )
            return None, final

        _, mid_out = for_each_tile(
            middle_body, (x_b, y_b), dims=(0, None), tile_size=2, out_dim=0
        )
        return None, mid_out

    _, out = for_each_tile(outer_body, (X, Y), dims=(0, 0), tile_size=1, out_dim=0)
    return out


def triple_nested_stardep_inner_reference(
    X: torch.Tensor, Y: torch.Tensor
) -> torch.Tensor:
    """Eager Python nesting of the same tiling, for value assertions."""
    b_tile_size, m_tile_size, k_tile_size = 1, 2, 3
    batches = []
    for b_start in range(0, X.shape[0], b_tile_size):
        x_b_tile = X[b_start : b_start + b_tile_size] * 1.0
        y_b_tile = Y[b_start : b_start + b_tile_size]
        for b_idx in range(x_b_tile.shape[0]):
            x_b = x_b_tile[b_idx]
            y_b = y_b_tile[b_idx]
            m_rows = []
            for m_start in range(0, x_b.shape[0], m_tile_size):
                x_m_tile = x_b[m_start : m_start + m_tile_size] * 1.0
                acc = torch.zeros(x_m_tile.shape[0], N, device=X.device, dtype=X.dtype)
                for k_start in range(0, x_m_tile.shape[1], k_tile_size):
                    x_k_tile = x_m_tile[:, k_start : k_start + k_tile_size]
                    y_k_tile = y_b[k_start : k_start + k_tile_size]
                    acc = acc + x_k_tile @ y_k_tile
                m_rows.append(acc)
            batches.append(torch.cat(m_rows, dim=0))
    return torch.cat(batches, dim=0)


def triple_nested_stardep_multilevel_fn(
    X: torch.Tensor, Y: torch.Tensor
) -> torch.Tensor:
    """Three levels deep (outer B, middle M, inner K); StarDep at outer AND inner.

    Combines triple_nested_stardep_outer_fn's outer STAR_DEP_KEPT markers
    with triple_nested_stardep_inner_fn's inner STAR_DEP_KEPT marker in
    one fixture. The outer level's tiles (x_b_tile, y_b_tile) feed directly
    into the middle for_each_tile call (STAR_DEP_KEPT, 2 markers). The
    middle level's x_m_tile is routed through elementwise (`* 1.0`), causing
    its marker to be INLINE_ERASED. The inner level's tiles (x_k_tile,
    y_k_tile) feed directly into the matmul (STAR_DEP_KEPT, 1 marker from
    x_k_tile after the elementwise on x_m_tile). Catches any interaction
    between two independent marker_resolution stamps in the same splice
    chain that the single-level variants cannot surface.

    NOTE (Step 2 finding, CORRECTED at final whole-branch review): This
    fixture was intended to exercise both outer and inner STAR_DEP_KEPT
    markers independently. Empirical verification (instrumenting
    `_consume_tile_dim_markers` per splice call) confirms the inner-level
    marker is completely lost/erased here -- but the originally-documented
    contrast case was wrong: triple_nested_stardep_inner_fn does NOT
    preserve an inner-level marker either (see its own corrected docstring)
    -- its survivor is also at the outer level, via the same `y_b_tile`
    asymmetry as this fixture. So inner-marker erasure for this operand
    shape looks like the norm across outer_fn, inner_fn, and the
    pre-existing nested_split_m_then_k_fn, not something specific to
    stacking two STAR_DEP_KEPT levels. The underlying "does stacking two
    STAR_DEP_KEPT levels lose the deeper one" question (likely issue #4581
    territory) remains open, but this fixture's measurement alone does not
    establish it -- a fixture where the inner marker demonstrably survives
    in isolation (e.g. with the `y_b_tile`-style asymmetry fixed at every
    level) would be needed to actually test that hypothesis. This fixture
    currently exercises only the outer 2 STAR_DEP_KEPT markers, identical to
    triple_nested_stardep_outer_fn. The middle-level elementwise is
    preserved as a structural probe for future investigation. See
    task-3-report.md for the original investigation and the final
    whole-branch review's ledger entry for the correction.

    Note: .squeeze(0) on outer tiles reconciles the outer loop's tile_size=1
    singleton batch dimension with the middle loop's 2-D input expectation.
    """

    def outer_body(_, ops):
        x_b_tile, y_b_tile = ops
        x_b = x_b_tile.squeeze(0)
        y_b = y_b_tile.squeeze(0)

        def middle_body(_, mid_ops):
            x_m_tile, y_m_whole = mid_ops
            x_m_tile = x_m_tile * 1.0
            m_tile = x_m_tile.shape[0]

            def inner_body(acc, inner_ops):
                x_k_tile, y_k_tile = inner_ops
                return acc + x_k_tile @ y_k_tile, None

            final, _ = for_each_tile(
                inner_body,
                (x_m_tile, y_m_whole),
                dims=(-1, 0),
                tile_size=3,
                init=torch.zeros(m_tile, N, device=X.device, dtype=X.dtype),
            )
            return None, final

        _, mid_out = for_each_tile(
            middle_body, (x_b, y_b), dims=(0, None), tile_size=2, out_dim=0
        )
        return None, mid_out

    _, out = for_each_tile(outer_body, (X, Y), dims=(0, 0), tile_size=1, out_dim=0)
    return out


def triple_nested_stardep_multilevel_reference(
    X: torch.Tensor, Y: torch.Tensor
) -> torch.Tensor:
    """Eager Python nesting of the same tiling, for value assertions."""
    b_tile_size, m_tile_size, k_tile_size = 1, 2, 3
    batches = []
    for b_start in range(0, X.shape[0], b_tile_size):
        x_b_tile = X[b_start : b_start + b_tile_size]
        y_b_tile = Y[b_start : b_start + b_tile_size]
        for b_idx in range(x_b_tile.shape[0]):
            x_b = x_b_tile[b_idx]
            y_b = y_b_tile[b_idx]
            m_rows = []
            for m_start in range(0, x_b.shape[0], m_tile_size):
                x_m_tile = x_b[m_start : m_start + m_tile_size] * 1.0
                acc = torch.zeros(x_m_tile.shape[0], N, device=X.device, dtype=X.dtype)
                for k_start in range(0, x_m_tile.shape[1], k_tile_size):
                    x_k_tile = x_m_tile[:, k_start : k_start + k_tile_size]
                    y_k_tile = y_b[k_start : k_start + k_tile_size]
                    acc = acc + x_k_tile @ y_k_tile
                m_rows.append(acc)
            batches.append(torch.cat(m_rows, dim=0))
    return torch.cat(batches, dim=0)


def sibling_nested_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """One outer for_each_tile level (M) whose body runs two independent
    (non-nested, sibling) inner for_each_tile loops: one splits K over X/Y
    for a matmul-style partial, the other splits K again over the same
    x_tile with an elementwise scale, and the two partials are summed. Both
    sibling loops route their tiles through elementwise ops first.

    NOTE: This fixture is still UNCOMPILABLE, but no longer at the marker
    pass. There is exactly ONE marker here: the outer for_each_tile's own
    dim=0 marker on x_tile itself (stamped once, by the outer loop, before
    either sibling runs). x_tile is then handed directly as an operand to
    both sibling_a's and sibling_b's inner for_each_tile calls, each of
    which lowers to its own WhileLoop op. Both WhileLoops end up as
    consumers of the single x_tile marker via StarDep (whole-tensor,
    name-only dependency) -- so the marker has 2 consuming reads, not 1.
    _consume_tile_dim_markers now resolves every consumer of a marker
    rather than asserting on the second one (the multi-consumer shape
    paged attention needs -- see paged_gather_kv_fn below), so this fixture
    now gets past that pass and fails further downstream instead, in the
    same OpSpecValidationError symbol-consistency gap that
    triple_nested_stardep_multilevel_fn hits (issue #4581 follow-up OS-5:
    an inner-level `u`-symbol reaching an OpSpec whose iteration_space only
    has the outer level's `c` keys). The bug site is still this outer
    x_tile hand-off, NOT inside either sibling's own K-tiling body -- don't
    go looking for duplicated K-dim markers there. Fixture retained as a
    structural probe and as documentation of the remaining limitation.
    """

    def outer_body(_, ops):
        x_tile, y_whole = ops
        m_tile = x_tile.shape[0]

        def sibling_a_body(acc, inner_ops):
            x_k_tile, y_k_tile = inner_ops
            x_k_tile = x_k_tile * 1.0
            return acc + x_k_tile @ y_k_tile, None

        partial_a, _ = for_each_tile(
            sibling_a_body,
            (x_tile, y_whole),
            dims=(-1, 0),
            tile_size=3,
            init=torch.zeros(m_tile, N, device=X.device, dtype=X.dtype),
        )

        def sibling_b_body(acc, inner_ops):
            (x_k_tile,) = inner_ops
            x_k_tile = x_k_tile * 1.0
            return acc + x_k_tile.sum(dim=-1, keepdim=True), None

        partial_b, _ = for_each_tile(
            sibling_b_body,
            (x_tile,),
            dims=(-1,),
            tile_size=3,
            init=torch.zeros(m_tile, 1, device=X.device, dtype=X.dtype),
        )

        return None, partial_a + partial_b

    _, out = for_each_tile(outer_body, (X, Y), dims=(0, None), tile_size=2, out_dim=0)
    return out


def sibling_nested_reference(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Eager Python nesting of the same tiling, for value assertions."""
    m_tile_size, k_tile_size = 2, 3
    rows = []
    for m_start in range(0, X.shape[0], m_tile_size):
        x_m_tile = X[m_start : m_start + m_tile_size]
        acc_a = torch.zeros(x_m_tile.shape[0], N, device=X.device, dtype=X.dtype)
        for k_start in range(0, x_m_tile.shape[1], k_tile_size):
            x_k_tile = x_m_tile[:, k_start : k_start + k_tile_size] * 1.0
            y_k_tile = Y[k_start : k_start + k_tile_size]
            acc_a = acc_a + x_k_tile @ y_k_tile
        acc_b = torch.zeros(x_m_tile.shape[0], 1, device=X.device, dtype=X.dtype)
        for k_start in range(0, x_m_tile.shape[1], k_tile_size):
            x_k_tile = x_m_tile[:, k_start : k_start + k_tile_size] * 1.0
            acc_b = acc_b + x_k_tile.sum(dim=-1, keepdim=True)
        rows.append(acc_a + acc_b)
    return torch.cat(rows, dim=0)


def sibling_nested_stardep_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Same sibling shape as sibling_nested_fn, but sibling A's tile feeds
    its matmul directly (no intervening elementwise op), while sibling B
    routes through elementwise.

    NOTE: This fixture is still UNCOMPILABLE for the same remaining reason
    as sibling_nested_fn: there is exactly ONE marker, the outer
    for_each_tile's dim=0 marker on x_tile itself, and x_tile is handed
    directly to both sibling_a's and sibling_b's inner for_each_tile calls.
    Each lowers to its own WhileLoop op, and both WhileLoops consume the
    single x_tile marker via StarDep -- giving it 2 consuming reads instead
    of 1. _consume_tile_dim_markers now resolves both of them (see
    sibling_nested_fn's own note), so the failure has moved downstream to
    the same OpSpecValidationError symbol-consistency gap
    triple_nested_stardep_multilevel_fn hits (issue #4581 follow-up OS-5).
    The bug site is the outer x_tile hand-off, not inside either sibling's
    own K-tiling body. The intended differentiation (sibling A
    STAR_DEP_KEPT, sibling B INLINE_ERASED) still cannot be observed end to
    end until that downstream gap is closed. Fixture retained as a
    structural probe for the eventual fix.
    """

    def outer_body(_, ops):
        x_tile, y_whole = ops
        m_tile = x_tile.shape[0]

        def sibling_a_body(acc, inner_ops):
            x_k_tile, y_k_tile = inner_ops
            return acc + x_k_tile @ y_k_tile, None

        partial_a, _ = for_each_tile(
            sibling_a_body,
            (x_tile, y_whole),
            dims=(-1, 0),
            tile_size=3,
            init=torch.zeros(m_tile, N, device=X.device, dtype=X.dtype),
        )

        def sibling_b_body(acc, inner_ops):
            (x_k_tile,) = inner_ops
            x_k_tile = x_k_tile * 1.0
            return acc + x_k_tile.sum(dim=-1, keepdim=True), None

        partial_b, _ = for_each_tile(
            sibling_b_body,
            (x_tile,),
            dims=(-1,),
            tile_size=3,
            init=torch.zeros(m_tile, 1, device=X.device, dtype=X.dtype),
        )

        return None, partial_a + partial_b

    _, out = for_each_tile(outer_body, (X, Y), dims=(0, None), tile_size=2, out_dim=0)
    return out


def sibling_nested_stardep_reference(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Eager Python nesting of the same tiling, for value assertions."""
    m_tile_size, k_tile_size = 2, 3
    rows = []
    for m_start in range(0, X.shape[0], m_tile_size):
        x_m_tile = X[m_start : m_start + m_tile_size]
        acc_a = torch.zeros(x_m_tile.shape[0], N, device=X.device, dtype=X.dtype)
        for k_start in range(0, x_m_tile.shape[1], k_tile_size):
            x_k_tile = x_m_tile[:, k_start : k_start + k_tile_size]
            y_k_tile = Y[k_start : k_start + k_tile_size]
            acc_a = acc_a + x_k_tile @ y_k_tile
        acc_b = torch.zeros(x_m_tile.shape[0], 1, device=X.device, dtype=X.dtype)
        for k_start in range(0, x_m_tile.shape[1], k_tile_size):
            x_k_tile = x_m_tile[:, k_start : k_start + k_tile_size] * 1.0
            acc_b = acc_b + x_k_tile.sum(dim=-1, keepdim=True)
        rows.append(acc_a + acc_b)
    return torch.cat(rows, dim=0)


LQ, LK, D = 128, 256, 128
SOFTMAX_TILE_SIZE = 128


def attention_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    Q = torch.randn(LQ, D, dtype=torch.float16)
    K = torch.randn(LK, D, dtype=torch.float16)
    V = torch.randn(LK, D, dtype=torch.float16)
    return Q, K, V


def online_softmax_fn(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """Case D: single for_each_tile loop, 3-leaf carry (m, denom, acc).

    Q is closed over whole (not tiled -- the outer Q-loop is deferred to a
    follow-on nested fixture). K/V are Kind.SLICE, co-indexed and tiled along
    Lk. carry = (m, denom, acc): running max, running sum-of-exp, weighted
    accumulator -- the online-softmax recurrence flash attention's inner
    loop needs.
    """

    def body(carry, tiles):
        m, denom, acc = carry
        k_tile, v_tile = tiles
        scores = Q @ k_tile.transpose(-1, -2)
        m_new = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
        correction = torch.exp(m - m_new)
        p = torch.exp(scores - m_new)
        denom_new = denom * correction + p.sum(dim=-1, keepdim=True)
        acc_new = acc * correction + p @ v_tile
        return (m_new, denom_new, acc_new), None

    m0 = torch.full((Q.shape[0], 1), float("-inf"), device=Q.device, dtype=Q.dtype)
    denom0 = torch.zeros((Q.shape[0], 1), device=Q.device, dtype=Q.dtype)
    acc0 = torch.zeros_like(Q)

    (m, denom, acc), _ = for_each_tile(
        body,
        (K, V),
        dims=(0, 0),
        tile_size=SOFTMAX_TILE_SIZE,
        init=(m0, denom0, acc0),
    )
    return acc / denom


def online_softmax_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    tile_size: int = SOFTMAX_TILE_SIZE,
) -> torch.Tensor:
    """Eager fp32 online-softmax, looped in Python -- isolates fixture-math bugs.

    Computed once here and compared against a straightforward softmax(Q @
    K.T) @ V, so a mismatch between "compiled online_softmax_fn" and this
    function isolates a lowering bug, while a mismatch between this function
    and the straightforward softmax would indicate a fixture-math bug caught
    before it ever reaches the compiler.
    """
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    lk = Kf.shape[0]
    m = torch.full((Qf.shape[0], 1), float("-inf"), dtype=torch.float32)
    denom = torch.zeros((Qf.shape[0], 1), dtype=torch.float32)
    acc = torch.zeros_like(Qf)
    for start in range(0, lk, tile_size):
        k_tile = Kf[start : start + tile_size]
        v_tile = Vf[start : start + tile_size]
        scores = Qf @ k_tile.transpose(-1, -2)
        m_new = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
        correction = torch.exp(m - m_new)
        p = torch.exp(scores - m_new)
        denom = denom * correction + p.sum(dim=-1, keepdim=True)
        acc = acc * correction + p @ v_tile
        m = m_new
    naive = torch.softmax(Qf @ Kf.transpose(-1, -2), dim=-1) @ Vf
    torch.testing.assert_close(acc / denom, naive, atol=1e-2, rtol=1e-2)
    return acc / denom


PAGE_POOL, PAGE_BLOCKS, PAGE_SIZE, PAGE_HS, PAGE_LQ = 8, 4, 32, 64, 32
INT32_ELEMS_PER_STICK = 32
PAGE_ORDER = (5, 2, 7, 0)


def paged_gather_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pages, block table, query) for paged attention's in-body page gather.

    The table mirrors what spyre-inference's paged attention passes: one
    stick-wide int32 row per active block, page index at column 0.
    """
    torch.manual_seed(0)
    pages = torch.randn(PAGE_POOL, PAGE_SIZE, PAGE_HS, dtype=torch.float16)
    q = torch.randn(PAGE_LQ, PAGE_HS, dtype=torch.float16)
    table = torch.zeros(PAGE_BLOCKS, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    for i, page in enumerate(PAGE_ORDER):
        table[i, 0] = page
    return pages, table, q


def paged_gather_fn(
    pages: torch.Tensor, table: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Case 3 (Kind.GATHER): gather one page per trip from inside the body.

    The tiled operand is the block table, not the pages: the body reads its
    own page index out of the tile and gathers with it, so the whole page pool
    stays invariant and only one page is live per trip. That index is a POINT
    read -- one int32 element whose address moves with the spliced loop var and
    which carries no iteration dim of its own -- so it is the shape coarse
    tiling handles through squeezed_advance_per_read rather than a tiled dim
    (see coarse_tile._point_splice_advance_for_dep). Pre-gathering the pages
    ahead of the loop would compile without any of that, at the cost of keeping
    the whole sequence's K/V live across it.
    """

    def body(acc, tiles):
        table_row, pages_all, q_whole = tiles
        page_idx = table_row[0, 0:1]
        page = pages_all.index_select(0, page_idx).squeeze(0)
        scores = q_whole @ page.transpose(0, 1)
        return acc + scores @ page, None

    acc0 = torch.zeros(PAGE_LQ, PAGE_HS, device=q.device, dtype=q.dtype)
    final, _ = for_each_tile(
        body,
        (table, pages, q),
        dims=(0, None, None),
        tile_size=1,
        init=acc0,
    )
    return final


def paged_gather_reference(pages: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The same accumulation in fp32 on CPU, looped in Python over PAGE_ORDER."""
    pf, qf = pages.float(), q.float()
    acc = torch.zeros(PAGE_LQ, PAGE_HS)
    for p in PAGE_ORDER:
        page = pf[p]
        acc = acc + (qf @ page.transpose(0, 1)) @ page
    return acc


def paged_gather_kv_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """(k pages, v pages, block table, query) -- two pools, one shared index."""
    torch.manual_seed(0)
    k_pages = torch.randn(PAGE_POOL, PAGE_SIZE, PAGE_HS, dtype=torch.float16)
    v_pages = torch.randn(PAGE_POOL, PAGE_SIZE, PAGE_HS, dtype=torch.float16)
    q = torch.randn(PAGE_LQ, PAGE_HS, dtype=torch.float16)
    table = torch.zeros(PAGE_BLOCKS, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    for i, page in enumerate(PAGE_ORDER):
        table[i, 0] = page
    return k_pages, v_pages, table, q


def paged_gather_kv_fn(
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    table: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """paged_gather_fn with separate K and V pools -- ONE marker, TWO consumers.

    The only structural difference from paged_gather_fn is the one that
    matters to _consume_tile_dim_markers: the page index sliced out of the
    tiled block table feeds two ``index_select``s (K's page and V's page)
    instead of one, so the block table's single dim=0 tile_dim_marker ends up
    with two consuming reads, both ComputedBuffer/MemoryDep-shaped. That is
    exactly what spyre-inference's page_attn_kernel does, and it is the shape
    an earlier "expected exactly one consumer" assertion in that pass
    rejected outright.

    Keeping K and V in separate matmuls (Q@K^T, then P@V) matters too: with
    one shared pool, or with both gathers feeding a single elementwise
    expression, Inductor fuses the two gathers into one pointwise
    ComputedBuffer whose two identical marker deps dedupe to a single read --
    which silently does not exercise the multi-consumer path at all.
    """

    def body(acc, tiles):
        table_row, k_all, v_all, q_whole = tiles
        page_idx = table_row[0, 0:1]
        k_page = k_all.index_select(0, page_idx).squeeze(0)
        v_page = v_all.index_select(0, page_idx).squeeze(0)
        scores = q_whole @ k_page.transpose(0, 1)
        return acc + scores @ v_page, None

    acc0 = torch.zeros(PAGE_LQ, PAGE_HS, device=q.device, dtype=q.dtype)
    final, _ = for_each_tile(
        body,
        (table, k_pages, v_pages, q),
        dims=(0, None, None, None),
        tile_size=1,
        init=acc0,
    )
    return final


def paged_gather_kv_reference(
    k_pages: torch.Tensor, v_pages: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """The same accumulation in fp32 on CPU, looped in Python over PAGE_ORDER."""
    kf, vf, qf = k_pages.float(), v_pages.float(), q.float()
    acc = torch.zeros(PAGE_LQ, PAGE_HS)
    for p in PAGE_ORDER:
        acc = acc + (qf @ kf[p].transpose(0, 1)) @ vf[p]
    return acc


@contextlib.contextmanager
def _post_grad_graphs():
    """Capture each post-grad graph right after decompose_scan_to_while_loop runs.

    post_grad_custom_post_pass fires BEFORE that decomposition, so a custom
    pass cannot see the while_loop node; wrapping the decomposition itself
    can. Inductor has no built-in hook for "give me the post-grad graph",
    so this monkey-patches the (now permanent, #4136-landed) decomposition
    function for the duration of one compile.
    """
    import torch._inductor.fx_passes.post_grad as pg

    seen: list[torch.fx.GraphModule] = []
    original = pg.decompose_scan_to_while_loop

    def wrapper(gm):
        out = original(gm)
        seen.append(gm)
        return out

    pg.decompose_scan_to_while_loop = wrapper
    try:
        yield seen
    finally:
        pg.decompose_scan_to_while_loop = original


def capture_post_grad_while_loop(
    fn: Callable[..., torch.Tensor], args: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.fx.GraphModule]:
    """Compile fn(*args); return (output, the post-grad graph module).

    Asserts the returned graph module contains a real `while_loop` node
    once fully lowered by decompose_scan_to_while_loop -- these fixtures
    exist specifically to drive that node into torch-spyre's
    CustomPreSchedulingPasses.
    """
    torch._dynamo.reset()
    with _post_grad_graphs() as graphs:
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        out, _code = run_and_get_code(compiled, *args)
    assert graphs, "no post-grad graph captured (FX graph cache hit?)"
    gm = graphs[-1]
    found = any(
        "while_loop" in str(node.target)
        for node in gm.graph.nodes
        if node.op == "call_function"
    )
    assert found, "expected a while_loop node in the post-grad graph"
    return out, gm

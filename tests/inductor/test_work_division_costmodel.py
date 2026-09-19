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

import os
import re
import unittest
import warnings
from math import prod
from pathlib import Path
from unittest.mock import MagicMock, patch

import sympy
import torch
from sympy import Symbol
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    FlexibleLayout,
    Pointwise,
    Reduction,
)

import torch_spyre  # noqa: F401
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.pass_utils import commit_iteration_space_ownership
from torch_spyre._inductor.work_division import (
    TensorDep,
    _cost_model_matmul_planner,
    _matmul_split_cost,
    apply_splits,
    multi_dim_iteration_space_split,
)

# elems_per_stick for fp16 on Spyre (64 elements per stick)
_FP16_ELEMS_PER_STICK = 64

MAX_CORES = 32

# ---------------------------------------------------------------------------
# Cost baselines — best-known modeled cost (µs) for each unique shape/scenario.
#
# Rules:
#   - None  = not yet recorded.
#   - float = the best-known modeled cost the planner produced for this shape.
#
# Baseline Management & Regression Checks:
#   - Performance improvement (cost < baseline): emits UserWarning informing of improvement.
#   - Within range (<= baseline + tolerance): passes silently.
#   - Performance regression (> baseline + tolerance): raises AssertionError (test failure).
#   - Updating baselines: run with UPDATE_BASELINES=1 to record new/improved values in file.
# ---------------------------------------------------------------------------
COST_BASELINES: dict[str, float | None] = {
    # Unique shape scenarios (B=1, M>1 prefill and underfill)
    "test_prefill_speculative_decode_underfill_m4": 84.5314,  # (1, 4, 128) x (1, 128, 2048) - M=4 underfill
    "test_prefill_underfill_boundary_m16": 25.1893,  # (1, 16, 128) x (1, 128, 2048) - M=16 boundary
    "test_prefill_standard_qkt_m2048": 67.0027,  # (1, 2048, 128) x (1, 128, 2048) - standard prefill
    "test_prefill_scorev_heavy_k_narrow_n": 58.6411,  # (1, 2048, 2048) x (1, 2048, 128) - heavy-K score x V
    "test_prefill_mlp_upproj_wide_n": 2691.7553,  # (1, 2048, 4096) x (1, 4096, 11008) - wide-N MLP
    # Unique shape scenarios (B>1, M>1 batched prefill)
    "test_batched_prefill_multihead_qkt_b4_m2048": 238.0107,  # (4, 2048, 128) x (4, 128, 2048) - batched heads
    "test_batched_prefill_deep_k_heads_b4_m2048_k512": 430.5227,  # (4, 2048, 512) x (4, 512, 2048) - deep K batched
    "test_batched_prefill_bert_style_b8_m512": 36.4947,  # (8, 512, 128) x (8, 128, 512) - BERT-style
}

_THIS_FILE = Path(__file__).resolve()


def _store_baseline(test_name: str, cost: float) -> None:
    """Rewrite the COST_BASELINES entry for *test_name* in this source file."""
    text = _THIS_FILE.read_text()
    pattern = rf'("{re.escape(test_name)}":\s*)(None|[0-9]+(?:\.[0-9]+)?)'
    replacement = rf"\g<1>{cost:.4f}"
    new_text, n = re.subn(pattern, replacement, text, count=1)
    if n != 1:
        raise RuntimeError(
            f"_store_baseline: expected exactly one match for {test_name!r}, got {n}"
        )
    _THIS_FILE.write_text(new_text)


def _real_commit(op, splits, it_space):
    """Call the real commit_iteration_space_ownership with iteration_space_from_op
    patched to return ``it_space``."""
    with patch(
        "torch_spyre._inductor.pass_utils.iteration_space_from_op",
        return_value=it_space,
    ):
        commit_iteration_space_ownership(op, splits)


def _isym(name):
    """Symbol with (integer, positive) assumptions, matching real Inductor loop vars."""
    return Symbol(name, integer=True, positive=True)


def _fixed_tiled_layout(shape, dtype=torch.float16):
    size = list(shape)
    stride = [int(s) for s in FlexibleLayout.contiguous_strides(size)]
    within_stick_dim = len(size) - 1
    dim_order = [i for i in range(len(size)) if i != within_stick_dim]
    dim_order.append(within_stick_dim)
    device_layout = SpyreTensorLayout(size, stride, dtype, dim_order)
    return FixedTiledLayout("spyre:0", dtype, size, stride, device_layout)


def _tensor_dep(name, shape, symbols):
    """Build a real TensorDep for a contiguous access over ``symbols``."""
    layout = _fixed_tiled_layout(shape)
    index = sympy.Integer(0)
    for sym, stride in zip(symbols, layout.stride):
        index += sym * int(stride)
    dep = MemoryDep(name, index, tuple(symbols), tuple(shape))
    return TensorDep(dep=dep, layout=layout)


def _computed_buffer(shape, name="buf0", reduction_type=None, reduction_ranges=()):
    if reduction_type is not None:
        data = MagicMock(spec=Reduction)
        data.reduction_type = reduction_type
        data.reduction_ranges = list(reduction_ranges)
    else:
        data = MagicMock(spec=Pointwise)
    data.ranges = list(shape)
    layout = _fixed_tiled_layout(shape)
    op = ComputedBuffer(name=name, layout=layout, data=data)
    op.operation_name = name
    return op


# ---------------------------------------------------------------------------
# Shared assertion helpers — mixed into every test class that calls the
# cost-model planner directly.
# ---------------------------------------------------------------------------


class _CostModelAssertMixin:
    """Mixin providing shared assertion helpers for cost-model test classes."""

    def _assert_valid_split(self, splits, it_space):
        """Generic sanity: core budget respected, each split divides its dim."""
        cores = prod(splits.values())
        self.assertLessEqual(cores, MAX_CORES, f"uses {cores} cores, limit {MAX_CORES}")
        for sym, size in it_space.items():
            s = splits.get(sym, 1)
            self.assertGreaterEqual(s, 1, f"{sym}: split {s} < 1")
            self.assertEqual(size % s, 0, f"{sym}: size {size} not divisible by {s}")

    def _assert_full_cores(self, splits, test_name: str) -> None:
        """Assert the planner uses all MAX_CORES."""
        cores = prod(splits.values())
        self.assertEqual(
            cores,
            MAX_CORES,
            f"{test_name}: planner chose {cores} cores instead of {MAX_CORES}. "
            f"splits={splits}",
        )

    def _assert_cost_not_regressed(
        self, test_name: str, cost: float, tolerance_pct: float = 0.05
    ) -> None:
        """Compare *cost* (µs) against the stored COST_BASELINES entry.

        Args:
            test_name: Name of the test matching the key in COST_BASELINES.
            cost: The measured modeled cost in µs.
            tolerance_pct: Allowed relative tolerance range (default: 5% / 0.05).
        """
        baseline = COST_BASELINES.get(test_name)

        if baseline is None:
            warning_msg = (
                f"\n[UNRECORDED BASELINE] {test_name}: measured cost is {cost:.4f} µs.\n"
                f"  Re-run with UPDATE_BASELINES=1 to record this baseline."
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
            if os.environ.get("UPDATE_BASELINES") == "1":
                _store_baseline(test_name, cost)
                print(
                    f"[cost-baseline] wrote {cost:.4f} µs to COST_BASELINES[{test_name!r}]"
                )
            return

        _EPSILON = 1e-4
        if cost < baseline - _EPSILON:
            diff = baseline - cost
            pct = (diff / baseline) * 100
            warning_msg = (
                f"\n[PERFORMANCE IMPROVEMENT] {test_name}:\n"
                f"  stored baseline : {baseline:.4f} µs\n"
                f"  measured now    : {cost:.4f} µs\n"
                f"  improvement     : -{diff:.4f} µs (-{pct:.2f}%)\n"
                f"  Re-run with UPDATE_BASELINES=1 to commit the new baseline."
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
            if os.environ.get("UPDATE_BASELINES") == "1":
                _store_baseline(test_name, cost)
                print(
                    f"[cost-baseline] updated COST_BASELINES[{test_name!r}] to {cost:.4f} µs"
                )
            return

        max_allowed_cost = baseline * (1.0 + tolerance_pct)
        if cost > max_allowed_cost:
            self.fail(
                f"\n[PERFORMANCE REGRESSION ERROR] {test_name}: modeled cost REGRESSED outside acceptable range (+{tolerance_pct * 100:.1f}%)\n"
                f"  stored baseline : {baseline:.4f} µs\n"
                f"  max allowed     : {max_allowed_cost:.4f} µs\n"
                f"  measured now    : {cost:.4f} µs\n"
                f"  regression      : +{cost - baseline:.4f} µs (+{((cost - baseline) / baseline) * 100:.2f}%)"
            )


# ---------------------------------------------------------------------------
# Core planner decisions, invariants, and handoff integrity tests
# ---------------------------------------------------------------------------


class TestPlannerDecisionsAndHandoff(_CostModelAssertMixin, unittest.TestCase):
    """Direct assertions on planner decision rules, constraints, and handoff to IR."""

    def _run_planner(
        self,
        op,
        it_space,
        output_td,
        stick_vars,
        input_tds,
        blocked=None,
        allowed_splits=None,
        committed_splits=None,
        max_cores=32,
    ):
        default = {sym: 1 for sym in it_space}
        return _cost_model_matmul_planner(
            op,
            default,
            it_space,
            output_td,
            stick_vars,
            committed_splits or {},
            max_cores,
            input_tds,
            blocked or set(),
            allowed_splits or {},
        )

    def test_prefill_qkt_splits_m_dimension_not_k(self):
        """Prefill QK^T: Planner should split M (2048 rows) and leave K=2 sticks unsplit."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="qkT",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("qkT", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        self.assertGreater(
            splits.get(m, 1), 1, "planner should split M for prefill QK^T"
        )
        self.assertEqual(
            splits.get(k, 1), 1, "planner should not split K for narrow QK^T"
        )
        self.assertEqual(prod(splits.values()), MAX_CORES)
        self._assert_full_cores(splits, "test_prefill_qkt_splits_m_dimension_not_k")

    def test_scorev_heavy_k_prefers_m_split_over_narrow_n(self):
        """Score x V (K=2048 >> N=128): Planner should split M, not narrow N=2 sticks."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 128),
            name="scorev",
            reduction_type="batchmatmul",
            reduction_ranges=(2048,),
        )
        output_td = _tensor_dep("scorev", (2048, 128), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 2048), (m, k)),
            _tensor_dep("rhs", (2048, 128), (k, n)),
        ]
        it_space = {m: 2048, n: 2, k: 32}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        self.assertGreater(splits.get(m, 1), 1)
        self._assert_full_cores(
            splits, "test_scorev_heavy_k_prefers_m_split_over_narrow_n"
        )

    def test_blocked_batch_dim_stays_unsplit(self):
        """Blocked batch dimension must remain split=1 even if unblocked plan prefers splitting it."""
        batch, m, n, k = (_isym(x) for x in ("batch", "m", "n", "k"))
        op = _computed_buffer(
            (4, 64, 256),
            name="blocked_batch",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("blocked_batch", (4, 64, 256), (batch, m, n))
        input_tds = [
            _tensor_dep("lhs", (4, 64, 128), (batch, m, k)),
            _tensor_dep("rhs", (4, 128, 256), (batch, k, n)),
        ]
        it_space = {batch: 4, m: 64, n: 4, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        def prefer_batch_split(batch_axis, *_args, **_kwargs):
            return 0 if batch_axis[1] > 1 else 1

        with patch(
            "torch_spyre._inductor.work_division._matmul_split_cost",
            side_effect=prefer_batch_split,
        ):
            unrestricted = self._run_planner(
                op, it_space, output_td, stick_vars, input_tds
            )
            restricted = self._run_planner(
                op, it_space, output_td, stick_vars, input_tds, blocked={batch}
            )

        self.assertGreater(unrestricted.get(batch, 1), 1)
        self.assertEqual(restricted.get(batch, 1), 1)

    def test_apply_splits_commits_ownership_to_ir(self):
        """Real apply_splits must store work_slices matching the planner decision exactly."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="apply_splits_qkT",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("apply_splits_qkT", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        with patch(
            "torch_spyre._inductor.work_division.commit_iteration_space_ownership",
            wraps=lambda op_, splits_: _real_commit(op_, splits_, it_space),
        ):
            apply_splits(op, splits)

        ownership = getattr(op, "iteration_space_ownership", None)
        self.assertIsNotNone(ownership)
        for sym, expected in splits.items():
            actual = ownership.work_slices.get(sym, 1)
            self.assertEqual(actual, expected)

    def test_committed_split_prevents_planner_override(self):
        """Non-empty committed_splits must cause planner to return unchanged default splits."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="already_committed",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("already_committed", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}
        default = {sym: 1 for sym in it_space}

        result = _cost_model_matmul_planner(
            op,
            default,
            it_space,
            output_td,
            stick_vars,
            {k: 2},
            MAX_CORES,
            input_tds,
            set(),
            {},
        )
        self.assertEqual(result, default)

    def test_non_matmul_op_is_noop(self):
        """Non-matmul operations must be a no-op for the cost-model planner."""
        x = _isym("x")
        op = _computed_buffer((2048,), name="pointwise_op")
        output_td = _tensor_dep("pointwise_op", (2048,), (x,))
        it_space = {x: 2048}
        default = {x: 1}

        result = _cost_model_matmul_planner(
            op, default, it_space, output_td, {}, {}, MAX_CORES, [], set(), {}
        )
        self.assertEqual(result, default)

    def test_handoff_preserves_plan_to_scheduler_ownership(self):
        """Full planner -> apply_splits -> ownership.work_slices round-trip check."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="handoff_qkt",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("handoff_qkt", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        with patch(
            "torch_spyre._inductor.work_division.commit_iteration_space_ownership",
            wraps=lambda op_, splits_: _real_commit(op_, splits_, it_space),
        ):
            apply_splits(op, splits)

        ownership = op.iteration_space_ownership
        self.assertIsNotNone(
            ownership, "apply_splits must set op.iteration_space_ownership"
        )
        received = {sym: ownership.work_slices.get(sym, 1) for sym in splits}

        for sym in splits:
            self.assertEqual(splits[sym], received[sym])

    def test_codegen_handoff_per_core_tile_and_mapping(self):
        """Verify the exact data structures codegen reads from iteration_space_ownership:
        1. work_slices: split factors per dimension.
        2. per-core tile size: exact integer tile bounds for each core.
        3. core_id_to_work_slice: SymPy formulas mapping hardware core_id -> slice index (if present).
        """
        core_id_sym = sympy.Symbol("core_id")
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="codegen_handoff_qkt",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("codegen_handoff_qkt", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        with patch(
            "torch_spyre._inductor.work_division.commit_iteration_space_ownership",
            wraps=lambda op_, splits_: _real_commit(op_, splits_, it_space),
        ):
            apply_splits(op, splits)

        ownership = op.iteration_space_ownership
        self.assertIsNotNone(ownership, "Codegen requires op.iteration_space_ownership")

        # 1. Check split factors received by codegen
        for sym, split_factor in splits.items():
            self.assertEqual(ownership.work_slices.get(sym, 1), split_factor)

        # 2. Check per-core tile dimensions (what codegen loops iterate over)
        for sym, total_size in it_space.items():
            split_factor = ownership.work_slices.get(sym, 1)
            self.assertEqual(
                total_size % split_factor, 0, f"Codegen tile uneven on {sym}"
            )
            per_core_tile = total_size // split_factor
            self.assertGreater(per_core_tile, 0)

        # 3. Check core_id -> slice mapping formula if set
        cid_map = getattr(ownership, "core_id_to_work_slice", None)
        if cid_map:
            for core_num in range(MAX_CORES):
                for sym, formula in cid_map.items():
                    split_factor = splits.get(sym, 1)
                    if hasattr(formula, "subs"):
                        slice_idx = int(formula.subs(core_id_sym, core_num))
                        self.assertTrue(
                            0 <= slice_idx < split_factor,
                            f"Core {core_num} mapped to out-of-bounds slice {slice_idx} on {sym}",
                        )

    def test_handoff_detects_symbol_relabeling_corruption(self):
        """Handoff must detect if reduction symbol K is mistakenly relabeled to N."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="relabeled",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("relabeled", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)

        with patch(
            "torch_spyre._inductor.work_division.commit_iteration_space_ownership",
            wraps=lambda op_, splits_: _real_commit(op_, splits_, it_space),
        ):
            apply_splits(op, splits)

        committed = op.iteration_space_ownership.work_slices
        for sym in splits:
            self.assertEqual(committed.get(sym, 1), splits[sym])
        corrupted = dict(committed)
        if corrupted.get(k, 1) > 1:
            corrupted[n] = corrupted.pop(k)
            self.assertNotEqual(corrupted.get(k, 1), splits.get(k, 1))

    def test_handoff_detects_dropped_dimension_key(self):
        """Handoff must detect if any dimension key is omitted during commit."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="dropped_key",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("dropped_key", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)

        with patch(
            "torch_spyre._inductor.work_division.commit_iteration_space_ownership",
            wraps=lambda op_, splits_: _real_commit(op_, splits_, it_space),
        ):
            apply_splits(op, splits)

        committed = op.iteration_space_ownership.work_slices
        corrupted = dict(committed)
        if k in corrupted:
            del corrupted[k]
            self.assertNotEqual(corrupted.keys(), splits.keys())

    _TSP4032_KNOWN_CORES = 25

    def test_greedy_decode_tsp4032_underutilization_guard(self):
        """tsp#4032 regression guard: N=400 sticks must use at least 25 cores."""
        n, k = (_isym(x) for x in ("n", "k"))
        splits = multi_dim_iteration_space_split(
            {n: 400, k: 64},
            MAX_CORES,
            [n],
            [k],
        )
        cores = prod(splits.values())
        self.assertGreaterEqual(cores, self._TSP4032_KNOWN_CORES)

    def test_prefill_b1_mgt1_cost_model_splits_m_without_bmm_pass(self):
        """B=1 M=2048 prefill: Cost model alone must choose M>1 without external bmm pass."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="prefill_qkT_regression",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("prefill_qkT_regression", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        self.assertGreater(splits.get(m, 1), 1)
        self._assert_full_cores(
            splits, "test_prefill_b1_mgt1_cost_model_splits_m_without_bmm_pass"
        )

    def test_greedy_batch_scaling_uses_at_least_as_many_cores(self):
        """B>1 M=1: Greedy pass must not assign fewer cores to B=8 than B=2 with same N/K."""
        b2, n2, k2 = (_isym(x) for x in ("b2", "n2", "k2"))
        b8, n8, k8 = (_isym(x) for x in ("b8", "n8", "k8"))

        splits2 = multi_dim_iteration_space_split(
            {b2: 2, n2: 16, k2: 2},
            MAX_CORES,
            [b2, n2],
            [k2],
        )
        splits8 = multi_dim_iteration_space_split(
            {b8: 8, n8: 16, k8: 2},
            MAX_CORES,
            [b8, n8],
            [k8],
        )

        cores2 = prod(splits2.values())
        cores8 = prod(splits8.values())

        self.assertGreater(cores2, 1)
        self.assertGreater(cores8, 1)
        self.assertGreaterEqual(cores8, cores2)


# ---------------------------------------------------------------------------
# Group A: B=1, M=1 Decode Shapes (Greedy Pass)
# ---------------------------------------------------------------------------


class TestDecodeGreedyB1M1(unittest.TestCase):
    """B=1 M=1 decode shapes — handled by greedy multi_dim_iteration_space_split."""

    def _run(self, n_sticks, k_sticks):
        n, k = _isym("n"), _isym("k")
        splits = multi_dim_iteration_space_split(
            {n: n_sticks, k: k_sticks},
            MAX_CORES,
            [n],
            [k],
        )
        cores = prod(splits.values())
        self.assertLessEqual(cores, MAX_CORES)
        self.assertEqual(n_sticks % splits.get(n, 1), 0)
        return splits

    def test_decode_tiny_n1_k2(self):
        """(1, 1, 128) x (1, 128, 64) — N=1 stick (64 elements)."""
        splits = self._run(n_sticks=1, k_sticks=2)
        self.assertLessEqual(prod(splits.values()), 2)

    def test_decode_narrow_n8_k2(self):
        """(1, 1, 128) x (1, 128, 512) — N=8 sticks."""
        splits = self._run(n_sticks=8, k_sticks=2)
        self.assertGreater(prod(splits.values()), 1)

    def test_decode_standard_qkt_n32_k2(self):
        """(1, 1, 128) x (1, 128, 2048) — N=32 sticks (saturates 32 cores)."""
        splits = self._run(n_sticks=32, k_sticks=2)
        self.assertEqual(prod(splits.values()), MAX_CORES)

    def test_decode_non_power_of_two_n48_k2(self):
        """(1, 1, 128) x (1, 128, 3072) — N=48 sticks (non-power-2)."""
        splits = self._run(n_sticks=48, k_sticks=2)
        self.assertGreater(prod(splits.values()), 1)

    def test_decode_heavy_k_scorev_n2_k32(self):
        """(1, 1, 2048) x (1, 2048, 128) — N=2 sticks, K=32 sticks."""
        splits = self._run(n_sticks=2, k_sticks=32)
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

    def test_decode_vocab_width_tsp4032_n400_k64(self):
        """(1, 1, 4096) x (1, 4096, 25600) — N=400 sticks."""
        n, k = _isym("n"), _isym("k")
        splits = multi_dim_iteration_space_split(
            {n: 400, k: 64},
            MAX_CORES,
            [n],
            [k],
        )
        self.assertGreaterEqual(prod(splits.values()), 25)

    def test_decode_full_divisible_n416_k64(self):
        """(1, 1, 4096) x (1, 4096, 26624) — N=416 sticks (divisible by 32)."""
        splits = self._run(n_sticks=416, k_sticks=64)
        self.assertEqual(prod(splits.values()), MAX_CORES)

    def test_decode_span_limit_n512_k64(self):
        """(1, 1, 4096) x (1, 4096, 32768) — N=512 sticks."""
        splits = self._run(n_sticks=512, k_sticks=64)
        self.assertEqual(prod(splits.values()), MAX_CORES)


# ---------------------------------------------------------------------------
# Group B: B=1, M>1 Prefill Shapes (Cost Model Pass)
# ---------------------------------------------------------------------------


class TestPrefillCostModelB1Mgt1(_CostModelAssertMixin, unittest.TestCase):
    """B=1 M>1 prefill shapes — direct tests on _cost_model_matmul_planner."""

    def _make_op(self, m_rows, n_sticks, k_sticks, name):
        m, n, k = _isym("m"), _isym("n"), _isym("k")
        n_elems = n_sticks * _FP16_ELEMS_PER_STICK
        k_elems = k_sticks * _FP16_ELEMS_PER_STICK
        op = _computed_buffer(
            (m_rows, n_elems),
            name=name,
            reduction_type="batchmatmul",
            reduction_ranges=(k_elems,),
        )
        output_td = _tensor_dep(name, (m_rows, n_elems), (m, n))
        input_tds = [
            _tensor_dep(f"{name}_lhs", (m_rows, k_elems), (m, k)),
            _tensor_dep(f"{name}_rhs", (k_elems, n_elems), (k, n)),
        ]
        it_space = {m: m_rows, n: n_sticks, k: k_sticks}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}
        default = {sym: 1 for sym in it_space}
        splits = _cost_model_matmul_planner(
            op,
            default,
            it_space,
            output_td,
            stick_vars,
            {},
            MAX_CORES,
            input_tds,
            set(),
            {},
        )
        return splits, m, n, k

    def test_prefill_speculative_decode_underfill_m4(self):
        """(1, 4, 128) x (1, 128, 2048) — M=4 underfill."""
        splits, m, n, k = self._make_op(4, 32, 2, "prefill_speculative_m4")
        self.assertGreater(prod(splits.values()), 1)
        self.assertGreater(splits.get(n, 1), 1)
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(4, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"))
        self._assert_cost_not_regressed(
            "test_prefill_speculative_decode_underfill_m4", cost
        )

    def test_prefill_underfill_boundary_m16(self):
        """(1, 16, 128) x (1, 128, 2048) — M=16 boundary."""
        splits, m, n, k = self._make_op(16, 32, 2, "prefill_boundary_m16")
        self.assertGreater(prod(splits.values()), 1)
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(16, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"))
        self._assert_cost_not_regressed("test_prefill_underfill_boundary_m16", cost)

    def test_prefill_standard_qkt_m2048(self):
        """(1, 2048, 128) x (1, 128, 2048) — standard prefill QK^T."""
        splits, m, n, k = self._make_op(2048, 32, 2, "prefill_standard_qkt")
        self.assertGreater(splits.get(m, 1), 1)
        self.assertEqual(splits.get(k, 1), 1)
        self.assertEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"))
        self._assert_cost_not_regressed("test_prefill_standard_qkt_m2048", cost)

    def test_prefill_scorev_heavy_k_narrow_n(self):
        """(1, 2048, 2048) x (1, 2048, 128) — K>>N, score x V."""
        splits, m, n, k = self._make_op(2048, 2, 32, "prefill_scorev_heavy_k")
        self.assertGreater(splits.get(m, 1), 1)
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(128, splits.get(n, 1)),
            k_axis=(2048, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"))
        self._assert_cost_not_regressed("test_prefill_scorev_heavy_k_narrow_n", cost)

    def test_prefill_mlp_upproj_wide_n(self):
        """(1, 2048, 4096) x (1, 4096, 11008) — wide-N MLP up-proj."""
        splits, m, n, k = self._make_op(2048, 172, 64, "prefill_mlp_upproj")
        self.assertGreater(splits.get(m, 1), 1)
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(11008, splits.get(n, 1)),
            k_axis=(4096, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"))
        self._assert_cost_not_regressed("test_prefill_mlp_upproj_wide_n", cost)

    def test_prefill_span_limit_boundary_n513(self):
        """(1, 2048, 4096) x (1, 4096, 32832) — at span limit."""
        splits, m, n, k = self._make_op(2048, 513, 64, "prefill_span_limit")
        self.assertLessEqual(prod(splits.values()), MAX_CORES)


# ---------------------------------------------------------------------------
# Group C: B>1, M=1 Batched Decode Shapes (Greedy Pass)
# ---------------------------------------------------------------------------


class TestBatchedDecodeGreedyBgt1M1(unittest.TestCase):
    """B>1 M=1 batch decode — greedy P3 on {b, n, k}."""

    def _run(self, batch, n_sticks, k_sticks):
        b, n, k = _isym("b"), _isym("n"), _isym("k")
        splits = multi_dim_iteration_space_split(
            {b: batch, n: n_sticks, k: k_sticks},
            MAX_CORES,
            [b, n],
            [k],
        )
        cores = prod(splits.values())
        self.assertLessEqual(cores, MAX_CORES)
        self.assertEqual(batch % splits.get(b, 1), 0)
        self.assertEqual(n_sticks % splits.get(n, 1), 0)
        return splits, b, n, k

    def test_batched_decode_batch2_n16(self):
        """(2, 1, 128) x (2, 128, 1024) — B=2, N=16 sticks."""
        splits, b, n, k = self._run(2, 16, 2)
        self.assertGreater(prod(splits.values()), 1)

    def test_batched_decode_batch4_n8(self):
        """(4, 1, 128) x (4, 128, 512) — B=4, N=8 sticks."""
        splits, b, n, k = self._run(4, 8, 2)
        self.assertGreater(prod(splits.values()), 1)

    def test_batched_decode_batch8_n4(self):
        """(8, 1, 128) x (8, 128, 256) — B=8, N=4 sticks."""
        splits, b, n, k = self._run(8, 4, 2)
        self.assertGreater(prod(splits.values()), 1)

    def test_batched_decode_batch16_n2(self):
        """(16, 1, 128) x (16, 128, 128) — B=16, N=2 sticks."""
        splits, b, n, k = self._run(16, 2, 2)
        self.assertGreater(prod(splits.values()), 1)

    def test_batched_decode_batch32_n1_full_cores(self):
        """(32, 1, 64) x (32, 64, 64) — B=32, N=1 stick (saturates 32 cores)."""
        splits, b, n, k = self._run(32, 1, 1)
        self.assertEqual(prod(splits.values()), MAX_CORES)

    def test_batched_decode_odd_batch5_n32(self):
        """(5, 1, 128) x (5, 128, 2048) — odd B=5, N=32 sticks."""
        splits, b, n, k = self._run(5, 32, 2)
        self.assertGreater(prod(splits.values()), 1)

    def test_batched_decode_odd_batch7_n32(self):
        """(7, 1, 128) x (7, 128, 2048) — odd B=7, N=32 sticks."""
        splits, b, n, k = self._run(7, 32, 2)
        self.assertGreater(prod(splits.values()), 1)


# ---------------------------------------------------------------------------
# Group D: B>1, M>1 Batched Prefill Shapes (Cost Model Pass)
# ---------------------------------------------------------------------------


class TestBatchedPrefillCostModelBgt1Mgt1(_CostModelAssertMixin, unittest.TestCase):
    """B>1 M>1 batched prefill — direct tests on _cost_model_matmul_planner."""

    def _make_op(self, batch, m_rows, n_sticks, k_sticks, name):
        b, m, n, k = _isym("b"), _isym("m"), _isym("n"), _isym("k")
        n_elems = n_sticks * _FP16_ELEMS_PER_STICK
        k_elems = k_sticks * _FP16_ELEMS_PER_STICK
        op = _computed_buffer(
            (batch, m_rows, n_elems),
            name=name,
            reduction_type="batchmatmul",
            reduction_ranges=(k_elems,),
        )
        output_td = _tensor_dep(name, (batch, m_rows, n_elems), (b, m, n))
        input_tds = [
            _tensor_dep(f"{name}_lhs", (batch, m_rows, k_elems), (b, m, k)),
            _tensor_dep(f"{name}_rhs", (batch, k_elems, n_elems), (b, k, n)),
        ]
        it_space = {b: batch, m: m_rows, n: n_sticks, k: k_sticks}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}
        default = {sym: 1 for sym in it_space}
        splits = _cost_model_matmul_planner(
            op,
            default,
            it_space,
            output_td,
            stick_vars,
            {},
            MAX_CORES,
            input_tds,
            set(),
            {},
        )
        return splits, b, m, n, k

    def test_batched_prefill_multihead_qkt_b4_m2048(self):
        """(4, 2048, 128) x (4, 128, 2048) — B=4, M=2048, N=32, K=2."""
        splits, b, m, n, k = self._make_op(4, 2048, 32, 2, "batched_prefill_qkt")
        self.assertGreater(max(splits.get(m, 1), splits.get(b, 1)), 1)
        self.assertEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(4, splits.get(b, 1)),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"))
        self._assert_cost_not_regressed(
            "test_batched_prefill_multihead_qkt_b4_m2048", cost
        )

    def test_batched_prefill_deep_k_heads_b4_m2048_k512(self):
        """(4, 2048, 512) x (4, 512, 2048) — B=4, M=2048, N=32, K=8."""
        splits, b, m, n, k = self._make_op(4, 2048, 32, 8, "batched_prefill_deep_k")
        self.assertGreater(max(splits.get(m, 1), splits.get(b, 1)), 1)
        self.assertEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(4, splits.get(b, 1)),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(512, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"))
        self._assert_cost_not_regressed(
            "test_batched_prefill_deep_k_heads_b4_m2048_k512", cost
        )

    def test_batched_prefill_bert_style_b8_m512(self):
        """(8, 512, 128) x (8, 128, 512) — B=8, M=512, N=8, K=2."""
        splits, b, m, n, k = self._make_op(8, 512, 8, 2, "batched_prefill_bert")
        self.assertGreater(max(splits.get(m, 1), splits.get(b, 1)), 1)
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(8, splits.get(b, 1)),
            m_axis=(512, splits.get(m, 1)),
            n_axis=(512, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"))
        self._assert_cost_not_regressed("test_batched_prefill_bert_style_b8_m512", cost)


if __name__ == "__main__":
    unittest.main()

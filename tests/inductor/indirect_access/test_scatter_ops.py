# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Scatter-family and related indirect writes: scatter / scatter_add /
scatter_reduce, index_reduce, index_fill, masked_fill, and put.

Multi-sencores is compiled-only: dests whose indexed dim can split across cores
request patch_sencores (eager once at default config; compiled at 1 and chip-max).
Small dests stay eager + compiled at default config. Eager is never a SENCORES variant.
"""

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor.exc import InductorError

from utils_inductor import DEVICE, cached_randn, compare_with_cpu
from conftest import (
    gap_one_core,
    _xfail_existing,
    compare_mode,
    expect_dynamic_true_refused,
    rejection,
)

# fp16 compute after a gather (unary, binary, small reductions).  The largest error
# measured against an fp32 reference is 5.1e-3 (fp16's own compute error: up to 1.7e-3),
# so 1e-2 leaves about 2x margin.  Pure data movement uses atol=0 instead.
_TOL_F16 = 1e-2
# fp32 compute.  The largest error measured against an fp32 CPU reference is 4.3e-7
# (layer norm).
_TOL_F32 = 1e-5


_MASKED_FILL_FP32 = rejection(
    4714,
    "masked_fill fp32: the bool mask cast to fp32 gives a staggered EA whose shuffle "
    "needs whole stick pairs; the stick dim of 32 is not a whole pair.",
    r"requires whole stick pairs",
)
_IR_DUP_EAGER = (
    6638,
    "index_reduce_ with a repeated index returns wrong values: the decomposition "
    "overwrites, so only one source row per destination row is kept (unique indices "
    "are a documented precondition).",
    r"eager spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_IR_DUP_COMPILED = (
    6638,
    "compile index_reduce_ with a repeated index returns wrong values: the "
    "decomposition overwrites, so only one source row per destination row is kept "
    "(unique indices are a documented precondition).",
    r"compiled spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_IR_STICK_DIM = rejection(
    4404,
    "index_reduce along the stick (last) dim is rejected: its gather/scatter along the stick does not lower.",
    r"index_reduce along the stick \(last\) dim",
)
_PUT_EAGER = rejection(
    4540,
    "aten::put_ is not registered on Spyre.",
    r"Could not run 'aten::put_'",
)
_PUT_COMPILED = rejection(
    4636,
    "compile put_: AssertionError restickify op_spec has 3 args.",
    r"restickify op_spec has 3 args",
)
_MUTATING_FALLBACK = r"upstream lowers this form to a mutating fallback"


def _scatter_fallback(issue, what):
    """Compiled scatter forms that upstream lowers to scatter_reduce_.two are rejected."""
    return rejection(
        issue,
        f"compile {what} lowers to aten.scatter_reduce_.two, a mutating fallback with no Spyre kernel.",
        _MUTATING_FALLBACK,
    )


_SR_TWO = rejection(
    4361,
    "aten::scatter_reduce.two_out is not registered on Spyre.",
    r"Could not run 'aten::scatter_reduce.two_out'",
)


def _mode_flags(mode: str):
    """Return (run_eager, run_compile) booleans from the 'mode' parameter string."""
    return (mode == "eager", mode == "compile")


# Shared execution-mode parametrizations. Pass each case's issue, exception, and
# failure description here so the XFAIL policy is defined in one place.
def _xfail(issue, exception, reason):
    return pytest.mark.xfail(
        strict=True, raises=exception, reason=f"{reason}, #{issue}"
    )


def _mode_params(*, eager=None, compile=None):
    """Build eager/compile parameters; each XFAIL is (issue, exception, reason)."""
    params = []
    for mode, xfail in (("eager", eager), ("compile", compile)):
        params.append(pytest.param(mode, marks=_xfail(*xfail)) if xfail else mode)
    return tuple(params)


_MODE_PARAMS_SCATTER_EAGER_ONLY_UNSUPPORTED = _mode_params(
    eager=(
        4473,
        NotImplementedError,
        "Spyre eager scatter dispatch is unsupported for this case",
    )
)
_MODE_PARAMS_SCATTER_HIGH_RANK_XFAIL = _mode_params(
    eager=(
        4473,
        NotImplementedError,
        "Spyre eager scatter dispatch is unsupported for this case",
    ),
    compile=(
        6574,
        InductorError,
        "dbo-opt GatherIndexConversion fails its divisibility assertion",
    ),
)
_MODE_PARAMS_SCATTER_ADD_UNSUPPORTED = _mode_params(
    eager=(
        4396,
        NotImplementedError,
        "aten::scatter_add.out has no registered Spyre eager kernel",
    ),
    compile=(
        6575,
        InductorError,
        "dbo-opt scheduler rejects the scatter page-size constraint",
    ),
)
_MODE_PARAMS_SCATTER_ADD_EAGER_ONLY_XFAIL = _mode_params(
    eager=(
        4396,
        NotImplementedError,
        "aten::scatter_add.out has no registered Spyre eager kernel",
    )
)
_MODE_PARAMS_EAGER_COMPILE = ("eager", "compile")
_REDUCE_MODES = ("prod", "mean", "amax", "amin")
_PUT_ACCUMULATE_OPTIONS = (False, True)

_INDEX_REDUCE_NARROW_CASES = tuple(
    pytest.param(
        reduce_mode,
        mode,
        marks=_xfail(issue, InductorError, reason),
    )
    for reduce_mode in _REDUCE_MODES
    for mode, issue, reason in (
        (
            "eager",
            6210,
            "Indirect indexing on a sliced view with a non-zero storage offset fails OpSpec validation",
        ),
        (
            "compile",
            6707,
            "Spyre Inductor rejects the graph because it has 3 inputs but 4 real inputs",
        ),
    )
)

_MODE_PARAMS_INDEX_FILL_UNSUPPORTED = _mode_params(
    eager=(
        4414,
        NotImplementedError,
        "aten::index_fill_.int_Scalar has no registered Spyre eager kernel",
    ),
    compile=(
        4472,
        InductorError,
        "Inductor scatter/index_fill lowering rejects a Constant store value",
    ),
)
_MODE_PARAMS_MASKED_SCATTER_UNSUPPORTED = _mode_params(
    eager=(
        4437,
        NotImplementedError,
        "Spyre eager masked_scatter_ overload has no registered kernel",
    ),
    compile=(
        4437,
        BackendCompilerFailed,
        "Spyre masked_scatter lowering rejects unsupported mask/source indexing",
    ),
)
_MODE_PARAMS_MASKED_SCATTER_ELEMENTWISE = _mode_params(
    eager=(
        4437,
        RuntimeError,
        "Spyre masked_scatter rejects element-level mask/source indexing",
    ),
    compile=(
        4437,
        BackendCompilerFailed,
        "Spyre masked_scatter lowering rejects unsupported mask/source indexing",
    ),
)
_MODE_PARAMS_INDEX_PUT_UNSUPPORTED = _mode_params(
    eager=(
        692,
        NotImplementedError,
        "aten::_index_put_impl_ has no registered Spyre eager kernel",
    ),
    compile=(
        4451,
        InductorError,
        "index_put_ on a 3-D view of a 4-D tensor fails with a stride assertion in Inductor scheduling",
    ),
)
_MODE_PARAMS_PUT_UNSUPPORTED = _mode_params(
    eager=(692, NotImplementedError, "aten::put_ has no registered Spyre eager kernel"),
    compile=(
        6578,
        InductorError,
        "Compiled put_ fails the restickify op_spec three-argument assertion",
    ),
)

_INDEX_REDUCE_OUTPLACE_CASES = [
    (torch.bfloat16, reduce_mode, index_dtype, 0.005, 0.005)
    for reduce_mode in ("prod", "mean", "amax", "amin")
    for index_dtype in (torch.int32, torch.int64)
] + [
    (dtype, reduce_mode, torch.int64, tol, tol)
    for dtype, tol in ((torch.float32, 0.0001), (torch.float16, 0.05))
    for reduce_mode in ("mean", "amax", "amin")
]


def _router_index(T, E, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.empty(T, K, dtype=torch.int64)
    for t in range(T):
        idx[t] = torch.randperm(E, generator=g)[:K]
    return idx


# ---------------------------------------------------------------------------
# index_reduce
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexReduceEagerCompile:
    """index_reduce_ standalone: row-level indexed reduction on dim=0 (mean/prod/amax/amin)."""

    def setup_method(self):
        torch.manual_seed(0xBEEF)

    def _xfail_duplicates(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/6638
        _xfail_existing(execution_mode, eager=_IR_DUP_EAGER, compiled=_IR_DUP_COMPILED)

    def _run(self, execution_mode, dst, src, idx, reduce_op, tol):
        def fn(dst, src, idx):
            return dst.clone().index_reduce_(
                0, idx, src, reduce=reduce_op, include_self=False
            )

        compare_mode(execution_mode, fn, dst, src, idx, atol=tol, rtol=tol)

    def test_index_reduce_mean_dim0(self, execution_mode):
        """index_reduce_ mean on dim=0; duplicate indices average rows into destination.

        PyTorch index_reduce_ does not accept reduce='sum' (use scatter_add for sum).
        The index repeats row 2, and the decomposition overwrites, so the result is wrong
        in eager and compiled (#6638).
        """
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_mean01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_mean01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "mean", _TOL_F16)

    def test_index_reduce_prod_dim0(self, execution_mode):
        """index_reduce_ prod on dim=0; duplicate indices multiply rows into destination."""
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_prod01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_prod01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "prod", _TOL_F16)

    def test_index_reduce_amax_dim0(self, execution_mode):
        """index_reduce_ amax on dim=0; duplicate indices take element-wise max across rows."""
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_amax01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_amax01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "amax", _TOL_F16)

    def test_index_reduce_amin_dim0(self, execution_mode):
        """index_reduce_ amin on dim=0; duplicate indices take element-wise min across rows."""
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_amin01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_amin01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "amin", _TOL_F16)

    def test_index_reduce_mean_float32(self, execution_mode):
        """index_reduce_ mean in float32; validates 4-byte word path for row accumulation."""
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_f32_01", dtype=torch.float32)
        src = cached_randn(
            (3, 8), differentiation="irdc_f32_01_src", dtype=torch.float32
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "mean", _TOL_F32)

    def test_index_reduce_single_index(self, execution_mode):
        """index_reduce_ with a single-element index; boundary case — P=1 row update."""
        dst = cached_randn((8, 16), differentiation="irdc_p1_01", dtype=torch.float16)
        src = cached_randn(
            (1, 16), differentiation="irdc_p1_01_src", dtype=torch.float16
        )
        idx = torch.tensor([3], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "amax", _TOL_F16)

    @pytest.mark.parametrize("reduce_op", ["mean", "prod", "amax", "amin"])
    @pytest.mark.parametrize("include_self", [False, True])
    def test_index_reduce_unique_indices(self, execution_mode, reduce_op, include_self):
        """index_reduce_ with unique indices on dim=0: supported (#6187), matches CPU.

        Every destination row receives at most one source row. Measured against CPU
        fp16 the largest error is 2e-3, so _TOL_F16 leaves 5x margin.
        """
        dst = cached_randn((4, 8), differentiation="irdc_uniq", dtype=torch.float16)
        src = cached_randn((3, 8), differentiation="irdc_uniq_src", dtype=torch.float16)
        idx = torch.tensor([2, 0, 3], dtype=torch.int64)

        def fn(dst, src, idx):
            return dst.clone().index_reduce_(
                0, idx, src, reduce=reduce_op, include_self=include_self
            )

        compare_mode(execution_mode, fn, dst, src, idx, atol=_TOL_F16, rtol=_TOL_F16)

    def test_index_reduce_stick_dim_rejected(self, execution_mode):
        """index_reduce_ along the last (stick) dim is rejected with Unsupported (#4404)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4404
        _xfail_existing(execution_mode, eager=_IR_STICK_DIM, compiled=_IR_STICK_DIM)
        dst = cached_randn((4, 64), differentiation="irdc_stick", dtype=torch.float16)
        src = cached_randn(
            (4, 2), differentiation="irdc_stick_src", dtype=torch.float16
        )
        idx = torch.tensor([1, 5], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, s, i: d.clone().index_reduce_(
                1, i, s, reduce="amax", include_self=False
            ),
            dst,
            src,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )


# ---------------------------------------------------------------------------
# index_fill
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexFillEagerCompile:
    """index_fill_ standalone: fills selected rows/columns with a scalar constant."""

    def setup_method(self):
        torch.manual_seed(0xF111)

    def test_index_fill_dim0_float16(self, execution_mode):
        """index_fill_ on dim=0 with float16; fills rows [0, 2] with 3.0."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4414
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4472
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4414,
                "aten::index_fill_.int_Scalar is not registered on Spyre.",
                r"Could not run 'aten::index_fill_.int_Scalar'",
            ),
            compiled=rejection(
                4472,
                "compile index_fill_ stores a rank-0 Constant.",
                r"store value of unexpected type <class '[\w.]*Constant'>",
            ),
        )
        x = cached_randn((4, 16), differentiation="ifill_f16_01", dtype=torch.float16)
        idx = torch.tensor([0, 2], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x.clone().index_fill_(0, i, 3.0),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_fill_dim1_float16(self, execution_mode):
        """index_fill_ on dim=1 with float16; fills columns [1, 3] with -1.0."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4414
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4414,
                "aten::index_fill_.int_Scalar is not registered on Spyre.",
                r"Could not run 'aten::index_fill_.int_Scalar'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        x = cached_randn((8, 16), differentiation="ifill_f16_02", dtype=torch.float16)
        idx = torch.tensor([1, 3], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x.clone().index_fill_(1, i, -1.0),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_fill_1d_dest_length_8(self, execution_mode):
        """index_fill_ 1-D dest length 8 — compile leftover (#4399)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4414
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4414,
                "aten::index_fill_.int_Scalar is not registered on Spyre.",
                r"Could not run 'aten::index_fill_.int_Scalar'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        x = torch.zeros(8, dtype=torch.float16)
        idx = torch.tensor([0, 2], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda t, i: t.clone().index_fill_(0, i, 1.0),
            x,
            idx,
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# masked_fill (standalone)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestMaskedFillStandaloneEagerCompile:
    """masked_fill as the primary operation on a plain tensor (not downstream after gather)."""

    def setup_method(self):
        torch.manual_seed(0xA55A)

    def test_masked_fill_float16(self, execution_mode):
        """masked_fill on (8,16) float16; every even position receives -inf sentinel.

        Exact (atol=0): masked_fill copies or overwrites, so on dl16-snapped
        inputs the device result is bit-identical to the CPU result.
        """
        x = cached_randn((8, 16), differentiation="mfill_f16_01", dtype=torch.float16)
        mask = torch.arange(8 * 16).reshape(8, 16) % 2 == 0
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, float("-inf")),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_masked_fill_float32(self, execution_mode):
        """masked_fill on (8,16) float32; validates 4-byte word path for mask-based fill."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4714
        _xfail_existing(
            execution_mode, eager=_MASKED_FILL_FP32, compiled=_MASKED_FILL_FP32
        )
        x = cached_randn((8, 16), differentiation="mfill_f32_01", dtype=torch.float32)
        mask = torch.arange(8 * 16).reshape(8, 16) % 2 == 0
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, float("-inf")),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_masked_fill_bfloat16(self, execution_mode):
        """masked_fill on (8,16) bfloat16; dtype parity check for bf16 masked writes."""
        x = cached_randn((8, 16), differentiation="mfill_bf16_01", dtype=torch.bfloat16)
        mask = torch.arange(8 * 16).reshape(8, 16) % 3 == 0
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, 0.0),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_masked_fill_2d_row_mask(self, execution_mode):
        """masked_fill with a column-broadcast mask (shape [8,1]); fills entire rows."""
        x = cached_randn((8, 32), differentiation="mfill_row_01", dtype=torch.float16)
        mask = torch.tensor(
            [True, False, True, False, True, False, True, False]
        ).unsqueeze(1)
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, -1e4),
            x,
            mask,
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# put (flat-indexed write)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestPutFlatIndexWriteEagerCompile:
    """torch.Tensor.put_ standalone: flat-indexed write into the flattened tensor view."""

    def setup_method(self):
        torch.manual_seed(0x9876)

    def _xfail_put(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4540
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4636
        _xfail_existing(execution_mode, eager=_PUT_EAGER, compiled=_PUT_COMPILED)

    def test_put_overwrite_float16(self, execution_mode):
        """put_ accumulate=False on float16; flat positions [0, 5, 11] receive values [1, 2, 3]."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_ow_f16_01", dtype=torch.float16)
        idx = torch.tensor([0, 5, 11], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=False),
            x,
            idx,
            vals,
            atol=0,
            rtol=0,
        )

    def test_put_accumulate_float16(self, execution_mode):
        """put_ accumulate=True on float16; duplicate index [0,0] adds twice to flat pos 0."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_acc_f16_01", dtype=torch.float16)
        idx = torch.tensor([0, 0, 5], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=True),
            x,
            idx,
            vals,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_put_overwrite_float32(self, execution_mode):
        """put_ accumulate=False on float32; validates 4-byte word path for flat writes."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_ow_f32_01", dtype=torch.float32)
        idx = torch.tensor([0, 5, 11], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=False),
            x,
            idx,
            vals,
            atol=0,
            rtol=0,
        )

    def test_put_accumulate_float32(self, execution_mode):
        """put_ accumulate=True on float32; duplicate index [0,0] adds twice to flat pos 0."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_acc_f32_01", dtype=torch.float32)
        idx = torch.tensor([0, 0, 5], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=True),
            x,
            idx,
            vals,
            atol=_TOL_F32,
            rtol=_TOL_F32,
        )

    def test_put_bfloat16(self, execution_mode):
        """put_ accumulate=False on bfloat16; validates 2-byte word flat write."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_bf16_01", dtype=torch.bfloat16)
        idx = torch.tensor([0, 5, 11], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=False),
            x,
            idx,
            vals,
            atol=0,
            rtol=0,
        )

    def test_put_single_position(self, execution_mode):
        """put_ with a single flat index; boundary case P=1.

        Eager is missing aten::put_ (#4540). Compiled matches CPU.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4540
        _xfail_existing(execution_mode, eager=_PUT_EAGER)
        x = cached_randn((4, 8), differentiation="put_p1_01", dtype=torch.float16)
        idx = torch.tensor([7], dtype=torch.int64)
        vals = torch.tensor([-9.0], dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=False),
            x,
            idx,
            vals,
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# scatter_add
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterAddGroupedMmOffs:
    """Gemma grouped_mm offs — int64 dest [128], T=192. Eager #4396; compiled is rejected (#4396).

    Dest length 128 splits across cores; compiled runs at sencores=1 and chip-max.
    """

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    @gap_one_core
    def test_scatter_add_grouped_mm_offs_int64(self, execution_mode):
        """scatter_add dim=0 int64 counts [128], T=192 — Gemma grouped_mm offs (#4396)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4396
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4396,
                "aten::scatter_add.out is not registered on Spyre.",
                r"Could not run 'aten::scatter_add.out'",
            ),
            compiled=_scatter_fallback(4396, "scatter_add int64"),
        )
        G, T = 128, 192
        counts = torch.zeros(G, dtype=torch.int64)
        index = torch.randint(0, G, (T,), dtype=torch.int64)
        src = torch.ones(T, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda c, i, s: c.scatter_add(0, i, s),
            counts,
            index,
            src,
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# torch.scatter / scatter_(value=)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterRouterMaps:
    """Llama 4 K=1 and Qwen3 K=8 densify maps — dest (192, 128). #4403 / #4404 / #4354.

    Dest (192, 128) splits across cores; compiled runs at sencores=1 and chip-max.
    """

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    @gap_one_core
    def test_scatter_llama4_router_k1(self, execution_mode):
        """torch.scatter dim=1 K=1 into -inf dest (192, 128) — Llama 4 router (#4403)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4404
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                1002,
                "aten::scatter.src_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.src_out'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        T, E, K = 192, 128, 1
        dst = torch.full((T, E), float("-inf"), dtype=torch.float16)
        idx = _router_index(T, E, K)
        src = (
            (torch.arange(T, dtype=torch.float32) / 7.0).to(torch.float16).unsqueeze(1)
        )
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter(1, i, s),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_scatter_qwen3_router_k8(self, execution_mode):
        """torch.scatter dim=1 K=8 into zeros dest (192, 128) — Qwen3 MoE router (#4403)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4404
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                1002,
                "aten::scatter.src_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.src_out'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        T, E, K = 192, 128, 8
        dst = torch.zeros(T, E, dtype=torch.float16)
        idx = _router_index(T, E, K)
        src = cached_randn((T, K), differentiation="sc_qwen3", dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter(1, i, s),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterValueStamp:
    """Scalar scatter_ extra dest ranks — grouped_mm offs and KV-cache slots. #4436 / #4472."""

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    @gap_one_core
    def test_scatter_value_grouped_mm_offs(self, execution_mode):
        """scatter_(0, idx, 1.0) onto fp16 dest [128], T=192 — leftover extra (#4436 Python).

        Eager is aten::scatter.value_out (#4473). Compiled fails in
        insert_restickify_padding (#6163): the restickify is written into a
        slice of another buffer, which cannot be padded in place.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4473
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/6163
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4473,
                "aten::scatter.value_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.value_out'",
            ),
            compiled=rejection(
                6163,
                "compile scatter_(value=) on an fp16 dest [128]: insert_restickify_padding cannot pad a restickify written into a slice of another buffer.",
                r"insert_restickify_padding: restickify written into a slice of another buffer",
            ),
        )
        G, T = 128, 192
        dst = torch.zeros(G, dtype=torch.float16)
        idx = torch.randint(0, G, (T,), dtype=torch.int64)

        def fn(d, i):
            out = d.clone()
            out.scatter_(0, i, 1.0)
            return out

        compare_mode(execution_mode, fn, dst, idx, atol=0, rtol=0)

    @gap_one_core
    def test_scatter_value_kv_cache_slots(self, execution_mode):
        """scatter_ stamp 0.0 into bf16 dest (256, 8, 128) at slots [3, 17] (#4436)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4473
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4472
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4473,
                "aten::scatter.value_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.value_out'",
            ),
            compiled=rejection(
                4472,
                "compile scatter_(value=) stores a rank-0 Constant.",
                r"store value of unexpected type <class '[\w.]*Constant'>",
            ),
        )
        dst = cached_randn((256, 8, 128), differentiation="sc_kv", dtype=torch.bfloat16)
        idx = torch.zeros(2, 8, 128, dtype=torch.int64)
        idx[0] = 3
        idx[1] = 17

        def fn(d, i):
            out = d.clone()
            out.scatter_(0, i, 0.0)
            return out

        compare_mode(execution_mode, fn, dst, idx, atol=0, rtol=0)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatter1DDestRestickify:
    """1-D dest below stick size / dest [64] — extra, not a YAML unique-abort name."""

    def test_scatter_1d_dest_length_5_fp32(self, execution_mode):
        """scatter_ dest length 5 fp32 — leftover extra (#4474 Python).

        Eager is aten::scatter.src_out (#1002). Compiled this log was
        normalize_coordinates offset==0 (#4399), not the filed #4474
        insert_restickify_padding string.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                1002,
                "aten::scatter.src_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.src_out'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        x = torch.zeros(5, dtype=torch.float32)
        index = torch.tensor([1, 3], dtype=torch.int32)
        src = torch.tensor([100.0, 200.0], dtype=torch.float32)

        def fn(d, i, s):
            out = d.clone()
            out.scatter_(0, i, s)
            return out

        compare_mode(execution_mode, fn, x, index, src, atol=0, rtol=0)

    @gap_one_core
    def test_scatter_1d_dest_length_64(self, execution_mode, patch_sencores):
        """scatter_ dest [64] fp16 — compile assert offset == 0 (#4399)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                1002,
                "aten::scatter.src_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.src_out'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        x = torch.zeros(64, dtype=torch.float16)
        index = torch.tensor([1, 3], dtype=torch.int64)
        src = torch.tensor([1.0, 2.0], dtype=torch.float16)

        def fn(d, i, s):
            out = d.clone()
            out.scatter_(0, i, s)
            return out

        compare_mode(execution_mode, fn, x, index, src, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# scatter_reduce / scatter_(reduce=)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterReduceLegacyGroupedMm:
    """Legacy Tensor.scatter_(reduce=) on grouped_mm dest ranks. Eager and compiled: #4413.

    Dest length 128 splits across cores; compiled runs at sencores=1 and chip-max.
    """

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    @gap_one_core
    def test_scatter_reduce_add_grouped_mm_int64(self, execution_mode):
        """scatter_(reduce='add') int64 dest [128], T=192 (#4413)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4413
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4413,
                "aten::scatter.reduce_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.reduce_out'",
            ),
            compiled=_scatter_fallback(4413, "scatter_(reduce=)"),
        )
        G, T = 128, 192
        dst = torch.ones(G, dtype=torch.int64)
        idx = torch.randint(0, G, (T,), dtype=torch.int64)
        src = torch.ones(T, dtype=torch.int64)

        def fn(d, i, s):
            out = d.clone()
            out.scatter_(0, i, s, reduce="add")
            return out

        compare_mode(execution_mode, fn, dst, idx, src, atol=0, rtol=0)

    @gap_one_core
    def test_scatter_reduce_multiply_grouped_mm_fp16(self, execution_mode):
        """scatter_(reduce='multiply') fp16 dest [128], T=192; dest starts at ones (#4413)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4413
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4413,
                "aten::scatter.reduce_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.reduce_out'",
            ),
            compiled=_scatter_fallback(4413, "scatter_(reduce=)"),
        )
        G, T = 128, 192
        dst = torch.ones(G, dtype=torch.float16)
        idx = torch.randint(0, G, (T,), dtype=torch.int64)
        src = cached_randn((T,), differentiation="sr_mul_src", dtype=torch.float16) + 1

        def fn(d, i, s):
            out = d.clone()
            out.scatter_(0, i, s, reduce="multiply")
            return out

        compare_mode(execution_mode, fn, dst, idx, src, atol=_TOL_F16, rtol=_TOL_F16)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterReduceTwoColliding:
    """aten::scatter_reduce.two_out colliding dest. Eager and compiled: #4361.

    dim=0 dest (32, 64) splits at chip-max; dim=1 dest (4, 64) stays single-config.
    """

    def setup_method(self):
        torch.manual_seed(0)

    def _xfail_colliding_dim0(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4361
        _xfail_existing(
            execution_mode,
            eager=_SR_TWO,
            compiled=_scatter_fallback(4361, "scatter_reduce with colliding indices"),
        )

    @gap_one_core
    def test_scatter_reduce_amax_colliding_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce amax dim=0 dest/src (32, 64) non-unique index (#4361)."""
        self._xfail_colliding_dim0(execution_mode)
        dst = torch.randn(32, 64, dtype=torch.float16)
        src = torch.randn(32, 64, dtype=torch.float16)
        idx = torch.randint(0, 32, (32, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="amax"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_scatter_reduce_prod_colliding_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce prod dim=0 dest/src (32, 64) non-unique index (#4361)."""
        self._xfail_colliding_dim0(execution_mode)
        dst = torch.randn(32, 64, dtype=torch.float16).abs() + 0.5
        src = torch.randn(32, 64, dtype=torch.float16).abs() + 0.5
        idx = torch.randint(0, 32, (32, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="prod"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_scatter_reduce_mean_colliding_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce mean dim=0 dest/src (32, 64) non-unique index (#4361)."""
        self._xfail_colliding_dim0(execution_mode)
        dst = torch.randn(32, 64, dtype=torch.float16)
        src = torch.randn(32, 64, dtype=torch.float16)
        idx = torch.randint(0, 32, (32, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="mean"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_scatter_reduce_amin_must_write_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce amin dim=0 dest=100 so dest is not already the min (#4361)."""
        self._xfail_colliding_dim0(execution_mode)
        dst = torch.full((32, 64), 100.0, dtype=torch.float16)
        src = torch.randn(32, 64, dtype=torch.float16)
        idx = torch.randint(0, 32, (32, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="amin"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_scatter_reduce_sum_dim1_fp16(self, execution_mode):
        """scatter_reduce sum dim=1 fp16 dest (4, 64) src (4, 32).

        Eager is aten::scatter_reduce.two_out (#4361). Compiled is KeyError tmp0 (#4637),
        not #4346 (indirect_sizes {}) and not #4354 (ranges_from_index_vars).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4361
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4637
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=_SR_TWO,
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        dst = (
            (torch.arange(4 * 64, dtype=torch.float32) / 7.0)
            .reshape(4, 64)
            .to(torch.float16)
        )
        src = (
            (torch.arange(4 * 32, dtype=torch.float32) / 7.0)
            .reshape(4, 32)
            .to(torch.float16)
        )
        idx = torch.zeros(4, 32, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(1, i, s, reduce="sum"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_scatter_reduce_sum_dim1_fp32(self, execution_mode):
        """scatter_reduce sum dim=1 fp32 — ReStickifyOpHBM leftover (#1796)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4361
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1796
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=_SR_TWO,
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        dst = cached_randn((4, 64), differentiation="sr_f32", dtype=torch.float32)
        src = cached_randn((4, 32), differentiation="sr_f32s", dtype=torch.float32)
        idx = torch.zeros(4, 32, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(1, i, s, reduce="sum"),
            dst,
            idx,
            src,
            atol=_TOL_F32,
            rtol=_TOL_F32,
        )


class TestScatterReduceCompileDynamic:
    """compile(dynamic=True) scatter_reduce sum dim=0 is refused (#5578)."""

    @expect_dynamic_true_refused
    def test_scatter_reduce_sum_dim0_dynamic(self):
        """compile(dynamic=True) scatter_reduce sum dim=0 is refused."""
        dst = (
            (torch.arange(4 * 64, dtype=torch.float32) / 7.0)
            .reshape(4, 64)
            .to(torch.float16)
        )
        src = (
            (torch.arange(2 * 64, dtype=torch.float32) / 3.0)
            .reshape(2, 64)
            .to(torch.float16)
        )
        idx = torch.zeros(2, 64, dtype=torch.int64)
        ref = dst.scatter_reduce(0, idx, src, reduce="sum")
        got = torch.compile(
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="sum"),
            dynamic=True,
        )(dst.to(DEVICE), idx.to(DEVICE), src.to(DEVICE)).cpu()
        torch.testing.assert_close(got, ref, atol=_TOL_F16, rtol=_TOL_F16)


# ---------------------------------------------------------------------------
# Additional indirect-write coverage (parametrized over mode: eager / compile)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterEdgeCasesEagerCompile:
    """Tensor-src scatter / scatter_ edge cases: zero-volume, strided, empty and single-element operands."""

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_EAGER_ONLY_UNSUPPORTED)
    def test_scatter_zero_volume_inner_dim(self, mode: str):
        """Zero-volume inner dimension scatter — shape (8, 0, 128) must be a safe no-op.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).
        """
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(8, 0, 128, dtype=torch.bfloat16)
        src_t = torch.zeros(4, 0, 128, dtype=torch.bfloat16)
        idx_t = torch.zeros(4, 0, 128, dtype=torch.int64)

        def fn(dest, index, source):
            return dest.scatter(0, index, source)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="Spyre eager scatter dispatch is unsupported for this case, #4473",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
        ],
    )
    def test_scatter_transposed_source_strided_index(self, mode: str):
        """Transposed (non-contiguous) source + strided index into strided destination.

        Synthetic KV-cache-like shape (8, 128); exercises a strided layout, not a full
        model configuration. Exercises the full non-contiguous path: dest is a step-2
        strided view, src is transposed (T), index is a narrow sub-view — all three
        inputs have non-unit strides simultaneously.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        base_dest = torch.zeros(16, 128, dtype=torch.bfloat16)
        src_base = torch.randn(128, 8, dtype=torch.bfloat16)
        src_t = src_base.t()
        idx_base = torch.empty((20, 128), dtype=torch.int64)
        idx_t = idx_base[2:18:2, :]
        for col in range(128):
            idx_t[:, col] = torch.randperm(8)

        def fn(base, index, source):
            out = base.clone()
            view = out[::2, :]
            view.scatter_(0, index, source)
            return out

        compare_with_cpu(
            fn,
            base_dest,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_EAGER_ONLY_UNSUPPORTED)
    @pytest.mark.parametrize("dim", [0, 1])
    def test_scatter_empty_row_index(self, mode: str, dim: int):
        """Zero-element index and source — safe no-op, dest returned unchanged.

        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128, hidden=5120).
        dim=0: dest (8, 128), src/index (0, 128) — zero rows, empty along the scatter dim.
        dim=1: dest (8, 128), src/index (8, 0) — zero cols, empty along the scatter dim.
        Both cases must return dest unchanged.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.randn(8, 128, dtype=torch.bfloat16)
        if dim == 0:
            src_t = torch.randn(0, 128, dtype=torch.bfloat16)
            idx_t = torch.zeros(0, 128, dtype=torch.int64)
        else:
            src_t = torch.randn(8, 0, dtype=torch.bfloat16)
            idx_t = torch.zeros(8, 0, dtype=torch.int64)

        def fn(dest, index, source):
            return dest.scatter(dim, index, source)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="Spyre eager scatter dispatch is unsupported for this case, #4473",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor indirect-scatter alignment fails with KeyError: tmp0, #4637",
                ),
            ),
        ],
    )
    def test_scatter_1d_single_element_source(self, mode: str):
        """Single-element source scatter — src shape (1,) writes one value into a 1D dest.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).
        Scatters exactly one source element to one indexed position in a 128-element
        buffer.  This is not a broadcasting test; index and source both have shape (1,).
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(128, dtype=torch.float32)
        src_t = torch.randn(1, dtype=torch.float32)
        idx_t = torch.randint(0, 128, (1,), dtype=torch.int64)

        def fn(dest, index, source):
            return dest.scatter(0, index, source)

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="Spyre eager scatter dispatch is unsupported for this case, #4473",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor cannot pad the sliced scatter output in place during restickification, #4474",
                ),
            ),
        ],
    )
    def test_scatter_1d_inplace_scalar_fill(self, mode: str):
        """1D in-place scatter_ scalar fill into a head_dim vector.

        Model shape: Gemma-4-26B (head_dim=256).
        Covers 1D-IP: the only existing 1D scatter test (test_scatter_dim_broadcasting)
        uses the out-of-place form. This exercises scatter_() on a 1D tensor by filling
        16 randomly selected positions in a head_dim=256 buffer with a constant 1.0.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(256, dtype=torch.bfloat16)
        idx_t = torch.randperm(256, dtype=torch.int64)[:16]

        def fn(dest, index):
            out = dest.clone()
            out.scatter_(0, index, 1.0)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterHighRankEagerCompile:
    """4D / 5D / 6D scatter on KV-cache geometries (out-of-place and in-place)."""

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_HIGH_RANK_XFAIL)
    def test_scatter_4d_outofplace_attention_logits(self, mode: str):
        """4D out-of-place scatter on (batch, kv_heads, seq, head_dim) attention geometry.

        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128).
        Covers 4D-OOP: no existing test calls scatter() on a 4D tensor.
        Scatters updated token logits into 4 selected sequence positions (dim=2).
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 10, 16, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 10, 4, 128, dtype=torch.bfloat16)
        seq_idx = torch.randperm(16, dtype=torch.int64)[:4]
        idx_t = seq_idx.view(1, 1, 4, 1).expand(2, 10, 4, 128)

        def fn(dest, index, source):
            return dest.scatter(2, index, source)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_HIGH_RANK_XFAIL)
    def test_scatter_4d_inplace_channel_update(self, mode: str):
        """4D in-place scatter_ — updates selected kv_head channels in a KV-cache buffer.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).
        Covers 4D-IP: scatter_() on a 4D tensor. Scatters 4 source rows into selected
        kv_heads positions (dim=1) inside a (batch, kv_heads, seq, head_dim) buffer.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 8, 16, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 4, 16, 128, dtype=torch.bfloat16)
        kv_idx = torch.randperm(8, dtype=torch.int64)[:4]
        idx_t = kv_idx.view(1, 4, 1, 1).expand(2, 4, 16, 128)

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_(1, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_HIGH_RANK_XFAIL)
    def test_scatter_5d_inplace_layer_cache(self, mode: str):
        """5D in-place scatter_ on layer-KV-cache geometry (batch, layers, kv_heads, seq, head_dim).

        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128).
        Covers 5D-IP: the only existing 5D scatter test (test_scatter_5d_middle_axis)
        uses the out-of-place form. Scatters 4 updated token positions into dim=3 (seq)
        of a full layer-cache buffer in-place.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 4, 8, 16, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        seq_idx = torch.randperm(16, dtype=torch.int64)[:4]
        idx_t = seq_idx.view(1, 1, 1, 4, 1).expand(2, 4, 8, 4, 128)

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_(3, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_HIGH_RANK_XFAIL)
    def test_scatter_6d_inplace_multi_layer_cache(self, mode: str):
        """6D in-place scatter_ on multi-layer multi-expert cache geometry.

        Layout: (experts, batch, layers, kv_heads, seq, head_dim).
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equivalent).
        Covers 6D-IP: no existing test uses scatter_() on a 6D tensor. Scatters 2
        updated token positions into dim=4 (seq) of a 6D expert-layer-KV-cache buffer.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 2, 2, 8, 8, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 2, 2, 8, 2, 128, dtype=torch.bfloat16)
        seq_idx = torch.randperm(8, dtype=torch.int64)[:2]
        idx_t = seq_idx.view(1, 1, 1, 1, 2, 1).expand(2, 2, 2, 8, 2, 128)

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_(4, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterAddShapesEagerCompile:
    """scatter_add: storage offsets, bf16 subnormals, and 4D / 5D / 6D KV-cache geometries."""

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="aten::scatter_add.out has no registered Spyre eager kernel, #4396",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor layout propagation cannot resolve sharding because graph and real input counts differ, #6707",
                ),
            ),
        ],
    )
    def test_scatter_add_nonzero_storage_offset(self, mode: str):
        """Non-zero storage_offset on self, index, and src simultaneously.

        Synthetic KV-cache-like shape (16, 256); not a Gemma-4-26B configuration.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        base_self = torch.zeros(32, 256, dtype=torch.bfloat16)
        base_src = torch.randn(32, 256, dtype=torch.bfloat16)
        base_idx = torch.randint(0, 16, (32, 256), dtype=torch.int64)

        def fn(dest_base, index_base, source_base):
            out = dest_base.clone()
            index = index_base.narrow(0, 2, 16)
            source = source_base.narrow(0, 4, 16)
            dest = out.narrow(0, 8, 16)
            assert dest.storage_offset() > 0
            assert index.storage_offset() > 0
            assert source.storage_offset() > 0
            dest.scatter_add_(0, index, source)
            return out

        compare_with_cpu(
            fn,
            base_self,
            base_idx,
            base_src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="aten::scatter_add.out has no registered Spyre eager kernel, #4396",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="dbo-opt GatherIndexConversion fails its divisibility assertion, #6574",
                ),
            ),
        ],
    )
    def test_scatter_add_bf16_subnormal_accumulation(self, mode: str):
        """BF16 subnormal/denormal accumulation using smallest_subnormal.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).
        Uses torch.finfo(bfloat16).smallest_subnormal to guarantee a true
        denormal value rather than a magic constant that may flush to zero on
        some platforms. Verifies the Spyre backend does not trap on denormals
        or silently flush them before accumulation.
        """
        run_eager, run_compile = _mode_flags(mode)
        finfo = torch.finfo(torch.bfloat16)
        val = getattr(finfo, "smallest_subnormal", 2 ** (-133))
        self_t = torch.zeros(8, 128, dtype=torch.bfloat16)
        src_t = torch.full((4, 128), val, dtype=torch.bfloat16)
        idx_t = torch.zeros(4, 128, dtype=torch.int64)

        def fn(dest, index, source):
            return torch.scatter_add(dest, 0, index, source)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.0001,
            rtol=0.0001,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_ADD_UNSUPPORTED)
    def test_scatter_add_4d_both(self, mode: str):
        """4D scatter_add on (batch, kv_heads, seq, head_dim) — both OOP and IP.

        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128). Covers 4D-OOP and 4D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 10, 8, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 10, 4, 128, dtype=torch.bfloat16)
        idx_t = torch.randint(0, 8, (2, 10, 4, 128), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return torch.scatter_add(dest, 2, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_add_(2, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_ADD_UNSUPPORTED)
    def test_scatter_add_5d_both(self, mode: str):
        """5D scatter_add on (batch, layers, kv_heads, seq, head_dim) — both OOP and IP.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128). Covers 5D-OOP and 5D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 4, 8, 8, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        idx_t = torch.randint(0, 8, (2, 4, 8, 4, 128), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return torch.scatter_add(dest, 3, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_add_(3, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_ADD_UNSUPPORTED)
    def test_scatter_add_6d_both(self, mode: str):
        """6D scatter_add on (experts, batch, layers, kv_heads, seq, head_dim) — both OOP and IP.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equiv). Covers 6D-OOP and 6D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 2, 2, 8, 2, 128, dtype=torch.bfloat16)
        idx_t = torch.randint(0, 4, (2, 2, 2, 8, 2, 128), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return torch.scatter_add(dest, 4, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_add_(4, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterReduceModesEagerCompile:
    """scatter_reduce across all reduction modes and with dynamic shapes."""

    @pytest.mark.parametrize(
        ("mode", "reduce_mode"),
        [
            pytest.param(
                "eager",
                "sum",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="indexed-reduction eager dispatch is reported unsupported, #4361",
                ),
                id="sum-eager",
            ),
            pytest.param(
                "compile",
                "sum",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="dbo-opt scheduler rejects the scatter page-size constraint, #6575",
                ),
                id="sum-compile",
            ),
            pytest.param(
                "eager",
                "prod",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="indexed-reduction eager dispatch is reported unsupported, #4361",
                ),
                id="prod-eager",
            ),
            pytest.param(
                "compile",
                "prod",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="indexed-reduction compile path is reported unsupported, #4361",
                ),
                id="prod-compile",
            ),
            pytest.param(
                "eager",
                "mean",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="indexed-reduction eager dispatch is reported unsupported, #4361",
                ),
                id="mean-eager",
            ),
            pytest.param(
                "compile",
                "mean",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="indexed-reduction compile path is reported unsupported, #4361",
                ),
                id="mean-compile",
            ),
            pytest.param(
                "eager",
                "amax",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="indexed-reduction eager dispatch is reported unsupported, #4361",
                ),
                id="amax-eager",
            ),
            pytest.param(
                "compile",
                "amax",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="indexed-reduction compile path is reported unsupported, #4361",
                ),
                id="amax-compile",
            ),
            pytest.param(
                "eager",
                "amin",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="indexed-reduction eager dispatch is reported unsupported, #4361",
                ),
                id="amin-eager",
            ),
            pytest.param(
                "compile",
                "amin",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="indexed-reduction compile path is reported unsupported, #4361",
                ),
                id="amin-compile",
            ),
        ],
    )
    def test_scatter_reduce_nan_inf_all_modes(self, mode: str, reduce_mode: str):
        """NaN and Inf propagation across all 5 reduction modes (not just sum).

        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128).
        """
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(8, 128, dtype=torch.float32)
        src_t = torch.ones(4, 128, dtype=torch.float32)
        src_t[0, 0] = float("nan")
        src_t[1, 1] = float("inf")
        idx_t = torch.tensor(
            [[0] * 128, [1] * 128, [2] * 128, [3] * 128], dtype=torch.int64
        )

        def fn(dest, index, source):
            return dest.scatter_reduce(
                0, index, source, reduce=reduce_mode, include_self=True
            )

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="indexed-reduction eager dispatch is reported unsupported, #4361",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="dbo-opt scheduler rejects the scatter page-size constraint, #6575",
                ),
            ),
        ],
    )
    def test_scatter_reduce_dynamic_symint_shapes(self, mode: str):
        """Dynamic SymInt shape tracing through scatter_reduce in TorchInductor.

        Model shape: Granite-4.1-20B (hidden=8192, kv_heads=8, head_dim=128).
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        B, S, D = (2, 8, 8192)
        max_S = 32
        self_t = torch.zeros(B, max_S, D, dtype=torch.float32)
        src_t = torch.randn(B, S, D, dtype=torch.float32)
        idx_t = torch.randint(0, max_S, (B, S, D), dtype=torch.int64)

        def fn(dest, index, source):
            return dest.scatter_reduce(
                1, index, source, reduce="sum", include_self=True
            )

        if run_compile and hasattr(torch, "_dynamo"):
            torch._dynamo.mark_dynamic(src_t, 1)
        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexAddEagerCompile:
    """index_add / index_add_: storage offsets, high-rank KV-cache geometries and channels-last layout."""

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="indexed-reduction eager dispatch is reported unsupported, #4634",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor layout propagation cannot resolve sharding because graph and real input counts differ, #6707",
                ),
            ),
        ],
    )
    def test_index_add_nonzero_storage_offset(self, mode: str):
        """Non-zero storage_offset on sliced window inputs — pointer arithmetic correctness.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        pool_self = torch.zeros(64, 128, dtype=torch.bfloat16)
        pool_src = torch.randn(64, 128, dtype=torch.bfloat16)
        idx_t = torch.randperm(8)[:4]

        def fn(dest_base, index, source_base):
            dest_storage = dest_base.clone()
            source_storage = source_base.clone()
            dest = dest_storage.narrow(0, 16, 8)
            source = source_storage.narrow(0, 8, 4)
            assert dest.storage_offset() > 0
            assert source.storage_offset() > 0
            dest.index_add_(0, index, source)
            return dest_storage

        compare_with_cpu(
            fn,
            pool_self,
            idx_t,
            pool_src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.05,
            rtol=0.05,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_ADD_EAGER_ONLY_XFAIL)
    def test_index_add_5d_both(self, mode: str):
        """5D index_add on (batch, layers, kv_heads, seq, head_dim) — both OOP and IP.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128). Covers 5D-OOP and 5D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 4, 8, 8, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        idx_t = torch.randperm(8, dtype=torch.int64)[:4]

        def fn_oop(dest, index, source):
            return torch.index_add(dest, 3, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.index_add_(3, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_SCATTER_ADD_EAGER_ONLY_XFAIL)
    def test_index_add_6d_both(self, mode: str):
        """6D index_add on expert-layer-KV-cache geometry — both OOP and IP.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equiv). Covers 6D-OOP and 6D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 2, 2, 8, 2, 128, dtype=torch.bfloat16)
        idx_t = torch.randperm(4, dtype=torch.int64)[:2]

        def fn_oop(dest, index, source):
            return torch.index_add(dest, 4, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.index_add_(4, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_EAGER_COMPILE)
    def test_index_add_channels_last_4d(self, mode: str):
        """channels-last 4D tensor index_add along channel dimension (dim=1).

        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128) — channels=kv_heads=10, spatial=head_dim=128.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 10, 8, 16, dtype=torch.float32).to(
            memory_format=torch.channels_last
        )
        src_shape = list(self_t.shape)
        src_shape[1] = 4
        src_t = torch.randn(*src_shape, dtype=torch.float32).to(
            memory_format=torch.channels_last
        )
        idx_t = torch.randperm(10)[:4]

        def fn(dest, index, source):
            return torch.index_add(dest, 1, index, source)

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexFillShapesEagerCompile:
    """index_fill / index_fill_: special IEEE-754 values, 1D / 4D / 6D geometries and strided views."""

    @pytest.mark.parametrize("mode", _MODE_PARAMS_INDEX_FILL_UNSUPPORTED)
    @pytest.mark.parametrize("fill_val", [float("nan"), float("inf"), float("-inf")])
    def test_index_fill_special_ieee754_values(self, mode: str, fill_val: float):
        """Special IEEE-754 fills — NaN, +Inf, -Inf bit-pattern preservation.

        Synthetic KV-cache-like shape (16, 256); not a Gemma-4-26B configuration.
        """
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.ones(16, 256, dtype=torch.float32)
        idx_t = torch.tensor([0, 8], dtype=torch.int64)

        def fn(dest, index):
            return torch.index_fill(dest, 0, index, fill_val)

        compare_with_cpu(
            fn, self_t, idx_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_INDEX_FILL_UNSUPPORTED)
    def test_index_fill_1d_both(self, mode: str):
        """1D index_fill — both OOP and IP on a 1D head_dim vector.

        Model shape: Granite-4.1-20B (head_dim=128). Covers 1D-OOP and 1D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(128, dtype=torch.bfloat16)
        idx_t = torch.randperm(128, dtype=torch.int64)[:8]

        def fn_oop(dest, index):
            return torch.index_fill(dest, 0, index, 99.0)

        def fn_ip(dest, index):
            out = dest.clone()
            out.index_fill_(0, index, 99.0)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_INDEX_FILL_UNSUPPORTED)
    def test_index_fill_4d_both(self, mode: str):
        """4D index_fill on (batch, kv_heads, seq, head_dim) — both OOP and IP.

        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128). Covers 4D-OOP and 4D-IP.
        """
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 10, 8, 128, dtype=torch.bfloat16)
        idx_t = torch.tensor([1, 3, 5, 7], dtype=torch.int64)

        def fn_oop(dest, index):
            return torch.index_fill(dest, 2, index, 42.0)

        def fn_ip(dest, index):
            out = dest.clone()
            out.index_fill_(2, index, 42.0)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_INDEX_FILL_UNSUPPORTED)
    def test_index_fill_6d_both(self, mode: str):
        """6D index_fill on expert-layer-KV-cache geometry — both OOP and IP.

        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256). Covers 6D-OOP and 6D-IP.
        """
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 2, 2, 16, 4, 256, dtype=torch.float32)
        idx_t = torch.tensor([1, 3], dtype=torch.int64)

        def fn_oop(dest, index):
            return torch.index_fill(dest, 4, index, 7.0)

        def fn_ip(dest, index):
            out = dest.clone()
            out.index_fill_(4, index, 7.0)
            return out

        compare_with_cpu(
            fn_oop, self_t, idx_t, run_eager=run_eager, run_compile=run_compile
        )
        compare_with_cpu(
            fn_ip, self_t, idx_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="aten::index_fill_.int_Scalar has no registered Spyre eager kernel, #6587",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
        ],
    )
    def test_index_fill_nonunit_strided_slice(self, mode: str):
        """Non-unit strided slice fill — interleaved rows must remain unmodified.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128) — step-3 stride across kv-heads.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        base = torch.randn(24, 128, dtype=torch.bfloat16)
        idx_t = torch.tensor([0, 1], dtype=torch.int64)

        def fn(base_tensor, index):
            out = base_tensor.clone()
            view = out[::3, :]
            view.index_fill_(0, index, 99.0)
            return out

        compare_with_cpu(
            fn,
            base,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexSelectDynamicEagerCompile:
    """index_select with a varying index length."""

    @pytest.mark.parametrize("mode", _MODE_PARAMS_EAGER_COMPILE)
    def test_index_select_dynamic_index_length(self, mode: str):
        """Dynamic index length — compiled graph must handle varying K without recompilation.

        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128).
        Calls the compiled function three times with index tensors of different
        lengths (K=2, 5, 10) to verify TorchInductor emits a kernel that
        accepts a symbolic index size rather than specialising on a fixed K.
        In eager mode this simply validates correctness across all three sizes.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        x = torch.randn(10, 128, dtype=torch.bfloat16)

        def fn(source, index):
            return torch.index_select(source, 0, index)

        compiled_fn = (
            torch.compile(fn) if run_compile and hasattr(torch, "compile") else None
        )
        for k in (2, 5, 10):
            idx_t = torch.randperm(10)[:k]
            if run_eager:
                compare_with_cpu(
                    fn,
                    x,
                    idx_t,
                    run_eager=True,
                    run_compile=False,
                    atol=0.01,
                    rtol=0.016,
                )
            if compiled_fn is not None:
                x_dev = x.to(DEVICE)
                idx_dev = idx_t.to(DEVICE)
                expected = fn(x, idx_t)
                result = compiled_fn(x_dev, idx_dev)
                torch.testing.assert_close(
                    result.cpu(), expected, atol=0.01, rtol=0.016
                )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestMaskedScatterEagerCompile:
    """masked_scatter / masked_scatter_: dynamic shapes, high-rank, contiguity and zero-volume cases."""

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=RuntimeError,
                    reason="Spyre masked_scatter rejects element-level mask/source indexing, #4437",
                ),
            ),
            "compile",
        ],
    )
    def test_masked_scatter_dynamic_shapes_inductor(self, mode: str):
        """Dynamic batch dimension and dynamic source length under TorchInductor.

        Model shape: Ministral-3-14B (hidden=5120, kv_heads=8) — dynamic (B, S) slices.
        masked_scatter requires a prefix-sum scan over the mask to compute per-
        element source offsets. This test verifies TorchInductor can compile
        that scan kernel with a symbolic batch size B and symbolic True-count K
        rather than specialising on fixed constants.
        The compiled function is called three times with distinct (B, S) pairs —
        (8, 40), (4, 20), (6, 32) — varying both dimensions to exercise
        guard-free re-use of a single compiled kernel. In eager mode the test
        simply validates correctness across all three sizes.
        """
        run_eager, run_compile = _mode_flags(mode)

        def fn(dest, m, source):
            return torch.masked_scatter(dest, m, source)

        compiled_fn = (
            torch.compile(fn) if run_compile and hasattr(torch, "compile") else None
        )
        torch.manual_seed(7)
        for B, S in ((8, 40), (4, 20), (6, 32)):
            self_t = torch.zeros(B, S, dtype=torch.bfloat16)
            mask = torch.rand(B, S) > 0.5
            num_true = int(mask.sum().item())
            src_t = torch.randn(num_true, dtype=torch.bfloat16)
            expected = fn(self_t, mask, src_t)
            if run_eager:
                compare_with_cpu(
                    fn,
                    self_t,
                    mask,
                    src_t,
                    run_eager=True,
                    run_compile=False,
                    atol=0.01,
                    rtol=0.016,
                )
            if compiled_fn is not None:
                result = compiled_fn(self_t, mask, src_t)
                torch.testing.assert_close(
                    result.cpu(), expected, atol=1e-05, rtol=1e-05
                )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_UNSUPPORTED)
    def test_masked_scatter_0d_inplace(self, mode: str):
        """0D in-place masked_scatter_ — scalar self with scalar True mask.

        Covers 0D-IP (existing 0D test uses OOP form only).
        """
        run_eager, run_compile = _mode_flags(mode)
        x = torch.tensor(0.0, dtype=torch.float32)
        mask = torch.tensor(True)
        src = torch.tensor([42.0], dtype=torch.float32)

        def fn(d, m, s):
            out = d.clone()
            out.masked_scatter_(m, s)
            return out

        compare_with_cpu(fn, x, mask, src, run_eager=run_eager, run_compile=run_compile)

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_UNSUPPORTED)
    def test_masked_scatter_4d_inplace(self, mode: str):
        """4D in-place masked_scatter_ — (batch, kv_heads, seq, head_dim).

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128). Covers 4D-IP.
        """
        run_eager, run_compile = _mode_flags(mode)
        torch.manual_seed(5)
        self_t = torch.zeros(2, 8, 8, 128, dtype=torch.bfloat16)
        mask = torch.rand(2, 8, 8, 128) > 0.75
        src_t = torch.randn(int(mask.sum().item()), dtype=torch.bfloat16)

        def fn(dest, m, source):
            out = dest.clone()
            out.masked_scatter_(m, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_UNSUPPORTED)
    def test_masked_scatter_5d_inplace(self, mode: str):
        """5D in-place masked_scatter_ — (batch, layers, kv_heads, seq, head_dim).

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equiv). Covers 5D-IP.
        """
        run_eager, run_compile = _mode_flags(mode)
        torch.manual_seed(6)
        self_t = torch.zeros(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        mask = torch.rand(2, 4, 8, 4, 128) > 0.75
        src_t = torch.randn(int(mask.sum().item()), dtype=torch.bfloat16)

        def fn(dest, m, source):
            out = dest.clone()
            out.masked_scatter_(m, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_ELEMENTWISE)
    def test_masked_scatter_6d_both(self, mode: str):
        """6D masked_scatter — both OOP and IP on expert-layer-KV-cache geometry.

        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128). Covers 6D-OOP and 6D-IP.
        """
        run_eager, run_compile = _mode_flags(mode)
        torch.manual_seed(7)
        self_t = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        mask = torch.rand(2, 2, 2, 8, 4, 128) > 0.75
        src_t = torch.randn(int(mask.sum().item()), dtype=torch.bfloat16)

        def fn_oop(dest, m, source):
            return torch.masked_scatter(dest, m, source)

        def fn_ip(dest, m, source):
            out = dest.clone()
            out.masked_scatter_(m, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_ELEMENTWISE)
    def test_masked_scatter_all_true_full_replacement(self, mode: str):
        """Dense all-True mask replaces every element from the flat source."""
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(10, 128, dtype=torch.bfloat16)
        mask = torch.ones(10, 128, dtype=torch.bool)
        src_t = torch.arange(1280, dtype=torch.bfloat16)

        def fn(dest, m, source):
            return torch.masked_scatter(dest, m, source)

        compare_with_cpu(
            fn,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_ELEMENTWISE)
    def test_masked_scatter_no_side_effects(self, mode: str):
        """Out-of-place masked_scatter must not mutate self, mask, or source."""
        run_eager, run_compile = _mode_flags(mode)
        x = torch.zeros(5, dtype=torch.float32)
        mask = torch.tensor([True, False, True, False, True])
        src = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        x_before = x.clone()
        mask_before = mask.clone()
        src_before = src.clone()

        def fn(dest, m, source):
            return dest.masked_scatter(m, source)

        compare_with_cpu(fn, x, mask, src, run_eager=run_eager, run_compile=run_compile)
        torch.testing.assert_close(x, x_before)
        torch.testing.assert_close(src, src_before)
        assert torch.equal(mask, mask_before)

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_ELEMENTWISE)
    def test_masked_scatter_scalar_0d_self(self, mode: str):
        """MSK-NEW-04b: 0-dimensional scalar self with scalar True mask → value replaced, shape preserved."""
        run_eager, run_compile = _mode_flags(mode)
        x = torch.tensor(0, dtype=torch.float32)
        mask = torch.tensor(True)
        src = torch.tensor([42.0], dtype=torch.float32)

        def fn(d, m, s):
            return d.masked_scatter(m, s)

        compare_with_cpu(fn, x, mask, src, run_eager=run_eager, run_compile=run_compile)

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_ELEMENTWISE)
    def test_masked_scatter_scalar_false_mask_noop(self, mode: str):
        """0-dimensional scalar False mask — dest returned unchanged (no True positions)."""
        run_eager, run_compile = _mode_flags(mode)
        x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        mask = torch.tensor(0, dtype=torch.bool)
        src = torch.tensor([99.0, 88.0], dtype=torch.float32)

        def fn(d, m, s):
            return d.masked_scatter(m, s)

        compare_with_cpu(fn, x, mask, src, run_eager=run_eager, run_compile=run_compile)

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_UNSUPPORTED)
    @pytest.mark.parametrize(
        "dest_shape,mask_shape",
        [
            ((0,), (0,)),
            ((4, 0), (4, 0)),
            ((0, 5), (0, 5)),
            ((5, 0, 5), (5, 0, 5)),
            ((3, 0, 4), (3, 0, 4)),
        ],
    )
    def test_masked_scatter_zero_volume_shapes(
        self, mode: str, dest_shape: tuple, mask_shape: tuple
    ):
        """MSK-GAP-C: Zero-volume / empty-dimension destinations — must be a safe no-op.

        When the destination (or mask) contains a zero-size dimension the total
        number of elements is zero.  An all-False (or vacuously all-True) mask
        over zero elements has zero True positions, so an empty source is valid
        and the operation must return a tensor identical to the original
        destination.
        Mirrors the upstream empty-tensor cases in test_masked_scatter:
            dest = torch.empty((5, 0, 5), ...)
            mask = torch.ones_like(dest, dtype=torch.bool)
            src  = torch.empty((0,), ...)
            dest.masked_scatter_(mask, src)   # no-op, dest unchanged
        Both in-place and out-of-place variants are checked.
        compare_with_cpu is used: CPU result (unchanged dest) is the reference.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dest = torch.randn(*dest_shape, dtype=torch.float32)
        mask = torch.zeros(*mask_shape, dtype=torch.bool)
        src = torch.empty(0, dtype=torch.float32)

        def fn_inplace(d, m, s):
            out = d.clone()
            out.masked_scatter_(m, s)
            return out

        def fn_outplace(d, m, s):
            return torch.masked_scatter(d, m, s)

        compare_with_cpu(
            fn_inplace, dest, mask, src, run_eager=run_eager, run_compile=run_compile
        )
        compare_with_cpu(
            fn_outplace, dest, mask, src, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_UNSUPPORTED)
    @pytest.mark.parametrize("dest_shape", [(5, 0, 5), (3, 0, 4), (0, 8)])
    def test_masked_scatter_zero_volume_ones_mask(self, mode: str, dest_shape: tuple):
        """MSK-GAP-C (ones mask): Zero-volume dest with all-True mask and empty source.

        Mirrors the second upstream empty-tensor variant:
            mask = torch.ones_like(dest, dtype=torch.bool)
            src  = torch.empty((0,), ...)
            dest.masked_scatter_(mask, src)
        An all-True mask over a zero-volume tensor still has zero True elements
        (because numel() == 0), so the empty source is valid and the call must
        be a safe no-op.
        compare_with_cpu validates that both eager and compile modes match CPU.
        """
        run_eager, run_compile = _mode_flags(mode)
        dest = torch.empty(*dest_shape, dtype=torch.float32)
        mask = torch.ones(*dest_shape, dtype=torch.bool)
        src = torch.empty(0, dtype=torch.float32)

        def fn_inplace(d, m, s):
            out = d.clone()
            out.masked_scatter_(m, s)
            return out

        def fn_outplace(d, m, s):
            return torch.masked_scatter(d, m, s)

        compare_with_cpu(
            fn_inplace, dest, mask, src, run_eager=run_eager, run_compile=run_compile
        )
        compare_with_cpu(
            fn_outplace, dest, mask, src, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_UNSUPPORTED)
    def test_masked_scatter_noncontig_dest_mask_combos(self, mode: str):
        """All four contiguous/non-contiguous destination × mask combinations.

        Each tensor has logical shape (5, 2). Non-contiguous operands are
        transposed views with stride (1, 5), testing layout without changing
        the logical shape.
        """
        run_eager, run_compile = _mode_flags(mode)
        dest_contig = torch.zeros(5, 2, dtype=torch.float32)
        dest_noncontig = torch.zeros(2, 5, dtype=torch.float32).transpose(0, 1)
        mask_contig = torch.tensor(
            [[False, True], [False, True], [False, False], [True, True], [True, True]],
            dtype=torch.bool,
        )
        mask_noncontig = mask_contig.transpose(0, 1).contiguous().transpose(0, 1)
        source = torch.arange(1, 7, dtype=torch.float32)
        assert dest_contig.is_contiguous()
        assert not dest_noncontig.is_contiguous()
        assert mask_contig.is_contiguous()
        assert not mask_noncontig.is_contiguous()

        def fn(dest, mask, values):
            out = dest.clone()
            out.masked_scatter_(mask, values)
            return out

        for dest, mask in (
            (dest_contig, mask_contig),
            (dest_contig, mask_noncontig),
            (dest_noncontig, mask_contig),
            (dest_noncontig, mask_noncontig),
        ):
            compare_with_cpu(
                fn, dest, mask, source, run_eager=run_eager, run_compile=run_compile
            )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_MASKED_SCATTER_ELEMENTWISE)
    def test_masked_scatter_5d_transposed_target(self, mode: str):
        """True non-contiguous 5D target — row-major traversal of mask True bits.

        Model shape: Granite-3.3-8B (batch=2, kv_heads=8, layers=4, seq=4, head_dim=128).
        The destination is base.transpose(1, 3) with NO .contiguous() call, so
        it is a genuinely non-contiguous tensor (strides are non-unit and not
        in descending order).  masked_scatter must traverse both dest and mask
        in C-contiguous logical order regardless of the physical memory layout.
        CPU is the reference; the result is compared element-wise via
        compare_with_cpu.
        """
        run_eager, run_compile = _mode_flags(mode)
        torch.manual_seed(1)
        base = torch.zeros(2, 8, 4, 4, 128, dtype=torch.bfloat16)
        dest_t = base.transpose(1, 3)
        assert not dest_t.is_contiguous(), "dest must be non-contiguous for this test"
        mask = torch.rand(*dest_t.shape) > 0.75
        num_true = int(mask.sum().item())
        src_t = torch.randn(num_true, dtype=torch.bfloat16)

        def fn(dest, m, source):
            return torch.masked_scatter(dest, m, source)

        compare_with_cpu(
            fn,
            dest_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.05,
            rtol=0.05,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexReduceShapesEagerCompile:
    """index_reduce / index_reduce_: views, dtypes, index dtypes, high-rank geometries and non-contiguous operands."""

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Indirect indexing on a sliced view with a non-zero storage offset fails OpSpec validation, #6210",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Indirect indexing on a sliced view with a non-zero storage offset fails OpSpec validation, #6210",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("reduce_mode", _REDUCE_MODES)
    def test_index_reduce_sliced_view_dest(self, mode: str, reduce_mode: str):
        """IDR-GAP-F: index_reduce_ on a non-contiguous view slice of a larger tensor.

        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128) — (batch=2, kv_heads+pad=11, head_dim=128).
        Mirrors the upstream TorchInductor regression test (issue #144846):
            x_base[:, 3:, :]  — non-contiguous slice leaving 3 padding cols behind
        The base tensor is allocated with extra padding; index_reduce_ is
        applied directly to the slice (non-zero storage_offset and non-default strides).
        The result in the slice must match the CPU reference; unsliced cols remain unchanged.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        x_base = torch.rand(2, 11, 128, dtype=torch.bfloat16) + 0.5
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)
        source = torch.rand(2, 4, 128, dtype=torch.bfloat16) + 0.5

        def fn(base, i, s):
            out = base.clone()
            view = out[:, 3:, :]
            view.index_reduce_(1, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn,
            x_base,
            idx,
            source,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize(("reduce_mode", "mode"), _INDEX_REDUCE_NARROW_CASES)
    def test_index_reduce_narrow_view_dest(self, mode: str, reduce_mode: str):
        """IDR-GAP-F (narrow): index_reduce_ on a narrow() view — non-zero storage_offset.

        Synthetic KV-cache-like shape (16, 256) — narrow 16 of 32 rows.
        Uses narrow() instead of slicing to produce a view with a different
        storage_offset.  This exercises the pointer-arithmetic path through
        the backend's DMA address computation for non-zero offsets.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        pool = torch.rand(32, 256, dtype=torch.bfloat16) + 0.5
        src = torch.rand(4, 256, dtype=torch.bfloat16) + 0.5
        idx = torch.randint(0, 16, (4,), dtype=torch.int64)

        def fn(base, i, s):
            out = base.clone()
            dest = out.narrow(0, 8, 16)
            assert dest.storage_offset() > 0
            dest.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn,
            pool,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce with duplicate indices keeps one update instead of reducing them, #4411",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce with duplicate indices keeps one update instead of reducing them, #4411",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "dtype, reduce_mode, index_dtype, atol, rtol", _INDEX_REDUCE_OUTPLACE_CASES
    )
    def test_index_reduce_outplace_dtype_and_index_variants(
        self,
        mode: str,
        dtype: torch.dtype,
        reduce_mode: str,
        index_dtype: torch.dtype,
        atol: float,
        rtol: float,
    ):
        """Retain functional index_reduce cases; omit upstream-covered in-place cases."""
        run_eager, run_compile = _mode_flags(mode)
        dest = torch.rand(8, 128, dtype=dtype) + 0.5
        source = torch.rand(4, 128, dtype=dtype) + 0.5

        def fn_out(d, i, s):
            return d.index_reduce(0, i, s, reduce=reduce_mode, include_self=True)

        for index in (
            torch.tensor([0, 2, 4, 6], dtype=index_dtype),
            torch.tensor([0, 0, 2, 4], dtype=index_dtype),
        ):
            compare_with_cpu(
                fn_out,
                dest,
                index,
                source,
                run_eager=run_eager,
                run_compile=run_compile,
                atol=atol,
                rtol=rtol,
            )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_EAGER_COMPILE)
    @pytest.mark.parametrize("reduce_mode", _REDUCE_MODES)
    def test_index_reduce_4d_both(self, mode: str, reduce_mode: str):
        """4D index_reduce on (batch, kv_heads, seq, head_dim) — both OOP and IP.

        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256). Covers 4D-OOP and 4D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dest = torch.rand(2, 16, 8, 256, dtype=torch.float32) + 0.5
        src = torch.rand(2, 16, 4, 256, dtype=torch.float32) + 0.5
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)

        def fn_oop(d, i, s):
            return d.index_reduce(2, i, s, reduce=reduce_mode, include_self=True)

        def fn_ip(d, i, s):
            out = d.clone()
            out.index_reduce_(2, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn_oop,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_EAGER_COMPILE)
    @pytest.mark.parametrize("reduce_mode", _REDUCE_MODES)
    def test_index_reduce_5d_both(self, mode: str, reduce_mode: str):
        """5D index_reduce on (batch, layers, kv_heads, seq, head_dim) — both OOP and IP.

        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128). Covers 5D-OOP and 5D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dest = torch.rand(2, 4, 10, 8, 128, dtype=torch.float32) + 0.5
        src = torch.rand(2, 4, 10, 4, 128, dtype=torch.float32) + 0.5
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)

        def fn_oop(d, i, s):
            return d.index_reduce(3, i, s, reduce=reduce_mode, include_self=True)

        def fn_ip(d, i, s):
            out = d.clone()
            out.index_reduce_(3, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn_oop,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_EAGER_COMPILE)
    @pytest.mark.parametrize("reduce_mode", _REDUCE_MODES)
    def test_index_reduce_6d_both(self, mode: str, reduce_mode: str):
        """6D index_reduce on expert-layer-KV-cache geometry — both OOP and IP.

        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128). Covers 6D-OOP and 6D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dest = torch.rand(2, 2, 2, 8, 4, 128, dtype=torch.float32) + 0.5
        src = torch.rand(2, 2, 2, 8, 2, 128, dtype=torch.float32) + 0.5
        idx = torch.randint(0, 4, (2,), dtype=torch.int64)

        def fn_oop(d, i, s):
            return d.index_reduce(4, i, s, reduce=reduce_mode, include_self=True)

        def fn_ip(d, i, s):
            out = d.clone()
            out.index_reduce_(4, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn_oop,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize(
        ("reduce_mode", "mode"),
        [
            pytest.param(
                "prod",
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ with duplicate indices keeps one update instead of reducing them, #4411",
                ),
            ),
            pytest.param(
                "prod",
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
            pytest.param(
                "mean",
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ on a non-contiguous destination returns values that differ from CPU, #4932",
                ),
            ),
            pytest.param(
                "mean",
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
            pytest.param(
                "amax",
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ on a non-contiguous destination returns values that differ from CPU, #4932",
                ),
            ),
            pytest.param(
                "amax",
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
            pytest.param(
                "amin",
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ on a non-contiguous destination returns values that differ from CPU, #4932",
                ),
            ),
            pytest.param(
                "amin",
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
        ],
    )
    def test_index_reduce_noncontig_dest(self, mode: str, reduce_mode: str):
        """IDR-GAP-B (i): Non-contiguous destination — step-2 row stride.

        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128) — step-2 row view.
        dest is a [::2, :] view of a larger backing buffer (stride[0] = 2×cols),
        so it is non-contiguous.  src and idx are contiguous.
        index_reduce_ must correctly address the strided destination layout.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        base = torch.rand(16, 128, dtype=torch.bfloat16) + 0.5
        dest = base[::2, :]
        assert not dest.is_contiguous()
        src = torch.rand(4, 128, dtype=torch.bfloat16) + 0.5
        idx = torch.randint(0, dest.shape[0], (4,), dtype=torch.int64)

        def fn(backing, i, s):
            out = backing.clone()
            dest_view = out[::2, :]
            assert not dest_view.is_contiguous()
            dest_view.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn,
            base,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ with a non-contiguous source returns values that differ from CPU, #4932",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ with a non-contiguous source returns values that differ from CPU, #4932",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("reduce_mode", _REDUCE_MODES)
    def test_index_reduce_noncontig_source(self, mode: str, reduce_mode: str):
        """IDR-GAP-B (ii): Non-contiguous source — transposed 2D tensor.

        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128) — transposed (128, 8) → (8, 128).
        src is a transposed view (non-contiguous strides).  dest and idx are
        contiguous.  index_reduce_ must read src through its logical strides.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dest = torch.rand(8, 128, dtype=torch.bfloat16) + 0.5
        src_base = torch.rand(128, 4, dtype=torch.bfloat16) + 0.5
        src = src_base.t()
        assert not src.is_contiguous()

        def fn(d, i, s):
            out = d.clone()
            out.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        idx_unique = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
        compare_with_cpu(
            fn,
            dest,
            idx_unique,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        idx_dup = torch.tensor([0, 0, 2, 4], dtype=torch.int64)
        compare_with_cpu(
            fn,
            dest,
            idx_dup,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ with a non-contiguous index returns values that differ from CPU, #4932",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ with a non-contiguous index returns values that differ from CPU, #4932",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("reduce_mode", _REDUCE_MODES)
    def test_index_reduce_noncontig_index(self, mode: str, reduce_mode: str):
        """IDR-GAP-B (iii): Non-contiguous index — step-2 stride slice.

        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256) — step-2 index over kv-heads.
        idx is derived from a wider buffer via [::2], giving non-unit stride.
        dest and src are contiguous.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dest = torch.rand(16, 256, dtype=torch.bfloat16) + 0.5
        src = torch.rand(4, 256, dtype=torch.bfloat16) + 0.5

        def fn(d, i, s):
            out = d.clone()
            out.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        idx_base_unique = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
        idx_unique = idx_base_unique[::2]
        assert not idx_unique.is_contiguous()
        compare_with_cpu(
            fn,
            dest,
            idx_unique,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        idx_base_dup = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], dtype=torch.int64)
        idx_dup = idx_base_dup[::2]
        assert not idx_dup.is_contiguous()
        compare_with_cpu(
            fn,
            dest,
            idx_dup,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )

    @pytest.mark.parametrize(
        ("reduce_mode", "mode"),
        [
            pytest.param(
                "prod",
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ with a non-contiguous dest, source and index returns values that differ from CPU, #4932",
                ),
            ),
            pytest.param(
                "prod",
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
            pytest.param(
                "mean",
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ with a non-contiguous dest, source and index returns values that differ from CPU, #4911",
                ),
            ),
            pytest.param(
                "mean",
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
            pytest.param(
                "amax",
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ with a non-contiguous dest, source and index returns values that differ from CPU, #4911",
                ),
            ),
            pytest.param(
                "amax",
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
            pytest.param(
                "amin",
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_reduce_ with a non-contiguous dest, source and index returns values that differ from CPU, #4911",
                ),
            ),
            pytest.param(
                "amin",
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor does not support slice_scatter with step=2 for a strided destination view, #4927",
                ),
            ),
        ],
    )
    def test_index_reduce_all_three_noncontig(self, mode: str, reduce_mode: str):
        """IDR-GAP-B (iv): dest, src, and index all non-contiguous simultaneously.

        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128) — all-noncontig 2D kv cache.
        Corresponds to the upstream (dest_noncontig, src_noncontig, index_noncontig)
        = (True, True, True) parametrize combination.  All three tensors have
        non-unit strides; index_reduce_ must correctly resolve addresses for each.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        base_dest = torch.rand(20, 128, dtype=torch.bfloat16) + 0.5
        dest = base_dest[::2, :]
        src_base = torch.rand(128, 4, dtype=torch.bfloat16) + 0.5
        src = src_base.t()
        assert not dest.is_contiguous()
        assert not src.is_contiguous()

        def fn(backing, i, s):
            out = backing.clone()
            dest_view = out[::2, :]
            assert not dest_view.is_contiguous()
            dest_view.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        idx_base_unique = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
        idx_unique = idx_base_unique[::2]
        assert not idx_unique.is_contiguous()
        compare_with_cpu(
            fn,
            base_dest,
            idx_unique,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        idx_base_dup = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], dtype=torch.int64)
        idx_dup = idx_base_dup[::2]
        assert not idx_dup.is_contiguous()
        compare_with_cpu(
            fn,
            base_dest,
            idx_dup,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexPutShapesEagerCompile:
    """index_put / index_put_ and slice assignment: collisions, high-rank geometries, boolean masks."""

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="aten::_index_put_impl_ has no registered Spyre eager kernel, #692",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="index_put_ accumulate with colliding indices keeps one update instead of accumulating them, #4411",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_index_put_half_format_heavy_collision(self, mode: str, dtype: torch.dtype):
        """5000 low-precision updates collide on 8 rows (625 updates per row)."""
        run_eager, run_compile = _mode_flags(mode)
        num_bins, updates_per_bin = (8, 625)
        num_updates = num_bins * updates_per_bin
        self_t = torch.zeros(num_bins, 128, dtype=dtype)
        idx_t = torch.arange(num_updates, dtype=torch.int64) % num_bins
        vals = torch.ones(num_updates, 128, dtype=dtype)

        def fn(dest, index, values):
            out = dest.clone()
            out.index_put_((index,), values, accumulate=True)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1.0,
            rtol=0.01,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_EAGER_COMPILE)
    def test_full_tensor_slice_assignment(self, mode: str):
        """Full-tensor slice assignment out[:] = v — broadcasts scalar-filled tensor to all positions.

        Uses a synthetic (16, 256) tensor shape; it is not an exact Gemma-4-26B KV-cache shape.
        This is not an index_put test.  index_put_((), v) is the semantic equivalent
        but crashes with an internal assert on CPU in PyTorch 2.13 (IndexKernelUtils.h:8).
        This test documents the workaround (out[:] = v) and validates the slice-assignment
        path on Spyre for full-tensor overwrites.
        """
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(16, 256, dtype=torch.bfloat16)
        vals = torch.ones(16, 256, dtype=torch.bfloat16) * 7.0

        def fn(dest, v):
            out = dest.clone()
            out[:] = v
            return out

        compare_with_cpu(
            fn,
            self_t,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_INDEX_PUT_UNSUPPORTED)
    def test_index_put_4d_both(self, mode: str):
        """4D index_put on (batch, kv_heads, seq, head_dim) — both OOP and IP.

        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128). Covers 4D-OOP and 4D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 8, 8, 128, dtype=torch.bfloat16)
        n = 4
        flat = torch.randperm(2 * 8 * 8, dtype=torch.int64)[:n]
        i0 = flat // (8 * 8)
        i1 = flat // 8 % 8
        i2 = flat % 8
        vals = torch.randn(n, 128, dtype=torch.bfloat16)

        def fn_oop(dest, a, b, c, v):
            return torch.index_put(dest, (a, b, c), v, accumulate=False)

        def fn_ip(dest, a, b, c, v):
            out = dest.clone()
            out.index_put_((a, b, c), v, accumulate=False)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            i0,
            i1,
            i2,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            i0,
            i1,
            i2,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_INDEX_PUT_UNSUPPORTED)
    def test_index_put_6d_both(self, mode: str):
        """6D index_put on expert-layer-KV-cache geometry — both OOP and IP.

        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256). Covers 6D-OOP and 6D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 2, 2, 16, 4, 256, dtype=torch.bfloat16)
        n = 8
        flat = torch.randperm(2 * 2 * 2 * 16 * 4, dtype=torch.int64)[:n]
        i0 = flat // (2 * 2 * 16 * 4)
        i1 = flat // (2 * 16 * 4) % 2
        i2 = flat // (16 * 4) % 2
        i3 = flat // 4 % 16
        i4 = flat % 4
        vals = torch.randn(n, 256, dtype=torch.bfloat16)

        def fn_oop(dest, a, b, c, d, e, v):
            return torch.index_put(dest, (a, b, c, d, e), v, accumulate=False)

        def fn_ip(dest, a, b, c, d, e, v):
            out = dest.clone()
            out.index_put_((a, b, c, d, e), v, accumulate=False)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            i0,
            i1,
            i2,
            i3,
            i4,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            i0,
            i1,
            i2,
            i3,
            i4,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=NotImplementedError,
                    reason="aten::_index_put_impl_ has no registered Spyre eager kernel, #692",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="index_put_ with a sparse 3D boolean mask is not supported on Spyre, #6588",
                ),
            ),
        ],
    )
    def test_index_put_sparse_3d_bool_mask_transposed(self, mode: str):
        """Sparse 3D boolean mask on a transposed (non-contiguous) tensor.

        Model shape: Ministral-3-14B (kv_heads=8, seq=4, head_dim=128).
        """
        run_eager, run_compile = _mode_flags(mode)
        torch.manual_seed(0)
        base = torch.zeros(4, 8, 128, dtype=torch.bfloat16)
        dest_t = base.transpose(0, 1)
        mask = torch.rand(8, 4, 128) > 0.85
        num_true = int(mask.sum().item())
        vals = torch.randn(num_true, dtype=torch.bfloat16)

        def fn(dest, m, v):
            out = dest.clone()
            out.index_put_((m,), v, accumulate=False)
            return out

        compare_with_cpu(
            fn,
            dest_t,
            mask,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexCopyEagerCompile:
    """index_copy / index_copy_: non-contiguous 5D destination, zero-volume trailing dim and 6D geometry."""

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=RuntimeError,
                    reason="Eager index_copy.out on a non-contiguous 5D destination re-enters the compiled Spyre kernel, #6587",
                ),
            ),
            "compile",
        ],
    )
    def test_index_copy_5d_noncontiguous_strided(self, mode: str):
        """Non-contiguous 5D transposed destination — strided DMA coordinate correctness.

        Model shape: Granite-3.3-8B (batch=2, kv_heads=8, layers=4, seq=4, head_dim=128).
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        base = torch.zeros(2, 8, 4, 4, 128, dtype=torch.bfloat16)
        dest_t = base.transpose(1, 3)
        num_slices = 2
        src_shape = list(dest_t.shape)
        src_shape[2] = num_slices
        src_t = torch.randn(*src_shape, dtype=torch.bfloat16)
        idx_t = torch.tensor([0, 3], dtype=torch.int64)

        def fn(dest, index, source):
            out = dest.clone()
            out.index_copy_(2, index, source)
            return out

        compare_with_cpu(
            fn,
            dest_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.05,
            rtol=0.05,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor cannot pad the sliced output in place during restickification, #4474",
                ),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=InductorError,
                    reason="Inductor raises Unexpected stick expression for the zero-volume layout, #3455",
                ),
            ),
        ],
    )
    def test_index_copy_zero_volume_trailing_dim(self, mode: str):
        """Zero-volume trailing dimension — index_copy must be a safe no-op.

        Model shape: Ministral-3-14B (kv_heads=8, seq=4, head_dim=0 zero-volume).
        """
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(8, 4, 0, dtype=torch.bfloat16)
        src_t = torch.zeros(8, 2, 0, dtype=torch.bfloat16)
        idx_t = torch.tensor([1, 3], dtype=torch.int64)

        def fn(dest, index, source):
            out = dest.clone()
            out.index_copy_(1, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_EAGER_COMPILE)
    def test_index_copy_6d_both(self, mode: str):
        """6D index_copy on expert-layer-KV-cache geometry — both OOP and IP.

        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128). Covers 6D-OOP and 6D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        self_t = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 2, 2, 8, 2, 128, dtype=torch.bfloat16)
        idx_t = torch.randperm(4, dtype=torch.int64)[:2]

        def fn_oop(dest, index, source):
            return torch.index_copy(dest, 4, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.index_copy_(4, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.005,
            rtol=0.005,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestPutNonContigEagerCompile:
    """torch.put / put_: non-contiguous destination, source and index, and 5D / 6D geometries."""

    @pytest.mark.parametrize("mode", _MODE_PARAMS_PUT_UNSUPPORTED)
    @pytest.mark.parametrize("accumulate", _PUT_ACCUMULATE_OPTIONS)
    def test_put_2d_dst_noncontig(self, mode: str, accumulate: bool):
        """PUT-01b: 2D non-contiguous dst — flat index must respect multi-stride address layout.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128) — stride-2 view over kv-heads.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        base = torch.zeros(16, 128, dtype=torch.bfloat16)
        dst = base[::2, :]
        src = torch.randn(4, dtype=torch.bfloat16)
        if accumulate:
            idx = torch.randint(0, dst.numel(), (4,), dtype=torch.int64)
        else:
            idx = torch.randperm(dst.numel(), dtype=torch.int64)[:4]

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_PUT_UNSUPPORTED)
    @pytest.mark.parametrize("accumulate", _PUT_ACCUMULATE_OPTIONS)
    def test_put_noncontig_source(self, mode: str, accumulate: bool):
        """PUT-GAP-B (i): Non-contiguous source tensor.

        Model shape: Granite-4.1-20B (head_dim=128) — 1D flat KV slot buffer.
        Source is a step-2 slice of a larger 1-D buffer (non-unit stride).
        The destination and index are fully contiguous.  torch.put must
        flatten and read the source through its stride rather than assuming
        a packed memory layout.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dst = torch.zeros(128, dtype=torch.bfloat16)
        src_base = torch.randn(16, dtype=torch.bfloat16)
        src = src_base[::2]
        assert not src.is_contiguous()
        if accumulate:
            idx = torch.randint(0, 128, (8,), dtype=torch.int64)
        else:
            idx = torch.randperm(128, dtype=torch.int64)[:8]

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_PUT_UNSUPPORTED)
    @pytest.mark.parametrize("accumulate", _PUT_ACCUMULATE_OPTIONS)
    def test_put_noncontig_index(self, mode: str, accumulate: bool):
        """PUT-GAP-B (ii): Non-contiguous index tensor.

        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256) — 1D flat write into 16×256 cache.
        Index is a step-2 slice (non-unit stride); destination and source are
        contiguous.  torch.put flattens the index through its stride to resolve
        the target flat positions.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dst = torch.zeros(256, dtype=torch.bfloat16)
        src = torch.randn(8, dtype=torch.bfloat16)
        if accumulate:
            idx_base = torch.randint(0, 256, (16,), dtype=torch.int64)
        else:
            selected = torch.randperm(256, dtype=torch.int64)[:8]
            idx_base = torch.empty(16, dtype=torch.int64)
            idx_base[::2] = selected
            idx_base[1::2] = 0
        idx = idx_base[::2]
        assert not idx.is_contiguous()

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_PUT_UNSUPPORTED)
    @pytest.mark.parametrize("accumulate", _PUT_ACCUMULATE_OPTIONS)
    def test_put_all_three_noncontig(self, mode: str, accumulate: bool):
        """PUT-GAP-B (iii): dst, src, and index all non-contiguous simultaneously.

        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128) — 2D kv cache, all non-contig.
        This is the hardest non-contiguous path: torch.put must convert the
        destination to a flattened contiguous view and read both src and index
        through their respective non-unit strides.  Corresponds directly to the
        upstream parametrize over (dst_contig, src_contig, idx_contig) = (F,F,F).
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        base_dst = torch.zeros(20, 128, dtype=torch.bfloat16)
        dst = base_dst[::2, :]
        src_base = torch.randn(16, dtype=torch.bfloat16)
        src = src_base[::2]
        if accumulate:
            idx_raw = torch.randint(0, dst.numel(), (16,), dtype=torch.int64)
        else:
            selected = torch.randperm(dst.numel(), dtype=torch.int64)[:8]
            idx_raw = torch.empty(16, dtype=torch.int64)
            idx_raw[::2] = selected
            idx_raw[1::2] = 0
        idx = idx_raw[::2]
        assert not dst.is_contiguous()
        assert not src.is_contiguous()
        assert not idx.is_contiguous()

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_PUT_UNSUPPORTED)
    @pytest.mark.parametrize("accumulate", _PUT_ACCUMULATE_OPTIONS)
    def test_put_5d_both(self, mode: str, accumulate: bool):
        """5D torch.put and put_ — both OOP and IP on layer-KV-cache geometry.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128). Covers 5D-OOP and 5D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dst = torch.zeros(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        src = torch.randn(8, dtype=torch.bfloat16)
        if accumulate:
            idx = torch.randint(0, dst.numel(), (8,), dtype=torch.int64)
        else:
            idx = torch.randperm(dst.numel(), dtype=torch.int64)[:8]

        def fn_oop(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        def fn_ip(d, i, s):
            out = d.clone()
            out.put_(i, s, accumulate=accumulate)
            return out

        compare_with_cpu(
            fn_oop,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )
        compare_with_cpu(
            fn_ip,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

    @pytest.mark.parametrize("mode", _MODE_PARAMS_PUT_UNSUPPORTED)
    @pytest.mark.parametrize("accumulate", _PUT_ACCUMULATE_OPTIONS)
    def test_put_6d_both(self, mode: str, accumulate: bool):
        """6D torch.put and put_ on expert-layer-KV-cache geometry — both OOP and IP.

        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equiv). Covers 6D-OOP and 6D-IP.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)
        dst = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        src = torch.randn(8, dtype=torch.bfloat16)
        if accumulate:
            idx = torch.randint(0, dst.numel(), (8,), dtype=torch.int64)
        else:
            idx = torch.randperm(dst.numel(), dtype=torch.int64)[:8]

        def fn_oop(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        def fn_ip(d, i, s):
            out = d.clone()
            out.put_(i, s, accumulate=accumulate)
            return out

        compare_with_cpu(
            fn_oop,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )
        compare_with_cpu(
            fn_ip,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.01,
            rtol=0.016,
        )

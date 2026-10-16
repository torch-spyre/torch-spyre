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

"""Value dtype coverage (float16/bfloat16/float32/int32/int64/bool), index dtype variants (int32/int64), int64 auto-downcast warning, downstream dtype fusion, and SDSC wordLength validation."""

import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import cached_randn  # noqa: E402
from conftest import _xfail_existing, compare_mode  # noqa: E402

_INDEX_EAGER = (1219, "aten::index.Tensor_out is not registered on Spyre.")
_INT32_TO_INT64_CONV = (
    4334,
    "eager advanced index type conversion from torch.int32 to torch.int64.",
)
_BOOL_OPERANDS = (
    4332,
    "torch.bool result from operands with device format(s) {SEN169_FP16, IEEE_INT32}.",
)
_DOUBLE_UNSUPPORTED = (
    1228,
    "RuntimeError: Spyre backend does not support dtype Double.",
)


class TestGatherValueAndIndexDtypeCoverage:
    """Value dtype coverage (float16/bfloat16/float32/int32/int64/bool), index dtype variants (int32/int64), int64→int32 auto-downcast, downstream dtype fusion, and SDSC wordLength validation."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self):
        yield
        os.environ.pop("TORCH_SPYRE_DOWNCAST_WARN", None)

    # -- value dtype variants ---------------------------------------------

    def test_gather_bfloat16_value(self, execution_mode):
        """bfloat16 value + int32 index; SDSC wordLength=2."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4334
        _xfail_existing(execution_mode, eager=_INT32_TO_INT64_CONV)
        x = cached_randn((32, 128), differentiation="gdt01", dtype=torch.bfloat16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=2e-2, rtol=2e-2)

    def test_gather_float32_value(self, execution_mode):
        """float32 value + int32 index; 4 bytes/element layout."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4334
        _xfail_existing(execution_mode, eager=_INT32_TO_INT64_CONV)
        x = cached_randn((32, 128), differentiation="gdt02", dtype=torch.float32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=1e-5, rtol=1e-5)

    def test_gather_int32_value(self, execution_mode):
        """Integer value tensor gather; exact integer output."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4334
        _xfail_existing(execution_mode, eager=_INT32_TO_INT64_CONV)
        x = torch.randint(-1000, 1000, (32, 64), dtype=torch.int32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_gather_int64_value(self, execution_mode):
        """int64 value tensor; 8 bytes/element."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4334
        _xfail_existing(execution_mode, eager=_INT32_TO_INT64_CONV)
        x = torch.randint(-10000, 10000, (32, 64), dtype=torch.int64)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_gather_bool_value(self, execution_mode):
        """Boolean value gather; 1-byte elements."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4334
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4332
        _xfail_existing(
            execution_mode,
            eager=_INT32_TO_INT64_CONV,
            compiled=_BOOL_OPERANDS,
        )
        x = torch.randint(0, 2, (32, 64)).bool()
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # -- index dtype variants and downcast --------------------------------

    @pytest.mark.parametrize(
        "val_dtype,diff_key,atol",
        [
            (torch.float16, "gdt06", 1e-2),
            (torch.bfloat16, "gdt08", 2e-2),
            (torch.float32, "gdt10", 1e-5),
        ],
    )
    def test_gather_int64_index_downcast(
        self, execution_mode, val_dtype, diff_key, atol
    ):
        """int64 index auto-downcast to int32; warning emitted for each value dtype."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation=diff_key, dtype=val_dtype)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            compare_mode(
                execution_mode, lambda x, i: x[i], x, idx, atol=atol, rtol=atol
            )

    def test_gather_int32_index_bfloat16(self, execution_mode):
        """bfloat16 value + int32 index at production scale (512,256) — standard combo."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4334
        _xfail_existing(execution_mode, eager=_INT32_TO_INT64_CONV)
        x = cached_randn((512, 256), differentiation="gdt07", dtype=torch.bfloat16)
        idx = torch.randint(0, 512, (64,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=2e-2, rtol=2e-2)

    def test_gather_int32_index_float32(self, execution_mode):
        """float32 value + int32 index; no downcast path."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4334
        _xfail_existing(execution_mode, eager=_INT32_TO_INT64_CONV)
        x = cached_randn((32, 128), differentiation="gdt09", dtype=torch.float32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=1e-5, rtol=1e-5)

    def test_gather_float64_not_gathered(self, execution_mode):
        """float64 must NOT produce GATHER_OP_SPEC; falls back to CPU."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1228
        _xfail_existing(
            execution_mode, eager=_DOUBLE_UNSUPPORTED, compiled=_DOUBLE_UNSUPPORTED
        )
        x = cached_randn((32, 64), differentiation="gdt_f64", dtype=torch.float64)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_gather_downcast_warning_suppressed(self, execution_mode):
        """TORCH_SPYRE_DOWNCAST_WARN=0 suppresses int64→int32 warning; result still correct."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        os.environ["TORCH_SPYRE_DOWNCAST_WARN"] = "0"
        x = cached_randn((32, 128), differentiation="gdt12", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compare_mode(
                execution_mode, lambda x, i: x[i], x, idx, atol=1e-2, rtol=1e-2
            )
        spyre_warns = [
            str(m.message) for m in w if "downcast" in str(m.message).lower()
        ]
        assert len(spyre_warns) == 0, f"Unexpected downcast warnings: {spyre_warns}"

    # -- downstream dtype combinations ------------------------------------

    @pytest.mark.parametrize(
        "val_dtype,downstream,diff_key,atol",
        [
            (torch.bfloat16, torch.exp, "gdt13", 2e-2),
            (torch.float32, torch.abs, "gdt14", 1e-5),
        ],
    )
    def test_gather_with_downstream_dtype(
        self, execution_mode, val_dtype, downstream, diff_key, atol
    ):
        """gather + downstream op fused; dtype-specific tolerance."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4334
        _xfail_existing(execution_mode, eager=_INT32_TO_INT64_CONV)
        x = cached_randn((32, 128), differentiation=diff_key, dtype=val_dtype)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: downstream(x[i]), x, idx, atol=atol, rtol=atol
        )

    # -- SDSC wordLength checks (structural; run compile, check output) ---

    @pytest.mark.parametrize(
        "idx_dtype,diff_key",
        [
            (torch.int32, "gdt15"),
            (torch.int64, "gdt16"),
        ],
    )
    def test_sdsc_wordlength(self, execution_mode, idx_dtype, diff_key):
        """SDSC wordLength validated via compile run for int32 and int64 index dtypes."""
        if idx_dtype == torch.int32:
            # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4334
            _xfail_existing(execution_mode, eager=_INT32_TO_INT64_CONV)
        else:
            # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
            _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation=diff_key, dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=idx_dtype)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            compare_mode(
                execution_mode, lambda x, i: x[i], x, idx, atol=1e-2, rtol=1e-2
            )

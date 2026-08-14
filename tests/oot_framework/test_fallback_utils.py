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
"""Unit tests for fallback_utils classification (device-free)."""

import pytest

from oot_framework.fallback_utils import (
    _is_known_gap,
    assert_no_cpu_fallback,
)

_COS = "aten.cos.default is falling back to cpu"
_CONV = "conversion from torch.int64 to torch.float32 is falling back to cpu"
_INDEX_COPY = "aten.index_copy.out is falling back to cpu"


def test_no_messages_is_noop():
    assert_no_cpu_fallback("case", [])  # must not raise


def test_known_gap_ops_do_not_fail():
    assert _is_known_gap(_COS)
    assert _is_known_gap(_CONV)
    # sin/arange are also known gaps
    assert _is_known_gap("aten.sin.default is falling back to cpu")
    assert _is_known_gap("aten.arange.default is falling back to cpu")
    # reporting only; must not raise
    assert_no_cpu_fallback("case", [_COS, _CONV])


def test_kernel_backed_fallback_fails():
    # index_copy has a device kernel -> a fallback masks a bug -> must fail.
    assert not _is_known_gap(_INDEX_COPY)
    with pytest.raises(AssertionError, match="masking a device bug"):
        assert_no_cpu_fallback("case", [_INDEX_COPY])


def test_allow_env_downgrades_to_warning(monkeypatch):
    monkeypatch.setenv("SPYRE_OPTEST_ALLOW_CPU_FALLBACK", "1")
    # even a bug-masking fallback is downgraded to a print, not a failure
    assert_no_cpu_fallback("case", [_INDEX_COPY])

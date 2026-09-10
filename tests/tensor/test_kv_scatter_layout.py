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

# Owner(s): ["module: cpp"]

"""Regression guard for the KV-scatter layout requirement (torch-spyre#3705).

An indirect row scatter (``cache.index_copy(0, idx, src)``) is only correct on
Spyre when the *indexed* dimension -- the cache-position dim, dim 0 here -- is
pinned to device position 0 (outermost). With a position-major
``SpyreTensorLayout`` the scatter lands on exactly the rows named by ``idx``.

With the *default* layout, the last logical dim becomes the stick-count and is
placed outermost, so the indexed dim arrives at device position 1, behind the
stick dim (cf. ``canonical_device_layout`` in
``tests/inductor/indirect_access_common.py``). The indirect scatter then
silently writes the *wrong* rows (or the backend aborts on the bundle -- cf.
``tests/inductor/test_indirect_access_scatter.py``, which xfails the same
``index_copy`` on a plain layout). That is torch-spyre#3705.

This file pins the workaround the hf-adapters KV cache is built on: the
pinned-layout scatter is correct, reuses a single compiled binary across every
decode position, and survives a scatter->gather round trip. It also encodes the
negative case as an ``xfail`` so it flips to xpass the day #3705 is fixed.
"""

import pytest
import torch
from torch.spyre import SpyreTensorLayout, get_device_dtype
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
)

DEVICE = "spyre"


def _position_major_stl(L, rows, hd, dtype):
    """Layout for logical ``[L, rows, hd]`` with the indexed dim ``L`` pinned to
    device position 0 (outermost).

    Mirrors ``canonical_device_layout`` in
    ``tests/inductor/indirect_access_common.py`` for a 3-D operand: lead dims
    stay outermost, the last dim is split into (stick-count, elems-per-stick).
    """
    eps = SpyreTensorLayout([L, rows, hd], dtype).elems_per_stick()
    sticks = (hd + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[L, rows, sticks, eps],
        stride_map=[rows * hd, hd, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def _scatter(cache, idx, src):
    """Row scatter under test: write ``src`` rows into ``cache`` at ``idx``."""
    return cache.index_copy(0, idx, src)


def _gather(cache, idx):
    """Read the rows at ``idx`` back out (how attention consumes the cache)."""
    return cache[idx]


def _tol(dtype):
    """(atol, rtol) for a device round trip.

    Device fp16 storage rounds to ~4.9e-4, so 2e-3 sits above real precision
    without masking a wrong-row regression (see test_tensor_layout.py). bf16's
    ~8-bit mantissa is coarser (~3.9e-3 near 1.0), so it needs 1e-2.
    """
    return (1e-2, 1e-2) if dtype == torch.bfloat16 else (2e-3, 2e-3)


def _write_positions(L, nwrite):
    """A spread of valid start positions: origin, around a stick boundary (64),
    mid-cache, and the last in-bounds start."""
    last = L - nwrite
    candidates = {0, 1, 63, 64, 128, 320, 512, last}
    return sorted(p for p in candidates if 0 <= p <= last)


# (L, rows, hd, dtype, nwrite) -- the per-layer shapes hf-adapters exercises.
# rows folds [B * nkv]: B=1 -> rows=nkv; B=4, nkv=8 -> rows=32. nwrite=1 is the
# decode step (the compile-explosion case), nwrite=64 a prefill/block write.
_PINNED_CASES = [
    subtest((576, 8, 128, torch.float16, 1), name="granite8b_decode_fp16"),
    subtest((576, 8, 256, torch.float16, 1), name="padded_hd256_decode_fp16"),
    subtest((576, 1, 128, torch.float16, 1), name="mqa_decode_fp16"),
    subtest((576, 32, 128, torch.float16, 1), name="batched_b4_decode_fp16"),
    subtest((576, 8, 128, torch.bfloat16, 1), name="granite8b_decode_bf16"),
    subtest((576, 8, 128, torch.float16, 64), name="granite8b_block_fp16"),
    subtest((576, 32, 128, torch.float16, 64), name="batched_b4_block_fp16"),
    subtest((576, 8, 128, torch.bfloat16, 64), name="granite8b_block_bf16"),
    subtest((2048, 8, 128, torch.float16, 1), name="granite8b_L2048_decode"),
    subtest((2048, 8, 128, torch.float16, 64), name="granite8b_L2048_block"),
]

_ROUNDTRIP_CASES = [
    subtest((576, 8, 128, torch.float16, 1, 137), name="fp16_1row_pos137"),
    subtest((576, 8, 128, torch.float16, 64, 192), name="fp16_64row_pos192"),
    subtest((576, 8, 128, torch.bfloat16, 64, 256), name="bf16_64row_pos256"),
]


@instantiate_parametrized_tests
class TestKVScatterLayout(TestCase):
    """Pin the position-major KV-cache scatter contract (torch-spyre#3705)."""

    def setUp(self):
        torch.manual_seed(0xAFFE)
        # Lazy device init (mirrors test_tensor_layout.py's `x.to("spyre")`).
        torch.zeros(1, dtype=torch.float16).to(DEVICE)

    @parametrize("L,rows,hd,dtype,nwrite", _PINNED_CASES)
    def test_pinned_layout_scatters_correct_rows(self, L, rows, hd, dtype, nwrite):
        """A cache whose indexed dim is pinned to device position 0 scatters
        into exactly the rows named by ``idx``.

        Primary regression guard for torch-spyre#3705: the pinned-layout
        indirect ``index_copy`` must match the CPU ``index_copy_`` reference
        row-for-row at every write position. This is the workaround the
        hf-adapters KV cache relies on and must stay green.
        """
        stl = _position_major_stl(L, rows, hd, dtype)
        atol, rtol = _tol(dtype)
        compiled = torch.compile(_scatter, dynamic=False)
        for pos in _write_positions(L, nwrite):
            cache = torch.rand(L, rows, hd, dtype=dtype)
            src = torch.rand(nwrite, rows, hd, dtype=dtype)
            idx = torch.arange(pos, pos + nwrite, dtype=torch.int32)
            # CPU golden: index_copy_ needs an int64 index.
            want = cache.clone().index_copy_(0, idx.long(), src)
            got = compiled(
                cache.to(DEVICE, device_layout=stl),
                idx.to(DEVICE),
                src.to(DEVICE),
            ).cpu()
            self.assertEqual(got.shape, want.shape)
            self.assertEqual(
                got.float(),
                want.float(),
                msg=f"wrong rows scattered at pos={pos} (L={L}, nwrite={nwrite})",
                atol=atol,
                rtol=rtol,
            )

    @parametrize("L,rows,hd,dtype,nwrite,pos", _ROUNDTRIP_CASES)
    def test_scatter_gather_roundtrip(self, L, rows, hd, dtype, nwrite, pos):
        """Scatter rows in, then gather the same rows back out: the values must
        survive the scatter->gather round trip attention runs on the cache."""
        stl = _position_major_stl(L, rows, hd, dtype)
        atol, rtol = _tol(dtype)
        cache = torch.zeros(L, rows, hd, dtype=dtype)
        # Offset away from zero so a dropped write is not masked by the zeros.
        src = torch.rand(nwrite, rows, hd, dtype=dtype) + 0.5
        idx = torch.arange(pos, pos + nwrite, dtype=torch.int32)

        written = torch.compile(_scatter, dynamic=False)(
            cache.to(DEVICE, device_layout=stl),
            idx.to(DEVICE),
            src.to(DEVICE),
        )
        got = torch.compile(_gather, dynamic=False)(written, idx.to(DEVICE)).cpu()
        self.assertEqual(
            got.float(),
            src.float(),
            msg=f"round-trip corrupted rows at pos={pos}",
            atol=atol,
            rtol=rtol,
        )

    def test_pinned_layout_single_binary_across_positions(self):
        """One compiled binary must serve every write position -- the point of
        the pinned-layout rewrite.

        Decode steps at different cache positions must hit the SAME graph, not
        recompile per position, while still landing on the correct rows. Guards
        both correctness and the compile explosion the workaround avoids.
        """
        L, rows, hd, dtype, nwrite = 576, 8, 128, torch.float16, 1
        stl = _position_major_stl(L, rows, hd, dtype)
        atol, rtol = _tol(dtype)

        torch._dynamo.reset()
        torch._dynamo.utils.counters["stats"].clear()
        compiled = torch.compile(_scatter, dynamic=False)

        for pos in _write_positions(L, nwrite):
            cache = torch.rand(L, rows, hd, dtype=dtype)
            src = torch.rand(nwrite, rows, hd, dtype=dtype)
            idx = torch.arange(pos, pos + nwrite, dtype=torch.int32)
            want = cache.clone().index_copy_(0, idx.long(), src)
            got = compiled(
                cache.to(DEVICE, device_layout=stl),
                idx.to(DEVICE),
                src.to(DEVICE),
            ).cpu()
            self.assertEqual(
                got.float(),
                want.float(),
                msg=f"wrong rows scattered at pos={pos}",
                atol=atol,
                rtol=rtol,
            )

        graphs = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
        self.assertLessEqual(
            graphs,
            1,
            f"expected a single compiled binary across positions, got {graphs}",
        )

    @pytest.mark.xfail(
        reason=(
            "torch-spyre#3705: with the cache-position dim NOT pinned to device "
            "position 0 (the default/plain layout), the indirect scatter silently "
            "writes the wrong rows, or the backend aborts on the bundle. Remove "
            "this xfail (and promote to a hard assertion) when #3705 is fixed."
        ),
        strict=False,
    )
    def test_default_layout_scatters_wrong_rows(self):
        """Negative guard: the SAME ``index_copy`` on the DEFAULT (non-pinned)
        device layout does NOT land on the intended rows -- the bug behind
        torch-spyre#3705.

        We assert the *correct* placement; today that assertion fails (wrong
        rows) or the backend aborts, so this is xfail. If it ever passes, #3705
        has been fixed and this guard should be promoted/removed. Compare
        tests/inductor/test_indirect_access_scatter.py::test_index_copy, which
        xfails the same op on a plain layout.
        """
        L, rows, hd, dtype, nwrite, pos = 576, 8, 128, torch.float16, 1, 137
        atol, rtol = _tol(dtype)
        cache = torch.rand(L, rows, hd, dtype=dtype)
        src = torch.rand(nwrite, rows, hd, dtype=dtype)
        idx = torch.arange(pos, pos + nwrite, dtype=torch.int32)
        want = cache.clone().index_copy_(0, idx.long(), src)
        got = torch.compile(_scatter, dynamic=False)(
            cache.to(DEVICE),  # default layout: indexed dim NOT at position 0
            idx.to(DEVICE),
            src.to(DEVICE),
        ).cpu()
        self.assertEqual(
            got.float(),
            want.float(),
            msg="default (non-pinned) layout mis-scattered rows",
            atol=atol,
            rtol=rtol,
        )


if __name__ == "__main__":
    run_tests()

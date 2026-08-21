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

"""Consolidated scatter-style indirect-access tests (one file per op family).

Each scenario routes its compile through
self._stage_and_e2e(...): it asserts across every capture-path stage --
classification, op-spec structure (IndirectAccess on the output), and SDSC
fields -- and then runs the kernel end-to-end on the real backend.

All scatter scenarios run with SENCORES=1.

"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    CRASHED,
    GATHER_OP_SPEC,
    SCATTER_OP_SPEC,
    DIRECT_OP_SPEC,
    IndirectAccessTestCase,
)

from torch_spyre._C import (  # noqa: E402
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)
from torch_spyre._inductor import config  # noqa: E402


@config.patch({"sencores": 1})
class TestScatter(IndirectAccessTestCase):
    """torch scatter-family ops: one compile + all-stage checks per scenario."""

    def _row_store(self, M=128, N=256, P=3, dtype=torch.int32):
        """Common row-store operands: out[M,N], src[P,N], 1-D idx[P], all named."""
        out = torch.zeros(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(P, N, dtype=torch.float16).to("spyre")
        idx = torch.arange(P, dtype=dtype).to("spyre")
        self.name_dims(out, {"M": M, "N": N})
        self.name_dims(src, {"P": P, "N": N})
        self.name_dims(idx, {"P": P})
        return out, src, idx

    def _full_index_store(self, M=128, N=256, P=3, dtype=torch.int32):
        """Operands for scatter with a full [P,N] index tensor: out[M,N], src[P,N].
        Index is uniform per row (all N columns in a row scatter to the same target
        row) to ensure collision-free unique indices; per-element variation within
        a row is not exercised."""
        out = torch.zeros(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(P, N, dtype=torch.float16).to("spyre")
        index = (
            torch.randperm(M, dtype=dtype)[:P]
            .unsqueeze(1)
            .expand(P, N)
            .contiguous()
            .to("spyre")
        )
        self.name_dims(out, {"M": M, "N": N})
        self.name_dims(src, {"P": P, "N": N})
        self.name_dims(index, {"P": P, "N": N})
        return out, src, index

    # -- Working index-tensor scatters: op spec with output IndirectAccess --
    def test_index_put(self):
        """out[idx] = src"""
        out, src, idx = self._row_store()

        def kernel(out, src, idx):
            out[idx] = src
            return out

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

    def test_index_put_with_exp(self):
        """out[idx] = src.exp() -- index_put fused with a unary operation."""
        out, src, idx = self._row_store()

        def kernel(out, src, idx):
            out[idx] = src.exp()
            return out

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC, op="exp")

    def test_index_put_batched_dim1(self):
        """y[:, i] = src -- batched scatter indexing a non-leading dim.

        y[B,M,N], src[B,P,N], 1-D idx[P] shared across the batch: every batch
        slice scatters its P rows into the same M-dim positions.
        """
        Bn, M, N, P = 4, 16, 1024, 6
        y = torch.rand(Bn, M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(Bn, P, N, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(y, {"B": Bn, "M": M, "N": N})
        self.name_dims(src, {"B": Bn, "P": P, "N": N})
        self.name_dims(idx, {"P": P})

        def kernel(y, src, idx):
            y[:, idx] = src
            return y

        self._stage_and_e2e(kernel, y, src, idx, expect=SCATTER_OP_SPEC)

    def test_index_put_4d_dim2_default_layout_destination(self):
        """Scatter on dim 2 of a 4-D destination that has the DEFAULT device layout.

        Indirect access addresses the indexed dimension through the *device* layout and
        requires it at device position 0. A destination allocated plainly --
        ``torch.zeros(...).to("spyre")`` -- puts dim 2 elsewhere, so unless the layout
        is enforced the scatter writes the wrong rows **silently, with no error**
        (torch-spyre#3705).

        The existing dim-0 and dim-1 scatter tests above do not cover this: their
        shapes happen to place the indexed dim at device position 0 already. This is
        the shape a decode-time query stick uses (``[batch, heads, 64, head_dim]``,
        writing one row), which is where the silent corruption was first observed.
        """
        Bn, H, M, N, P = 1, 4, 64, 256, 1
        dst = torch.zeros(Bn, H, M, N, dtype=torch.float16)
        src = torch.rand(Bn, H, P, N, dtype=torch.float16)
        # int64: index_copy_ requires a long index, unlike the index_put tests above.
        idx = torch.tensor([7], dtype=torch.int64)

        def kernel(dst, src, idx):
            dst.index_copy_(2, idx, src)
            return dst

        expected = kernel(dst.clone(), src.clone(), idx.clone())
        actual = torch.compile(kernel, dynamic=False)(
            dst.clone().to("spyre"), src.to("spyre"), idx.to("spyre")
        ).to("cpu")
        torch.testing.assert_close(actual, expected)

    def test_index_put_p7(self):
        """y[idx] = src -- 1-D scatter with an odd (non-power-of-2) P=7."""
        M, N, P = 16, 1024, 7
        y = torch.rand(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(P, N, dtype=torch.float16).to("spyre")
        idx = torch.arange(P, dtype=torch.int32).to("spyre")
        self.name_dims(y, {"M": M, "N": N})
        self.name_dims(src, {"P": P, "N": N})
        self.name_dims(idx, {"P": P})

        def kernel(y, src, idx):
            y[idx] = src
            return y

        self._stage_and_e2e(kernel, y, src, idx, expect=SCATTER_OP_SPEC)

    def test_index_put_3d_dim0(self):
        """y[idx] = src -- 3-D scatter on dim 0.

        y[M,N,K], src[P,N,K], 1-D idx[P]: scatter P rows into M dimension
        of a 3-D tensor.
        """
        M, N, K, P = 32, 64, 128, 8
        y = torch.rand(M, N, K, dtype=torch.float16).to("spyre")
        src = torch.rand(P, N, K, dtype=torch.float16).to("spyre")
        idx = torch.arange(P, dtype=torch.int32).to("spyre")
        self.name_dims(y, {"M": M, "N": N, "K": K})
        self.name_dims(src, {"P": P, "N": N, "K": K})
        self.name_dims(idx, {"P": P})

        def kernel(y, src, idx):
            y[idx] = src
            return y

        self._stage_and_e2e(kernel, y, src, idx, expect=SCATTER_OP_SPEC)

    def test_index_put_4d_dim0(self):
        """y[idx] = src -- 4-D scatter on dim 0.

        y[M,N,K,L], src[P,N,K,L], 1-D idx[P]: scatter P rows into M dimension
        of a 4-D tensor.
        """
        M, N, K, L, P = 16, 32, 64, 256, 6
        y = torch.rand(M, N, K, L, dtype=torch.float16).to("spyre")
        src = torch.rand(P, N, K, L, dtype=torch.float16).to("spyre")
        idx = torch.arange(P, dtype=torch.int32).to("spyre")
        self.name_dims(y, {"M": M, "N": N, "K": K, "L": L})
        self.name_dims(src, {"P": P, "N": N, "K": K, "L": L})
        self.name_dims(idx, {"P": P})

        def kernel(y, src, idx):
            y[idx] = src
            return y

        self._stage_and_e2e(kernel, y, src, idx, expect=SCATTER_OP_SPEC)

    def test_scatter(self):
        """torch.scatter(out, 0, index, src)"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return torch.scatter(out, 0, index, src)

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    def test_scatter_method_without_unary(self):
        """out.scatter_(0, index, src) -- in-place method form without a unary."""
        out, src, index = self._full_index_store()

        def kernel(out, src, index):
            return out.scatter_(0, index, src)

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    def test_scatter_with_exp(self):
        """y.scatter_(0, index, src.exp()) -- fused unary, exp runs on Spyre.

        Also pins the detection gap: indirect_info_from_op flags gather
        loads but not scatter stores (the output is recognized later in
        superdsc via is_output_tensor), so detected=False here.
        """
        out, src, index = self._full_index_store()

        def kernel(out, src, index):
            return out.scatter_(0, index, src.exp())

        self._stage_and_e2e(
            kernel,
            out,
            src,
            index,
            expect=SCATTER_OP_SPEC,
            op="exp",
            detected=False,
        )

    def test_scatter_add(self):
        """y.scatter_add_(0, index, src)"""
        out, src, index = self._full_index_store()

        def kernel(out, src, index):
            return out.scatter_add_(0, index, src)

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    def test_index_copy(self):
        """torch.index_copy(out, 0, idx, src).

        index_copy requires a long (int64) index, unlike the int32-friendly
        index_put/index_add, so the CPU reference needs an int64 index here.
        """
        out, src, idx = self._row_store(dtype=torch.int64)

        def kernel(out, src, idx):
            return torch.index_copy(out, 0, idx, src)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

    def _paged_cache_layout(self, L=576, H=8, D=128):
        """The paged-KV-cache device layout for a [L, H, D] fp16 tensor: L
        (the indirectly-accessed dim) outermost, then H, then D split into
        (stick_count, elems_per_stick) -- mirroring the real paged-KV-cache
        layout used by attention decode/prefill (see test_paged.py)."""
        eps = get_elem_in_stick(torch.float16)
        return SpyreTensorLayout(
            device_size=[L, H, (D + eps - 1) // eps, eps],
            stride_map=[H * D, D, eps, 1],
            device_dtype=get_device_dtype(torch.float16),
        )

    def _paged_kv_cache_operands(self, L=576, H=8, D=128, P=3):
        """A paged-KV-store-shaped scatter target: cache[L, H, D] on the paged
        device layout, plus a matching src[P, H, D] and an int64 idx[P]."""
        stl = self._paged_cache_layout(L=L, H=H, D=D)
        cache = torch.rand(L, H, D, dtype=torch.float16)
        src = torch.rand(P, H, D, dtype=torch.float16)
        idx = torch.randperm(L, dtype=torch.int64)[:P]
        cache_dev = cache.to("spyre", device_layout=stl)
        src_dev = src.to("spyre")
        idx_dev = idx.to("spyre")
        return cache_dev, src_dev, idx_dev

    def _assert_compiled_matches_cpu(self, kernel, *dev_args):
        """Run `kernel` through torch.compile on device and require the result
        to match the CPU eager reference -- a hard assertion, not an xfail.
        Compute CPU reference from pristine inputs before compiled run to catch
        mutations that corrupt non-indexed regions."""
        cpu_args = [
            a.cpu()
            if isinstance(a, torch.Tensor) and a.device.type == "spyre"
            else a.clone()
            if isinstance(a, torch.Tensor)
            else a
            for a in dev_args
        ]
        reference = kernel(*cpu_args)
        result = torch.compile(kernel, dynamic=False)(*dev_args)
        torch.testing.assert_close(result.cpu(), reference)

    def test_index_copy_e2e(self):
        """torch.compile(index_copy) against a paged-KV-cache-shaped device
        layout ([576, 8, 128]), run e2e and
        require the result to match the CPU reference."""
        cache, src, idx = self._paged_kv_cache_operands()

        def kernel(c, s, i):
            return c.index_copy(0, i, s)

        self._assert_compiled_matches_cpu(kernel, cache, src, idx)

    def test_index_copy_decode_e2e(self):
        """Decode-shaped variant: a single-token write (P=1), the common
        per-step KV-cache update pattern during autoregressive decode."""
        cache, src, idx = self._paged_kv_cache_operands(P=1)

        def kernel(c, s, i):
            return c.index_copy(0, i, s)

        self._assert_compiled_matches_cpu(kernel, cache, src, idx)

    def test_index_put_e2e(self):
        """y[idx] = src against a paged-KV-cache-shaped device layout --
        index_put's simpler assignment form, run e2e against the CPU
        reference (cf. test_index_put's default-layout version)."""
        y, src, idx = self._paged_kv_cache_operands()

        def kernel(y, src, idx):
            y[idx] = src
            return y

        self._assert_compiled_matches_cpu(kernel, y, src, idx)

    def test_index_put_second_target_e2e(self):
        """z[idx] = src against a *second*, differently-shaped paged-cache
        tensor (fewer, wider rows) -- pins that the paged-cache path isn't
        special-cased to one shape."""
        z, src, idx = self._paged_kv_cache_operands(L=288, H=4, D=128, P=5)

        def kernel(z, src, idx):
            z[idx] = src
            return z

        self._assert_compiled_matches_cpu(kernel, z, src, idx)

    def test_index_put_two_targets_sum_e2e(self):
        """z[j] = src_z; y[i] = src_y; return z + y -- two independent
        indirect-output scatters into differently-shaped paged caches
        combined by a direct pointwise op in the same compiled graph."""
        y, src_y, idx_y = self._paged_kv_cache_operands(L=576, H=8, D=128)
        z, src_z, idx_z = self._paged_kv_cache_operands(L=576, H=8, D=128)

        def kernel(y, src_y, idx_y, z, src_z, idx_z):
            y[idx_y] = src_y
            z[idx_z] = src_z
            return z + y

        self._assert_compiled_matches_cpu(kernel, y, src_y, idx_y, z, src_z, idx_z)

    def test_index_put_3d_scatter_dim1_e2e(self):
        """3-D tensor [B, H, D] scatter on dim 1 (H): y[:, idx, :] = src.
        Tests indirect access on a non-leading dimension, validating stride
        handling for multi-dim indexing."""
        B, H, D = 576, 8, 128
        y = torch.rand(B, H, D, dtype=torch.float16).to("spyre")
        P = 3
        src = torch.rand(B, P, D, dtype=torch.float16).to("spyre")
        idx = torch.randperm(H, dtype=torch.int64)[:P].to("spyre")

        def kernel(y, src, idx):
            y[:, idx, :] = src
            return y

        self._assert_compiled_matches_cpu(kernel, y, src, idx)

    def test_index_put_4d_scatter_dim1_e2e(self):
        """4-D tensor [B, L, H, D] scatter on dim 1 (L): y[:, idx, :, :] = src.
        Tests indirect access on a mid-rank dimension."""
        B, L, H, D = 2, 576, 8, 128
        y = torch.rand(B, L, H, D, dtype=torch.float16).to("spyre")
        P = 4
        src = torch.rand(B, P, H, D, dtype=torch.float16).to("spyre")
        idx = torch.randperm(L, dtype=torch.int64)[:P].to("spyre")

        def kernel(y, src, idx):
            y[:, idx, :, :] = src
            return y

        self._assert_compiled_matches_cpu(kernel, y, src, idx)

    # ===== index_add: gather + add + overwrite-scatter decomposition =====
    # aten.index_add lowers (spyre_index_add in decompositions.py) to
    # index_select (indirect gather) + add + index_put (indirect overwrite
    # store). The final store is an indirect OUTPUT, so each scenario still
    # classifies as SCATTER_OP_SPEC (_label_for checks indirect-output first).

    def _index_add_operands(
        self,
        M=128,
        N=256,
        P=3,
        *,
        data_dtype=torch.float16,
        idx_dtype=torch.int32,
        nonzero_dest=False,
        idx=None,
    ):
        """Operands for index_add scenarios: out[M,N], src[P,N], 1-D idx[P].

        Parameters
        ----------
        M, N, P:
            Destination rows, feature width, and number of updated rows.
        data_dtype:
            Element dtype for out and src (float16 or float32).
        idx_dtype:
            Dtype for the index tensor (int32 or int64).
        nonzero_dest:
            When True, out is filled with random values instead of zeros.
            Use this to pin that the gather step reads the existing dest
            values (gathered + src, not just src).
        idx:
            Explicit CPU index tensor of length P.  When None the helper
            generates torch.arange(P), producing the consecutive unique
            indices used by the majority of scenarios.  Pass an explicit
            tensor for boundary rows, strided, or random-permutation indices.
        """
        out_data = (
            torch.rand(M, N, dtype=data_dtype)
            if nonzero_dest
            else torch.zeros(M, N, dtype=data_dtype)
        )
        out = out_data.to("spyre")
        src = torch.rand(P, N, dtype=data_dtype).to("spyre")
        if idx is None:
            idx = torch.arange(P, dtype=idx_dtype).to("spyre")
        else:
            idx = idx.to(idx_dtype).to("spyre")
        self.name_dims(out, {"M": M, "N": N})
        self.name_dims(src, {"P": P, "N": N})
        self.name_dims(idx, {"P": P})
        return out, src, idx

    # NOTE: index_add lowers to gather (index_select) + add + overwrite-scatter
    # (index_put), so the bundle is BOTH a gather and a scatter. It still
    # classifies as SCATTER_OP_SPEC (terminal indirect-output store), but
    # assert_indirect_sdsc_fields' scatter-only invariant ("every indirect value
    # tensor is the output") does not hold -- the gather's value tensor is an
    # input. Hence sdsc=False on these (classification + e2e still run).

    def test_index_add(self):
        """out.index_add_(0, idx, src) -- canonical row scatter-add."""
        out, src, idx = self._index_add_operands()

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC, sdsc=False)

    def test_index_add_narrow_feature(self):
        """Small feature dim (N=4), far below one stick."""
        out, src, idx = self._index_add_operands(N=4)

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC, sdsc=False)

    def test_index_add_single_index(self):
        """P == 1: a single-row update (decode-step shape)."""
        out, src, idx = self._index_add_operands(P=1)

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC, sdsc=False)

    def test_index_add_many_updates(self):
        """More update rows (P=32) into a 128-row destination."""
        out, src, idx = self._index_add_operands(M=128, P=32)

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC, sdsc=False)

    def test_index_add_int64_index(self):
        """int64 index (downcast to int32 on device)."""
        out, src, idx = self._index_add_operands(idx_dtype=torch.int64)

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC, sdsc=False)

    def test_index_add_alpha(self):
        """torch.index_add with a non-unit alpha scaling factor (0.5, 1.0, 2.0)."""
        for alpha in (0.5, 1.0, 2.0):
            out, src, idx = self._index_add_operands(
                M=8,
                N=4,
                P=3,
                data_dtype=torch.float32,
                idx_dtype=torch.int64,
                idx=torch.tensor([1, 3, 6]),
            )

            def kernel(out, src, idx, _alpha=alpha):
                return torch.index_add(out, 0, idx, src, alpha=_alpha)

            self._assert_compiled_matches_cpu(kernel, out, src, idx)

    def test_index_add_3d(self):
        """3-D destination [M, A, B], scatter-add along dim 0."""
        M, A, B, P = 128, 8, 64, 3
        out = torch.zeros(M, A, B, dtype=torch.float16).to("spyre")
        src = torch.rand(P, A, B, dtype=torch.float16).to("spyre")
        idx = torch.arange(P, dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": M, "A": A, "B": B})
        self.name_dims(src, {"P": P, "A": A, "B": B})
        self.name_dims(idx, {"P": P})

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC, sdsc=False)

    def test_index_add_moe_shape(self):
        """MoE shape [64,2816], P=3."""
        out, src, idx = self._index_add_operands(
            M=64,
            N=2816,
            P=3,
            nonzero_dest=True,
        )

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_tiny_zeros(self):
        """Sanity: [8,4] dest (zeros), 3 unique indices, functional API."""
        out, src, idx = self._index_add_operands(
            M=8,
            N=4,
            P=3,
            idx=torch.tensor([0, 2, 5], dtype=torch.int32),
        )

        def kernel(out, src, idx):
            return torch.index_add(out, 0, idx, src)

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_nonzero_dest(self):
        """Gather correctness: [8,4] non-zero dest -- gathered values must be
        added to src, not src written directly."""
        out, src, idx = self._index_add_operands(
            M=8,
            N=4,
            P=3,
            nonzero_dest=True,
            idx=torch.tensor([1, 4, 7], dtype=torch.int32),
        )

        def kernel(out, src, idx):
            return torch.index_add(out, 0, idx, src)

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_moe_inplace(self):
        """MoE full-coverage inplace: [64,2816] dest, all 64 rows updated via
        index_add_().  The original user-reported failing shape."""
        out, src, idx = self._index_add_operands(M=64, N=2816, P=64)

        def kernel(out, src, idx):
            out = out.clone()
            out.index_add_(0, idx, src)
            return out

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_moe_functional(self):
        """MoE full-coverage functional: same [64,2816] shape via
        torch.index_add (out-of-place)."""
        out, src, idx = self._index_add_operands(M=64, N=2816, P=64)

        def kernel(out, src, idx):
            return torch.index_add(out, 0, idx, src)

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_partial_update(self):
        """Partial update: [128,256] dest, only 3 rows (first, middle, last).
        Non-indexed rows must be unchanged."""
        out, src, idx = self._index_add_operands(
            M=128,
            N=256,
            P=3,
            nonzero_dest=True,
            idx=torch.tensor([0, 63, 127], dtype=torch.int32),
        )

        def kernel(out, src, idx):
            return torch.index_add(out, 0, idx, src)

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_dense_update(self):
        """Dense update: [64,256] dest, every other row updated (32 of 64)."""
        out, src, idx = self._index_add_operands(
            M=64,
            N=256,
            P=32,
            nonzero_dest=True,
            idx=torch.arange(0, 64, 2, dtype=torch.int32),  # 0,2,...,62
        )

        def kernel(out, src, idx):
            return torch.index_add(out, 0, idx, src)

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_large_p(self):
        """Large P: [32,512] dest, 24 of 32 rows updated (randomly chosen)."""
        _idx = (
            torch.randperm(32, generator=torch.Generator().manual_seed(0))[:24]
            .sort()
            .values
        )
        out, src, idx = self._index_add_operands(
            M=32,
            N=512,
            P=24,
            nonzero_dest=True,
            idx=_idx,
        )

        def kernel(out, src, idx):
            return torch.index_add(out, 0, idx, src)

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_ragged_n(self):
        """Non-aligned N: [64,63] dest -- N is not a multiple of stick size (64)."""
        out, src, idx = self._index_add_operands(M=64, N=63, P=8)

        def kernel(out, src, idx):
            return torch.index_add(out, 0, idx, src)

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_fp32(self):
        """fp32 dtype: [16,128] dest, 5 unique indices."""
        out, src, idx = self._index_add_operands(
            M=16,
            N=128,
            P=5,
            data_dtype=torch.float32,
            idx=torch.tensor([1, 3, 7, 10, 14], dtype=torch.int32),
        )

        def kernel(out, src, idx):
            return torch.index_add(out, 0, idx, src)

        self._stage_and_e2e(
            kernel,
            out,
            src,
            idx,
            expect=SCATTER_OP_SPEC,
            sdsc=False,
            expect_close=True,
        )

    def test_index_add_dim1_unsupported(self):
        """index_add along dim=1 (column scatter-add) is unsupported. The gather
        (index_select) leg puts the index-dependent coordinate on a non-outermost
        device dim, and propagate_layouts finds no supported output layout
        (indirect access requires the indexed dim outermost). The compile aborts,
        surfaced here as CRASHED."""
        M, N, P = 4, 8, 3
        out = torch.zeros(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(M, P, dtype=torch.float16).to("spyre")
        idx = torch.arange(P, dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": M, "N": N})
        self.name_dims(src, {"M": M, "P": P})
        self.name_dims(idx, {"P": P})

        def kernel(out, src, idx):
            return out.index_add_(1, idx, src)

        self.check(kernel, out, src, idx, expect=CRASHED)

    def test_scatter_reduce(self):
        """out.scatter_reduce_(0, index, src, "sum")"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return out.scatter_reduce_(0, index, src, "sum")

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    def test_index_put_accumulate(self):
        """out.index_put_((idx,), src, accumulate=True) -- out[idx] += src."""
        out, src, idx = self._row_store()

        def kernel(out, src, idx):
            return out.index_put_((idx,), src, accumulate=True)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

    def test_scatter_add_functional(self):
        """torch.scatter_add(out, 0, index, src) -- functional accumulating scatter."""
        out, src, index = self._full_index_store()

        def kernel(out, src, index):
            return torch.scatter_add(out, 0, index, src)

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    # ------------- Not Detected As Indirect Access Scatter -------------
    def test_scatter_reduce_amax(self):
        """out.scatter_reduce_(0, index, src, "amax")"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return out.scatter_reduce_(0, index, src, "amax")

        self._stage_and_e2e(kernel, out, src, index, expect=DIRECT_OP_SPEC)

    def test_scatter_reduce_amin(self):
        """out.scatter_reduce_(0, index, src, "amin")"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return out.scatter_reduce_(0, index, src, "amin")

        self._stage_and_e2e(kernel, out, src, index, expect=DIRECT_OP_SPEC)

    def test_scatter_reduce_prod(self):
        """out.scatter_reduce_(0, index, src, "prod")"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return out.scatter_reduce_(0, index, src, "prod")

        self._stage_and_e2e(kernel, out, src, index, expect=DIRECT_OP_SPEC)

    # -- Known crashes (separate from the indirect-store path) -------------
    def test_index_fill_crashes(self):
        """out.index_fill_(0, idx, 0.0) -- scalar fill -> rank-0 Constant codegen."""
        out = torch.rand(128, 256, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 128, (3,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 128, "N": 256})
        self.name_dims(idx, {"P": 3})

        def kernel(out, idx):
            return out.index_fill_(0, idx, 0.0)

        self.check(kernel, out, idx, expect=CRASHED)

    def test_masked_scatter_element_mask_unsupported(self):
        """Element-level mask (stride(-1) != 0): the decomposition rejects it.

        A full [M, N] per-element mask is not broadcast along the last dim, so
        spyre_masked_scatter raises Unsupported -- an element scatter would index
        into a packed 1-D source, and a lane within a stick is not addressable on
        Spyre. The raise propagates out of torch.compile before any op spec is
        produced (CRASHED in this harness); there is no bundle to run e2e.
        """
        M, N = 64, 64
        out = torch.zeros(M, N, dtype=torch.float16).to("spyre")
        mask = torch.randint(0, 2, (M, N), dtype=torch.bool).to("spyre")
        src = torch.rand(M, N, dtype=torch.float16).to("spyre")
        self.name_dims(out, {"M": M, "N": N})

        def kernel(out, mask, src):
            return torch.masked_scatter(out, mask, src)

        self.check(kernel, out, mask, src, expect=CRASHED)

    # -- Supported masked_scatter: row-broadcast mask lowers to a gather -------
    def test_masked_scatter_row_broadcast(self):
        """torch.masked_scatter with a mask broadcast along the last dim.

        A row-broadcast mask (stride(-1) == 0 -- e.g. an attention mask [B, S]
        expanded to [B, S, C]) selects whole rows, so spyre_masked_scatter lowers
        it to a stick gather source_2d[row_idx] plus a where. That is an indirect
        *read*, hence a GATHER_OP_SPEC (not an indirect-output scatter).
        """
        ROWS, COLS, SRC_ROWS, N_TRUE = 855, 5120, 266, 266
        inp = torch.rand(1, ROWS, COLS, dtype=torch.float16).to("spyre")
        src = torch.rand(SRC_ROWS, COLS, dtype=torch.float16).to("spyre")
        # Row-broadcast mask: [1, ROWS] expanded to [1, ROWS, COLS] (stride(-1)==0).
        mask_1d = torch.zeros(1, ROWS, dtype=torch.bool)
        mask_1d[0, torch.randperm(ROWS)[:N_TRUE]] = True
        mask = mask_1d.unsqueeze(-1).to("spyre").expand(1, ROWS, COLS)
        self.name_dims(inp, {"B": 1, "ROWS": ROWS, "COLS": COLS})
        self.name_dims(src, {"SRC_ROWS": SRC_ROWS, "COLS": COLS})

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self._stage_and_e2e(
            kernel, inp, mask, src, expect=GATHER_OP_SPEC, expect_close=True
        )

    def test_masked_scatter_unexpanded_row_broadcast(self):
        """torch.masked_scatter with a row mask left in its UN-EXPANDED form:
        a literal size-1 last dim [B, S, 1] (stride(-1) == 1), not broadcast up
        to [B, S, C]. This is what real models hand us (e.g. Mistral-Small-3.2's
        `inputs_embeds.masked_scatter(special_image_mask, image_features)` with
        mask [1, 855, 1] into self [1, 855, 5120]).

        It is the same whole-row selection as the expanded form -- mask[..., 0]
        collapses either spelling to one bool per row -- so it must also lower to
        a gather.
        """
        ROWS, COLS, SRC_ROWS, N_TRUE = 855, 5120, 266, 266
        inp = torch.rand(1, ROWS, COLS, dtype=torch.float16).to("spyre")
        src = torch.rand(SRC_ROWS, COLS, dtype=torch.float16).to("spyre")
        # Un-expanded: [1, ROWS, 1], NOT .expand()-ed to [1, ROWS, COLS].
        mask_1d = torch.zeros(1, ROWS, dtype=torch.bool)
        mask_1d[0, torch.randperm(ROWS)[:N_TRUE]] = True
        mask = mask_1d.unsqueeze(-1).to("spyre")  # shape [1, ROWS, 1]
        self.assertEqual(tuple(mask.shape), (1, ROWS, 1))
        self.name_dims(inp, {"B": 1, "ROWS": ROWS, "COLS": COLS})
        self.name_dims(src, {"SRC_ROWS": SRC_ROWS, "COLS": COLS})

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self._stage_and_e2e(
            kernel, inp, mask, src, expect=GATHER_OP_SPEC, expect_close=True
        )

    def _row_broadcast_operands(self, shape, src_rows, n_true):
        """Supported masked_scatter operands: `self` of `shape`, a mask broadcast
        along the last dim (stride(-1) == 0) with `n_true` selected rows, and a
        2-D `src[src_rows, shape[-1]]`. Returns (inp, mask, src) on "spyre"."""
        *lead, cols = shape
        rows = 1
        for d in lead:
            rows *= d
        inp = torch.rand(*shape, dtype=torch.float16).to("spyre")
        src = torch.rand(src_rows, cols, dtype=torch.float16).to("spyre")
        flat = torch.zeros(rows, dtype=torch.bool)
        flat[torch.randperm(rows)[:n_true]] = True
        mask = flat.reshape(*lead, 1).to("spyre").expand(*shape)
        # Share the column dim name "C" across self and src (house convention:
        # the two operands' common last dim is one named dim, cf. _row_store).
        lead_names = [f"L{i}" for i in range(len(lead))]
        self.name_dims(inp, dict(zip(lead_names + ["C"], list(lead) + [cols])))
        self.name_dims(src, {"SRC": src_rows, "C": cols})
        return inp, mask, src

    def test_masked_scatter_2d_row_broadcast(self):
        """2-D self [M, N] with a row-broadcast mask -- the smallest supported
        shape. Capture-only: assert it reaches a gather op spec (and a valid
        indirect-access SDSC bundle), no device run needed."""
        inp, mask, src = self._row_broadcast_operands(
            (128, 256), src_rows=40, n_true=40
        )

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self.check(kernel, inp, mask, src, expect=GATHER_OP_SPEC)

    def test_masked_scatter_batched_row_broadcast(self):
        """Batched self [B, S, C] (rows = B*S) with a row-broadcast mask. Pins
        that a leading batch dim still collapses to whole-row selection and
        reaches a gather op spec."""
        inp, mask, src = self._row_broadcast_operands(
            (2, 64, 128), src_rows=40, n_true=40
        )

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self.check(kernel, inp, mask, src, expect=GATHER_OP_SPEC)

    def test_masked_scatter_all_rows_selected(self):
        """All-True mask: every row is selected, so src must supply exactly `rows`
        rows and the result is `src` row-for-row. Exercises the max-selection
        boundary end-to-end."""
        M, N = 8, 128
        inp = torch.rand(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(M, N, dtype=torch.float16).to("spyre")
        mask = torch.ones(M, 1, dtype=torch.bool).to("spyre").expand(M, N)
        self.name_dims(inp, {"M": M, "N": N})
        self.name_dims(src, {"M": M, "N": N})

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self._stage_and_e2e(
            kernel, inp, mask, src, expect=GATHER_OP_SPEC, expect_close=True
        )

    def test_masked_scatter_no_rows_selected(self):
        """All-False mask: nothing is selected, so the result must equal `self`
        unchanged. The gather still lowers (the mask is a runtime input, not a
        compile-time constant), so this stays a GATHER_OP_SPEC."""
        M, N = 8, 128
        inp = torch.rand(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(4, N, dtype=torch.float16).to("spyre")
        mask = torch.zeros(M, 1, dtype=torch.bool).to("spyre").expand(M, N)
        self.name_dims(inp, {"M": M, "N": N})
        self.name_dims(src, {"SRC": 4, "N": N})

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self._stage_and_e2e(
            kernel, inp, mask, src, expect=GATHER_OP_SPEC, expect_close=True
        )

    def test_masked_scatter_degenerate_last_dim_unsupported(self):
        """Degenerate last dim (cols == 1): a single-column row is not a real
        shared row for the block-per-row equivalence, so the decomposition
        rejects it with Unsupported (CRASHED here)."""
        M = 128
        inp = torch.rand(M, 1, dtype=torch.float16).to("spyre")
        src = torch.rand(4, 1, dtype=torch.float16).to("spyre")
        mask = torch.zeros(M, 1, dtype=torch.bool)
        mask[torch.randperm(M)[:4]] = True
        mask = mask.to("spyre")
        self.name_dims(inp, {"M": M, "ONE": 1})

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self.check(kernel, inp, mask, src, expect=CRASHED)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()

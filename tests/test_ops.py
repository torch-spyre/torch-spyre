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

import pytest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOps(TestCase):
    def __init__(self, method_name="runTest", methodName="runTest"):
        super().__init__(method_name, methodName)

        # NOTE: needs to be at most 1e-3
        self.rtol = 1e-1
        self.atol = 1e-1
        self.dtype = torch.float16

        # TODO: The tensor size was changed (from 3, 5, 7 respectively) to avoid padding in the stick dimension.
        #   Once we have proper padding to stack handled, these values should be changed back
        self.mm_a = 67
        self.mm_b = 256
        self.mm_c = 128
        torch.random.manual_seed(42)

    def test_inplace_fill_scalar(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype, device="spyre")
        x.fill_(5.0)
        x_actual = x.cpu()
        x_expected = torch.tensor([5.0, 5.0, 5.0], dtype=self.dtype)
        torch.testing.assert_close(x_expected, x_actual, rtol=self.rtol, atol=self.atol)

    def test_copy_1d_padded_to_stick(self):
        x = torch.tensor([1, 2, 3], dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_2d_padded_to_stick(self):
        x = torch.tensor([[1, -2, 3], [4, 5, 6]], dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_3d_padded_to_stick(self):
        x = torch.tensor(
            [[[1, -2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=self.dtype,
        )
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_4d_padded_to_stick(self):
        x = torch.rand(2, 2, 2, 3, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_6d_padded_to_stick(self):
        x = torch.rand(1, 3, 5, 2, 4, 62, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_5d_padded_to_stick(self):
        x = torch.rand(1, 2, 3, 4, 5, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_4d_padded(self):
        x = torch.rand(2, 2, 2, 120, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_3d_padded(self):
        x = torch.rand(2, 2, 72, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_2d_padded(self):
        x = torch.rand(2, 205, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_1d_padded(self):
        x = torch.rand(511, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_1d(self):
        x = torch.rand(256, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_2d(self):
        x = torch.rand(256, 128, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_3d(self):
        x = torch.rand(256, 128, 512, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_4d(self):
        x = torch.rand(2, 6, 3, 128, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_5d(self):
        x = torch.rand(4, 8, 3, 64, 256, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_6d(self):
        x = torch.rand(4, 8, 16, 12, 64, 128, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_mish(self):
        x = torch.rand([2, 100, 12800], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.nn.functional.mish(x_spyre).to("cpu")
        torch.testing.assert_close(
            y, torch.nn.functional.mish(x), rtol=self.rtol, atol=self.atol
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_addmm_ab_bc_scaled(self):
        mat = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_a, self.mm_c
        )
        x = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_a, self.mm_b
        )
        y = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_b, self.mm_c
        )
        alpha = 0.5
        mat_spyre = mat.to("spyre")
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.addmm(mat_spyre, x_spyre, y_spyre, alpha=alpha).to("cpu")
        torch.testing.assert_close(
            z, torch.addmm(mat, x, y, alpha=alpha), rtol=self.rtol, atol=self.atol
        )

    def test_addmm_ab_bc_out(self):
        mat = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_a, self.mm_c
        )
        x = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_a, self.mm_b
        )
        y = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_b, self.mm_c
        )
        mat_spyre = mat.to("spyre")
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        out_spyre = torch.empty(self.mm_a, self.mm_c, dtype=self.dtype, device="spyre")
        torch.addmm(mat_spyre, x_spyre, y_spyre, out=out_spyre)
        torch.testing.assert_close(
            out_spyre.to("cpu"), torch.addmm(mat, x, y), rtol=self.rtol, atol=self.atol
        )

    def test_mean(self):
        x = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
            dtype=self.dtype,
        )
        x_spyre = x.to("spyre")
        y0 = torch.mean(x_spyre, dim=[0]).to("cpu")
        y1 = torch.mean(x_spyre, dim=[1]).to("cpu")
        y0_keepdim = torch.mean(x_spyre, dim=[0], keepdim=True).to("cpu")
        torch.testing.assert_close(
            y0, torch.mean(x, dim=[0]), rtol=self.rtol, atol=self.atol
        )
        torch.testing.assert_close(
            y1, torch.mean(x, dim=[1]), rtol=self.rtol, atol=self.atol
        )
        torch.testing.assert_close(
            y0_keepdim,
            torch.mean(x, dim=[0], keepdim=True),
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_normal_randn(self):
        gen = torch.manual_seed(42)

        y_spyre = torch.randn(3, 5, device="spyre", generator=gen)

        # torch.Generator is stateful, hence reset
        gen.manual_seed(42)

        y_cpu = torch.randn(3, 5, device="cpu", generator=gen)

        torch.testing.assert_close(
            y_spyre.to("cpu"), y_cpu, rtol=self.rtol, atol=self.atol
        )

    def test_zeros(self):
        x_spyre = torch.zeros(3, 64, device="spyre", dtype=self.dtype)
        x = torch.zeros(3, 64, dtype=self.dtype)
        torch.testing.assert_close(x_spyre.to("cpu"), x, rtol=self.rtol, atol=self.atol)

    def test_zeros_padded_last_dim(self):
        # Test zeros with last dimension requiring padding (not 64)
        x_spyre = torch.zeros(3, 50, device="spyre", dtype=self.dtype)
        x = torch.zeros(3, 50, dtype=self.dtype)
        torch.testing.assert_close(x_spyre.to("cpu"), x, rtol=self.rtol, atol=self.atol)

    def test_uniform_(self):
        x_spyre = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=self.dtype, device="spyre")
        x_spyre.uniform_()
        x_cpu = x_spyre.to("cpu")
        self.assertTrue(
            torch.all(x_cpu >= 0.0) and torch.all(x_cpu < 1.0),
            f"uniform_ values out of range [0, 1): {x_cpu}",
        )
        self.assertFalse(
            torch.all(x_cpu == x_cpu[0, 0]), "uniform_ produced all identical values"
        )

    def test_uniform_custom_range(self):
        x_spyre = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0], dtype=self.dtype, device="spyre"
        )
        x_spyre.uniform_(-5.0, 5.0)
        x_cpu = x_spyre.to("cpu")
        self.assertTrue(
            torch.all(x_cpu >= -5.0) and torch.all(x_cpu < 5.0),
            f"uniform_ values out of range [-5, 5): {x_cpu}",
        )
        self.assertFalse(
            torch.all(x_cpu == x_cpu[0]), "uniform_ produced all identical values"
        )

    # NOTE: embedding / indirect indexing / index_select are not supported yet
    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_embedding(self):
        # an embedding matrix containing 10 tensors of size 3
        embedding_matrix = torch.rand(10, 3, dtype=torch.float16)
        # a batch of 2 samples of 4 indices each
        indices = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.int64)
        cpu_y = torch.nn.functional.embedding(indices, embedding_matrix)

        embed_spyre = embedding_matrix.to("spyre")
        indices_spyre = indices.to("spyre")
        spyre_y = torch.nn.functional.embedding(indices_spyre, embed_spyre).to("cpu")

        torch.testing.assert_close(cpu_y, spyre_y, rtol=self.rtol, atol=self.atol)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_tensor_tensor(self):
        """Test aten.isin.Tensor_Tensor: both inputs are tensors."""
        elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        test_elements = torch.tensor([2, 4], dtype=torch.int64)
        expected = torch.isin(elements, test_elements)

        elements_spyre = elements.to("spyre")
        test_elements_spyre = test_elements.to("spyre")
        actual = torch.isin(elements_spyre, test_elements_spyre).cpu()

        torch.testing.assert_close(actual, expected)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_tensor_tensor_out(self):
        """Test aten.isin.Tensor_Tensor_out: out-variant."""
        elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        test_elements = torch.tensor([2, 4], dtype=torch.int64)
        out_cpu = torch.empty(elements.shape, dtype=torch.bool)
        torch.isin(elements, test_elements, out=out_cpu)

        elements_spyre = elements.to("spyre")
        test_elements_spyre = test_elements.to("spyre")
        out_spyre = torch.empty(elements.shape, dtype=torch.bool, device="spyre")
        torch.isin(elements_spyre, test_elements_spyre, out=out_spyre)

        torch.testing.assert_close(out_spyre.cpu(), out_cpu)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_tensor_scalar(self):
        """Test aten.isin.Tensor_Scalar: test_elements is a scalar."""
        elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        test_elements = 3
        expected = torch.isin(elements, test_elements)

        elements_spyre = elements.to("spyre")
        actual = torch.isin(elements_spyre, test_elements).cpu()

        torch.testing.assert_close(actual, expected)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_tensor_scalar_out(self):
        """Test aten.isin.Tensor_Scalar_out: test_elements is a scalar, out-variant."""
        elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        test_elements = 3
        out_cpu = torch.empty(elements.shape, dtype=torch.bool)
        torch.isin(elements, test_elements, out=out_cpu)

        elements_spyre = elements.to("spyre")
        out_spyre = torch.empty(elements.shape, dtype=torch.bool, device="spyre")
        torch.isin(elements_spyre, test_elements, out=out_spyre)

        torch.testing.assert_close(out_spyre.cpu(), out_cpu)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_scalar_tensor(self):
        """Test aten.isin.Scalar_Tensor: elements is a scalar."""
        elements = 3
        test_elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        expected = torch.isin(elements, test_elements)

        test_elements_spyre = test_elements.to("spyre")
        actual = torch.isin(elements, test_elements_spyre).cpu()

        # Compare boolean values (scalar tensor shape may differ due to Spyre backend)
        self.assertEqual(actual.item(), expected.item())

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_scalar_tensor_out(self):
        """Test aten.isin.Scalar_Tensor_out: elements is a scalar, out-variant."""
        elements = 3
        test_elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        out_cpu = torch.empty(0, dtype=torch.bool)
        torch.isin(elements, test_elements, out=out_cpu)

        test_elements_spyre = test_elements.to("spyre")
        out_spyre = torch.empty((), dtype=torch.bool, device="spyre")
        torch.isin(elements, test_elements_spyre, out=out_spyre)

        # Compare boolean values (scalar tensor shape may differ due to Spyre backend)
        self.assertEqual(out_spyre.cpu().item(), out_cpu.item())

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_embedding_with_padding_idx(self):
        # an embedding matrix containing 10 tensors of size 3
        embedding_matrix = torch.rand(10, 3, dtype=torch.float16)
        # a batch of 2 samples of 4 indices each
        indices = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.int64)
        cpu_y = torch.nn.functional.embedding(indices, embedding_matrix, padding_idx=0)

        embed_spyre = embedding_matrix.to("spyre")
        indices_spyre = indices.to("spyre")
        spyre_y = torch.nn.functional.embedding(
            indices_spyre, embed_spyre, padding_idx=0
        ).to("cpu")

        torch.testing.assert_close(cpu_y, spyre_y, rtol=self.rtol, atol=self.atol)


if __name__ == "__main__":
    run_tests()

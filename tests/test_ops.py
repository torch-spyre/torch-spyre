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

import pathlib
import yaml
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_methods_invocations import op_db


class TestOps(TestCase):
    def __init__(self, method_name="runTest", methodName="runTest"):
        super().__init__(method_name, methodName)
        declarations_path = (
            pathlib.Path(__file__).parent.parent
            / "codegen"
            / "outputs"
            / "GeneratedDeclarations.yaml"
        )
        declarations_path.resolve()

        with declarations_path.open() as f:
            self.declarations = yaml.safe_load(f)

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

    @unittest.skip("View tensors do not have SpyreTensorImpl")
    def test_t_1d(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = x_spyre.t().to("cpu")
        torch.testing.assert_close(y, x.t(), rtol=self.rtol, atol=self.atol)

    def test_t_2d(self):
        x = torch.tensor([[1, -2, 3], [4, 5, 6]], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = x_spyre.t().to("cpu")
        torch.testing.assert_close(y, x.t(), rtol=self.rtol, atol=self.atol)

    def test_transpose_2d(self):
        x = torch.tensor([[1, -2, 3], [4, 5, 6]], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = x_spyre.transpose(0, 1).to("cpu")
        torch.testing.assert_close(y, x.transpose(0, 1), rtol=self.rtol, atol=self.atol)

    def test_transpose_3d(self):
        x = torch.tensor(
            [[[1, -2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=self.dtype,
        )
        x_spyre = x.to("spyre")
        y = x_spyre.transpose(0, 1).to("cpu")
        torch.testing.assert_close(y, x.transpose(0, 1), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Implement permute properly on eager")
    def test_permute_2d(self):
        x = torch.tensor([[1, -2, 3], [4, 5, 6]], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = x_spyre.permute(1, 0).to("cpu")
        torch.testing.assert_close(y, x.permute(1, 0), rtol=self.rtol, atol=self.atol)

    def test_bool(self):
        dtype = torch.bool
        x = torch.randint(0, 2, (2, 64), dtype=dtype)
        x_spyre = x.to("spyre")
        y = torch.randint(0, 2, (2, 64), dtype=dtype)
        y_spyre = y.to("spyre")
        result = torch.eq(x_spyre, y_spyre).cpu()
        torch.testing.assert_close(result, torch.eq(x, y))

    def test_eq(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        y = torch.tensor([0, -2, 4], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        # FIXME: equal is currently returning back the same dtype as the original tensor, we need to have this return a bool
        actual = (x_spyre == y_spyre).cpu().bool()
        torch.testing.assert_close(actual, x == y, rtol=self.rtol, atol=self.atol)

    def test_ge(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        y = torch.tensor([0, -2, 4], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        # FIXME: equal is currently returning back the same dtype as the original tensor, we need to have this return a bool
        actual = (x_spyre >= y_spyre).cpu().bool()
        torch.testing.assert_close(actual, x >= y, rtol=self.rtol, atol=self.atol)

    def test_abs(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.abs(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.abs(x), rtol=self.rtol, atol=self.atol)

    def test_relu(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.relu(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.relu(x), rtol=self.rtol, atol=self.atol)

    def test_exp(self):
        x = torch.tensor([-10, -1, 0, 1, 10], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.exp(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.exp(x), rtol=self.rtol, atol=self.atol)

    def test_exp_transpose(self):
        x = torch.tensor([[-10, -1, 0, 1, 10], [1, 2, 3, 4, 5]], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.exp(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.exp(x), rtol=self.rtol, atol=self.atol)

    def test_log(self):
        x = torch.tensor([0.1, 1, 10, 100], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.log(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.log(x), rtol=self.rtol, atol=self.atol)

    def test_reciprocal(self):
        x = torch.tensor([-2, 1, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.reciprocal(x_spyre).to("cpu")
        torch.testing.assert_close(
            y, torch.reciprocal(x), rtol=self.rtol, atol=self.atol
        )

    def test_sigmoid(self):
        x = torch.tensor([-2, 1, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.sigmoid(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.sigmoid(x), rtol=self.rtol, atol=self.atol)

    def test_sqrt(self):
        x = torch.tensor([0, 1, 2.25, 4, 10000], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.sqrt(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.sqrt(x), rtol=self.rtol, atol=self.atol)

    def test_tanh(self):
        x = torch.tensor([-2, -1, 0, 1, 2], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.tanh(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.tanh(x), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Needs more debug")
    def test_clone(self):
        x = torch.tensor([-2, -1, 0, 1, 2], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.clone(x_spyre).to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_add_Tensor(self):
        x = torch.tensor([1, 2, 3], dtype=self.dtype)
        y = torch.tensor([4, 5, 6], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.add(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.add(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.expectedFailure
    def test_add_Scalar(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=self.dtype)
        y = 5
        x_spyre = x.to("spyre")
        z = (x_spyre + y).to("cpu")
        torch.testing.assert_close(z, x + y, rtol=self.rtol, atol=self.atol)

    @unittest.skip("xfail: Swapping stick dimension is unsupported in new DCI")
    def test_add_Tensor_transpose(self):
        x = torch.arange(8, dtype=self.dtype).view(2, 4)
        y = torch.arange(8, dtype=self.dtype).view(4, 2) * 10
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z_01 = torch.add(x_spyre, y_spyre.t().contiguous()).to("cpu")
        z_10 = torch.add(x_spyre.t().contiguous(), y_spyre).to("cpu")
        torch.testing.assert_close(
            z_01, torch.add(x, y.t()), rtol=self.rtol, atol=self.atol
        )
        torch.testing.assert_close(
            z_10, torch.add(x.t(), y), rtol=self.rtol, atol=self.atol
        )

    def test_sub(self):
        x = torch.tensor([10, 20, 3], dtype=self.dtype)
        y = torch.tensor([4, 5, 6], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.subtract(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(
            z, torch.subtract(x, y), rtol=self.rtol, atol=self.atol
        )

    def test_mul(self):
        x = torch.tensor([1, 0, -3], dtype=self.dtype)
        y = torch.tensor([4, 5, 6], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mul(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mul(x, y), rtol=self.rtol, atol=self.atol)

    def test_mm_ab_bc(self):
        x = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_a, self.mm_b
        )
        y = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_b, self.mm_c
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    def test_mm_ac_cb(self):
        x = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_a, self.mm_c
        )
        y = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_c, self.mm_b
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Debug accuracy error")
    def test_mm_ba_ac(self):
        x = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_b, self.mm_a
        )
        y = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_a, self.mm_c
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    def test_mm_bc_ca(self):
        x = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_b, self.mm_c
        )
        y = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_c, self.mm_a
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Debug accuracy error")
    def test_mm_ca_ab(self):
        x = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_c, self.mm_a
        )
        y = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_a, self.mm_b
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    def test_mm_cb_ba(self):
        x = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_c, self.mm_b
        )
        y = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_b, self.mm_a
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: bmm.out not implemented yet in eager")
    def test_bmm_ab_bc(self):
        B = 1
        x = torch.randn(B * self.mm_a * self.mm_b, dtype=self.dtype).view(
            B, self.mm_a, self.mm_b
        )
        y = torch.randn(B * self.mm_b * self.mm_c, dtype=self.dtype).view(
            B, self.mm_b, self.mm_c
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.bmm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.bmm(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: bmm.out not implemented yet in eager")
    def test_bmm_cb_ba(self):
        B = 1
        x = torch.randn(B * self.mm_c * self.mm_b, dtype=self.dtype).view(
            B, self.mm_c, self.mm_b
        )
        y = torch.randn(B * self.mm_b * self.mm_a, dtype=self.dtype).view(
            B, self.mm_b, self.mm_a
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.bmm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.bmm(x, y), rtol=self.rtol, atol=self.atol)

    def test_matmul_ab_bc(self):
        B = 1
        x = torch.randn(B * self.mm_a * self.mm_b, dtype=self.dtype).view(
            B, self.mm_a, self.mm_b
        )
        y = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_b, self.mm_c
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.matmul(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(
            z, torch.matmul(x, y), rtol=self.rtol, atol=self.atol
        )

    def test_matmul_cb_ba(self):
        B = 1
        x = torch.randn(B * self.mm_c * self.mm_b, dtype=self.dtype).view(
            B, self.mm_c, self.mm_b
        )
        y = torch.randn(self.mm_b * self.mm_a, dtype=self.dtype).view(
            self.mm_b, self.mm_a
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.matmul(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(
            z, torch.matmul(x, y), rtol=self.rtol, atol=self.atol
        )

    @unittest.skip("TODO: mean.out not implemented in eager mode")
    def test_mean(self):
        x = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=self.dtype)
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

    def test_sum(self):
        x = torch.arange(0, 64, dtype=self.dtype).unsqueeze(0).repeat(3, 1)
        x_spyre = x.to("spyre")
        y0 = torch.sum(x_spyre, dim=[0]).to("cpu")
        torch.testing.assert_close(
            y0, torch.sum(x, dim=[0]), rtol=self.rtol, atol=self.atol
        )

    def test_softmax(self):
        x = torch.arange(0, 64, dtype=self.dtype).unsqueeze(0).repeat(3, 1)
        x_spyre = x.to("spyre")
        y1 = torch.softmax(x_spyre, dim=1).to("cpu")
        torch.testing.assert_close(
            y1, torch.softmax(x, dim=1), rtol=self.rtol, atol=self.atol
        )

    # --- View layout: identity ---

    def test_view_identity_2d(self):
        """[512, 256] -> [512, 256]: no change."""
        x = torch.rand(512, 256, dtype=self.dtype).to("spyre")
        stl_before = x.device_tensor_layout()
        y = x.view(512, 256)
        stl_after = y.device_tensor_layout()
        self.assertEqual(stl_after.device_size, stl_before.device_size)
        self.assertEqual(stl_after.dim_map, stl_before.dim_map)

    def test_view_identity_1d(self):
        """[128] -> [128]: no change."""
        x = torch.rand(128, dtype=self.dtype).to("spyre")
        stl_before = x.device_tensor_layout()
        y = x.view(128)
        stl_after = y.device_tensor_layout()
        self.assertEqual(stl_after.device_size, stl_before.device_size)
        self.assertEqual(stl_after.dim_map, stl_before.dim_map)

    # --- View layout: size-1 insertion (unsqueeze equivalent) ---

    def test_view_insert_size1_middle(self):
        """[512, 256] -> [512, 1, 256]: insert size-1 dim in the middle."""
        x = torch.rand(512, 256, dtype=self.dtype).to("spyre")
        y = x.view(512, 1, 256)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.dim_map, [2, 0, 2])

    def test_view_insert_size1_front(self):
        """[512, 256] -> [1, 512, 256]: insert size-1 dim at the front."""
        x = torch.rand(512, 256, dtype=self.dtype).to("spyre")
        y = x.view(1, 512, 256)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.dim_map, [2, 1, 2])

    def test_view_insert_size1_end(self):
        """[512, 256] -> [512, 256, 1]: insert size-1 dim at the end."""
        x = torch.rand(512, 256, dtype=self.dtype).to("spyre")
        y = x.view(512, 256, 1)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.dim_map, [1, 0, 1])

    def test_view_insert_multiple_size1(self):
        """[512, 256] -> [1, 512, 1, 256, 1]: multiple size-1 insertions."""
        x = torch.rand(512, 256, dtype=self.dtype).to("spyre")
        y = x.view(1, 512, 1, 256, 1)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.dim_map, [3, 1, 3])

    # --- View layout: size-1 removal (squeeze equivalent) ---

    def test_view_remove_size1_middle(self):
        """[512, 1, 256] -> [512, 256]: remove size-1 dim from middle."""
        x = torch.rand(512, 1, 256, dtype=self.dtype).to("spyre")
        y = x.view(512, 256)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.dim_map, [1, 0, 1])

    def test_view_remove_size1_front(self):
        """[1, 512, 256] -> [512, 256]: remove size-1 dim from front."""
        x = torch.rand(1, 512, 256, dtype=self.dtype).to("spyre")
        y = x.view(512, 256)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [4, 512, 64])
        self.assertEqual(stl.dim_map, [1, 0, 1])

    # --- View layout: merge (N:1) ---

    def test_view_merge_non_stick_2d_to_1d(self):
        """[2, 3, 4] -> [6, 4]: merge first two (non-stick) dims."""
        x = torch.rand(2, 3, 4, dtype=self.dtype).to("spyre")
        y = x.view(6, 4)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [3, 1, 2, 64])
        self.assertEqual(stl.dim_map, [0, 1, 0, 1])

    def test_view_merge_all_to_1d(self):
        """[4, 2, 64] -> [512]: merge all dims into one."""
        x = torch.rand(4, 2, 64, dtype=self.dtype).to("spyre")
        y = x.view(512)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.dim_map, [0, 0, 0, 0])

    # --- View layout: split (1:M) non-stick ---

    def test_view_split_non_stick(self):
        """[6, 4] -> [2, 3, 4]: split first (non-stick) dim."""
        x = torch.rand(6, 4, dtype=self.dtype).to("spyre")
        y = x.view(2, 3, 4)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [1, 2, 3, 64])
        self.assertEqual(stl.dim_map, [2, 0, 1, 2])

    # --- View layout: split (1:M) involving stick dim ---

    def test_view_split_stick_dim(self):
        """[512] -> [4, 128]: split where innermost new dim >= elems_per_stick."""
        x = torch.rand(512, dtype=self.dtype).to("spyre")
        y = x.view(4, 128)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [2, 4, 64])
        self.assertEqual(stl.dim_map, [1, 0, 1])

    def test_view_split_stick_dim_exact(self):
        """[256] -> [4, 64]: split where innermost == elems_per_stick."""
        x = torch.rand(256, dtype=self.dtype).to("spyre")
        y = x.view(4, 64)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [1, 4, 64])
        self.assertEqual(stl.dim_map, [1, 0, 1])

    def test_view_split_stick_dim_2d(self):
        """[3, 512] -> [3, 4, 128]: split stick dim from 2d tensor."""
        x = torch.rand(3, 512, dtype=self.dtype).to("spyre")
        y = x.view(3, 4, 128)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [4, 2, 3, 64])
        self.assertEqual(stl.dim_map, [1, 2, 0, 2])

    # --- View layout: mixed cases ---

    def test_view_merge_and_insert_size1(self):
        """[2, 3, 4] -> [1, 6, 4]: merge first two dims and insert size-1."""
        x = torch.rand(2, 3, 4, dtype=self.dtype).to("spyre")
        y = x.view(1, 6, 4)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [3, 1, 2, 64])
        self.assertEqual(stl.dim_map, [1, 2, 1, 2])

    def test_view_split_and_insert_size1(self):
        """[6, 4] -> [2, 3, 1, 4]: split non-stick dim and insert size-1."""
        x = torch.rand(6, 4, dtype=self.dtype).to("spyre")
        y = x.view(2, 3, 1, 4)
        stl = y.device_tensor_layout()
        self.assertEqual(stl.device_size, [1, 2, 3, 64])
        self.assertEqual(stl.dim_map, [3, 0, 1, 3])

    # --- View layout: rejection cases ---

    def test_view_reject_nm_complex(self):
        """[4, 6] -> [3, 8]: N:M group (2 old, 2 new) should fail."""
        x = torch.rand(4, 6, dtype=self.dtype).to("spyre")
        with self.assertRaisesRegex(RuntimeError, "N:M dimension groups"):
            x.view(3, 8)

    def test_view_reject_stick_split_too_small(self):
        """[512] -> [16, 32]: innermost new dim (32) < elems_per_stick (64)."""
        x = torch.rand(512, dtype=self.dtype).to("spyre")
        with self.assertRaisesRegex(RuntimeError, "elems_per_stick"):
            x.view(16, 32)

    @unittest.skip("TODO: Needs more debug")
    def test_all_ops(self):
        def test_op(declaration):
            op_handle = getattr(torch.ops.aten, declaration["operator_name"])
            if declaration["overload_name"]:
                try:
                    op_handle = getattr(op_handle, declaration["overload_name"])
                except AttributeError:
                    pass

            op_info = [op for op in op_db if op.name == declaration["name"]]

            close = True
            if op_info:
                op_info = op_info[0]
                sample_inputs = list(
                    op_info.sample_inputs(device="cpu", dtype=torch.float16)
                )
                for s in sample_inputs:
                    sample_input = [
                        s.input,
                        *s.args[: len(declaration["arguments"]) - 1],
                    ]
                    try:
                        outputs_cpu = op_handle(*sample_input)

                        sample_input_spyre = [
                            s_.to("spyre") if isinstance(s_, torch.Tensor) else s_
                            for s_ in sample_input
                        ]
                        outputs_spyre = op_handle(*sample_input_spyre)

                        for j in range(len(outputs_cpu)):
                            close_ = torch.allclose(
                                outputs_cpu[j],
                                outputs_spyre[j].to("cpu"),
                                rtol=self.rtol,
                                atol=self.atol,
                            )
                            if not close_:
                                close = False
                                print(
                                    f"spyre output is different for {declaration['operator_name']}"
                                )

                        # check if something happens to inputs as well
                        for j in range(len(sample_input)):
                            if isinstance(sample_input[j], torch.Tensor):
                                close_ = torch.allclose(
                                    sample_input[j],
                                    sample_input_spyre[j].to("cpu"),
                                    rtol=self.rtol,
                                    atol=self.atol,
                                )
                                if not close_:
                                    close = False
                                    print(
                                        f"spyre inputs changed after operation for {declaration['operator_name']}"
                                    )

                    except Exception:
                        print(f"Could not run test for {declaration['operator_name']}")
            else:
                print(f"Could not find op_info for {declaration['operator_name']}")

            return close

        for dec in self.declarations:
            test_op(dec)


if __name__ == "__main__":
    run_tests()

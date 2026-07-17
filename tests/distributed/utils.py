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

import pytest
import torch


def _assert_tensor_equal(result, expected, dtype, message_prefix):
    if dtype.is_floating_point:
        matches = torch.allclose(result, expected, rtol=1e-5, atol=1e-5)
        if not matches:
            # Find first mismatch for detailed error reporting
            diff = torch.abs(result - expected)
            mismatch_mask = diff > (1e-5 + 1e-5 * torch.abs(expected))
            if mismatch_mask.any():
                mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)
                first_mismatch = mismatch_indices[0].tolist()
                first_mismatch_idx = (
                    tuple(first_mismatch)
                    if len(first_mismatch) > 1
                    else first_mismatch[0]
                )
                error_msg = (
                    f"{message_prefix}\n"
                    f"First mismatch at index {first_mismatch_idx}:\n"
                    f"  Expected: {expected[first_mismatch_idx]}\n"
                    f"  Got:      {result[first_mismatch_idx]}\n"
                    f"First 10 expected: {expected.flatten()[:10]}\n"
                    f"First 10 result:   {result.flatten()[:10]}"
                )
            else:
                error_msg = f"{message_prefix}: tensors not close"
            pytest.fail(error_msg)
    else:
        matches = torch.equal(result, expected)
        if not matches:
            # Find first mismatch for detailed error reporting
            mismatch_mask = result != expected
            if mismatch_mask.any():
                mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)
                first_mismatch = mismatch_indices[0].tolist()
                first_mismatch_idx = (
                    tuple(first_mismatch)
                    if len(first_mismatch) > 1
                    else first_mismatch[0]
                )
                error_msg = (
                    f"{message_prefix}\n"
                    f"First mismatch at index {first_mismatch_idx}:\n"
                    f"  Expected: {expected[first_mismatch_idx]}\n"
                    f"  Got:      {result[first_mismatch_idx]}\n"
                    f"First 10 expected: {expected.flatten()[:10]}\n"
                    f"First 10 result:   {result.flatten()[:10]}"
                )
            else:
                error_msg = f"{message_prefix}: tensors not equal"
            pytest.fail(error_msg)

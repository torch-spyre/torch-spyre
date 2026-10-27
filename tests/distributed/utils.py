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

import math

import pytest
import torch


def _assert_tensor_equal(
    result, expected, dtype, message_prefix, *, rtol=1e-5, atol=1e-5
):
    if dtype.is_floating_point:
        matches = torch.allclose(result, expected, rtol=rtol, atol=atol)
        if not matches:
            # Find first mismatch for detailed error reporting
            diff = torch.abs(result - expected)
            mismatch_mask = diff > (atol + rtol * torch.abs(expected))
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


def _collective_rank_input(rank, shape, dtype):
    # Signed powers of two are exact in BF16 and DL16; 2**20 also detects an
    # unintended conversion through IEEE FP16, whose range is smaller.
    large = 2.0**20 if dtype == torch.bfloat16 else 16.0
    pattern = torch.tensor(
        [large, -large, 2.0**-20, -(2.0**-20), 0.5, -0.5, 1.5, -1.5], dtype=dtype
    )
    return (pattern.repeat(math.prod(shape) // len(pattern)) * (rank + 1)).reshape(
        shape
    )


def _bfloat16_rank_input(rank, shape):
    return _collective_rank_input(rank, shape, torch.bfloat16)


def _collective_expected(op, shape, world_size, dtype):
    if op == "broadcast":
        return _collective_rank_input(world_size - 1, shape, dtype)
    values = [_collective_rank_input(rank, shape, dtype) for rank in range(world_size)]
    if op in ("allgather", "gather"):
        return torch.cat(values, dim=0)
    return torch.stack(values).sum(dim=0)


def _bfloat16_collective_expected(op, shape, world_size):
    return _collective_expected(op, shape, world_size, torch.bfloat16)


def _native_bfloat16_tensor(device):
    from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout

    # Explicit native BF16 storage must not be reinterpreted as DL16.
    layout = SpyreTensorLayout([2, 64], [64, 1], DataFormats.BFLOAT16, {(0, 1): 128})
    return spyre_empty_with_layout([128], [1], torch.bfloat16, layout, device=device)


def _overpadded_collective_tensor(values, padded_width, device):
    from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout

    # Match the materialized input in Granite's compiled logits allgather:
    # logical [1, 24640], device [416, 1, 64], stride map [64, -1, 1].
    # The compiler retains the projection's 26624-element physical allocation.
    layout = SpyreTensorLayout(
        [padded_width // 64, 1, 64], [64, -1, 1], DataFormats.SEN169_FP16, {}
    )
    tensor = spyre_empty_with_layout(
        values.shape, values.stride(), values.dtype, layout, device=device
    )
    tensor.copy_(values)
    return tensor

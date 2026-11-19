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

"""Raw host-transfer primitives; no loader or scheduling policy."""

import gc
import mmap

import pytest
import torch
from inductor.utils_inductor import expects_raise

DEVICE = torch.device("spyre")


def _default(t):
    return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=DEVICE)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("num_threads", [0, 1, 4, 64])
def test_raw_copy_from_owned_host_buffer(transposed, num_threads):
    from torch_spyre._C import (
        convert_h2d_into,
        copy_raw_h2d_batch,
        copy_tensor,
        dma_bytes,
    )

    entries, references, conversions = [], [], []
    for width in (64, 128, 192, 256, 320):
        src = torch.randn(64, width).half()
        if transposed:
            src = src.t()
        dst = _default(src)
        reference = _default(src)
        copy_tensor(src, reference, non_blocking=False)
        backing = mmap.mmap(-1, dma_bytes(dst))
        buffer = torch.frombuffer(backing, dtype=torch.uint8)
        del backing
        gc.collect()
        conversions.append((memoryview(buffer.numpy()), src, dst))
        entries.append((buffer, dst))
        references.append(reference)
    convert_h2d_into(conversions, num_threads=num_threads)
    copy_raw_h2d_batch(entries)
    for (_, dst), reference in zip(entries, references):
        torch.testing.assert_close(dst.cpu(), reference.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("num_threads", [0, 1, 4])
def test_host_conversion_accepts_empty_batch(num_threads):
    from torch_spyre._C import convert_h2d_into

    convert_h2d_into([], num_threads=num_threads)


@expects_raise
def test_raw_copy_rejects_unaligned_buffer():
    from torch_spyre._C import copy_raw_h2d_batch, dma_bytes

    dst = _default(torch.randn(64, 64).half())
    buffer = torch.frombuffer(mmap.mmap(-1, dma_bytes(dst) + 1), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="IOMMU-aligned"):
        copy_raw_h2d_batch([(buffer[1:], dst)])


@expects_raise
def test_raw_copy_rejects_insufficient_capacity():
    from torch_spyre._C import copy_raw_h2d_batch

    dst = _default(torch.randn(64, 64).half())
    buffer = torch.frombuffer(mmap.mmap(-1, 1), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="buffer size must equal"):
        copy_raw_h2d_batch([(buffer, dst)])


@expects_raise
def test_host_conversion_rejects_insufficient_capacity():
    from torch_spyre._C import convert_h2d_into

    src = torch.randn(64, 64).half()
    dst = _default(src)
    tiny = torch.empty(1, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="single-chunk allocation of at most"):
        convert_h2d_into([(memoryview(tiny.numpy()), src, dst)], num_threads=1)


@expects_raise
@pytest.mark.parametrize("num_threads", [1, 4, 64])
def test_host_conversion_drains_valid_entries_after_validation_failure(num_threads):
    from torch_spyre._C import convert_h2d_into, dma_bytes

    src = torch.randn(64, 64).half()
    dst = _default(src)
    size = dma_bytes(dst)
    expected = torch.zeros(size, dtype=torch.uint8)
    convert_h2d_into([(memoryview(expected.numpy()), src, dst)], num_threads=1)
    buffers = [torch.zeros_like(expected) for _ in range(5)]
    entries = [(memoryview(buffer.numpy()), src, dst) for buffer in buffers]
    tiny = torch.full((1,), 0xA5, dtype=torch.uint8)
    entries.insert(2, (memoryview(tiny.numpy()), src, dst))
    with pytest.raises(RuntimeError, match="single-chunk allocation of at most"):
        convert_h2d_into(entries, num_threads)
    for buffer in buffers:
        torch.testing.assert_close(buffer, expected, rtol=0, atol=0)
    assert tiny.item() == 0xA5


@expects_raise
def test_raw_batch_validates_every_entry_before_dma():
    from torch_spyre._C import copy_raw_h2d_batch, copy_tensor, dma_bytes

    src = torch.ones(64, 64, dtype=torch.float16)
    dst = _default(src)
    copy_tensor(src, dst, non_blocking=False)
    buffer = torch.frombuffer(mmap.mmap(-1, dma_bytes(dst)), dtype=torch.uint8)
    buffer.zero_()
    with pytest.raises(RuntimeError, match="buffer size must equal"):
        copy_raw_h2d_batch([(buffer, dst), (buffer[:1], dst)])
    torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)

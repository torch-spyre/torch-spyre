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

"""Caller-owned byte buffer primitives; no loader or scheduling policy."""

import pytest
import torch
from inductor.utils_inductor import expects_raise

DEVICE = torch.device("spyre")


def test_buffer_conversion_needs_only_logical_metadata():
    import struct

    from torch_spyre._C import convert_buffers

    # Ordinary safetensors shape/dtype only. Independent coordinates cover the
    # backend's row permutation, padding and a partially filled final stick.
    source = struct.pack("<66i", *range(66))
    output = bytearray(b"X" * 640)
    convert_buffers([(output, source, [2, 33], "I32")])
    expected = bytearray(512)
    for row in range(2):
        for col in range(33):
            offset = ((col // 32) * 2 + row) * 32 + col % 32
            struct.pack_into("<i", expected, offset * 4, row * 33 + col)
    assert output[:512] == expected
    assert output[512:] == b"X" * 128


@expects_raise
def test_buffer_conversion_batch_preflight_does_not_mutate_valid_output():
    from torch_spyre._C import convert_buffers

    output = bytearray(b"X" * 128)
    tiny = bytearray(127)
    with pytest.raises(RuntimeError, match="capacity"):
        convert_buffers(
            [(output, bytes(128), [32], "I32"), (tiny, bytes(128), [32], "I32")]
        )
    assert output == b"X" * 128 and tiny == bytes(127)
    for shape, dtype in [([True], "I32"), ([0], "I32"), ([32], "IEEE_INT32")]:
        with pytest.raises(RuntimeError, match="integers|shape|dtype"):
            convert_buffers(
                [
                    (output, bytes(128), [32], "I32"),
                    (bytearray(128), bytes(128), shape, dtype),
                ]
            )
        assert output == b"X" * 128


def test_buffer_range_uses_existing_handle_pinned_factory_and_captured_stream():
    import torch_spyre
    from torch_spyre._C import copy_buffer_ranges, get_composite_address

    destination = torch.zeros((64, 64), dtype=torch.float32, device=DEVICE)
    handle = get_composite_address(destination)
    # The host allocation is deliberately much smaller than the device tensor.
    host = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
    host.view(torch.float32).fill_(3)
    stream = torch_spyre.Stream()
    # No stream capsule and no current-stream lookup in the worker/API.
    from concurrent.futures import ThreadPoolExecutor

    source = memoryview(host.numpy())
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(
            copy_buffer_ranges,
            [(source, handle, 128, 128)],
            stream=stream,
        ).result()
    # Enqueue returning is not completion; keep source/handle/stream alive.
    stream.synchronize()
    assert torch_spyre._C.stream_get_error(stream._cdata) == (
        torch_spyre._C.SpyreStreamError.Success
    )
    result = destination.cpu()
    assert (result == 3).sum().item() == 32
    assert (result == 0).sum().item() == 64 * 64 - 32


@expects_raise
def test_buffer_range_batch_rejects_invalid_tail_before_dma():
    import torch_spyre
    from torch_spyre._C import copy_buffer_ranges, get_composite_address

    destination = torch.ones((64, 64), dtype=torch.float32, device=DEVICE)
    host = torch.zeros(4096, dtype=torch.uint8, pin_memory=True)
    handle = get_composite_address(destination)
    with pytest.raises(RuntimeError, match="range"):
        copy_buffer_ranges(
            [
                (memoryview(host.numpy()), handle, 0, 128),
                (memoryview(host.numpy()), handle, 128, 127),
            ],
            stream=torch_spyre.current_stream(),
        )
    assert torch.equal(destination.cpu(), torch.ones(64, 64))

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
"""Shared helpers for reconstructing a tensor's captured memory layout.

The model-op test configs record each input tensor's real ``shape``,
``stride`` and ``storage_offset`` (captured from a live HuggingFace forward
trace). Both op-test frameworks build a contiguous, *seeded* tensor and then
need to reproduce that captured layout without disturbing the seeded values --
so a case is reproducible and identical across the two frameworks.

``apply_layout`` takes a contiguous seeded tensor and returns a view/copy with
the requested stride/storage_offset whose *logical* values are unchanged.
"""

from typing import List, Optional, Sequence

import torch


def _contiguous_stride(shape: Sequence[int]) -> List[int]:
    """C-contiguous stride for ``shape`` (matches ``torch.empty(shape).stride()``)."""
    stride = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = acc
        acc *= shape[i]
    return stride


def apply_layout(
    t: torch.Tensor,
    stride: Optional[Sequence[int]],
    storage_offset: int = 0,
) -> torch.Tensor:
    """Return ``t``'s seeded values laid out with ``stride``/``storage_offset``.

    ``t`` must be the contiguous, seeded tensor. The returned tensor has the
    same shape, dtype and *logical* values as ``t`` but the requested physical
    layout, so downstream ops see the real captured strides. A trivial layout
    (no stride and zero offset, or a stride equal to ``t``'s own) returns ``t``
    unchanged.

    Broadcast layouts (a ``0`` in ``stride``, e.g. GQA ``repeat_kv`` expand)
    cannot hold independent values in the broadcast dim, so they are rebuilt by
    seeding the collapsed (size-1) shape and expanding -- matching how the real
    tensor was produced.
    """
    if stride is None and not storage_offset:
        return t

    shape = list(t.shape)
    resolved = list(stride) if stride is not None else list(t.stride())
    if storage_offset == 0 and resolved == _contiguous_stride(shape):
        return t

    if 0 in resolved:
        # Broadcast/expand view: seed only the physically distinct positions
        # (collapse each broadcast dim to size 1) and expand back to `shape`.
        collapsed = [1 if st == 0 else s for s, st in zip(shape, resolved)]
        base = torch.as_strided(
            t.reshape(-1)[: _numel(collapsed)].clone(),
            collapsed,
            _contiguous_stride(collapsed),
        )
        return base.expand(shape)

    needed = storage_offset + (
        sum((s - 1) * st for s, st in zip(shape, resolved)) + 1 if shape else 1
    )
    backing = torch.empty(needed, dtype=t.dtype)
    view = torch.as_strided(backing, shape, resolved, storage_offset)
    with torch.no_grad():
        view.copy_(t)  # place the seeded values into the requested layout
    return view


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n

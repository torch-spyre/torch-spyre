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

"""Public sharded transfer API."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed.tensor import Shard

from torch_spyre._native_placement import (
    _ensure_spyre_runtime_initialized,
    sharded_receipt_from_native,
)
from torch_spyre.sharding import (
    ShardedReceipt,
    SpyreShardingSpec,
    _require_exact_int,
    _require_shard_count,
)

_SUPPORTED_TRANSFER_DTYPES = frozenset(
    {
        torch.bool,
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.int32,
        torch.uint8,
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class _ValidatedDistributeRequest:
    """Validated host-to-Spyre transfer request metadata."""

    source: torch.Tensor
    spec: SpyreShardingSpec
    global_shape: tuple[int, ...]


def _validate_host_transfer_source(source: object) -> torch.Tensor:
    """Validate a CPU host tensor before native sharded transfer is attempted."""

    if not isinstance(source, torch.Tensor):
        raise TypeError("source must be a torch.Tensor")
    if type(source) is not torch.Tensor and not isinstance(source, torch.nn.Parameter):
        raise TypeError("source must be a torch.Tensor or torch.nn.Parameter")
    if source.device.type != "cpu":
        raise ValueError("source must be a CPU tensor")
    if source.layout is not torch.strided:
        raise ValueError("source must use strided layout")
    if source.dim() == 0:
        raise ValueError("source must not be scalar")
    if any(type(extent) is not int or extent <= 0 for extent in source.shape):
        raise ValueError("source dimensions must be concrete and positive")
    if source.dtype not in _SUPPORTED_TRANSFER_DTYPES:
        raise TypeError("source dtype is not supported by sharded transfer")
    if not source.is_contiguous():
        raise ValueError("source must be contiguous")
    if source.storage_offset() != 0:
        raise ValueError("source must have zero storage offset")
    if source._base is not None:
        raise ValueError("source must be a base-like tensor")
    if source.is_conj() or source.is_neg():
        raise ValueError("source must not carry conjugate or negative view state")
    if source.requires_grad:
        raise ValueError("source must not require gradients")
    if source.untyped_storage().nbytes() != source.numel() * source.element_size():
        raise ValueError("source must use compact storage")
    return source


def _active_core_count() -> int:
    """Return the active compute core count used by compiled Spyre kernels.

    This is the same ``SENCORES`` setting the Inductor work-division planner
    reads, so sharded storage and the kernels that consume it agree on how many
    cores participate. Imported lazily to keep this module side-effect free.
    """

    from torch_spyre._inductor import config

    return config.sencores


def _validate_core_match(num_shards: int) -> None:
    """Enforce the sharding rule: one placement slot per active compute core."""

    active_cores = _active_core_count()
    if num_shards != active_cores:
        raise ValueError(
            f"distribute_tensor(num_shards={num_shards}) requires {num_shards} "
            f"active cores (one core per shard), but SENCORES={active_cores}; "
            f"set SENCORES={num_shards}"
        )


def _normalize_placement(placement: object):
    """Accept ``Shard(0)`` or the tensor-parallel idiom ``[Shard(0)]``.

    PyTorch's ``distribute_tensor`` takes one placement per mesh dimension;
    this device has no mesh, so exactly one placement is admitted and the
    list form is accepted purely for familiarity.
    """

    if isinstance(placement, (list, tuple)):
        if len(placement) != 1:
            raise TypeError(
                "placement must be a single Shard (one placement per tensor; "
                "this device has no multi-dimensional mesh)"
            )
        placement = placement[0]
    return placement


def _validate_distribute_request(
    source: object, placement: object, num_shards: object
) -> _ValidatedDistributeRequest:
    """Validate and plan one host-to-Spyre transfer request."""

    spec = SpyreShardingSpec(
        # SpyreShardingSpec.__post_init__ enforces the exact-Shard placement.
        _normalize_placement(placement),
        _require_shard_count(num_shards),
    )
    _validate_core_match(spec.num_shards)
    validated_source = _validate_host_transfer_source(source)
    global_shape = tuple(validated_source.shape)
    # Planning gate: degenerate shapes and out-of-range placement dims are
    # rejected here, before the runtime is touched.
    spec.plan(global_shape)
    return _ValidatedDistributeRequest(
        source=validated_source,
        spec=spec,
        global_shape=global_shape,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ShardedTensorInfo:
    """Read-only sharding record of a sharded Spyre tensor."""

    placement: Shard
    num_shards: int
    shard_device_bytes: int
    receipt: ShardedReceipt


def is_sharded(tensor: object) -> bool:
    """Return whether ``tensor`` is a sharded Spyre tensor."""

    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "spyre":
        return False
    from torch_spyre import _C

    return bool(_C._spyre_is_sharded(tensor))


def sharding_info(tensor: torch.Tensor) -> ShardedTensorInfo:
    """Return the validated sharding record and placement receipt of ``tensor``.

    Physical placement (reported opaquely per entry) is for inspection only;
    it is not part of the tensor's semantic identity.
    """

    if not is_sharded(tensor):
        raise TypeError("expected a sharded Spyre tensor")
    from torch_spyre import _C

    native = _C._spyre_sharding_info(tensor)
    shard_dim = _require_exact_int("shard_dim", native["shard_dim"])
    num_shards = _require_exact_int("num_shards", native["num_shards"])
    shard_device_bytes = _require_exact_int(
        "shard_device_bytes", native["shard_device_bytes"]
    )
    receipt = sharded_receipt_from_native(
        native["chunks"],
        num_shards,
        shard_device_bytes,
        _require_exact_int("total_bytes", native["total_bytes"]),
    )
    from torch.distributed.tensor import Shard

    return ShardedTensorInfo(
        placement=Shard(shard_dim),
        num_shards=num_shards,
        shard_device_bytes=shard_device_bytes,
        receipt=receipt,
    )


def distribute_tensor(
    tensor: object, placement: object, *, num_shards: object = None
) -> torch.Tensor:
    """Copy a CPU tensor into a sharded Spyre tensor.

    Example::

        y = torch.spyre.distribute_tensor(x, Shard(0))
        x_back = y.to("cpu")

    The name and call shape mirror ``torch.distributed.tensor.distribute_tensor``
    so tensor-parallel users recognize the pattern, with the device playing
    the role of the mesh. Unlike that function the result is a plain
    ``torch.Tensor`` on the ``spyre`` device — there is no DTensor, no mesh,
    no collectives (no ``redistribute()``, no autograd), and shards must be
    equal.

    ``placement`` is a ``Shard(dim)`` or the single-element list
    ``[Shard(dim)]``. ``num_shards`` must be 8 or 16 and must equal the
    active core count (``SENCORES``) — like tensor parallel, the shard count
    is a property of the hardware, not a free parameter. ``num_shards``
    omitted means "the active core count"; since the default ``SENCORES`` is
    32, a default environment must set ``SENCORES`` to 8 or 16 first.

    The result carries the global shape and dtype of ``tensor``. Its storage
    holds ``num_shards`` (8 or 16) backend
    placement slots; shard *i* (the *i*-th equal slice along the ``Shard``
    dimension) lives in slot *i*. In the current backend each slot is also
    bound to core *i* (an identity slot-to-core map); the placement receipt
    records the actual binding.

    Copying back to the CPU (``full_tensor``, ``.to("cpu")``, ``.cpu()``)
    gathers every shard. Operations that do not support sharded storage
    raise instead of reading part of the tensor. Fails closed: no tensor is
    returned if allocation or any shard copy fails, and nothing falls back
    to unified storage.
    """

    if num_shards is None:
        num_shards = _active_core_count()
    request = _validate_distribute_request(tensor, placement, num_shards)
    source = request.source
    dim = request.spec.placement.dim % source.dim()

    _ensure_spyre_runtime_initialized()
    from torch_spyre import _C

    result = _C._spyre_empty_sharded(
        list(request.global_shape),
        request.source.dtype,
        dim,
        request.spec.num_shards,
    )
    # Validate the native receipt before any data is copied.
    sharding_info(result)
    # Blocking per-shard layout-aware copy (spyre_copy_from handles sharded
    # destinations); on failure ``result`` is dropped and freed.
    _C.copy_tensor(source.detach(), result, non_blocking=False)
    return result


def full_tensor(tensor: object) -> torch.Tensor:
    """Return the gathered CPU copy of a sharded Spyre tensor.

    The tensor-parallel counterpart of ``DTensor.full_tensor()``: it gathers
    every shard into one ordinary CPU tensor. ``.to("cpu")`` and ``.cpu()``
    do the same thing.
    """

    if not isinstance(tensor, torch.Tensor) or not is_sharded(tensor):
        raise TypeError("expected a sharded Spyre tensor")
    return tensor.to("cpu")

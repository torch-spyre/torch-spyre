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

"""Immutable local-read facts; no graph, storage admission or planner imports."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from .core_mapping import (
    DenseSubrangeTransfer,
    PerCoreView,
    _EVALUATION_ERRORS,
    certify_dense_subrange_transfer,
    dense_read_rectangle,
    local_subrange_byte_offset,
    core_slices,
    work_division_from_view,
)
from .op_spec import TensorArg


@dataclass(frozen=True)
class LocalReadGeometry:
    parent_shape: tuple[int, ...]
    origin: tuple[int, ...]
    extent: tuple[int, ...]
    element_bytes: int


@dataclass(frozen=True)
class LocalReadEmission:
    certificate: DenseSubrangeTransfer
    source_view: PerCoreView
    device_dtype: Any
    element_arrangement: Any
    source_allocation: tuple[tuple[str, Any], ...]
    consumer_name: str
    consumer_allocation: tuple[tuple[str, Any], ...]


def has_partial_read(parent_shape, reads) -> bool:
    """Unknown coverage is partial; coverage alone grants no storage permission."""
    for dep in reads:
        try:
            if not bool(dep.get_numel() == math.prod(parent_shape)):
                return True
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            return True
    return False


def local_read_certificate(
    geometry: LocalReadGeometry | None,
    source_view: PerCoreView | None,
    read_view: PerCoreView | None,
) -> DenseSubrangeTransfer | None:
    if (
        geometry is None
        or source_view is None
        or read_view is None
        or source_view.num_cores is None
        or read_view.num_cores is None
        or not source_view.same_partition(read_view)
    ):
        return None
    try:
        maps = [
            core_slices(source_view, source_view.num_cores),
            core_slices(read_view, read_view.num_cores),
        ]
        certificate = certify_dense_subrange_transfer(
            geometry.parent_shape,
            geometry.origin,
            geometry.extent,
            dict(source_view.work_slice_dims),
            dict(read_view.work_slice_dims),
            maps[0],
            maps[1],
            element_bytes=geometry.element_bytes,
        )
        local_subrange_byte_offset(certificate)
        return certificate
    except (*_EVALUATION_ERRORS, NotImplementedError):
        return None


def normalize_local_read(
    contract: LocalReadEmission | None,
    name: str,
    argument: TensorArg,
    iteration_space,
    *,
    allow_extended: bool,
) -> TensorArg:
    """Validate the final access against fresh facts before rebasing its frame."""
    if contract is None:
        raise ValueError(f"{name}: partial LX input lacks a local certificate")
    certificate = contract.certificate
    allocation = dict(contract.source_allocation)
    if (
        tuple(argument.device_size) != certificate.parent_shape
        or argument.device_dtype != contract.device_dtype
        or argument.element_arrangement != contract.element_arrangement
        or argument.allocation != allocation
        or set(argument.allocation) != {"lx"}
        or argument.device_tile_advance_expr is not None
        or dense_read_rectangle(
            argument.device_size, argument.device_coordinates, iteration_space
        )
        != (certificate.origin, certificate.extent)
    ):
        raise ValueError(f"{name}: emitted LX access differs from its certificate")
    base = allocation["lx"]
    if type(base) is not int or base < 0 or base % 128:
        raise ValueError(f"{name}: local LX base is not a concrete stick address")
    consumer_allocation = dict(contract.consumer_allocation)
    if "lx" in consumer_allocation:
        child_base = consumer_allocation["lx"]
        if (
            base < child_base + certificate.destination_reservation_bytes
            and child_base < base + certificate.source_reservation_bytes
        ):
            raise ValueError(f"{name}: compact child overlaps its live parent")
    coordinates = [
        coordinate - start
        for coordinate, start in zip(argument.device_coordinates, certificate.origin)
    ]
    division = work_division_from_view(
        contract.source_view,
        certificate.extent,
        coordinates,
        iteration_space,
        allow_extended=allow_extended,
    )
    if (
        argument.work_division is None
        or division is None
        or not argument.work_division.same_ownership(division)
    ):
        raise ValueError(f"{name}: rebasing changed the committed LX ownership")
    return replace(
        argument,
        device_size=list(certificate.extent),
        device_coordinates=coordinates,
        allocation={"lx": base + local_subrange_byte_offset(certificate)},
        work_division=division,
    )

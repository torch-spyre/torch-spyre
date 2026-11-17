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

"""CPU geometry tests; these do not enable or test native subrange emission."""

from array import array
from dataclasses import FrozenInstanceError
from itertools import product
import math

import pytest

from torch_spyre._inductor.core_mapping import (
    certify_dense_subrange_transfer,
    local_subrange_byte_offset,
    transfer_edges,
)


def _owners(rows, features, replicas=1):
    return {
        core: {0: (core // rows) % features, 1: core % rows}
        for core in range(rows * features * replicas)
    }


def _certificate(
    shape=(24, 512, 64),
    origin=(20, 0, 0),
    extent=(4, 512, 64),
    source_splits=None,
    destination_splits=None,
    source_map=None,
    destination_map=None,
):
    return certify_dense_subrange_transfer(
        shape,
        origin,
        extent,
        {0: 8, 1: 4} if source_splits is None else source_splits,
        {1: 32} if destination_splits is None else destination_splits,
        _owners(4, 8) if source_map is None else source_map,
        _owners(32, 1) if destination_map is None else destination_map,
        element_bytes=2,
    )


def _address_oracle(certificate):
    """Compare copied global address tags with independently enumerated child data.

    The reference never uses the certificate's fragment origins, offsets or
    strides. It enumerates parent/child coordinate boxes from their owner maps.
    """
    shape = certificate.parent_shape
    rank = len(shape)
    parent_strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(rank))

    def tags(local_shape, base):
        result = array("I")
        for prefix in product(*(range(size) for size in local_shape[:-1])):
            address = (
                sum(
                    (start + coordinate) * stride
                    for start, coordinate, stride in zip(
                        base[:-1], prefix, parent_strides[:-1]
                    )
                )
                + base[-1]
            )
            result.extend(range(address, address + local_shape[-1]))
        return result

    source = [
        tags(
            certificate.source_local_shape,
            tuple(
                slot * size for slot, size in zip(owner, certificate.source_local_shape)
            ),
        )
        for owner in certificate.source_owners
    ]
    original = [values.tobytes() for values in source]
    count = math.prod(certificate.destination_local_shape)
    destinations = [array("I", [0]) * count for _ in certificate.destination_owners]
    coverage = [bytearray(count) for _ in destinations]
    for fragment in certificate.fragments:
        for prefix in product(*(range(size) for size in fragment.extent[:-1])):
            src = (
                fragment.source_offset_bytes
                + sum(
                    index * stride
                    for index, stride in zip(prefix, fragment.source_strides_bytes[:-1])
                )
            ) // certificate.element_bytes
            dst = (
                fragment.destination_offset_bytes
                + sum(
                    index * stride
                    for index, stride in zip(
                        prefix, fragment.destination_strides_bytes[:-1]
                    )
                )
            ) // certificate.element_bytes
            width = fragment.extent[-1]
            assert 0 <= src < src + width <= len(source[fragment.source_core])
            assert 0 <= dst < dst + width <= count
            marked = coverage[fragment.destination_core]
            assert not any(marked[dst : dst + width])
            marked[dst : dst + width] = b"\x01" * width
            destinations[fragment.destination_core][dst : dst + width] = source[
                fragment.source_core
            ][src : src + width]
    for core, owner in enumerate(certificate.destination_owners):
        base = tuple(
            start + slot * size
            for start, slot, size in zip(
                certificate.origin, owner, certificate.destination_local_shape
            )
        )
        assert all(coverage[core])
        assert destinations[core] == tags(certificate.destination_local_shape, base)
    assert original == [values.tobytes() for values in source]


@pytest.mark.parametrize(
    "origin,extent,fanin",
    [
        ((0, 0, 0), (16, 512, 64), 6),
        ((16, 0, 0), (4, 512, 64), 2),
        ((20, 0, 0), (4, 512, 64), 2),
    ],
)
def test_fused_qkv_subranges_match_address_oracle(origin, extent, fanin):
    certificate = _certificate(origin=origin, extent=extent)
    _address_oracle(certificate)
    assert certificate.source_reservation_bytes == 49152
    assert certificate.destination_reservation_bytes == math.prod(extent) * 2 // 32
    assert (
        max(
            sum(f.destination_core == c for f in certificate.fragments)
            for c in range(32)
        )
        == fanin
    )


def test_full_v_replica_keeps_unequal_fragments_and_parent_storage():
    certificate = _certificate(destination_splits={}, destination_map=_owners(1, 1, 32))
    with pytest.raises(ValueError, match="local subrange"):
        local_subrange_byte_offset(certificate)
    _address_oracle(certificate)
    incoming = [f for f in certificate.fragments if f.destination_core == 0]
    assert len(incoming) == 8
    assert {f.extent[0] for f in incoming} == {1, 3}
    assert {f.source_offset_bytes for f in incoming} == {0, 32768}
    assert certificate.source_reservation_bytes == 49152
    assert certificate.destination_reservation_bytes == 262144


def test_gate_up_siblings_have_distinct_copy_identity():
    options = dict(
        shape=(100, 512, 64),
        extent=(50, 512, 64),
        source_splits={0: 4, 1: 8},
        source_map=_owners(8, 4),
    )
    gate = _certificate(origin=(0, 0, 0), **options)
    up = _certificate(origin=(50, 0, 0), **options)
    for certificate in (gate, up):
        _address_oracle(certificate)
        assert certificate.source_reservation_bytes == 204800
        assert certificate.destination_reservation_bytes == 102400
    assert gate.destination_identity != up.destination_identity
    assert (
        gate.destination_identity
        == _certificate(origin=(0, 0, 0), **options).destination_identity
    )
    with pytest.raises(FrozenInstanceError):
        gate.origin = up.origin
    with pytest.raises(FrozenInstanceError):
        gate.fragments[0].source_offset_bytes = 1


@pytest.mark.parametrize("first", [0, 2])
def test_cache_head_subrange_preserves_post_rope_parent(first):
    from sympy import Mod, Symbol, floor
    from torch_spyre._inductor.core_mapping import dense_read_rectangle

    row, feature = (
        Symbol(name, integer=True, nonnegative=True) for name in ("row", "feature")
    )
    coordinates = (floor(feature / 64) + first, row, Mod(feature, 64))
    origin, extent = dense_read_rectangle(
        (4, 512, 64), coordinates, {row: 512, feature: 128}
    )
    assert origin == (first, 0, 0) and extent == (2, 512, 64)
    certificate = _certificate(
        shape=(4, 512, 64),
        origin=origin,
        extent=extent,
        source_splits={1: 32},
        source_map=_owners(32, 1),
    )
    _address_oracle(certificate)
    assert certificate.source_reservation_bytes == 8192
    assert certificate.destination_reservation_bytes == 4096
    assert {f.source_offset_bytes for f in certificate.fragments} == {first * 16 * 128}
    assert local_subrange_byte_offset(certificate) == first * 2048


def test_permuted_physical_owners_keep_their_real_source_addresses():
    owners = _owners(4, 8)
    certificate = _certificate(source_map={core: owners[31 - core] for core in owners})
    _address_oracle(certificate)
    assert {f.source_core for f in certificate.fragments} == set(range(8))


@pytest.mark.parametrize(
    "options,match",
    [
        ({"origin": (-1, 0, 0)}, "inside the parent"),
        ({"origin": (21, 0, 0)}, "inside the parent"),
        ({"extent": (0, 512, 64)}, "inside the parent"),
        ({"origin": (20, 0, 1), "extent": (4, 512, 63)}, "complete aligned"),
        ({"destination_splits": {1: 3}}, "divide each axis"),
        ({"source_splits": {0: 8, 1: 4, 2: 2}}, "complete sticks"),
        ({"source_map": {**_owners(4, 8), 31: {0: 0, 1: 0}}}, "complete domain"),
        ({"source_map": {**_owners(4, 8), 32: {0: 0, 1: 0}}}, "complete domain"),
        (
            {"source_map": {core + 1: owner for core, owner in _owners(4, 8).items()}},
            "physical domain",
        ),
        ({"source_map": {**_owners(4, 8), 0: {0: 0.0, 1: 0}}}, "nonintegral"),
        ({"destination_map": {**_owners(32, 1), 32: {0: 0, 1: 0}}}, "complete domain"),
        (
            {"destination_map": {**_owners(32, 1, 2), 32: {0: 0, 1: 1}}},
            "replicas uniform",
        ),
        (
            {"shape": (10**12, 512, 64), "source_splits": {0: 10**12}},
            "complete domain",
        ),
        (
            {
                "origin": (0, 0, 0),
                "extent": (16, 512, 64),
                "destination_splits": {},
                "destination_map": _owners(1, 1, 32),
            },
            "eight-source",
        ),
    ],
)
def test_invalid_subrange_contract_is_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        _certificate(**options)


def test_legacy_transfer_edges_keep_exact_partition_semantics():
    for source_split, destination_split in product(range(1, 10), repeat=2):
        source = {core: {0: core} for core in range(source_split)}
        destination = {core: {0: core} for core in range(destination_split)}
        expected = {
            (src, dst)
            for src in source
            for dst in destination
            if src * destination_split < (dst + 1) * source_split
            and dst * source_split < (src + 1) * destination_split
        }
        assert (
            transfer_edges(
                {0: source_split}, {0: destination_split}, source, destination
            )
            == expected
        )


def test_gapped_local_cache_read_is_not_compact():
    gapped = certify_dense_subrange_transfer(
        (512, 4, 64),
        (0, 2, 0),
        (512, 2, 64),
        {0: 32},
        {0: 32},
        {core: {0: core} for core in range(32)},
        {core: {0: core} for core in range(32)},
        element_bytes=2,
    )
    with pytest.raises(ValueError, match="not a compact"):
        local_subrange_byte_offset(gapped)

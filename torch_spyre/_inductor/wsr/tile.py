# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import sympy

# An irregular dimension is a dimension with size one or stride zero.


def compute_tile_stride(size, stride, tile_size):
    """
    Convert a tensor stride to a tile stride.

    Irregular dimensions are supported. Tile sizes must divide tensor sizes.
    Cumulative tile counts must divide strides. Padding is reduced in proportion
    of tile counts. Tile strides of irregular tile dimensions are set to zero.
    """
    assert all(x % y == 0 for x, y in zip(size, tile_size))
    # exclude irregular tensor dimensions (size==1 or stride==0)
    dims = [d for d, (s, t) in enumerate(zip(size, stride)) if s != 1 and t != 0]
    # order regular dimensions in increasing stride order
    dims.sort(key=lambda i: stride[i])
    tile_stride = [sympy.S.Zero] * len(tile_size)
    running_tile_count = sympy.S.One
    for d in dims:
        assert stride[d] % running_tile_count == 0
        if tile_size[d] > 1:
            tile_stride[d] = stride[d] // running_tile_count
        running_tile_count *= size[d] // tile_size[d]
    return tile_stride


def compute_tile_offset(offset, paired_strides):
    """
    Convert tensor offset to tile offset.

    Tensor and tile stride must be paired in decreasing tensor stride order.
    Irregular tensor dimensions must be excluded. Min stride must divide offset.
    """
    tile_offset = sympy.S.Zero
    for s, t in paired_strides:
        q, offset = divmod(offset, s)
        tile_offset += q * t
    assert offset == 0
    return tile_offset


def compute_tile_index(index, size, stride, tile_stride):
    """
    Convert a tensor index to a tile index.

    Tensor and tile are required to have the same dimensions laid out in the
    same order. Irregular tensor and tile dimensions are supported. Index
    derived from views are supported.
    """
    # exclude irregular tensor dimensions (size==1 or stride==0)
    dims = [d for d, (s, t) in enumerate(zip(size, stride)) if s != 1 and t != 0]
    stride = [stride[d] for d in dims]
    tile_stride = [tile_stride[d] for d in dims]
    paired_strides = list(reversed(sorted(zip(stride, tile_stride))))
    # sanitize index
    index = index.replace(sympy.floor, lambda x: x).expand()
    # handle constant offset
    offset = index.xreplace({var: sympy.S.Zero for var in index.free_symbols})
    tile_index = compute_tile_offset(offset, paired_strides)
    # handle iteration variables
    for term in (index - offset).as_ordered_terms():
        assert len(term.free_symbols) == 1
        if term.is_Symbol or term.func == sympy.Mod:
            offset = sympy.S.One
        else:
            assert term.func == sympy.Mul and term.args[0].is_Rational
            offset = term.args[0].numerator
        tile_index += compute_tile_offset(offset, paired_strides) * term // offset
    return tile_index

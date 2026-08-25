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

from torch.utils._sympy.functions import ModularIndexing

from ..errors import Unsupported

# An irregular dimension is a dimension with size one or stride zero.


def compute_tile_stride(size, stride, tile_size):
    """
    Convert a tensor stride to a tile stride.

    Irregular dimensions are supported. Tile sizes must divide tensor sizes.
    Cumulative tile counts must divide strides. Padding is reduced in proportion
    of tile counts. Tile strides of irregular tile dimensions are set to zero.
    """
    if not all(x % y == 0 for x, y in zip(size, tile_size)):
        raise Unsupported(f"tile sizes {tile_size} do not divide tensor sizes {size}")
    # exclude irregular tensor dimensions (size==1 or stride==0)
    dims = [d for d, (s, t) in enumerate(zip(size, stride)) if s != 1 and t != 0]
    # order regular dimensions in increasing stride order
    dims.sort(key=lambda i: stride[i])
    tile_stride = [sympy.S.Zero] * len(tile_size)
    running_tile_count = sympy.S.One
    for d in dims:
        if stride[d] % running_tile_count != 0:
            raise Unsupported(
                f"stride {stride[d]} at dim {d} is not divisible by cumulative"
                f" tile count {running_tile_count}"
            )
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
    if offset != 0:
        raise Unsupported(
            f"offset {offset} is not expressible in terms of the given strides"
        )
    return tile_offset


def decompose_index_for_tiling(index, var_ranges):
    """
    Decompose index into atoms + offset and validate assumptions.

    An atom is a tuple (positive integer coefficient, iteration variable). An
    offset is a non-negative integer. Coefficients and offset may include
    symbols. Tiling of indexes with ModularIndexing is not supported. Tiling of
    indexes with negative coefficients is not supported.
    """
    vars = set(var_ranges.keys())
    vars_found = set()
    offset = sympy.S.Zero
    atoms = []
    for term in index.as_ordered_terms():
        term_vars = list(set(term.free_symbols) & vars)
        if len(term_vars) == 0:
            offset += term
            continue
        if len(term_vars) != 1:
            raise Unsupported(
                f"index term {term} depends on multiple iteration variables {term_vars}"
            )
        var = term_vars[0]
        if var in vars_found:
            raise Unsupported(
                f"iteration variable {var} appears in multiple index terms"
            )
        vars_found.add(var)
        if isinstance(term, ModularIndexing):
            raise Unsupported(
                f"ModularIndexing in index term {term} is not supported for tiling"
            )
        if term == var:
            atoms.append((sympy.S.One, var))
            continue
        if term.func != sympy.Mul:
            raise Unsupported(f"index term {term} is not a linear monomial")
        prod = sympy.S.One
        var_found = False
        for arg in term.args:
            if isinstance(arg, ModularIndexing):
                raise Unsupported(
                    f"ModularIndexing in index term argument {arg} is not"
                    " supported for tiling"
                )
            if arg == var:
                if var_found:
                    raise Unsupported(
                        f"iteration variable {var} appears more than once in"
                        f" term {term}"
                    )
                var_found = True
                continue
            prod *= arg
        if not (prod > 0 and prod.is_integer):
            raise Unsupported(f"index coefficient {prod} is not a positive integer")
        atoms.append((prod, var))
    if not (offset >= 0 and offset.is_integer):
        raise Unsupported(f"index offset {offset} is not a non-negative integer")
    return atoms, offset


def compute_tile_index(index, var_ranges, size, stride, tile_stride):
    """
    Convert a tensor index to a tile index.

    Tensor and tile are required to have the same dimensions laid out in the
    same order. Irregular tensor and tile dimensions are supported. Index
    derived from views are supported.
    """
    atoms, offset = decompose_index_for_tiling(index, var_ranges)
    # exclude irregular tensor dimensions (size==1 or stride==0)
    dims = [d for d, (s, t) in enumerate(zip(size, stride)) if s != 1 and t != 0]
    stride = [stride[d] for d in dims]
    tile_stride = [tile_stride[d] for d in dims]
    paired_strides = list(reversed(sorted(zip(stride, tile_stride))))
    tile_index = compute_tile_offset(offset, paired_strides)
    # handle iteration variables
    for atom in atoms:
        tile_index += compute_tile_offset(atom[0], paired_strides) * atom[1]
    return tile_index

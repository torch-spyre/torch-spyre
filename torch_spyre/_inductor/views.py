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

# Helper methods to handle views

import sympy
from sympy import simplify
from typing import Sequence


def compute_relative_stride(
    rank: int, device_size: Sequence[sympy.Expr], dim_map: Sequence[int]
) -> list[sympy.Expr]:
    """
    Compute strides of device dimensions with respect to host dimensions
    """
    acc = [sympy.S.One] * rank
    rel_stride = [-1] * len(dim_map)
    for device_dim in range(len(dim_map) - 1, -1, -1):
        dim = dim_map[device_dim]
        if dim != -1:
            rel_stride[device_dim] = acc[dim]
            acc[dim] *= device_size[device_dim]
    return rel_stride


# def compute_coordinates(
#     size: Sequence[sympy.Expr],
#     stride: Sequence[sympy.Expr],
#     var_ranges: dict[sympy.Symbol, sympy.Expr],
#     index: sympy.Expr,
# ) -> list[sympy.Expr]:
#     """
#     Derive an array of coordinate expressions into a tensor from an index
#     """
#     coordinates = [sympy.S.Zero] * len(size)
#     vars = index.free_symbols
#     for var in vars:
#         #if var_ranges[var] <= 1:
#         if var_ranges[var] == 1 or (var_ranges[var].is_number and var_ranges[var] <= 1):
#             continue
#         term = index.subs({v: 0 for v in vars - {var}})
#         step = term.subs(var, 1)
#         limit = term.subs(var, var_ranges[var])
#         primary_stride = 0
#         primary_dim = -1
#         for dim in range(len(size)):
#             if size[dim] == 1:
#                 continue
#             st = stride[dim]
#             if st > step and st < limit:
#                 coordinates[dim] += var * step // st
#             elif st <= step and st > primary_stride:
#                 primary_stride = st
#                 primary_dim = dim
#         coordinates[primary_dim] += var * step // primary_stride
#     return coordinates

def compute_coordinates(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """
    Derive an array of coordinate expressions from a symbolic index.
    """
    coordinates = [sympy.S.Zero] * len(size)
    vars = index.free_symbols
    
    for var in vars:
        v_range = var_ranges.get(var, sympy.S.One)
        
        # Symbolic check for range <= 1
        if v_range == 1 or (v_range.is_number and v_range <= 1):
            continue
            
        # Extract 'step' as the coefficient of the variable
        step = index.diff(var)
        limit = step * v_range
        
        primary_stride = sympy.S.Zero
        primary_dim = -1
        
        for dim in range(len(size)):
            if size[dim] == 1:
                continue
            
            st = stride[dim]
            
            # Use simplify() == True to resolve symbolic inequalities
            is_between = (sympy.simplify(st > step) == True and 
                          sympy.simplify(st < limit) == True)
            
            is_primary = (sympy.simplify(st <= step) == True and 
                          sympy.simplify(st > primary_stride) == True)
            
            if is_between:
                coordinates[dim] += var * step // st
            elif is_primary:
                primary_stride = st
                primary_dim = dim
        
        if primary_dim != -1:
            coordinates[primary_dim] += var * step // primary_stride
            
    return [sympy.simplify(c) for c in coordinates]

def compute_device_coordinates(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    device_size: Sequence[sympy.Expr],
    dim_map: Sequence[int],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """
    Derive an array of coordinate expressions into a device tensor from an index
    """
    rel_stride = compute_relative_stride(len(size), device_size, dim_map)
    host_coordinates = compute_coordinates(size, stride, var_ranges, index)
    coordinates = [sympy.S.Zero] * len(device_size)
    for dim in range(len(device_size)):
        if dim_map[dim] == -1:
            continue
        expr = host_coordinates[dim_map[dim]]
        vars = expr.free_symbols
        for var in vars:
            term = expr.subs({v: 0 for v in vars - {var}})
            #step = term.subs(var, 1)
            step = term.diff(var)
            limit = step * var_ranges[var]
            #limit = term.subs(var, var_ranges[var])
            quotient = simplify(term / rel_stride[dim])
            if not quotient.has(var_ranges[var]): # Simple check to ensure it doesn't 'bleed'
                coordinates[dim] += quotient
            # if limit > rel_stride[dim] and step < rel_stride[dim] * device_size[dim]:
            #     coordinates[dim] += term // rel_stride[dim]
    return coordinates

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

from sympy import Expr, Symbol


"""Pure work-division helpers shared by scheduler and codegen paths.

This module intentionally avoids importing heavy backend/runtime components,
so unit tests can validate work-division behavior without requiring the native
`torch_spyre._C` extension to be present.
"""


def extend_iteration_space_with_core_division(
    it_space: dict[Symbol, Expr],
    core_division: dict[Symbol, int],
) -> dict[Symbol, tuple[Expr, int]]:
    """Attach per-dimension core split factors to an iteration space."""
    return {k: (v, core_division.get(k, 1)) for k, v in it_space.items()}


def core_split(size: int, max_cores: int) -> int:
    """Find the largest divisor of size that doesn't exceed max_cores.

    Args:
        size: The dimension size to split.
        max_cores: Maximum number of cores to use for this dimension.

    Returns:
        Number of cores to use (always divides size evenly).
    """
    for i in range(max_cores, 0, -1):
        if size % i == 0:
            return i
    return 1


def multi_dim_core_split(
    sizes: list[int], max_cores: int, priorities: list[int] | None = None
) -> list[int]:
    """Distribute max_cores across multiple dimensions optimally.

    This function tries to split cores across multiple dimensions to maximize
    parallelism while ensuring even division. It uses a greedy approach that
    prioritizes dimensions based on:
    1. User-specified priorities (if provided)
    2. Dimension size (larger dimensions get priority)

    Dimensions with negative priorities are excluded from splitting and will
    always have a split value of 1.

    Args:
        sizes: List of dimension sizes that can be parallelized.
        max_cores: Total number of cores available.
        priorities: Optional list of priority values (higher = more important).
            If None, uses dimension sizes as priorities.
            Use negative values to exclude dimensions from splitting.

    Returns:
        List of core splits for each dimension (same length as sizes).
        The product of all splits will be <= max_cores.
    """
    if not sizes:
        return []

    n_dims = len(sizes)
    splits = [1] * n_dims

    if priorities is None:
        priorities = sizes.copy()

    dim_info = [
        (i, sizes[i], priorities[i]) for i in range(n_dims) if priorities[i] >= 0
    ]
    dim_info.sort(key=lambda x: (x[2], x[1]), reverse=True)

    n_cores_to_split = max_cores
    for dim_idx, size, _ in dim_info:
        if n_cores_to_split <= 1:
            break

        best_split = core_split(size, n_cores_to_split)
        if best_split > 1:
            splits[dim_idx] = best_split
            n_cores_to_split = n_cores_to_split // best_split

    return splits


def multi_dim_iteration_space_split(
    iteration_space: dict[Symbol, Expr],
    max_cores: int,
    priorities: list[Symbol],
) -> dict[Symbol, int]:
    """Distribute max_cores across multiple dimensions of an iteration space.

    This function tries to split cores across multiple dimensions to maximize
    parallelism while ensuring even division. It uses a greedy approach that
    prioritizes dimensions of the iteration space based on caller-provided
    priority order.

    Args:
        iteration_space: The iteration space to be parallelized.
        max_cores: Total number of cores available.
        priorities: Order in which to consider the dimensions.

    Returns:
        Core splits for iteration_space dimensions.
        The product of all splits will be <= max_cores.
    """
    n_cores_to_split = max_cores
    splits = {v: 1 for v in iteration_space.keys()}

    for v in priorities:
        if n_cores_to_split <= 1:
            break
        best_split = core_split(iteration_space[v], n_cores_to_split)
        if best_split > 1:
            splits[v] = best_split
            n_cores_to_split = n_cores_to_split // best_split

    return splits

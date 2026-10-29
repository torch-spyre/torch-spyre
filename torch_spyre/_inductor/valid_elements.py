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

"""Device-dim groups of a SpyreTensorLayout, and coordinates derived through them.

A group is the set of device dims that hold one host dim, as the layout's
``valid_elements`` records it, mapped to that host dim's real extent.
``valid_elements_of`` returns a layout's groups: its stored map, or one derived
from the buffer's host layout when the layout carries none. Derivation is exact
for a layout whose tiles are full, and refused for one with padding, which only
the pass that added it can record.

``grouped_coordinates`` derives device coordinates through the groups:
``stride_map`` identifies each group's host dim, and the group's device sizes,
not the nesting of ``stride_map`` entries, spread that host dim's index over its
members. The two agree on full tiles; only the grouped form stays correct when a
tile is padded beyond its host extent.

``config.valid_elements_coords`` (``SPYRE_VALID_ELEMENTS_COORDS``) selects how
alignment uses them: ``on`` (the default) uses grouped coordinates wherever they
can be derived, ``off`` always uses stride matching, and ``shadow`` uses stride
matching while comparing both: disagreements are logged as warnings on the
``spyre.inductor.valid_elements`` logger (for debugging).
"""

import dataclasses
import functools
import math
from collections.abc import Sequence

import sympy

from . import config
from .logging_utils import get_inductor_logger
from .views import compute_coordinates

logger = get_inductor_logger("valid_elements")

Groups = dict[tuple[int, ...], int]


def mode() -> str:
    """``config.valid_elements_coords``: ``on``, ``off`` or ``shadow``."""
    return config.valid_elements_coords


@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """The groups of a layout, and where they came from."""

    groups: Groups
    derived: bool  # True when computed from the host layout, not read from the STL


class Underivable(Exception):
    """The groups of a layout cannot be determined from it and its host layout."""


def _layout_key(stl) -> tuple:
    return (
        tuple(int(s) for s in stl.device_size),
        tuple(int(s) for s in stl.stride_map),
    )


def valid_elements_of(
    stl, host_size: Sequence[int], host_stride: Sequence[int]
) -> GroupInfo:
    """The layout's groups: its ``valid_elements``, or derived from the host layout.

    A stored map is used as is when the layout cannot be derived from its host
    layout, which is the case for a padded one. When it can be, a stored map that
    disagrees with the derived one (beyond where size-1 dims are attached) does
    not describe this buffer, and the derived map is used instead: for a layout
    whose tiles are full the host layout is exact.

    Raises ``Underivable`` when the map is empty and the layout cannot be matched
    to its host layout, including when it holds more capacity than the host
    layout needs: that padding must be recorded by whoever added it.
    """
    stored = {tuple(int(d) for d in k): int(v) for k, v in stl.valid_elements.items()}
    try:
        derived = _derive(
            _layout_key(stl),
            tuple(int(s) for s in host_size),
            tuple(int(s) for s in host_stride),
        )
    except Underivable:
        if not stored:
            raise
        return GroupInfo(stored, derived=False)
    sizes = [int(s) for s in stl.device_size]
    if stored and _moving(stored, sizes) == _moving(derived, sizes):
        return GroupInfo(stored, derived=False)
    return GroupInfo(derived, derived=True)


def _moving(groups: Groups, device_size: Sequence[int]) -> Groups:
    """``groups`` without their size-1 members, which coordinates never move.

    ``init()`` attaches a size-1 dim to a group by its position, which strides
    cannot always reproduce (a stick dim of extent 1 has stride_map -1).
    """
    moving: Groups = {}
    for members, extent in groups.items():
        live = tuple(m for m in members if device_size[m] > 1)
        if live:
            moving[live] = extent
    return moving


def derived_from_host(
    device_size: Sequence[int],
    stride_map: Sequence[int],
    host_size: Sequence[int],
    host_stride: Sequence[int],
) -> Groups:
    """The groups of a layout derived from its host layout, or ``{}`` if it can't be.

    For a pass that builds a layout from the host layout, where deriving the map
    is exact and transforming the input's map is not.
    """
    try:
        return dict(
            _derive(
                (tuple(int(s) for s in device_size), tuple(int(s) for s in stride_map)),
                tuple(int(s) for s in host_size),
                tuple(int(s) for s in host_stride),
            )
        )
    except Underivable:
        return {}


@functools.lru_cache(maxsize=4096)
def _derive(layout_key: tuple, host_size: tuple, host_stride: tuple) -> Groups:
    device_size, stride_map = layout_key
    n = len(device_size)
    lane = n - 1
    eps = device_size[lane]
    real = [h for h in range(len(host_size)) if host_size[h] > 1]

    def host_dims_with_stride(stride):
        return [h for h in real if host_stride[h] == stride]

    # A sparse stick: another device dim steps the lane dim's host dim, one element
    # per stick in lane 0, as init() records it (valid extent 1).
    sparse = stride_map[lane] > 0 and any(
        device_size[d] > 1 and stride_map[d] == stride_map[lane] for d in range(lane)
    )
    stick_host = None
    if stride_map[lane] > 0 and not sparse:
        matches = host_dims_with_stride(stride_map[lane])
        if len(matches) != 1:
            raise Underivable(f"lane stride {stride_map[lane]} matches {matches}")
        stick_host = matches[0]

    groups: Groups = {}
    stick_members = [lane]
    matched = {stick_host} if stick_host is not None else set()
    if stick_host is not None:
        # The num-sticks dim steps the stick's host dim a stick at a time; for a
        # dim shorter than a stick, init() records the whole dim's extent instead,
        # and a restickify the stick. It belongs to the stick group even when it
        # holds a single stick.
        extent = host_size[stick_host]
        tile_strides = {
            min(eps, extent) * host_stride[stick_host],
            eps * host_stride[stick_host],
        }
        tiles = math.ceil(extent / eps)
    num_sticks_found = False
    for d in range(lane):
        size, sm = device_size[d], stride_map[d]
        if (
            stick_host is not None
            and not num_sticks_found
            and sm in tile_strides
            and size == tiles
        ):
            stick_members.append(d)
            num_sticks_found = True
            continue
        if size == 1:
            groups[(d,)] = 1
            continue
        if sm == 0:
            # An expanded host dim: every element of it is the same host element.
            groups[(d,)] = size
            continue
        matches = [h for h in host_dims_with_stride(sm) if host_size[h] == size]
        if len(matches) != 1:
            raise Underivable(
                f"device dim {d} (size {size}, stride_map {sm}) matches {matches}"
            )
        groups[(d,)] = host_size[matches[0]]
        matched.add(matches[0])
    unmatched = [h for h in real if h not in matched]
    if unmatched:
        raise Underivable(f"host dims {unmatched} have no device dim")
    if sparse:
        groups[(lane,)] = 1
        return groups
    stick_extent = host_size[stick_host] if stick_host is not None else 1
    stick_capacity = math.prod(device_size[d] for d in stick_members)
    if stick_capacity > math.ceil(stick_extent / eps) * eps:
        raise Underivable(
            f"stick group capacity {stick_capacity} exceeds whole sticks of "
            f"{stick_extent}: padding without valid_elements"
        )
    groups[tuple(sorted(stick_members))] = stick_extent
    return groups


def grouped_coordinates(
    stl,
    host_size: Sequence[int],
    host_stride: Sequence[int],
    it_space: dict,
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """Device coordinates of ``index`` through the layout's groups.

    ``index`` addresses the buffer's storage, so a view's access is decomposed with
    the buffer's host layout. Raises ``Underivable`` when the groups are unknown, a
    group's host dim cannot be identified, the access is not a plain walk over the
    host dims, a group has more than two members that move, or two that do not
    nest; stride matching covers those.
    """
    info = valid_elements_of(stl, host_size, host_stride)
    device_size = [int(s) for s in stl.device_size]
    stride_map = [int(s) for s in stl.stride_map]
    host_coords = [
        _strip_symbol_floors(c)
        for c in compute_coordinates(
            list(host_size), list(host_stride), it_space, index, None
        )
    ]
    # The access must decompose into the buffer's host dims exactly; an overlapping
    # view (unfold) does not.
    rebuilt = sum(c * st for c, st in zip(host_coords, host_stride))
    if sympy.expand(rebuilt - index) != 0:
        raise Underivable(f"index {index} does not decompose into host dims")
    real = [h for h in range(len(host_size)) if host_size[h] > 1]
    coords: list[sympy.Expr] = [sympy.S.Zero] * len(device_size)
    for members, extent in info.groups.items():
        live = [m for m in members if device_size[m] > 1]
        if not live or extent <= 1:
            # Size-1 members, and groups of one valid element, never move.
            continue
        if len(live) > 2:
            raise Underivable(f"group {members} has more than two non-unit members")
        step = stride_map[live[-1]]
        if step <= 0:
            # An expanded host dim: the access never moves along it.
            continue
        # The group holds the host dims whose strides fall within its valid
        # extent, [step, step * extent): one host dim, or several a view merged.
        # Bounding by the valid extent rather than the capacity keeps a padded
        # tile from swallowing the next host dim.
        span = [h for h in real if step <= host_stride[h] < step * extent]
        if not span or any(host_stride[h] % step for h in span):
            raise Underivable(f"group {members}: no host dims in its span from {step}")
        flat = sum(host_coords[h] * (host_stride[h] // step) for h in span)
        if len(live) > 1 and len(flat.free_symbols) > 1:
            # A host index summing several loop vars (a conv window over a
            # flattened image) is spread term by term downstream; stride matching
            # produces that form, a single mixed-radix split does not.
            raise Underivable(f"group {members}: index {flat} has several loop vars")
        if len(live) == 1:
            coords[live[0]] = flat
            continue
        outer, inner = live
        width = device_size[inner]
        if stride_map[outer] != width * stride_map[inner]:
            # The members do not nest: an outer step is not a whole run of the
            # inner one (an FP8 stick of 128 lanes tiled 64 elements at a time),
            # so the device sizes do not say where an element is.
            raise Underivable(f"group {members}: members do not nest")
        # A constant offset (a slice start) is split over the group separately and
        # added outside the floor/Mod, as stride matching does: downstream reads it
        # as the access's start, so a lane coordinate may exceed the stick width.
        offset, moving = sympy.sympify(flat).as_coeff_Add()
        offset = int(offset)
        bounds = _linear_bounds(moving, it_space)
        if bounds is not None and bounds[0] // width == bounds[1] // width:
            # The access stays within one block of the group, e.g. a one-stick
            # slice: the outer coordinate is a constant and the inner one linear.
            # Alignment cannot place a floor whose range is a single block.
            block = bounds[0] // width
            outer_coord, inner_coord = sympy.Integer(block), moving - block * width
        else:
            outer_coord = sympy.floor(moving / width)
            inner_coord = sympy.Mod(moving, width)
        coords[outer] = outer_coord + offset // width
        coords[inner] = inner_coord + offset % width
    return coords


def _strip_symbol_floors(expr):
    """``floor(v)`` is ``v`` for a loop var, which is always an integer."""
    return sympy.sympify(expr).replace(
        lambda e: isinstance(e, sympy.floor) and e.args[0].is_Symbol,
        lambda e: e.args[0],
    )


def _range_of(var, it_space: dict) -> int | None:
    """The concrete range of loop var ``var``, or None when it has none."""
    r = it_space.get(var)
    r = r[0] if isinstance(r, tuple) else r
    if r is None:
        return None
    try:
        return int(r)
    except TypeError:
        return None


def _linear_bounds(expr, it_space: dict) -> tuple[int, int] | None:
    """Min and max of a linear ``expr`` with non-negative integer coefficients."""
    expr = sympy.expand(expr)
    if not expr.is_polynomial(*expr.free_symbols):
        return None
    lo = hi = 0
    for term in sympy.Add.make_args(expr):
        coeff, rest = term.as_coeff_Mul()
        if not coeff.is_Integer or coeff < 0:
            return None
        if rest == 1:
            lo += int(coeff)
            hi += int(coeff)
            continue
        size = _range_of(rest, it_space) if rest.is_Symbol else None
        if size is None:
            return None
        hi += int(coeff) * (size - 1)
    return lo, hi


def coordinates_agree(a, b, it_space: dict, samples: int = 16) -> bool:
    """Whether two coordinate lists are equal over sampled points of ``it_space``."""
    if len(a) != len(b):
        return False
    if all(sympy.sympify(x) == sympy.sympify(y) for x, y in zip(a, b)):
        return True
    variables = sorted(set().union(*(e.free_symbols for e in [*a, *b])), key=str)
    maybe_ranges = {v: _range_of(v, it_space) for v in variables}
    ranges = {v: r for v, r in maybe_ranges.items() if r is not None}
    if len(ranges) != len(variables):
        return all(sympy.simplify(x - y) == 0 for x, y in zip(a, b))
    points = [{v: 0 for v in variables}, {v: ranges[v] - 1 for v in variables}]
    for k in range(samples):
        points.append(
            {v: (k * 2654435761 + 97 * j) % ranges[v] for j, v in enumerate(variables)}
        )
    return all(
        int(sympy.sympify(x).subs(p)) == int(sympy.sympify(y).subs(p))
        for p in points
        for x, y in zip(a, b)
    )

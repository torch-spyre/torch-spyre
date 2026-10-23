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

"""The annealing backend's native first-fit packer (``_C.pack_boxes``),
against a from-scratch Python reference. Device-free."""

from __future__ import annotations

import random

import pytest

from torch_spyre import _C


def reference(order, x0, x1, size, present, pinned, ties):
    offset = [0] * len(x0)
    placed = {}
    for i in range(len(x0)):
        if present[i] and pinned[i] >= 0:
            offset[i] = pinned[i]
            placed[i] = (x0[i], x1[i], pinned[i], pinned[i] + size[i])
    for i in order:
        if not present[i]:
            offset[i] = 0
            continue
        tied = [t for t in ties[i] if t in placed]
        if tied:
            offset[i] = offset[tied[0]]
        else:
            near = [
                (y0, y1)
                for p0, p1, y0, y1 in placed.values()
                if p0 < x1[i] and x0[i] < p1 and y0 < y1
            ]
            offset[i] = min(
                c
                for c in (0, *(y1 for _, y1 in near))
                if size[i] == 0 or all(c + size[i] <= y0 or y1 <= c for y0, y1 in near)
            )
        placed[i] = (x0[i], x1[i], offset[i], offset[i] + size[i])
    return offset


def random_boxes(rng, n):
    x0 = [rng.randrange(0, 20) for _ in range(n)]
    x1 = [s + rng.randrange(0, 8) for s in x0]
    size = [rng.choice([0, 16, 64, 128, 512]) for _ in range(n)]
    present = [rng.random() < 0.8 for _ in range(n)]
    pinned = [rng.choice([-1, -1, -1, 0, 256]) for _ in range(n)]
    ties = [[] for _ in range(n)]
    for _ in range(n // 4):  # merges: a child abuts its parent in time
        p, c = rng.sample(range(n), 2)
        x0[c], x1[c] = x1[p], max(x1[c], x1[p])
        ties[p].append(c)
        ties[c].append(p)
    order = [i for i in range(n) if pinned[i] < 0]
    rng.shuffle(order)
    return order, x0, x1, size, present, pinned, ties


@pytest.mark.parametrize("seed", range(200))
def test_matches_reference(seed):
    rng = random.Random(seed)
    args = random_boxes(rng, rng.randrange(1, 40))
    assert list(_C.pack_boxes(*args)) == reference(*args)


@pytest.mark.parametrize("seed", range(50))
def test_untied_packing_never_overlaps(seed):
    rng = random.Random(seed)
    order, x0, x1, size, present, _, _ = random_boxes(rng, 30)
    n = len(x0)
    order = list(range(n))
    offsets = _C.pack_boxes(order, x0, x1, size, present, [-1] * n, [[]] * n)
    live = [i for i in range(n) if present[i] and size[i] and x0[i] < x1[i]]
    for i in live:
        for j in live:
            if i < j and x0[i] < x1[j] and x0[j] < x1[i]:
                assert (
                    offsets[i] + size[i] <= offsets[j]
                    or offsets[j] + size[j] <= offsets[i]
                )


def test_first_fit_order_decides_the_stack():
    # Two boxes alive together: whichever is placed first sits at 0.
    args = ([0, 1], [0, 0], [4, 4], [64, 32], [True, True], [-1, -1], [[], []])
    assert list(_C.pack_boxes(*args)) == [0, 64]
    args = ([1, 0], *args[1:])
    assert list(_C.pack_boxes(*args)) == [32, 0]


def test_lifetimes_that_do_not_overlap_share_offset_zero():
    args = ([0, 1], [0, 4], [4, 8], [64, 64], [True, True], [-1, -1], [[], []])
    assert list(_C.pack_boxes(*args)) == [0, 0]


def test_pinned_box_is_an_obstacle():
    args = ([1], [0, 0], [4, 4], [64, 32], [True, True], [0, -1], [[], []])
    assert list(_C.pack_boxes(*args)) == [0, 64]


def test_active_merge_takes_the_partner_offset():
    # The child would fit at 0 above nothing, but its merge pins it to the
    # parent, which sits above an unrelated box.
    args = (
        [2, 0, 1],
        [0, 4, 0],
        [4, 8, 4],
        [64, 64, 32],
        [True, True, True],
        [-1, -1, -1],
        [[1], [0], []],
    )
    assert list(_C.pack_boxes(*args)) == [32, 32, 0]


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="differ in length"):
        _C.pack_boxes([0], [0], [1, 2], [1], [True], [-1], [[]])

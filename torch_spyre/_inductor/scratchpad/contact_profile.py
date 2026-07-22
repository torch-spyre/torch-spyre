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


"""The contact-profile data structure shared by the permutation-based layout
solvers.

A :class:`Profile` is a step function from a half-open column span to labels
(neighbouring buffer indices, or ``None``). It is the substrate on which
:class:`PermutationBasedLayoutSolver` maintains each buffer's below/above
contacts incrementally.
"""

from typing import Optional
import bisect


def _coalesce_segments(
    starts: list[int], labels: list[Optional[int]]
) -> tuple[list[int], list[Optional[int]]]:
    """Merge adjacent segments carrying equal labels. ``starts`` has length
    ``len(labels) + 1``; segment ``i`` covers ``[starts[i], starts[i+1])``."""
    out_starts = [starts[0]]
    out_labels: list[Optional[int]] = []
    for i, label in enumerate(labels):
        if out_labels and out_labels[-1] == label:
            out_starts[-1] = starts[i + 1]  # extend the previous segment
        else:
            out_labels.append(label)
            out_starts.append(starts[i + 1])
    return out_starts, out_labels


class Profile:
    """A step function from a half-open span ``[span_start, span_end)`` to labels
    (each an ``Optional[int]``; ``None`` means "no neighbour here").

    Stored as parallel lists: ``starts`` of length ``n + 1`` and ``labels`` of
    length ``n``; segment ``i`` covers ``[starts[i], starts[i + 1])`` carrying
    ``labels[i]``, with ``starts[-1] == span_end``.

    Canonical form (every mutating operation restores it): ``starts`` strictly
    increasing, and no two adjacent segments carry equal labels.
    """

    __slots__ = ("starts", "labels")

    def __init__(self, starts: list[int], labels: list[Optional[int]]):
        self.starts = starts
        self.labels = labels

    @classmethod
    def uniform(cls, span_start: int, span_end: int, label: Optional[int]) -> "Profile":
        """A single-segment profile over ``[span_start, span_end)``."""
        assert span_start < span_end
        return cls([span_start, span_end], [label])

    @classmethod
    def from_segments(cls, starts: list[int], labels: list[Optional[int]]) -> "Profile":
        """Build a canonical profile from segments that tile the span (coalescing
        adjacent equal labels)."""
        assert len(starts) == len(labels) + 1 and len(labels) >= 1
        return cls(*_coalesce_segments(starts, labels))

    @property
    def span_start(self) -> int:
        return self.starts[0]

    @property
    def span_end(self) -> int:
        return self.starts[-1]

    def label_at(self, t: int) -> Optional[int]:
        """The label of the segment containing column ``t`` (``t`` in span)."""
        assert self.starts[0] <= t < self.starts[-1]
        return self.labels[bisect.bisect_right(self.starts, t) - 1]

    def segments(self, a: int, b: int) -> tuple[list[int], list[Optional[int]]]:
        """The segments clipped to ``[a, b)`` as fresh lists (no aliasing): the
        first segment's start is clamped to ``a`` and the last end to ``b``.
        An empty range yields ``([a], [])``."""
        assert self.starts[0] <= a <= b <= self.starts[-1]
        if a == b:
            return [a], []
        out_starts = [a]
        out_labels: list[Optional[int]] = []
        i = bisect.bisect_right(self.starts, a) - 1
        while self.starts[i] < b:
            out_labels.append(self.labels[i])
            out_starts.append(min(self.starts[i + 1], b))
            i += 1
        return out_starts, out_labels

    def splice(
        self, a: int, b: int, seg_starts: list[int], seg_labels: list[Optional[int]]
    ) -> None:
        """Replace the function on ``[a, b)`` with the given segments (which must
        exactly tile ``[a, b)``), coalescing at both seams. No-op if ``a == b``."""
        assert self.starts[0] <= a <= b <= self.starts[-1]
        if a == b:
            return
        assert seg_starts[0] == a and seg_starts[-1] == b
        left_s, left_l = self.segments(self.starts[0], a)
        right_s, right_l = self.segments(b, self.starts[-1])
        new_s = left_s[:-1] + list(seg_starts[:-1]) + right_s
        new_l = left_l + list(seg_labels) + right_l
        self.starts, self.labels = _coalesce_segments(new_s, new_l)

    def relabel(self, a: int, b: int, mapping: dict) -> None:
        """For every segment within ``[a, b)`` whose label is a key of
        ``mapping``, replace it with ``mapping[label]`` (splitting straddling
        segments at the boundaries); coalesce afterwards. No-op if ``a == b``."""
        if a == b:
            return
        seg_s, seg_l = self.segments(a, b)
        new_l = [mapping[label] if label in mapping else label for label in seg_l]
        self.splice(a, b, seg_s, new_l)

    def label_set(self) -> set:
        """The set of labels appearing anywhere in the profile.

        (Named ``label_set`` rather than ``labels`` because ``labels`` is the
        segment-label list attribute.)"""
        return set(self.labels)

    def validate(self) -> None:
        """Raise ``AssertionError`` if the canonical-form invariants are broken."""
        assert len(self.starts) == len(self.labels) + 1, "length mismatch"
        assert len(self.labels) >= 1, "profile must have at least one segment"
        for i in range(len(self.starts) - 1):
            assert self.starts[i] < self.starts[i + 1], "starts not strictly increasing"
        for i in range(len(self.labels) - 1):
            assert self.labels[i] != self.labels[i + 1], "adjacent labels equal"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Profile)
            and self.starts == other.starts
            and self.labels == other.labels
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        segs = ", ".join(
            f"[{self.starts[i]},{self.starts[i + 1]})={self.labels[i]}"
            for i in range(len(self.labels))
        )
        return f"Profile({segs})"

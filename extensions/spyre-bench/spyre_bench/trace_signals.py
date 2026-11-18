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

"""Resolve role-named trace signals tolerantly: a role with no declared or no stored signal is NOT_CAPTURED.

A run declares which stored signal plays which role through `role.<family>.<role>` run properties; the roles a pack
knows are its registered `role.<family>.*` attributes. Unknown signals and roles are ignored, so an older reader keeps
working on newer data, and a missing signal is never reported as zero.

The stored contract is a JSON array (a list). Any other array-like with `__len__` and `__getitem__` (a numpy array, a
tuple) is accepted; str, bytes and mappings are not arrays.

Result per role: NOT_CAPTURED (no string name declared, the name is not stored, or the stored value is not an array),
or a Signal; an array-like that cannot be read (0-d, indexing that raises) is NOT_CAPTURED. An empty array is captured but empty (`Signal((), ())`). An entry is missing, and its position is listed
in `missing_at`, when it is None or NaN or, for a nested array, when any leaf inside it is; a partial array is never
passed off as complete.
"""

import math
from dataclasses import dataclass


class _NotCaptured:
    """Singleton marker distinct from None, 0 and []; always falsy."""

    def __bool__(self):
        return False

    def __repr__(self):
        return "NOT_CAPTURED"


NOT_CAPTURED = _NotCaptured()
_MAX_DEPTH = 8


@dataclass(frozen=True)
class Signal:
    values: tuple
    missing_at: tuple

    @property
    def complete(self):
        return not self.missing_at


def _is_array(v):
    return hasattr(v, "__len__") and hasattr(v, "__getitem__") and not isinstance(v, (str, bytes, dict))


def _items(arr):
    """Tuple of the elements of an array-like, or None when it cannot be read (0-d, hostile or odd indexing)."""
    try:
        return tuple(arr[i] for i in range(len(arr)))
    except Exception:  # noqa: BLE001 - duck-typed calls may raise anything; BaseException still propagates
        return None


def _leaf_absent(v):
    if v is None:
        return True
    if isinstance(v, (str, bytes, bool)):
        return False
    try:
        return math.isnan(float(v))  # also numpy scalars
    except (TypeError, ValueError, OverflowError):
        return False


def _absent(v, depth=0):
    if _is_array(v):
        items = _items(v)
        return items is None or (depth < _MAX_DEPTH and any(_absent(x, depth + 1) for x in items))
    return _leaf_absent(v)


def declared_roles(reg, family):
    """Role names registered for a family, e.g. declared_roles(reg, "moe")."""
    prefix = f"role.{family}."
    return sorted(k[len(prefix) :] for k in reg.attrs if k.startswith(prefix))


def _stored(measurements, name):
    if not isinstance(name, str) or measurements is None:
        return None
    try:
        return measurements[name]
    except (KeyError, TypeError, IndexError):
        return None


def resolve_roles(reg, family, run_props, measurements):
    """{role: Signal or NOT_CAPTURED} for every registered role of the family. None props/measurements read as empty."""
    props = run_props if hasattr(run_props, "get") else {}
    out = {}
    for role in declared_roles(reg, family):
        arr = _stored(measurements, props.get(f"role.{family}.{role}"))
        if not _is_array(arr):
            out[role] = NOT_CAPTURED
            continue
        vals = _items(arr)
        if vals is None:
            out[role] = NOT_CAPTURED
            continue
        out[role] = Signal(vals, tuple(i for i, v in enumerate(vals) if _absent(v)))
    return out

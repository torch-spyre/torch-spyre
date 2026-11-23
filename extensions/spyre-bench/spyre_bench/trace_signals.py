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
in `missing_at`, when it is None or NaN or, for a nested array, when it is empty, holds any missing leaf, contains
itself, nests deeper than MAX_DEPTH, holds more than MAX_NODES items, or cannot be read; a partial array is never
passed off as complete. Ordinary exceptions from duck-typed values are swallowed; BaseExceptions propagate.
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
MAX_DEPTH = 8  # deepest nesting inside one entry that is read
MAX_NODES = 10_000  # nested items visited per entry


@dataclass(frozen=True)
class Signal:
    values: tuple
    missing_at: tuple

    @property
    def complete(self):
        return not self.missing_at


def _is_array(v):
    try:
        return hasattr(v, "__len__") and hasattr(v, "__getitem__") and not isinstance(v, (str, bytes, dict))
    except Exception:  # noqa: BLE001 - hostile attribute access
        return False


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
    except Exception:  # noqa: BLE001
        return False


def _nested_absent(v, budget, path, depth):
    """True when a nested array is empty, unreadable, cyclic, too deep or large, or holds a missing leaf."""
    items = _items(v)
    if not items or depth > MAX_DEPTH or id(v) in path:
        return True
    path.add(id(v))
    try:
        for x in items:
            budget[0] -= 1
            if budget[0] < 0:
                return True
            if _nested_absent(x, budget, path, depth + 1) if _is_array(x) else _leaf_absent(x):
                return True
        return False
    finally:
        path.discard(id(v))


def _absent(v):
    return _nested_absent(v, [MAX_NODES], set(), 1) if _is_array(v) else _leaf_absent(v)


def declared_roles(reg, family):
    """Role names registered for a family, e.g. declared_roles(reg, "moe")."""
    prefix = f"role.{family}."
    return sorted(k[len(prefix) :] for k in reg.attrs if k.startswith(prefix))


def _stored(measurements, name):
    if not isinstance(name, str) or measurements is None:
        return None
    try:
        return measurements[name]
    except Exception:  # noqa: BLE001
        return None


def _prop(props, key):
    try:
        return props.get(key)
    except Exception:  # noqa: BLE001
        return None


def resolve_roles(reg, family, run_props, measurements):
    """{role: Signal or NOT_CAPTURED} for every registered role of the family. None props/measurements read as empty."""
    out = {}
    for role in declared_roles(reg, family):
        arr = _stored(measurements, _prop(run_props, f"role.{family}.{role}"))
        vals = _items(arr) if _is_array(arr) else None
        if vals is None:
            out[role] = NOT_CAPTURED
            continue
        out[role] = Signal(vals, tuple(i for i, v in enumerate(vals) if _absent(v)))
    return out

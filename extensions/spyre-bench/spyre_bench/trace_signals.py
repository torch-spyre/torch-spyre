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

Result per role: NOT_CAPTURED (no string name declared, the name is not stored, or the stored value is not an array),
or a Signal. An empty array is captured but empty (`Signal([], ())`). Entries that are None or NaN stay in `values` and
their positions are listed in `missing_at`, so a partial array is never passed off as complete.
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


@dataclass(frozen=True)
class Signal:
    values: tuple
    missing_at: tuple

    @property
    def complete(self):
        return not self.missing_at


def _absent(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def declared_roles(reg, family):
    """Role names registered for a family, e.g. declared_roles(reg, "moe")."""
    prefix = f"role.{family}."
    return sorted(k[len(prefix) :] for k in reg.attrs if k.startswith(prefix))


def resolve_roles(reg, family, run_props, measurements):
    """{role: Signal or NOT_CAPTURED} for every registered role of the family."""
    out = {}
    for role in declared_roles(reg, family):
        name = run_props.get(f"role.{family}.{role}")
        arr = measurements.get(name) if isinstance(name, str) else None
        if not isinstance(arr, (list, tuple)):
            out[role] = NOT_CAPTURED
            continue
        out[role] = Signal(tuple(arr), tuple(i for i, v in enumerate(arr) if _absent(v)))
    return out

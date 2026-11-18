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
"""

NOT_CAPTURED = None


def declared_roles(reg, family):
    """Role names registered for a family, e.g. declared_roles(reg, "moe")."""
    prefix = f"role.{family}."
    return sorted(k[len(prefix) :] for k in reg.attrs if k.startswith(prefix))


def resolve_roles(reg, family, run_props, measurements):
    """{role: values or NOT_CAPTURED} for every registered role of the family."""
    out = {}
    for role in declared_roles(reg, family):
        name = run_props.get(f"role.{family}.{role}")
        out[role] = measurements[name] if name and name in measurements else NOT_CAPTURED
    return out

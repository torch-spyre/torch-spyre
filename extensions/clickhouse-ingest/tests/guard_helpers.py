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

"""Test helpers: one policy object that carries both the retry and the breaker settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from spyre_clickhouse_ingest.breaker import Limits, guarded_insert
from spyre_clickhouse_ingest.resilience import Policy


@dataclass
class Pol(Policy):
    breaker_s: float = 60.0
    breaker_probe_s: float = 10.0
    budget_s: float = 900.0
    budget_window_s: float = 1800.0

    def split(self) -> tuple[Policy, Limits]:
        d = asdict(self)
        keys = ("breaker_s", "breaker_probe_s", "budget_s", "budget_window_s")
        return Policy(**{k: v for k, v in d.items() if k not in keys}), Limits(
            **{k: d[k] for k in keys}
        )


def guarded(*args, policy=None, **kw):
    """`guarded_insert` that also takes a `Pol`."""
    if isinstance(policy, Pol):
        policy, kw["limits"] = policy.split()
    return guarded_insert(*args, policy=policy, **kw)

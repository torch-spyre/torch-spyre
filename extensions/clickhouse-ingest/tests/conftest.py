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

import pytest

from spyre_clickhouse_ingest import resilience, writes
from spyre_clickhouse_ingest.writer import TestResultWriter


@pytest.fixture(autouse=True)
def no_counter_settle(monkeypatch):
    """The counters settle wait is for concurrent writers; tests have none."""
    monkeypatch.setattr(TestResultWriter, "COUNTER_SETTLE_S", 0.0)


@pytest.fixture(autouse=True)
def fresh_breaker(monkeypatch):
    """The breakers are shared state; a test that trips one must not leak into the next."""
    monkeypatch.setattr(writes, "_breakers", {})


@pytest.fixture(autouse=True)
def fresh_env_warnings():
    resilience._warned.clear()

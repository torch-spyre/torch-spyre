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

import sys

import pytest


@pytest.fixture(autouse=True)
def _fresh_write_state():
    """A park or a spent deadline in one test must not change the next one's writes."""
    writes = sys.modules.get("spyre_clickhouse_ingest.writes")
    if writes:
        writes.reset()
    yield
    writes = sys.modules.get("spyre_clickhouse_ingest.writes")
    if writes:
        writes.reset()

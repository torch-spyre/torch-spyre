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

"""Pins the connection factory's handling of BLANK env vars.

GitHub Actions exports an unset secret as the empty string, not as an absent variable, so every
`os.environ.get(name, default)` default in this factory used to be unreachable and a missing
CLICKHOUSE_PORT surfaced as `int('')`.
"""

import clickhouse_connect
import pytest
from spyre_clickhouse_ingest import client

REQUIRED = {"CLICKHOUSE_HOST": "ch.example.com", "CLICKHOUSE_PASS": "secret"}


@pytest.fixture
def captured(monkeypatch):
    """Capture the kwargs get_client would connect with."""
    seen = {}

    def fake_get_client(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(clickhouse_connect, "get_client", fake_get_client)
    for name in (
        "CLICKHOUSE_HOST",
        "CLICKHOUSE_PORT",
        "CLICKHOUSE_USER",
        "CLICKHOUSE_PASS",
        "CLICKHOUSE_DB",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)
    return seen


def test_blank_optional_vars_fall_back_to_the_documented_defaults(
    captured, monkeypatch
):
    for name in ("CLICKHOUSE_PORT", "CLICKHOUSE_USER", "CLICKHOUSE_DB"):
        monkeypatch.setenv(name, "")
    client.get_client()
    assert captured["port"] == 443
    assert captured["user"] == "default"
    assert captured["database"] == "spyre"


def test_absent_optional_vars_use_the_same_defaults(captured):
    client.get_client()
    assert (captured["port"], captured["user"], captured["database"]) == (
        443,
        "default",
        "spyre",
    )


def test_values_are_honoured_and_stripped(captured, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_PORT", " 8443 ")
    monkeypatch.setenv("CLICKHOUSE_DB", " other ")
    client.get_client()
    assert captured["port"] == 8443
    assert captured["database"] == "other"


def test_verify_defaults_true_and_can_be_relaxed(captured):
    client.get_client()
    assert captured["verify"] is True
    captured.clear()
    client.get_client(verify=False)
    assert captured["verify"] is False


@pytest.mark.parametrize("name", ["CLICKHOUSE_HOST", "CLICKHOUSE_PASS"])
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_required_var_names_itself(captured, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit, match=name):
        client.get_client()


def test_non_numeric_port_says_so(captured, monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_PORT", "not-a-port")
    with pytest.raises(SystemExit, match="CLICKHOUSE_PORT"):
        client.get_client()


def test_client_summary_reports_the_resolved_values(captured, monkeypatch):
    # Read through the same resolver as the connection, so a log banner cannot claim a port the
    # client did not use.
    monkeypatch.setenv("CLICKHOUSE_PORT", "")
    assert client.client_summary() == "ch.example.com:443/spyre"

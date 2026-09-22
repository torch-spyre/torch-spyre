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

"""ClickHouse connection factory and the v2 table-presence gate."""

import os

import clickhouse_connect

from . import schema


class ClickHouseEnv:
    """The connection settings, read from the environment through one resolver."""

    DEFAULT_PORT = "443"
    DEFAULT_USER = "default"
    DEFAULT_DB = "spyre"

    @staticmethod
    def get(name: str, default: str = "") -> str:
        """An env var, treating BLANK as absent -- GHA exports an unset secret as ''."""
        return (os.environ.get(name) or "").strip() or default

    @classmethod
    def host(cls) -> str:
        """CLICKHOUSE_HOST, which has no default."""
        return cls.get("CLICKHOUSE_HOST")

    @classmethod
    def port(cls) -> str:
        """CLICKHOUSE_PORT as a raw string, so a non-numeric value can name itself."""
        return cls.get("CLICKHOUSE_PORT", cls.DEFAULT_PORT)

    @classmethod
    def database(cls) -> str:
        """CLICKHOUSE_DB, the connection's own database."""
        return cls.get("CLICKHOUSE_DB", cls.DEFAULT_DB)

    @classmethod
    def secure(cls) -> bool:
        """CLICKHOUSE_SECURE=0 drops to plain HTTP, to reach a local box with no TLS."""
        return cls.get("CLICKHOUSE_SECURE", "1") not in ("0", "false", "no")

    @classmethod
    def target_database(cls) -> str:
        """CLICKHOUSE_DB_V2: the v2 database NAME, or '' when v2 is not configured."""
        return os.environ.get("CLICKHOUSE_DB_V2", "").strip()

    @classmethod
    def summary(cls) -> str:
        """A host:port/database string, resolved exactly as the connection is."""
        return f"{cls.host()}:{cls.port()}/{cls.database()}"


class ClickHouse:
    """The one connection factory for every ingest, plus the v2 write gate."""

    ENV = ClickHouseEnv

    @classmethod
    def connect(cls, *, verify: bool = True):
        """Connect with the resolved settings; `verify=False` for an unverified cert."""
        host = cls.ENV.host()
        if not host:
            raise SystemExit(
                "CLICKHOUSE_HOST is unset or empty -- check the secrets mapping"
            )
        secure = cls.ENV.secure()
        password = cls.ENV.get("CLICKHOUSE_PASS")
        if not password and secure:
            raise SystemExit(
                "CLICKHOUSE_PASS is unset or empty -- check the secrets mapping"
            )
        port_raw = cls.ENV.port()
        try:
            port = int(port_raw)
        except ValueError:
            raise SystemExit(f"CLICKHOUSE_PORT is not a number: {port_raw!r}") from None
        return clickhouse_connect.get_client(
            host=host,
            port=port,
            user=cls.ENV.get("CLICKHOUSE_USER", cls.ENV.DEFAULT_USER),
            password=password,
            database=cls.ENV.database(),
            secure=secure,
            verify=verify,
        )

    @staticmethod
    def tables_present(
        client, db: str, tables=None, check_columns: bool = True
    ) -> bool:
        """The v2 write gate: every table (default: functional pair) has its columns."""
        return all(
            t.present(client, db, check_columns=check_columns)
            for t in (tables or (schema.TestCases, schema.TestCaseRuns))
        )


# Function API, kept so installed consumers import one definition, not a copy.
_env = ClickHouseEnv.get
get_client = ClickHouse.connect
client_summary = ClickHouseEnv.summary
target_database = ClickHouseEnv.target_database
tables_present = ClickHouse.tables_present

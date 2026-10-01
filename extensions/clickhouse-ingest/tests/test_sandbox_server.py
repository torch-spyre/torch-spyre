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

"""Pins the sandbox server's fences: who a token is, which sandboxes they may touch, and
which schema sources it will fetch."""

import json

import pytest
from spyre_clickhouse_ingest.sandbox_server import (
    Config,
    Identity,
    Sandboxes,
    SchemaSource,
)

SECRET = b"s" * 48


@pytest.fixture
def identity(tmp_path):
    f = tmp_path / "tokens.conf"
    f.write_text("# comment\nshared-tok=Jane.Doe@ibm.com / profiling\n")
    return Identity(SECRET, f)


@pytest.mark.parametrize(
    "label, owner",
    [
        ("ashokponkumar", "ashokponkumar"),
        ("Jane.Doe@ibm.com / profiling", "jane_doe"),
        ("IAM#jane.doe@in.ibm.com", "jane_doe"),
        ("42ops", "u42ops"),
        ("a" * 40, "a" * 24),
        ("x__y", "x_y"),
    ],
)
def test_owner_is_normalized(label, owner):
    assert Identity.owner(label) == owner


def test_shared_file_and_minted_tokens_resolve(identity):
    assert identity.resolve("shared-tok") == "jane_doe"
    assert identity.resolve(identity.mint("bob")) == "bob"


def test_forged_or_foreign_tokens_do_not_resolve(identity, tmp_path):
    token = identity.mint("bob")
    assert identity.resolve(token.replace("sbx_bob", "sbx_ann")) is None
    assert identity.resolve("sbx_bob.AAAAAAAAAAAAAAAAAAAAAA") is None
    other = Identity(b"t" * 48, tmp_path / "missing.conf")
    assert other.resolve(token) is None
    assert other.resolve("shared-tok") is None


def test_short_secret_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        Identity(b"short", tmp_path / "t")


def test_passwords_are_per_database_and_stable(identity):
    assert identity.password("sandbox_a") == identity.password("sandbox_a")
    assert identity.password("sandbox_a") != identity.password("sandbox_b")


@pytest.mark.parametrize(
    "repo, ref",
    [
        ("torch-spyre", "main"),
        ("a/b/c", "main"),
        ("evil.com/x", "main"),
        ("me/torch-spyre", "../../etc"),
        ("me/torch-spyre", "main; rm"),
    ],
)
def test_schema_source_is_validated(repo, ref):
    with pytest.raises(ValueError):
        with SchemaSource.fetch(repo, ref):
            pass


class FakeAdmin:
    """Answers the one system.databases query Sandboxes.meta makes."""

    def __init__(self, dbs):
        self.dbs = dbs

    def query(self, sql, parameters=None):
        class R:
            result_rows = list(self.dbs.items())

        return R()


def ops(dbs):
    cfg = Config(secret=SECRET, max_sandboxes=2, max_days=30, max_runs=100, max_ttl=7)
    return Sandboxes(
        cfg, Identity(SECRET, cfg.tokens_file), lambda db: FakeAdmin(dbs), None
    )


def managed(owner):
    return json.dumps({"managed_by": "sandbox-server", "owner": owner, "expires": "x"})


def test_only_the_owner_may_act():
    dbs = {
        "sandbox_ann": managed("ann"),
        "sandbox_hand_made": "",
        "sandbox_bob": managed("bob"),
    }
    o = ops(dbs)
    admin = FakeAdmin(dbs)
    assert o.owned(admin, "ann", "") == "ann"
    with pytest.raises(PermissionError):
        o.owned(admin, "ann", "x")
    with pytest.raises(PermissionError):
        o.owned(FakeAdmin({"sandbox_ann": managed("bob")}), "ann", "")
    assert set(o.meta(admin)) == {"sandbox_ann", "sandbox_bob"}


@pytest.mark.parametrize("bad", ["A", "a_b", "x" * 13, "../", "a-b"])
def test_suffix_is_validated(bad):
    with pytest.raises(ValueError):
        ops({}).name("ann", bad)


def test_seed_filter_is_clamped_to_limits():
    f = ops({}).seed_filter({"days": 999, "runs_per_component": 0, "components": ["c"]})
    assert (f.days, f.runs_per_component, f.components) == (30, 1, ("c",))


def test_ttl_is_clamped():
    c = json.loads(ops({}).comment("ann", 999, "r@main"))
    assert c["owner"] == "ann" and c["managed_by"] == "sandbox-server"
    import datetime as dt

    left = dt.datetime.fromisoformat(c["expires"]) - dt.datetime.fromisoformat(
        c["created"]
    )
    assert left == dt.timedelta(days=7)


def test_reads_and_statements_are_told_apart():
    assert Sandboxes.READS.match("  with x as (select 1) select * from x")
    assert Sandboxes.READS.match("DESCRIBE TABLE t")
    assert not Sandboxes.READS.match("ALTER TABLE t ADD COLUMN c UInt8")
    assert not Sandboxes.READS.match("INSERT INTO t SELECT 1")

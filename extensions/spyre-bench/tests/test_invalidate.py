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

"""spyre-bench invalidate: flag handling, the dry-run default, the confirmation rule and the hand-off to the ingest.

The matching, the state machine and the ordering rule are the ingest's (tested there on chdb); here the backend is a fake.
python -m pytest -q -p no:cacheprovider tests/test_invalidate.py
"""

import json
import subprocess
import sys

import pytest

from spyre_bench import cli, ingest_adapter, invalidate
from spyre_bench import store as stores

RUN = "11111111-1111-4111-8111-111111111111"
BENCH = "22222222-2222-4222-8222-222222222222"
INV = "018f0000-0000-7000-8000-000000000001"
CODE = ["--reason-code", "faulty_hardware", "--reason", "card ran hot"]


class Refused(ValueError):
    pass


class Store(stores.Store):
    name = "fake"

    def client(self):
        return "client"


def preview(**kw):
    p = {
        "invalidation_id": INV, "scope_kind": "device", "state_before": None, "state_after": "invalidated", "noop": False,
        "matched_runs": 3, "matched_repeats": 9, "truncated": False, "threshold": 50, "needs_confirm": False,
        "confirm_token": "abc123abc123", "benchmarks": [{"benchmark_id": BENCH, "name": "op-alpha", "component": "c"}],
        "series": [{"component": "c", "benchmark_id": BENCH, "backend": "b", "arch": "x86_64", "source_job": "j", "name": "op-alpha", "runs": 3}],
        "runs": [{"run_id": RUN, "run_ts": "2026-10-01 02:00:00", "repeats": [0, 1, 2], "benchmarks": 1}],
        "sample_runs": [RUN],
        "verdict_changes": {"available": True, "total": 1, "transitions": {"regressed->unchanged": 1}, "truncated": False,
                            "rows": [{"ts": "2026-10-03", "name": "op-alpha", "metric": "m", "before": "regressed", "after": "unchanged"}]},
    }  # fmt: skip
    return {**p, **kw}


class Backend:
    InvalidationError = Refused

    def __init__(self, **kw):
        self.kw, self.previews, self.bundles = kw, [], []
        self.stored = {
            INV: {
                "invalidation_id": INV,
                "scope_kind": "device",
                "scope": {"device_serial": "SN1"},
                "window_from": None,
                "window_to": None,
                "state": "invalidated",
            }
        }

    def preview(self, client, ev, limit=None):
        self.previews.append(ev)
        if self.kw.get("refuse"):
            raise Refused(self.kw["refuse"])
        return preview(**self.kw.get("p", {}))

    def stored_event(self, client, inv):
        return self.stored.get(inv)

    def write_bundle(self, directory, events, uploader=""):
        self.bundles.append((str(directory), events, uploader))
        return directory

    def list_invalidations(self, client, state, limit):
        return [{"invalidation_id": INV, "state": "invalidated", "scope_kind": "device", "scope_key": "SN1", "reason_code": "faulty_hardware",
                 "actor": "amy", "updated_at": "t", "runs": 3}]  # fmt: skip

    def history(self, client, inv):
        return (
            [
                {
                    "ts": "t0",
                    "state": "invalidated",
                    "actor": "amy",
                    "reason_code": "faulty_hardware",
                    "reason": "hot",
                    "evidence_url": "",
                    "matched_runs": 3,
                    "scope_kind": "device",
                }
            ]
            if inv == INV
            else []
        )

    def impact(self, client, inv):
        return [{"run_id": RUN, "run_ts": "t", "repeats": 3, "benchmarks": 4}]


class Runner:
    def __init__(self, rc=0, out='{"status": "ingested"}'):
        self.rc, self.out, self.calls = rc, out, []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, self.rc, self.out, "")


def run(argv, backend=None, runner=None, capsys=None):
    backend, runner = backend or Backend(), runner or Runner()
    rc = invalidate.main(argv, st=Store(), backend=backend, runner=runner)
    return rc, backend, runner, capsys.readouterr() if capsys else None


def test_the_verb_is_registered():
    assert cli.VERBS["invalidate"][0] == "spyre_bench.invalidate"


def test_the_default_is_a_preview_that_writes_nothing(capsys):
    rc, be, runner, out = run(["--device", "SN1", *CODE], capsys=capsys)
    assert rc == 0 and be.bundles == [] and runner.calls == []
    assert "covers 3 runs, 9 repeats, 1 benchmarks, 1 series" in out.out and "regressed->unchanged 1" in out.out
    assert "dry run: nothing written" in out.out
    ev = be.previews[0]
    assert ev["scope_kind"] == "device" and ev["scope"] == {"device_serial": "SN1"} and ev["state"] == "invalidated"


def test_apply_records_the_event_through_a_bundle_and_the_ingest(capsys, tmp_path):
    rc, be, runner, out = run(
        [
            "--node",
            "n1",
            "--from",
            "2026-10-01T00:00:00Z",
            *CODE,
            "--apply",
            "--out",
            str(tmp_path / "b"),
            "--actor",
            "amy@example.com",
        ],
        capsys=capsys,
    )
    assert rc == 0
    ((directory, events, uploader),) = be.bundles
    assert directory == str(tmp_path / "b") and uploader == "amy@example.com"
    assert (
        events[0]["reason"] == "card ran hot"
        and events[0]["window_from"] == "2026-10-01T00:00:00Z"
        and "confirmed" not in events[0]
    )
    ((argv, kw),) = runner.calls
    assert argv[-4:] == ["--from-bundle", str(tmp_path / "b"), "--uploader", "amy@example.com"] and argv[1:4] == [
        "-I",
        "-m",
        "spyre_clickhouse_ingest",
    ]
    assert kw["env"].get("PYTHONPATH") is None


def test_bundle_only_stops_before_the_ingest(capsys, tmp_path):
    rc, be, runner, out = run(
        ["--run", RUN, *CODE, "--apply", "--bundle-only", "--out", str(tmp_path / "b")], capsys=capsys
    )
    assert rc == 0 and len(be.bundles) == 1 and runner.calls == [] and "not ingested" in out.out


def test_restore_and_set_take_their_scope_from_the_stored_invalidation(capsys):
    rc, be, runner, _ = run(["restore", INV, *CODE, "--apply"], capsys=capsys)
    assert (
        rc == 0
        and be.bundles[0][1][0]["state"] == "restored"
        and be.bundles[0][1][0]["scope"] == {"device_serial": "SN1"}
    )
    rc, be, _, _ = run(["set", INV, "--state", "suspect", *CODE, "--apply"], capsys=capsys)
    assert be.bundles[0][1][0]["state"] == "suspect"
    rc, *_, out = run(["restore", "018f0000-0000-7000-8000-0000000000ff", *CODE], capsys=capsys)
    assert rc == 2 and "no invalidation" in out.err


def test_a_noop_is_reported_and_writes_nothing(capsys):
    rc, be, runner, out = run(["--device", "SN1", *CODE, "--apply"], backend=Backend(p={"noop": True}), capsys=capsys)
    assert rc == 0 and be.bundles == [] and runner.calls == [] and "nothing to do" in out.out


def test_above_the_threshold_apply_needs_the_confirmation(capsys):
    big = Backend(p={"needs_confirm": True, "matched_runs": 80})
    rc, be, runner, out = run(["--node", "n1", *CODE, "--apply"], backend=big, capsys=capsys)
    assert rc == 2 and be.bundles == [] and "--confirm abc123abc123" in out.err
    rc, be, runner, out = run(
        ["--node", "n1", *CODE, "--apply", "--confirm", "000000000000"], backend=big, capsys=capsys
    )
    assert rc == 2 and be.bundles == [] and "does not match" in out.err
    rc, be, runner, out = run(
        ["--node", "n1", *CODE, "--apply", "--confirm", "abc123abc123"], backend=big, capsys=capsys
    )
    assert (
        rc == 0 and be.bundles[0][1][0]["confirmed"] is True and be.bundles[0][1][0]["confirm_token"] == "abc123abc123"
    )
    rc, be, *_ = run(["--node", "n1", *CODE, "--apply", "--confirm"], backend=big, capsys=capsys)
    assert rc == 0 and be.bundles[0][1][0]["confirmed"] is True


def test_the_preview_of_a_big_scope_says_what_to_pass(capsys):
    rc, *_, out = run(["--node", "n1", *CODE], backend=Backend(p={"needs_confirm": True}), capsys=capsys)
    assert rc == 0 and "confirmation required" in out.out and "--confirm abc123abc123" in out.out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--device", "SN1"], "--reason-code is required"),
        (["--device", "SN1", "--reason-code", "faulty_hardware"], "--reason is required"),
        (["--device", "SN1", "--reason-code", "faulty_hardware", "--reason", "  "], "--reason is required"),
        (["--device", "SN1", "--reason-code", "nope", "--reason", "x"], "--reason-code is required"),
        ([*CODE], "exactly one scope"),
        (["--device", "SN1", "--node", "n1", *CODE], "exactly one scope"),
        (["--benchmark", BENCH, *CODE], "exactly one scope"),
        (["--run", RUN, "--repeat", "1", *CODE], "--repeat needs"),
        (["--run", RUN, "--from", "2026-01-01", *CODE], "--from/--to bound"),
        (["--device", "SN1", "--state", "restored", *CODE], "use `restore`"),
        (["--device", "SN1", *CODE, "--apply", "--dry-run"], "exclude each other"),
        (["restore", *CODE], "needs the invalidation id"),
        (["show"], "needs the invalidation id"),
        ([INV, "--device", "SN1", *CODE], "invalid choice"),
    ],
)
def test_bad_flags_are_refused_before_the_store_is_read(argv, message, capsys):
    be = Backend()
    with pytest.raises(SystemExit) if message == "invalid choice" else _nullcontext():
        rc, *_ = run(argv, backend=be, capsys=None)
    if message != "invalid choice":
        assert rc == 2 and be.previews == []
        assert message in capsys.readouterr().err


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_run_benchmark_repeat_scope_passes_all_three(capsys):
    _, be, *_ = run(["--run", RUN, "--benchmark", BENCH, "--repeat", "2", *CODE], capsys=capsys)
    assert be.previews[0]["scope_kind"] == "run_benchmark" and be.previews[0]["scope"] == {
        "run_id": RUN,
        "benchmark_id": BENCH,
        "repeat": 2,
    }


def test_the_ingest_failing_is_reported(capsys):
    rc, *_, out = run(
        ["--device", "SN1", *CODE, "--apply"],
        runner=Runner(rc=1, out='{"status": "failed", "reason": "down"}'),
        capsys=capsys,
    )
    assert rc == 5 and "down" in out.err
    rc, *_ = run(
        ["--device", "SN1", *CODE, "--apply"],
        runner=Runner(rc=2, out='{"status": "rejected", "reason": "no"}'),
        capsys=capsys,
    )
    assert rc == 2


def test_a_refusal_from_the_backend_is_exit_2(capsys):
    rc, *_, out = run(
        ["--device", "SN1", *CODE], backend=Backend(refuse="invalidated cannot become suspect"), capsys=capsys
    )
    assert rc == 2 and "cannot become" in out.err


def test_list_and_show(capsys):
    rc, *_, out = run(["list", "--json"], capsys=capsys)
    assert rc == 0 and json.loads(out.out)[0]["invalidation_id"] == INV
    rc, *_, out = run(["show", INV], capsys=capsys)
    assert rc == 0 and "covers 1 runs" in out.out and "card" not in out.err
    rc, *_ = run(["show", "018f0000-0000-7000-8000-0000000000ff"], capsys=capsys)
    assert rc == 2


def test_without_the_ingest_the_message_says_how_to_install_it(monkeypatch, capsys):
    def gone():
        raise ingest_adapter.IngestUnavailable("not importable")

    monkeypatch.setattr(ingest_adapter, "invalidation", gone)
    assert invalidate.main(["--device", "SN1", *CODE], st=Store()) == 2
    assert "invalidate needs the ingest" in capsys.readouterr().err


def test_the_reason_codes_are_the_ingests():
    be = pytest.importorskip("spyre_clickhouse_ingest.bench_invalidation")
    assert tuple(be.REASON_CODES) == invalidate.REASON_CODES


def test_the_stdout_report_of_a_real_ingest_run_is_what_we_parse():
    rc, report = invalidate.ingest_bundle(
        ".", "amy", runner=Runner(out='noise\n{"status": "ingested", "events": []}\n')
    )
    assert rc == 0 and report["status"] == "ingested"
    assert invalidate.INGEST_ARGV[0] == sys.executable


def test_the_ingest_writes_to_the_database_the_preview_read(monkeypatch):
    monkeypatch.setenv("BENCH_CLICKHOUSE_DB", "sandbox_me")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "other_db")
    monkeypatch.setenv("BENCH_INVALIDATION_CONFIRM_ABOVE", "500")
    r = Runner()
    invalidate.ingest_bundle(".", "amy", runner=r)
    env = r.calls[0][1]["env"]
    assert env["CLICKHOUSE_DB_V2"] == "sandbox_me" and env["BENCH_INVALIDATION_CONFIRM_ABOVE"] == "500"


def test_a_timeout_or_a_statusless_report_is_not_success(capsys):
    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)

    rc, *_ = run(["--device", "SN1", *CODE, "--apply"], runner=slow, capsys=capsys)
    assert rc == 5
    rc, *_ = run(["--device", "SN1", *CODE, "--apply"], runner=Runner(out="{}"), capsys=capsys)
    assert rc == 5


def test_json_output_stays_one_document(capsys):
    rc, *_, out = run(["--device", "SN1", *CODE, "--json"], capsys=capsys)
    assert rc == 0 and json.loads(out.out)["invalidation_id"] == INV


def test_apply_asks_the_ingest_for_the_lab_channel(capsys, tmp_path):
    _, _, runner, _ = run(["--node", "n1", *CODE, "--apply", "--out", str(tmp_path / "b")], capsys=capsys)
    argv = runner.calls[0][0]
    assert argv[argv.index("--channel-trust") + 1] == "lab"
    assert argv.index("--channel-trust") < argv.index("--from-bundle")


def test_the_event_and_the_channel_are_accepted_by_the_real_ingest(capsys, tmp_path, monkeypatch):
    """The bundle the verb writes, ingested with the trust the verb asks for, through the ingest's own writer."""
    be = pytest.importorskip("spyre_clickhouse_ingest.bench_invalidation")
    pytest.importorskip("chdb.session")
    from pathlib import Path

    ingest_tests = Path(__file__).resolve().parents[2] / "clickhouse-ingest" / "tests"
    if not (ingest_tests / "bench_chdb.py").exists():
        pytest.skip("the ingest's chdb test helpers are not in this tree")
    sys.path.insert(0, str(ingest_tests))
    try:
        from bench_chdb import ChdbClient, Db
        from spyre_clickhouse_ingest import bundle, offline
        from spyre_clickhouse_ingest import client as client_mod
    finally:
        sys.path.remove(str(ingest_tests))
    _, fake, runner, _ = run(
        ["--node", "n1", "--from", "2026-10-01T00:00:00Z", *CODE, "--apply", "--out", str(tmp_path / "b")],
        capsys=capsys,
    )
    argv = runner.calls[0][0]
    trust = argv[argv.index("--channel-trust") + 1]
    ((_, events, _),) = fake.bundles
    out = be.write_bundle(tmp_path / "real", events, uploader="amy@example.com")
    db = Db()
    c = ChdbClient(db)
    monkeypatch.setattr(client_mod.ClickHouse, "connect", staticmethod(lambda **k: c))
    monkeypatch.setattr(client_mod.ClickHouseEnv, "target_database", staticmethod(lambda: "default"))
    try:
        from types import SimpleNamespace

        args = SimpleNamespace(
            from_bundle=str(out), dry_run=False, uploader="amy@example.com", bundle_prop=[], channel_trust="",
            expect_key="", trusted_job_prefix=[], append_to_run=False,
        )  # fmt: skip
        with offline.opened(out) as root:
            with pytest.raises(offline.BundleError, match="not none"):
                bundle._ingest(root, args)
            args.channel_trust = trust
            assert bundle._ingest(root, args)["status"] == "ingested"
    finally:
        db.s.close()

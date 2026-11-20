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

"""relay.py against an in-memory Artifactory and a scripted `results --from-bundle`."""

import json
import regex as re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from spyre_clickhouse_ingest import relay

AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
relay.ROOT = "next/{arch}/v2-results"
ROOT = "next/s390x/v2-results"


class FakeArtifactory(relay.Artifactory):
    """Files by full path; AQL answered by the two shapes the relay sends."""

    def __init__(self, files):
        super().__init__("https://art/artifactory", "repo", "t")
        self.files = dict(files)  # path -> (bytes, created, created_by)
        self.moves, self.puts, self.props = [], [], {}

    def find(self, criteria):
        out = []
        for path, (_, created, by) in self.files.items():
            folder, _, name = path.rpartition("/")
            item = {
                "path": folder,
                "name": name,
                "type": "file",
                "created": created,
                "created_by": by,
                "size": len(self.files[path][0]),
                "actual_sha1": str(hash(self.files[path][0])),
            }
            if "$or" in criteria and "type" in criteria:  # one folder's files
                f = criteria["$or"][0]["path"]
                if folder == f or folder.startswith(f + "/"):
                    out.append(item)
            elif name == "bundle.json" or name.endswith((".tgz", ".tar.gz")):
                out.append(item)
        return out

    def prop(self, path, name):
        return self.props.get((path, name), "")

    def set_prop(self, path, name, value):
        self.props[(path, name)] = value

    def exists(self, path):
        return any(p == path or p.startswith(path + "/") for p in self.files)

    def download(self, path, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[path][0])

    def put(self, path, data):
        self.puts.append(path)
        self.files[path] = (data, NOW.isoformat(), "relay")

    def move(self, src, dst):
        self.moves.append((src, dst))
        for p in [p for p in self.files if p == src or p.startswith(src + "/")]:
            self.files[dst + p[len(src) :]] = self.files.pop(p)


def _args(**kw):
    base = {
        "apply": True,
        "inbox": "inbox",
        "processed": "processed",
        "rejected": "rejected",
        "trusted_job_prefix": ["ci-jobs/nightly/"],
        "ci_account": [],
        "stale_hours": 24.0,
    }
    return SimpleNamespace(**{**base, **kw})


def _dir(folder, created="2026-10-09T11:00:00.000Z", by="jdoe", meta=b'{"files": []}'):
    return {f"{folder}/bundle.json": (meta, created, by),
            f"{folder}/results/junit.xml": (b"<testsuites/>", created, by)}  # fmt: skip


def _relay(files, outcomes, tmp_path, **kw):
    """Run once; `outcomes` maps a bundle's local name to (code, report)."""
    art = FakeArtifactory(files)
    calls = []

    def ingest(local, argv):
        calls.append((local, argv))
        return outcomes.get(
            local.name if local.suffix else "dir", (0, {"status": "ingested"})
        )

    entries = relay.Relay(art, _args(**kw), ingest=ingest, now=NOW).run(tmp_path)
    return art, entries, calls


def test_an_ingested_folder_moves_to_processed_with_its_url_as_bundle_url(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/manual_jdoe_x-fvt"
    art, entries, calls = _relay(_dir(folder), {}, tmp_path)
    dst = f"{ROOT}/processed/{AID}/manual_jdoe_x-fvt"
    assert art.moves == [(folder, dst)] and entries[0]["outcome"] == "ingested"
    local, argv = calls[0]
    assert (local / "results" / "junit.xml").read_bytes() == b"<testsuites/>"
    flags = dict(zip(argv[1::2], argv[2::2]))
    assert flags["--uploader"] == "jdoe" and flags["--expect-key"] == AID
    assert flags["--bundle-url"] == f"https://art/artifactory/repo/{dst}"
    assert (
        flags["--trusted-job-prefix"] == "ci-jobs/nightly/" and "--dry-run" not in argv
    )


def test_a_rejected_bundle_moves_with_its_reason(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    art, entries, _ = _relay(
        _dir(folder),
        {"dir": (2, {"status": "rejected", "reason": "sha256 mismatch"})},
        tmp_path,
    )
    dst = f"{ROOT}/rejected/{AID}/b-fvt"
    assert art.moves == [(folder, dst)] and art.puts == [f"{dst}/REJECTED.json"]
    assert (
        json.loads(art.files[f"{dst}/REJECTED.json"][0])["reason"] == "sha256 mismatch"
    )
    assert entries[0]["outcome"] == "rejected"


@pytest.mark.parametrize("code, created, outcome, moved", [
    (3, "2026-10-09T11:00:00Z", "waiting", False),   # still uploading
    (3, "2026-10-07T11:00:00Z", "rejected", True),   # incomplete for over a day
    (1, "2026-10-01T11:00:00Z", "failed", False),    # ClickHouse down: never the bundle's fault
])  # fmt: skip
def test_unfinished_bundles_wait_unless_stale(tmp_path, code, created, outcome, moved):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    art, entries, _ = _relay(
        _dir(folder, created), {"dir": (code, {"reason": "x"})}, tmp_path
    )
    assert entries[0]["outcome"] == outcome and bool(art.moves) == moved


def test_a_tgz_is_ingested_as_a_file(tmp_path):
    path = f"{ROOT}/inbox/{AID}/b-fvt.tgz"
    files = {path: (b"tgz", "2026-10-09T11:00:00Z", "jdoe")}
    art, _, calls = _relay(files, {}, tmp_path)
    assert calls[0][0].name == "b-fvt.tgz"
    assert art.moves == [(path, f"{ROOT}/processed/{AID}/b-fvt.tgz")]


def test_processed_rejected_and_other_trees_are_not_inboxes(tmp_path):
    files = {**_dir(f"{ROOT}/processed/{AID}/a-fvt"), **_dir(f"{ROOT}/rejected/{AID}/b-fvt"),
             **_dir(f"{ROOT}/inbox-test/inbox/{AID}/c-fvt")}  # fmt: skip
    assert _relay(files, {}, tmp_path)[1] == []
    art, _, _ = _relay(files, {}, tmp_path, inbox="inbox-test/inbox",
                             processed="inbox-test/processed")  # fmt: skip
    assert art.moves == [
        (
            f"{ROOT}/inbox-test/inbox/{AID}/c-fvt",
            f"{ROOT}/inbox-test/processed/{AID}/c-fvt",
        )
    ]


def test_a_path_without_an_artifact_id_is_rejected_unread(tmp_path):
    folder = f"{ROOT}/inbox/not-an-id/b-fvt"
    art, entries, calls = _relay(_dir(folder), {}, tmp_path)
    assert calls == [] and entries[0]["outcome"] == "rejected"
    assert art.moves == [(folder, f"{ROOT}/rejected/not-an-id/b-fvt")]


def test_a_name_already_taken_downstream_gets_a_suffix(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    files = {**_dir(folder), **_dir(f"{ROOT}/processed/{AID}/b-fvt")}
    art, _, _ = _relay(files, {}, tmp_path)
    assert art.moves == [(folder, f"{ROOT}/processed/{AID}/b-fvt-dup20261009T120000Z")]


def test_dry_run_moves_nothing(tmp_path):
    art, entries, calls = _relay(
        _dir(f"{ROOT}/inbox/{AID}/b-fvt"),
        {"dir": (2, {"reason": "x"})},
        tmp_path,
        apply=False,
    )
    assert "--dry-run" in calls[0][1] and art.moves == [] and art.puts == []
    assert entries[0]["outcome"] == "rejected"


def test_an_attachment_tgz_is_not_a_bundle_of_its_own(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    files = {
        **_dir(folder),
        f"{folder}/attachments/logs.tgz": (b"x", "2026-10-09T11:00:00Z", "jdoe"),
    }
    _, entries, calls = _relay(files, {}, tmp_path)
    assert len(calls) == 1 and len(entries) == 1


def test_bundles_are_found_under_root_only():
    item = {
        "name": "b-fvt.tgz",
        "created": "2026-10-09T11:00:00Z",
        "created_by": "jdoe",
    }
    found = relay.bundles(
        [
            {**item, "path": f"{ROOT}/inbox/{AID}"},
            {**item, "path": f"other/{ROOT}/inbox/{AID}"},
        ],
        "inbox",
    )
    assert [(b["arch"], b["key"], b["rel"]) for b in found] == [
        ("s390x", AID, f"{AID}/b-fvt.tgz")
    ]


def test_an_image_only_bundle_is_filed_under_its_digest(tmp_path):
    key = "sha256-" + "b6" * 32
    folder = f"{ROOT}/inbox/{key}/b-fvt"
    art, _, calls = _relay(_dir(folder), {}, tmp_path)
    argv = calls[0][1]
    assert argv[argv.index("--expect-key") + 1] == key
    assert art.moves == [(folder, f"{ROOT}/processed/{key}/b-fvt")]
    assert not relay.is_key("sha256-" + "b6" * 31) and not relay.is_key("b6" * 32)


def test_a_folder_without_files_waits_until_its_listing_is_unchanged(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    art = FakeArtifactory(_dir(folder, meta=b'{"test_type": "fvt"}'))
    calls = []

    def ingest(local, argv):
        calls.append(local)
        return 0, {"status": "ingested"}

    def run():
        return relay.Relay(art, _args(), ingest=ingest, now=NOW).run(
            tmp_path / str(len(calls))
        )

    assert run()[0]["outcome"] == "waiting" and calls == []
    art.files[f"{folder}/results/more.xml"] = (
        b"<testsuites/>",
        NOW.isoformat(),
        "jdoe",
    )
    assert run()[0]["outcome"] == "waiting" and calls == []  # it changed: wait again
    assert run()[0]["outcome"] == "ingested"
    assert (calls[0] / "results" / "more.xml").exists() and art.moves


def test_a_folder_still_changing_after_a_day_is_rejected(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    files = _dir(folder, created="2026-10-07T11:00:00Z", meta=b'{"test_type": "fvt"}')
    _, entries, calls = _relay(files, {}, tmp_path)
    assert entries[0]["outcome"] == "rejected" and calls == []


def test_a_dry_run_records_no_listing(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    art, entries, _ = _relay(_dir(folder, meta=b"{}"), {}, tmp_path, apply=False)
    assert entries[0]["outcome"] == "waiting" and art.props == {}


def test_an_exit_without_a_report_is_a_retry_not_a_rejection(tmp_path, monkeypatch):
    """argparse exits 2 on a library without --from-bundle; that must not reject bundles."""
    done = SimpleNamespace(
        returncode=2, stdout="", stderr="error: unrecognized arguments: --from-bundle"
    )
    monkeypatch.setattr(relay.subprocess, "run", lambda *a, **kw: done)
    code, report = relay.run_ingest(tmp_path, [])
    assert code == relay.FAILED and "unrecognized arguments" in report["reason"]


# --- result_uploads rows -----------------------------------------------------------------------

RID = "e4b1dc2a-985b-5295-9aaf-38cc48cfe00e"
SHA = "ab" * 32
REPORT = {"status": "ingested", "run_id": RID, "artifact_id": AID, "test_type": "fvt", "arch": "s390x",
          "bundle_sha256": SHA, "run_key": f"manual:jdoe:{SHA}"}  # fmt: skip
UUID7 = r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"  # the dashboard's _UUID_RE


class FakeClickHouse:
    """Answers the statements UploadLog sends; `order` is shared with the Artifactory moves."""

    def __init__(
        self, order, bundle_kind="junit", verdict=True, result_kind="functional"
    ):
        self.order, self.rows, self.fail = order, [], False
        self.verdict = [[result_kind, bundle_kind]] if verdict else []

    def query(self, sql, parameters=None):
        if "result_uploads" in sql:
            n = sum(1 for r in self.rows if str(r["run_id"]) == parameters["rid"])
            return SimpleNamespace(result_rows=[[n]])
        assert "FROM spy.artifact_results" in sql and "state != 'running'" in sql
        return SimpleNamespace(result_rows=self.verdict)

    def command(self, sql, parameters=None, settings=None):
        assert (
            sql.startswith("INSERT INTO spy.result_uploads")
            and "WHERE NOT EXISTS" in sql
        )
        if self.fail:
            raise OSError("INSERT denied")
        if not any(str(r["run_id"]) == parameters["run_id"] for r in self.rows):
            self.order.append("row")
            props = {
                parameters[f"pk{n}"]: parameters[f"pv{n}"]
                for n in range(8)
                if f"pk{n}" in parameters
            }
            self.rows.append({**parameters, "props": props})


class OrderedArtifactory(FakeArtifactory):
    def __init__(self, files, order):
        super().__init__(files)
        self.order = order

    def move(self, src, dst):
        self.order.append("move")
        super().move(src, dst)


def _recording(files, tmp_path, outcome=(0, REPORT), client=None, order=None, **kw):
    order = [] if order is None else order
    art = OrderedArtifactory(files, order)
    client = client or FakeClickHouse(order)
    uploads = relay.UploadLog(client, "spy")
    entries = relay.Relay(art, _args(**kw), ingest=lambda local, argv: outcome, now=NOW,
                          uploads=uploads).run(tmp_path)  # fmt: skip
    return art, entries, client, order


def test_an_ingested_bundle_gets_one_published_row_before_it_moves(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/manual_jdoe_x-fvt"
    _, entries, ch, order = _recording(_dir(folder), tmp_path)
    (row,) = ch.rows
    assert order == ["row", "move"] and entries[0]["upload"] == "recorded"
    assert (
        row["state"] == "published" and row["actor"] == "jdoe" and row["reason"] == ""
    )
    assert (str(row["run_id"]), str(row["artifact_id"])) == (RID, AID)
    assert (row["format"], row["result_kind"], row["test_type"], row["arch"]) == (
        "junit",
        "functional",
        "fvt",
        "s390x",
    )
    assert row["content_sha256"] == SHA and row["filename"] == "manual_jdoe_x-fvt"
    assert (
        row["bundle_url"]
        == f"https://art/artifactory/repo/{ROOT}/processed/{AID}/manual_jdoe_x-fvt"
    )
    assert row["size_bytes"] == len(b'{"files": []}') + len(b"<testsuites/>")
    assert row["props"] == {
        "channel": "relay",
        "trust": "external",
        "ingest_status": "ingested",
        "bundle_key": AID,
    }
    assert re.match(UUID7, str(row["upload_id"]))


def test_a_tgz_row_names_the_file_and_its_size(tmp_path):
    path = f"{ROOT}/inbox/{AID}/b-fvt.tgz"
    _, _, ch, _ = _recording(
        {path: (b"tgz!", "2026-10-09T11:00:00Z", "jdoe")}, tmp_path
    )
    assert (ch.rows[0]["filename"], ch.rows[0]["size_bytes"]) == ("b-fvt.tgz", 4)


def test_the_bundle_kind_on_the_verdict_is_the_rows_format(tmp_path):
    order = []
    _, _, ch, _ = _recording(_dir(f"{ROOT}/inbox/{AID}/b"), tmp_path, order=order,
                             client=FakeClickHouse(order, bundle_kind="vllm"))  # fmt: skip
    assert ch.rows[0]["format"] == "vllm"


def test_trust_comes_from_the_uploading_account_never_from_the_run_key(tmp_path):
    spoofed = {
        **REPORT,
        "run_key": "ci-jobs/nightly/main#12",
    }  # the bundle claims a CI job
    folder = f"{ROOT}/inbox/{AID}/b"
    _, _, ch, _ = _recording(_dir(folder), tmp_path / "1", (0, spoofed))
    assert ch.rows[0]["props"]["trust"] == "external"
    _, _, ch, _ = _recording(
        _dir(folder), tmp_path / "2", (0, spoofed), ci_account=["other"]
    )
    assert ch.rows[0]["props"]["trust"] == "external"
    _, _, ch, _ = _recording(
        _dir(folder), tmp_path / "3", (0, REPORT), ci_account=["other", "jdoe"]
    )
    assert ch.rows[0]["props"]["trust"] == "ci"  # a listed account, with a manual: key


def test_trust_of_treats_an_unknown_uploader_as_external():
    assert relay.trust_of("", [""]) == "external" and relay.trust_of("a", ["a"]) == "ci"


def test_an_uploader_the_store_does_not_name_is_recorded_as_unknown(tmp_path):
    files = _dir(f"{ROOT}/inbox/{AID}/b", by="")
    _, _, ch, _ = _recording(files, tmp_path)
    assert ch.rows[0]["actor"] == "unknown"


def test_a_duplicate_of_a_recorded_run_writes_no_second_row(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    order = []
    ch = FakeClickHouse(order)
    for n, status in enumerate(("ingested", "duplicate")):
        art, entries, _, _ = _recording(_dir(folder), tmp_path / str(n), (0, {**REPORT, "status": status}),
                                        client=ch, order=order)  # fmt: skip
        assert art.moves  # each is moved out of the inbox
    assert (
        len(ch.rows) == 1
        and entries[0]["outcome"] == "duplicate"
        and entries[0]["upload"] == "exists"
    )


def test_a_duplicate_with_no_row_yet_gets_one(tmp_path):
    """A crash between the ingest and the row, or a bundle ingested before rows existed."""
    _, _, ch, _ = _recording(
        _dir(f"{ROOT}/inbox/{AID}/b"), tmp_path, (0, {**REPORT, "status": "duplicate"})
    )
    assert len(ch.rows) == 1 and ch.rows[0]["props"]["ingest_status"] == "duplicate"


@pytest.mark.parametrize("outcome, label", [
    ((2, {"status": "rejected", "reason": "bad"}), "rejected"),
    ((1, {"reason": "ClickHouse down"}), "failed"),
    ((3, {"reason": "file missing"}), "waiting"),
])  # fmt: skip
def test_a_bundle_that_was_not_ingested_writes_no_row(tmp_path, outcome, label):
    _, entries, ch, order = _recording(_dir(f"{ROOT}/inbox/{AID}/b"), tmp_path, outcome)
    assert entries[0]["outcome"] == label and ch.rows == [] and "row" not in order


def test_a_path_rejected_unread_writes_no_row(tmp_path):
    _, entries, ch, _ = _recording(_dir(f"{ROOT}/inbox/not-an-id/b"), tmp_path)
    assert entries[0]["outcome"] == "rejected" and ch.rows == []


def test_a_dry_run_records_nothing(tmp_path):
    """main() builds no UploadLog without --apply; one handed in is still never called."""
    _, _, ch, order = _recording(_dir(f"{ROOT}/inbox/{AID}/b"), tmp_path, apply=False)
    assert ch.rows == [] and order == []


def test_a_failed_row_write_keeps_the_bundle_in_the_inbox_until_it_lands(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    order = []
    ch = FakeClickHouse(order)
    ch.fail = True
    art, entries, _, _ = _recording(
        _dir(folder), tmp_path / "1", client=ch, order=order
    )
    assert (
        entries[0]["outcome"] == "unrecordable"
        and "INSERT denied" in entries[0]["reason"]
    )
    assert art.moves == [] and ch.rows == []
    ch.fail = False  # next poll: the ingest answers duplicate, the row lands, the bundle moves
    art, entries, _, _ = _recording(_dir(folder), tmp_path / "2", (0, {**REPORT, "status": "duplicate"}),
                                    client=ch, order=order)  # fmt: skip
    assert (
        len(ch.rows) == 1
        and len(art.moves) == 1
        and entries[0]["outcome"] == "duplicate"
    )


def test_a_verdict_the_ingest_did_not_leave_is_a_retry(tmp_path):
    order = []
    art, entries, _, _ = _recording(_dir(f"{ROOT}/inbox/{AID}/b"), tmp_path, order=order,
                                    client=FakeClickHouse(order, verdict=False))  # fmt: skip
    assert (
        entries[0]["outcome"] == "unrecordable"
        and "no fvt verdict" in entries[0]["reason"]
        and art.moves == []
    )


def test_the_upload_id_is_fixed_by_the_run_and_shaped_like_a_uuid7():
    other = "0a1b2c3d-985b-5295-9aaf-38cc48cfe00e"
    assert (
        relay.upload_id_of(RID) == relay.upload_id_of(RID) != relay.upload_id_of(other)
    )
    assert re.match(UUID7, relay.upload_id_of(RID)) and re.match(
        UUID7, relay.upload_id_of(other)
    )


def test_a_verdict_without_a_bundle_kind_is_recorded_as_junit_and_its_result_kind_is_kept(
    tmp_path,
):
    order = []
    ch = FakeClickHouse(order, bundle_kind="", result_kind="performance")
    _recording(_dir(f"{ROOT}/inbox/{AID}/b"), tmp_path, client=ch, order=order)
    assert (ch.rows[0]["format"], ch.rows[0]["result_kind"]) == ("junit", "performance")


class ExitingClickHouse(FakeClickHouse):
    def command(self, sql, parameters=None, settings=None):
        raise SystemExit("CLICKHOUSE_PORT is not a number: 'x'")


def test_a_connect_that_exits_is_an_unrecordable_bundle_not_a_dead_relay(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    files = {
        **_dir(folder),
        **_dir(f"{ROOT}/inbox/{AID}/c-fvt", created="2026-10-09T11:30:00Z"),
    }
    order = []
    art, entries, _, _ = _recording(
        files, tmp_path, client=ExitingClickHouse(order), order=order
    )
    assert [e["outcome"] for e in entries] == [
        "unrecordable",
        "unrecordable",
    ]  # the second bundle was still tried
    assert "CLICKHOUSE_PORT" in entries[0]["reason"] and art.moves == []


# --- main() wiring -----------------------------------------------------------------------------


@pytest.fixture
def wired(monkeypatch, tmp_path):
    seen = {}

    class SpyRelay:
        def __init__(self, art, args, uploads=None):
            seen.update(args=args, uploads=uploads)

        def run(self, work):
            seen["ran"] = "run"
            return []

        def backfill(self, work):
            seen["ran"] = "backfill"
            return []

    monkeypatch.setattr(relay, "Relay", SpyRelay)
    monkeypatch.setattr(relay, "UploadLog", lambda: "log")
    monkeypatch.setenv("ARTIFACTORY_TOKEN", "t")
    monkeypatch.setenv("ARTIFACTORY_BASE_URL", "https://art.example.com/artifactory")
    monkeypatch.setenv("ARTIFACTORY_V2_RESULTS_REPO", "results")
    monkeypatch.setenv("ARTIFACTORY_V2_RESULTS_ROOT", "next/{arch}/v2-results")

    def run(*argv):
        seen.clear()
        relay.main(
            [
                *argv,
                "--work",
                str(tmp_path / "w"),
                "--summary",
                str(tmp_path / "s.json"),
            ]
        )
        return seen

    return run


def test_main_hands_the_upload_log_only_to_an_applying_relay(wired):
    assert wired("--apply")["uploads"] == "log"
    assert wired()["uploads"] is None
    assert wired("--apply", "--no-record-uploads")["uploads"] is None


def test_main_passes_the_ci_accounts_and_picks_the_pass(wired):
    seen = wired("--apply", "--ci-account", "svc-a", "--ci-account", "svc-b")
    assert seen["args"].ci_account == ["svc-a", "svc-b"] and seen["ran"] == "run"
    assert wired("--apply")["args"].ci_account == []
    seen = wired("--backfill-processed")  # reads the log even as a dry run
    assert seen["ran"] == "backfill" and seen["uploads"] == "log"


def test_a_backfill_without_recording_is_refused(wired):
    with pytest.raises(SystemExit):
        wired("--backfill-processed", "--no-record-uploads")


# --- backfill of processed/ ----------------------------------------------------------------------


def _backfill(files, tmp_path, report=None, code=0, client=None, **kw):
    report = report or {**REPORT, "status": "duplicate"}
    order = []
    art = OrderedArtifactory(files, order)
    client = client or FakeClickHouse(order)
    calls = []

    def ingest(local, argv):
        calls.append(argv)
        return code, report

    entries = relay.Relay(art, _args(**kw), ingest=ingest, now=NOW,
                          uploads=relay.UploadLog(client, "spy")).backfill(tmp_path)  # fmt: skip
    return art, entries, client, calls


def test_backfill_writes_the_missing_row_of_a_processed_bundle_and_moves_nothing(
    tmp_path,
):
    folder = f"{ROOT}/processed/{AID}/b-fvt"
    art, entries, ch, calls = _backfill(
        {**_dir(folder), **_dir(f"{ROOT}/inbox/{AID}/waiting")}, tmp_path
    )
    assert [e["outcome"] for e in entries] == ["recorded"] and art.moves == []
    assert "--dry-run" in calls[0] and calls[0][
        calls[0].index("--bundle-url") + 1
    ].endswith(folder)
    (row,) = ch.rows
    assert (
        row["props"]["backfilled"] == "1"
        and row["bundle_url"].endswith(folder)
        and row["actor"] == "jdoe"
    )


def test_backfill_is_a_no_op_for_a_run_with_a_row_and_for_a_dry_run(tmp_path):
    files = _dir(f"{ROOT}/processed/{AID}/b-fvt")
    order = []
    ch = FakeClickHouse(order)
    _backfill(files, tmp_path / "1", client=ch)
    assert (
        _backfill(files, tmp_path / "2", client=ch)[1][0]["outcome"] == "exists"
        and len(ch.rows) == 1
    )
    _, entries, ch2, _ = _backfill(files, tmp_path / "3", apply=False)
    assert entries[0]["outcome"] == "would-backfill" and ch2.rows == []


@pytest.mark.parametrize(
    "code, report",
    [(0, {"status": "would-ingest"}), (2, {"status": "rejected", "reason": "x"})],
)
def test_backfill_skips_what_the_ingest_does_not_know_as_recorded(
    tmp_path, code, report
):
    _, entries, ch, _ = _backfill(
        _dir(f"{ROOT}/processed/{AID}/b-fvt"), tmp_path, report, code
    )
    assert entries[0]["outcome"] == "skipped" and ch.rows == []


@pytest.mark.parametrize(
    "extra_writer, want",
    [("someone-else", "external"), ("", "external"), ("jdoe", "ci")],
)
def test_backfill_records_the_lowest_trust_of_every_account_that_wrote_the_folder(
    tmp_path, extra_writer, want
):
    folder = f"{ROOT}/processed/{AID}/b-fvt"
    files = {
        **_dir(folder),
        f"{folder}/attachments/logs.tgz": (b"x", "2026-10-09T11:00:00Z", extra_writer),
    }
    _, _, ch, calls = _backfill(files, tmp_path, ci_account=["jdoe"])
    argv = calls[0]
    assert ch.rows[0]["props"]["trust"] == want
    assert argv[argv.index("--channel-trust") + 1] == want


def test_backfill_reads_a_tgz_and_survives_a_bad_row(tmp_path):
    path = f"{ROOT}/processed/{AID}/b-fvt.tgz"
    files = {path: (b"tgz!", "2026-10-09T11:00:00Z", "jdoe")}
    order = []
    ch = FakeClickHouse(order)
    ch.fail = True
    _, entries, _, _ = _backfill(files, tmp_path, client=ch)
    assert entries[0]["outcome"] == "unrecordable"


def test_the_ingest_is_told_the_trust_of_the_uploading_account(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    for accounts, want in (([], "external"), (["jdoe"], "ci")):
        _, _, calls = _relay(_dir(folder), {}, tmp_path, ci_account=accounts)
        argv = calls[0][1]
        assert argv[argv.index("--channel-trust") + 1] == want


def test_main_needs_the_deployments_location(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTORY_TOKEN", "t")
    for var in (
        "ARTIFACTORY_BASE_URL",
        "ARTIFACTORY_V2_RESULTS_REPO",
        "ARTIFACTORY_V2_RESULTS_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit, match="ARTIFACTORY_V2_RESULTS_ROOT"):
        relay.main(
            ["--work", str(tmp_path / "w"), "--summary", str(tmp_path / "s.json")]
        )


def test_without_trusted_prefixes_no_ci_looking_run_key_is_accepted(wired):
    assert wired("--apply")["args"].trusted_job_prefix == [relay.NO_JOB_KEYS]
    assert wired("--apply", "--trusted-job-prefix", "jobs/")[
        "args"
    ].trusted_job_prefix == ["jobs/"]


def test_the_row_records_the_trust_the_ingest_applied(tmp_path):
    folder = f"{ROOT}/inbox/{AID}/b-fvt"
    order = []
    client = FakeClickHouse(order)
    outcome = (0, {**REPORT, "trust": "external"})
    uploads = relay.UploadLog(client, "spy")
    relay.Relay(FakeArtifactory(_dir(folder)), _args(ci_account=["jdoe"]), ingest=lambda local, argv: outcome,
                now=NOW, uploads=uploads).run(tmp_path)  # fmt: skip
    assert client.rows[0]["props"]["trust"] == "external"


def test_a_folder_is_only_ci_when_every_file_was_written_by_a_ci_account():
    assert relay.min_trust(["svc"], ["svc"]) == "ci"
    assert relay.min_trust(["svc", "jdoe"], ["svc"]) == "external"


def test_only_live_upload_events_stop_the_row():
    sqls = []

    class Spy:
        def query(self, sql, parameters=None):
            sqls.append(sql)
            return SimpleNamespace(result_rows=[[0]])

        def command(self, sql, parameters=None, settings=None):
            sqls.append(sql)
            assert settings == {"select_sequential_consistency": 1}

    log = relay.UploadLog(Spy(), "spy")
    log.has(RID)
    log.append(upload_id="u", state="published", actor="a", artifact_id=AID, run_id=RID, format="junit",
               result_kind="functional", test_type="fvt", arch="s390x", filename="f", size_bytes=1,
               content_sha256="c", bundle_url="u", props={"channel": "relay"})  # fmt: skip
    assert all("state IN ('published', 'retracted')" in q for q in sqls)

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

# ruff: noqa: F811  (the imported `online` fixture is requested by name)
"""Sealing for sites with no route to the database: deterministic bundles, redaction, trust
labels, one time-zone rule, the compatibility check and the printable receipt."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_bundle import (  # noqa: F401
    AID,
    JUNIT,
    _flags,
    _ingest,
    _minimal,
    _offline,
    _rejected,
    online,
)

from spyre_clickhouse_ingest import bundle, offline, results
from spyre_clickhouse_ingest.__main__ import main as cli

NAIVE = JUNIT.replace("2026-10-08T10:00:00+00:00", "2026-10-08T10:00:00")
HOSTED = JUNIT.replace("<testsuite ", '<testsuite hostname="lab-node-7" ')


REAL_WRITE_JUNIT = bundle.write_junit


def _seal(tmp_path, name, xml, *extra):
    d = tmp_path / f"xml-{name}"
    d.mkdir()
    (d / "junit.xml").write_text(xml)
    out = tmp_path / name
    argv = ["results", "--offline", "--out", str(out), "--xml-dir", str(d),
            "--artifact", f"id:{AID}", "--arch", "s390x", "--trigger-type", "fvt", *extra]  # fmt: skip
    assert cli(argv) == 0
    return out


def _dig(path):
    with offline.opened(path) as root:
        return offline.digest(root)


# --- deterministic sealing ---------------------------------------------------------------


def test_sealing_the_same_results_twice_gives_one_digest(tmp_path):
    a = _seal(tmp_path, "a.tgz", JUNIT)
    time.sleep(1.1)
    b = _seal(tmp_path, "b.tgz", JUNIT)
    assert _dig(a) == _dig(b)
    assert a.read_bytes() == b.read_bytes()


def test_a_folder_and_its_tgz_share_a_digest(tmp_path):
    folder = _seal(tmp_path, "f", JUNIT)
    assert _dig(folder) == _dig(_seal(tmp_path, "f.tgz", JUNIT))


def test_the_seal_time_is_not_stamped_into_the_bundle(tmp_path):
    timed = JUNIT.replace('tests="2"', 'tests="2" time="5"')
    meta = json.loads((_seal(tmp_path, "m", timed) / "bundle.json").read_text())
    assert meta["started_at"] == "2026-10-08T10:00:00Z"
    assert meta["ended_at"] == "2026-10-08T10:00:05Z"  # last suite start + its time


def test_without_timestamps_no_end_time_is_invented(tmp_path):
    bare = JUNIT.replace(' timestamp="2026-10-08T10:00:00+00:00"', "")
    meta = json.loads((_seal(tmp_path, "m", bare) / "bundle.json").read_text())
    assert "started_at" not in meta and "ended_at" not in meta
    pinned = _seal(tmp_path, "p", bare, "--triggered-at", "2026-10-08T10:00:00Z",
                   "--ended-at", "2026-10-08T10:05:00Z")  # fmt: skip
    assert json.loads((pinned / "bundle.json").read_text())["ended_at"].endswith(
        "05:00Z"
    )


# --- redaction -----------------------------------------------------------------------------


def test_the_sealing_host_and_user_are_recorded_unless_redacted(tmp_path):
    runner = json.loads((_seal(tmp_path, "d", JUNIT) / "bundle.json").read_text())[
        "runner"
    ]
    assert set(runner) == {"host", "user"}
    only_user = _seal(tmp_path, "h", JUNIT, "--redact-host")
    assert set(json.loads((only_user / "bundle.json").read_text())["runner"]) == {
        "user"
    }
    both = _seal(tmp_path, "b", JUNIT, "--redact-host", "--redact-user")
    assert "runner" not in json.loads((both / "bundle.json").read_text())


def test_redacting_the_host_blanks_the_junit_hostname_and_still_validates(
    tmp_path, capsys
):
    plain = _seal(tmp_path, "p", HOSTED)
    assert "lab-node-7" in (plain / "results" / "junit.xml").read_text()
    red = _seal(tmp_path, "r", HOSTED, "--redact-host", "--redact-user")
    assert "lab-node-7" not in (red / "results" / "junit.xml").read_text()
    assert offline.check(red)["test_type"] == "fvt"
    assert _dig(red) != _dig(plain)


def test_redacted_seals_agree_across_sealing_users(tmp_path, monkeypatch):
    a = _seal(tmp_path, "a", JUNIT, "--redact-host", "--redact-user")
    monkeypatch.setattr(offline.getpass, "getuser", lambda: "someone-else")
    monkeypatch.setattr(offline.socket, "gethostname", lambda: "other-host")
    b = _seal(tmp_path, "b", JUNIT, "--redact-host", "--redact-user")
    assert _dig(a) == _dig(b)


# --- trust ---------------------------------------------------------------------------------


def test_an_offline_seal_is_external_unless_the_sealer_says_lab(tmp_path):
    assert (
        json.loads((_seal(tmp_path, "e", JUNIT) / "bundle.json").read_text())["trust"]
        == "external"
    )
    lab = _seal(tmp_path, "l", JUNIT, "--seal-trust", "lab")
    assert json.loads((lab / "bundle.json").read_text())["trust"] == "lab"


def test_a_bundle_cannot_claim_ci(tmp_path):
    root = _minimal(tmp_path, {"artifact_id": AID, "test_type": "fvt", "trust": "ci"})
    assert "'ci' is not one of" in _rejected(root)
    with pytest.raises(SystemExit):
        cli(
            ["results", "--offline", "--seal-trust", "ci", "--out", str(tmp_path / "x")]
        )


def _trust_props(online):
    argv = online.calls[0]
    return {argv[i + 1] for i, a in enumerate(argv) if a == "--result-prop"}


@pytest.mark.parametrize(
    "label, channel, recorded",
    [
        ("external", "", "external"),
        ("lab", "", "external"),
        ("", "", None),
        ("external", "ci", "external"),
        ("lab", "external", "external"),
        ("lab", "lab", "lab"),
        ("", "ci", "ci"),
        ("", "lab", "lab"),
    ],
)
def test_the_recorded_trust_is_the_lower_of_the_label_and_the_channel(
    tmp_path,
    online,
    capsys,
    label,
    channel,
    recorded,
):
    meta = {
        "artifact_id": AID,
        "test_type": "fvt",
        **({"trust": label} if label else {}),
    }
    flags = ["--channel-trust", channel] if channel else []
    code, report = _ingest(_minimal(tmp_path, meta), capsys, *flags)
    assert code == 0 and report.get("trust") == recorded
    assert online.trusts == [recorded]
    assert not any(p.startswith("run.trust=") for p in _trust_props(online))


def test_a_channel_assigned_trust_cannot_be_passed_as_a_bundle_prop(
    tmp_path, online, capsys
):
    root = _minimal(tmp_path)
    code, report = _ingest(root, capsys, "--bundle-prop", "run.trust=ci")
    assert (
        code == offline.REJECTED
        and "assigned by the receiving channel" in report["reason"]
    )
    assert online.calls == []


def test_a_row_cannot_set_run_trust(tmp_path):
    root = _minimal(tmp_path, {"artifact_id": AID, "test_type": "perf", "kind": "benchmarks",
                               "component": "suite"})  # fmt: skip
    for p in (root / "results").iterdir():
        p.unlink()
    row = {"name": "b", "measurements": {"m": 1.0}, "run_props": {"run.trust": "ci"}}
    (root / "results" / "r.json").write_text(json.dumps({"benchmarks": [row]}))
    assert "run.trust is assigned" in _rejected(root)


# --- one time-zone rule ----------------------------------------------------------------------


def test_a_zoneless_timestamp_is_read_as_utc_and_warned_about(tmp_path, capsys):
    out = _seal(tmp_path, "n", NAIVE)
    assert "carry no time zone; read as UTC" in capsys.readouterr().err
    assert (
        json.loads((out / "bundle.json").read_text())["started_at"]
        == "2026-10-08T10:00:00Z"
    )


def test_a_zoned_timestamp_is_not_warned_about(tmp_path, capsys):
    _seal(tmp_path, "z", JUNIT)
    assert "no time zone" not in capsys.readouterr().err


def test_the_local_zone_of_the_sealing_host_does_not_matter(tmp_path, monkeypatch):
    digests = []
    for i, tz in enumerate(("UTC", "Asia/Kolkata")):
        monkeypatch.setenv("TZ", tz)
        time.tzset()
        digests.append(
            _dig(_seal(tmp_path, f"t{i}", NAIVE, "--redact-host", "--redact-user"))
        )
    monkeypatch.delenv("TZ")
    time.tzset()
    assert digests[0] == digests[1]


# --- compatibility check ---------------------------------------------------------------------


def test_check_lists_what_this_ingest_accepts(capsys):
    assert cli(["results", "--check"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (
        out["schema_versions"] == [offline.SCHEMA_VERSION] and "junit" in out["kinds"]
    )
    assert out["trust_labels"] == ["lab", "external"] and "compatible" not in out


def test_check_accepts_a_bundle_this_kit_sealed(tmp_path, capsys):
    out = _seal(tmp_path, "c", JUNIT)
    capsys.readouterr()
    assert cli(["results", "--check", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["compatible"] and report["bundle"]["producer"]["name"] == offline.KIT


@pytest.mark.parametrize(
    "extra, why",
    [
        ({"schema_version": 2}, "newer than this ingest"),
        ({"site": "x"}, "fields this ingest does not know: ['site']"),
    ],
)
def test_check_names_the_side_to_upgrade(tmp_path, capsys, extra, why):
    root = _minimal(tmp_path, {"artifact_id": AID, "test_type": "fvt", **extra})
    assert cli(["results", "--check", str(root)]) == offline.REJECTED
    report = json.loads(capsys.readouterr().out)
    assert not report["compatible"] and why in report["problems"][0]
    assert "upgrade the connected ingest" in report["problems"][0]


def test_check_without_a_bundle_json_is_incomplete(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert cli(["results", "--check", str(tmp_path / "empty")]) == offline.INCOMPLETE


# --- receipt ---------------------------------------------------------------------------------


def test_a_receipt_id_follows_the_digest_and_the_upload():
    a, b = offline.receipt_id("d" * 64, "u1"), offline.receipt_id("d" * 64, "u2")
    assert a != b and a == offline.receipt_id("d" * 64, "u1")
    assert offline.receipt_id("e" * 64, "u1") != a
    assert len(a.split("-")) == 5 and a.startswith("RCPT-")


def test_the_ingest_prints_the_receipt(tmp_path, online, capsys):
    root = _minimal(tmp_path)
    capsys.readouterr()
    code = cli(
        ["results", "--from-bundle", str(root), "--bundle-prop", "upload_id=U-1"]
    )
    out, err = capsys.readouterr()
    report = json.loads(out.splitlines()[-1])
    expected = offline.receipt_id(offline.digest(root), "U-1")
    assert code == 0 and report["receipt"] == expected
    assert report["receipt"] in err and "bundle sha256" in err
    assert _flags(online.calls[0])["--run-id"] == report["run_id"]


def test_a_duplicate_or_dry_run_prints_no_receipt(tmp_path, online, capsys):  # noqa: F811
    root = _minimal(tmp_path)
    code, report = _ingest(root, capsys, "--dry-run")
    assert report["status"] == "would-ingest" and "receipt" not in report
    online.client.verdicts = [offline.digest(root)]
    code, report = _ingest(root, capsys)
    assert report["status"] == "duplicate" and "receipt" not in report
    assert "RCPT" not in capsys.readouterr().err


def test_run_trust_cannot_be_passed_as_a_result_prop():
    from spyre_clickhouse_ingest import results

    with pytest.raises(SystemExit, match="assigned by the receiving channel"):
        results._write_named_artifact_verdicts(
            None, "db", SimpleNamespace(result_props=[("run.trust", "ci")]), {}
        )


def test_a_junit_that_mentions_run_trust_seals_and_ingests(tmp_path, online, capsys):
    mention = JUNIT.replace("<testsuite ", '<testsuite data-x="run.trust" ')
    root = _seal(tmp_path, "m", mention)
    code, report = _ingest(root, capsys)
    assert code == 0 and report["status"] == "ingested"


def test_check_runs_the_full_validation(tmp_path, capsys):
    root = _minimal(tmp_path, {"artifact_id": AID, "test_type": "fvt", "trust": "ci"})
    assert cli(["results", "--check", str(root)]) == offline.REJECTED
    report = json.loads(capsys.readouterr().out)
    assert not report["compatible"] and "does not validate" in report["problems"][0]


@pytest.mark.parametrize("bad", ["inf", "nan", "-3", "1e30", "1e13", "1,5", "abc", ""])
def test_an_unusable_suite_time_falls_back_to_the_suite_start(tmp_path, bad):
    xml = JUNIT.replace('tests="2"', f'tests="2" time="{bad}"')
    meta = json.loads((_seal(tmp_path, "t", xml) / "bundle.json").read_text())
    assert meta["started_at"] == meta["ended_at"] == "2026-10-08T10:00:00Z"


def test_redaction_warns_about_the_short_name_of_a_qualified_host(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(offline.socket, "gethostname", lambda: "lab-node-7.example.org")
    _seal(tmp_path, "q", JUNIT.replace("</testsuites>", "<system-out>lab-node-7</system-out></testsuites>"),
          "--redact-host")  # fmt: skip
    assert "still in ['junit.xml']" in capsys.readouterr().err


def test_redaction_blanks_every_quote_form_and_touches_nothing_else(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(offline.socket, "gethostname", lambda: "lab-node-7")
    crlf = (
        "<?xml version='1.0'?>\r\n<testsuites><testsuite name=\"a>b\" hostname='lab-node-7' "
        'tests="2" timestamp="2026-10-08T10:00:00+00:00">\r\n'
        '<testcase classname="a.b" name="t"/></testsuite>'
        '<testsuite hostname="lab-node-7" name="c" tests="0"/>'
        "<system-out>lab-node-7</system-out></testsuites>"
    )
    out = _seal(tmp_path, "r", crlf, "--redact-host")
    text = (out / "results" / "junit.xml").read_bytes()
    assert text.count(b'hostname=""') == 2 and text.count(b"\r\n") == 2
    err = capsys.readouterr().err
    assert "blanked 2 testsuite hostname attribute(s)" in err
    assert "still in ['junit.xml']" in err


def test_redaction_copes_with_a_read_only_source(tmp_path):
    d = tmp_path / "xml-ro"
    d.mkdir()
    (d / "junit.xml").write_text(HOSTED)
    (d / "junit.xml").chmod(0o444)
    argv = ["results", "--offline", "--out", str(tmp_path / "ro"), "--xml-dir", str(d),
            "--artifact", f"id:{AID}", "--arch", "s390x", "--trigger-type", "fvt", "--redact-host"]  # fmt: skip
    assert cli(argv) == 0
    assert b"lab-node-7" not in (tmp_path / "ro" / "results" / "junit.xml").read_bytes()


def test_an_unlabelled_ingest_without_a_channel_writes_exactly_the_old_props(
    tmp_path, online, capsys
):
    root = _minimal(tmp_path)
    code, report = _ingest(
        root, capsys, "--uploader", "jdoe", "--bundle-url", "https://art/b/"
    )
    argv = online.calls[0]
    props = sorted(argv[i + 1] for i, a in enumerate(argv) if a == "--result-prop")
    sha = offline.digest(root)
    assert code == 0 and "trust" not in report
    assert online.trusts == [None]
    assert props == sorted([
        "source=bundle", "uploader=jdoe", f"bundle_sha256={sha}", "bundle_url=https://art/b/",
        "bundle_kind=junit", "bundle_files=junit.xml",
    ])  # fmt: skip


def test_an_offline_sealed_bundle_without_a_channel_is_external(
    tmp_path, online, capsys
):
    root = _seal(tmp_path, "s", JUNIT, "--seal-trust", "lab")
    code, report = _ingest(root, capsys)
    assert code == 0 and report["trust"] == "external"


def test_a_timestamp_that_cannot_be_converted_is_ignored(tmp_path, capsys):
    odd = JUNIT.replace("2026-10-08T10:00:00+00:00", "0001-01-01T00:00:00+05:00")
    meta = json.loads((_seal(tmp_path, "o", odd) / "bundle.json").read_text())
    assert "started_at" not in meta
    assert "could not be read and were ignored" in capsys.readouterr().err


def test_a_bad_or_late_seal_time_is_a_clear_result(tmp_path, capsys):
    d = tmp_path / "x"
    d.mkdir()
    (d / "junit.xml").write_text(JUNIT)

    def argv(out, *extra):
        return ["results", "--offline", "--out", str(tmp_path / out), "--xml-dir", str(d),
                "--artifact", f"id:{AID}", "--arch", "s390x", "--trigger-type", "fvt", *extra]  # fmt: skip

    assert cli(argv("a", "--triggered-at", "yesterday")) == offline.REJECTED
    assert (
        "--triggered-at 'yesterday' is not an ISO-8601 time" in capsys.readouterr().out
    )
    assert cli(argv("b", "--ended-at", "2026-10-08T09:00:00")) == offline.REJECTED
    capsys.readouterr()
    # a start later than the last suite end never makes a bundle that fails its own check
    assert cli(argv("c", "--triggered-at", "2026-10-09T00:00:00Z")) == 0
    meta = json.loads((tmp_path / "c" / "bundle.json").read_text())
    assert meta["ended_at"] == meta["started_at"] == "2026-10-09T00:00:00Z"
    assert (
        cli(
            argv(
                "d",
                "--triggered-at",
                "2026-10-09T00:00:00Z",
                "--ended-at",
                "2026-10-08T00:00:00Z",
            )
        )
        == offline.REJECTED
    )


@pytest.mark.parametrize("extra", [["--offline"], ["--from-bundle", "x"], ["--upload"]])
def test_check_stands_alone(capsys, extra):
    assert cli(["results", "--check", "", *extra]) == offline.REJECTED
    assert "--check stands alone" in capsys.readouterr().out


def test_an_empty_check_path_is_an_error(capsys):
    assert cli(["results", "--check", ""]) == offline.REJECTED
    assert "wants a bundle path" in capsys.readouterr().out


def test_a_junit_property_named_run_trust_does_not_change_the_assigned_trust(
    tmp_path, online, capsys
):
    prop = JUNIT.replace(
        "<testcase ",
        '<properties><property name="run.trust" value="ci"/></properties><testcase ',
        1,
    )
    code, report = _ingest(_seal(tmp_path, "p", prop), capsys)
    assert code == 0 and report["trust"] == "external"
    argv = online.calls[0]
    assert not any(a.startswith("run.trust") for a in argv)


def test_a_late_start_warns_when_it_clamps_the_end(tmp_path, capsys):
    d = tmp_path / "x"
    d.mkdir()
    (d / "junit.xml").write_text(JUNIT)
    capsys.readouterr()
    assert cli(["results", "--offline", "--out", str(tmp_path / "c"), "--xml-dir", str(d),
                "--artifact", f"id:{AID}", "--arch", "s390x", "--trigger-type", "fvt",
                "--triggered-at", "2026-10-09T00:00:00Z"]) == 0  # fmt: skip
    assert "ended_at set to the start" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [[], ["--triggered-at", "2026-10-08T10:00:00Z"],
                                   ["--ended-at", "2026-10-08T11:00:00Z"]])  # fmt: skip
def test_a_zoneless_timestamp_warns_unless_both_times_are_given(
    tmp_path, capsys, flags
):
    _seal(tmp_path, "w", NAIVE, *flags)
    assert "carry no time zone; read as UTC" in capsys.readouterr().err


def test_the_direct_path_does_not_pre_validate_times(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "junit.xml").write_text(JUNIT)
    args = offline.parser().parse_args(
        [
            "--offline",
            "--out",
            str(tmp_path / "o"),
            "--xml-dir",
            str(d),
            "--artifact",
            f"id:{AID}",
            "--arch",
            "s390x",
            "--trigger-type",
            "fvt",
            "--triggered-at",
            "2026-10-08T10:00:00",
        ]  # fmt: skip
    )
    with pytest.raises(offline.BundleError, match="--triggered-at"):
        offline.write(args)
    args.out = str(tmp_path / "o2")
    with pytest.raises(offline.BundleError, match="started_at"):
        offline.write(args, validate_times=False)  # the old check() message


def test_no_command_line_flag_sets_the_assigned_trust(capsys):
    with pytest.raises(SystemExit):
        cli(
            [
                "results",
                "--assigned-trust",
                "ci",
                "--xml-dir",
                "d",
                "--artifact",
                f"id:{AID}",
            ]
        )
    assert "unrecognized arguments: --assigned-trust" in capsys.readouterr().err


def test_a_channel_level_is_refused_where_it_cannot_apply(tmp_path, capsys):
    assert (
        cli(
            [
                "results",
                "--offline",
                "--channel-trust",
                "ci",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == offline.REJECTED
    )
    assert "--channel-trust applies to --from-bundle" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--channel-trust applies"):
        cli(["results", "--xml-dir", str(tmp_path), "--artifact", f"id:{AID}", "--arch", "s390x",
             "--channel-trust", "ci", "--schema", "v2"])  # fmt: skip


@pytest.mark.parametrize(
    "label, channel, recorded",
    [
        ("external", "", "external"),
        ("lab", "ci", "lab"),
        ("", "ci", "ci"),
        ("", "", ""),
    ],
)
def test_the_real_write_junit_hands_the_trust_to_results(
    tmp_path, online, capsys, monkeypatch, label, channel, recorded
):
    seen = []
    monkeypatch.setattr(bundle, "write_junit", REAL_WRITE_JUNIT)
    monkeypatch.setattr(
        results,
        "main",
        lambda argv, assigned_trust="": seen.append((argv, assigned_trust)),
    )
    meta = {
        "artifact_id": AID,
        "test_type": "fvt",
        **({"trust": label} if label else {}),
    }
    flags = ["--channel-trust", channel] if channel else []
    args = results.build_parser().parse_args(
        ["--from-bundle", str(_minimal(tmp_path, meta)), "--strict", *flags]
    )
    assert bundle.ingest(args) == 0
    ((argv, assigned),) = seen
    assert assigned == recorded
    assert not any(a.startswith("run.trust") or a == "--assigned-trust" for a in argv)


@pytest.mark.parametrize("assigned", ["external", "lab", "ci", ""])
def test_the_verdict_props_carry_the_assigned_trust(monkeypatch, assigned):
    written = []
    monkeypatch.setattr(
        results, "ensure",
        lambda *a, **kw: SimpleNamespace(dry_run=False, identity=SimpleNamespace(artifact_id=AID)),
    )  # fmt: skip
    monkeypatch.setattr(
        results, "insert_artifact_result", lambda *a, **kw: written.append(kw) or True
    )
    args = SimpleNamespace(
        artifact=f"id:{AID}", artifact_id="", lookup="only", registry="off", tag_family="",
        tags=[], tag_date=None, origin="promoted", sources=[], identity_deps=[], context_deps=[],
        props=[], tag_props=[], run_url="https://art/b/", dry_run=False, arch="s390x",
        repository="", branch="", sha="", jenkins_run_key="J#1", gha_run_id="", component="",
        result_props=[("source", "bundle")], run_attempt=0, assigned_trust=assigned,
    )  # fmt: skip
    legs = {
        ("d3ea9749-67a5-5bd1-8471-a290d0c67fc9", "fvt"): {
            "failed": 0,
            "total": 2,
            "duration_s": 1.0,
        }
    }
    assert results._write_named_artifact_verdicts(None, "spyre_v2", args, legs)
    props = written[0]["props"]
    assert props.get("run.trust") == (assigned or None)


def test_a_direct_ingest_of_identical_files_is_never_a_duplicate(tmp_path, monkeypatch):
    digests, clock = [], iter(f"2026-11-14T10:00:0{n}Z" for n in range(9))
    monkeypatch.setattr(offline, "_now", lambda: next(clock))
    monkeypatch.setattr(
        bundle,
        "ingest",
        lambda a: digests.append(offline.digest(Path(a.from_bundle))) or 0,
    )
    src = tmp_path / "bench"
    src.mkdir()
    (src / "r.json").write_text(json.dumps({"benchmarks": [
        {"name": "b", "measurements": {"m": 1.0}}]}))  # fmt: skip
    argv = ["results", "--benchmarks-dir", str(src), "--trigger-type", "quality",
            "--component", "suite", "--artifact", f"id:{AID}", "--arch", "s390x"]  # fmt: skip
    assert cli(argv) == 0 and cli(argv) == 0
    assert digests[0] != digests[1]


def test_two_offline_seals_of_identical_files_share_a_digest(tmp_path):
    assert _dig(_seal(tmp_path, "a", JUNIT)) == _dig(_seal(tmp_path, "b", JUNIT))


def test_channel_trust_is_also_refused_with_validate_only_and_check(tmp_path, capsys):
    root = _seal(tmp_path, "v", JUNIT)
    for extra in (
        ["--from-bundle", str(root), "--validate-only"],
        ["--check", str(root)],
    ):
        assert cli(["results", "--channel-trust", "ci", *extra]) == offline.REJECTED
        assert "--channel-trust applies to --from-bundle" in capsys.readouterr().out

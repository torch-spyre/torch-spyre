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

"""The advisory check poster: conclusions, payload shape, idempotent upsert, and token hygiene."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "bench_check.py"
spec = importlib.util.spec_from_file_location("bench_check", SCRIPT)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

SHA = "0123456789abcdef0123456789abcdef01234567"


def row(
    verdict,
    name="op-a",
    metric="kernel_mean_ms",
    kind="perf",
    tier="blocking",
    worse=10.0,
    invalidated=False,
    reason="",
):
    return {
        "run_id": "r1", "benchmark_id": f"b-{name}", "name": name, "metric": metric, "kind": kind, "tier": tier,
        "verdict": verdict, "reason": reason, "series": {"arch": "x86_64"}, "value": 11.0, "worse_pct": worse, "invalidated": invalidated,
        "baseline": {"median": 10.0}, "noise": {"n_repeats": 3},
    }  # fmt: skip


def report(*rows, invalidated=0, notes=()):
    return {
        "schema": "bench-verdicts/v1", "query": {"artifact_id": "a1"}, "runs": ["r1"], "notes": list(notes),
        "results": list(rows), "summary": {"invalidated": invalidated},
    }  # fmt: skip


class Args:
    dashboard_url_template = "https://dash.example/runs/{run_id}/{benchmark_id}"
    annotation_path = ""
    max_annotations = 10
    top = 10
    run_broken = ""


def payload(rep, kind="perf", **over):
    args = Args()
    for k, v in over.items():
        setattr(args, k, v)
    return bc.build_payload(rep, kind, "x86_64", SHA, args)


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([row("regressed")], "neutral"),
        ([row("suspected", tier="advisory")], "neutral"),
        ([row("pass"), row("improved")], "success"),
        ([row("no_baseline")], "neutral"),
        ([], "neutral"),
        ([row("regressed", tier="informational"), row("pass")], "success"),
        ([row("regressed", invalidated=True), row("pass")], "success"),
    ],
)
def test_conclusion_is_never_failure_for_a_verdict(rows, expected):
    assert payload(report(*rows))["conclusion"] == expected


def test_failure_only_when_the_run_itself_broke():
    assert (
        payload(report(row("regressed")), run_broken="ingest exited 2")["conclusion"]
        == "failure"
    )


def test_names_and_kinds_are_split():
    rep = report(row("regressed"), row("pass", metric="exact_match", kind="quality"))
    assert (
        payload(rep)["name"] == "Spyre Perf (x86_64)"
        and payload(rep)["conclusion"] == "neutral"
    )
    q = payload(rep, kind="quality")
    assert q["name"] == "Spyre Quality (x86_64)" and q["conclusion"] == "success"


def test_summary_lists_worst_first_with_links_and_caps_rows():
    rows = [row("suspected", name=f"op-{i}", worse=float(i)) for i in range(5)] + [
        row("regressed", name="op-r", worse=1.0)
    ]
    out = payload(report(*rows), top=3)
    text = out["output"]["summary"]
    assert text.index("op-r") < text.index("op-4") < text.index("op-3")
    assert "op-0" not in text and "3 more flagged" in text
    assert "https://dash.example/runs/r1/b-op-r" in text
    assert out["details_url"] == "https://dash.example/runs/r1/b-op-r"


def test_bad_template_gives_no_link_instead_of_failing():
    out = payload(report(row("regressed")), dashboard_url_template="https://x/{nope}")
    assert "details_url" not in out


def test_annotations_are_capped_and_never_failures():
    rows = [row("regressed", name=f"op-{i}") for i in range(80)]
    ann = payload(
        report(*rows), annotation_path="bench/definitions.yaml", max_annotations=500
    )["output"]["annotations"]
    assert len(ann) == bc.MAX_ANNOTATIONS
    assert {a["annotation_level"] for a in ann} == {"warning"}
    assert payload(report(*rows))["output"]["annotations"] == []


def test_invalidated_rows_are_not_counted_or_listed():
    out = payload(
        report(
            row("regressed", name="gone", invalidated=True), row("pass"), invalidated=1
        )
    )
    assert (
        "gone" not in out["output"]["summary"]
        and "1 invalidated" in out["output"]["summary"]
    )


def test_dry_run_prints_the_payload_and_needs_no_token(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    f = tmp_path / "r.json"
    f.write_text(json.dumps(report(row("regressed"))))
    assert (
        bc.main(
            [
                "--report",
                str(f),
                "--repo",
                "o/r",
                "--sha",
                SHA,
                "--arch",
                "x86_64",
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)[0]["name"] == "Spyre Perf (x86_64)"


def test_a_missing_token_warns_and_posts_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    f = tmp_path / "r.json"
    f.write_text(json.dumps(report(row("pass"))))
    argv = ["--report", str(f), "--repo", "o/r", "--sha", SHA, "--arch", "x86_64"]
    assert bc.main(argv) == 0
    assert "GITHUB_TOKEN" in capsys.readouterr().out


class Fake(bc.GitHub):
    def __init__(self, existing):
        super().__init__("sekret-token")
        self.existing, self.calls = existing, []

    def call(self, method, path, body=None):
        self.calls.append((method, path.split("?")[0], body))
        if method == "GET":
            return {"check_runs": self.existing}
        return {"id": 99}


def test_upsert_creates_then_updates_by_name_and_sha():
    p = payload(report(row("regressed")))
    gh = Fake([])
    assert gh.upsert_check("o/r", p)[0] == "created"
    assert gh.calls[-1][:2] == ("POST", "/repos/o/r/check-runs")
    gh = Fake([{"id": 5}, {"id": 7}])
    assert gh.upsert_check("o/r", p)[0] == "updated"
    method, path, body = gh.calls[-1]
    assert (method, path) == ("PATCH", "/repos/o/r/check-runs/7")
    assert "name" not in body and "head_sha" not in body


def test_status_fallback_is_never_failure_for_a_verdict():
    gh = Fake([])
    gh.post_status("o/r", payload(report(row("regressed"))))
    assert (
        gh.calls[-1][2]["state"] == "success"
        and gh.calls[-1][2]["context"] == "Spyre Perf (x86_64)"
    )


def test_the_token_never_appears_in_output(capsys):
    gh = bc.GitHub("sekret-token", api="http://127.0.0.1:9")
    with pytest.raises(SystemExit) as exc:
        gh.call("GET", "/x")
    assert (
        "sekret-token" not in str(exc.value)
        and "sekret-token" not in capsys.readouterr().out
    )


def test_rows_of_other_arches_are_not_judged():
    other = row("regressed")
    other["series"] = {"arch": "ppc64le"}
    mine = row("pass")
    mine["series"] = {"arch": "x86_64"}
    assert payload(report(other, mine))["conclusion"] == "success"


def test_title_and_cells_for_no_baseline_and_pipes():
    assert payload(report(row("no_baseline")))["output"]["title"] == "No baseline yet"
    out = payload(report(row("regressed", name="a|b")))
    assert "a\\|b" in out["output"]["summary"]


def test_negative_annotation_cap_gives_none_and_titles_are_bounded():
    rows = [row("regressed", name="n" * 400)]
    assert (
        payload(report(*rows), annotation_path="f", max_annotations=-3)["output"][
            "annotations"
        ]
        == []
    )
    ann = payload(report(*rows), annotation_path="f")["output"]["annotations"]
    assert len(ann[0]["title"]) <= 255


def test_values_are_hidden_unless_asked_for():
    rep = report(row("regressed", name="op-x"))
    hidden = payload(rep, annotation_path="f")
    shown = payload(rep, annotation_path="f", show_values=True)
    for out, visible in ((hidden, False), (shown, True)):
        text = out["output"]["summary"] + out["output"]["annotations"][0]["message"]
        assert ("11" in text and "10%" in text) is visible
    assert "baseline median" not in hidden["output"]["summary"]


def test_an_at_sign_in_a_name_cannot_mention_anyone():
    out = payload(report(row("regressed", name="@someone/op")))
    assert "@someone" not in out["output"]["summary"]


def test_an_update_does_not_resend_annotations():
    gh = Fake([{"id": 7}])
    gh.upsert_check("o/r", payload(report(row("regressed")), annotation_path="f"))
    assert "annotations" not in gh.calls[-1][2]["output"]


def test_posting_failures_do_not_fail_the_job(tmp_path, monkeypatch, capsys):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(report(row("regressed"))))
    monkeypatch.setenv("GITHUB_TOKEN", "t")

    def boom(self, *a):
        raise SystemExit("GitHub API POST failed: HTTP 403")

    monkeypatch.setattr(bc.GitHub, "upsert_check", boom)
    argv = ["--report", str(f), "--repo", "o/r", "--sha", SHA, "--arch", "x86_64"]
    assert bc.main(argv) == 0
    assert "::warning::" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name", ["a\\|b", "`x`", "[t](http://e)", "![i](http://e)", "<b>x</b>", "a\rb"]
)
def test_names_cannot_inject_markdown(name):
    out = payload(report(row("regressed", name=name)), run_broken="")
    line = next(
        ln
        for ln in out["output"]["summary"].splitlines()
        if ln.startswith("| regressed")
    )
    assert line.count("|") - line.count("\\|") == 6  # 5 cells, no extra separator
    assert not any(c in line.rsplit("| [run]", 1)[0] for c in "`[]<>\r")


def test_run_broken_text_is_escaped():
    out = payload(report(row("pass")), run_broken="boom @someone `x`")
    assert (
        "@someone" not in out["output"]["summary"]
        and "`" not in out["output"]["summary"]
    )


def test_notes_are_private_unless_values_are_shown():
    rep = report(row("regressed"), notes=["benchmark secret-key is slow"])
    assert "secret-key" not in payload(rep)["output"]["summary"]
    assert "secret-key" in payload(rep, show_values=True)["output"]["summary"]


@pytest.mark.parametrize("bad", ["not json", json.dumps({"schema": "other/v9"})])
def test_a_bad_report_warns_and_exits_zero(tmp_path, capsys, bad):
    f = tmp_path / "r.json"
    f.write_text(bad)
    argv = ["--report", str(f), "--repo", "o/r", "--sha", SHA, "--arch", "x86_64"]
    assert bc.main(argv) == 0
    assert "::warning::" in capsys.readouterr().out


def test_a_non_http_error_while_posting_does_not_fail(tmp_path, monkeypatch, capsys):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(report(row("regressed"))))
    monkeypatch.setenv("GITHUB_TOKEN", "t")

    def boom(self, *a):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(bc.GitHub, "upsert_check", boom)
    argv = ["--report", str(f), "--repo", "o/r", "--sha", SHA, "--arch", "x86_64"]
    assert bc.main(argv) == 0


def test_a_rejected_annotation_is_retried_without_annotations(tmp_path, monkeypatch):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(report(row("regressed"))))
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    seen = []

    def flaky(self, repo, payload):
        seen.append(len(payload["output"]["annotations"]))
        if len(seen) == 1:
            raise SystemExit("HTTP 422")
        return "created", {}

    monkeypatch.setattr(bc.GitHub, "upsert_check", flaky)
    argv = ["--report", str(f), "--repo", "o/r", "--sha", SHA, "--arch", "x86_64",
            "--annotation-path", "missing.yaml"]  # fmt: skip
    assert bc.main(argv) == 0
    assert seen == [1, 0]


@pytest.mark.parametrize(
    "name", ["https://evil.example/x", "see www.evil.com", "**b** #12", "~~x~~ &amp;"]
)
def test_names_cannot_plant_links_or_references(name):
    out = payload(report(row("regressed", name=name)))["output"]["summary"]
    line = next(ln for ln in out.splitlines() if ln.startswith("| regressed"))
    body = line.rsplit("| [run]", 1)[0]
    for bad in ("://", "www.", "#1", "**", "~~", "&amp"):
        assert bad not in body


def test_a_reason_outside_the_known_set_is_not_published():
    out = payload(report(row("suspected", reason="12.5 percent over")))["output"][
        "summary"
    ]
    assert "12.5" not in out


def test_a_broken_run_posts_failure_even_when_the_report_is_unreadable(tmp_path):
    f = tmp_path / "r.json"
    f.write_text("not json")
    args = Args()
    args.report, args.run_broken = str(f), "crashed"
    assert bc.load_report(args)["results"] == []


def test_one_failing_kind_does_not_stop_the_other(tmp_path, monkeypatch, capsys):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(report(row("regressed"), row("pass", kind="quality"))))
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    posted = []

    def flaky(self, repo, payload):
        posted.append(payload["name"])
        if len(posted) == 1:
            raise SystemExit("HTTP 500\nline2")
        return "created", {}

    monkeypatch.setattr(bc.GitHub, "upsert_check", flaky)
    argv = [
        "--report",
        str(f),
        "--repo",
        "o/r",
        "--sha",
        SHA,
        "--arch",
        "x86_64",
        "--kind",
        "both",
    ]
    assert bc.main(argv) == 0
    assert len(posted) == 2
    out = capsys.readouterr().out
    assert "%0A" in out and "line2" in out.replace("%0A", "") and "\nline2" not in out


def anchored_row(verdict, name="op-a", rolling="pass", tier="blocking", **a):
    r = row(rolling, name=name, tier=tier)
    r["anchored"] = {
        "verdict": verdict, "version": "2", "since_ts": "2026-11-01 02:00:00", "n_exceeding_runs": 12, "days_exceeding": 11,
        "worse_pct": 12.3, "anchor_value": 10.0, **a,
    }  # fmt: skip
    return r


def test_a_persistent_regression_keeps_the_check_neutral_though_the_rolling_verdict_passes():
    rep = report(anchored_row("regressed"), row("pass", name="op-b"))
    p = payload(rep)
    assert p["conclusion"] == "neutral"
    assert (
        p["output"]["title"]
        == "Advisory: persistent regression against the accepted baseline"
    )
    assert "**Persistent regressions**" in p["output"]["summary"]
    assert (
        "op-a kernel_mean_ms: regressed vs anchor v2 since 2026-11-01 (12 runs), rolling verdict `pass`"
        in p["output"]["summary"]
    )
    assert "12.3% worse" not in p["output"]["summary"]
    shown = payload(rep, show_values=True)["output"]["summary"]
    assert "(12 runs), 12.3% worse, rolling verdict `pass`" in shown
    assert "op-b" not in p["output"]["summary"].split("Persistent regressions")[1]


def test_only_a_regressed_anchored_verdict_counts_and_informational_and_invalidated_rows_never_do():
    for verdict in ("suspected", "improved", "pass", "no_anchor"):
        assert payload(report(anchored_row(verdict)))["conclusion"] == "success"
    assert (
        payload(report(anchored_row("regressed", tier="informational")))["conclusion"]
        == "success"
    )
    inv = anchored_row("regressed")
    inv["invalidated"] = True
    assert payload(report(inv, row("pass", name="op-b")))["conclusion"] == "success"


def test_a_report_without_anchored_keys_reads_as_before():
    assert payload(report(row("pass")))["conclusion"] == "success"
    assert "Persistent" not in payload(report(row("pass")))["output"]["summary"]


def test_persistent_regressions_are_annotated_as_warnings_within_the_cap():
    rep = report(
        *[anchored_row("regressed", name=f"op-{i}", days_exceeding=i) for i in range(5)]
    )
    ann = payload(rep, annotation_path="README.md", max_annotations=3)["output"][
        "annotations"
    ]
    assert len(ann) == 3 and {a["annotation_level"] for a in ann} == {"warning"}
    assert [a["title"] for a in ann][0] == "regressed vs anchor: op-4 kernel_mean_ms"
    assert (
        ann[0]["message"]
        == "regressed against the accepted baseline since 2026-11-01 (12 runs)."
    )
    shown = payload(rep, annotation_path="README.md", show_values=True)["output"][
        "annotations"
    ]
    assert shown[0]["message"].startswith(
        "12.3% worse than the accepted baseline 10 since 2026-11-01"
    )


def test_a_flagged_and_persistent_row_is_annotated_once():
    r = anchored_row("regressed", rolling="regressed")
    ann = payload(report(r), annotation_path="README.md")["output"]["annotations"]
    assert len(ann) == 1 and ann[0]["title"].startswith("regressed: ")


def test_the_persistent_list_is_capped_at_top_and_says_how_many_are_hidden():
    rep = report(*[anchored_row("regressed", name=f"op-{i}") for i in range(4)])
    s = payload(rep, top=2)["output"]["summary"]
    assert s.count("regressed vs anchor v2") == 2 and "2 more not shown" in s


def test_rolling_flagged_rows_keep_their_annotation_when_persistent_ones_fill_the_cap():
    rep = report(
        *[anchored_row("regressed", name=f"op-p{i}") for i in range(3)],
        row("regressed", name="op-new"),
    )
    ann = payload(rep, annotation_path="README.md", max_annotations=2)["output"][
        "annotations"
    ]
    assert [a["title"] for a in ann][0] == "regressed: op-new kernel_mean_ms"
    assert len(ann) == 2


def test_a_null_days_exceeding_does_not_break_the_ordering():
    rep = report(
        anchored_row("regressed", name="op-a", days_exceeding=None),
        anchored_row("regressed", name="op-b", days_exceeding=4),
    )
    s = payload(rep)["output"]["summary"]
    assert s.index("op-b") < s.index("op-a")

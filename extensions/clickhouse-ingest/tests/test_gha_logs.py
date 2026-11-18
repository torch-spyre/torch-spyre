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

"""The retry/dedupe/abort behaviour that was previously unreachable inside a YAML heredoc."""

import json
import subprocess

import pytest
from spyre_clickhouse_ingest import gha_logs, paging


class FakeCompleted:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(gha_logs.time, "sleep", lambda _s: None)


def _sequence(monkeypatch, results):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return results[min(len(calls) - 1, len(results) - 1)]

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_a_transient_502_is_retried_then_succeeds(monkeypatch):
    calls = _sequence(
        monkeypatch,
        [
            FakeCompleted(1, stderr=b"HTTP 502 Bad Gateway"),
            FakeCompleted(1, stderr=b"HTTP 503"),
            FakeCompleted(0, stdout=b"log body"),
        ],
    )
    result = gha_logs.run_with_retry(["gh", "api", "/x"])
    assert result.returncode == 0
    assert len(calls) == 3


def test_a_404_is_not_retried(monkeypatch):
    # Retrying a permanent answer burns the budget a real 502 needs.
    calls = _sequence(monkeypatch, [FakeCompleted(1, stderr=b"HTTP 404 Not Found")])
    result = gha_logs.run_with_retry(["gh", "api", "/x"])
    assert result.returncode == 1
    assert len(calls) == 1


def test_retries_are_bounded(monkeypatch):
    calls = _sequence(monkeypatch, [FakeCompleted(1, stderr=b"HTTP 502")])
    gha_logs.run_with_retry(["gh", "api", "/x"], max_attempts=4)
    assert len(calls) == 4


def test_duplicate_job_ids_are_dropped():
    # A duplicate listing would otherwise download and parse the same job twice, doubling every
    # row the run produces.
    jobs = [{"id": 1, "name": "a"}, {"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    assert gha_logs.dedupe_jobs(jobs) == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
    ]


def test_load_jobs_reads_jsonl_and_dedupes(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(j)
            for j in (
                {"id": 1, "name": "a"},
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"},
            )
        )
        + "\n"
    )
    assert [j["id"] for j in gha_logs.load_jobs(path)] == [1, 2]


def test_logs_are_written_with_sanitised_names(monkeypatch, tmp_path):
    _sequence(monkeypatch, [FakeCompleted(0, stdout=b"body")])
    monkeypatch.setattr(gha_logs, "escape_sequence_flag", lambda: [])
    count = gha_logs.download_job_logs(
        [{"id": 7, "name": "Inductor / Test: Coarse Tile"}], "org/repo", tmp_path
    )
    assert count == 1
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert "/" not in written[0].name and ":" not in written[0].name
    assert written[0].read_bytes() == b"body"


def test_zero_downloads_aborts(monkeypatch, tmp_path):
    # Every completed run has at least one job with a log, so zero means the mechanism broke.
    # Exiting non-zero is deliberate: the downstream empty-data guard would report success.
    _sequence(monkeypatch, [FakeCompleted(1, stderr=b"HTTP 404")])
    monkeypatch.setattr(gha_logs, "escape_sequence_flag", lambda: [])
    with pytest.raises(SystemExit) as exc:
        gha_logs.download_job_logs_or_die(
            [{"id": 1, "name": "a"}], "org/repo", tmp_path
        )
    assert exc.value.code == 1


def test_no_jobs_at_all_is_not_an_abort(monkeypatch, tmp_path):
    assert gha_logs.download_job_logs_or_die([], "org/repo", tmp_path) == 0


def test_escape_sequence_flag_is_probed(monkeypatch):
    # gh >= 2.97.0 needs the flag or every log download returns an empty body; older gh has no
    # such flag, so it is probed rather than version-pinned.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: FakeCompleted(0, stdout="--allow-escape-sequences  do it"),
    )
    assert gha_logs.escape_sequence_flag() == ["--allow-escape-sequences"]
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: FakeCompleted(0, stdout="no flag")
    )
    assert gha_logs.escape_sequence_flag() == []


def _jobs_gh(monkeypatch, fail_over, total=203):
    """subprocess.run that answers 502 for a listing larger than fail_over, as GitHub does for a many-job run."""
    calls = []

    def fake_run(args, **kwargs):
        size = int(
            next(a for a in args if str(a).startswith("per_page=")).split("=")[1]
        )
        calls.append(size)
        if size > fail_over:
            return FakeCompleted(1, stderr=b"gh: Server Error (HTTP 502)")
        rows = [{"id": i, "name": f"job {i}"} for i in range(total)]
        return FakeCompleted(
            0, stdout="".join(json.dumps(r) + "\n" for r in rows).encode()
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_list_jobs_steps_the_page_size_down_on_502(monkeypatch, tmp_path):
    calls = _jobs_gh(monkeypatch, fail_over=50)
    n = gha_logs.list_jobs("o/r", "1", tmp_path / "jobs.jsonl")
    assert n == 203
    assert calls == [100, 100, 50]
    assert len((tmp_path / "jobs.jsonl").read_text().splitlines()) == 203


def test_list_jobs_is_one_call_when_the_first_size_works(monkeypatch, tmp_path):
    calls = _jobs_gh(monkeypatch, fail_over=100, total=12)
    assert gha_logs.list_jobs("o/r", "1", tmp_path / "jobs.jsonl") == 12
    assert calls == [100]


def test_list_jobs_last_size_keeps_the_long_backoff_then_aborts(monkeypatch, tmp_path):
    calls = _jobs_gh(monkeypatch, fail_over=0)
    with pytest.raises(SystemExit) as stopped:
        gha_logs.list_jobs("o/r", "1", tmp_path / "jobs.jsonl")
    assert stopped.value.code == 1
    assert calls.count(10) == paging.LAST_SIZE_ATTEMPTS
    for size in paging.JOB_PAGE_SIZES[:-1]:
        assert calls.count(size) == paging.STEP_ATTEMPTS
    assert not (tmp_path / "jobs.jsonl").exists()


def test_list_jobs_does_not_retry_a_404_at_any_size(monkeypatch, tmp_path):
    calls = _sequence(
        monkeypatch, [FakeCompleted(1, stderr=b"gh: Not Found (HTTP 404)")]
    )
    with pytest.raises(SystemExit):
        gha_logs.list_jobs("o/r", "1", tmp_path / "jobs.jsonl")
    assert len(calls) == len(paging.JOB_PAGE_SIZES)


def test_list_jobs_asks_for_the_latest_attempt_with_get(monkeypatch, tmp_path):
    calls = _sequence(monkeypatch, [FakeCompleted(0, stdout=b"")])
    gha_logs.list_jobs("o/r", "77", tmp_path / "jobs.jsonl")
    (args,) = calls
    assert args[args.index("--method") + 1] == "GET"
    assert "filter=latest" in args and "--paginate" in args
    assert "/repos/o/r/actions/runs/77/jobs" in args


def test_list_jobs_keeps_only_a_prefix_and_strips_it(monkeypatch, tmp_path):
    rows = [
        {"id": 1, "name": "model-ops / llama"},
        {"id": 2, "name": "build"},
        {"id": 3, "name": "model-ops / gemma"},
    ]
    _sequence(
        monkeypatch,
        [FakeCompleted(0, stdout="".join(json.dumps(r) + "\n" for r in rows).encode())],
    )
    out = tmp_path / "jobs.jsonl"
    assert gha_logs.list_jobs("o/r", "1", out, name_prefix="model-ops / ") == 2
    assert [json.loads(line) for line in out.read_text().splitlines()] == [
        {"id": 1, "name": "llama"},
        {"id": 3, "name": "gemma"},
    ]
    assert gha_logs.load_jobs(out)[0]["name"] == "llama"


def test_list_jobs_with_no_match_writes_an_empty_file(monkeypatch, tmp_path):
    _sequence(monkeypatch, [FakeCompleted(0, stdout=b'{"id": 2, "name": "build"}\n')])
    out = tmp_path / "jobs.jsonl"
    assert gha_logs.list_jobs("o/r", "1", out, name_prefix="model-ops / ") == 0
    assert out.read_text() == ""

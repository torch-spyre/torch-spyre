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

"""The perf-zero guard keys on benchmark XML reaching the ingest, not on the tier's name.

A perf tier whose measuring leg writes its benchmark rows inline uploads only
test-result XML, so 0 parsed records is its steady state; refusing it threw away an
otherwise complete ingest. A perf upload that does carry benchmark XML still has to
parse some of it.
"""

import pytest

from spyre_clickhouse_ingest import results

from test_dry_run import RUN, WHEEL, XML, LiveClient

# classname carries "benchmark" so it dispatches as a benchmark XML, but the testcase
# name does not match _PERF_NAME_RE, so it parses to zero records -- the shape of a
# perf upload that lost its report.xml.
EMPTY_BENCH_XML = (
    '<?xml version="1.0"?><testsuites><testsuite name="s" timestamp="2026-01-01T00:00:00Z">'
    '<testcase classname="benchmark" name="nothing_parseable" time="1.0"/>'
    "</testsuite></testsuites>"
)


def ingest(monkeypatch, tmp_path, xml, name, tier="perf"):
    """A whole `results` ingest of one XML file at `tier`, against a stub client."""
    (tmp_path / name).write_text(xml)
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setattr(results, "get_client", lambda: LiveClient())
    results.main(
        ["--xml-dir", str(tmp_path), "--schema", "v2", "--run-id", RUN, "--trigger-type", tier,
         "--arch", "x86_64", "--artifact", WHEEL, "--repository", "torch-spyre/torch-spyre",
         "--sha", "ab12"]
    )  # fmt: skip


@pytest.mark.parametrize(
    ("tier", "parsed", "expected"),
    [
        ("perf", 0, False),  # the inline-ingest perf tier: nothing to parse
        ("perf", 3, True),
        ("perf", 3, False),
        ("regression", 0, True),
        ("", 0, True),
    ],
)
def test_the_guard_passes_everything_but_a_barren_benchmark_upload(
    tier, parsed, expected
):
    results._exit_if_perf_zero(tier, parsed, expected)


def test_a_perf_run_whose_benchmark_xml_parsed_nothing_is_still_refused(capsys):
    with pytest.raises(SystemExit) as exc:
        results._exit_if_perf_zero("perf", 0, True)
    assert exc.value.code == 1
    assert "refusing ingest=ok" in capsys.readouterr().err


def test_zero_records_without_benchmark_xml_says_why_it_is_not_a_refusal(capsys):
    results._exit_if_perf_zero("perf", 0, False)
    assert "not a refusal" in capsys.readouterr().out


def test_a_perf_tier_of_test_result_xml_only_ingests_clean(monkeypatch, tmp_path):
    """The regression: junit-perf-*.xml are test-result XMLs, and used to exit 1 here."""
    ingest(monkeypatch, tmp_path, XML, "junit-perf-granite-tp1.xml")


def test_a_perf_tier_whose_benchmark_xml_is_barren_still_fails(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        ingest(monkeypatch, tmp_path, EMPTY_BENCH_XML, "report.xml")
    assert exc.value.code == 1


def test_no_xml_at_all_still_fails_a_perf_tier(monkeypatch, tmp_path):
    """A perf run whose artifact never arrived is lost data, not a tier that ingests inline."""
    with pytest.raises(SystemExit) as exc:
        ingest(monkeypatch, tmp_path, XML, "ignored.txt")
    assert exc.value.code == 1

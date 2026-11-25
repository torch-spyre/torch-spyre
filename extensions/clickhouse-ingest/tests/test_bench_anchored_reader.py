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

"""The verdict reader and the check poster on a persistent regression the rolling baseline has absorbed (chdb)."""

import importlib.util
from pathlib import Path

import pytest
from bench_anchor_fixture import FILES, SHIFT, Db, accept, history
from bench_verdict_fixture import Fixture
from spyre_clickhouse_ingest import bench_verdicts as bv

SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "bench_check.py"
spec = importlib.util.spec_from_file_location("bench_check", SCRIPT)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

SHA = "0123456789abcdef0123456789abcdef01234567"


class ReaderDb(Db):
    """Db whose `rows` raises what bv.Source raises for a missing view."""

    def __init__(self, files=FILES):
        self.skip = set(FILES) - set(files)
        super().__init__()

    def rows(self, sql):
        try:
            return super().rows(sql)
        except RuntimeError as exc:
            if bv.is_missing(exc):
                raise bv.ViewMissing(str(exc).splitlines()[0]) from exc
            raise


class Args:
    dashboard_url_template = ""
    annotation_path = "README.md"
    max_annotations = 10
    top = 10
    run_broken = ""


@pytest.fixture
def world():
    d = ReaderDb()
    f = Fixture(d)
    runs = history(f)
    accept(f, 40, 1)
    yield f, runs
    d.s.close()


def report(world, run=-1, **kw):
    f, runs = world
    return bv.read_verdicts(f.db, run_id=[runs[run]], **kw)


def test_the_rolling_verdict_passes_while_the_reader_reports_the_persistent_regression(
    world,
):
    rep = report(world)
    assert {r["verdict"] for r in rep["results"]} == {"pass"}
    assert {r["anchored"]["verdict"] for r in rep["results"]} == {"regressed"}
    a = next(r for r in rep["results"] if r["name"] == "op-alpha")["anchored"]
    assert (a["version"], a["n_exceeding_runs"], a["days_exceeding"]) == ("1", 60, 59)
    assert a["since_ts"].startswith("2026-11-15")
    assert 11 < a["worse_pct"] < 13
    assert (
        rep["summary"]["persistent"] == 4
        and rep["summary"]["anchored"]["regressed"] == 4
    )
    assert rep["summary"]["actionable"] == 0
    assert [r["name"] for r in bv.persistent(rep)][0] in {
        "op-alpha",
        "op-beta",
        "op-gamma",
        "op-delta",
    }


def test_markdown_lists_the_persistent_regressions(world):
    md = bv.to_markdown(report(world))
    assert (
        "Persistent regressions" in md
        and "regressed vs anchor since 2026-11-15 (60 runs, 59 days)" in md
    )


def test_the_poster_stays_neutral_and_names_the_open_regressions(world):
    rep = report(world)
    p = bc.build_payload(rep, "perf", "x86_64", SHA, Args())
    assert p["conclusion"] == "neutral"
    assert (
        p["output"]["title"]
        == "Advisory: persistent regression against the accepted baseline"
    )
    assert "**Persistent regressions**" in p["output"]["summary"]
    assert (
        "since 2026-11-15" in p["output"]["summary"]
        and "rolling verdict `pass`" in p["output"]["summary"]
    )
    ann = p["output"]["annotations"]
    assert len(ann) == 4 and {a["annotation_level"] for a in ann} == {"warning"}
    assert ann[0]["title"].startswith("regressed vs anchor: ")


def test_few_repeats_cap_the_anchored_verdict_like_the_rolling_one(world):
    f, runs = world
    run = f.run(repeats=2, slow=SHIFT)
    rep = bv.read_verdicts(f.db, run_id=[run])
    assert {r["anchored"]["verdict"] for r in rep["results"]} == {"suspected"}
    assert {r["anchored"]["reason"] for r in rep["results"]} == {"few_repeats"}
    assert rep["summary"]["persistent"] == 0


def test_accepting_a_new_normal_clears_the_persistent_list():
    d = ReaderDb()
    f = Fixture(d)
    runs = history(f)
    accept(f, 40, 1)
    assert bv.read_verdicts(d, run_id=[runs[-1]])["summary"]["persistent"] == 4
    accept(f, 100, 2)
    rep = bv.read_verdicts(d, run_id=[runs[-1]])
    assert rep["summary"]["persistent"] == 0
    assert {r["anchored"]["verdict"] for r in rep["results"]} == {"pass"}
    assert (
        bc.build_payload(rep, "perf", "x86_64", SHA, Args())["conclusion"] == "success"
    )
    d.s.close()


def test_a_database_without_the_anchored_view_reports_a_note_and_no_anchored_verdicts():
    d = ReaderDb(files=[f for f in FILES if not f.startswith(("86", "88"))])
    f = Fixture(d)
    runs = history(f, nights=30)
    rep = bv.read_verdicts(d, run_id=[runs[-1]])
    assert any("v_benchmark_anchored_verdicts" in n for n in rep["notes"])
    assert {r["anchored"] for r in rep["results"]} == {None}
    assert (
        rep["summary"]["persistent"] == 0
        and rep["summary"]["anchored"]["regressed"] == 0
    )
    d.s.close()

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

"""The verdict view reads the run.purpose the ingest writes, on real chdb (see test_bench_purpose.py)."""

import pytest

session = pytest.importorskip("chdb.session")

from bench_verdict_fixture import Fixture  # noqa: E402
from test_bench_purpose import ingested_props  # noqa: E402
from test_bench_verdict_exclusions import KEY, SLOW, Db  # noqa: E402


@pytest.fixture
def inj():
    d = Db()
    f = Fixture(d)
    f.history(22)
    yield f
    d.s.close()


def verdict(inj, run):
    return inj.verdicts(run)[KEY][:2]


@pytest.mark.parametrize("purpose", ["nightly", "weekly", "merge"])
def test_scheduled_purposes_join_baselines_and_can_regress(inj, tmp_path, purpose):
    props = {"run.purpose": ingested_props(tmp_path, purpose)["run.purpose"]}
    inj.run(slow=SLOW, props=props)
    second = inj.run(slow=SLOW, props=props)
    assert verdict(inj, second)[0] == "regressed"


@pytest.mark.parametrize("purpose", ["on-demand", "pr-check", "release"])
def test_other_purposes_are_judged_but_never_regress(inj, tmp_path, purpose):
    props = {"run.purpose": ingested_props(tmp_path, purpose)["run.purpose"]}
    inj.run(slow=SLOW, props=props)
    second = inj.run(slow=SLOW, props=props)
    assert verdict(inj, second) == ("suspected", "purpose_not_baseline")
    nightly = inj.run()
    assert second not in inj.baseline(nightly)[0]


def test_an_ingest_purpose_matches_the_view_eligible_list(inj):
    listed = {
        r["purpose"]
        for r in inj.db.rows("SELECT purpose FROM benchmark_baseline_purposes")
    }
    assert listed == {"nightly", "weekly", "merge"}

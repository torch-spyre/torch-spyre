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

"""The experiment ledger (75-/76-) against chdb: models match the DDL, bundles load once, views read them."""

import json
import math
from datetime import datetime

import pytest
from spyre_clickhouse_ingest import schema
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier
from spyre_clickhouse_ingest.experiments import Bundle, abab_bundle

AUDIT = {"audit_uuid", "audit_timestamp"}


class ChdbClient:
    """The two calls Table.insert/insert_identities make, served by a chdb session."""

    def __init__(self, session):
        self.s = session

    def insert(self, table, rows, column_names=None, database=None):
        def enc(v):
            if isinstance(v, datetime):
                return v.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            return v

        body = "\n".join(
            json.dumps({c: enc(v) for c, v in zip(column_names, r)}) for r in rows
        )
        self.s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")

    def query(self, sql, parameters=None):
        ids = ", ".join(f"'{i}'" for i in (parameters or {}).get("ids", [])) or "''"
        sql = sql.replace("{ids:Array(UUID)}", f"({ids})")

        class R:
            result_rows = [
                tuple(r.values())
                for r in rows(self.s, sql.replace(" IN ('')", " IN (NULL)"))
            ]

        return R()


def rows(s, sql):
    raw = s.query(sql, "JSONEachRow").data()
    raw = raw.decode() if isinstance(raw, bytes) else raw
    return [json.loads(line) for line in raw.splitlines()]


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    for f in ("75-experiment-ledger.sql", "76-experiment-views.sql"):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / f).read_text()):
            s.query(stmt)
    yield s
    s.close()


def _driver_rows(aa: bool):
    """A paired driver run with synthetic values: B is 10% faster unless aa."""
    rows_ = [
        {
            "kind": "env",
            "ts": 1.0,
            "report": {
                "ts": 1.0,
                "host": "h",
                "node": "n1",
                "kernel": "5.14.el9.x86_64",
                "cpu_model": "cpu",
                "ncpu": 8,
                "load": [0.5],
                "env": {"FLEX_DEVICE": "PF"},
                "rpms": "ibm-flex-1.0-1.x86_64",
                "profiler_build": "false",
                "image_artifact": '{"artifact_id": "05ab2b13-374e-59de-9872-689d9333cd30"}',
                "sha": "e1",
            },
        },
        {
            "kind": "warmup",
            "arm": "A",
            "status": "ok",
            "wall_s": 9.0,
            "metrics": {"lat": 120.0},
            "ts": 2.0,
        },
        {
            "kind": "warmup",
            "arm": "B",
            "status": "ok",
            "wall_s": 9.0,
            "metrics": {"lat": 120.0},
            "ts": 3.0,
        },
    ]
    for p, noise in enumerate((1.0, 1.02, 0.99, 1.01)):
        b = 100.0 * noise * (1.0 if aa else 0.9)
        rows_ += [
            {
                "kind": "sample",
                "pair": p,
                "order": "AB",
                "arm": "A",
                "status": "ok",
                "wall_s": 10.0,
                "metrics": {"lat": 100.0, "aux": 5.0},
                "ts": 10.0 + p,
            },
            {
                "kind": "sample",
                "pair": p,
                "order": "AB",
                "arm": "B",
                "status": "ok",
                "wall_s": 10.0,
                "metrics": {"lat": b, "aux": 5.0},
                "ts": 10.5 + p,
            },
            {
                "kind": "interim",
                "pair": p,
                "n": p + 1,
                "effect_pct": 0.0 if aa else 10.0,
                "ci_lo_pct": math.nan,
                "ci_hi_pct": math.nan,
                "sd_log": math.nan,
                "e_value": None,
                "t_p": math.nan,
                "decision": "continue",
                "ts": 10.6 + p,
            },
        ]
    rows_.append(
        {
            "kind": "result",
            "ts": 20.0,
            "stop": "effect" if not aa else "max_n",
            "metrics": {
                "lat": {
                    "n": 4,
                    "decision": "max_n" if aa else "effect",
                    "effect_pct": 0.0 if aa else 10.5,
                    "ci_lo_pct": -1.0,
                    "ci_hi_pct": 1.0 if aa else 12.0,
                    "sd_log": 0.0135,
                    "e_value": None,
                    "t_p": 0.5,
                    "confirmed": not aa,
                    "floor_sd_log": None if aa else 0.0135,
                }
            },
        }
    )
    return rows_


SPEC = {
    "workload": "toy_model_b1",
    "primary": "lat",
    "arms": [{"name": "A", "env": {}}, {"name": "B", "env": {"K": "1"}}],
}
AA_SPEC = {**SPEC, "arms": [{"name": "A", "env": {}}, {"name": "B", "env": {}}]}


def _load(s, *records):
    bundle = Bundle(actor="ci@example.com").load(r for recs in records for r in recs)
    return bundle.write(ChdbClient(s), "")


def test_models_match_the_ddl_columns(chdb_session):
    for table in schema.EXPERIMENT_TABLES:
        ddl = [r["name"] for r in rows(chdb_session, f"DESCRIBE TABLE {table.name}")]
        assert [c for c in ddl if c not in AUDIT] == list(table.columns), table.name


@pytest.mark.parametrize(
    "table, bad",
    [
        ("experiment_events", {"kind": "guess"}),
        ("experiment_events", {"actor_kind": "root"}),
        ("experiment_arms", {"role": "control"}),
        ("experiment_arms", {"hazard_class": "risky"}),
        ("experiment_findings", {"gain_evidence": "T2"}),
        ("experiment_findings", {"gain_evidence": ""}),
    ],
)
def test_the_ddl_refuses_what_the_model_refuses(chdb_session, table, bad):
    base = {
        "experiment_events": {
            "experiment_id": "018f0000-0000-7000-8000-000000000001",
            "name": "x",
            "kind": "ab",
            "state": "running",
            "actor": "a",
        },
        "experiment_arms": {
            "experiment_id": "018f0000-0000-7000-8000-000000000001",
            "arm": "A",
            "role": "baseline",
            "knob_sha": "k",
            "actor": "a",
        },
        "experiment_findings": {
            "finding_id": "018f0000-0000-7000-8000-000000000002",
            "experiment_id": "018f0000-0000-7000-8000-000000000001",
            "kind": "effect",
            "metric": "lat",
            "estimate": 1.0,
            "verdict": "better",
            "gain_evidence": "G2",
            "actor": "a",
        },
    }[table]
    chdb_session.query(f"INSERT INTO {table} FORMAT JSONEachRow\n" + json.dumps(base))
    with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
        chdb_session.query(
            f"INSERT INTO {table} FORMAT JSONEachRow\n" + json.dumps({**base, **bad})
        )


def test_an_effect_finding_needs_gain_evidence_in_the_model_too():
    with pytest.raises(schema.SchemaError, match="gain_evidence"):
        Bundle(actor="a").load(
            [
                {
                    "record": "finding",
                    "experiment": "x",
                    "key": "k",
                    "ts": 0,
                    "kind": "effect",
                    "metric": "lat",
                    "estimate": 1.0,
                    "verdict": "better",
                }
            ]
        )


def test_a_bundle_written_twice_adds_nothing(chdb_session):
    recs = abab_bundle(_driver_rows(aa=False), SPEC, "exp-ab", actor="ci@example.com")
    first = _load(chdb_session, recs)
    assert first["experiment_samples"] == 4 * 2 * 2 + 2 * 1
    assert first["run_env_reports"] == 1
    assert set(_load(chdb_session, recs).values()) == {0}


def test_views_read_an_ab_and_an_aa_run(chdb_session):
    s = chdb_session
    _load(
        s,
        [
            {
                "record": "experiment",
                "name": "camp",
                "kind": "other",
                "state": "running",
                "ts": 0,
                "budget": {"card_min": 100},
            }
        ],
        abab_bundle(
            _driver_rows(aa=False),
            SPEC,
            "exp-ab",
            parent="camp",
            status={"budget_cap_card_min": 100, "budget_used_card_min": 25},
        ),
        abab_bundle(_driver_rows(aa=True), AA_SPEC, "exp-aa", parent="camp"),
    )

    verdict = rows(
        s,
        "SELECT arm, metric, verdict, gain_evidence, decision, final FROM v_arm_verdicts "
        "WHERE experiment_id = (SELECT experiment_id FROM v_experiments WHERE name = 'exp-ab')",
    )
    assert verdict == [
        {
            "arm": "B",
            "metric": "lat",
            "verdict": "better",
            "gain_evidence": "G2",
            "decision": "effect",
            "final": True,
        }
    ]

    floors = rows(
        s,
        "SELECT source, metric, n_pairs, round(run_sd_pct, 3) AS sd FROM v_noise_floors ORDER BY source, metric",
    )
    measured = {r["metric"]: r for r in floors if r["source"] == "measured"}
    assert set(measured) == {"lat", "aux"}  # only the A/A experiment's pairs
    assert measured["lat"]["n_pairs"] == 4 and measured["lat"]["sd"] > 0
    assert measured["aux"]["sd"] == 0
    assert [r["metric"] for r in floors if r["source"] == "reported"] == ["lat"]

    prog = {
        r["name"]: r
        for r in rows(
            s,
            "SELECT name, parent_id, state, runs, trials, card_min_used, findings "
            "FROM v_experiment_progress",
        )
    }
    assert prog["exp-ab"]["runs"] == 10 and prog["exp-ab"]["trials"] == 4
    assert prog["exp-ab"]["state"] == "done" and prog["exp-ab"]["card_min_used"] == 0.25
    assert prog["exp-ab"]["parent_id"] == Bundle.experiment_id("camp")

    tried = rows(
        s,
        "SELECT experiment, arm, knobs, gain_evidence FROM v_subject_tried "
        "WHERE subject_key = 'toy_model_b1' AND role = 'candidate' ORDER BY experiment",
    )
    assert tried == [
        {"experiment": "exp-aa", "arm": "B", "knobs": {}, "gain_evidence": None},
        {
            "experiment": "exp-ab",
            "arm": "B",
            "knobs": {"env.K": "1"},
            "gain_evidence": "G2",
        },
    ]


def test_a_profiler_build_demotes_a_confirmed_effect():
    driver = _driver_rows(aa=False)
    driver[0]["report"]["profiler_build"] = "true"
    finding = [r for r in abab_bundle(driver, SPEC, "exp") if r["record"] == "finding"]
    assert [(f["gain_evidence"], f["demoted_by"]) for f in finding] == [
        ("G1", ["profiler_build"])
    ]


def test_a_later_finding_version_supersedes_and_a_retraction_hides_it(chdb_session):
    s = chdb_session
    f = {
        "record": "finding",
        "experiment": "x",
        "key": "k",
        "kind": "noise_floor",
        "metric": "lat",
        "estimate": 2.0,
    }
    _load(
        s,
        [
            {"record": "experiment", "name": "x", "state": "done", "ts": 0},
            {**f, "ts": 1},
            {**f, "ts": 2, "estimate": 3.0},
        ],
    )
    assert rows(s, "SELECT estimate FROM v_experiment_findings") == [{"estimate": 3.0}]
    _load(s, [{**f, "ts": 3, "estimate": 3.0, "state": "retracted"}])
    assert rows(
        s, "SELECT count() AS n FROM v_noise_floors WHERE source = 'reported'"
    ) == [{"n": 0}]

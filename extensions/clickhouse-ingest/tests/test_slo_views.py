"""SLO views: the pitfalls this repo has already paid for, checked on the view text."""

import re
from pathlib import Path

SQL = (Path(__file__).resolve().parents[1] / "schema" / "64-slo-views.sql").read_text()
VIEWS = re.findall(r"CREATE VIEW IF NOT EXISTS (\w+) AS(.*?);\n", SQL, re.S)
BODY = {name: body for name, body in VIEWS}


def _code(text):
    return "\n".join(line.split("--")[0] for line in text.splitlines())


def test_the_file_declares_exactly_the_documented_views():
    assert list(BODY) == [
        "v_slo_targets",
        "v_slo_pr_verdict_daily",
        "v_slo_main_green_daily",
        "v_slo_ingest_freshness",
        "v_slo_incident_response",
        "v_slo_error_budget",
    ]


def test_no_table_is_created_so_the_registry_set_is_untouched():
    assert "CREATE TABLE" not in SQL.upper()


def test_percentiles_use_the_ornull_combinator():
    # quantileIf over no rows is NaN, which a page cannot tell from a number.
    code = _code(SQL)
    assert not re.search(r"\bquantileIf\(", code)
    assert len(re.findall(r"quantileIfOrNull\(", code)) >= 8


def test_no_unguarded_division():
    # Every `/ x` outside a unit conversion divides by nullIf(...), or by a literal.
    for name, body in BODY.items():
        for m in re.finditer(r"/\s*([^\s,)]+)", _code(body)):
            token = m.group(1)
            assert token.startswith("nullIf") or re.fullmatch(r"\d+", token), (
                name,
                token,
            )


def test_empty_minif_is_never_reported_as_the_epoch():
    # minIf over no rows is 1970-01-01; this file takes the newest/oldest with max/min over a
    # non-empty group or guards on the count.
    assert "minIf(" not in _code(SQL)
    assert "if(s.rows_7d = 0, NULL" in SQL


def test_no_aggregate_is_aliased_to_a_source_column_another_aggregate_reads():
    # ClickHouse substitutes the alias into the other aggregates (ILLEGAL_AGGREGATION). A select-list
    # alias is a problem only when a later aggregate in the same SELECT reads a column of that name.
    agg = r"(?:count|countIf|max|min|any|sum|sumIf|argMax|quantileIfOrNull)"
    for name, body in BODY.items():
        for select in re.split(r"\bFROM\b", _code(body)):
            aliases = set(re.findall(agg + r"\([^;]*?\)\s+AS\s+([a-z_]+)\b", select))
            reads = set()
            for call in re.finditer(agg + r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", select):
                reads |= set(re.findall(r"\b(?:[a-z]\.)?([a-z_]+)\b", call.group(1)))
            clash = aliases & reads
            # `runs`, `passed` and the like are reused on purpose as aliases of earlier aggregates;
            # only an alias that is also a real source column is a clash.
            real = {
                "day",
                "lane",
                "repo",
                "outcome",
                "severity",
                "cause_class",
                "source",
                "window_days",
            }
            assert not (clash & real), (name, clash & real)


def test_every_target_the_views_use_is_in_one_place():
    targets = re.search(r"arrayJoin\(\[(.*?)\]\)", SQL, re.S).group(1)
    rows = re.findall(r"\('(\w+)',\s*'([^']+)',\s*(\d+),\s*([\d.]+)\)", targets)
    assert {(s, subj) for s, subj, *_ in rows} >= {
        ("verdict", "merge-queue"),
        ("verdict", "spyre-test"),
        ("verdict", "main-push"),
        ("main_green", "*"),
        ("freshness", "jenkins"),
        ("freshness", "gha"),
        ("ack", "critical"),
        ("ack", "warning"),
    }
    assert all(0 < float(share) <= 1 for *_, share in rows)
    for name, body in BODY.items():
        if name != "v_slo_targets":
            assert not re.search(
                r"within_min\s*=\s*\d|target_share\s*=\s*[\d.]+", _code(body)
            )


def test_targets_are_marked_as_proposals_in_the_file():
    assert "PROPOSAL" in SQL and "not a" in SQL and "commitment" in SQL


def test_superseded_runs_are_left_out_of_every_figure():
    for name in ("v_slo_pr_verdict_daily", "v_slo_error_budget"):
        assert "superseded" in BODY[name]


def test_a_source_with_no_rows_still_appears_in_freshness():
    body = BODY["v_slo_ingest_freshness"]
    assert "LEFT JOIN" in body and "within_limit" in body

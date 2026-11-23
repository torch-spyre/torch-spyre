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

"""Read benchmark verdicts for one run or artifact from the existing views.

`python -m spyre_clickhouse_ingest bench-verdicts --run-id <stored run id> [--run-id ...] [--format json|markdown]`

Sources: v_benchmark_metric_verdicts (verdict, baseline), v_bench_results (repeats, CV),
v_invalidated_runs (invalidations), benchmark_runs.props (run.trust / run.validity). A missing
optional view degrades to a note, never an error. The verdict view drops invalidated samples and
caps suspect or non-baseline-purpose runs to `suspected` (reasons `suspect_data`,
`purpose_not_baseline`); this reader adds caps for thin evidence (few repeats, a run whose
validity is below `unqualified_env`) and lists wholly invalidated runs in `excluded_runs`. Ids
are the STORED run ids (`stored_run_ids` in the engine's run.json), not the engine uuid.
"""

import argparse
import json
import math
import sys
import uuid

SCHEMA = "bench-verdicts/v1"
VERDICTS = ("regressed", "suspected", "improved", "pass", "no_baseline")
# The view's "unchanged" is reported as "pass".
VIEW_STATUS = {"unchanged": "pass"}
ANCHORED_VERDICTS = ("regressed", "suspected", "improved", "pass", "no_anchor")
# Validity states ordered worst first, as in the engine; only these may support a `regressed`.
VALIDITY_ORDER = (
    "invalid",
    "suspect",
    "external",
    "unqualified_env",
    "shared_env",
    "valid",
)
REGRESSION_VALIDITY = ("unqualified_env", "shared_env", "valid")
QUALITY_METRICS = frozenset(
    "exact_match acc_norm lm_eval_acc rouge1 rouge2 rougeL bleu_4 meteor ndcg_at_10 spearman "
    "accuracy v_measure retrieval_pct".split()
)
# Engine quality metrics. The diagnostic ones (standard errors, counts, error shares) support a score
# and are never judged or counted themselves.
QUALITY_PREFIXES = ("llm.eval.", "llm.parity.")
DIAGNOSTIC_SUFFIXES = (
    "_stderr",
    ".n_samples",
    ".error_frac",
    ".truncated_frac",
    ".invalid_frac",
)
DEFAULT_MIN_REPEATS = 3
_MISSING = (
    "unknown_table",
    "unknown table",
    "doesn't exist",
    "does not exist",
    "no such table",
)


class ViewMissing(Exception):
    """An optional view or table is not in this database."""


def is_missing(exc):
    """True when a ClickHouse error means a table or view does not exist."""
    return any(m in str(exc).lower() for m in _MISSING)


class Source:
    """Rows as dicts from a ClickHouse database. Tests substitute any object with `rows(sql)`."""

    def __init__(self, client):
        self.client = client

    def rows(self, sql):
        try:
            return list(self.client.query(sql).named_results())
        except Exception as exc:  # clickhouse_connect raises driver-specific errors
            if is_missing(exc):
                raise ViewMissing(str(exc).splitlines()[0]) from exc
            raise


def _uuid(value):
    """Canonical text of a UUID, or ValueError; the only way an id reaches SQL."""
    return str(uuid.UUID(str(value)))


def _ident(value):
    """A plain identifier-like token (arch); anything else is refused before it reaches SQL."""
    if not value or not all(c.isalnum() or c in "_-" for c in value):
        raise ValueError(f"not a valid identifier: {value!r}")
    return value


def _in(ids):
    return ", ".join(f"toUUID('{i}')" for i in ids)


def _since(v):
    """A streak start, or None for NULL or the epoch default of a streak with no eligible run yet."""
    return None if not v or str(v).startswith(("NULL", "1970")) else v


def is_diagnostic(metric):
    """A companion of a quality score: its standard error, sample count or error share."""
    suffix = next((s for s in DIAGNOSTIC_SUFFIXES if metric.endswith(s)), None)
    if suffix is None:
        return False
    return (
        metric.startswith(QUALITY_PREFIXES) or metric[: -len(suffix)] in QUALITY_METRICS
    )


def is_quality_metric(metric):
    return (
        metric in QUALITY_METRICS
        or metric.startswith(QUALITY_PREFIXES)
        or is_diagnostic(metric)
    )


def _num(v):
    return None if v is None or not math.isfinite(float(v)) else float(v)


def _worst_validity(states):
    states = [s for s in states if s]
    # A value outside the engine's states is treated as the conservative "suspect".
    rank = {v: i for i, v in enumerate(VALIDITY_ORDER)}
    return min(states, key=lambda v: rank.get(v, rank["suspect"])) if states else ""


class VerdictReader:
    def __init__(self, source, min_repeats=DEFAULT_MIN_REPEATS):
        self.src = source
        self.min_repeats = min_repeats
        self.notes = []

    def _optional(self, label, sql, hint):
        try:
            return self.src.rows(sql)
        except ViewMissing:
            self.notes.append(f"{label} is not available in this database ({hint})")
            return None

    def run_ids(self, run_id=None, artifact_id=None):
        if run_id:
            return sorted(
                {_uuid(r) for r in ([run_id] if isinstance(run_id, str) else run_id)}
            )
        sql = (
            "SELECT DISTINCT toString(t.run_id) AS run_id FROM v_benchmark_results_enriched AS t "
            f"WHERE t.artifact_id = toUUID('{_uuid(artifact_id)}') ORDER BY run_id"
        )
        try:
            return [r["run_id"] for r in self.src.rows(sql)]
        except ViewMissing:
            self.notes.append(
                "v_benchmark_results_enriched is missing; cannot resolve an artifact to runs"
            )
            return []

    def read(self, run_id=None, artifact_id=None, arch=None):
        if bool(run_id) == bool(artifact_id):
            raise ValueError("pass exactly one of run_id or artifact_id")
        ids = self.run_ids(run_id, artifact_id)
        report = {
            "schema": SCHEMA,
            "query": {"run_id": run_id, "artifact_id": artifact_id, "arch": arch},
            "runs": ids,
            "excluded_runs": [],
            "excluded_benchmarks": [],
            "notes": self.notes,
            "results": [],
        }
        if not ids:
            self.notes.append("no benchmark rows found for this id")
            return self._finish(report)
        rows = self._optional(
            "v_benchmark_metric_verdicts",
            "SELECT toString(run_id) AS run_id, toString(benchmark_id) AS benchmark_id, component, backend, arch, "
            "source_job, name, input_shapes, toString(ts) AS ts, metric, tier, higher_is_better, n_baseline, "
            "n_current, baseline_value, current_value, delta_pct, worse_pct, floor_pct, status, reason "
            f"FROM v_benchmark_metric_verdicts AS t WHERE t.run_id IN ({_in(ids)})"
            + (f" AND t.arch = '{_ident(arch)}'" if arch else "")
            + " ORDER BY run_id, name, metric",
            "apply schema 60 and 62",
        )
        if rows is None:
            return self._finish(report)
        ctx = self._context(ids)
        ctx["anchored"] = self._anchored(ids, arch)
        report["results"] = [self._result(r, ctx) for r in rows]
        seen = {r["run_id"] for r in rows}
        report["excluded_runs"] = sorted(
            run
            for run, hits in ctx["inval"].items()
            if run not in seen
            and any(
                h["state"] == "invalidated"
                and h["benchmark_id"] is None
                and h["repeat"] is None
                for h in hits
            )
        )
        for run in report["excluded_runs"]:
            self.notes.append(
                f"run {run} is invalidated; the verdict view drops its samples, so it has no verdicts"
            )
        have = {(r["run_id"], r["benchmark_id"]) for r in rows}
        report["excluded_benchmarks"] = sorted(
            f"{run}/{h['benchmark_id']}"
            for run, hits in ctx["inval"].items()
            for h in hits
            if h["state"] == "invalidated"
            and h["benchmark_id"] is not None
            and h["repeat"] is None
            and (run, h["benchmark_id"]) not in have
        )
        for key in report["excluded_benchmarks"]:
            self.notes.append(f"benchmark {key} is invalidated and has no verdicts")
        if (
            not rows
            and not report["excluded_runs"]
            and not report["excluded_benchmarks"]
        ):
            self.notes.append(
                "no verdict rows for these run ids (unknown id, nothing ingested, or no match for --arch)"
            )
        return self._finish(report)

    def _anchored(self, ids, arch):
        """Anchored verdicts (schema 88) by series and metric; an empty dict when the view is missing."""
        rows = self._optional(
            "v_benchmark_anchored_verdicts",
            "SELECT toString(run_id) AS run_id, toString(benchmark_id) AS benchmark_id, component, backend, arch, "
            "source_job, metric, status, reason, version, anchor_value, current_value, delta_pct, worse_pct, floor_pct, "
            "toString(since_ts) AS since_ts, n_exceeding_runs, days_exceeding "
            f"FROM v_benchmark_anchored_verdicts AS t WHERE t.run_id IN ({_in(ids)})"
            + (f" AND t.arch = '{_ident(arch)}'" if arch else ""),
            "apply schema 86 and 88; persistent regressions are not reported",
        )
        return {
            (
                r["run_id"],
                r["benchmark_id"],
                r["component"],
                r["backend"],
                r["arch"],
                r["source_job"],
                r["metric"],
            ): r
            for r in rows or []
        }

    def _context(self, ids):
        where = f"t.run_id IN ({_in(ids)})"
        noise = {}
        rows = self._optional(
            "v_bench_results",
            "SELECT toString(run_id) AS run_id, toString(benchmark_id) AS benchmark_id, backend, metric, n_repeats, cv_pct, "
            f"is_suspect FROM v_bench_results AS t WHERE {where}",
            "apply schema 81-84; repeats and CV are unknown",
        )
        for r in rows or []:
            noise[(r["run_id"], r["benchmark_id"], r["backend"], r["metric"])] = r
        inval = {}
        rows = self._optional(
            "v_invalidated_runs",
            "SELECT toString(run_id) AS run_id, toString(benchmark_id) AS benchmark_id, repeat, state, reason_code "
            f"FROM v_invalidated_runs AS t WHERE {where}",
            "apply schema 83; invalidations are unknown",
        )
        for r in rows or []:
            inval.setdefault(r["run_id"], []).append(r)
        props = {}
        rows = self._optional(
            "benchmark_runs",
            "SELECT toString(run_id) AS run_id, groupUniqArray(props['run.validity']) AS validity, "
            "groupUniqArray(props['run.trust']) AS trust, groupUniqArray(props['run.purpose']) AS purpose, "
            "groupUniqArray(props['run.validity_reasons']) AS reasons "
            f"FROM benchmark_runs AS t WHERE {where} GROUP BY t.run_id",
            "trust and validity are unknown",
        )
        for r in rows or []:
            props[r["run_id"]] = r
        ttype = {}
        rows = self._optional(
            "v_benchmark_results_enriched",
            f"SELECT toString(t.run_id) AS run_id, groupUniqArray(t.test_type) AS test_type FROM v_benchmark_results_enriched AS t "
            f"WHERE {where} GROUP BY t.run_id",
            "kind falls back to the metric name",
        )
        for r in rows or []:
            ttype[r["run_id"]] = set(r["test_type"])
        return {"noise": noise, "inval": inval, "props": props, "ttype": ttype}

    def _result(self, r, ctx):
        run, bench, metric = r["run_id"], r["benchmark_id"], r["metric"]
        noise = ctx["noise"].get((run, bench, r["backend"], metric))
        p = ctx["props"].get(run, {})
        validity = _worst_validity(p.get("validity", []))
        trust = (
            "external"
            if "external" in p.get("trust", [])
            else next((t for t in p.get("trust", []) if t), "")
        )
        reasons = sorted({x for s in p.get("reasons", []) for x in s.split(",") if x})
        hits = [
            h
            for h in ctx["inval"].get(run, [])
            if h["benchmark_id"] in (None, bench) and h["state"] == "invalidated"
        ]
        suspect_hits = [
            h
            for h in ctx["inval"].get(run, [])
            if h["benchmark_id"] in (None, bench) and h["state"] == "suspect"
        ]
        whole = any(h["repeat"] is None for h in hits)
        base_med, floor = _num(r["baseline_value"]), _num(r["floor_pct"])
        n_repeats = noise["n_repeats"] if noise else None
        view_verdict = VIEW_STATUS.get(r["status"], r["status"])
        verdict, reason = view_verdict, r["reason"]
        # A run that mixes test types is classed by metric name; only an all-quality run is quality by type.
        is_quality = ctx["ttype"].get(run) == {"quality"} or is_quality_metric(metric)
        unjudged = not validity or "" in p.get("validity", [])
        if view_verdict == "regressed":
            # A score is one value over n samples, so the repeat-count and repeat-noise caps do not apply; instead
            # nothing may be called a regression unless the data behind it was judged and is trustworthy.
            if (
                not is_quality
                and n_repeats is not None
                and n_repeats < self.min_repeats
            ):
                verdict, reason = "suspected", "few_repeats"
            elif is_quality and suspect_hits:
                verdict, reason = "suspected", "suspect_data"
            elif is_quality and (unjudged or trust == "external"):
                verdict, reason = (
                    "suspected",
                    "validity_unknown" if unjudged else "trust_external",
                )
            elif validity and validity not in REGRESSION_VALIDITY:
                verdict, reason = "suspected", f"validity_{validity}"
            elif not is_quality and noise and noise["is_suspect"]:
                verdict, reason = "suspected", "suspect_repeats"
        anchored = self._anchored_result(
            ctx["anchored"].get(
                (
                    run,
                    bench,
                    r["component"],
                    r["backend"],
                    r["arch"],
                    r["source_job"],
                    metric,
                )
            ),
            n_repeats,
            validity,
            noise,
        )
        series = {
            k: r[k]
            for k in ("component", "benchmark_id", "backend", "arch", "source_job")
        }
        return {
            "run_id": run,
            "benchmark_id": bench,
            "name": r["name"],
            "input_shapes": r["input_shapes"],
            "metric": metric,
            "kind": "quality" if is_quality else "perf",
            "diagnostic": is_diagnostic(metric),
            "tier": r["tier"],
            "higher_is_better": bool(r["higher_is_better"]),
            "verdict": verdict,
            "view_verdict": view_verdict,
            "reason": reason,
            "value": _num(r["current_value"]),
            "delta_pct": _num(r["delta_pct"]),
            "worse_pct": _num(r["worse_pct"]),
            "baseline": {
                "n_runs": r["n_baseline"],
                "median": base_med,
                "band_pct": floor,
                "low": None
                if base_med is None or floor is None
                else base_med * (1 - floor / 100),
                "high": None
                if base_med is None or floor is None
                else base_med * (1 + floor / 100),
            },
            "noise": None
            if noise is None
            else {
                "n_repeats": n_repeats,
                "cv_pct": _num(noise["cv_pct"]),
                "suspect": bool(noise["is_suspect"]),
            },
            "series_key": "/".join(str(series[k]) for k in series),
            "series": series,
            "trust": trust,
            "validity": validity,
            "validity_reasons": reasons,
            "purpose": next((x for x in p.get("purpose", []) if x), ""),
            "invalidated": bool(hits) and whole,
            "invalidated_repeats": sorted(
                h["repeat"] for h in hits if h["repeat"] is not None
            ),
            "invalidation_reason": hits[0]["reason_code"] if hits else "",
            "suspect_invalidation": bool(suspect_hits),
            "anchored": anchored,
        }

    def _anchored_result(self, a, n_repeats, validity, noise):
        """The anchored verdict of a row with the reader's evidence caps; None when the series has no anchored row."""
        if a is None:
            return None
        view_status = VIEW_STATUS.get(a["status"], a["status"])
        verdict, reason = view_status, a["reason"]
        if view_status == "regressed":
            if n_repeats is not None and n_repeats < self.min_repeats:
                verdict, reason = "suspected", "few_repeats"
            elif validity and validity not in REGRESSION_VALIDITY:
                verdict, reason = "suspected", f"validity_{validity}"
            elif noise and noise["is_suspect"]:
                verdict, reason = "suspected", "suspect_repeats"
        return {
            "verdict": "no_anchor" if view_status == "no_anchor" else verdict,
            "view_verdict": view_status,
            "reason": reason,
            "version": a["version"] or None,
            "anchor_value": _num(a["anchor_value"]),
            "value": _num(a["current_value"]),
            "delta_pct": _num(a["delta_pct"]),
            "worse_pct": _num(a["worse_pct"]),
            "floor_pct": _num(a["floor_pct"]),
            "since_ts": _since(a["since_ts"])
            if view_status in ("regressed", "suspected")
            else None,
            "n_exceeding_runs": int(a["n_exceeding_runs"] or 0),
            "days_exceeding": int(a["days_exceeding"] or 0),
        }

    @staticmethod
    def _finish(report):
        judged = [r for r in report["results"] if not r.get("diagnostic")]
        live = [r for r in judged if not r["invalidated"]]
        counts = {v: 0 for v in VERDICTS}
        by_kind = {}
        for r in live:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
            k = by_kind.setdefault(
                r["kind"], {"counts": {v: 0 for v in VERDICTS}, "actionable": 0}
            )
            k["counts"][r["verdict"]] = k["counts"].get(r["verdict"], 0) + 1
            k["actionable"] += int(
                r["verdict"] in ("regressed", "suspected")
                and r["tier"] != "informational"
            )
        for r in judged:
            k = by_kind.setdefault(
                r["kind"], {"counts": {v: 0 for v in VERDICTS}, "actionable": 0}
            )
            k["invalidated"] = k.get("invalidated", 0) + int(r["invalidated"])
        report["summary"] = {
            "counts": counts,
            "by_kind": by_kind,
            "invalidated": len(judged) - len(live),
            "reasons": {
                k: sum(
                    1 for r in live if r["verdict"] == "suspected" and r["reason"] == k
                )
                for k in sorted(
                    {
                        r["reason"]
                        for r in live
                        if r["verdict"] == "suspected" and r["reason"]
                    }
                )
            },
            "anchored": {
                v: sum(1 for r in live if (r["anchored"] or {}).get("verdict") == v)
                for v in ANCHORED_VERDICTS
            },
            "persistent": len(persistent(report)),
            "actionable": sum(
                1
                for r in live
                if r["verdict"] in ("regressed", "suspected")
                and r["tier"] != "informational"
            ),
        }
        return report


def read_verdicts(
    client, run_id=None, artifact_id=None, arch=None, min_repeats=DEFAULT_MIN_REPEATS
):
    """The report dict (see SCHEMA) for one run id or artifact id; `client` is a clickhouse_connect client."""
    source = client if hasattr(client, "rows") else Source(client)
    return VerdictReader(source, min_repeats).read(run_id, artifact_id, arch)


def actionable(report, kind=None):
    """Live, non-informational regressed/suspected rows (optionally of one kind), worst first."""
    rows = [
        r
        for r in report["results"]
        if not r["invalidated"]
        and not r.get("diagnostic")
        and r["verdict"] in ("regressed", "suspected")
        and r["tier"] != "informational"
        and (kind is None or r["kind"] == kind)
    ]
    return sorted(
        rows, key=lambda r: (r["verdict"] != "regressed", -(r["worse_pct"] or 0))
    )


def persistent(report, kind=None):
    """Live, non-informational rows `regressed` against the accepted baseline, longest first. The rolling verdict may have
    absorbed them already, which is what this list is for."""
    rows = [
        r
        for r in report["results"]
        if not r["invalidated"]
        and (r.get("anchored") or {}).get("verdict") == "regressed"
        and r["tier"] != "informational"
        and (kind is None or r["kind"] == kind)
    ]
    return sorted(
        rows,
        key=lambda r: (
            -r["anchored"]["days_exceeding"],
            -(r["anchored"]["worse_pct"] or 0),
        ),
    )


def persistent_since(r):
    a = r["anchored"]
    when = f" since {a['since_ts'][:10]}" if a["since_ts"] else ""
    return f", regressed vs anchor{when} ({a['n_exceeding_runs']} runs, {a['days_exceeding']} days)"


REASON_TEXT = {
    "single_run": "one run over the floor; regressed needs the previous run over as well",
    "suspect_data": "the run is flagged suspect and is kept out of baselines",
    "purpose_not_baseline": "the run's purpose is not baseline-eligible and it is kept out of baselines",
    "few_repeats": "too few valid repeats to call a regression",
    "validity_unknown": "no policy judged the data behind this score, so it cannot be called a regression",
    "trust_external": "the score came from an external source and is not trusted to call a regression",
}


def reason_text(reason):
    return REASON_TEXT.get(reason) or (
        "run validity is below the level that can support a regression"
        if reason.startswith("validity_")
        else ""
    )


def to_json(report):
    return json.dumps(report, indent=2, sort_keys=True)


def _cell(v, nd=3):
    return "" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))


def to_markdown(report, limit=20):
    """Every kind, with values and baselines: internal use only, never posted on a public pull request."""
    s = report["summary"]
    out = [f"Verdicts for {', '.join(report['runs']) or 'no run'}", ""]
    out += [f"- {k}: {v}" for k, v in s["counts"].items() if v]
    if s["invalidated"]:
        out.append(f"- invalidated (excluded): {s['invalidated']}")
    out += [
        f"- suspected because {k}: {n} ({reason_text(k)})"
        for k, n in s["reasons"].items()
        if reason_text(k)
    ]
    out += [""] + [f"Note: {n}" for n in report["notes"]]
    pers = persistent(report)
    if pers:
        out += [
            "Persistent regressions (worse than the accepted baseline, whatever the rolling verdict says):",
            "",
        ]
        out += [
            f"- {r['name']} {r['metric']}: {_cell(r['anchored']['worse_pct'])}% worse than baseline v{r['anchored']['version']}"
            f"{persistent_since(r)}"
            for r in pers[:limit]
        ]
        out.append("")
    rows = sorted(
        (r for r in report["results"] if not r.get("diagnostic")),
        key=lambda r: (VERDICTS.index(r["verdict"]), -(r["worse_pct"] or 0)),
    )
    if rows:
        out += [
            "| verdict | benchmark | metric | value | baseline median (n runs, band %) | delta % | CV % | trust | validity |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    for r in rows[:limit]:
        b, n = r["baseline"], r["noise"] or {}
        verdict = r["verdict"] + (f" ({r['reason']})" if r["reason"] else "")
        if r["invalidated"]:
            verdict = f"invalidated ({r['invalidation_reason']})"
        out.append(
            f"| {verdict} | {r['name']} | {r['metric']} | {_cell(r['value'])} | "
            f"{_cell(b['median'])} ({b['n_runs']}, {_cell(b['band_pct'])}) | {_cell(r['delta_pct'])} | "
            f"{_cell(n.get('cv_pct'))} | {r['trust']} | {r['validity']} |"
        )
    if len(rows) > limit:
        out.append(f"\n{len(rows) - limit} more rows not shown.")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="bench-verdicts", description=__doc__.splitlines()[0]
    )
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument(
        "--run-id",
        action="append",
        help="stored run id (run.json stored_run_ids); repeatable",
    )
    who.add_argument("--artifact-id")
    ap.add_argument("--arch")
    ap.add_argument("--format", choices=("json", "markdown"), default="json")
    ap.add_argument("--min-repeats", type=int, default=DEFAULT_MIN_REPEATS)
    args = ap.parse_args(argv)
    from .client import ClickHouse

    report = read_verdicts(
        ClickHouse.connect(database=ClickHouse.ENV.target_database() or ""),
        args.run_id,
        args.artifact_id,
        args.arch,
        args.min_repeats,
    )
    print(to_json(report) if args.format == "json" else to_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

"""Evaluate the `derive:` and `compare:` blocks of an Analysis over stored Results.

  spyre-bench analyze ANALYSIS --catalog DIR --results PATH... [--pack DIR]... [--central DIR]... [--json]

PATH is a JSON file (a list of Results or one Result) or the output directory of `spyre-bench run`; repeat --results to
combine runs (for example one run per pinning policy).

derive combines two cases of one subject and environment per repeat pair; compare tabulates a metric (or the derived
one) by a run/label/observation attribute within each `across` combination. A cell may hold only one Result (add
env.node to `across` to keep nodes apart): pooling Results of another subject, environment, node or case, or repeated
runs of one case, is refused. Non-finite samples and pairs with a zero denominator are skipped and listed under
`warnings`. Report is YAML, or JSON with --json.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from spyre_bench import engine, quality

PIN_KEY = "run.exec.cpu_pinning"


class AnalysisError(Exception):
    pass


def _case(result):
    """The case name an Analysis refers to; a Result built by `run` is named after its run and labels the case."""
    meta = result["metadata"]
    return (meta.get("labels") or {}).get("bench.case") or meta["name"]


def _props(result, repeat):
    """Run properties of a repeat: the Result's run_props, overridden by the repeat's own."""
    return {**(result["status"].get("run_props") or {}), **(repeat.get("props") or {})}


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def repeat_values(result, metric, warn=None):
    """{run.repeat: median of the repeat's finite samples} over the ok repeats that measured the metric."""
    out, seen = {}, set()
    for r in result["status"]["repeats"]:
        if r["run.repeat"] in seen:
            raise AnalysisError(f"{_case(result)}: run.repeat {r['run.repeat']} appears twice")
        seen.add(r["run.repeat"])
        if r.get("status", "ok") != "ok":
            continue
        raw = r["samples"].get(metric) or []
        ok = [x for x in raw if _finite(x)]
        if len(ok) != len(raw) and warn is not None:
            warn.append(
                f"{_case(result)} repeat {r['run.repeat']}: {len(raw) - len(ok)} non-finite {metric} sample(s) ignored"
            )
        if ok:
            out[r["run.repeat"]] = statistics.median(ok)
    return out


def _row_props(result, metric):
    """Properties from the first ok repeat that measured the metric (the one rule both compare paths use)."""
    for r in result["status"]["repeats"]:
        if r.get("status", "ok") == "ok" and any(_finite(x) for x in r["samples"].get(metric) or []):
            return _props(result, r)
    return dict(result["status"].get("run_props") or {})


def _pinnings(result):
    """Pinning policies seen across the ok repeats; a repeat without the property takes the label, else none."""
    st = result["status"]
    label = (st["environment"].get("labels") or {}).get(engine.PIN_LABEL) or "none"
    return {_props(result, r).get(PIN_KEY) or label for r in st["repeats"] if r.get("status", "ok") == "ok"}


def _labels(result, pinning):
    """Environment labels; a missing pinning label is `none`, and it is left out when pinning is not compared."""
    labels = {**(result["status"]["environment"].get("labels") or {})}
    labels[engine.PIN_LABEL] = labels.get(engine.PIN_LABEL) or "none"
    if not pinning:
        del labels[engine.PIN_LABEL]
    return labels


def _relaxed(relax, key):
    """key is named by a relax pattern: exact, or a prefix with a trailing `*` (a bare `*` relaxes nothing)."""
    return any(p == key or (p.endswith("*") and len(p) > 1 and key.startswith(p[:-1])) for p in relax)


def _strict(labels, relax):
    return {k: v for k, v in labels.items() if not _relaxed(relax, k)}


def incomparable(a, b, pinning=True, relax=()):
    """Why two Results cannot be combined: empty means same subject, environment, node and pinning.

    pinning=False leaves the pinning policy out (the compared dimension itself); a Result whose own repeats disagree
    on pinning is never comparable.
    """
    sa, sb = a["status"], b["status"]
    why = [f"mixed cpu pinning within {_case(r)}" for r in (a, b) if len(_pinnings(r)) > 1]
    if sa["subject"]["artifact_id"] != sb["subject"]["artifact_id"] and not _relaxed(relax, "subject.id"):
        why.append("different subject")
    if sa["environment"]["env_id"] != sb["environment"]["env_id"] and not _relaxed(relax, "env.id"):
        why.append("different env_id")
    if _strict(_labels(a, pinning), relax) != _strict(_labels(b, pinning), relax):
        why.append("different environment labels")
    if sa["environment"].get("observations", {}).get("env.node") != sb["environment"].get("observations", {}).get(
        "env.node"
    ) and not _relaxed(relax, "env.node"):
        why.append("different node")
    if pinning and _pinnings(a) != _pinnings(b):
        why.append("different cpu pinning")
    return why


def spread(values):
    """(mean, median, cv_pct); cv_pct is None below two values or at a zero mean."""
    mean = statistics.fmean(values)
    cv = statistics.stdev(values) / abs(mean) * 100 if len(values) > 1 and mean else None
    return mean, statistics.median(values), cv


def _combine(op, a, b, x):
    if op == "difference":
        return a - b
    if op == "ratio":
        return a / b
    return (a - b) / (x["first"] - x["second"])


def derive_pair(derive, metric, first, second, warn=None):
    """The derived value per repeat pair, its median/CV, and the same operation on the two aggregates."""
    why = incomparable(first, second)
    if why:
        raise AnalysisError(
            f"derive {derive['name']}: {_case(first)} and {_case(second)} are not comparable: {', '.join(why)}"
        )
    va, vb = repeat_values(first, metric, warn), repeat_values(second, metric, warn)
    reps = sorted(set(va) & set(vb))
    values, used, skipped = [], [], 0
    for r in reps:
        try:
            v = _combine(derive["op"], va[r], vb[r], derive.get("x"))
        except ZeroDivisionError:
            v, why = None, "zero denominator"
        else:
            why = None if _finite(v) else "non-finite result"
        if why:
            skipped += 1
            if warn is not None:
                warn.append(f"derive {derive['name']}: repeat {r} skipped ({why})")
            continue
        values.append(v)
        used.append(r)
    if not values:
        raise AnalysisError(f"derive {derive['name']}: no usable repeat pair of {_case(first)} and {_case(second)}")
    mean, med, cv = spread(values)

    def agg(res, vals):
        v = res["status"]["aggregate"].get("metrics", {}).get(metric, {}).get("value")
        return v if _finite(v) else statistics.median(vals.values())

    try:
        of_agg = _combine(derive["op"], agg(first, va), agg(second, vb), derive.get("x"))
    except ZeroDivisionError:
        of_agg = None
    of_agg = of_agg if _finite(of_agg) else None
    return {
        "name": derive["name"],
        "op": derive["op"],
        "first": _case(first),
        "second": _case(second),
        "repeats": used,
        "values": values,
        "n": len(values),
        "skipped_pairs": skipped,
        "mean": mean,
        "median": med,
        "cv_pct": cv,
        "of_aggregates": of_agg,
    }


def _derive_unit(derive, unit):
    return derive.get("unit") or ("1" if derive["op"] == "ratio" else unit)


def _row(result, metric, values, **extra):
    st = result["status"]["environment"]
    if len(_pinnings(result)) > 1:
        raise AnalysisError(f"{_case(result)}: repeats disagree on cpu pinning ({sorted(_pinnings(result))})")
    return {
        "labels": st.get("labels", {}),
        "obs": st.get("observations", {}),
        "props": _row_props(result, metric),
        "values": values,
        "results": [result],
        **extra,
    }


def _plain_rows(results, metric, warn):
    rows = []
    for r in results:
        vals = list(repeat_values(r, metric, warn).values())
        if vals:
            rows.append(_row(r, metric, vals))
    return rows


def _derived_rows(derive, metric, results, warn):
    """One row per pair of Results of the two cases in the same context. Raises when none is comparable."""
    firsts = [r for r in results if _case(r) == derive["cases"]["first"]]
    seconds = [r for r in results if _case(r) == derive["cases"]["second"]]
    for k, found in (("first", firsts), ("second", seconds)):
        if not found:
            raise AnalysisError(f"derive {derive['name']}: no Result for case {derive['cases'][k]!r}")
    rows, unpaired = [], 0
    for a in firsts:
        mates = [b for b in seconds if not incomparable(a, b)]
        if len(mates) > 1:
            raise AnalysisError(
                f"derive {derive['name']}: {len(mates)} Results of {derive['cases']['second']} share the context of "
                f"one Result of {derive['cases']['first']}; select one run per context"
            )
        if not mates:
            unpaired += 1
            continue
        d = derive_pair(derive, metric, a, mates[0], warn)
        rows.append({**_row(a, metric, d["values"]), "results": [a, mates[0]], "pair": d})
    if not rows:
        probe = incomparable(firsts[0], seconds[0])
        raise AnalysisError(
            f"derive {derive['name']}: {derive['cases']['first']} and {derive['cases']['second']} are not "
            f"comparable: {', '.join(probe) or 'no repeats in common'}"
        )
    return rows, unpaired


def _attr(row, key):
    for src in (row["props"], row["labels"], row["obs"]):
        if key in src:
            return str(src[key])
    return "(unset)"


def _check_cell(key, rows):
    """A cell holds one Result (pair): anything else pools incomparable or repeated runs."""
    if len(rows) < 2:
        return
    a = rows[0]["results"][0]
    for r in rows[1:]:
        b = r["results"][0]
        why = (
            ["different cases"]
            if _case(a) != _case(b)
            else incomparable(a, b) or ["repeated Results of one case in one context"]
        )
        raise AnalysisError(
            f"compare: cell {' / '.join(key[0] + (key[1],))} holds {len(rows)} Results ({', '.join(why)}); "
            "narrow the selection or add env.node to `across`"
        )


def _reading(effect, direction):
    if effect is None or direction not in ("lower_is_better", "higher_is_better") or effect == 0:
        return None
    return "better" if (effect < 0) == (direction == "lower_is_better") else "worse"


def compare(cmp_, rows, unit="1", direction=None, relax=()):
    """Per (across, by) cell: n, mean, median, cv_pct, effect_pct vs the baseline cell, within_noise, reading.

    relax: attributes (subject.id, env.id, env.node, env.label.<k>, trailing * allowed) that may differ between cells,
    for a comparison whose point is that difference (card count, build); a cell itself still holds one Result.
    Relaxing env.id drops the whole environment fingerprint from the check, not only the attribute being compared;
    derive pairs stay strict."""
    across = cmp_.get("across", [])
    cells = {}
    for row in rows:
        key = (tuple(_attr(row, k) for k in across), _attr(row, cmp_["by"]))
        cells.setdefault(key, []).append(row)
    for key, group in cells.items():
        _check_cell(key, group)
    for (ctx, by), group in cells.items():
        for other in [k[1] for k in cells if k[0] == ctx and k[1] != by]:
            why = incomparable(cells[(ctx, other)][0]["results"][0], group[0]["results"][0], pinning=False, relax=relax)
            if why:
                raise AnalysisError(
                    f"compare: cells {other!r} and {by!r} are not comparable: {', '.join(why)}; "
                    "narrow the selection or add the differing attribute to `across`"
                )
    vals = {k: [v for r in g for v in r["values"]] for k, g in cells.items()}
    levels = sorted({k[1] for k in vals})
    base = cmp_.get("baseline")
    if base is None:
        if "none" not in levels:
            raise AnalysisError(f"compare: no `none` level among {levels}; set compare.baseline")
        base = "none"
    elif base not in levels:
        raise AnalysisError(f"compare: baseline {base!r} is not one of {levels}")
    out = []
    for (ctx, by), v in sorted(vals.items()):
        mean, med, cv = spread(v)
        ref = vals.get((ctx, base))
        effect = noise = None
        if ref is not None and by != base:
            _, rmed, rcv = spread(ref)
            effect = (med - rmed) / abs(rmed) * 100 if rmed else None
            if effect is not None and cv is not None and rcv is not None:
                noise = abs(effect) <= max(cv, rcv)
        out.append(
            {
                **dict(zip(across, ctx)),
                cmp_["by"]: by,
                "n": len(v),
                "mean": mean,
                "median": med,
                "cv_pct": cv,
                "effect_pct": effect,
                "within_noise": noise,
                "reading": _reading(effect, direction) if noise is not None else None,
            }
        )
    return {
        "metric": cmp_["metric"],
        "unit": unit,
        "direction": direction,
        "by": cmp_["by"],
        "across": across,
        "baseline": base,
        "noise_rule": "within_noise: |effect_pct| <= max(cv_pct of the cell, cv_pct of the baseline cell)",
        "cells": out,
    }


def _unit(results, metric):
    return next(
        (
            r["status"]["aggregate"]["metrics"][metric].get("unit", "1")
            for r in results
            if metric in r["status"]["aggregate"].get("metrics", {})
        ),
        "1",
    )


def select_results(results, benchmarks):
    """Results of the named Benchmarks; an ingest-mode Result keeps the Benchmark name apart from the ingest name."""
    return [r for r in results if r["status"]["case"].get("benchmark", r["status"]["case"]["name"]) in benchmarks]


def analyze(spec, results, metrics=None):
    """Report for an Analysis spec over Result documents; raises AnalysisError when the inputs cannot be combined.

    metrics: the registry's metric table, for the direction of the compared metric.
    """
    metrics = metrics or {}
    sel = set(spec["select"]["benchmarks"])
    everything, results = results, select_results(results, sel)
    if not results:
        raise AnalysisError(f"no Result of {sorted(sel)}")
    derive, cmp_ = spec.get("derive"), spec.get("compare")
    warn = []
    report = {"metric": spec["metric"]}
    drows = None
    if derive:
        drows, unpaired = _derived_rows(derive, spec["metric"], results, warn)
        report["derive"] = {
            "name": derive["name"],
            "unit": _derive_unit(derive, _unit(results, spec["metric"])),
            "pairs": [r["pair"] for r in drows],
            "unpaired_results": unpaired,
        }
    if cmp_:
        m = cmp_["metric"]
        if derive and m == derive["name"]:
            rows, unit, base_metric = drows, report["derive"]["unit"], spec["metric"]
        else:
            rows, unit, base_metric = _plain_rows(results, m, warn), _unit(results, m), m
        if not rows:
            raise AnalysisError(f"compare: no values of {m}")
        report["compare"] = compare(
            cmp_, rows, unit, metrics.get(base_metric, {}).get("direction"), spec.get("relax", [])
        )
    if spec.get("reference"):
        report["reference"] = quality.reference_report(spec, everything, results, metrics)
    report["warnings"] = warn
    return report


def load_results(paths):
    """Result documents from JSON files and `run` output directories (cases/*/result.json)."""
    out = []
    for p in paths:
        p = Path(p)
        docs = [json.loads(f.read_text()) for f in sorted(p.glob("cases/*/result.json"))] if p.is_dir() else None
        if docs is None:
            doc = json.loads(p.read_text())
            docs = doc if isinstance(doc, list) else [doc]
        out += docs
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--results", required=True, action="append", help="Results JSON or a run output directory")
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("--central", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cat = engine.Catalog([a.catalog], a.pack, a.central)
    if cat.errors:
        print("\n".join(cat.errors), file=sys.stderr)
        return 2
    doc = cat.by.get(("Analysis", a.name))
    if doc is None:
        print(f"error: no Analysis named {a.name!r}", file=sys.stderr)
        return 2
    try:
        report = analyze(doc.spec, load_results(a.results), cat.reg.metrics)
    except (AnalysisError, OSError, ValueError, KeyError) as x:
        print(f"error: {x}", file=sys.stderr)
        return 2
    except (AttributeError, TypeError, IndexError) as x:
        print(f"error: malformed results: {type(x).__name__}: {x}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1) if a.json else engine.dump_yaml(report), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

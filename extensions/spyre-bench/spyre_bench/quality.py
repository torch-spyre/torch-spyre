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

"""Quality results: validity gates and the comparison of a score with a reference.

A quality score is a statistic over n samples, so its trust comes from the data checks (right dataset, every record
scored, no infrastructure errors, harness and model pinned) and its judgement from its standard error, not from
node cleanliness or timing. Everything here is advisory: a Result that is `invalid` or `suspect` is reported as such
and is never `regressed`.
"""

import math

# Gates a quality reference-purpose row must list: the exception to the clean-run rule is paid for with data checks.
REFERENCE_GATES = (
    "llm.dataset_digest",
    "llm.records_complete",
    "llm.no_errors",
    "llm.harness_pinned",
    "llm.model_pinned",
)
TRUNCATION_MAX = 0.01
STRONG_Z = 4.0  # a single run is `regressed` only this far beyond the reference; between tolerance and this: suspected
MDE_Z = 1.645  # one-sided 95%


def _first(samples, metric):
    vals = [x for x in (samples or {}).get(metric) or [] if isinstance(x, (int, float)) and math.isfinite(x)]
    return vals[0] if vals else None


def _digest(value):
    return str(value or "").removeprefix("sha256:").lower()


def _g_dataset(f):
    want, have = _digest(f.get("expect_digest")), _digest(f.get("dataset_digest"))
    if not want:
        return "suspect", "dataset_unpinned"
    if not have:
        return "invalid", "dataset_unreported"
    return None if have == want else ("invalid", "dataset_mismatch")


def _g_records(f):
    want, have = f.get("expect_records"), f.get("n_samples")
    if want is None:
        return "suspect", "records_undeclared"
    if have is None:
        return "invalid", "records_unreported"
    try:
        return None if int(have) == int(want) else ("invalid", "records_incomplete")
    except (TypeError, ValueError):
        return "invalid", "records_unreported"


def _g_errors(f):
    frac = f.get("error_frac")
    if frac is None:
        return "suspect", "errors_unreported"
    return None if frac == 0 else ("invalid", "infrastructure")


def _g_truncation(f):
    frac = f.get("truncated_frac")
    if frac is None:
        return "suspect", "truncation_unreported"
    return None if frac <= TRUNCATION_MAX else ("suspect", "truncation")


def _g_harness(f):
    return None if f.get("harness_version") and f.get("protocol") else ("suspect", "protocol_unpinned")


def _g_model(f):
    return None if f.get("model_revision") else ("suspect", "model_unpinned")


def _g_determinism(f):
    temp = f.get("temperature")
    if temp is None:
        return "suspect", "seed_unpinned"
    want = f.get("expect_temperature", 0)
    if want == 0:
        return None if temp == 0 else ("suspect", "sampling_in_greedy_case")
    if f.get("seed") is None:
        return "suspect", "seed_unpinned"
    return None if math.isclose(temp, want) else ("suspect", "sampling_mismatch")


GATES = {
    "llm.dataset_digest": _g_dataset,
    "llm.records_complete": _g_records,
    "llm.no_errors": _g_errors,
    "llm.no_truncation": _g_truncation,
    "llm.harness_pinned": _g_harness,
    "llm.model_pinned": _g_model,
    "llm.determinism": _g_determinism,
}


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def expected(case):
    """What the Benchmark declares about its scored input: records, selection digest, sampling temperature."""
    wl = case["workload"]
    inputs = [i for i in wl.get("inputs") or [] if isinstance(i, dict) and i.get("expect")]
    exp = inputs[0]["expect"] if inputs else {}
    return {
        "expect_records": exp.get("records"),
        "expect_digest": exp.get("selection_digest"),
        "expect_temperature": (wl.get("eval") or {}).get("temperature", 0),
    }


def repeat_facts(repeat, want):
    """The facts one ok repeat contributes: its reported samples and quality props against what the case declares."""
    s, p = repeat.get("samples") or {}, repeat.get("props") or {}
    n = _first(s, "llm.eval.n_samples")
    return {
        **want,
        "n_samples": None if n is None else int(n),
        "error_frac": _first(s, "llm.eval.error_frac"),
        "truncated_frac": _first(s, "llm.eval.truncated_frac"),
        "dataset_digest": p.get("run.quality.dataset_digest"),
        "harness_version": p.get("run.quality.harness_version"),
        "protocol": p.get("run.quality.protocol"),
        "model_revision": p.get("run.quality.model_revision"),
        "temperature": _float(p.get("run.quality.temperature")),
        "seed": _float(p.get("run.quality.seed")),
    }


def gate_caps(gates, facts):
    """(state, reason) for every listed gate that fails on the facts."""
    return [cap for g in gates if (cap := GATES[g](facts))]


def validity_of(policy, purpose, repeats_facts, measured=True, trust="internal"):
    """(state, reasons) of a quality Result: the worst cap any listed gate puts on any ok repeat.

    repeats_facts: repeat_facts() of each ok repeat. Clean, qualification, cv and warm-up never apply to a score.
    """
    from spyre_bench import engine

    req = policy["purposes"][purpose]["requires"]
    caps = []
    if not measured:
        caps.append(("invalid", "not_measured"))
    if len(repeats_facts) < req.get("min_repeats", 1):
        caps.append(("suspect", "repeats_below_min"))
    for facts in repeats_facts:
        caps += gate_caps(req.get("gates", []), facts)
    if trust == "external":
        caps.append(("external", "trust_external"))
    if not caps:
        return "valid", []
    reasons = list(dict.fromkeys(r for _, r in caps))
    return min(caps, key=lambda c: engine.VALIDITY_ORDER.index(c[0]))[0], reasons


def policy_errors(doc, reg):
    """Rule violations of a ValidityPolicy with applies_to: [quality]."""
    errs = []
    for name, row in doc.spec["purposes"].items():
        w = f"{doc.where} purposes.{name}"
        if name not in reg.vocab["purposes"]:
            errs.append(f"{w}: not a registered purpose {sorted(reg.vocab['purposes'])}")
        req = row["requires"]
        for k in ("max_cv_pct", "discard_warmup"):
            if k in req:
                errs.append(f"{w}: requires.{k} has no meaning for a quality score")
        for k in ("qualification", "clean", "placement"):
            if req.get(k, "any") not in ("any", "optional"):
                errs.append(f"{w}: requires.{k} has no effect on a quality score; use any")
        for g in req.get("gates", []):
            if g not in GATES:
                errs.append(f"{w}: requires.gates: {g!r} is not a quality gate {sorted(GATES)}")
            elif g not in reg.gates:
                errs.append(f"{w}: requires.gates: {g!r} is not registered by a loaded pack")
        if name == "pr" and (row["baseline"] or row["reference"]):
            errs.append(f"{w}: a PR run can never be a baseline or reference")
        if row["reference"]:
            if row["floor"] != "valid" or not row["baseline"]:
                errs.append(f"{w}: a reference purpose needs floor valid and baseline true")
            missing = [g for g in REFERENCE_GATES if g not in req.get("gates", [])]
            if missing:
                errs.append(f"{w}: a quality reference purpose must list the gates {missing}")
    return errs


def case_errors(case):
    """Rule violations of a quality case's workload: a greedy case carries no seed in its identity."""
    ev = (case["workload"].get("eval") or {}) if case["type"] == "quality" else {}
    if ev.get("temperature") == 0 and "seed" in ev:
        return ["eval.seed must not be set when temperature is 0 (a seed would split a greedy series)"]
    return []


def binomial_stderr(p, n):
    """Standard error of a proportion; a half-count shrink keeps it above zero at p = 0 or 1."""
    if not n or not 0 <= p <= 1:
        return None
    p = (p * n + 0.5) / (n + 1) if p in (0, 1) else p
    return math.sqrt(p * (1 - p) / n)


def score_of(result, metric, stderr_metric=None):
    """(score, stderr, n) of a Result's aggregate; stderr from the stderr metric, else binomial from n."""
    agg = result["status"]["aggregate"]["metrics"]
    if metric not in agg:
        return None, None, None
    score = agg[metric]["value"]
    n = agg.get("llm.eval.n_samples", {}).get("value")
    se = agg.get(stderr_metric or metric + "_stderr", {}).get("value")
    if se is None and n and score is not None:
        se = binomial_stderr(score, n)
    return score, se, n


def judge_against(score, se, ref, ref_se, direction, tol, validity=None, reasons=()):
    """One row: how the score stands against a reference, in combined standard errors.

    status: invalid | suspected (validity), no_stderr, no_power, regressed, improved, pass. `regressed` needs a valid-enough
    Result and a shortfall at least STRONG_Z combined standard errors; a smaller shortfall past the tolerance is
    `suspected` (single_run).
    """
    row = {"score": score, "stderr": se, "reference": ref, "reference_stderr": ref_se, "validity": validity}
    if validity == "invalid":
        return {**row, "status": "invalid", "reason": ",".join(reasons)}
    if validity not in (None, "valid"):  # only data that passed every check can support a regression
        return {**row, "status": "suspected", "reason": ",".join(reasons) or f"validity_{validity}"}
    if score is None or ref is None:
        return {**row, "status": "no_value", "reason": "metric_missing"}
    if direction not in ("higher_is_better", "lower_is_better"):
        return {**row, "status": "no_value", "reason": "no_direction"}
    delta = score - ref
    worse = -delta if direction == "higher_is_better" else delta
    sigma = math.hypot(se, ref_se) if se is not None and ref_se is not None else 0.0
    if not sigma > 0:
        return {**row, "delta": delta, "worse": worse, "status": "no_stderr", "reason": "stderr_missing"}
    allow = max(tol.get("stderr", 0.0) * sigma, tol.get("abs", 0.0))
    z = worse / sigma
    row |= {"delta": delta, "worse": worse, "sigma": sigma, "z_worse": z, "allowance": allow}
    if worse > allow:
        if z >= STRONG_Z:
            return {**row, "status": "regressed", "reason": ""}
        return {**row, "status": "suspected", "reason": "single_run"}
    if "abs" in tol and MDE_Z * sigma > tol["abs"]:
        return {**row, "mde": MDE_Z * sigma, "status": "no_power", "reason": "underpowered"}
    if worse < -allow:
        return {**row, "status": "improved", "reason": ""}
    return {**row, "status": "pass", "reason": ""}


def reference_errors(spec, case_names, metrics, w):
    """Catalog errors of an Analysis `reference:` block."""
    ref = spec.get("reference")
    if ref is None:
        return []
    errs = []
    if bool(ref.get("case")) == ("value" in ref):
        errs.append(f"{w}: reference needs exactly one of case or value")
    if ref.get("case") and ref["case"] not in case_names:
        errs.append(f"{w}: reference.case {ref['case']!r} is not a case")
    if "value" in ref and not ("stderr" in ref or "n" in ref):
        errs.append(f"{w}: an inline reference needs stderr or n")
    tol = ref["tolerance"]
    if not tol or any(not v > 0 for v in tol.values()):
        errs.append(f"{w}: reference.tolerance needs stderr or abs, each above 0")
    m = metrics.get(spec["metric"])
    if m is None or m["category"] != "quality" or m["direction"] == "none":
        errs.append(
            f"{w}: reference compares a registered quality metric with a direction; {spec['metric']} is not one"
        )
    sm = ref.get("stderr_metric")
    if sm and sm not in metrics:
        errs.append(f"{w}: reference.stderr_metric {sm!r} is not registered")
    for k in ("derive", "compare", "decompose"):
        if spec.get(k):
            errs.append(f"{w}: reference cannot be combined with {k}")
    return errs


def _validity(result):
    v = result["status"].get("validity") or {}
    return v.get("state"), v.get("reasons") or []


def reference_report(spec, results, selected, metrics):
    """The `reference` section of an Analysis report: one row per selected Result."""
    from spyre_bench.analysis import AnalysisError

    ref, metric = spec["reference"], spec["metric"]
    smetric = ref.get("stderr_metric")
    direction = metrics.get(metric, {}).get("direction")
    bad_ref = None
    if ref.get("case"):
        mates = [r for r in results if r["status"]["case"].get("case") == ref["case"]]
        if len(mates) != 1:
            raise AnalysisError(
                f"reference: {len(mates)} Results of case {ref['case']!r} in the results file; pass exactly one "
                "(the Result must carry status.case.case)"
            )
        rscore, rse, rn = score_of(mates[0], metric, smetric)
        label = ref["case"]
        skip = mates[0]
        if _validity(mates[0])[0] != "valid":
            bad_ref = "reference_not_valid"
    else:
        rscore = ref["value"]
        rse = ref.get("stderr", binomial_stderr(rscore, ref.get("n")))
        rn, label, skip = ref.get("n"), ref.get("label") or "inline", None
    rows = []
    for r in selected:
        if r is skip:
            continue
        score, se, n = score_of(r, metric, smetric)
        state, reasons = _validity(r)
        if state is None:  # no policy judged it: never trust a score nothing checked
            state, reasons = "suspect", ["validity_unjudged"]
        elif bad_ref and state not in ("invalid", "suspect"):
            state, reasons = "suspect", [bad_ref]
        rows.append(
            {
                "case": r["status"]["case"].get("case", r["status"]["case"]["name"]),
                "n": n,
                **judge_against(score, se, rscore, rse, direction, ref["tolerance"], state, reasons),
            }
        )
    if not rows:
        raise AnalysisError("reference: no Result besides the reference itself")
    return {
        "metric": metric,
        "direction": direction,
        "reference": {"label": label, "value": rscore, "stderr": rse, "n": rn},
        "tolerance": ref["tolerance"],
        "rule": f"regressed needs a shortfall past the tolerance and at least {STRONG_Z:g} combined standard errors; "
        "invalid and suspect Results are never regressed",
        "rows": rows,
    }

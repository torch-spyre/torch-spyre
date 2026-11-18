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

"""Metric context: the facets a number must carry, the context line, comparability and the validate lint.

A metric class (vocabulary file core/metric-context.yaml) lists the facets its numbers must carry. A record is a dict
{metric, value, unit, statistic, n, ctx}; `ctx` maps facet ids to values. Required means KNOWN: a missing facet is
rendered as an empty chip and the number is never hidden, but it is excluded from comparisons.
"""

import fnmatch
import functools
from pathlib import Path

import yaml

VOCAB_PATH = Path(__file__).resolve().parent / "core" / "metric-context.yaml"
MISSING = (None, "", "unknown")


class Vocab:
    def __init__(self, facets, classes):
        self.facets = {f["id"]: f for f in facets}
        self.classes = classes

    @classmethod
    def load(cls, path=None):
        d = yaml.safe_load(Path(path or VOCAB_PATH).read_text(encoding="utf-8"))
        return cls(d["facets"], d["classes"])

    def classify(self, metric):
        m = (metric or "").lower()
        for c in self.classes:
            if any(fnmatch.fnmatchcase(m, g) for g in c["match"]):
                return c
        return None

    def class_by_id(self, cid):
        return next((c for c in self.classes if c["id"] == cid), None)

    def lint(self):
        """Vocabulary self-check: every class facet exists."""
        errs = []
        for c in self.classes:
            for f in c["required"] + c.get("optional", []):
                for alt in f.split("|"):
                    if alt not in self.facets:
                        errs.append(f"class {c['id']}: unknown facet {alt}")
        return errs


@functools.lru_cache(maxsize=1)
def default_vocab():
    return Vocab.load()


# ---------------------------------------------------------------- resolving values
def resolve(rec, vocab):
    """Facet values after derivation: (values, derived_ids). Unknown values are dropped."""
    vals = {k: v for k, v in (rec.get("ctx") or {}).items() if v not in MISSING}
    derived = set()
    st = rec.get("statistic")
    if st and "statistic" not in vals:
        vals["statistic"] = st
        derived.add("statistic")
    if rec.get("n") and "sample_n" not in vals:
        vals["sample_n"] = rec["n"]
        derived.add("sample_n")
    if "kv_span" not in vals and "input_len" in vals and "output_len" in vals:
        vals["kv_span"] = f"{int(vals['input_len'])}-{int(vals['input_len']) + int(vals['output_len'])}"
        derived.add("kv_span")
    if "cache_state" not in vals and "compile_cache" in vals and "shape_warmup" in vals:
        cc, sw = vals["compile_cache"], vals["shape_warmup"]
        vals["cache_state"] = (
            "warm" if (cc == "warm" and sw == "complete") else ("cold" if (cc == "cold" or sw == "none") else "partial")
        )
        derived.add("cache_state")
    return vals, derived


def _alts(req):
    return req.split("|")


def _applies(fid, vocab, vals):
    """A facet with `when: key=value` applies unless that key is known and differs."""
    cond = vocab.facets[fid].get("when")
    if not cond:
        return True
    k, _, v = cond.partition("=")
    return k not in vals or str(vals[k]).lower() == v.lower()


def missing(rec, vocab):
    """Required facets with no known value. A "a|b" requirement is met by either alternative."""
    cls = vocab.classify(rec["metric"])
    if cls is None:
        return ["metric class"]
    vals, _ = resolve(rec, vocab)
    return [
        r
        for r in cls["required"]
        if not any(a in vals for a in _alts(r)) and all(_applies(a, vocab, vals) for a in _alts(r))
    ]


def completeness(rec, vocab):
    cls = vocab.classify(rec["metric"])
    if cls is None:
        return 0, 0
    total = len(cls["required"])
    return total - len(missing(rec, vocab)), total


def is_complete(rec, vocab):
    return not missing(rec, vocab)


# ---------------------------------------------------------------- chips
def _k(n, approx=False):
    n = int(n)
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    if approx and n >= 1024:
        return f"~{n / 1024:.1f}k"
    return str(n)


def _short_build(b):
    b = str(b)
    s = b.lstrip("0")
    return s if len(s) >= 3 else b[-3:]


EMPTY = "\u2014"  # em dash: "required, not recorded". Never a legal value of any facet.
CLEAN_CHIP = {"enforced": "clean", "observed": "observed-clean", "none": "shared", "unknown": "clean?"}
STAT_CHIP = {
    "value": "",
    "mean": "mean",
    "median": "p50",
    "p50": "p50",
    "p90": "p90",
    "p95": "p95",
    "p99": "p99",
    "p100": "max",
    "std": "std",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "count": "",
    "ratio": "",
}
DEF_SHORT = {
    "arrival_to_first_token": "arrival",
    "send_to_first_token": "send",
    "token_gap": "gap",
    "tpot_mean": "mean",
}
# Fixed chip order; facets in the same tuple render as one chip.
ORDER = [
    ("cache_state",),
    ("window",),
    ("concurrency", "request_rate"),
    ("batch_size",),
    ("eval_task",),
    ("dataset",),
    ("sampling",),
    ("tool",),
    ("input_len", "output_len"),
    ("op_shape",),
    ("kv_span",),
    ("kv_ladder",),
    ("length_mode",),
    ("prefix_cache",),
    ("dtype",),
    ("tp",),
    ("model",),
    ("backend",),
    ("arch",),
    ("device_type",),
    ("device_count",),
    ("cpu_pinning",),
    ("mem_placement",),
    ("build",),
    ("clean",),
    ("purpose",),
    ("ratio_basis",),
    ("roofline_basis",),
    ("timer",),
    ("ttft_def",),
    ("itl_def",),
    ("token_basis",),
    ("statistic",),
    ("sample_n",),
]
LABEL = {
    "cache_state": "cache",
    "window": "window",
    "concurrency": "c",
    "request_rate": "rate",
    "batch_size": "bs",
    "eval_task": "task",
    "dataset": "data",
    "sampling": "sampling",
    "tool": "tool",
    "op_shape": "shape",
    "kv_span": "KV",
    "kv_ladder": "ladder",
    "length_mode": "lengths",
    "prefix_cache": "prefix-cache",
    "dtype": "dtype",
    "tp": "TP",
    "model": "model",
    "backend": "backend",
    "arch": "arch",
    "device_type": "device",
    "device_count": "cards",
    "cpu_pinning": "pin",
    "mem_placement": "mem",
    "build": "build",
    "clean": "clean",
    "purpose": "purpose",
    "ratio_basis": "ratio-of",
    "roofline_basis": "roofline-of",
    "timer": "timer",
    "ttft_def": "ttft",
    "itl_def": "itl",
    "token_basis": "tokens",
    "statistic": "stat",
    "sample_n": "n",
}
# Facets whose value is part of the head (value unit (stat, n)); they still count as required but need no chip.
IN_HEAD = {"statistic", "sample_n"}


def _chip_value(fid, v, derived):
    if fid == "kv_span":
        lo, _, hi = str(v).partition("-")
        return f"{_k(lo)}\u2192{_k(hi)}"
    if fid == "build":
        return _short_build(v)
    if fid == "clean":
        return CLEAN_CHIP.get(v, v)
    if fid == "cpu_pinning":
        return "on" if v != "none" else "off"
    if fid in ("ttft_def", "itl_def"):
        return DEF_SHORT.get(v, v)
    return str(v)


def _text(group, vals, derived):
    """Text of one chip for a tuple of facets; None when the chip has nothing known; empty slot via EMPTY."""
    if group == ("concurrency", "request_rate"):
        if "concurrency" in vals:
            return f"c={vals['concurrency']}"
        if "request_rate" in vals:
            return f"rate {vals['request_rate']}/s"
        return None
    if group == ("input_len", "output_len"):
        a, b = vals.get("input_len"), vals.get("output_len")
        if a is None and b is None:
            return None
        return f"in {_k(a) if a is not None else EMPTY}/out {_k(b) if b is not None else EMPTY}"
    fid = group[0]
    if fid not in vals:
        return None
    v = _chip_value(fid, vals[fid], derived)
    lab = LABEL[fid]
    if fid in ("cache_state", "dtype", "arch", "backend", "window") and fid != "window":
        return v
    if fid == "window":
        return f"window {v}"
    if fid == "device_count":
        return f"{v} card" + ("s" if str(v) != "1" else "")
    if fid == "tp":
        return f"TP{v}"
    if fid == "build":
        return f"build {v}"
    if fid == "clean":
        return v
    if fid == "batch_size":
        return f"bs={v}"
    if fid == "prefix_cache":
        return f"prefix-cache {v}"
    if fid == "mem_placement":
        return f"mem {v}"
    if fid in ("eval_task", "sampling", "tool"):
        return v
    return f"{lab} {v}"


def _empty_text(group):
    if group == ("concurrency", "request_rate"):
        return f"load {EMPTY}"
    if group == ("input_len", "output_len"):
        return f"in {EMPTY}/out {EMPTY}"
    return f"{LABEL[group[0]]} {EMPTY}"


def chips(rec, vocab, expanded=False):
    """Ordered (facet_ids, text, state) chips; state is 'known' or 'empty'.

    Every REQUIRED facet gets a chip: known ones show the value, missing ones show `<label> \u2014`.
    Expanded adds the known optional facets and judgement badges.
    """
    cls = vocab.classify(rec["metric"])
    vals, derived = resolve(rec, vocab)
    req = set()
    miss = set()
    for r in cls["required"] if cls else []:
        alts = _alts(r)
        if not all(_applies(a, vocab, vals) for a in alts):
            continue
        req.update(alts)
        if not any(a in vals for a in alts):
            miss.update(alts)
    opt = set(cls.get("optional", [])) if (cls and expanded) else set()
    out = []
    for group in ORDER:
        in_req = any(g in req for g in group)
        in_opt = any(g in opt and g in vals for g in group)
        if not (in_req or in_opt) or any(g in IN_HEAD for g in group):
            continue
        text = _text(group, vals, derived)
        if text is not None and not (group == ("input_len", "output_len") and EMPTY in text and not in_req):
            state = "empty" if EMPTY in text else "known"
            out.append((list(group), text, state))
        elif in_req:
            out.append((list(group), _empty_text(group), "empty"))
    if expanded and "trust" in vals:
        out.append((["trust"], f"trust {vals['trust']}", "known"))
    return out


def context_line(rec, vocab, expanded=False):
    return " \u00b7 ".join(t for _, t, _ in chips(rec, vocab, expanded))


def stat_label(rec):
    st = rec.get("statistic", "value")
    s = STAT_CHIP.get(st, st)
    n = rec.get("n")
    parts = [p for p in (s, f"n={n}" if n else "") if p]
    return " ".join(parts)


def render(rec, vocab):
    """One display string: value unit (stat, n), the context line with empty slots, then the badge if incomplete.

    The number is ALWAYS shown. Exclusion from comparisons is a separate rule (see compare)."""
    head = f"{rec['value']} {rec.get('unit', '')}".strip()
    st = stat_label(rec)
    if st:
        head += f" ({st})"
    line = context_line(rec, vocab)
    miss = missing(rec, vocab)
    out = f"{head} \u00b7 {line}" if line else head
    if miss:
        out += f" [CONTEXT INCOMPLETE: {', '.join(miss)}]"
    return out


def expanded_view(rec, vocab):
    """Group -> list of (facet, value, flag) for the expanded panel; flag in {'', 'derived', 'required, not recorded'}."""
    cls = vocab.classify(rec["metric"])
    vals, derived = resolve(rec, vocab)
    rows = {g: [] for g in ("workload", "state", "build", "environment", "method", "judgement")}
    req = set()
    for r in cls["required"] if cls else []:
        alts = _alts(r)
        if all(_applies(a, vocab, vals) for a in alts):
            req.update(alts)
    show = []
    for r in cls["required"] if cls else []:
        alts = _alts(r)
        if not all(_applies(a, vocab, vals) for a in alts):
            continue
        present = [a for a in alts if a in vals]
        show.extend(present if present else [r])
    show += [o for o in (cls.get("optional", []) if cls else []) if o in vals]
    for fid in show:
        f = vocab.facets[_alts(fid)[0]]
        if fid in vals:
            rows[f["group"]].append((fid, vals[fid], "derived" if fid in derived else ""))
        else:
            rows[f["group"]].append((fid, EMPTY, "required, not recorded"))
    return rows


def length_drift(rec, tol=0.1):
    """Measured vs requested lengths: [(facet, requested, actual)] where they differ by more than `tol` (relative)."""
    vals, _ = resolve(rec, default_vocab())
    out = []
    for req, act in (("input_len", "actual_input_len"), ("output_len", "actual_output_len")):
        if req in vals and act in vals and abs(float(vals[act]) - float(vals[req])) > tol * float(vals[req]):
            out.append((act, vals[req], vals[act]))
    return out


def page_header(recs, vocab):
    """Split the chips of several records into (common chips, per-record varying chips) for one page."""
    per = [[(tuple(ids), t) for ids, t, _ in chips(r, vocab)] for r in recs]
    if not per:
        return [], []
    common = [c for c in per[0] if all(c in p for p in per[1:])]
    return [t for _, t in common], [[t for c, t in p if (c, t) not in common] for p in per]


# ---------------------------------------------------------------- comparability
def compare(a, b, vocab, vary=("build",)):
    """Comparability of two records.

    status: 'comparable' (verdict allowed), 'differs' (side by side, differences listed),
            'incomplete' (a side lacks required context), 'different_metric'.
    `vary` names the facets the comparison is ABOUT (default: the build); they may differ.
    """
    ca, cb = vocab.classify(a["metric"]), vocab.classify(b["metric"])
    if ca is None or cb is None or ca["id"] != cb["id"] or a["metric"] != b["metric"]:
        return {"status": "different_metric", "differs": [], "unknown": [], "line": "different metric"}
    ma, mb = missing(a, vocab), missing(b, vocab)
    if ma or mb:
        unk = sorted(set(ma) | set(mb))
        return {
            "status": "incomplete",
            "differs": [],
            "unknown": unk,
            "line": "not comparable: context unknown: " + ", ".join(unk),
        }
    va, _ = resolve(a, vocab)
    vb, _ = resolve(b, vocab)
    vary = set(vary)
    differs = []
    for r in ca["required"]:
        for alt in _alts(r):
            if alt in vary or vocab.facets[alt].get("compare") == "note" or not _applies(alt, vocab, va):
                continue
            x, y = va.get(alt), vb.get(alt)
            if (x is None) != (y is None) or (x is not None and str(x).lower() != str(y).lower()):
                differs.append((alt, x, y))
    also = []
    for fid in dict.fromkeys(
        ca.get("optional", []) + [f for f in vocab.facets if vocab.facets[f].get("compare") == "note"]
    ):
        if fid in vary:
            continue
        x, y = va.get(fid), vb.get(fid)
        if x is not None and y is not None and str(x).lower() != str(y).lower():
            also.append((fid, x, y))
    if differs:
        txt = ", ".join(f"{_name(f)} {_fmt(f, x)} vs {_fmt(f, y)}" for f, x, y in differs)
        return {"status": "differs", "differs": differs, "also": also, "unknown": [], "line": "differs: " + txt}
    line = "comparable" + (
        "; note, not required but differs: " + ", ".join(f"{f} {x} vs {y}" for f, x, y in also) if also else ""
    )
    return {"status": "comparable", "differs": [], "also": also, "unknown": [], "line": line}


def _name(f):
    return {"cache_state": "cold/warm", "kv_span": "KV", "tp": "TP", "input_len": "input", "output_len": "output"}.get(
        f, f
    )


def _fmt(f, v):
    if f == "kv_span" and v is not None:
        return _chip_value(f, v, set())
    if f in ("input_len", "output_len") and v is not None:
        return _k(v)
    return "unset" if v is None else str(v)


# ---------------------------------------------------------------- table scan (used by the deck/report check)
def scan_rows(rows, vocab):
    """rows: list of records. Returns (n_complete, n_total, [(index, missing)])."""
    bad = []
    for i, r in enumerate(rows):
        m = missing(r, vocab)
        if m:
            bad.append((i, m))
    return len(rows) - len(bad), len(rows), bad


# ---------------------------------------------------------------- pack declarations and the validate lint
def metric_class(spec, vocab):
    """The class of a registered pack metric: its declared `class`, else the name-glob match."""
    cid = spec.get("class")
    return vocab.class_by_id(cid) if cid else vocab.classify(spec["name"])


def required_facets(spec, vocab):
    """Facet requirements of a pack metric: its own `context.required`, else the class default."""
    own = (spec.get("context") or {}).get("required")
    if own is not None:
        return list(own)
    cls = metric_class(spec, vocab)
    return list(cls["required"]) if cls else []


def pack_errors(metrics, vocab):
    """Errors for pack metric declarations: unknown class, unknown facet ids."""
    errs = []
    for m in metrics:
        if m.get("class") and vocab.class_by_id(m["class"]) is None:
            errs.append(f"metric {m['name']}: class {m['class']!r} is not a metric class")
        ctx = m.get("context") or {}
        for fid in ctx.get("required", []) + ctx.get("optional", []):
            for alt in fid.split("|"):
                if alt not in vocab.facets:
                    errs.append(f"metric {m['name']}: context facet {alt!r} is not in the vocabulary")
    return errs


def lint_catalog(cat, vocab=None):
    """Warnings for Benchmarks whose metrics need author-declared facets the Benchmark does not state.

    Only facets with `auto: declared` can be missing here; engine, probe, derive and ingest facets are captured.
    A Benchmark states them in its `context:` block. Warn only: an incomplete number is still shown and stored.
    """
    vocab = vocab or default_vocab()
    benches = {b.name: b for b in cat.of("Benchmark")}
    warns, seen = [], set()
    for b in benches.values():
        for k in b.spec.get("context") or {}:
            if k not in vocab.facets:
                warns.append(f"Benchmark/{b.name}: context.{k} is not a facet in the metric-context vocabulary")
    for c in cat.cases:
        b = benches.get(c["benchmark"])
        if b is None:
            continue
        stated = set(b.spec.get("context") or {})
        for name in cat.case_metrics(c)[0]:
            if (b.name, name) in seen:
                continue
            seen.add((b.name, name))
            spec = cat.reg.metrics.get(name)
            if spec is None:
                continue
            cls = metric_class(spec, vocab)
            if cls is None:
                warns.append(f"Benchmark/{b.name}: metric {name} matches no metric class")
                continue
            for req in required_facets(spec, vocab):
                alts = req.split("|")
                if any(a in stated or vocab.facets[a].get("auto") != "declared" for a in alts if a in vocab.facets):
                    continue
                warns.append(
                    f"Benchmark/{b.name}: metric {name} (class {cls['id']}) needs facet {req}; "
                    f"declare it under the Benchmark's context: block"
                )
    return warns

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
import math
import functools
from pathlib import Path

import yaml

VOCAB_PATH = Path(__file__).resolve().parent / "core" / "metric-context.yaml"
# Placeholder spellings that mean "not recorded". "none" is also one unless it is a legal value of the facet.
PLACEHOLDERS = {"", "unknown", "unset", "null", "n/a", "na", "?", "-", "\u2014", "\u2013", "none"}


class Vocab:
    def __init__(self, facets, classes):
        self.facets = {f["id"]: f for f in facets}
        self.classes = classes
        self.specs = {}
        self.builtin = None

    @classmethod
    def load(cls, path=None):
        d = yaml.safe_load(Path(path or VOCAB_PATH).read_text(encoding="utf-8"))
        return cls(d["facets"], d["classes"])

    def classify(self, metric):
        m = metric.lower() if isinstance(metric, str) else ""
        for c in self.classes:
            if any(fnmatch.fnmatchcase(m, g) for g in c["match"]):
                return c
        return None

    def class_by_id(self, cid):
        return next((c for c in self.classes if c["id"] == cid), None)

    def bind(self, metrics, builtin=None):
        """A copy that resolves classes and required facets through registered pack metric specs.

        `builtin` names the metrics of the engine's own packs (None: all). Only those are glob-classified when the
        pack declares no `class`; any other metric without a class is "unclassified" and has no requirements.
        """
        v = Vocab(list(self.facets.values()), self.classes)
        v.specs = dict(metrics)
        v.builtin = None if builtin is None else set(builtin)
        return v

    def cls_for(self, metric):
        """The class dict that governs a metric: its pack declaration when bound, else the name-glob match.

        A pack `context.required` / `context.optional` replaces the class lists; `x-` classes and facets carry no
        requirements.
        """
        metric = metric if isinstance(metric, str) else ""
        spec = self.specs.get(metric)
        if spec is None:
            return self.classify(metric)
        cid = spec.get("class")
        if cid and cid.startswith("x-"):
            base = {"id": cid, "match": [], "required": [], "optional": []}
        else:
            base = self.class_by_id(cid) if cid else self.classify(metric)
        ctx = spec.get("context") or {}
        if not cid and self.builtin is not None and metric not in self.builtin:
            base = None if "required" in ctx else {"id": "unclassified", "match": [], "required": [], "optional": []}
        if base is None and "required" not in ctx:
            return None
        base = base or {"id": "custom", "match": [], "required": [], "optional": []}
        req = ctx.get("required", base["required"])
        opt = ctx.get("optional", base.get("optional", []))
        keep = lambda fs: [f for f in fs if not any(a.startswith("x-") for a in f.split("|"))]  # noqa: E731
        return {**base, "required": keep(req), "optional": keep(opt)}

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
MAX_DEPTH = 8  # deepest container nesting that is read; deeper is malformed
MAX_NODES = 10_000  # items visited in one container before it is malformed


class _Box:
    """A container value reduced to what the renderer needs: its length, whether all leaves are unrecorded."""

    __slots__ = ("count", "unrecorded")

    def __init__(self, count, unrecorded):
        self.count, self.unrecorded = count, unrecorded

    def __repr__(self):
        return f"<{self.count} values>"


def _safe_str(v):
    try:
        return str(v)
    except Exception:  # noqa: BLE001 - hostile __str__
        return "?"


def _nonfinite(v):
    """NaN or infinity, including numpy scalars (duck-typed: anything float() accepts that is not a str)."""
    if isinstance(v, (str, bytes, bool)):
        return False
    try:
        return not math.isfinite(float(v))
    except Exception:  # noqa: BLE001 - duck-typed conversion may raise anything but BaseException
        return False


def _clen(v):
    """Length of a non-string container, or None when v is not one (or its length cannot be read)."""
    if isinstance(v, _Box):
        return v.count
    try:
        if isinstance(v, (str, bytes)) or not hasattr(v, "__len__"):
            return None
        return len(v)
    except Exception:  # noqa: BLE001 - hostile or 0-d array-likes
        return None


def _items(v, n):
    try:
        return list(v.values()) if hasattr(v, "values") and callable(v.values) else [v[i] for i in range(n)]
    except Exception:  # noqa: BLE001
        try:
            return list(v)
        except Exception:  # noqa: BLE001
            return None


def _leaf_unrecorded(x):
    return x is None or _nonfinite(x) or (isinstance(x, str) and x.strip().lower() in PLACEHOLDERS)


def _walk(v):
    """(length, all_unrecorded) of a container, or None when v is not one.

    A cycle, more than MAX_NODES items, nesting deeper than MAX_DEPTH, or an unreadable level makes the value
    malformed, reported as all_unrecorded.
    """
    n = _clen(v)
    if n is None:
        return None
    seen, budget, stack, all_unrec = {id(v)}, MAX_NODES, [(v, n, 0)], True
    while stack:
        node, size, depth = stack.pop()
        items = _items(node, size)
        if items is None:
            return n, True
        for x in items:
            budget -= 1
            if budget < 0:
                return n, True
            m = _clen(x)
            if m is None:
                all_unrec = all_unrec and _leaf_unrecorded(x)
            elif id(x) in seen or depth + 1 > MAX_DEPTH:
                return n, True
            else:
                seen.add(id(x))
                stack.append((x, m, depth + 1))
    return n, all_unrec


def _sanitize(v):
    """A context value made safe to handle: scalars as is, containers as a _Box, other objects as float or text."""
    if v is None or isinstance(v, (str, int, float, bool, _Box)):
        return v
    w = _walk(v)
    if w is not None:
        return _Box(*w)
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return _safe_str(v)


def _norm(rec):
    """A record of any shape as {metric: str, value, unit, statistic: str|None, n, ctx: dict}; never raises."""
    if not isinstance(rec, dict):
        rec = {}
    st, n, ctx = rec.get("statistic"), rec.get("n"), rec.get("ctx")
    return {
        "metric": rec["metric"] if isinstance(rec.get("metric"), str) else "",
        "value": _safe_str(rec.get("value", "")),
        "unit": rec.get("unit") if isinstance(rec.get("unit"), str) else "",
        "statistic": st if isinstance(st, str) and st else None,
        "n": n if isinstance(n, (int, float, str)) and not isinstance(n, bool) else None,
        "ctx": {k: _sanitize(v) for k, v in ctx.items() if isinstance(k, str)} if isinstance(ctx, dict) else {},
    }


def _unrecorded(fid, v, vocab):
    """True for None, placeholder spellings, NaN/inf, empty containers and containers of unrecorded leaves.

    "none" is a value where the facet lists it in `values` or sets `none_ok`; everywhere else (model, ...) it is a
    placeholder.
    """
    if v is None or _nonfinite(v):
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "none":
            f = vocab.facets.get(fid) or {}
            return not (f.get("none_ok") or "none" in [str(x).lower() for x in f.get("values", [])])
        return s in PLACEHOLDERS
    if isinstance(v, _Box):
        return v.count == 0 or v.unrecorded
    return False


def _show(v):
    n = _clen(v)
    return _safe_str(v) if n is None else f"{n} values"


def _canon(fid, v, vocab):
    """YAML 1.1 reads an unquoted `off` as False: an enum facet that lists "off" takes False as "off"."""
    if v is False and "off" in [str(x).lower() for x in (vocab.facets.get(fid) or {}).get("values", [])]:
        return "off"
    return v


def resolve(rec, vocab):
    """Facet values after derivation: (values, derived_ids). Unknown values are dropped."""
    rec = _norm(rec)
    vals = {k: _canon(k, v, vocab) for k, v in rec["ctx"].items() if not _unrecorded(k, v, vocab)}
    derived = set()
    st = rec.get("statistic")
    if st and "statistic" not in vals:
        vals["statistic"] = st
        derived.add("statistic")
    if rec.get("n") and "sample_n" not in vals:
        vals["sample_n"] = rec["n"]
        derived.add("sample_n")
    if "kv_span" not in vals and "input_len" in vals and "output_len" in vals:
        try:
            lo = int(float(vals["input_len"]))
            vals["kv_span"] = f"{lo}-{lo + int(float(vals['output_len']))}"
            derived.add("kv_span")
        except (TypeError, ValueError, OverflowError):
            pass
    if "cache_state" not in vals and "compile_cache" in vals and "shape_warmup" in vals:
        cc, sw = vals["compile_cache"], vals["shape_warmup"]
        vals["cache_state"] = (
            "warm" if (cc == "warm" and sw == "complete") else ("cold" if (cc == "cold" or sw == "none") else "partial")
        )
        derived.add("cache_state")
    return vals, derived


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _same(fid, vocab, x, y):
    """Equality of two known values: numeric facets compare as numbers, others case-insensitively as text."""
    if (vocab.facets.get(fid) or {}).get("type") in ("int", "float"):
        a, b = _num(x), _num(y)
        if a is not None and b is not None:
            return a == b
    return str(x).lower() == str(y).lower()


def _alts(req):
    return req.split("|")


def _applies(fid, vocab, vals):
    """A facet with `when: key=value` applies unless that key is known and differs."""
    cond = vocab.facets.get(fid, {}).get("when")
    if not cond:
        return True
    k, _, v = cond.partition("=")
    return k not in vals or str(vals[k]).lower() == v.lower()


def missing(rec, vocab):
    """Required facets with no known value. A "a|b" requirement is met by either alternative."""
    rec = _norm(rec)
    cls = vocab.cls_for(rec["metric"])
    if cls is None:
        return ["metric class"]
    vals, _ = resolve(rec, vocab)
    return [
        r
        for r in cls["required"]
        if not any(a in vals for a in _alts(r)) and all(_applies(a, vocab, vals) for a in _alts(r))
    ]


def completeness(rec, vocab):
    rec = _norm(rec)
    cls = vocab.cls_for(rec["metric"])
    if cls is None:
        return 0, 0
    total = len(cls["required"])
    return total - len(missing(rec, vocab)), total


def is_complete(rec, vocab):
    return not missing(rec, vocab)


# ---------------------------------------------------------------- chips
def _k(n, approx=False):
    try:
        n = int(float(n))
    except (TypeError, ValueError, OverflowError):
        return _show(n)
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
CLEAN_CHIP = {"enforced": "clean", "observed": "observed-clean", "none": "shared"}
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
        lo, sep, hi = str(v).partition("-")
        if not sep:
            return str(v)
        return f"{_k(lo)}\u2192{_k(hi)}"
    if fid == "build":
        return _short_build(v)
    if fid == "clean":
        return CLEAN_CHIP.get(str(v), str(v))
    if isinstance(v, bool):
        return "on" if v else "off"
    if fid == "cpu_pinning":
        return "off" if str(v).strip().lower() == "none" else "on"
    if fid in ("ttft_def", "itl_def"):
        return DEF_SHORT.get(str(v), str(v))
    return str(v)


def _label(fid):
    return LABEL.get(fid, fid.replace("_", "-"))


def _order(vocab):
    """The fixed chip order, then every other facet in vocabulary order, so a required facet always has a slot."""
    known = {f for g in ORDER for f in g}
    return ORDER + [(f,) for f in vocab.facets if f not in known]


def _text(group, vals, derived):
    """Text of one chip for a tuple of facets; None when the chip has nothing known; empty slot via EMPTY."""
    if group == ("concurrency", "request_rate"):
        if "concurrency" in vals:
            return f"c={_show(vals['concurrency'])}"
        if "request_rate" in vals:
            return f"rate {_show(vals['request_rate'])}/s"
        return None
    if group == ("input_len", "output_len"):
        a, b = vals.get("input_len"), vals.get("output_len")
        if a is None and b is None:
            return None
        return f"in {_k(a) if a is not None else EMPTY}/out {_k(b) if b is not None else EMPTY}"
    fid = group[0]
    if fid not in vals:
        return None
    if _clen(vals[fid]) is not None:
        return f"{_label(fid)} {_show(vals[fid])}"
    v = _chip_value(fid, vals[fid], derived)
    lab = _label(fid)
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
    return f"{_label(group[0])} {EMPTY}"


def chips(rec, vocab, expanded=False):
    """Ordered (facet_ids, text, state) chips; state is 'known' or 'empty'.

    Every REQUIRED facet gets a chip: known ones show the value, missing ones show `<label> \u2014`.
    Expanded adds the known optional facets and judgement badges.
    """
    rec = _norm(rec)
    cls = vocab.cls_for(rec["metric"])
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
    for group in _order(vocab):
        in_req = any(g in req for g in group)
        in_opt = any(g in opt and g in vals for g in group)
        if not (in_req or in_opt) or any(g in IN_HEAD for g in group):
            continue
        text = _text(group, vals, derived)
        text = None if text is None else str(text)
        half = group == ("input_len", "output_len") and not ("input_len" in vals and "output_len" in vals)
        if text is not None and not (half and not in_req):
            out.append((list(group), text, "empty" if half else "known"))
        elif in_req:
            out.append((list(group), _empty_text(group), "empty"))
    if expanded and "trust" in vals:
        out.append((["trust"], f"trust {vals['trust']}", "known"))
    return out


def context_line(rec, vocab, expanded=False):
    return " \u00b7 ".join(t for _, t, _ in chips(rec, vocab, expanded))


def stat_label(rec):
    rec = _norm(rec)
    st = rec["statistic"] or "value"
    s = STAT_CHIP.get(st, st)
    n = rec["n"]
    parts = [p for p in (s, f"n={n}" if n else "") if p]
    return " ".join(parts)


def render(rec, vocab):
    """One display string: value unit (stat, n), the context line with empty slots, then the badge if incomplete.

    The number is ALWAYS shown. Exclusion from comparisons is a separate rule (see compare)."""
    rec = _norm(rec)
    head = f"{rec['value']} {rec['unit']}".strip()
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
    rec = _norm(rec)
    cls = vocab.cls_for(rec["metric"])
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
        f = vocab.facets.get(_alts(fid)[0])
        if f is None:
            continue
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
        a, b = _num(vals.get(act)), _num(vals.get(req))
        if a is not None and b is not None and abs(a - b) > tol * b:
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
    a, b = _norm(a), _norm(b)
    ca, cb = vocab.cls_for(a["metric"]), vocab.cls_for(b["metric"])
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
    vary = {vary} if isinstance(vary, str) else set(vary)
    differs = []
    for r in ca["required"]:
        for alt in _alts(r):
            if (
                alt in vary
                or vocab.facets.get(alt, {}).get("compare") == "note"
                or not (_applies(alt, vocab, va) or _applies(alt, vocab, vb))
            ):
                continue
            x, y = va.get(alt), vb.get(alt)
            if (x is None) != (y is None) or (x is not None and not _same(alt, vocab, x, y)):
                differs.append((alt, x, y))
    also = []
    for fid in dict.fromkeys(
        ca.get("optional", []) + [f for f in vocab.facets if vocab.facets[f].get("compare") == "note"]
    ):
        if fid in vary:
            continue
        x, y = va.get(fid), vb.get(fid)
        if x is not None and y is not None and not _same(fid, vocab, x, y):
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
    """The class that governs a pack metric spec (declared `class`/`context` first, else the name glob)."""
    return vocab.bind({spec["name"]: spec}).cls_for(spec["name"])


def required_facets(spec, vocab):
    """Facet requirements of a pack metric: its own `context.required`, else the class default."""
    cls = metric_class(spec, vocab)
    return list(cls["required"]) if cls else []


def _custom(name):
    return str(name).startswith("x-")


def pack_errors(metrics, vocab):
    """Errors for pack metric declarations. Classes and facets are closed except the `x-` namespace."""
    errs = []
    for m in metrics:
        cid = m.get("class")
        if cid and not _custom(cid) and vocab.class_by_id(cid) is None:
            errs.append(f"metric {m['name']}: class {cid!r} is not a metric class")
        ctx = m.get("context") or {}
        for fid in ctx.get("required", []) + ctx.get("optional", []):
            for alt in fid.split("|"):
                if not _custom(alt) and alt not in vocab.facets:
                    errs.append(f"metric {m['name']}: context facet {alt!r} is not in the vocabulary")
    return errs


def _context_value_warning(f, key, v):
    """Why a Benchmark `context:` value is not usable, or None. YAML 1.1 turns unquoted off/on into booleans."""
    legal = [str(x).lower() for x in f.get("values", [])]
    if v is None:
        return f"context.{key} has no value (YAML null); give a value or remove the key"
    if f.get("type") != "enum" or not legal:
        return None
    if v is False and "off" in legal:
        return None
    if v is True and "off" in legal:
        return f"context.{key}: true is ambiguous (YAML reads an unquoted on as true); quote one of {legal}"
    if str(v).lower() not in legal:
        return f"context.{key}={v!r} is not one of {legal}"
    return None


def lint_catalog(cat, vocab=None, detail=False):
    """Warnings about metric context in a catalog. Never an error.

    Always reported: a Benchmark `context:` key that is not a facet. Author-only facets a Benchmark does not state, and
    metrics without a class, are one summary line unless `detail`. Engine, probe, derive and ingest facets are captured
    and not checked.
    """
    vocab = (vocab or default_vocab()).bind(cat.reg.metrics, cat.reg.builtin_metrics)
    benches = {b.name: b for b in cat.of("Benchmark")}
    warns, notes, seen, undeclared = [], [], set(), set()
    for b in benches.values():
        for k in b.spec.get("context") or {}:
            if k not in vocab.facets and not _custom(k):
                warns.append(f"Benchmark/{b.name}: context.{k} is not a facet in the metric-context vocabulary")
            elif k in vocab.facets:
                w = _context_value_warning(vocab.facets[k], k, b.spec["context"][k])
                if w:
                    warns.append(f"Benchmark/{b.name}: {w}")
    for c in cat.cases:
        b = benches.get(c["benchmark"])
        if b is None:
            continue
        stated = set(b.spec.get("context") or {})
        for name in cat.case_metrics(c)[0]:
            if (b.name, name) in seen or name not in cat.reg.metrics:
                continue
            seen.add((b.name, name))
            cls = vocab.cls_for(name)
            if cls is not None and cls["id"] == "unclassified":
                continue
            if cls is None:
                undeclared.add(b.name)
                notes.append(f"Benchmark/{b.name}: metric {name} matches no metric class")
                continue
            for req in cls["required"]:
                alts = req.split("|")
                if any(a in stated or vocab.facets[a].get("auto") != "declared" for a in alts if a in vocab.facets):
                    continue
                undeclared.add(b.name)
                notes.append(
                    f"Benchmark/{b.name}: metric {name} (class {cls['id']}) needs facet {req}; "
                    f"declare it under the Benchmark's context: block"
                )
    if detail:
        return warns + notes
    if undeclared:
        warns.append(
            f"{len(undeclared)} benchmarks have undeclared context facets; run validate --lint-context for details"
        )
    return warns

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

"""spyre-perf-suite's Multi-AIU report XMLs -> spyre_v2 benchmark_runs (report_kind 'perf-suite-multi-aiu-xml').

generate_multi_aiu_report.py writes four kinds of file, told apart by the <testsuites name>:

    spyre-perf-suite         report_rank_N.xml              per-rank latency (TTFT/ITL steps, summaries)
    spyre-kernel-suite       spyre_kernel_report_rank_N.xml per-rank Spyre kernel durations
    cpu-kernel-suite         cpu_kernel_report_rank_N.xml   per-rank host-side kernel durations
    spyre-multi-aiu-summary  multi_aiu_summary.xml          cross-card means / min / max

Their testcase names (`summary_mean_rank_0_...`, `kernel_rank_0_...`) do not fit the op/model
shape results.parse_benchmark_xml reads, so they get their own parser. One testcase is one
benchmark: the value is the testcase `time` (ms), and the tags the generator wrote that say WHAT
was measured (card_rank, step, scope, metric, kernel) are the benchmark's identity tags, so every
card, step and kernel has its own trend. A card is never averaged into another.
"""

import math
import xml.etree.ElementTree as etree
from datetime import UTC, datetime
from pathlib import Path

import regex as re

REPORT_KIND = "perf-suite-multi-aiu-xml"

# <testsuites name> -> (suite, backend). CPU kernels are host work; the rest ran on the card.
SUITES = {
    "spyre-perf-suite": ("latency", "spyre"),
    "spyre-kernel-suite": ("spyre_kernel", "spyre"),
    "cpu-kernel-suite": ("cpu_kernel", "cpu"),
    "spyre-multi-aiu-summary": ("summary", "spyre"),
}
# Tags that name what was measured. The others (time_ms__...) repeat the value, which is
# already the measurement.
_IDENTITY_TAGS = ("card_rank", "step", "scope", "metric", "kernel")
# Suite-level properties kept on every run row; source_report is a path on the producer's host.
_RUN_PROPS = ("source_trace", "pid", "workload_id", "duration_slices")
_RANK_RE = re.compile(r"_?rank_\d+")
# Suite properties a producer may name the measured model by, in order of preference.
_MODEL_PROPS = ("model", "hf_model", "model_name")
# The three layers read off flexRoundTrip / aiuRoundtrip spans and the device slices placed in them:
# `flex_rt_ms` on a step, `ttft_flex_rt_ms` in a summary.
_DEVICE_LAYER_RE = re.compile(r"^(?:(?:ttft|itl)_)?(?:flex_rt|aiu_rt|aiu_hw)_ms$")
# Hashed with the name and tags (BenchmarkId.disc_keys): the suite and the size of the run
# (config_name, `<N>-card`) are part of what a benchmark is, so card 0 of a 1-card run and card 0
# of a 4-card run are two series: their numbers are not comparable.
_ID_KEYS = ("record_type", "run_mode", "config_name")


def _properties(node) -> dict:
    return {
        p.get("name"): p.get("value", "") for p in node.findall("./properties/property")
    }


def _tags(case) -> dict:
    out = {}
    for p in case.findall("./properties/property"):
        key, sep, val = p.get("value", "").partition("__")
        if p.get("name") == "tag" and sep:
            out.setdefault(key, val)
    return out


def is_multi_aiu_xml(root) -> bool:
    """True for the four Multi-AIU report files, and not for a plain spyre-perf-suite report.xml.

    The per-rank latency report shares its <testsuites name> with report.xml, so it also has to
    carry the `card_rank` suite property only the Multi-AIU generator writes.
    """
    name = root.get("name", "")
    if name not in SUITES:
        return False
    if name != "spyre-perf-suite":
        return True
    return any("card_rank" in _properties(s) for s in root.findall("testsuite"))


def _utc(stamp: str) -> str:
    """A testsuite `timestamp` (ISO-8601) as the 'YYYY-MM-DD HH:MM:SS' UTC the run props carry."""
    try:
        ts = datetime.fromisoformat(stamp)
    except ValueError:
        return ""
    # A stamp with no offset is the generator's, which writes UTC.
    return (
        ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
        if ts.tzinfo
        else (ts.strftime("%Y-%m-%d %H:%M:%S"))
    )


def num_cards_of(paths) -> int:
    """How many cards a run used: the producer's declared count, else the distinct `card_rank`s.

    Declared is the summary's `total_cards`, or the `num_cards` a strict producer writes on every
    per-rank file (from its `# cards: N, rank: R` report header), so a rank file read alone still
    knows the run's size. Inferred (the ranks seen) is only a fallback: with a card's report
    missing it undercounts, and the run lands in the wrong `<N>-card` series.

    A run is several files (a report per rank, kernel reports, a summary); the summary is the
    only one that names the whole run, and `multi_aiu_summary.xml` sorts between the CPU- and
    Spyre-kernel reports, so the files have to be read before any is parsed.
    """
    total, ranks = 0, set()
    for path in paths:
        root = etree.parse(str(path)).getroot()
        if not is_multi_aiu_xml(root):
            continue
        for ts in root.findall("testsuite"):
            props = _properties(ts)
            for key in ("total_cards", "num_cards"):
                if props.get(key, "").isdigit():
                    total = max(total, int(props[key]))
            if props.get("card_rank", "").isdigit():
                ranks.add(props["card_rank"])
    return total or len(ranks)


def model_of(paths) -> str:
    """The model a run measured, as its producer declared it on any of its files; "" if none does.

    The generator does not write one yet, so the caller passes the model the run was
    started with (results --model) as the fallback. The model is part of every benchmark's
    identity: granite's TTFT and gpt2's TTFT are two series, not one.
    """
    for path in paths:
        root = etree.parse(str(path)).getroot()
        if not is_multi_aiu_xml(root):
            continue
        for ts in root.findall("testsuite"):
            props = _properties(ts)
            for key in _MODEL_PROPS:
                if props.get(key, "").strip():
                    return props[key].strip()
    return ""


def workload_props(workload, prefix="workload.") -> dict:
    """A run's workload description (the producer's RUN.json) as flat run props.

    Nested objects are joined with dots (`workload.versions.torch`), every value is a string, and empty
    values are dropped. These describe the run, never the benchmark: they are run props, so the
    benchmark identity is unchanged.
    """
    out = {}
    for key, value in (workload or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(workload_props(value, name + "."))
        elif value not in (None, "", [], {}):
            out[name] = (
                ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            )
    return out


def lacks_device_spans(entries: list) -> bool:
    """True when a latency report has the device layers and every one is 0.

    The generator prints 0 for flex_rt / aiu_rt / aiu_hw when the trace has no flexRoundTrip /
    aiuRoundtrip spans (the flex build predates that instrumentation): TTFT/ITL totals are valid,
    the layer split is not there (spyre-perf-suite scripts/README.md, "Reading the numbers"). Those
    zeros mean "not measured", so the run is incomplete, not a pass. A report with no such layer
    at all (a kernel or summary file) says nothing either way.
    """
    layers = [
        e["measurements"]["duration_ms"][0]
        for e in entries
        if e["props"].get("run_mode") == "latency"
        and _DEVICE_LAYER_RE.match(e["props"].get("metric", ""))
        and "duration_ms" in e["measurements"]
    ]
    return bool(layers) and not any(layers)


def parse(
    path: Path, num_cards: int = 0, model: str = "", workload: dict | None = None
) -> list:
    """One entry per testcase, in the shape BenchmarkWriter.insert takes.

    Entries are built here rather than through results._bench_entries: its metric allowlist
    drops `n_flex`, and its props are op/model fields. `n_flex` is a span COUNT the generator
    carries in `time`, so it is stored as its own measurement, not as a duration.

    `num_cards` is the run's size (num_cards_of over all its files); read from this file alone
    when not given. `model` is the model the run measured (model_of, else results --model);
    read from this file when not given, and left out when nobody names it. `workload` is the
    run's workload description (results --workload-json), kept as `workload.*` run props.
    """
    root = etree.parse(str(path)).getroot()
    suite, backend = SUITES[root.get("name", "")]
    cards = num_cards or num_cards_of([path])
    model = model or model_of([path])
    config = f"{cards}-card" if cards else ""
    entries = []
    for ts in root.findall("testsuite"):
        run_props = {
            k: v for k, v in _properties(ts).items() if k in _RUN_PROPS and v != ""
        }
        # The run's own time, so a late or back-filled ingest is still dated by the measurement.
        if _utc(ts.get("timestamp", "")):
            run_props["run_utc"] = _utc(ts.get("timestamp", ""))
        if cards:
            run_props["num_cards"] = str(cards)
        if model:
            run_props["model"] = model
        run_props.update(workload_props(workload))
        for case in ts.findall("testcase"):
            try:
                value = float(case.get("time", ""))
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            tags = _tags(case)
            ident = {k: tags[k] for k in _IDENTITY_TAGS if k in tags}
            if model:
                ident["model"] = model
            entries.append(
                {
                    "name": f"multi_aiu/{suite}/{_RANK_RE.sub('', case.get('name', ''))}",
                    "tags": sorted(f"{k}__{v}" for k, v in ident.items()),
                    "backend": backend,
                    "props": {
                        "record_type": "multi_aiu",
                        "run_mode": suite,
                        "config_name": config,
                        "num_cards": str(cards) if cards else "",
                        **ident,
                    },
                    "measurements": {
                        "n_flex"
                        if ident.get("metric") == "n_flex"
                        else "duration_ms": [value]
                    },
                    "iterations": 1,
                    "run_props": run_props,
                    "disc": {
                        "record_type": "multi_aiu",
                        "run_mode": suite,
                        "config_name": config,
                    },
                    "disc_keys": _ID_KEYS,
                }
            )
    return entries

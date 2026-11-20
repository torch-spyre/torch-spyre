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


"""``tools/cost_model/eval_model.py`` is run as a script, never imported as a package
module, so nothing else in CI notices when it stops loading. The rest checks the
relayout validation on synthetic sweep records built from a known law: no device."""

import collections
import dataclasses
import importlib.util
import math
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOOL_DIR = os.path.join(_ROOT, "tools", "cost_model")


@pytest.fixture
def em(monkeypatch):
    """``eval_model.py`` loaded from its path, as the script runs."""
    # The script puts its own directory and the repo root first on sys.path and
    # imports its siblings (``records``) as top-level modules. Undo both, so no
    # later test in this worker resolves an import against the tool directory.
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    try:
        path = os.path.join(_TOOL_DIR, "eval_model.py")
        spec = importlib.util.spec_from_file_location("eval_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for name in set(sys.modules) - before:
            origin = getattr(sys.modules[name], "__file__", None) or ""
            if os.path.dirname(os.path.abspath(origin)) == _TOOL_DIR:
                del sys.modules[name]


def test_eval_model_is_importable(em):
    assert callable(em.main)


# A [rows, cols] fp16 tensor over 32 cores, and two divisions of it.
_ROWS, _COLS, _X, _Y = 512, 4096, "R:16,H:2", "R:32"
# The producer and consumer time of each division: it cancels in the estimator.
_BASE_US = {_X: 100.0, _Y: 140.0}


def _copy(split, run_elems, out_elems=_ROWS * _COLS):
    return {
        "name": "lx_relayout",
        "is_reduction": False,
        "out_elems": out_elems,
        "cores": 32,
        "dtype_bytes": 2,
        "args": [],
        "is_lx_relayout": True,
        "relayout_cores": 32,
        "relayout_run_elems": run_elems,
        "relayout_split": split,
    }


def _rec(src, dst, kernel_us, feats, *, cols=_COLS, std=0.0, mode="on"):
    return {
        "op": "relayout",
        "model_sha": "abc1234",
        "rows": _ROWS,
        "cols": cols,
        "lx": 1,
        "sencores": 32,
        "rl_src": src,
        "rl_dst": dst,
        "rl_mode": mode,
        "kernel_us": kernel_us,
        "kernel_us_std": std,
        "reps": 7,
        "feats": feats,
    }


def _swap_group(em, params, copy, *, cols=_COLS, jitter=()):
    """The four `on` runs of the X/Y pair, the cross runs carrying ``copy`` priced by
    ``params``. Each ``jitter`` entry adds one process per run, offset by it."""
    shuffle = em.cm.relayout_ns(em.cm.op_from_dict(copy), params) / 1000
    base = {
        (_X, _Y): _BASE_US[_X] + _BASE_US[_Y] + shuffle,
        (_Y, _X): _BASE_US[_X] + _BASE_US[_Y] + shuffle,
        (_X, _X): 2 * _BASE_US[_X],
        (_Y, _Y): 2 * _BASE_US[_Y],
    }
    rows = []
    for (src, dst), t in base.items():
        feats = [copy] if src != dst else []
        for d in jitter or (0.0,):
            rows.append(_rec(src, dst, t + d, feats, cols=cols, std=1.0))
    return rows, shuffle


def test_swap_estimator_isolates_the_shuffle(em):
    params = em.cm.CostParams()
    rows, shuffle = _swap_group(em, params, _copy(32, 1024))
    (pair,) = em.relayout_pairs(rows, params)
    assert pair["meas_us"] == pytest.approx(shuffle)
    assert pair["pred_us"] == pytest.approx(shuffle)
    assert pair["split"] == 32 and pair["repeats"] == 1


def test_repeats_are_pooled_and_set_the_noise(em):
    params = em.cm.CostParams()
    rows, shuffle = _swap_group(em, params, _copy(32, 1024), jitter=(-0.3, 0.3))
    notes = collections.Counter()
    (pair,) = em.relayout_pairs(rows, params, notes)
    assert notes == {"configuration(s) measured in several processes: pooled": 4}
    assert pair["meas_us"] == pytest.approx(shuffle)
    assert pair["repeats"] == 2
    # Each run: two processes at t -+ 0.3, so sd = 0.3 * sqrt(2) and the standard
    # error of their mean is 0.3. Four of those in quadrature, halved: 0.3.
    assert pair["noise_us"] == pytest.approx(0.3)


def test_one_process_noise_is_the_error_of_its_median(em):
    params = em.cm.CostParams()
    rows, _ = _swap_group(em, params, _copy(32, 1024))
    (pair,) = em.relayout_pairs(rows, params)
    # kernel_us_std 1.0 over 7 repetitions; four runs in quadrature, halved.
    assert pair["noise_us"] == pytest.approx(math.sqrt(math.pi / 2) / math.sqrt(7))


def test_a_cross_run_with_two_copies_is_dropped(em):
    params = em.cm.CostParams()
    rows, _ = _swap_group(em, params, _copy(32, 1024))
    rows[0]["feats"] = [_copy(32, 1024), _copy(32, 1024)]
    notes = collections.Counter()
    assert em.relayout_pairs(rows, params, notes) == []
    assert notes == {"pair(s) dropped: a cross run without exactly one copy": 1}


def test_a_copy_that_differs_between_repeats_is_dropped(em):
    params = em.cm.CostParams()
    rows, _ = _swap_group(em, params, _copy(32, 1024), jitter=(0.0, 0.0))
    rows[1]["feats"] = [_copy(16, 1024)]  # the second process of X->Y
    notes = collections.Counter()
    assert em.relayout_pairs(rows, params, notes) == []
    assert notes["pair(s) dropped: the copy differs between repeats"] == 1


def _law_sweep(em, params):
    """Pairs at every split in the law's range and several run lengths, each a
    separate shape so they never share runs, measured exactly under ``params``."""
    rows = []
    for i, (split, run) in enumerate(
        (s, r) for s in (2, 4, 8, 16, 32) for r in (512, 2048, 8192)
    ):
        cols = _COLS * (i + 1)
        jitter = (-0.1 * (i + 1), 0.1 * (i + 1))  # noise varies across pairs
        group, _ = _swap_group(
            em, params, _copy(split, run, _ROWS * cols), cols=cols, jitter=jitter
        )
        rows += group
    return rows


@pytest.mark.parametrize("weight", ["noise", "relative"])
def test_fit_recovers_the_law(em, weight):
    truth = dataclasses.replace(
        em.cm.CostParams(),
        relayout_run_a_ns=2.0,
        relayout_run_b_ns=3.0,
        relayout_span_gbps=500.0,
    )
    rows = _law_sweep(em, truth)
    # Priced under the defaults: the regressors do not depend on a, b or K.
    pairs = em.relayout_pairs(rows, em.cm.CostParams())
    fitted, stderr = em.fit_relayout(pairs, weight)
    assert fitted["relayout_run_a_ns"] == pytest.approx(2.0, rel=1e-6)
    assert fitted["relayout_run_b_ns"] == pytest.approx(3.0, rel=1e-6)
    assert fitted["relayout_span_gbps"] == pytest.approx(500.0, rel=1e-6)
    assert all(v == pytest.approx(0.0, abs=1e-6) for v in stderr.values())
    assert em._physical(fitted, truth) == ""
    chi2, within, n = em._chi2(em.relayout_pairs(rows, truth), fitted_params=3)
    assert chi2 == pytest.approx(0.0, abs=1e-9) and within == n == len(pairs)


def test_noise_weights_discount_a_poorly_measured_pair(em):
    truth = dataclasses.replace(
        em.cm.CostParams(),
        relayout_run_a_ns=2.0,
        relayout_run_b_ns=3.0,
        relayout_span_gbps=500.0,
    )
    rows = _law_sweep(em, truth)
    # One more pair, 1 us off the law and measured 100x worse than the rest.
    group, _ = _swap_group(
        em, truth, _copy(2, 512, _ROWS * 512), cols=512, jitter=(-100.0, 100.0)
    )
    for r in group:
        if r["rl_src"] != r["rl_dst"]:
            r["kernel_us"] += 1.0
    pairs = em.relayout_pairs(rows + group, em.cm.CostParams())
    by = {
        w: em.fit_relayout(pairs, w)[0]["relayout_span_gbps"]
        for w in ("noise", "relative")
    }
    assert by["noise"] == pytest.approx(500.0, rel=1e-3)
    assert abs(by["relative"] - 500.0) > 10 * abs(by["noise"] - 500.0)


def test_fit_needs_two_splits(em):
    params = em.cm.CostParams()
    rows, _ = _swap_group(em, params, _copy(32, 1024), jitter=(-0.1, 0.1))
    with pytest.raises(ValueError, match="two or more governing splits"):
        em.fit_relayout(em.relayout_pairs(rows, params))


def test_physical_rejects_a_per_run_cost_that_falls_with_the_split(em):
    params = em.cm.CostParams()
    fitted = {
        "relayout_run_a_ns": 30.0,
        "relayout_run_b_ns": -7.5,
        "relayout_span_gbps": 500.0,
    }
    why = em._physical(fitted, params)
    assert "b < 0" in why and "negative at split 32" in why
    # Judged over the run's own split range, not the default one.
    narrow = dataclasses.replace(params, relayout_split_max=8)
    assert "negative" not in em._physical(fitted, narrow)


@pytest.mark.parametrize(
    "timing, log_date, scored",
    [
        ("device", "20261029_120000", True),  # the fixed harness
        (None, "20261028_162107", False),  # unmarked, after #3424
        (None, "20260804_013002", True),  # before the launch::* markers existed
        (None, "", True),  # undated: predates the sweep runner
    ],
)
def test_rows_timed_with_the_host_markers_are_out_of_scope(
    em, timing, log_date, scored
):
    rec = {"op": "neg", "cores": 32, "log_date": log_date}
    if timing:
        rec["timing"] = timing
    assert em.in_scope(rec) is scored

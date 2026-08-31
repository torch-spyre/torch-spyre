# Copyright 2025 The Torch-Spyre Authors.
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

"""Unit tests for the cost-model reporting pass (``cost_model_pass.py``).

The properties guarded here are the ones the report's arithmetic depends on.

* **A kernel is priced once.** ``predict_ops`` is defined over a fused bundle and is
  not additive over its ops -- it de-duplicates external inputs shared inside the
  bundle. Pricing ops separately and summing gives a different number entirely.
  ``_feats`` therefore gives every op the SAME external input ``arg0``, which is what
  makes ``_fused_hbm_bytes`` de-duplicate: with distinct input names the de-dup branch
  never runs and a sum-of-parts implementation would pass every test here.
* **Grouping follows contiguity, not tiling.** ``spyre_fuse_nodes`` fuses every
  contiguous run of Spyre nodes into one bundle, so the report must too. An earlier
  version gave each untiled op its own group and under-predicted a 5-op softmax by 45 %.
* **The per-op column sums back to its kernel total**, because the report prints both.

No Spyre device or backend compiler is required -- ops are injected directly.
"""

import dataclasses
import threading

import pytest

import torch_spyre._inductor.cost_model_pass as cmp
from torch_spyre._inductor import config
from torch_spyre._inductor.constants import DEVICE_NAME
from torch_spyre._inductor import cost_model
from torch_spyre._inductor.cost_model import ArgTraffic, OpFeatures


class _Device:
    def __init__(self, type_):
        self.type = type_


class _FakeLoopInfo:
    def __init__(self, group_id, loop_count):
        self.loop_group_id = group_id
        self.loop_count = loop_count


class _NotComputedBuffer:
    """Stands in for a fallback op the extractor cannot model."""


def _feats(
    name, *, extra_reads=0, write_mb=2, lx_mb=0, shared_input=True, device=DEVICE_NAME
):
    """OpFeatures with a known byte count.

    ``shared_input`` gives the op the external input ``arg0``. That name matters:
    ``_fused_hbm_bytes`` de-duplicates only args whose name starts with ``arg``, so it
    is what makes bundle pricing differ from per-op pricing.
    """
    elems = 512 * 1024  # 1 MB at 2 bytes
    args = []
    if shared_input:
        args.append(ArgTraffic("arg0", "input", False, elems, False, [], []))
    for i in range(extra_reads):
        args.append(ArgTraffic(f"{name}_in{i}", "input", False, elems, False, [], []))
    for i in range(write_mb):
        args.append(ArgTraffic(f"{name}_out{i}", "output", False, elems, False, [], []))
    for i in range(lx_mb):
        args.append(ArgTraffic(f"{name}_lx{i}", "output", True, elems, False, [], []))
    f = OpFeatures(
        name=name,
        is_reduction=False,
        out_elems=elems,
        cores=32,
        dtype_bytes=2,
        args=args,
    )
    # The pass breaks a kernel at any op not on the Spyre device, exactly as
    # spyre_fuse_nodes does, so the fixture has to answer get_device().
    f.get_device = lambda d=device: _Device(d) if d else None
    return f


def _ops(feats_list, monkeypatch, loop_infos=None):
    """Route prepared OpFeatures through the real grouping and pricing code."""
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    for f, li in zip(feats_list, loop_infos or [None] * len(feats_list)):
        if li is not None:
            f.loop_info = li
    return feats_list


# --------------------------------------------------------------------------- grouping


def test_empty_graph_is_zero():
    report = cmp.build_report([])
    assert report.total_us == 0.0
    assert report.groups == []


def test_contiguous_ops_form_one_kernel(monkeypatch):
    """This is what spyre_fuse_nodes does -- untiled ops still fuse together."""
    ops = _ops([_feats("a"), _feats("b"), _feats("c")], monkeypatch)
    report = cmp.build_report(ops)
    assert len(report.groups) == 1
    assert report.groups[0].op_names == ["a", "b", "c"]


def test_an_unmodellable_op_breaks_the_kernel(monkeypatch):
    """A non-Spyre node breaks a real bundle; an unmodellable op breaks a group."""
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    report = cmp.build_report(
        [_feats("a"), _feats("b"), _NotComputedBuffer(), _feats("c")]
    )
    assert [g.op_names for g in report.groups] == [["a", "b"], ["c"]]


def test_a_non_spyre_op_breaks_the_kernel(monkeypatch):
    """A CPU op's buffer IS a ComputedBuffer, so the type test alone is not enough.

    spyre_fuse_nodes breaks a bundle at any non-Spyre node; without the device check
    a CPU op would both join the kernel and be priced as Spyre traffic.
    """
    ops = _ops([_feats("a"), _feats("cpu_b", device="cpu"), _feats("c")], monkeypatch)
    report = cmp.build_report(ops)
    assert [g.op_names for g in report.groups] == [["a"], ["c"]]


def test_no_fusion_means_one_kernel_per_op(monkeypatch):
    """With bundle_symbolic_args off the backend does not fuse, so neither do we."""
    ops = _ops([_feats("a"), _feats("b"), _feats("c")], monkeypatch)
    with config.patch({"bundle_symbolic_args": False}):
        report = cmp.build_report(ops)
    assert [g.op_names for g in report.groups] == [["a"], ["b"], ["c"]]


def test_extraction_failure_also_breaks_the_kernel(monkeypatch):
    """An op the extractor rejects must not silently join its neighbours."""

    def _flaky(op):
        if op.name == "bad":
            raise ValueError("cannot model this one")
        return op

    monkeypatch.setattr(cmp, "extract_op_features", _flaky)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    report = cmp.build_report([_feats("a"), _feats("bad"), _feats("c")])
    assert [g.op_names for g in report.groups] == [["a"], ["c"]]


def test_loop_ids_are_recorded_for_labelling(monkeypatch):
    """Tiling no longer decides grouping, but it is still reported."""
    li = _FakeLoopInfo((0,), [8])
    ops = _ops([_feats("a"), _feats("b")], monkeypatch, [li, li])
    group = cmp.build_report(ops).groups[0]
    assert group.loop_group_ids == [(0,)]
    assert group.has_loop
    assert group.loop_trip == 1  # loop_trip comes from OpFeatures, not loop_info


def test_a_malformed_loop_id_does_not_raise(monkeypatch):
    """A bare int where a tuple is expected must not sink the report."""

    class _BadLoopInfo:
        loop_group_id = 1  # not a tuple

    ops = _ops([_feats("a")], monkeypatch, [_BadLoopInfo()])
    report = cmp.build_report(ops)
    assert len(report.groups) == 1
    assert report.groups[0].loop_group_ids == []


# ---------------------------------------------------------------------------- pricing


def test_kernel_is_priced_once_not_per_op(monkeypatch):
    """The kernel total must equal predict_ops over the whole kernel.

    Regression guard against a sum-of-parts implementation. The fixture shares ``arg0``
    across ops so the two genuinely differ -- with distinct input names they coincide
    and this test would pass either way.
    """
    from torch_spyre._inductor.cost_model import CostParams, predict_ops

    feats = [_feats("a"), _feats("b"), _feats("c")]
    ops = _ops(feats, monkeypatch)
    report = cmp.build_report(ops)

    bundle_us = predict_ops(feats, CostParams()) / 1000.0
    parts_us = sum(predict_ops([f], CostParams()) / 1000.0 for f in feats)

    assert report.groups[0].predicted_us == pytest.approx(bundle_us, rel=1e-9)
    # The fixture must actually distinguish the two, or this guards nothing.
    assert parts_us > bundle_us * 1.1


def test_attribution_sums_to_the_kernel_total(monkeypatch):
    """The printed parts must add up to the printed total."""
    ops = _ops(
        [_feats("a", extra_reads=8), _feats("b", extra_reads=2), _feats("c")],
        monkeypatch,
    )
    group = cmp.build_report(ops).groups[0]
    assert sum(o.predicted_us for o in group.ops) == pytest.approx(
        group.predicted_us, rel=1e-9
    )


def test_attribution_follows_main_memory_bytes(monkeypatch):
    """An op with twice the traffic takes twice the share."""
    ops = _ops(
        [
            _feats("big", extra_reads=3, write_mb=0, shared_input=False),
            _feats("small", extra_reads=1, write_mb=0, shared_input=False),
        ],
        monkeypatch,
    )
    by_name = {o.name: o for o in cmp.build_report(ops).groups[0].ops}
    assert by_name["big"].predicted_us == pytest.approx(
        3 * by_name["small"].predicted_us, rel=1e-9
    )


def test_op_with_no_main_memory_traffic_takes_no_share(monkeypatch):
    """An op fused into on-chip memory adds no traffic, so it attracts no time."""
    ops = _ops(
        [
            _feats("streams"),
            _feats("onchip", write_mb=0, lx_mb=4, shared_input=False),
        ],
        monkeypatch,
    )
    by_name = {o.name: o for o in cmp.build_report(ops).groups[0].ops}
    assert by_name["onchip"].predicted_us == 0.0
    assert by_name["onchip"].lx_bytes > 0


def test_total_is_the_sum_over_kernels(monkeypatch):
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    report = cmp.build_report(
        [_feats("a"), _NotComputedBuffer(), _feats("b"), _NotComputedBuffer()]
    )
    assert len(report.groups) == 2
    assert report.total_us == pytest.approx(
        sum(g.predicted_us for g in report.groups), rel=1e-9
    )


def test_one_unpriceable_kernel_does_not_lose_the_others(monkeypatch):
    """A group predict_ops cannot price is dropped; the rest still report."""
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    real = cmp.predict_ops

    def _flaky(feats, params=None):
        if any(f.name == "boom" for f in feats):
            raise ZeroDivisionError("cannot price")
        return real(feats, params)

    monkeypatch.setattr(cmp, "predict_ops", _flaky)
    report = cmp.build_report(
        [
            _feats("a"),
            _NotComputedBuffer(),
            _feats("boom"),
            _NotComputedBuffer(),
            _feats("c"),
        ]
    )
    assert [g.op_names for g in report.groups] == [["a"], ["c"]]
    assert report.total_us > 0.0


# ---------------------------------------------------------------------------- gating


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", False),
        ("0", False),
        ("false", False),
        ("off", False),
        ("no", False),
        ("2", False),
        ("banana", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        (" ON ", True),
    ],
)
def test_gate_parsing(value, expected):
    """ "0" must mean OFF. Plain truthiness would enable the pass for SPYRE_DUMP_COST=0.

    "2" was a second verbosity that has been removed; it must now read as off rather
    than silently behaving like "1".
    """
    with config.patch({"cost_model": value}):
        assert cmp._enabled() is expected


def test_disabled_returns_none_and_does_not_extract(monkeypatch):
    """Off must mean off: no extraction, so leaving it off is free."""
    calls = []
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: calls.append(op) or op)

    class _Graph:
        operations = [_feats("a")]

    for off in ("", "0", "false"):
        with config.patch({"cost_model": off}):
            assert cmp.cost_model_pass(_Graph()) is None
    assert calls == []


def test_enabled_returns_a_report(monkeypatch):
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)

    class _Graph:
        operations = [_feats("a")]

    with config.patch({"cost_model": "1"}):
        report = cmp.cost_model_pass(_Graph())
    assert report is not None
    assert report.total_us > 0.0
    assert cmp.LAST_REPORT is report


def test_last_report_does_not_leak_across_threads(monkeypatch):
    """A concurrent compile must not overwrite another thread's report.

    ``torch.compile`` can be driven from several threads, each running this pipeline
    over its own graph. When ``LAST_REPORT`` was a plain module global, whichever
    thread finished last won and the others silently read someone else's number.
    """
    monkeypatch.setattr(cmp, "extract_op_features", lambda op: op)
    monkeypatch.setattr(cmp, "ComputedBuffer", OpFeatures)
    # Enable the pass directly instead of through ``config.patch``. Whether a config
    # patch made on this thread is visible on a worker thread is a PyTorch detail that
    # has changed between releases, and it is not what this test is about -- in
    # production the switch is an environment variable read once at import.
    monkeypatch.setattr(cmp, "_enabled", lambda: True)

    all_compiled = threading.Barrier(4)
    produced: dict[int, bool] = {}
    isolated: dict[int, bool] = {}

    def compile_one(i):
        class _Graph:
            operations = [_feats(f"op{i}", write_mb=i + 1)]

        mine = cmp.cost_model_pass(_Graph())
        produced[i] = mine is not None
        # Nobody reads until every thread has produced a report, so a shared global
        # would by now hold whichever thread happened to finish last.
        all_compiled.wait(timeout=10)
        isolated[i] = cmp.LAST_REPORT is mine

    before = cmp.LAST_REPORT
    threads = [threading.Thread(target=compile_one, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # Asserted apart so a failure says WHICH half broke: the pass not running at all,
    # or a thread reading someone else's report.
    assert produced == {i: True for i in range(4)}, (
        "the pass did not run on every thread"
    )
    assert isolated == {i: True for i in range(4)}, (
        "a thread saw another thread's report"
    )
    # This thread compiled nothing, so its own view is untouched by all four.
    assert cmp.LAST_REPORT is before


def test_pass_never_raises(monkeypatch):
    """Instrumentation must not be able to break a compilation."""
    monkeypatch.setattr(
        cmp, "build_report", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    class _Graph:
        operations = [1, 2, 3]

    with config.patch({"cost_model": "1"}):
        assert cmp.cost_model_pass(_Graph()) is None


# --------------------------------------------------------------------------- reporting


def test_report_formats_without_error(monkeypatch):
    ops = _ops(
        [_feats("a"), _feats("b", write_mb=0, lx_mb=2, shared_input=False)],
        monkeypatch,
        [_FakeLoopInfo((0,), [8]), _FakeLoopInfo((0,), [8])],
    )
    text = cmp.build_report(ops).format()
    assert "predicted total" in text
    # An op that writes only to on-chip memory still has to show its bytes, so its
    # 0.0 share reads as "fused away", not "free".
    assert "HBM" in text and "-" in text
    assert "LX" in text
    assert "attribution" in text


def test_empty_report_formats_without_dividing_by_zero():
    assert "predicted total" in cmp.CostReport(total_us=0.0, groups=[]).format()


def test_dataclasses_are_plain_values():
    """The report is a value another pass can hold on to and compare."""
    assert dataclasses.is_dataclass(cmp.CostReport)
    assert dataclasses.is_dataclass(cmp.GroupCost)
    assert dataclasses.is_dataclass(cmp.OpCost)


# -------------------------------------------------------------- per-iteration traffic


def _looped(name, trip, args, matmul=False):
    """OpFeatures for one op inside a coarse-tiling loop of ``trip`` iterations."""
    f = OpFeatures(
        name=name,
        is_reduction=False,
        out_elems=1024,
        cores=32,
        dtype_bytes=2,
        args=args,
        loop_trip=trip,
        is_matmul=matmul,
        tiles_output_dim=matmul,
        # A concrete matmul geometry so a bundle containing this op is priceable by
        # ``_matmul_axes_for_split_cost`` (M=rows_per_core, N=cols_per_core, K backed
        # out of matmul_a_bytes) -- unused when ``matmul`` is False.
        matmul_rows_per_core=64.0,
        matmul_cols_per_core=64.0,
        matmul_a_bytes=64 * 64 * 2,
    )
    f.get_device = lambda: _Device(DEVICE_NAME)
    return f


def _arg(name, role, elems, factor=1, mem="hbm"):
    a = ArgTraffic(name, role, mem == "lx", elems, False, [], [])
    a.loop_factor = factor
    return a


@pytest.mark.parametrize(
    "label,factor",
    [
        # An operand that walks the tensor once. hbm_bytes is elems.
        ("advancing", 1),
        # One the loop holds fixed: hbm_bytes is elems * trip, so one iteration is
        # still the operand's full size -- the SAME answer plain division gives.
        ("fixed", 8),
        # And the case that is neither. `mm` under mm_nested_m_k really carries
        # 1 < loop_factor < trip, because the factor is a product over nesting
        # levels; a rule that branched on "advancing or fixed" is wrong here.
        ("nested", 2),
    ],
)
def test_per_iteration_is_the_whole_loop_over_the_trip(label, factor):
    f = _looped("k", 8, [_arg("arg0", "input", 4096, factor=factor)])
    assert cmp._per_iteration(f)["hbm_per_iter"] == f.hbm_bytes() // 8
    assert f.hbm_bytes() == 4096 * 2 * factor  # the premise: hbm_bytes folds the factor


@pytest.mark.parametrize(
    "label,args",
    [
        ("all advancing", [_arg("a", "input", 8 * 4096)]),
        ("all fixed", [_arg("b", "input", 4096, factor=8)]),
        ("nested", [_arg("c", "input", 4096, factor=2)]),
        (
            "mixed",
            [
                _arg("a", "input", 8 * 4096),
                _arg("b", "input", 4096, factor=8),
                _arg("c", "input", 4096, factor=2),
                _arg("o", "output", 8 * 2048),
            ],
        ),
    ],
)
def test_whole_loop_equals_trip_times_per_iteration(label, args):
    """The identity the report prints as ``whole = trip x per-iteration``.

    The nested rows are the ones that matter: they are the only inputs where a
    branch on advancing-vs-fixed gives a different answer from plain division.
    """
    f = _looped("k", 8, args)
    assert f.hbm_bytes() == 8 * cmp._per_iteration(f)["hbm_per_iter"]


def test_on_chip_bytes_are_not_divided_again():
    """An LX buffer is allocated per-tile, so lx_bytes is ALREADY one iteration.

    Dividing it by the trip count would under-report the working set by that factor --
    for the one number that decides whether a tile fits on chip.
    """
    f = _looped("k", 8, [_arg("buf", "output", 4096, mem="lx")])
    assert "lx_per_iter" not in cmp._per_iteration(f)
    op = cmp.OpCost(
        name="k",
        loop_group_id=None,
        hbm_bytes=0,
        lx_bytes=f.lx_bytes(),
        predicted_us=0.0,
        trip=8,
        hbm_per_iter=0,
    )
    assert "8.2 KB" in cmp._traffic(op)  # 4096 * 2, undivided


def test_untiled_op_has_no_split():
    per = cmp._per_iteration(_looped("plain", 1, [_arg("arg0", "input", 4096)]))
    assert per["trip"] == 1
    assert per["hbm_per_iter"] == 4096 * 2
    assert per["reread_bytes"] == 0


@pytest.mark.parametrize(
    "label,matmul,expected",
    [
        # An input of a matmul whose loop tiles an output dim: established re-fetch.
        ("output-tiled matmul", True, 4096 * 2 * 7),
        # The same loop_factor on anything else means something different -- under
        # reduction tiling each iteration takes a fresh K-slice, so the input
        # ADVANCES. Reporting it as re-fetched would be wrong, not just uncharged.
        ("not a matmul", False, 0),
    ],
)
def test_refetch_never_diverges_from_what_the_model_charges(label, matmul, expected):
    f = _looped("k", 8, [_arg("arg0", "input", 4096, factor=8)], matmul=matmul)
    assert cmp._per_iteration(f)["reread_bytes"] == expected
    assert cmp._per_iteration(f)["reread_bytes"] == cost_model._loop_reread_bytes([f])


def test_small_buffers_do_not_render_as_zero():
    """Real intermediates are kilobytes; fixed MB formatting printed them as 0.0."""
    op = cmp.OpCost(
        name="k",
        loop_group_id=None,
        hbm_bytes=8192,
        lx_bytes=0,
        predicted_us=0.0,
        trip=1,
        hbm_per_iter=8192,
    )
    text = cmp._traffic(op)
    assert "8.2 KB" in text
    assert "0.0" not in text


def test_long_op_names_keep_the_columns_aligned(monkeypatch):
    """Real graphs carry names like constant_pad_nd_default; a fixed width breaks."""
    ops = _ops([_feats("mm"), _feats("constant_pad_nd_default")], monkeypatch)
    text = cmp.build_report(ops).format()
    body = [ln for ln in text.splitlines() if "MB" in ln]
    assert len(body) == 2
    assert len({ln.index("HBM") for ln in body}) == 1


def test_per_iteration_flows_through_build_report(monkeypatch):
    """End to end: the fields the printout reads are populated by the real path."""
    ops = _ops(
        [
            _looped(
                "mm",
                8,
                [
                    _arg("arg0", "input", 8 * 4096),
                    _arg("arg1", "input", 4096, factor=8),
                    _arg("out", "output", 8 * 2048),
                ],
                matmul=True,
            )
        ],
        monkeypatch,
    )
    op = cmp.build_report(ops).groups[0].ops[0]
    assert op.trip == 8
    assert op.reread_bytes == 4096 * 2 * 7
    assert op.hbm_bytes == 8 * op.hbm_per_iter


def test_a_broken_attribution_keeps_the_price(monkeypatch):
    """The price is the answer; the per-op split is only presentation."""
    ops = _ops([_feats("a"), _feats("b")], monkeypatch)
    monkeypatch.setattr(cmp, "_per_iteration", lambda f: 1 / 0)
    report = cmp.build_report(ops)
    assert report.total_us > 0
    assert report.groups[0].ops == []
    assert "predicted total" in report.format()


def test_ops_that_run_different_numbers_of_times_are_grouped_apart(monkeypatch):
    """A coarse_tile_fill sits OUTSIDE the loop and runs once beside ops that do not.

    coarse_tiling_loops.md: the fill op is inserted "outside the loop, no loop_info".
    A kernel therefore has no single iteration count, so the count is stated per loop
    rather than in one column beside every op in the kernel.
    """
    ops = _ops(
        [
            _looped("fill", 1, [_arg("acc", "output", 4096)]),
            _looped("bmm", 8, [_arg("arg0", "input", 8 * 4096)]),
        ],
        monkeypatch,
    )
    text = cmp.build_report(ops).format()
    assert "loop, runs 8 times" in text
    assert "not in a loop, runs once" in text
    # each op sits under the block that states ITS count -- never one shared number
    body = text.splitlines()
    fill = next(i for i, ln in enumerate(body) if " fill " in ln or ln.endswith("fill"))
    bmm = next(i for i, ln in enumerate(body) if "bmm" in ln and "HBM" in ln)
    once = next(i for i, ln in enumerate(body) if "not in a loop" in ln)
    eight = next(i for i, ln in enumerate(body) if "runs 8 times" in ln)
    assert once < fill and eight < bmm


def test_a_kernel_in_one_loop_states_the_count_once(monkeypatch):
    ops = _ops(
        [
            _looped("a", 8, [_arg("arg0", "input", 8 * 4096)]),
            _looped("b", 8, [_arg("arg1", "input", 8 * 4096)]),
        ],
        monkeypatch,
    )
    text = cmp.build_report(ops).format()
    assert text.count("runs 8 times") == 1
    assert "not in a loop" not in text


def test_an_untiled_kernel_gets_no_loop_headers(monkeypatch):
    """Nothing loops, so a "runs once" header would be noise on every kernel."""
    ops = _ops([_feats("a"), _feats("b")], monkeypatch)
    text = cmp.build_report(ops).format()
    assert "not in a loop" not in text
    assert "runs" not in text

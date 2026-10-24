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

"""The op-coverage report toolkit (tests/op_coverage/spyre_op_coverage).

Everything here runs on the CPU: the toolkit traces on fake Spyre tensors and
reads torch-spyre's registries, so no device is needed. Expectations are
derived from those registries, not from today's list of supported ops, so the
tests keep passing as ops are added.

``spyre_op_coverage`` sits next to this file. This directory has no
``__init__.py``, so pytest puts it on ``sys.path`` before importing the test.
"""

import json
import operator
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._ops import OpOverload

from spyre_op_coverage import (
    CapturedCall,
    Knowledge,
    SpyreRegistry,
    Support,
    TensorMeta,
    analyze_calls,
    analyze_model,
    capture,
    to_json,
    to_markdown,
)
from spyre_op_coverage import cli, model_ops_yaml, serialize
from spyre_op_coverage.analyze import replay_spec
from spyre_op_coverage.knowledge import name_aliases, parse_docs_table
from spyre_op_coverage.model import (
    CoverageReport,
    KnownIssue,
    LeafOp,
    Verdict,
)
from spyre_op_coverage.registry import lowering_kind, op_name
from spyre_op_coverage.trace import Tracer

ROOT = Path(__file__).resolve().parents[2]
aten = torch.ops.aten


@pytest.fixture(scope="module")
def registry():
    return SpyreRegistry()


@pytest.fixture(scope="module")
def tracer(registry):
    return Tracer(registry)


def _no_knowledge():
    return Knowledge(docs=(), issues={})


def _meta(*shape, dtype=torch.float16):
    return TensorMeta(tuple(shape), serialize.contiguous_stride(tuple(shape)), dtype)


def _call(target, *args, kind="call_function", **kwargs):
    if kind == "call_method":
        name = f"torch.Tensor.{target}"
    else:
        name = torch.fx.node._get_qualified_name(target)
    return CapturedCall(name, target, kind, tuple(args), kwargs)


def _pick_lowered(registry, kind, supported):
    """An op whose in-tree lowering is of ``kind`` and is (not) handled by Spyre."""
    pool = (
        registry.pointwise_handlers if kind == "pointwise" else registry.reduction_types
    )
    for op, fn in registry.lowerings.items():
        if (
            not isinstance(op, OpOverload)
            or op in registry.spyre_lowerings
            or op.overloadpacket in registry.spyre_lowerings
            or op_name(op) in registry.cpu_fallbacks
            or op in registry.fallback_lowerings
            or op.is_view
        ):
            continue
        found_kind, detail = lowering_kind(fn)
        if found_kind == kind and (detail in pool) == supported:
            return op, detail
    pytest.skip(f"no {kind} lowering with supported={supported}")


# -- statuses -----------------------------------------------------------------


def test_rollup_counts_views_only_when_alone():
    assert Support.rollup([Support.VIEW, Support.SPYRE]) is Support.SPYRE
    assert Support.rollup([Support.VIEW, Support.VIEW]) is Support.VIEW
    assert (
        Support.rollup([Support.SPYRE, Support.CPU_FALLBACK, Support.VIEW])
        is Support.CPU_FALLBACK
    )
    assert (
        Support.worst([Support.UNVERIFIED, Support.UNSUPPORTED]) is Support.UNSUPPORTED
    )
    assert Support.worst([]) is Support.UNKNOWN


# -- registry -----------------------------------------------------------------


def test_registered_cpu_fallbacks_are_cpu(registry):
    from torch_spyre.ops.fallbacks import fallback_ops

    overloads = [op for op in fallback_ops if isinstance(op, OpOverload)]
    assert overloads
    for op in overloads:
        assert registry.compile_verdict(op).status is Support.CPU_FALLBACK, op


def test_spyre_lowerings_are_spyre(registry):
    from torch_spyre._inductor.lowering import spyre_lowerings

    checked = 0
    for op in spyre_lowerings:
        if not isinstance(op, OpOverload) or op_name(op) in registry.cpu_fallbacks:
            continue
        verdict = registry.compile_verdict(op)
        assert verdict.status is Support.SPYRE, (op, verdict)
        assert verdict.mechanism.startswith("Spyre lowering")
        checked += 1
    assert checked


def test_pointwise_lowering_needs_a_spyre_handler(registry, monkeypatch):
    op, name = _pick_lowered(registry, "pointwise", supported=True)
    assert registry.compile_verdict(op).status is Support.SPYRE
    monkeypatch.setattr(
        registry, "pointwise_handlers", registry.pointwise_handlers - {name}
    )
    verdict = registry.compile_verdict(op)
    assert verdict.status is Support.UNSUPPORTED
    assert f"ops.{name}" in verdict.mechanism


def test_reduction_lowering_needs_a_spyre_reduction(registry, monkeypatch):
    op, name = _pick_lowered(registry, "reduction", supported=True)
    assert registry.compile_verdict(op).status is Support.SPYRE
    monkeypatch.setattr(registry, "reduction_types", registry.reduction_types - {name})
    assert registry.compile_verdict(op).status is Support.UNSUPPORTED


def test_view_lowering_is_a_view(registry):
    assert registry.compile_verdict(aten.permute.default).status is Support.VIEW


@pytest.fixture(scope="module")
def custom_ops(registry):
    """One op with only a CPU kernel; one whose Spyre kernel goes through the CPU."""

    @torch.library.custom_op(
        "op_coverage_test::cpu_only", mutates_args=(), device_types="cpu"
    )
    def cpu_only(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @torch.library.custom_op(
        "op_coverage_test::round_trip", mutates_args=(), device_types="spyre"
    )
    def round_trip(x: torch.Tensor) -> torch.Tensor:
        return x.to("cpu").clone().to(x.device)

    return (
        torch.ops.op_coverage_test.cpu_only.default,
        torch.ops.op_coverage_test.round_trip.default,
    )


def test_no_lowering_and_no_spyre_kernel_is_unsupported(registry, custom_ops):
    cpu_only, _ = custom_ops
    assert registry.eager_verdict(cpu_only).status is Support.UNSUPPORTED
    assert registry.compile_verdict(cpu_only).status is Support.UNSUPPORTED


def test_custom_op_through_the_cpu_is_a_cpu_fallback(registry, custom_ops):
    _, round_trip = custom_ops
    assert registry.eager_verdict(round_trip).status is Support.CPU_FALLBACK
    verdict = registry.compile_verdict(round_trip)
    assert verdict.status is Support.CPU_FALLBACK
    assert verdict.mechanism.startswith("no lowering")


# -- tracing ------------------------------------------------------------------


def test_trace_records_the_decomposition_path(tracer):
    sig = tracer.analyze(_call(F.linear, _meta(4, 64), _meta(32, 64)))
    assert sig.error is None
    assert sig.top_level == ["aten.linear.default"]
    first_steps = {leaf.path[0].op for leaf in sig.leaves if leaf.path}
    assert "aten.linear.default" in first_steps
    assert all(
        step.source in ("spyre", "inductor", "composite")
        for leaf in sig.leaves
        for step in leaf.path
    )


def test_trace_records_composite_expansions(tracer):
    sig = tracer.analyze(
        _call(F.interpolate, _meta(1, 16, 8, 8), scale_factor=2.0, mode="nearest")
    )
    assert sig.error is None
    assert any(step.source == "composite" for leaf in sig.leaves for step in leaf.path)


def test_trace_functionalizes_in_place_ops(tracer):
    sig = tracer.analyze(_call("add_", _meta(4, 64), _meta(4, 64), kind="call_method"))
    ops = {leaf.op for leaf in sig.leaves}
    assert ops, sig
    # AOTAutograd hands Inductor the functional op, not add_.
    assert not any(op.split(".")[1].endswith("_") for op in ops), ops


def test_identity_call_traces_to_nothing(tracer):
    sig = tracer.analyze(_call("contiguous", _meta(4, 64), kind="call_method"))
    assert sig.error is None
    assert sig.leaves == []
    assert sig.status is Support.VIEW


def test_unresolved_op_is_unknown(tracer):
    sig = tracer.analyze(CapturedCall("torch.not_an_op", None, "unresolved", (), {}))
    assert sig.status is Support.UNKNOWN
    assert "torch.not_an_op" in sig.error


# -- whole model --------------------------------------------------------------


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.proj = nn.Linear(8, 16)

    def forward(self, x):
        y = F.silu(self.conv(x))
        y = torch.cat([y, y], 1)[:, :8]
        return self.proj(y.flatten(2).transpose(1, 2)).softmax(-1)


def build_toy():
    """Factory for the CLI's ``module`` recipe."""
    return _Toy().eval(), (torch.randn(1, 3, 8, 8),)


@pytest.fixture(scope="module")
def toy_report(registry):
    model, args = build_toy()
    return analyze_model(model, args, name="toy", knowledge=_no_knowledge())


def test_model_report_names_ops_and_modules(toy_report):
    by_name = {op.name: op for op in toy_report.ops}
    expected = {
        "torch.conv2d",
        "torch.nn.functional.silu",
        "torch.cat",
        "torch.getitem",
        "torch.flatten",
        "torch.transpose",
        "torch.nn.functional.linear",
        "torch.softmax",
    }
    assert expected <= set(by_name), sorted(by_name)
    assert by_name["torch.conv2d"].modules == ["conv"]
    assert by_name["torch.nn.functional.linear"].modules == ["proj"]
    assert toy_report.graphs == 1
    assert toy_report.graph_breaks == {}
    for op in toy_report.ops:
        for sig in op.signatures:
            assert sig.error is None, (op.name, sig.error)
            assert sig.replay is not None


def test_report_survives_json(toy_report):
    again = CoverageReport.from_dict(json.loads(to_json(toy_report)))
    assert to_markdown(again) == to_markdown(toy_report)
    text = to_markdown(toy_report)
    assert "## What does toy require?" in text
    for op in toy_report.ops:
        assert f"`{op.name}`" in text


def test_capture_locates_graph_breaks():
    def fn(x):
        y = x.sin()
        n = int(y.sum().item())  # data-dependent: Dynamo breaks the graph here
        return y + n

    result = capture(lambda compiled: compiled(torch.ones(4)), fn)
    assert result.graph_breaks
    assert any(Path(__file__).name in reason for reason in result.graph_breaks), (
        result.graph_breaks
    )


# -- knowledge ----------------------------------------------------------------

_DOCS = """
| Operation | Eager | Compiled | Execution | Notes |
|-----------|:-----:|:--------:|-----------|-------|
| **Reduction** | | | | |
| `torch.sum` | Y | Y | Spyre | |
| `torch.reshape` / `torch.view` | | Y | Spyre | Includes `_reshape_alias` |
| **CPU Fallback** | | | | |
| `torch.cumsum` | Y | Y | CPU fallback | Runs on CPU |
"""


def test_docs_table_rows():
    entries = parse_docs_table(_DOCS)
    assert [e.names for e in entries] == [
        ["torch.sum"],
        ["torch.reshape", "torch.view"],
        ["torch.cumsum"],
    ]
    assert [e.category for e in entries] == ["Reduction", "Reduction", "CPU Fallback"]
    assert entries[1].eager is False and entries[1].compiled is True


def test_checkout_docs_table_parses():
    path = ROOT / "docs" / "source" / "user_guide" / "supported_operations.md"
    entries = parse_docs_table(path.read_text())
    assert len(entries) > 20
    assert any("torch.mm" in e.names for e in entries)


def test_docs_lookup_and_refine():
    knowledge = Knowledge(parse_docs_table(_DOCS), issues={})
    assert "torch.view" in name_aliases("torch.Tensor.view")
    assert knowledge.doc_for("torch.Tensor.view").names == [
        "torch.reshape",
        "torch.view",
    ]
    assert knowledge.doc_for("aten.sum.dim_IntList").names == ["torch.sum"]

    documented = LeafOp(
        "aten.sum.dim_IntList",
        Verdict(Support.UNVERIFIED, "in-tree lowering `sum_`, not verified on Spyre"),
    )
    knowledge.refine(documented)
    assert documented.verdict.status is Support.SPYRE

    # Listed, but as a CPU fallback: no promotion.
    on_cpu = LeafOp(
        "aten.cumsum.default", Verdict(Support.UNVERIFIED, "in-tree lowering")
    )
    knowledge.refine(on_cpu)
    assert on_cpu.verdict.status is Support.UNVERIFIED


def test_known_issues_match_without_overload(tmp_path):
    notes = tmp_path / "notes.json"
    notes.write_text(
        json.dumps(
            {"aten.gather": {"note": "tracked", "links": ["https://x/issues/7"]}}
        )
    )
    knowledge = Knowledge.load(notes_paths=[notes], use_docs=False)
    issues = knowledge.issues_for(["torch.gather", "aten.gather.default"])
    assert KnownIssue("tracked", ["https://x/issues/7"]) in issues


# -- model_ops YAML -----------------------------------------------------------


def _tensor(shape, dtype="torch.bfloat16"):
    return {
        "tensor": {
            "shape": shape,
            "stride": list(serialize.contiguous_stride(tuple(shape))),
            "dtype": dtype,
        }
    }


def test_model_ops_yaml_calls(tmp_path, registry):
    yaml = pytest.importorskip("yaml")
    include = [
        {
            "name": "torch.nn.functional.silu",
            "sample_inputs_func": {"args": [_tensor([2, 64])]},
            "description": "https://github.com/huggingface/transformers/blob/v5/src/"
            "transformers/models/x/modeling_x.py#L12 in forward",
        },
        {
            "name": "torch.getitem",
            "sample_inputs_func": {
                "args": [
                    _tensor([2, 64]),
                    {"py": "(slice(None, None, None), slice(0, 8, None))"},
                ]
            },
        },
        {
            "name": "torch.Tensor.view",
            "sample_inputs_func": {"args": [_tensor([2, 64]), {"value": "(2, 2, 32)"}]},
        },
        {
            "name": "torch.to",
            "sample_inputs_func": {
                "args": [_tensor([2, 64])],
                "kwargs": {"dtype": "torch.float32"},
            },
        },
        {"name": "torch.not_an_op", "sample_inputs_func": {"args": []}},
    ]
    config = {
        "test_suite_config": {
            "files": [
                {
                    "path": "x",
                    "tests": [
                        {
                            "names": ["TestSpyreModelOps::test_model_ops_db"],
                            "tags": ["model__tiny"],
                            "edits": {"ops": {"include": include}},
                        }
                    ],
                }
            ]
        }
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(config))

    name, calls, warnings = model_ops_yaml.load(path, cast_float=torch.float16)
    assert name == "tiny"
    assert [c.kind for c in calls] == [
        "call_function",
        "call_function",
        "call_method",
        "call_method",
        "unresolved",
    ]
    assert calls[0].args[0].dtype is torch.float16
    assert calls[0].location == "transformers/models/x/modeling_x.py:12"
    assert calls[1].target is operator.getitem
    assert calls[1].args[1] == (slice(None), slice(0, 8))
    assert calls[2].args[1] == (2, 2, 32)
    assert calls[3].kwargs["dtype"] is torch.float32
    assert len(warnings) == 1 and "torch.not_an_op" in warnings[0]

    analysis = analyze_calls(calls, registry=registry, knowledge=_no_knowledge())
    by_name = {op.name: op for op in analysis.ops}
    assert by_name["torch.not_an_op"].status is Support.UNKNOWN
    for name in ("torch.nn.functional.silu", "torch.getitem", "torch.Tensor.view"):
        assert all(sig.error is None for sig in by_name[name].signatures), name


# -- serialization ------------------------------------------------------------


def test_encode_decode_round_trip():
    value = (
        _meta(2, 3),
        [1, 2],
        slice(None, 4, None),
        Ellipsis,
        torch.float16,
        {"k": None},
    )
    assert serialize.decode(json.loads(json.dumps(serialize.encode(value)))) == value


def test_resolve_model_ops_names():
    assert serialize.resolve_op_name("torch.Tensor.view") == ("call_method", "view")
    assert serialize.resolve_op_name("torch.getitem") == (
        "call_function",
        operator.getitem,
    )
    assert serialize.resolve_op_name("torch.nn.functional.silu") == (
        "call_function",
        F.silu,
    )
    # torch.float is a dtype; the op is the Tensor method.
    assert serialize.resolve_op_name("torch.float") == ("call_method", "float")
    with pytest.raises(LookupError):
        serialize.resolve_op_name("torch.not_an_op")


# -- command line and probe ---------------------------------------------------


def test_cli_writes_and_rerenders(tmp_path):
    out = tmp_path / "out"
    code = cli.main(
        [
            "module",
            f"{__name__}:build_toy",
            "-o",
            str(out),
            "--no-docs",
            "-q",
            "--fail-on",
            "unsupported",
        ]  # fmt: skip
    )
    report = json.loads((out / "build_toy.json").read_text())
    worst = max(Support(op["status"]).severity for op in report["ops"])
    assert code == (1 if worst >= Support.UNSUPPORTED.severity else 0)

    again = tmp_path / "again"
    assert cli.main(["json", str(out / "build_toy.json"), "-o", str(again), "-q"]) == 0
    assert (again / "build_toy.md").read_text() == (out / "build_toy.md").read_text()


def test_probe_compares_with_the_cpu():
    from spyre_op_coverage.probe import run_cases

    good = replay_spec(_call(torch.sigmoid, _meta(4, 8, dtype=torch.float32)))
    bad = dict(good, target="torch.not_an_op")
    # The worker runs Inductor on the CPU in place of the device.
    results = run_cases(
        [{"kind": "op", "replay": good}, {"kind": "op", "replay": bad}],
        device="cpu",
        timeout=600,
    )
    assert results[0].outcome == "pass", results[0]
    assert results[1].outcome == "skipped", results[1]
    assert "not_an_op" in results[1].detail

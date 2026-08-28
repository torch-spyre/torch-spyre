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

"""Device integration test: provenance survives a real Spyre compile end-to-end."""

import json
import logging
import logging.handlers
import os

import pytest
import torch
from torch._inductor.utils import fresh_cache


def _spyre_available() -> bool:
    try:
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME

        torch.zeros(1, dtype=torch.float16, device=DEVICE_NAME)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _spyre_available(), reason="requires an available Spyre device"
)


class _MLP(torch.nn.Module):
    # Stick-aligned dims (multiples of the 64-element fp16 stick): compiles and
    # runs end-to-end without padding. Mirrors the provenance example
    # reference_mlp (SimpleMLP(128, 256, 128)).
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 128)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _RichMLP(torch.nn.Module):
    # Same stick-aligned dims as _MLP, but adds layernorm + gelu between the two
    # matmuls so provenance survival is asserted across more production passes
    # (norm and activation lowering, not just a pointwise relu).
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.ln = torch.nn.LayerNorm(256)
        self.fc2 = torch.nn.Linear(256, 128)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.ln(self.fc1(x))))


class _ArtifactModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1.0)


class _GraphBreakArtifactModel(torch.nn.Module):
    def forward(self, x):
        y = torch.relu(x + 1.0)
        torch._dynamo.graph_break()
        return torch.relu(y + 1.0)


class _EquivalentGraphBreakArtifactModel(torch.nn.Module):
    def _segment(self, x):
        return torch.relu(x + 1.0)

    def forward(self, x):
        y = self._segment(x)
        torch._dynamo.graph_break()
        return self._segment(y)


def _compile_lifecycle_shapes(monkeypatch, tmp_path, model, shapes):
    import torch_spyre  # noqa: F401

    from torch_spyre._inductor import config as spyre_config
    from torch_spyre.constants import DEVICE_NAME
    from torch_spyre.execution.async_compile import SpyreAsyncCompile

    collections = []
    original_wait = SpyreAsyncCompile.wait

    def capture_wait(compiler, scope):
        original_wait(compiler, scope)
        collections.append(compiler._last_provenance_collection)

    monkeypatch.setattr(SpyreAsyncCompile, "wait", capture_wait)
    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    torch._dynamo.reset()

    path = tmp_path / "spyre_provenance.json"
    compiled = torch.compile(model.half().to(DEVICE_NAME).eval(), dynamic=False)
    outputs = []
    try:
        with (
            spyre_config.patch({"provenance_artifact_path": str(path)}),
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            fresh_cache(),
            torch.no_grad(),
        ):
            for shape in shapes:
                outputs.append(
                    compiled(
                        torch.randn(
                            *shape,
                            dtype=torch.float16,
                            device=DEVICE_NAME,
                        )
                    )
                )
    finally:
        torch._dynamo.reset()

    return collections, json.loads(path.read_text(encoding="utf-8")), path, outputs


def _assert_handles_survive_real_compile(monkeypatch, model, expect_rewrite):
    """Compile ``model`` on-device and assert the provenance invariants.

    Shared by every model parametrization of
    ``test_handles_survive_real_compile``: (a) the observer inspected real
    buffers and no pass dropped provenance, (b) a production reconstruction uses
    ``preserve_provenance`` when the model
    exercises one, (c) at least one handle resolves to a source line in this test
    module, and (d) a fused handle carries that line among its constituents.
    """
    from torch_spyre.constants import DEVICE_NAME
    import torch_spyre._inductor.pass_utils as pass_utils
    import torch_spyre._inductor.provenance as prov
    import torch_spyre._inductor.spyre_kernel as sk

    collected = []
    preserved = []
    _orig = prov.build_debug_handle
    _orig_preserve = prov.preserve_provenance
    _orig_observer_enter = prov.SpyreGraphTransformObserver.__enter__
    observer_snapshot_sizes = []

    def _collect(buffer):
        h = _orig(buffer)
        collected.append(h)
        return h

    def _preserve(old, new, *args, **kwargs):
        _orig_preserve(old, new, *args, **kwargs)
        preserved.append((old, new))

    def _observer_enter(observer):
        result = _orig_observer_enter(observer)
        if observer._active:
            observer_snapshot_sizes.append(len(observer._before))
        return result

    # These modules import the helpers by name, so patch each bound reference.
    monkeypatch.setattr(prov, "build_debug_handle", _collect)
    monkeypatch.setattr(sk, "build_debug_handle", _collect)
    monkeypatch.setattr(pass_utils, "preserve_provenance", _preserve)
    monkeypatch.setattr(
        prov.SpyreGraphTransformObserver,
        "__enter__",
        _observer_enter,
    )

    # Defeat Inductor's on-disk FX graph cache so codegen (and therefore
    # build_debug_handle) actually runs this process. Same pattern as the
    # provenance audit tooling (audit.py): a cache hit silently skips
    # create_op_spec/define_kernel, which would make this test flaky across
    # repeated runs with identical dims rather than a genuine provenance signal.
    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    torch._dynamo.reset()

    model = model.half().to(DEVICE_NAME).eval()
    x = torch.randn(2, 128, dtype=torch.float16, device=DEVICE_NAME)

    prov_logger = logging.getLogger("spyre.inductor.provenance")
    handler = logging.handlers.MemoryHandler(capacity=10000)
    previous_level = prov_logger.level
    prov_logger.setLevel(logging.WARNING)
    prov_logger.addHandler(handler)
    try:
        # The observer is opt-in like upstream provenance tracing; the handle
        # construction and forwarding assertions below remain unconditional.
        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            with torch.no_grad():
                torch.compile(model)(x)
    finally:
        prov_logger.removeHandler(handler)
        prov_logger.setLevel(previous_level)

    # (a) No pass dropped provenance (observer emitted no drop warnings).
    drops = [r for r in handler.buffer if "spyre-provenance" in r.getMessage()]
    assert any(observer_snapshot_sizes), (
        "observer did not snapshot any real provenance-bearing buffers"
    )
    assert not drops, (
        f"observer reported provenance drops: {[r.getMessage() for r in drops]}"
    )

    # (b) Models that trigger a real buffer reconstruction use the helper.
    if expect_rewrite:
        assert preserved, "no production rewrite called preserve_provenance"

    # (c) At least one handle resolved to a real source line (the matmul traces
    #     back to the model via the linear's weight-transpose origin).
    resolved = [
        h
        for h in collected
        if h is not None and h.source is not None and h.aten_op is not None
    ]
    assert resolved, (
        "no debug_handle resolved to a source; provenance did not reach the kernel"
    )
    # The resolved source should point at this test module (the model's forward).
    this_file = os.path.basename(__file__)
    assert any(h.source.file.endswith(this_file) for h in resolved)

    # (d) A fused op's handle references all its constituent sources via
    #     fused_from. Each linear lowers to permute + mm fused into one buffer,
    #     so its handle carries a multi-entry fused_from, at least one entry of
    #     which resolves back to the model source line.
    fused = [h for h in collected if h is not None and len(h.fused_from) >= 2]
    assert fused, "no fused handle with a multi-source fused_from was produced"
    assert any(
        c.source is not None and c.source.file.endswith(this_file)
        for h in fused
        for c in h.fused_from
    ), "fused_from did not carry the constituent source line"


@pytest.mark.parametrize(
    "model_cls,expect_rewrite",
    [(_MLP, False), (_RichMLP, True)],
    ids=["mlp_relu", "mlp_gelu_ln"],
)
def test_handles_survive_real_compile(monkeypatch, model_cls, expect_rewrite):
    import torch_spyre  # noqa: F401

    prov_logger = logging.getLogger("spyre.inductor.provenance")
    previous_level = prov_logger.level
    prov_logger.setLevel(logging.ERROR)
    try:
        _assert_handles_survive_real_compile(monkeypatch, model_cls(), expect_rewrite)
        assert prov_logger.level == logging.ERROR
    finally:
        prov_logger.setLevel(previous_level)


def test_graph_break_merges_distinct_source_segments(monkeypatch, tmp_path):
    from torch_spyre.provenance import resolve_provenance_event

    collections, document, path, outputs = _compile_lifecycle_shapes(
        monkeypatch,
        tmp_path,
        _GraphBreakArtifactModel(),
        [(64, 64)],
    )

    assert [tuple(output.shape) for output in outputs] == [(64, 64)]
    assert len(collections) == 2
    assert all(collection is not None for collection in collections)
    compile_ids = {
        collection.compile_id for collection in collections if collection is not None
    }
    assert len(compile_ids) == 2
    assert set(document["upstreamProjections"]) == compile_ids
    assert len(document["kernelIdentities"]) == 2
    assert len(document["kernelOccurrences"]) == 2
    assert document["mergeGeneration"] == 2
    assert document["status"] == "complete"

    for identity in document["kernelIdentities"].values():
        resolved = resolve_provenance_event(identity["eventNameBase"], path)
        assert resolved["status"] == "complete"
        assert len(resolved["occurrences"]) == 1


def test_graph_break_equivalent_source_segments_merge_context(
    monkeypatch, tmp_path, caplog
):
    from torch_spyre.provenance import resolve_provenance_event

    caplog.set_level(logging.WARNING)
    collections, document, path, outputs = _compile_lifecycle_shapes(
        monkeypatch,
        tmp_path,
        _EquivalentGraphBreakArtifactModel(),
        [(64, 64)],
    )

    assert [tuple(output.shape) for output in outputs] == [(64, 64)]
    assert not any(
        "provenance sidecar publication" in record.getMessage()
        for record in caplog.records
    )

    assert len(collections) == 2
    assert all(collection is not None for collection in collections)
    compile_ids = {
        collection.compile_id for collection in collections if collection is not None
    }
    assert len(compile_ids) == 1
    assert set(document["upstreamProjections"]) == compile_ids
    assert len(document["kernelIdentities"]) == 1
    assert len(document["kernelOccurrences"]) == 1
    assert document["mergeGeneration"] == 2
    assert document["status"] == "complete"
    projection = next(iter(document["upstreamProjections"].values()))
    stack_context = next(iter(projection["kernelStackTraces"].values()))
    stack_traces = stack_context["stackTraces"]
    assert any("y = self._segment(x)" in trace for trace in stack_traces)
    assert any("return self._segment(y)" in trace for trace in stack_traces)
    assert projection["upstreamProjectionFailed"] is False
    assert document["diagnostics"] == {}

    identity = next(iter(document["kernelIdentities"].values()))
    resolved = resolve_provenance_event(identity["eventNameBase"], path)
    assert resolved["status"] == "complete"
    assert len(resolved["occurrences"]) == 1


def test_shape_recompile_merges_distinct_compiles(monkeypatch, tmp_path):
    from torch_spyre.provenance import resolve_provenance_event

    collections, document, path, outputs = _compile_lifecycle_shapes(
        monkeypatch,
        tmp_path,
        _ArtifactModel(),
        [(64, 64), (128, 64)],
    )

    assert [tuple(output.shape) for output in outputs] == [
        (64, 64),
        (128, 64),
    ]
    assert len(collections) == 2
    assert all(collection is not None for collection in collections)
    compile_ids = {
        collection.compile_id for collection in collections if collection is not None
    }
    assert len(compile_ids) == 2
    assert set(document["upstreamProjections"]) == compile_ids
    assert len(document["kernelIdentities"]) == 2
    assert len(document["kernelOccurrences"]) == 2
    assert document["mergeGeneration"] == 2
    assert document["status"] == "complete"

    for identity in document["kernelIdentities"].values():
        resolved = resolve_provenance_event(identity["eventNameBase"], path)
        assert resolved["status"] == "complete"
        assert len(resolved["occurrences"]) == 1


def test_artifact_collection_joins_fresh_aliases_and_survives_cache_replay(
    monkeypatch,
    tmp_path,
):
    torch._dynamo.reset()
    monkeypatch.delenv("TORCH_TRACE", raising=False)
    import torch_spyre  # noqa: F401

    from torch._inductor import debug as inductor_debug
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre.constants import DEVICE_NAME
    from torch_spyre.execution.async_compile import SpyreAsyncCompile

    collections = []
    mappings = []
    registration_calls = []
    original_wait = SpyreAsyncCompile.wait
    original_register = inductor_debug.set_kernel_post_grad_provenance_tracing

    def capture_wait(compiler, scope):
        original_wait(compiler, scope)
        collections.append(compiler._last_provenance_collection)
        mappings.append(inductor_debug.dump_inductor_provenance_info())

    def capture_registration(node_schedule, kernel_name, is_extern=False):
        ordinal = original_register(node_schedule, kernel_name, is_extern)
        registration_calls.append((kernel_name, ordinal))
        return ordinal

    monkeypatch.setattr(SpyreAsyncCompile, "wait", capture_wait)
    monkeypatch.setattr(
        inductor_debug,
        "set_kernel_post_grad_provenance_tracing",
        capture_registration,
    )

    model = _ArtifactModel().half().to(DEVICE_NAME).eval()
    x = torch.randn(64, 64, dtype=torch.float16, device=DEVICE_NAME)
    compiled = torch.compile(model, dynamic=False)
    sidecar_path = tmp_path / "spyre_provenance.json"

    with (
        spyre_config.patch({"provenance_artifact_path": str(sidecar_path)}),
        torch._inductor.config.patch("trace.provenance_tracking_level", 1),
    ):
        with fresh_cache():
            with torch.no_grad():
                compiled(x)

        fresh_sidecar_bytes = sidecar_path.read_bytes()
        artifacts = torch.compiler.save_cache_artifacts()
        assert artifacts is not None
        artifact_bytes, _ = artifacts

        torch._dynamo.reset()
        with fresh_cache():
            torch.compiler.load_cache_artifacts(artifact_bytes)
            with torch.no_grad():
                torch.compile(model, dynamic=False)(x)

    assert sidecar_path.read_bytes() == fresh_sidecar_bytes

    assert len(collections) == 2
    fresh, replay = collections
    assert fresh is not None
    assert replay is not None
    assert fresh.has_graph_lowering
    assert not replay.has_graph_lowering

    fresh_registrations = [
        registration
        for occurrence in fresh.kernel_occurrences.values()
        for registration in occurrence.registrations
    ]
    assert fresh_registrations
    assert len(registration_calls) == len(fresh_registrations)
    assert {(kernel_name, ordinal) for kernel_name, ordinal in registration_calls} == {
        (occurrence.compiler_kernel_name, registration.ordinal)
        for occurrence in fresh.kernel_occurrences.values()
        for registration in occurrence.registrations
    }
    assert {registration.alias for registration in fresh_registrations} == set(
        mappings[0]["cppCodeToPost"]
    )
    registration_aliases = {registration.alias for registration in fresh_registrations}
    sidecar = json.loads(fresh_sidecar_bytes)
    assert sidecar["status"] == "complete"
    assert sidecar["diagnostics"] == {}
    projection = sidecar["upstreamProjections"][fresh.compile_id]
    assert projection["upstreamJoin"] == "ok"
    assert set(projection["cppCodeToPost"]) == registration_aliases
    assert projection["cppCodeToPost"] == {
        alias: mappings[0]["cppCodeToPost"][alias]
        for alias in sorted(registration_aliases)
    }
    reachable_post_nodes = {
        post_node
        for post_nodes in projection["cppCodeToPost"].values()
        for post_node in post_nodes
    }
    assert projection["postToPre"] == {
        post_node: mappings[0]["postToPre"][post_node]
        for post_node in sorted(reachable_post_nodes)
    }
    assert set(projection["kernelStackTraces"]) == registration_aliases
    assert all(
        context["postGradNodes"] == projection["cppCodeToPost"][alias]
        for alias, context in projection["kernelStackTraces"].items()
    )

    from torch_spyre._inductor.profiler_event import (
        extract_kernel_provenance_key,
    )

    assert all(
        extract_kernel_provenance_key(identity["eventNameBase"]) == identity_key
        for identity_key, identity in sidecar["kernelIdentities"].items()
    )

    assert all(
        not occurrence.registrations
        for occurrence in replay.kernel_occurrences.values()
    )
    assert replay.compile_id == fresh.compile_id
    assert replay.handles == fresh.handles
    assert replay.kernel_identities == fresh.kernel_identities
    assert replay.kernel_occurrences.keys() == fresh.kernel_occurrences.keys()


def test_split_multi_ops_records_history_during_real_compile(monkeypatch):
    import torch_spyre  # noqa: F401

    from torch_spyre.constants import DEVICE_NAME
    import torch_spyre._inductor.provenance as prov
    import torch_spyre._inductor.split_multi_ops as split

    collected = []
    _orig_decompose = split.decompose_provenance
    _orig_build = prov.build_debug_handle

    # Wrap the production writer, then inspect its exact child carriers before
    # later passes may eliminate compiler-generated intermediates.

    def _decompose(old, news, *args, **kwargs):
        _orig_decompose(old, news, *args, **kwargs)
        collected.extend(_orig_build(child) for child in news)

    monkeypatch.setattr(split, "decompose_provenance", _decompose)
    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    torch._dynamo.reset()

    def split_pointwise(x):
        return torch.relu(x + 1.0)

    x = torch.randn((4, 8, 128), dtype=torch.float16, device=DEVICE_NAME)
    with torch.no_grad():
        torch.compile(split_pointwise, backend="inductor")(x)

    assert collected, "split_multi_ops produced no child handles"
    assert any(
        transform.kind == "decomposition" and transform.pass_name == "split_multi_ops"
        for handle in collected
        if handle is not None
        for transform in handle.transform_history
    ), "split_multi_ops did not record decomposition on a child handle"

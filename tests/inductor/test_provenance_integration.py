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

import logging
import logging.handlers
import os

import pytest
import torch


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
    import torch_spyre._inductor.insert_restickify as restickify
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
    monkeypatch.setattr(restickify, "preserve_provenance", _preserve)
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

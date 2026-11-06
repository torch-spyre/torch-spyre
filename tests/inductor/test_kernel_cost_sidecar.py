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

"""Kernel cost sidecar and compile summary, compiled frontend-only on FakeTensors."""

import json

import pytest
import torch

from torch_spyre._inductor import compile_summary, config, timing_recorder
from torch_spyre._inductor import kernel_cost
from torch_spyre.execution.kernel_cache import get_kernel_registry


@pytest.fixture
def no_card(monkeypatch):
    import torch._inductor.fx_passes.fuse_attention as fuse_attention
    import torch_spyre
    import torch_spyre._C as _C
    from torch_spyre._C import SpyreTensorLayout

    torch_spyre._autoload()
    # FakeTensors never touch the card, so stand in for the runtime.
    monkeypatch.setattr(_C, "device_count", lambda: 1)
    monkeypatch.setattr(torch.spyre._impl, "_initialized", True)
    monkeypatch.setattr(
        torch.Tensor,
        "device_tensor_layout",
        lambda self: (
            SpyreTensorLayout(list(self.size()), self.dtype)
            if self.device.type == "spyre"
            else None
        ),
        raising=False,
    )
    monkeypatch.setattr(fuse_attention, "_sfdp_init", lambda *a, **k: None)
    timing_recorder.RECORDER._reset()
    kernel_cost._written.clear()
    yield
    timing_recorder.RECORDER._reset()


def _compile_mlp():
    from torch._inductor.compile_fx import compile_fx
    from torch._subclasses.fake_tensor import FakeTensorMode

    def model(x, w):
        return torch.nn.functional.gelu(x @ w) + 1.0

    with FakeTensorMode(allow_non_fake_inputs=True):
        inputs = [
            torch.empty(shape, dtype=torch.float16, device="spyre")
            for shape in ((64, 256), (256, 512))
        ]
    compile_fx(torch.fx.symbolic_trace(model), inputs)


def _compile(tmp_path, **overrides):
    patch = {
        "frontend_only": True,
        "kernel_cost_dir": str(tmp_path),
        **overrides,
    }
    with config.patch(patch), torch._inductor.config.patch(force_disable_caches=True):
        _compile_mlp()


def test_sidecar_records_each_kernel_with_cost(no_card, tmp_path):
    _compile(tmp_path)
    (path,) = tmp_path.glob("kernel_cost.*.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records
    for rec in records:
        assert len(rec["provenance_key"]) == 16
        cost = rec["cost"]
        assert cost["predicted_us"] > 0
        assert cost["hbm_read_bytes"] > 0 and cost["hbm_write_bytes"] > 0
        assert cost["cores"] >= 1
        assert cost["floor_inputs"]["bw_peak_gbps"] > 0
    assert any(r["cost"]["macs"] == 64 * 256 * 512 for r in records)
    assert any("mm" in "".join(r["aten_ops"]) for r in records)


def test_sidecar_is_off_by_default(no_card, tmp_path):
    with (
        config.patch({"frontend_only": True}),
        torch._inductor.config.patch(force_disable_caches=True),
    ):
        _compile_mlp()
    assert not list(tmp_path.iterdir())


def test_cost_model_report_is_not_printed_for_the_sidecar(no_card, tmp_path, capsys):
    _compile(tmp_path)
    assert "predicted total" not in capsys.readouterr().err


def test_compile_summary_shape(no_card, tmp_path):
    _compile(tmp_path, timing=True)
    path = compile_summary.dump(str(tmp_path / "summary"))
    summary = json.loads(open(path).read())
    assert set(summary) == {"total_s", "passes", "cache"}
    assert summary["total_s"] > 0
    assert any(k.startswith("CustomPreSchedulingPasses:") for k in summary["passes"])
    assert all(v >= 0 for v in summary["passes"].values())
    assert set(summary["cache"]) == {"entries", "hits", "misses"}


def test_compile_summary_counts_cache_activity(monkeypatch):
    reg = get_kernel_registry()
    monkeypatch.setattr(reg, "_registry", {})
    reg.register("k1", "a", [], [], [])
    reg.register("k2", "b", [], [], [])
    reg.record_miss("k1")
    reg.record_hit("k1")
    reg.record_hit("k2")
    assert compile_summary.build()["cache"] == {"entries": 2, "hits": 2, "misses": 1}


def test_compile_summary_unconfigured_writes_nothing():
    with config.patch({"compile_summary_dir": ""}):
        assert compile_summary.dump() is None

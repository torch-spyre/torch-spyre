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

"""The per-kernel cost sidecar, compiled frontend-only on FakeTensors (no card)."""

import json

import pytest
import torch

from torch_spyre._inductor import config, cost_model_pass, kernel_cost, timing_recorder


@pytest.fixture
def no_card(monkeypatch):
    """Stand in for the runtime so compile_fx runs on FakeTensors without a device."""
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
    # The dedup set is process-wide; a test must not inherit another's keys.
    kernel_cost._written.clear()
    yield
    timing_recorder.RECORDER._reset()
    kernel_cost._written.clear()


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
    patch = {"frontend_only": True, "kernel_cost_dir": str(tmp_path), **overrides}
    with config.patch(patch), torch._inductor.config.patch(force_disable_caches=True):
        _compile_mlp()


def _records(tmp_path):
    (path,) = tmp_path.glob("kernel_cost.*.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_sidecar_records_each_kernel_with_cost(no_card, tmp_path):
    _compile(tmp_path)
    records = _records(tmp_path)
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
    """Pricing for the sidecar is silent; SPYRE_DUMP_COST alone prints the report."""
    _compile(tmp_path)
    assert "predicted total" not in capsys.readouterr().err


def test_each_kernel_is_recorded_once(no_card, tmp_path):
    """One line per provenance key: a re-entered compile must not double-count."""
    _compile(tmp_path)
    keys = [r["provenance_key"] for r in _records(tmp_path)]
    assert len(keys) == len(set(keys))


def test_a_kernel_with_no_priced_group_records_absence(no_card, tmp_path, monkeypatch):
    """No report (an FX-cache replay) means `cost: null`, never a fabricated zero."""
    monkeypatch.setattr(cost_model_pass, "build_report", lambda *a, **k: None)
    _compile(tmp_path)
    records = _records(tmp_path)
    assert records
    assert all(r["cost"] is None for r in records)


def test_records_carry_the_keys_the_ingest_joins_on(no_card, tmp_path):
    """torch-spyre#6548's roofline joins on these names; renaming one drops the join."""
    _compile(tmp_path)
    rec = _records(tmp_path)[0]
    assert {"v", "provenance_key", "kernel", "aten_ops", "cost"} == set(rec)
    assert {
        "predicted_us",
        "hbm_read_bytes",
        "hbm_write_bytes",
        "lx_bytes",
        "macs",
        "cores",
        "groups",
        "floor_inputs",
    } == set(rec["cost"])

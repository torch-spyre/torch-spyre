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

"""The compile summary sidecar, compiled frontend-only on FakeTensors (no card)."""

import json
import logging
import os

import pytest
import torch

from torch_spyre._inductor import config, timing_recorder
from torch_spyre.execution.kernel_cache import get_kernel_registry
from torch_spyre.profiler import _compile_summary


@pytest.fixture
def clean_recorder():
    """The recorder is a process singleton: a test reading it must inherit no events."""
    timing_recorder.RECORDER._reset()
    yield
    timing_recorder.RECORDER._reset()


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


def _compile_timed():
    patch = {"frontend_only": True, "timing": True}
    with config.patch(patch), torch._inductor.config.patch(force_disable_caches=True):
        _compile_mlp()


def test_summary_shape(no_card, tmp_path):
    _compile_timed()
    path = _compile_summary.dump(str(tmp_path))
    summary = json.loads(open(path).read())
    assert set(summary) == {"v", "total_s", "passes", "cache"}
    assert summary["total_s"] > 0
    assert any(k.startswith("CustomPreSchedulingPasses:") for k in summary["passes"])
    assert all(v >= 0 for v in summary["passes"].values())
    assert set(summary["cache"]) == {"entries", "hits", "misses"}


def test_summary_lands_at_the_pid_path(no_card, tmp_path):
    """The name the ingest globs for, written whole rather than in place."""
    _compile_timed()
    path = _compile_summary.dump(str(tmp_path))
    assert os.path.basename(path) == f"compile_summary.{os.getpid()}.json"
    assert [p.name for p in tmp_path.iterdir()] == [os.path.basename(path)]


def test_counts_cache_activity(monkeypatch, clean_recorder):
    reg = get_kernel_registry()
    monkeypatch.setattr(reg, "_registry", {})
    reg.register("k1", "a", [], [], [])
    reg.register("k2", "b", [], [], [])
    reg.record_miss("k1")
    reg.record_hit("k1")
    reg.record_hit("k2")
    assert _compile_summary.build()["cache"] == {"entries": 2, "hits": 2, "misses": 1}


def test_unconfigured_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(_compile_summary.DIR_VAR, raising=False)
    assert _compile_summary.dump() is None
    assert not list(tmp_path.iterdir())


def test_configured_dir_reads_the_environment(monkeypatch):
    monkeypatch.setenv(_compile_summary.DIR_VAR, "/tmp/somewhere")
    assert _compile_summary.configured_dir() == "/tmp/somewhere"
    monkeypatch.delenv(_compile_summary.DIR_VAR)
    assert _compile_summary.configured_dir() == ""


def test_untimed_dump_says_why_it_is_empty(clean_recorder, tmp_path, caplog):
    """Without the recorder the compile fields are empty; say so, don't imply it."""
    with config.patch({"timing": False}), caplog.at_level(logging.WARNING):
        path = _compile_summary.dump(str(tmp_path))
    assert path is not None
    assert _compile_summary.TIMING_VAR in caplog.text
    summary = json.loads(open(path).read())
    assert summary["passes"] == {} and summary["total_s"] == 0


def test_timed_dump_does_not_warn(no_card, tmp_path, caplog):
    """Silent when the recorder is on, so the warning means what it says.

    `timing` is held across the dump, not just the compile: in a real run it comes
    from the environment at import and is still on at exit, which is when the hook
    dumps. `_compile_timed`'s own patch has closed by then.
    """
    with config.patch({"timing": True}), caplog.at_level(logging.WARNING):
        _compile_timed()
        _compile_summary.dump(str(tmp_path))
    assert _compile_summary.TIMING_VAR not in caplog.text

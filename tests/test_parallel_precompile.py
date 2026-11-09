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

"""CPU tests of the forked inner-compile helper (no device needed)."""

import os
import subprocess
import time

import pytest
import torch

import torch_spyre._inductor as spyre_inductor
from torch_spyre._inductor import parallel_precompile as pp


def test_compiles_run_in_children_and_stand_ins_refuse_to_run(monkeypatch, tmp_path):
    parent = os.getpid()

    def fake_inner(name, **kwargs):
        assert os.getpid() != parent
        assert torch._inductor.config.compile_threads == 1
        assert torch.get_num_threads() == 1
        (tmp_path / name).write_text(repr(kwargs))

    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", fake_inner)
    stand_ins = []
    with pp.fork_inner_compiles(3) as pool:
        for i in range(7):
            stand_ins.append(spyre_inductor._spyre_inner_compile(f"g{i}", x=1))
            with pytest.raises(pp.CompileOnlyAbort):
                stand_ins[-1]([1])
            assert len(pool.running) <= 3
    assert spyre_inductor._spyre_inner_compile is fake_inner
    assert (pool.ok, pool.failed) == (7, 0)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"g{i}" for i in range(7)]
    # A stand-in still run after the precompile must fail loudly.
    with pytest.raises(RuntimeError, match="reset_code"):
        stand_ins[0]([1])


def test_failed_hung_and_device_children_are_counted(monkeypatch, tmp_path):
    def fake_inner(kind):
        if kind == "fail":
            raise RuntimeError("boom")
        if kind == "hang":
            # A grandchild, like dbo-opt, must die with the child on a timeout.
            p = subprocess.Popen(["sleep", "60"])
            (tmp_path / "grandchild").write_text(str(p.pid))
            time.sleep(60)
        if kind == "device":
            import torch_spyre._C as spyre_c

            spyre_c.copy_tensor(None, None, False)

    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", fake_inner)
    started = time.time()
    with pp.fork_inner_compiles(2, timeout_s=3) as pool:
        for kind in ("ok", "fail", "hang", "device"):
            spyre_inductor._spyre_inner_compile(kind)
    assert time.time() - started < 30
    assert (pool.ok, pool.failed) == (1, 3)
    grandchild = int((tmp_path / "grandchild").read_text())
    time.sleep(0.5)
    try:  # killed with its group: gone, or a zombie nobody has reaped yet
        with open(f"/proc/{grandchild}/stat") as f:
            assert f.read().split()[2] == "Z"
    except FileNotFoundError:
        pass


def test_a_failure_in_the_caller_kills_the_children(monkeypatch):
    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", lambda: time.sleep(60))
    started = time.time()
    with pytest.raises(KeyboardInterrupt):
        with pp.fork_inner_compiles(2) as pool:
            spyre_inductor._spyre_inner_compile()
            raise KeyboardInterrupt
    assert time.time() - started < 10
    assert not pool.running

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

"""CPU tests: ``aot_compile`` reuses the graphs a parallel precompile pool left in
the FX graph cache (``TORCH_SPYRE_AOT_REUSE_FX=1``)."""

import inspect
import os
import subprocess
import sys
import textwrap

import pytest
import torch
from torch._inductor import compile_fx as cfx
from torch._inductor.codecache import FxGraphCache
from torch._inductor.utils import fresh_cache

import torch_spyre._inductor as spyre_inductor
from torch_spyre._inductor import parallel_precompile as pp


def kern(a, b):
    return ((a * 2 + b).sin() + 1).sum(dim=1)


def other(a, b):
    return ((a * 3 - b).cos() + 1).sum(dim=1)


def _backend(gm, example_inputs):
    # Look the inner compile up at call time: the pool swaps it.
    return cfx.compile_fx(
        gm,
        example_inputs,
        inner_compile=lambda *a, **k: spyre_inductor._spyre_inner_compile(*a, **k),
    )


@pytest.fixture
def compiles(monkeypatch):
    """Counts real inner compiles (code generation) in this process."""
    count = [0]
    real = cfx.fx_codegen_and_compile

    def counting(*args, **kwargs):
        count[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cfx, "fx_codegen_and_compile", counting)
    monkeypatch.setattr(pp, "_pool_fx_keys", {}, raising=False)
    monkeypatch.setattr(
        pp, "_aot_reuse_counts", {"reused": 0, "not seen": 0, "miss": 0}, raising=False
    )
    # A pool child runs on one thread, and the key of a graph with C++ code holds
    # the thread count unless it is fixed: fix it, as a Spyre graph has no C++ code.
    with torch._inductor.config.patch({"cpp.threads": 1}):
        yield count


@pytest.fixture
def reuse_log(monkeypatch):
    """The "aot_compile reuse:" lines the pool module logs."""
    lines = []

    def recording(real):
        def log(msg, *args, **kwargs):
            if msg.startswith("aot_compile reuse:"):
                lines.append(msg % args)
            return real(msg, *args, **kwargs)

        return log

    monkeypatch.setattr(pp.logger, "info", recording(pp.logger.info))
    monkeypatch.setattr(pp.logger, "warning", recording(pp.logger.warning))
    return lines


def _inputs(rows=4):
    torch.manual_seed(0)
    return torch.randn(rows, 8), torch.randn(rows, 8)


def _precompile_in_pool(fn, dynamic):
    torch._dynamo.reset()
    a, b = _inputs()
    with pp.fork_inner_compiles(1) as pool:
        with pytest.raises(pp.CompileOnlyAbort):
            torch.compile(fn, backend=_backend, fullgraph=True, dynamic=dynamic)(a, b)
    assert (pool.ok, pool.failed) == (1, 0)


def _aot_compile(fn, dynamic):
    torch._dynamo.reset()
    a, b = _inputs()
    compiled = torch.compile(
        fn, backend=_backend, fullgraph=True, dynamic=dynamic
    ).aot_compile(((a, b), {}))
    assert torch.allclose(compiled(a, b), fn(a, b))
    return compiled


def _load_in_new_process(path, tmp_path, dynamic):
    """Loads the saved function in a fresh process with empty caches."""
    script = tmp_path / "load.py"
    script.write_text(
        inspect.getsource(kern)
        + textwrap.dedent(
            f"""
            import torch
            from torch._inductor import compile_fx as cfx

            calls = []
            real = cfx.fx_codegen_and_compile
            cfx.fx_codegen_and_compile = lambda *a, **k: calls.append(1) or real(*a, **k)
            with open({str(path)!r}, "rb") as f:
                fn = torch.compiler.load_compiled_function(f)
            ok = True
            for rows in ((4, 6) if {dynamic!r} else (4,)):
                torch.manual_seed(0)
                a, b = torch.randn(rows, 8), torch.randn(rows, 8)
                ok = ok and torch.allclose(fn(a, b), kern(a, b))
            print("RESULT", ok, len(calls))
            """
        )
    )
    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "empty-cache"))
    out = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return out.stdout.split("RESULT")[-1].split()


@pytest.mark.parametrize("dynamic", [False, True])
def test_aot_compile_takes_the_pool_graph(
    monkeypatch, compiles, reuse_log, tmp_path, dynamic
):
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    with fresh_cache():
        _precompile_in_pool(kern, dynamic)
        assert len(pp._pool_fx_keys) == 1
        compiles[0] = 0
        compiled = _aot_compile(kern, dynamic)
        assert compiles[0] == 0
        assert reuse_log[0] == "aot_compile reuse: the pool noted 1 FX keys"
        assert reuse_log[1].startswith(
            "aot_compile reuse: reused pool FX entry (1 reused, 0 not seen, 0 miss"
        )
        assert len(reuse_log) == 2
        path = tmp_path / "kern.aot"
        compiled.save_compiled_function(str(path))
    assert _load_in_new_process(path, tmp_path, dynamic) == ["True", "0"]


def test_switch_off_keeps_the_old_behavior(monkeypatch, compiles, reuse_log, tmp_path):
    monkeypatch.delenv(pp.AOT_REUSE_ENV, raising=False)
    with fresh_cache():
        _precompile_in_pool(kern, False)
        compiles[0] = 0
        compiled = _aot_compile(kern, False)
        assert compiles[0] == 1
        assert not pp._pool_fx_keys and reuse_log == []
        path = tmp_path / "kern.aot"
        compiled.save_compiled_function(str(path))
    assert _load_in_new_process(path, tmp_path, False) == ["True", "0"]


def test_another_graph_is_compiled(monkeypatch, compiles, reuse_log):
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    with fresh_cache():
        _precompile_in_pool(kern, False)
        assert len(pp._pool_fx_keys) == 1
        compiles[0] = 0
        _aot_compile(other, False)
        assert compiles[0] == 1
        assert "no pool entry (key not seen)" in reuse_log[-1]
        # The line names the key fields that differ from the closest pool key.
        assert "differs in: gm" in reuse_log[-1]


def test_an_entry_the_pool_did_not_see_is_not_reused(monkeypatch, compiles, reuse_log):
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    with fresh_cache():
        torch._dynamo.reset()
        a, b = _inputs()
        torch.compile(kern, backend=_backend, fullgraph=True)(a, b)  # no pool
        assert compiles[0] == 1 and not pp._pool_fx_keys
        _aot_compile(kern, False)
        assert compiles[0] == 2
        assert "no pool entry (key not seen)" in reuse_log[-1]
        assert reuse_log[-1].endswith("the pool noted no key")


def test_a_corrupt_entry_is_compiled_again(monkeypatch, compiles, reuse_log, tmp_path):
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    with fresh_cache():
        _precompile_in_pool(kern, False)
        (key,) = pp._pool_fx_keys
        key_dir = FxGraphCache._get_tmp_dir_for_key(key)
        for name in os.listdir(key_dir):
            with open(os.path.join(key_dir, name), "wb") as f:
                f.write(b"not a pickle")
        compiles[0] = 0
        compiled = _aot_compile(kern, False)
        assert compiles[0] == 1
        assert "pool key seen but FX miss" in reuse_log[-1]
        path = tmp_path / "kern.aot"
        compiled.save_compiled_function(str(path))
    assert _load_in_new_process(path, tmp_path, False) == ["True", "0"]


def test_a_graph_built_for_the_childs_thread_count_is_not_reused(
    monkeypatch, compiles, reuse_log
):
    """A child runs on one thread. C++ kernels are built for the thread count, which
    is then part of the key, so the parent must not take the child's entry."""
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch._inductor.config.patch({"cpp.threads": -1}), fresh_cache():
            _precompile_in_pool(kern, False)
            assert len(pp._pool_fx_keys) == 1
            compiles[0] = 0
            _aot_compile(kern, False)
            assert compiles[0] == 1
            assert "pool key seen but FX miss" in reuse_log[-1]
    finally:
        torch.set_num_threads(threads)


def with_constant(a, b):
    table = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    return ((a * 2 + b).sin() * table).sum(dim=1)


def test_the_pool_notes_the_key_its_child_writes(monkeypatch, compiles, reuse_log):
    """The parent keys the graph while the fake mode is on; the child, inside
    compile_fx_inner, with it off. A graph constant is hashed by value."""
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    with fresh_cache():
        _precompile_in_pool(with_constant, False)
        (key,) = pp._pool_fx_keys
        assert os.listdir(FxGraphCache._get_tmp_dir_for_key(key))
        compiles[0] = 0
        _aot_compile(with_constant, False)
        assert compiles[0] == 0 and "reused pool FX entry" in reuse_log[-1]


@pytest.mark.parametrize("plain_mode", [True, False])
def test_aot_compile_under_inference_mode(monkeypatch, compiles, reuse_log, plain_mode):
    """Inputs made under inference_mode are inference tensors. A plain torch.compile
    compiles with inference mode off and drops that flag from its fake inputs;
    aot_compile does not, so the caller must compile in the plain mode for the key
    to match (else the log names the inputs, and only them)."""
    from torch._dynamo.utils import (
        maybe_disable_inference_mode,
        maybe_disable_inference_mode_for_fake_prop,
    )

    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    with fresh_cache(), torch.inference_mode():
        torch._dynamo.reset()
        a, b = _inputs()
        assert a.is_inference()
        with pp.fork_inner_compiles(1):
            with pytest.raises(pp.CompileOnlyAbort):
                torch.compile(kern, backend=_backend, fullgraph=True)(a, b)
        assert len(pp._pool_fx_keys) == 1
        torch._dynamo.reset()
        compiles[0] = 0
        # Shape guards only, like a caller that keys the saved function itself: the
        # tensor guards a plain-mode trace records do not hold for inference tensors.
        aot = torch.compile(
            kern,
            backend=_backend,
            fullgraph=True,
            options={
                "guard_filter_fn": lambda es: [e.guard_type == "SHAPE_ENV" for e in es]
            },
        )
        if plain_mode:
            with (
                maybe_disable_inference_mode(),
                maybe_disable_inference_mode_for_fake_prop(),
            ):
                compiled = aot.aot_compile(((a, b), {}))
        else:
            compiled = aot.aot_compile(((a, b), {}))
        assert torch.allclose(compiled(a, b), kern(a, b))
    if plain_mode:
        assert compiles[0] == 0 and "reused pool FX entry" in reuse_log[-1]
    else:
        assert compiles[0] == 1
        assert reuse_log[-1].endswith(
            "differs in: example_inputs[0], example_inputs[1]"
        )


def test_the_diagnosis_compares_hashes_not_text(monkeypatch):
    """Two fresh Spyre post-fusion pass objects hash the same, but their text holds
    their address: the closest-key line must not name that field."""
    from torch._inductor.codecache import FxGraphCachePickler
    from torch_spyre._inductor.passes import CustomPostFusionPasses

    gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    first, second = CustomPostFusionPasses(), CustomPostFusionPasses()
    pickler = FxGraphCachePickler(gm)
    assert pickler.get_hash(first) == pickler.get_hash(second)
    assert str(first) != str(second)

    def lines(passes, inference):
        name = "inductor_config[_post_fusion_custom_pass]"
        return [
            f"[{pickler.get_hash(passes)}] {name}: {passes}",
            f"[{'b' if inference else 'a'}] example_inputs[0]: is_inference={inference}",
        ]

    monkeypatch.setattr(pp, "_pool_fx_keys", {"pool": lines(first, False)})
    assert pp._closest_pool_key(lines(second, True)) == (
        "closest pool key pool differs in: example_inputs[0]"
    )

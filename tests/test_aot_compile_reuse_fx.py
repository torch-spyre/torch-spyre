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
the FX graph cache (``TORCH_SPYRE_AOT_REUSE_FX=1``).

These tests do not skip on a torch release outside ``pp.AOT_REUSE_RELEASES``: there
the switch is ignored and the reuse tests fail. That is on purpose. A torch upgrade
re-checks ``_pool_graph_state`` and moves ``AOT_REUSE_RELEASES``."""

import os

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
    # A CPU generator: torch.manual_seed would also seed, and so open, a Spyre card.
    g = torch.Generator().manual_seed(0)
    return torch.randn(rows, 8, generator=g), torch.randn(rows, 8, generator=g)


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


def _load_with_empty_caches(path, compiles, dynamic):
    """Loads the saved function with empty caches and runs it: it must compile
    nothing. In this process, not a new one: on a Spyre host a new process would
    open the card that this one already holds (torch.compile asks the accelerator
    for its current stream)."""
    with fresh_cache():
        torch._dynamo.reset()
        compiles[0] = 0
        with open(path, "rb") as f:
            fn = torch.compiler.load_compiled_function(f)
        for rows in (4, 6) if dynamic else (4,):
            a, b = _inputs(rows)
            assert torch.allclose(fn(a, b), kern(a, b))
        assert compiles[0] == 0


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
            "aot_compile reuse: reused FX entry (1 reused, 0 not seen, 0 miss"
        )
        assert len(reuse_log) == 2
        path = tmp_path / "kern.aot"
        compiled.save_compiled_function(str(path))
    _load_with_empty_caches(path, compiles, dynamic)


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
    _load_with_empty_caches(path, compiles, False)


def test_another_graph_is_compiled(monkeypatch, compiles, reuse_log):
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    with fresh_cache():
        _precompile_in_pool(kern, False)
        assert len(pp._pool_fx_keys) == 1
        compiles[0] = 0
        _aot_compile(other, False)
        assert compiles[0] == 1
        assert "FX miss, compiled" in reuse_log[-1]
        # The line names the key fields that differ from the closest pool key.
        assert "differs in: gm" in reuse_log[-1]


def test_an_entry_of_a_plain_compile_is_reused(
    monkeypatch, compiles, reuse_log, tmp_path
):
    """A start with warm caches forks no pool, so no key is noted. When its saved
    aot_compile files no longer match (an upgrade), aot_compile takes the plain
    compile's entry of an earlier start instead of compiling every graph again."""
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    with fresh_cache():
        torch._dynamo.reset()
        a, b = _inputs()
        torch.compile(kern, backend=_backend, fullgraph=True)(a, b)  # no pool
        assert compiles[0] == 1 and not pp._pool_fx_keys
        compiled = _aot_compile(kern, False)
        assert compiles[0] == 1
        assert "reused FX entry" in reuse_log[-1]
        assert reuse_log[-1].endswith("not a pool key")
        path = tmp_path / "kern.aot"
        compiled.save_compiled_function(str(path))
    _load_with_empty_caches(path, compiles, False)


def test_a_stale_entry_is_not_reused(monkeypatch, compiles, reuse_log):
    """An entry made under another Inductor config is another key: compiled."""
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    with fresh_cache():
        torch._dynamo.reset()
        a, b = _inputs()
        with torch._inductor.config.patch({"fallback_random": True}):
            torch.compile(kern, backend=_backend, fullgraph=True)(a, b)
        assert compiles[0] == 1
        _aot_compile(kern, False)
        assert compiles[0] == 2
        assert "FX miss, compiled" in reuse_log[-1]
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
        assert "FX miss, compiled" in reuse_log[-1]
        path = tmp_path / "kern.aot"
        compiled.save_compiled_function(str(path))
    _load_with_empty_caches(path, compiles, False)


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
            assert "FX miss, compiled" in reuse_log[-1]
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
        assert compiles[0] == 0 and "reused FX entry" in reuse_log[-1]


def test_switch_off_the_pool_keys_a_graph_with_the_modes_off(monkeypatch, compiles):
    """With the switch off too, the pool takes a graph's key as compile_fx_inner
    does, with the dispatch modes off, so a second pool finds the entry its child
    wrote and forks no child for it. On the card the key helpers can raise under
    Dynamo's fake mode: the old probe, keyed with the fake mode on, then had no key
    and forked again for a graph the cache held. Simulated here on CPU."""
    real = FxGraphCache.prepare_key

    def raising_under_fake_mode(*args, **kwargs):
        if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE):
            raise RuntimeError("a key helper that needs real tensors")
        return real(*args, **kwargs)

    monkeypatch.setattr(FxGraphCache, "prepare_key", raising_under_fake_mode)
    monkeypatch.delenv(pp.AOT_REUSE_ENV, raising=False)
    with fresh_cache():
        _precompile_in_pool(with_constant, False)
        torch._dynamo.reset()
        a, b = _inputs()
        compiles[0] = 0
        with pp.fork_inner_compiles(1) as pool:
            out = torch.compile(with_constant, backend=_backend, fullgraph=True)(a, b)
        assert (pool.submitted, compiles[0]) == (0, 0)
        assert torch.allclose(out, with_constant(a, b))


def test_an_unchecked_torch_release_ignores_the_switch(
    monkeypatch, compiles, reuse_log
):
    monkeypatch.setenv(pp.AOT_REUSE_ENV, "1")
    monkeypatch.setattr(pp, "AOT_REUSE_RELEASES", ((2, 0),))
    monkeypatch.setattr(pp, "_aot_reuse_release_warned", False)
    with fresh_cache():
        _precompile_in_pool(kern, False)
        compiles[0] = 0
        _aot_compile(kern, False)
    assert compiles[0] == 1 and not pp._pool_fx_keys
    assert len(reuse_log) == 1 and "ignored" in reuse_log[0]


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
        assert compiles[0] == 0 and "reused FX entry" in reuse_log[-1]
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

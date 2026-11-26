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
import pickle
import subprocess
import time
import types

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


@pytest.mark.parametrize("name", pp._DEVICE_ENTRY_POINTS)
def test_a_child_cannot_reach_the_runtime(monkeypatch, name):
    import torch_spyre._C as spyre_c

    if not hasattr(spyre_c, name):
        pytest.skip(f"this build of torch_spyre._C has no {name}")

    def fake_inner():
        getattr(spyre_c, name)()  # refused before any argument is checked

    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", fake_inner)
    with pp.fork_inner_compiles(1) as pool:
        spyre_inductor._spyre_inner_compile()
    assert (pool.ok, pool.failed) == (0, 1)


def test_fork_while_threads_run_and_hold_a_lock(monkeypatch, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock, release = threading.Lock(), threading.Event()

    def holder():
        with lock:
            release.wait(30)

    held = threading.Thread(target=holder)
    held.start()
    executor = ThreadPoolExecutor(4)  # live pool threads at fork time
    list(executor.map(abs, range(8)))

    def fake_inner(name):
        assert threading.active_count() == 1  # only this thread survives fork
        (tmp_path / name).write_text("done")  # compile work that needs no lock

    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", fake_inner)
    try:
        with pp.fork_inner_compiles(2, timeout_s=20) as pool:
            for i in range(4):
                spyre_inductor._spyre_inner_compile(f"g{i}")
        assert (pool.ok, pool.failed) == (4, 0)
        assert len(list(tmp_path.iterdir())) == 4
    finally:
        release.set()
        held.join()
    assert lock.acquire(timeout=5)  # the parent's lock and pool are unaffected
    lock.release()
    assert list(executor.map(abs, [-1])) == [1]
    executor.shutdown()


def test_second_pass_check_reports_hits_and_misses(monkeypatch):
    from torch._dynamo.utils import counters

    warnings = []
    monkeypatch.setattr(pp.logger, "warning", lambda *a: warnings.append(a))
    with pp.SecondPassCheck(expected=2) as check:
        counters["inductor"]["fxgraph_cache_hit"] += 2
    assert (check.hits, check.misses) == (2, 0)
    assert not warnings
    with pp.SecondPassCheck(expected=2) as check:  # a graph no child compiled
        counters["inductor"]["fxgraph_cache_hit"] += 2
        counters["inductor"]["fxgraph_cache_miss"] += 1
    assert (check.hits, check.misses) == (2, 1)
    assert not warnings
    with pp.SecondPassCheck(expected=2) as check:  # a child's compile was lost
        counters["inductor"]["fxgraph_cache_hit"] += 1
        counters["inductor"]["fxgraph_cache_miss"] += 1
    assert (check.hits, check.misses) == (1, 1)
    assert len(warnings) == 1


def _graph_with_outputs(*specs):
    """A GraphModule whose output node carries fake metadata like Inductor's."""
    g = torch.fx.Graph()
    x = g.placeholder("x")
    outs = []
    for spec in specs:
        if spec is None:
            outs.append(0)  # a constant output, passed through as is
            continue
        shape, dtype = spec
        node = g.call_function(torch.ops.aten.mul.Tensor, (x, 2))
        node.meta["val"] = torch.empty(shape, dtype=dtype)  # a fake tensor in Inductor
        outs.append(node)
    g.output(tuple(outs))
    return torch.fx.GraphModule(torch.nn.Module(), g)


def test_placeholders_follow_the_output_metadata_when_a_child_reports_nothing(
    monkeypatch, tmp_path
):
    parent = os.getpid()

    def fake_inner(gm, example_inputs, **kwargs):
        assert os.getpid() != parent
        (tmp_path / str(len(gm.graph.output_node().args[0]))).write_text("")

    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", fake_inner)
    gm = _graph_with_outputs(((2, 3), torch.float16), None, ((4,), torch.int64))
    with pp.fork_inner_compiles(2, placeholders=True) as pool:
        assert pool.submitted == 0
        stand_in = spyre_inductor._spyre_inner_compile(gm, [torch.empty(1)])
        assert pool.submitted == 1
        out = stand_in([torch.empty(1)])
        assert [getattr(o, "shape", o) for o in out] == [(2, 3), 0, (4,)]
        assert [getattr(o, "dtype", None) for o in out] == [
            torch.float16,
            None,
            torch.int64,
        ]
        assert out[0].device.type == "cpu" and not out[0].any()
        # The same graph passed by keyword is read the same way.
        stand_in = spyre_inductor._spyre_inner_compile(gm=gm, example_inputs=[])
        assert stand_in([])[2].shape == (4,)
    assert (pool.submitted, pool.ok, pool.failed) == (2, 2, 0)
    assert not pool.reports  # every report pipe is closed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["3"]
    with pytest.raises(RuntimeError, match="reset_code"):
        stand_in([])


def test_placeholders_take_the_layouts_the_child_reports(monkeypatch):
    def fake_inner(gm, example_inputs):
        time.sleep(0.2)  # the parent waits for the report, not for the exit
        pp._report_outputs(
            [
                ("new", [2, 3], [1, 2], 0, torch.float32, "cpu", None),
                None,
                ("alias", 0, [2], [2], 1),
                ("view", [6], [1], [2, 1], [2, 1], 1, torch.float32, "cpu", None),
            ]
        )
        time.sleep(0.5)

    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", fake_inner)
    gm = _graph_with_outputs(
        ((2, 3), torch.float16), None, ((2,), torch.float32), ((2, 1), torch.float32)
    )
    x = torch.arange(6.0)
    with pp.fork_inner_compiles(1, placeholders=True) as pool:
        stand_in = spyre_inductor._spyre_inner_compile(gm, [x])
        t0 = time.time()
        out = stand_in([x])
        assert time.time() - t0 < 0.45  # returned before the child exited
        assert pool.running  # the child is still compiling
        assert out[0].shape == (2, 3) and out[0].stride() == (1, 2)
        assert out[0].dtype == torch.float32 and not out[0].any()
        assert out[1] == 0
        assert torch.equal(out[2], x[1:5:2]) and out[2].data_ptr() == x[1:].data_ptr()
        # A view of a fresh buffer: zeros with the view's shape, strides and offset.
        assert out[3].shape == (2, 1) and out[3].stride() == (2, 1)
        assert out[3].storage_offset() == 1 and not out[3].any()
        # A second run of the same stand-in reuses the report.
        assert stand_in([x])[2].stride() == (2,)
    assert (pool.ok, pool.failed) == (1, 0)


def test_a_single_tensor_output_is_still_a_list(monkeypatch):
    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", lambda gm, inputs: None)
    g = torch.fx.Graph()
    x = g.placeholder("x")
    node = g.call_function(torch.ops.aten.neg.default, (x,))
    node.meta["val"] = torch.empty(5)
    g.output(node)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    with pp.fork_inner_compiles(1, placeholders=True):
        out = spyre_inductor._spyre_inner_compile(gm, [])([])
    assert len(out) == 1 and out[0].shape == (5,)


def test_output_specs_tell_fresh_buffers_from_views_of_inputs():
    from torch._inductor import ir
    from torch._inductor.graph import GraphLowering
    from torch._inductor.virtualized import V

    cpu = torch.device("cpu")
    arg = ir.InputBuffer(
        name="arg0_1", layout=ir.FixedLayout(cpu, torch.float32, [6], [1])
    )
    view = ir.ReinterpretView(
        data=ir.StorageBox(arg),
        layout=ir.FixedLayout(cpu, torch.float32, [2], [2], offset=1),
    )
    buf = ir.Buffer(
        name="buf0", layout=ir.FixedLayout(cpu, torch.float16, [2, 3], [1, 2])
    )
    # A view of a fresh device buffer: the layout lives on the buffer, not the view.
    laid_out = ir.FixedLayout(cpu, torch.float16, [4, 2], [2, 1])
    laid_out.device_layout = "L"
    buf1 = ir.Buffer(name="buf1", layout=laid_out)
    buf1_view = ir.ReinterpretView(
        data=ir.StorageBox(buf1),
        layout=ir.FixedLayout(cpu, torch.float16, [4, 2, 1], [2, 1, 1]),
    )
    plain_view = ir.ReinterpretView(
        data=ir.StorageBox(buf),
        layout=ir.FixedLayout(cpu, torch.float16, [3, 2], [2, 1]),
    )
    graph = types.SimpleNamespace(
        graph_input_names=["arg0_1"],
        graph_outputs=[
            ir.TensorBox(ir.StorageBox(buf)),
            7,
            ir.TensorBox(view),
            ir.TensorBox(buf1_view),
            ir.TensorBox(plain_view),
        ],
    )
    specs = pp._output_specs(graph)
    assert specs == [
        ("new", [2, 3], [1, 2], 0, torch.float16, "cpu", None),
        None,
        ("alias", 0, [2], [2], 1),
        ("view", [4, 2], [2, 1], [4, 2, 1], [2, 1, 1], 0, torch.float16, "cpu", "'L'"),
        ("new", [3, 2], [2, 1], 0, torch.float16, "cpu", None),
    ]
    del GraphLowering, V


def test_a_graph_the_fx_cache_holds_compiles_in_place(monkeypatch):
    """An AOT-cache bypass reaches the inner compile on a warm start; a fork for a
    cache hit would cost a child and turn the warm start into a second pass."""
    calls = []
    monkeypatch.setattr(
        spyre_inductor,
        "_spyre_inner_compile",
        lambda *a, **k: calls.append(k) or "code",
    )
    monkeypatch.setattr(pp, "_fx_cache_holds", lambda *a, **k: k.get("x") == "hit")
    with pp.fork_inner_compiles(2, placeholders=True) as pool:
        hit = spyre_inductor._spyre_inner_compile("gm", [], x="hit")
        miss = spyre_inductor._spyre_inner_compile(
            torch.fx.symbolic_trace(torch.nn.Identity()), [torch.empty(1)], x="miss"
        )
    assert hit == "code" and calls == [{"x": "hit"}]
    assert isinstance(miss, pp._StandIn) and pool.submitted == 1


def test_the_cache_probe_never_raises():
    assert pp._fx_cache_holds("not a graph", x=1) is False


def test_without_placeholders_nothing_changes(monkeypatch):
    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", lambda *a, **k: None)
    with pp.fork_inner_compiles(1) as pool:
        stand_in = spyre_inductor._spyre_inner_compile(
            _graph_with_outputs(((1,), torch.float32)), []
        )
        with pytest.raises(pp.CompileOnlyAbort):
            stand_in([])
    assert pool.submitted == 1 and not pool.reports


def test_a_caller_can_switch_an_open_pool_to_abort_stand_ins(monkeypatch):
    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", lambda *a, **k: None)
    gm = _graph_with_outputs(((1,), torch.float32))
    with pp.fork_inner_compiles(1, placeholders=True) as pool:
        pool.placeholders = False
        stand_in = spyre_inductor._spyre_inner_compile(gm, [])
        with pytest.raises(pp.CompileOnlyAbort):
            stand_in([])
        assert not pool.reports  # nothing to wait for
        pool.placeholders = True
        out = spyre_inductor._spyre_inner_compile(gm, [])([])
        assert len(out) == 1 and out[0].shape == (1,)


def test_the_report_pass_sends_the_specs_once_and_only_in_a_child(monkeypatch):
    graph = types.SimpleNamespace(graph_input_names=[], graph_outputs=[3])
    # Not a child: nothing to send, and the graph is left alone.
    pp.report_output_layouts(graph)
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(pp, "_report_fd", write_fd)
    pp.report_output_layouts(types.SimpleNamespace(parent=object()))  # a subgraph
    assert pp._report_fd == write_fd
    pp.report_output_layouts(graph)
    assert pp._report_fd is None and pp._reported == [None]
    pp.report_output_layouts(graph)  # a second top-level codegen reports nothing more
    assert pickle.loads(os.read(read_fd, 1 << 16)) == [None]
    assert os.read(read_fd, 1) == b""  # the write end is closed
    os.close(read_fd)


def test_a_child_solver_gets_its_share_of_the_cpus(monkeypatch):
    import torch.distributed as dist

    from torch_spyre._inductor import config as spyre_config

    monkeypatch.setattr(pp, "_host_cpus", lambda: 64)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    with spyre_config.patch(num_cpus=None):  # nothing explicit: autodetected host
        assert pp._solver_cpu_share(7) == 16  # 64 CPUs / 4 local ranks, any workers
        assert pp._solver_cpu_share(1) == 16
        monkeypatch.delenv("LOCAL_WORLD_SIZE")
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 8)
        assert pp._solver_cpu_share(7) == 8  # no local count: the process group's size
        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        assert pp._solver_cpu_share(7) == 16  # one rank: capped per child
        monkeypatch.setattr(pp, "_host_cpus", lambda: 9)
        assert pp._solver_cpu_share(7) == 9  # a small host stays whole
        monkeypatch.setattr(pp, "_host_cpus", lambda: 384)
        assert pp._solver_cpu_share(8) == 48  # a big host split across the workers
        assert pp._solver_cpu_share(64) == 16
        monkeypatch.setattr(pp, "_host_cpus", lambda: 1)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        assert pp._solver_cpu_share(7) == 2  # never below two
    with spyre_config.patch(num_cpus=6):
        assert pp._solver_cpu_share(7) == 6  # an explicit SPYRE_NUM_CPUS is the budget


def test_a_time_limited_solve_is_counted_not_failed(monkeypatch):
    def compile_in_child(*args, **kwargs):
        pp._time_limited += 1  # what the solver hook does in the child

    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", compile_in_child)
    with pp.fork_inner_compiles(1) as pool:
        spyre_inductor._spyre_inner_compile("g", graph_id=4)
        spyre_inductor._spyre_inner_compile("g", graph_id=5)
    assert (pool.ok, pool.failed, pool.time_limited) == (2, 0, 2)
    assert pp._time_limited == 0  # the parent's counter is untouched


def test_the_solver_hook_counts_feasible_plans_under_a_time_limit(monkeypatch):
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    from torch_spyre._inductor.scratchpad import allocator
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver

    monkeypatch.setattr(
        CpSatLayoutSolver,
        "_solve_and_record",
        lambda self, solver, model, **kw: solver.status,
    )
    monkeypatch.setattr(allocator, "_cpsat_solver_class", lambda: CpSatLayoutSolver)
    monkeypatch.setattr(pp, "_time_limited", 0)
    monkeypatch.setattr(pp, "_label", "graph 7 (9 nodes)")
    warnings = []
    monkeypatch.setattr(pp.logger, "warning", lambda *a: warnings.append(a))
    pp._count_time_limited_solves()  # what a child does at setup
    assert allocator._cpsat_solver_class() is CpSatLayoutSolver  # the first plan asks
    assert allocator._cpsat_solver_class() is CpSatLayoutSolver  # and it hooks once
    parameters = types.SimpleNamespace(max_time_in_seconds=60.0, num_search_workers=16)
    solver = types.SimpleNamespace(
        status=cp_model.FEASIBLE, parameters=parameters, WallTime=lambda: 60.2
    )
    solve = CpSatLayoutSolver._solve_and_record
    assert solve(None, solver, None, objective=True) == cp_model.FEASIBLE
    solver.status = cp_model.OPTIMAL  # proven: not counted
    assert solve(None, solver, None) == cp_model.OPTIMAL
    solver.status, parameters.max_time_in_seconds = cp_model.FEASIBLE, 0  # no limit set
    assert solve(None, solver, None) == cp_model.FEASIBLE
    assert pp._time_limited == 1
    assert len(warnings) == 1 and warnings[0][1] == "graph 7 (9 nodes)"


def test_a_collective_under_lx_context_switching_warns_once(monkeypatch):
    from torch_spyre._inductor import config as spyre_config

    g = torch.fx.Graph()
    x = g.placeholder("x")
    y = g.call_function(torch.ops._c10d_functional.all_reduce.default, (x, "sum", "tp"))
    g.output((y,))
    with_collective = torch.fx.GraphModule(torch.nn.Module(), g)
    warnings = []
    monkeypatch.setattr(pp.logger, "warning", lambda *a: warnings.append(a))
    monkeypatch.setattr(pp, "_warned_collectives", False)
    with spyre_config.patch(enable_lx_context_switching=False):
        pp._warn_collectives_in_lx(with_collective)
    assert not warnings  # the workaround is in place
    with spyre_config.patch(enable_lx_context_switching=True):
        pp._warn_collectives_in_lx(_graph_with_outputs(((2,), torch.float32)))
        assert not warnings  # no collective, nothing to warn about
        pp._warn_collectives_in_lx(with_collective)
        pp._warn_collectives_in_lx(with_collective)
    assert len(warnings) == 1 and "6563" in warnings[0][0]


def test_the_probe_computes_the_key_a_real_compile_writes(monkeypatch):
    """Pins ``_fx_cache_key`` to the key ``compile_fx_inner`` writes: a torch upgrade
    that changed the key would otherwise turn every probe into a silent miss."""
    from torch._inductor.codecache import FxGraphCache
    from torch._inductor.utils import fresh_cache

    original = spyre_inductor._spyre_inner_compile
    keys = []

    def record(*args, **kwargs):
        keys.append(pp._fx_cache_key(*args, **kwargs))
        assert not pp._fx_cache_holds(*args, **kwargs)  # a fresh cache misses
        return original(*args, **kwargs)

    monkeypatch.setattr(spyre_inductor, "_spyre_inner_compile", record)
    x = torch.rand(8, 64, dtype=torch.float16).to("spyre")
    with fresh_cache():
        torch._dynamo.reset()
        torch.compile(lambda t: t + 1)(x)
        assert keys and all(keys)
        for key in keys:  # the compile wrote its entry under exactly this key
            key_dir = FxGraphCache._get_tmp_dir_for_key(key)
            assert os.path.isdir(key_dir) and os.listdir(key_dir)

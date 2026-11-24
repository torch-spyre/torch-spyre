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

"""Run the Spyre inner compiles of many graphs in forked child processes.

Dynamo, AOTAutograd and the joint-graph passes stay in this process, because they
may run small ops on the device. The inner compile (lowering, LX planning, code
generation, dbo-opt) is host work only, and it writes the FX graph cache and the
Spyre kernel cache.

Inside ``fork_inner_compiles(n)``, each inner compile runs in a forked child, at
most ``n`` at once, and this process gets a stand-in graph that raises
``CompileOnlyAbort`` when run. Leaving the block waits for all children. The
caller then resets those frames (``torch._dynamo.reset_code``) and runs the graphs
again: each compile now finds the children's entries in the caches. A child that
fails only costs time, because the second run compiles whatever is missing.
``SecondPassCheck`` around that second run logs a warning with the numbers when
any compile missed the FX graph cache, so a lost speedup does not go unnoticed.

With ``placeholders=True`` the stand-in returns placeholder tensors instead of
raising, so a caller that traces a whole forward (a model warm-up bucket) gets past
the first graph and submits every graph of that forward. The FX graph cache key
holds each input's device layout and storage offset, so a placeholder must carry
the layout the real output will have: the child reports its graph's output layouts
through a pipe as soon as the pre-scheduling passes have fixed them, before LX
planning (most of a compile's time) and the dbo-opt backend, and the stand-in waits
for that report before it allocates. A child that reports nothing (its compile hit
the cache, or failed) gets default-layout zeros, and the graphs downstream of it
compile again in the second run.

The stand-ins are zeros, so the traced forward must not branch on tensor values or
take shapes from them (an ``.item()``, a data-dependent mask): the parent would then
trace other graphs than the real run does, which costs a recompile in the second run
but no wrong answers. The first call of a stand-in blocks until its child has
reported, so the parent's trace serializes on each child's frontend (Dynamo through
the pre-scheduling passes); only LX planning and dbo-opt overlap, which is where most
of a compile's time goes, so frontend-bound graphs gain little. The pipe's writer is
our own forked child, so the specs it pickles and the layout repr the parent evaluates
are trusted input.

What a child may use: the compile itself (Python, the Inductor and Spyre lowering
code, CP-SAT, and dbo-opt as a subprocess). What it may not use: the device. Every
Python entry point of the Spyre runtime (start, allocation, copies, fills, streams,
synchronize, kernel preparation) raises in a child, and so does the eager copy op.
The child runs on one thread with ``compile_threads = 1``, so it starts no worker
pool of its own (#4427's pool stays in the parent), and it never takes a lock that
another thread of the parent could hold at fork time: such a lock stays held in
the child, so a compile that needed it would hang and be killed by the timeout.
Its LX planning solver gets the rank's share of the CPUs: the autodetected CPU count
divided by the ranks on this host (``LOCAL_WORLD_SIZE``, else the process group's
size), at least two, and at most 16 unless the share split across the workers is
more; an explicit ``SPYRE_NUM_CPUS`` is this process's budget already and is used as
it is. Sized to the whole host, as each parent's is, the concurrent solvers starve
each other, and a starved solve returns a worse plan for the same graph (slower
kernels, not just a slower compile). A solve that stops at
``cpsat_time_limit_seconds`` returns the best plan found so far, and the FX graph
cache keeps that plan: the child counts such solves and exits with code 3, the pool
reports them in ``time_limited`` and in its summary line, so a slow kernel can be
traced back to a starved warm-up. With ``enable_lx_context_switching`` on (the
default) the LX planner may keep an all-reduce input in LX, and the wrapper a child
caches then calls the collective on a buffer it never allocated, a ``NameError`` in
the second run (torch-spyre#6563): the first graph with a collective logs a warning
once, naming ``ENABLE_LX_CONTEXT_SWITCHING=0`` as the workaround until the planner is
fixed.

Limits: Linux only (``os.fork``). Each child is a copy-on-write copy of the caller.
For a vLLM engine with Granite 3.3 8B and 8 workers, the peak resident memory of the
process tree grew from 16.9 GB to 25.1 GB (about 1 GB per child).
"""

import contextlib
import os
import pickle
import select
import signal
import time
from typing import Any, Callable, Optional

import torch

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("parallel_precompile")


class CompileOnlyAbort(Exception):
    """Raised by a stand-in graph while its compile runs in a child."""


class _StandIn:
    """The graph this process gets back for a compile that runs in a child.

    ``outputs`` is the graph's output metadata (fake tensors and constants) when the
    pool hands out placeholders, and ``pid`` the child whose output-layout report
    the first call waits for.
    """

    _boxed_call = True

    def __init__(
        self, pool: "_Pool", outputs: Optional[list] = None, pid: Optional[int] = None
    ) -> None:
        self._pool = pool
        self._outputs = outputs
        self._pid = pid
        self._specs: Optional[list] = None

    def __call__(self, args: list) -> Any:
        if not self._pool.open:
            raise RuntimeError(
                "parallel precompile: a stand-in graph ran after the precompile; "
                "reset its frame with torch._dynamo.reset_code"
            )
        if self._outputs is None:
            raise CompileOnlyAbort()
        if self._pid is not None:
            self._specs = self._pool.report(self._pid)
            self._pid = None
        return [
            _placeholder(meta, self._specs[i] if self._specs else None, args)
            if isinstance(meta, torch.Tensor)
            else meta
            for i, meta in enumerate(self._outputs)
        ]


def _output_metadata(gm: Any) -> list:
    """The fake tensors (or constants) the graph returns, from its output node."""
    outputs = gm.graph.output_node().args[0]
    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]
    found = []
    for out in outputs:
        if isinstance(out, torch.fx.Node):
            val = out.meta.get("val")
            found.append(val if isinstance(val, torch.Tensor) else None)
        else:
            found.append(out)
    return found


def _placeholder(meta: torch.Tensor, spec: Optional[tuple], args: list) -> torch.Tensor:
    """A tensor with the metadata of a graph output, from the child's spec if any.

    A spec is ``("alias", input_index, size, stride, offset)`` for an output that
    is a view of a graph input, ``("new", size, stride, offset, dtype, device,
    layout_repr)`` for a buffer the graph allocates, or ``("view", base_size,
    base_stride, size, stride, offset, dtype, device, layout_repr)`` for a view of
    such a buffer, where ``layout_repr`` is the ``SpyreTensorLayout`` as the generated
    wrapper spells it (None off the device).
    """
    if spec is not None and spec[0] == "alias":
        _, index, size, stride, offset = spec
        base = args[index]
        layout = (
            base.device_tensor_layout()
            if hasattr(base, "device_tensor_layout")
            else None
        )
        if layout is not None:
            from torch_spyre._C import as_strided_with_layout

            return as_strided_with_layout(
                base, tuple(size), tuple(stride), offset, layout
            )
        return torch.as_strided(base, size, stride, offset)
    if spec is not None and spec[0] == "view":
        _, base_size, base_stride, size, stride, offset, dtype, device, layout_repr = (
            spec
        )
        base = _fresh(base_size, base_stride, dtype, torch.device(device), layout_repr)
        if layout_repr is not None:
            from torch_spyre._C import as_strided_with_layout

            return as_strided_with_layout(
                base, tuple(size), tuple(stride), offset, base.device_tensor_layout()
            )
        return torch.as_strided(base, size, stride, offset)
    if spec is not None and spec[0] == "new":
        _, size, stride, _offset, dtype, device, layout_repr = spec
        return _fresh(size, stride, dtype, torch.device(device), layout_repr)
    return torch.zeros(
        tuple(int(d) for d in meta.shape), dtype=meta.dtype, device=meta.device
    )


def _fresh(
    size: list,
    stride: list,
    dtype: torch.dtype,
    device: torch.device,
    layout_repr: Optional[str],
) -> torch.Tensor:
    """A zeroed buffer with the layout the child reported, or the eager one without it."""
    if layout_repr is not None:
        from torch_spyre._C import fill_tensor, spyre_empty_with_layout
        from torch_spyre._inductor.wrapper import spec_eval_namespace

        layout = eval(layout_repr, spec_eval_namespace())  # noqa: S307 - our own repr
        out = spyre_empty_with_layout(
            tuple(size), tuple(stride), dtype, layout, device=device
        )
        fill_tensor(out, 0.0)
        return out
    out = torch.empty_strided(size, stride, dtype=dtype, device=device)
    try:
        return out.zero_()
    except RuntimeError:
        # torch-spyre fills dense storage only; a padded-stride output gets a dense stand-in.
        return torch.zeros(tuple(size), dtype=dtype, device=device)


def _output_specs(graph: Any) -> list:
    """The specs of a lowered graph's outputs (see ``_placeholder``), after codegen."""
    from torch._inductor import ir

    specs: list = []
    for out in graph.graph_outputs:
        if not isinstance(out, ir.IRNode):
            specs.append(None)
            continue
        node = out
        while isinstance(node, ir.MutableBox):
            node = node.data
        try:
            layout = node.get_layout()
            size = [int(s) for s in layout.size]
            stride = [int(s) for s in layout.stride]
            offset = int(layout.offset)
        except Exception:  # noqa: BLE001 - symbolic or layout-less output: no spec
            specs.append(None)
            continue
        base = node.data if isinstance(node, ir.ReinterpretView) else None
        while isinstance(base, ir.MutableBox):
            base = base.data
        if (
            isinstance(base, ir.InputBuffer)
            and base.get_name() in graph.graph_input_names
        ):
            index = graph.graph_input_names.index(base.get_name())
            specs.append(("alias", index, size, stride, offset))
            continue
        device_layout = getattr(layout, "device_layout", None)
        if base is not None and device_layout is None:
            # A view of a fresh buffer: the device layout lives on the buffer.
            try:
                base_layout = base.get_layout()
                base_device_layout = getattr(base_layout, "device_layout", None)
                base_size = [int(s) for s in base_layout.size]
                base_stride = [int(s) for s in base_layout.stride]
            except Exception:  # noqa: BLE001 - symbolic or layout-less base: a plain buffer
                base_device_layout = None
            if base_device_layout is not None:
                specs.append(
                    (
                        "view",
                        base_size,
                        base_stride,
                        size,
                        stride,
                        offset,
                        layout.dtype,
                        str(layout.device),
                        repr(base_device_layout),
                    )
                )
                continue
        specs.append(
            (
                "new",
                size,
                stride,
                offset,
                layout.dtype,
                str(layout.device),
                repr(device_layout) if device_layout is not None else None,
            )
        )
    return specs


_report_fd: Optional[int] = None  # in a child: the pipe its output specs go to
_reported: Optional[list] = None  # in a child: the specs it sent


def _report_outputs(specs: Optional[list]) -> None:
    """Child: send the graph's output specs to the parent, once, and close the pipe."""
    global _report_fd, _reported
    if _report_fd is None:
        return
    fd, _report_fd = _report_fd, None
    _reported = specs
    try:
        data = pickle.dumps(specs)
        while data:
            data = data[os.write(fd, data) :]
    finally:
        os.close(fd)


def report_output_layouts(graph: Any) -> None:
    """Pre-scheduling pass: in a forked compile child, send the top-level graph's
    output specs to the parent. A no-op in every other process."""
    if _report_fd is None or getattr(graph, "parent", None) is not None:
        return
    try:
        _report_outputs(_output_specs(graph))
    except Exception:  # noqa: BLE001 - the parent falls back to default layouts
        logger.warning(
            "parallel precompile: could not report the output layouts", exc_info=True
        )
        _report_outputs(None)


def _install_output_report() -> None:
    """Child: after codegen, report the output specs if no pass did, and warn if
    they no longer match what was reported (the second run then misses the cache)."""
    from torch._inductor.graph import GraphLowering

    original = GraphLowering.codegen

    def codegen(self: Any) -> Any:
        result = original(self)
        if getattr(self, "parent", None) is None:
            if _report_fd is not None:
                report_output_layouts(self)
            elif _reported is not None:
                try:
                    final = _output_specs(self)
                except Exception:  # noqa: BLE001 - a check only
                    final = None
                if final != _reported:
                    logger.warning(
                        "parallel precompile: the output layouts changed after the "
                        "report: reported %s, final %s",
                        _reported,
                        final,
                    )
        return result

    GraphLowering.codegen = codegen


# The ``torch_spyre._C`` functions that reach the runtime or the device.
_DEVICE_ENTRY_POINTS = (
    "start_runtime",
    "free_runtime",
    "set_device",
    "synchronize",
    "spyre_empty_with_layout",
    "spyre_empty_reserved",
    "empty_with_layout",
    "copy_tensor",
    "copy_kv_page_raw",
    "fill_tensor",
    "get_composite_address",
    "get_stream_from_pool",
    "set_current_stream",
    "prepare_kernel",
)


def _no_device(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("parallel precompile: a forked compile tried to use the device")


def _host_cpus() -> int:
    """The CPUs this process may use, as torch-spyre's config autodetects them."""
    from torch_spyre._inductor import config as spyre_config

    return spyre_config.get_cpu_count()


def _local_ranks() -> int:
    """Ranks on this host: ``LOCAL_WORLD_SIZE`` when set (vLLM's workers set it),
    else the process group's size, else one. The global world size would over-divide
    the CPUs of a host that holds only some of the ranks."""
    try:
        return max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "")))
    except ValueError:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return max(1, dist.get_world_size())
    except Exception:  # noqa: BLE001 - no process group means one rank
        pass
    return 1


_SOLVER_CPUS_PER_CHILD = 16  # a child's solver share, unless the rank's split is more


def _solver_cpu_share(workers: int) -> int:
    """CPUs for one child's LX planning solver.

    Every rank on the host forks ``workers`` children at once, and each child's
    CP-SAT solve would otherwise take every CPU of the host, as the parent's does.
    Starved solves return worse plans, not only later ones. A child gets its rank's
    share of the host, capped at ``_SOLVER_CPUS_PER_CHILD`` unless that share split
    across the workers is larger. With one rank on 64 CPUs and 7 workers, 16 or 8
    CPUs per child took a cold Gemma-4 start from 589 s to 397 s and 388 s; at four
    ranks the per-rank share (16) took the block warm-up from 338 s to 213 s. An
    explicit ``SPYRE_NUM_CPUS`` (``config.num_cpus``) is already this process's
    budget, so it is used as it is; only the autodetected host count is divided.
    """
    from torch_spyre._inductor import config as spyre_config

    if spyre_config.num_cpus is not None and spyre_config.num_cpus > 0:
        return spyre_config.num_cpus
    rank_share = max(2, _host_cpus() // _local_ranks())
    return min(rank_share, max(_SOLVER_CPUS_PER_CHILD, rank_share // max(1, workers)))


_TIME_LIMITED_EXIT = 3  # a child's exit code: compiled, but a solve hit its time limit
_time_limited = 0  # in a child: LX planning solves that stopped at their time limit
_label = ""  # in a child: the graph it compiles, for its log lines
_warned_collectives = False  # the torch-spyre#6563 warning goes out once per process


def _count_time_limited_solves() -> None:
    """Child: count the LX planning solves that stop at ``cpsat_time_limit_seconds``.

    CP-SAT then returns the best feasible plan it found (status ``FEASIBLE``, not
    ``OPTIMAL``), and the FX graph cache keeps that plan for every later start. The
    allocator imports the solver module lazily, so the hook sits on its accessor and
    reaches the solver class when the first plan asks for it.
    """
    from torch_spyre._inductor.scratchpad import allocator

    original = allocator._cpsat_solver_class

    def solver_class() -> type:
        cls = original()
        _hook_time_limited_solves(cls)
        return cls

    allocator._cpsat_solver_class = solver_class


def _hook_time_limited_solves(cls: type) -> None:
    """Wrap ``cls._solve_and_record``, the one way the CP-SAT solver solves, to count
    the solves that stopped at their time limit and warn about each; once per class.
    A solver class without it (the greedy fallback) is left alone."""
    solve = getattr(cls, "_solve_and_record", None)
    if solve is None or getattr(solve, "_counts_time_limited", False):
        return
    try:
        from ortools.sat.python import cp_model  # loaded along with the solver class
    except ImportError:  # pragma: no cover - no CP-SAT, nothing to count
        return

    def solve_and_record(self: Any, solver: Any, model: Any, **kwargs: Any) -> int:
        status = solve(self, solver, model, **kwargs)
        limit = solver.parameters.max_time_in_seconds
        if status == cp_model.FEASIBLE and limit:
            global _time_limited
            _time_limited += 1
            logger.warning(
                "parallel precompile: an LX planning solve of %s stopped at its %.0f s "
                "time limit after %.1f s on %d CPUs; the plan it caches may be slower "
                "than a finished one",
                _label or "a graph",
                limit,
                solver.WallTime(),
                solver.parameters.num_search_workers,
            )
        return status

    solve_and_record._counts_time_limited = True  # type: ignore[attr-defined]
    cls._solve_and_record = solve_and_record


def _graph_label(gm: Any, kwargs: dict) -> str:
    nodes = len(gm.graph.nodes) if hasattr(gm, "graph") else 0
    return f"graph {kwargs.get('graph_id', '?')} ({nodes} nodes)"


def _warn_collectives_in_lx(gm: Any) -> None:
    """Warn once when a graph with a collective compiles in a child while LX context
    switching is on: the planner may keep the collective's input in LX, and the
    cached wrapper then names a buffer it never allocated (torch-spyre#6563)."""
    global _warned_collectives
    if _warned_collectives or not hasattr(gm, "graph"):
        return
    from torch_spyre._inductor import config as spyre_config

    if not spyre_config.enable_lx_context_switching:
        return
    if not any(
        node.op == "call_function" and "_c10d_functional" in str(node.target)
        for node in gm.graph.nodes
    ):
        return
    _warned_collectives = True
    logger.warning(
        "parallel precompile: a graph with a collective compiles in a child while LX "
        "context switching is on; the LX planner may keep an all-reduce input in LX, "
        "and the second run then fails with a NameError in the generated wrapper "
        "(torch-spyre#6563). Set ENABLE_LX_CONTEXT_SWITCHING=0 for this process until "
        "the planner is fixed."
    )


def _child_setup(workers: int = 1) -> None:
    import torch

    import torch_spyre._C as spyre_c
    import torch_spyre.ops.eager as spyre_eager
    from torch_spyre._inductor import config as spyre_config

    os.setsid()  # own process group, so a kill also stops dbo-opt
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    torch.set_num_threads(1)  # the parent's OpenMP pool does not survive fork()
    torch._inductor.config.compile_threads = 1  # the parent's worker pool is not ours
    spyre_config.num_cpus = _solver_cpu_share(workers)
    _count_time_limited_solves()
    for name in _DEVICE_ENTRY_POINTS:  # the device stays with the caller
        if hasattr(spyre_c, name):
            setattr(spyre_c, name, _no_device)
    spyre_eager.copy_tensor = _no_device


class _Pool:
    """At most ``workers`` children at once. A child is killed after ``timeout_s``.

    ``submitted``, ``ok`` and ``failed`` count the children; ``time_limited`` the ok
    ones whose LX planning stopped at its time limit (``_count_time_limited_solves``).
    """

    def __init__(
        self, workers: int, timeout_s: float, placeholders: bool = False
    ) -> None:
        self.workers = workers
        self.placeholders = placeholders  # the caller may switch it while open
        self.timeout_s = timeout_s
        self.open = True
        self.running: dict[int, float] = {}  # pid -> start time
        self.reports: dict[int, int] = {}  # pid -> read end of its report pipe
        self.ok = 0
        self.failed = 0
        self.submitted = 0
        self.time_limited = 0

    @staticmethod
    def _kill(pid: int) -> None:
        for kill in (os.killpg, os.kill):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                kill(pid, signal.SIGKILL)

    def _reap(self) -> bool:
        """Collect finished children and kill late ones. True if one finished."""
        finished = False
        for pid, started in list(self.running.items()):
            if time.time() - started > self.timeout_s:
                self._kill(pid)
            try:
                done, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                done, status = pid, -1
            if done:
                del self.running[pid]
                finished = True
                exited = status >= 0 and os.WIFEXITED(status)
                code = os.WEXITSTATUS(status) if exited else -1
                if code in (0, _TIME_LIMITED_EXIT):
                    self.ok += 1
                    self.time_limited += code == _TIME_LIMITED_EXIT
                else:
                    self.failed += 1
        return finished

    def submit(
        self, compile_fn: Callable[[], Any], report: bool = False, label: str = ""
    ) -> int:
        """Fork a child for ``compile_fn``; returns its pid.

        With ``report`` the child gets a pipe to send its graph's output specs
        through (see ``report``). ``label`` names the graph in the child's log lines.
        """
        while len(self.running) >= self.workers:
            if not self._reap():
                time.sleep(0.02)
        self.submitted += 1
        read_fd = write_fd = -1
        if report:
            read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child: compile, then exit without returning to the caller
            code = 1
            try:
                global _label
                _label = label
                if report:
                    global _report_fd
                    os.close(read_fd)
                    _report_fd = write_fd
                    _install_output_report()
                _child_setup(self.workers)
                compile_fn()
                code = _TIME_LIMITED_EXIT if _time_limited else 0
            except BaseException:  # noqa: BLE001 - the caller compiles it again
                logger.warning("parallel precompile: a compile failed", exc_info=True)
            finally:
                os._exit(code)
        self.running[pid] = time.time()
        if report:
            os.close(write_fd)
            self.reports[pid] = read_fd
        return pid

    def report(self, pid: int) -> Optional[list]:
        """Wait for the child's output specs; None if it exited without reporting.

        The writer is our own forked child, so its pickled specs are trusted.
        """
        fd = self.reports.pop(pid, None)
        if fd is None:
            return None
        chunks: list[bytes] = []
        try:
            while True:
                ready, _, _ = select.select([fd], [], [], 0.05)
                if ready:
                    chunk = os.read(fd, 1 << 16)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    continue
                self._reap()
                if pid in self.running:
                    continue
                if not select.select([fd], [], [], 0)[0]:
                    break  # the child is gone and left nothing to read
        finally:
            os.close(fd)
        if not chunks:
            logger.info("parallel precompile: child %d reported no output layouts", pid)
            return None
        return pickle.loads(b"".join(chunks))

    def wait(self) -> None:
        while self.running:
            if not self._reap():
                time.sleep(0.05)
        for fd in self.reports.values():
            os.close(fd)
        self.reports.clear()

    def kill_all(self) -> None:
        for pid in list(self.running):
            self._kill(pid)
        self.wait()


def _fx_cache_key(*args: Any, **kwargs: Any) -> Optional[str]:
    """The local FX graph cache key of this inner compile's graph, or None.

    Mirrors the key ``compile_fx_inner`` computes, after the rebinding of
    ``get_decomp_fn`` that ``_spyre_inner_compile`` applies. A key-less or
    uncacheable graph, or anything unexpected, gives None.
    """
    try:
        key_info = _fx_cache_key_info(*args, **kwargs)
        return key_info[0] if key_info else None
    except Exception:
        return None


def _fx_cache_key_info(*args: Any, **kwargs: Any) -> Optional[tuple]:
    """``(key, debug lines)`` as ``_fx_cache_key`` computes them, or None for a
    key-less or uncacheable graph. Raises on anything unexpected."""
    from torch._inductor import config
    from torch._inductor.codecache import FxGraphCache
    from torch._inductor.compile_fx import get_input_idxs_to_check
    from torch.utils._python_dispatch import _disable_current_modes
    from torch_spyre._inductor.decompositions import get_spyre_decomp_table

    if config.force_disable_caches or not config.fx_graph_cache:
        return None
    gm = kwargs["gm"] if "gm" in kwargs else args[0]
    example_inputs = kwargs["example_inputs"] if "example_inputs" in kwargs else args[1]
    graph_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("gm", "example_inputs", "compile_region_name")
    }
    graph_kwargs["get_decomp_fn"] = get_spyre_decomp_table
    # The defaults compile_fx_inner fills in; they are part of the key.
    for name, default in (
        ("cudagraphs", None),
        ("static_input_idxs", ()),
        ("is_backward", False),
        ("graph_id", None),
        ("cpp_wrapper", False),
        ("fx_wrapper", False),
        ("is_inference", False),
        ("boxed_forward_device_index", None),
        ("layout_opt", None),
        ("extern_node_serializer", None),
    ):
        graph_kwargs.setdefault(name, default)
    inputs_to_check = get_input_idxs_to_check(
        example_inputs, graph_kwargs["static_input_idxs"]
    )
    # compile_fx_inner takes its key with the dispatch modes (the fake mode)
    # turned off.
    with _disable_current_modes():
        key_info, _ = FxGraphCache.prepare_key(
            gm, example_inputs, graph_kwargs, inputs_to_check, False
        )
    return key_info


def _fx_cache_holds(*args: Any, **kwargs: Any) -> bool:
    """Whether the local FX graph cache already holds this inner compile's graph
    (see ``_fx_cache_key``); anything unexpected counts as a miss."""
    return _fx_cache_has(_fx_cache_key(*args, **kwargs))


def _fx_cache_has(key: Optional[str]) -> bool:
    if key is None:
        return False
    try:
        from torch._inductor.codecache import FxGraphCache

        key_dir = FxGraphCache._get_tmp_dir_for_key(key)
        return os.path.isdir(key_dir) and any(True for _ in os.scandir(key_dir))
    except Exception:
        return False


AOT_REUSE_ENV = "TORCH_SPYRE_AOT_REUSE_FX"

# The reuse rests on internals of these torch releases: which flags
# ``compile_fx_inner`` reads, the ``precompile_enabled`` field of the FX key, and
# why torch keeps the two caches apart (pytorch#156611). On any other release the
# switch is ignored and ``aot_compile`` compiles as without it. A torch upgrade
# re-checks ``_pool_graph_state``'s argument and moves this tuple, as it does
# ``loop_scheduler.SUPPORTED_RELEASES``.
AOT_REUSE_RELEASES: tuple[tuple[int, int], ...] = ((2, 14),)

# Process-wide state, like the swap of ``_spyre_inner_compile`` in
# ``fork_inner_compiles``: compiles run on one thread at a time (Dynamo's compile
# lock), and neither of the two dicts below is ever cleared.
#
# ``_pool_fx_keys`` holds the FX graph cache keys of the graphs a pool of this
# process compiled in a child (or found in the FX graph cache), kept while
# ``TORCH_SPYRE_AOT_REUSE_FX=1``, with the key's debug lines, so a lookup that
# misses can name the fields that differ from the pool's keys. It grows by one
# entry per graph key a pool sees.
_pool_fx_keys: dict = {}
# ``_aot_reuse_counts`` counts the outcomes of ``compile_reusing_pool_graph``.
_aot_reuse_counts = {"reused": 0, "not seen": 0, "miss": 0}
_aot_reuse_release_warned = False


def _aot_reuse_on() -> bool:
    global _aot_reuse_release_warned
    if os.environ.get(AOT_REUSE_ENV, "0") != "1":
        return False
    release = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    if release in AOT_REUSE_RELEASES:
        return True
    if not _aot_reuse_release_warned:
        _aot_reuse_release_warned = True
        logger.warning(
            "aot_compile reuse: %s=1 ignored, torch %s is not a release it was "
            "checked on (%s)",
            AOT_REUSE_ENV,
            torch.__version__,
            ", ".join(f"{major}.{minor}" for major, minor in AOT_REUSE_RELEASES),
        )
    return False


def _pool_graph_state(*args: Any, **kwargs: Any) -> tuple:
    """Whether ``aot_compile`` may take this graph from the FX graph cache.

    ``aot_compile`` sets functorch's ``bundled_autograd_cache``, and then
    ``compile_fx_inner`` neither reads nor writes the FX graph cache, so every graph
    a pool child already compiled is compiled again here. The flag is also a field
    of the FX key (``precompile_enabled``), so the key differs from the child's.

    The key computed as the plain path computes it (flag off) is the child's key:
    the pool's own second pass finds the children's entries with it. Of what a child
    changes, only the thread count can enter the key, and only for a graph with CPU
    C++ code, whose kernels are built for that count; such a graph is not reused.

    Returns ``(state, detail)``. ``state`` is None when this does not apply (switch
    off, not inside ``aot_compile``), ``"found"`` when the FX graph cache holds the
    key, ``"not seen"`` when there is no key (a backward graph, or an error) and
    ``"missing"`` when the cache holds no entry for it; ``detail`` says why, for the
    log. The entry may come from a pool child or from a plain compile of an earlier
    start (an upgraded start whose saved ``aot_compile`` files no longer match, with
    no pool to fork). On ``"found"`` the
    caller runs the plain ``compile_fx_inner`` with the flag off, which loads that
    entry: the ``CompiledFxGraph`` a plain compile of this graph returns. The key
    covers the graph, the inputs (with the Spyre layouts) and the Inductor config,
    and the other functorch flags ``aot_compile`` sets are read by AOTAutograd only,
    before the inner compile. The entry holds no trace of the flag: torch adds it to
    the key so that the FX graph cache and the bundled AOTAutograd cache do not mix
    their entries (pytorch#156611). Forward graphs only.
    """
    # ``config.patch`` below is process-wide too: safe under the same one-thread
    # rule, and it is undone before the call returns.
    if not (_aot_reuse_on() and torch._functorch.config.bundled_autograd_cache):
        return None, ""
    if kwargs.get("is_backward", False):
        return "not seen", "a backward graph"
    try:
        with torch._functorch.config.patch(bundled_autograd_cache=False):
            key_info = _fx_cache_key_info(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - any error means no reuse
        return "not seen", f"no FX key: {type(e).__name__}: {e}"
    if not key_info:
        return "not seen", "no FX key (cache off or graph not cacheable)"
    key, lines = key_info
    if not _fx_cache_has(key):
        if key in _pool_fx_keys:
            return "missing", f"key {key}, a pool key, no entry on disk"
        return "missing", f"key {key}, no entry on disk; {_closest_pool_key(lines)}"
    source = "a pool key" if key in _pool_fx_keys else "not a pool key"
    return "found", f"key {key}, {source}"


def _closest_pool_key(lines: list) -> str:
    """Which key fields differ from the pool key closest to ``lines``."""
    if not _pool_fx_keys:
        return "the pool noted no key"

    def hashed(key_lines: list) -> set:
        # "[hash] field: text": compare the hash, not the text, which for an object
        # without a repr (a custom pass in the Inductor config) holds its address.
        return {line.split(": ", 1)[0] for line in key_lines}

    def fields(diff: set) -> list:
        return sorted({line.split("] ", 1)[-1] for line in diff})

    ours = hashed(lines)
    key, theirs = min(
        ((k, hashed(v)) for k, v in _pool_fx_keys.items()),
        key=lambda kv: len(ours ^ kv[1]),
    )
    return f"closest pool key {key} differs in: {', '.join(fields(ours ^ theirs))}"


def compile_reusing_pool_graph(compile_inner: Callable, *args: Any, **kwargs: Any):
    """``compile_inner(*args, **kwargs)``, but under ``aot_compile`` with
    ``TORCH_SPYRE_AOT_REUSE_FX=1`` a graph the FX graph cache holds (left by a pool
    child or a plain compile) is loaded from there (see ``_pool_graph_state``). Logs
    one line per such compile: a reuse at INFO, a compile as a warning.

    If torch's lookup then misses (a corrupt entry, or guards that fail under
    dynamic shapes), that call has already compiled the graph with
    ``bundled_autograd_cache`` off: the plain path, which writes an FX entry. The
    result is the same graph, compiled as without the switch."""
    state, detail = _pool_graph_state(*args, **kwargs)
    if state is None:
        return compile_inner(*args, **kwargs)
    if state != "found":
        outcome = "not seen" if state == "not seen" else "miss"
        out = compile_inner(*args, **kwargs)
    else:
        from torch._dynamo.utils import counters

        hits = counters["inductor"]["fxgraph_cache_hit"]
        with torch._functorch.config.patch(bundled_autograd_cache=False):
            out = compile_inner(*args, **kwargs)
        if counters["inductor"]["fxgraph_cache_hit"] > hits:
            outcome = "reused"
        else:
            outcome, detail = "miss", detail + ", the FX lookup missed"
    _aot_reuse_counts[outcome] += 1
    message = {
        "reused": "reused FX entry",
        "not seen": "no FX key, compiled",
        "miss": "FX miss, compiled",
    }[outcome]
    (logger.info if outcome == "reused" else logger.warning)(
        "aot_compile reuse: %s (%d reused, %d not seen, %d miss so far): %s",
        message,
        _aot_reuse_counts["reused"],
        _aot_reuse_counts["not seen"],
        _aot_reuse_counts["miss"],
        detail,
    )
    return out


@contextlib.contextmanager
def fork_inner_compiles(
    workers: int, timeout_s: float = 300.0, placeholders: bool = False
):
    """Run each Spyre inner compile in a forked child, at most ``workers`` at once.

    Yields the pool. Leaving the block waits for every child, or kills them all if
    the block raised. With ``placeholders`` (``pool.placeholders``, which the caller
    may switch while the block runs) the stand-ins return zero tensors in
    the layouts the children report instead of raising, so a traced forward runs
    to its end: a forward that branches on tensor values traces other graphs than
    the real run (a recompile in the second run, nothing wrong); see the module
    docstring for the contract and the serialization on each child's frontend.
    """
    import torch_spyre._inductor as spyre_inductor

    original = spyre_inductor._spyre_inner_compile
    pool = _Pool(max(1, workers), timeout_s, placeholders)

    def inner_compile(*args: Any, **kwargs: Any) -> Any:
        if _aot_reuse_on():
            # Key the graph once: note the key, then look it up.
            try:
                key_info = _fx_cache_key_info(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - only the reuse is lost
                key_info = None
                logger.warning("aot_compile reuse: the pool cannot key a graph: %r", e)
            if key_info:
                _pool_fx_keys[key_info[0]] = key_info[1]
            held = _fx_cache_has(key_info[0] if key_info else None)
        else:
            held = _fx_cache_holds(*args, **kwargs)
        if held:
            # A graph AOTAutograd would not cache (it bypasses graphs that call
            # torch-spyre's Tensor.to) reaches the inner compile on every start, and
            # the FX graph cache answers it in place faster than a fork.
            return original(*args, **kwargs)
        gm = kwargs.get("gm", args[0] if args else None)
        _warn_collectives_in_lx(gm)
        label = _graph_label(gm, kwargs)
        if not pool.placeholders:
            pool.submit(lambda: original(*args, **kwargs), label=label)
            return _StandIn(pool)
        outputs = _output_metadata(gm)
        pid = pool.submit(lambda: original(*args, **kwargs), report=True, label=label)
        return _StandIn(pool, outputs, pid)

    started = time.time()
    spyre_inductor._spyre_inner_compile = inner_compile
    try:
        yield pool
    except BaseException:
        pool.kill_all()
        raise
    finally:
        spyre_inductor._spyre_inner_compile = original
        pool.wait()
        pool.open = False
    if _aot_reuse_on():
        logger.info("aot_compile reuse: the pool noted %d FX keys", len(_pool_fx_keys))
    logger.info(
        "parallel precompile: %d compiles ok (%d with an LX planning solve at its time "
        "limit), %d failed, %d workers, %.1f s",
        pool.ok,
        pool.time_limited,
        pool.failed,
        pool.workers,
        time.time() - started,
    )


class SecondPassCheck:
    """Count FX graph cache hits and misses of the run after the precompile.

    Use it around the second run of the graphs, as a context manager or with
    ``finish()``. ``expected`` is the number of compiles the children finished
    (``pool.ok``). Fewer hits than that (or, without ``expected``, any miss) logs a
    warning with the numbers, because the children's work was then lost.
    """

    def __init__(self, expected: Optional[int] = None) -> None:
        from torch._dynamo.utils import counters

        self._counters = counters["inductor"]
        self._start = (
            self._counters["fxgraph_cache_hit"],
            self._counters["fxgraph_cache_miss"],
        )
        self.expected = expected
        self.hits = self.misses = 0

    def finish(self) -> tuple[int, int]:
        self.hits = self._counters["fxgraph_cache_hit"] - self._start[0]
        self.misses = self._counters["fxgraph_cache_miss"] - self._start[1]
        if self.expected is None:
            lost = self.misses > 0
        else:  # misses beyond that are graphs the children never compiled
            lost = self.hits < self.expected
        log = logger.warning if lost else logger.info
        log(
            "parallel precompile: the second run hit the FX graph cache %d times "
            "(children finished %s) and compiled %d graphs%s",
            self.hits,
            self.expected,
            self.misses,
            ": the children's work was lost" if lost else "",
        )
        return self.hits, self.misses

    def __enter__(self) -> "SecondPassCheck":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()

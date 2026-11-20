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

What a child may use: the compile itself (Python, the Inductor and Spyre lowering
code, CP-SAT, and dbo-opt as a subprocess). What it may not use: the device. Every
Python entry point of the Spyre runtime (start, allocation, copies, fills, streams,
synchronize, kernel preparation) raises in a child, and so does the eager copy op.
The child runs on one thread with ``compile_threads = 1``, so it starts no worker
pool of its own (#4427's pool stays in the parent), and it never takes a lock that
another thread of the parent could hold at fork time: such a lock stays held in
the child, so a compile that needed it would hang and be killed by the timeout.
Its LX planning solver gets the rank's share of the CPUs (the CPU count divided by
the ranks on the host, at least two; the whole host when there is one rank): sized
to the whole host on every rank, as each parent's is, the concurrent solvers starve
each other, and a starved solve returns a worse plan for the same graph (slower
kernels, not just a slower compile).

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


def _solver_cpu_share(workers: int) -> int:
    """CPUs for one child's LX planning solver: the host's share per rank.

    Every rank on the host forks ``workers`` children at once, and each child's
    CP-SAT solve would otherwise take every CPU of the host, as the parent's
    does. Starved solves return worse plans, not only later ones. The share is
    per rank, not per child: on one rank the children keep the whole host, which
    measured faster than a per-child share, while at four ranks on 64 CPUs the
    per-rank share took the block warm-up from 338 s to 213 s.
    """
    from torch_spyre._inductor import config as spyre_config

    ranks = 1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            ranks = dist.get_world_size()
    except Exception:  # noqa: BLE001 - no process group means one rank
        ranks = 1
    del workers  # the share is per rank; the workers stay in the signature for callers
    return max(2, spyre_config.get_cpu_count() // max(1, ranks))


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
    for name in _DEVICE_ENTRY_POINTS:  # the device stays with the caller
        if hasattr(spyre_c, name):
            setattr(spyre_c, name, _no_device)
    spyre_eager.copy_tensor = _no_device


class _Pool:
    """At most ``workers`` children at once. A child is killed after ``timeout_s``."""

    def __init__(self, workers: int, timeout_s: float) -> None:
        self.workers = workers
        self.timeout_s = timeout_s
        self.open = True
        self.running: dict[int, float] = {}  # pid -> start time
        self.reports: dict[int, int] = {}  # pid -> read end of its report pipe
        self.ok = 0
        self.failed = 0
        self.submitted = 0

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
                if status == 0:
                    self.ok += 1
                else:
                    self.failed += 1
        return finished

    def submit(self, compile_fn: Callable[[], Any], report: bool = False) -> int:
        """Fork a child for ``compile_fn``; returns its pid.

        With ``report`` the child gets a pipe to send its graph's output specs
        through (see ``report``).
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
                if report:
                    global _report_fd
                    os.close(read_fd)
                    _report_fd = write_fd
                    _install_output_report()
                _child_setup(self.workers)
                compile_fn()
                code = 0
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
        """Wait for the child's output specs; None if it exited without reporting."""
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


def _fx_cache_holds(*args: Any, **kwargs: Any) -> bool:
    """Whether the local FX graph cache already holds this inner compile's graph.

    Mirrors the key ``compile_fx_inner`` computes, after the rebinding of
    ``get_decomp_fn`` that ``_spyre_inner_compile`` applies. A key-less or
    uncacheable graph, or anything unexpected, counts as a miss.
    """
    try:
        from torch._inductor import config
        from torch._inductor.codecache import FxGraphCache
        from torch._inductor.compile_fx import get_input_idxs_to_check
        from torch_spyre._inductor.decompositions import get_spyre_decomp_table

        if config.force_disable_caches or not config.fx_graph_cache:
            return False
        gm = kwargs["gm"] if "gm" in kwargs else args[0]
        example_inputs = (
            kwargs["example_inputs"] if "example_inputs" in kwargs else args[1]
        )
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
        key_info, _ = FxGraphCache.prepare_key(
            gm, example_inputs, graph_kwargs, inputs_to_check, False
        )
        if key_info is None:
            return False
        key_dir = FxGraphCache._get_tmp_dir_for_key(key_info[0])
        return os.path.isdir(key_dir) and any(True for _ in os.scandir(key_dir))
    except Exception:
        return False


@contextlib.contextmanager
def fork_inner_compiles(
    workers: int, timeout_s: float = 300.0, placeholders: bool = False
):
    """Run each Spyre inner compile in a forked child, at most ``workers`` at once.

    Yields the pool. Leaving the block waits for every child, or kills them all if
    the block raised. With ``placeholders`` the stand-ins return zero tensors in
    the layouts the children report instead of raising, so a traced forward runs
    to its end.
    """
    import torch_spyre._inductor as spyre_inductor

    original = spyre_inductor._spyre_inner_compile
    pool = _Pool(max(1, workers), timeout_s)

    def inner_compile(*args: Any, **kwargs: Any) -> Any:
        if _fx_cache_holds(*args, **kwargs):
            # A graph AOTAutograd would not cache (it bypasses graphs that call
            # torch-spyre's Tensor.to) reaches the inner compile on every start, and
            # the FX graph cache answers it in place faster than a fork.
            return original(*args, **kwargs)
        if not placeholders:
            pool.submit(lambda: original(*args, **kwargs))
            return _StandIn(pool)
        gm = kwargs.get("gm", args[0] if args else None)
        outputs = _output_metadata(gm)
        pid = pool.submit(lambda: original(*args, **kwargs), report=True)
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
    logger.info(
        "parallel precompile: %d compiles ok, %d failed, %d workers, %.1f s",
        pool.ok,
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

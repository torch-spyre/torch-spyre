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

"""Stage 4 (optional, needs a Spyre device): compile and run on the device.

The static analysis cannot see layout or stick-alignment failures, numerics,
or graph-level passes. A probe compiles each unique signature on the device,
runs it on random inputs, and compares with the CPU. ``e2e`` does the same for
the whole model.

Probes run in a worker process, one case after another. A case that kills the
process (SIGABRT, SIGFPE) or hangs past the timeout is recorded as ``crash``
or ``timeout`` and the worker restarts at the next case.

The worker is ``main()`` below, started with ``python -c`` so that ``torch``
is imported before ``torch_spyre``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Optional

from .model import CoverageReport, ProbeResult

# fp16 against an fp32-accumulated CPU reference; the repo's compare_with_cpu
# default.
ATOL = RTOL = 0.1

Progress = Callable[[int, int, str], None]


# -- parent side --------------------------------------------------------------


def run_probes(
    report: CoverageReport,
    *,
    device: str = "spyre",
    timeout: float = 300.0,
    limit: Optional[int] = None,
    progress: Optional[Progress] = None,
) -> None:
    """Probe every signature in ``report`` and store the results on it."""
    cases: list[dict[str, Any]] = []
    owners: list[list[Any]] = []
    index: dict[str, int] = {}
    for record in report.ops:
        for sig in record.signatures:
            if sig.replay is None:
                continue
            key = json.dumps(sig.replay, sort_keys=True)
            if key not in index:
                if limit is not None and len(cases) >= limit:
                    continue
                index[key] = len(cases)
                cases.append({"kind": "op", "replay": sig.replay})
                owners.append([])
            owners[index[key]].append(sig)
    results = run_cases(cases, device=device, timeout=timeout, progress=progress)
    for i, sigs in enumerate(owners):
        for sig in sigs:
            sig.probe = results.get(i, ProbeResult("skipped", "not run"))


def run_e2e(
    recipe: str,
    target: str,
    options: dict[str, str],
    *,
    device: str = "spyre",
    timeout: float = 3600.0,
) -> ProbeResult:
    """Compile and run the whole model on the device, compared with the CPU."""
    case = {"kind": "e2e", "recipe": recipe, "target": target, "options": options}
    return run_cases([case], device=device, timeout=timeout)[0]


def run_cases(
    cases: list[dict[str, Any]],
    *,
    device: str,
    timeout: float,
    progress: Optional[Progress] = None,
) -> dict[int, ProbeResult]:
    results: dict[int, ProbeResult] = {}
    if not cases:
        return results
    with tempfile.TemporaryDirectory(prefix="spyre_probe_") as tmp:
        cases_path = pathlib.Path(tmp) / "cases.json"
        cases_path.write_text(json.dumps(cases))
        attempt = 0
        while len(results) < len(cases):
            pending = [i for i in range(len(cases)) if i not in results]
            attempt += 1
            out_path = pathlib.Path(tmp) / f"results-{attempt}.jsonl"
            todo_path = pathlib.Path(tmp) / f"todo-{attempt}.json"
            todo_path.write_text(json.dumps(pending))
            _run_worker(
                cases_path,
                todo_path,
                out_path,
                device,
                timeout,
                results,
                len(cases),
                progress,
            )
    return results


def _run_worker(
    cases_path: pathlib.Path,
    todo_path: pathlib.Path,
    out_path: pathlib.Path,
    device: str,
    timeout: float,
    results: dict[int, ProbeResult],
    total: int,
    progress: Optional[Progress],
) -> None:
    out_path.touch()
    # The package is not installed; put its parent directory on the path.
    package_parent = str(pathlib.Path(__file__).resolve().parents[1])
    cmd = [
        sys.executable,
        "-c",
        f"import sys; sys.path.insert(0, {package_parent!r}); import torch; "
        "from spyre_op_coverage.probe import main; main()",
        "--worker",
        str(cases_path),
        str(out_path),
        "--todo",
        str(todo_path),
        "--device",
        device,
    ]
    env = dict(os.environ)
    if not _uses_spyre(device):
        # Probing on another device must not start the Spyre runtime: the card
        # may be held by the calling process (one process per card).
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    log = open(out_path.with_suffix(".log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    current: Optional[int] = None
    any_started = False
    started = time.monotonic()
    offset = 0
    try:
        while True:
            with open(out_path) as f:
                f.seek(offset)
                chunk = f.read()
            complete = chunk[: chunk.rfind("\n") + 1]
            offset += len(complete)
            for line in complete.splitlines():
                event = json.loads(line)
                i = event["i"]
                if event.get("event") == "start":
                    current, started, any_started = i, time.monotonic(), True
                    if progress:
                        progress(len(results), total, f"case {i}")
                    continue
                results[i] = ProbeResult(
                    event["outcome"],
                    event.get("detail", ""),
                    event.get("max_abs_diff"),
                    event.get("fallbacks", []),
                )
                current = None
            if proc.poll() is not None:
                log.flush()
                last = _last_error(out_path.with_suffix(".log"))
                if current is not None and current not in results:
                    results[current] = ProbeResult(
                        "crash", f"worker exited with {proc.returncode}: {last}"
                    )
                elif not any_started:
                    # Died before starting any case: give up on the rest.
                    for i in json.loads(todo_path.read_text()):
                        results.setdefault(
                            i, ProbeResult("crash", f"worker failed: {last}")
                        )
                return
            if current is not None and time.monotonic() - started > timeout:
                proc.kill()
                proc.wait()
                results[current] = ProbeResult("timeout", f"over {timeout:.0f}s")
                return
            time.sleep(0.2)
    finally:
        if proc.poll() is None:
            proc.kill()
        log.close()


def _last_error(log_path: pathlib.Path) -> str:
    """The last exception line of a worker log, or its last line."""
    lines = [line.strip() for line in log_path.read_text(errors="replace").splitlines()]
    lines = [line for line in lines if line]
    for line in reversed(lines):
        if "Error" in line.split(":", 1)[0] or "Exception" in line.split(":", 1)[0]:
            return line[:300]
    return lines[-1][:300] if lines else "no output"


# -- worker side --------------------------------------------------------------


def _materialize(value: Any, device: str) -> Any:
    import torch
    from torch.utils import _pytree as pytree

    from .capture import TensorMeta

    # A local generator: torch.manual_seed would also seed (and so start) the
    # Spyre runtime.
    gen = torch.Generator().manual_seed(0)

    def make(x: Any) -> Any:
        if isinstance(x, TensorMeta):
            if x.dtype.is_floating_point:
                t = torch.randn(x.shape, generator=gen).to(x.dtype)
            elif x.dtype == torch.bool:
                t = torch.rand(x.shape, generator=gen) < 0.5
            else:
                # Small values stay valid as indices for gather/embedding/...
                t = torch.randint(0, 2, x.shape, generator=gen, dtype=x.dtype)
            return t.to(device)
        if isinstance(x, str) and x.split(":")[0] in ("cpu", "cuda"):
            return device
        return x

    return pytree.tree_map(make, value)


def _compare(expected: Any, actual: Any) -> tuple[bool, Optional[float], str]:
    import torch
    from torch.utils import _pytree as pytree

    exp = [t for t in pytree.tree_leaves(expected) if isinstance(t, torch.Tensor)]
    act = [t for t in pytree.tree_leaves(actual) if isinstance(t, torch.Tensor)]
    if len(exp) != len(act):
        return False, None, f"{len(act)} outputs, expected {len(exp)}"
    worst = 0.0
    for e, a in zip(exp, act):
        a = a.detach().to("cpu")
        if e.shape != a.shape:
            return False, None, f"shape {tuple(a.shape)}, expected {tuple(e.shape)}"
        if not e.is_floating_point():
            if not torch.equal(e, a.to(e.dtype)):
                return False, None, "integer outputs differ"
            continue
        e32, a32 = e.float(), a.float()
        diff = (e32 - a32).abs()
        finite = torch.isfinite(e32)
        if diff.numel():
            worst = max(worst, float(diff[finite].max()) if finite.any() else 0.0)
        if not torch.allclose(a32, e32, atol=ATOL, rtol=RTOL, equal_nan=True):
            return False, worst, f"max abs diff {worst:.3g}"
    return True, worst, ""


def _classify(exc: BaseException) -> str:
    names = {type(e).__name__ for e in _chain(exc)}
    compile_errors = {
        "BackendCompilerFailed",
        "InductorError",
        "Unsupported",
        "LoweringException",
    }
    return "compile_error" if names & compile_errors else "runtime_error"


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]


def _probe_callable(case: dict[str, Any], device: str):
    """``(fn, cpu_args, cpu_kwargs)`` for a case."""
    from .serialize import call_target, decode

    if case["kind"] == "op":
        replay = case["replay"]
        target = call_target(replay["kind"], replay["target"])
        args = _materialize(decode(replay["args"]), "cpu")
        kwargs = _materialize(decode(replay["kwargs"]), "cpu")

        def fn(*a: Any, **k: Any) -> Any:
            if isinstance(target, str):
                return getattr(a[0], target)(*a[1:], **k)
            return target(*a, **k)

        return fn, args, kwargs
    from .recipes import load

    spec = load(case["recipe"], case["target"], case["options"])
    return spec.model, spec.args, spec.kwargs


def probe_case(case: dict[str, Any], device: str) -> dict[str, Any]:
    import copy
    import warnings

    import torch
    from torch.utils import _pytree as pytree

    from .trace import short_error

    try:
        fn, args, kwargs = _probe_callable(case, device)
    except Exception as exc:  # noqa: BLE001 - reported
        return {"outcome": "skipped", "detail": "cannot rebuild: " + short_error(exc)}
    cpu_args = copy.deepcopy(args)
    try:
        expected = fn(*cpu_args, **copy.deepcopy(kwargs))
    except Exception as exc:  # noqa: BLE001 - reported
        return {"outcome": "skipped", "detail": "fails on CPU: " + short_error(exc)}
    if expected is None and cpu_args:
        # An in-place op such as setitem: compare the mutated input.
        expected = cpu_args[0]

    def to_device(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, str) and x == "cpu":
            return device
        return x

    if isinstance(fn, torch.nn.Module):
        fn = copy.deepcopy(fn).to(device)
    dev_args = pytree.tree_map(to_device, args)
    dev_kwargs = pytree.tree_map(to_device, kwargs)
    torch._dynamo.reset()
    compiled = torch.compile(fn, dynamic=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            actual = compiled(*dev_args, **dev_kwargs)
            if actual is None and dev_args:
                actual = dev_args[0]
        except Exception as exc:  # noqa: BLE001 - reported
            return {"outcome": _classify(exc), "detail": short_error(exc)}
    fallbacks = sorted(
        {
            str(w.message)
            for w in caught
            if "Fallback" in getattr(w.category, "__name__", "")
        }
    )
    ok, diff, detail = _compare(expected, actual)
    return {
        "outcome": "pass" if ok else "mismatch",
        "detail": detail,
        "max_abs_diff": diff,
        "fallbacks": fallbacks,
    }


def _uses_spyre(device: str) -> bool:
    return device.split(":")[0] == "spyre"


def _worker(cases_path: str, out_path: str, todo_path: str, device: str) -> None:
    if _uses_spyre(device):
        import torch_spyre

        torch_spyre._autoload()  # type: ignore[attr-defined]  # not in the .pyi
    cases = json.loads(pathlib.Path(cases_path).read_text())
    todo = json.loads(pathlib.Path(todo_path).read_text())
    with open(out_path, "a") as out:
        for i in todo:
            out.write(json.dumps({"i": i, "event": "start"}) + "\n")
            out.flush()
            os.fsync(out.fileno())
            result = probe_case(cases[i], device)
            out.write(json.dumps({"i": i, **result}) + "\n")
            out.flush()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker", nargs=2, metavar=("CASES", "RESULTS"), required=True
    )
    parser.add_argument("--todo", required=True)
    parser.add_argument("--device", default="spyre")
    ns = parser.parse_args(argv)
    _worker(ns.worker[0], ns.worker[1], ns.todo, ns.device)

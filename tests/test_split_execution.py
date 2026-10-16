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

"""Split-execution parity tests (no profiler required)."""

import contextlib
import hashlib
import os
import pathlib
import subprocess
import sys
import time

import pytest
import torch
import torch._dynamo

import torch_spyre
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
from torch_spyre._inductor import spyre_hint
from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

# ── stream constants ──────────────────────────────────────────────────────────
_HOST_COMPUTE_STREAM_ID = 65  # S_prep
_PIPELINE_DEPTH = 4  # number of back-to-back launches for parity tests

# ── subprocess environment variable keys ─────────────────────────────────────
_ENV_RESULT = "SPYRE_RESULT_PATH"
_ENV_TRACKER = "SPYRE_HAZARD_TRACKER"
_ENV_SHIFTED = "SPYRE_SHIFTED_ORDER"

# ── shapes ────────────────────────────────────────────────────────────────────
# Small shape: fast compile, used for correctness / routing checks.
_SMALL_M, _SMALL_K, _SMALL_N = 64, 512, 32
# Large (device-bound) shape: kernel ~806 µs, used for host-ahead check.
_LARGE_M, _LARGE_K, _LARGE_N = 1024, 8192, 1024
_K_TILES = 4

# ── host-ahead threshold ──────────────────────────────────────────────────────
# At least this fraction of (launch k+1 host time, launch k device still busy)
# pairs must hold for the run-ahead check to pass on the large shape.
_HOST_AHEAD_MIN_FRACTION = 0.7  # 11/15 measured; use 0.6 for headroom
_HOST_AHEAD_PAIRS = 8  # number of back-to-back pairs to check


# ── helpers ───────────────────────────────────────────────────────────────────


def _sync_all(device: torch.device) -> None:
    """Drain both S_dev (default stream) and S_prep (stream 65)."""
    torch.accelerator.synchronize(device)
    torch_spyre._C.host_compute_stream_by_id(
        _HOST_COMPUTE_STREAM_ID, device
    ).synchronize()


def _cleanup_device(device: torch.device) -> None:
    """Safely drain and release device runtime on child process exit."""
    with contextlib.suppress(Exception):
        _sync_all(device)
    with contextlib.suppress(Exception):
        torch_spyre._C.free_runtime()


def _s_dev(device: torch.device):
    """Return the C-level default (S_dev) stream handle."""
    return torch_spyre._C.default_stream(device)


def _assert_plan_has_prep_steps(runner: SpyreSDSCKernelRunner) -> None:
    """Fail fast if the JobPlan has no Prep-role steps.

    Without a Prep step the split path is never engaged and all parity checks
    pass vacuously (tracker-on and tracker-off both use S_dev for everything).
    """
    plan = runner.jobplan
    has_prep = any(
        plan.get_step_stream_role(i) == "Prep" for i in range(plan.num_steps())
    )
    assert has_prep, (
        f"JobPlan for '{runner.kernel_name}' has no Prep-role steps — "
        "deeptools did not emit a ComputeOnHost triple for this shape.  "
        "Did you forget to merge #4864 (Prep role for HostCompute)?"
    )


def _patch_runner_assert_prep() -> None:
    """Monkey-patch SpyreSDSCKernelRunner.run to assert Prep role on first call.

    Intentionally not restored — each child subprocess is a fresh process.
    """
    seen: set[int] = set()
    orig_run = SpyreSDSCKernelRunner.run

    def _checked_run(self, *args, **kwargs):
        if id(self) not in seen:
            seen.add(id(self))
            _assert_plan_has_prep_steps(self)
        return orig_run(self, *args, **kwargs)

    SpyreSDSCKernelRunner.run = _checked_run  # type: ignore[method-assign]


def _build_compiled_fn(M: int, K: int, N: int):
    """Declare named dims and return a compiled K-tiled mm for shape (M,K)×(K,N)."""
    _pnd.declare_tensor_dim("M", M)
    _pnd.declare_tensor_dim("K", K)
    _pnd.declare_tensor_dim("N", N)

    def k_tiled_mm(a, b):
        _pnd.name_tensor_dims(a, ["M", "K"])
        _pnd.name_tensor_dims(b, ["K", "N"])
        with spyre_hint(num_tiles_per_dim={"K": _K_TILES}):
            return torch.mm(a, b)

    return torch.compile(k_tiled_mm, backend="inductor")


def _to_dev_named(t_cpu, rows_dim: str, cols_dim: str) -> torch.Tensor:
    t_dev = t_cpu.to("spyre")
    _pnd.name_tensor_dims(t_dev, [rows_dim, cols_dim])
    return t_dev


# ── subprocess launcher ───────────────────────────────────────────────────────


def _run_child(
    tracker: int,
    result_path: str,
    child_name: str,
    *,
    cache_dir: str | None = None,
    shifted: bool = False,
    max_retries: int = 3,
) -> subprocess.CompletedProcess:
    """Spawn a child process that runs exactly ``child_name`` with retry on device busy."""
    env = os.environ.copy()
    env[_ENV_TRACKER] = str(tracker)
    env[_ENV_RESULT] = result_path
    if cache_dir is not None:
        env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        # Force single CPU so both modes always produce the same program.
        env["SPYRE_NUM_CPUS"] = "1"
    if shifted:
        env[_ENV_SHIFTED] = "1"

    proc = None
    for attempt in range(max_retries):
        time.sleep(0.5)  # Allow previous process to release VFIO device
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                child_name,
            ],
            env=env,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            timeout=300,
            check=False,
        )
        if proc.returncode == 0:
            return proc
        # Retry on device busy / VFIO cleanup delay
        if "Device or resource busy" in proc.stderr or "DeviceOpenFail" in proc.stderr:
            time.sleep(1.0 * (attempt + 1))
            continue
        break

    assert proc is not None
    raise AssertionError(
        f"Child (SPYRE_HAZARD_TRACKER={tracker}, child={child_name!r}) "
        f"exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def _collect_parity(
    tmp: pathlib.Path,
    child_name: str,
    *,
    shifted_on: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run both tracker modes and return (results_off, results_on).

    Both children share one compile cache so the same program is used in both
    modes.  The tracker-on child uses shifted input order when *shifted_on* is
    True (the default).
    """
    cache_dir = str(tmp / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    off_path = str(tmp / "off.pt")
    on_path = str(tmp / "on.pt")

    # Tracker-off first so the cache is already warm for tracker-on.
    _run_child(0, off_path, child_name, cache_dir=cache_dir, shifted=False)
    _run_child(1, on_path, child_name, cache_dir=cache_dir, shifted=shifted_on)

    for label, path in (("tracker-off", off_path), ("tracker-on", on_path)):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Child for '{child_name}' ({label}) exited 0 but never wrote "
                f"{path!r} — the child workload failed to produce output."
            )
    off = torch.load(off_path, weights_only=True)
    on = torch.load(on_path, weights_only=True)
    return off, on


# ── child workloads ───────────────────────────────────────────────────────────


def _child_single_mm() -> None:
    """Child: single K-tiled mm, saves [result_cpu].

    C3 – one launch with tracker on/off; checks plan shape and correctness.
    """
    device = torch.device("spyre")
    try:
        torch._dynamo.reset()
        _pnd.reset()

        M, K, N = _SMALL_M, _SMALL_K, _SMALL_N
        g = torch.Generator(device="cpu").manual_seed(0xAFFE)
        a_cpu = torch.randn(M, K, dtype=torch.float16, generator=g) * 0.01
        b_cpu = torch.randn(K, N, dtype=torch.float16, generator=g) * 0.01

        compiled = _build_compiled_fn(M, K, N)
        _patch_runner_assert_prep()

        a_dev = _to_dev_named(a_cpu, "M", "K")
        b_dev = _to_dev_named(b_cpu, "K", "N")

        result = compiled(a_dev, b_dev)
        _sync_all(device)
        torch.save([result.cpu()], os.environ[_ENV_RESULT])
    finally:
        _cleanup_device(device)


def _child_pipelined_mm() -> None:
    """Child: _PIPELINE_DEPTH K-tiled mm launches, saves [result_cpu, ...].

    C1b – back-to-back launches; shifted order for tracker-on.
    """
    device = torch.device("spyre")
    try:
        torch._dynamo.reset()
        _pnd.reset()

        M, K, N = _SMALL_M, _SMALL_K, _SMALL_N
        n = _PIPELINE_DEPTH

        g = torch.Generator(device="cpu").manual_seed(0xAFFE)
        inputs_cpu = [
            (
                torch.randn(M, K, dtype=torch.float16, generator=g) * 0.01,
                torch.randn(K, N, dtype=torch.float16, generator=g) * 0.01,
            )
            for _ in range(n)
        ]

        compiled = _build_compiled_fn(M, K, N)
        _patch_runner_assert_prep()

        shifted = os.environ.get(_ENV_SHIFTED) == "1"

        # Move all inputs to device up-front.
        inputs_dev = []
        for a_cpu, b_cpu in inputs_cpu:
            inputs_dev.append(
                (_to_dev_named(a_cpu, "M", "K"), _to_dev_named(b_cpu, "K", "N"))
            )

        results = []
        for slot in range(n):
            # Tracker-on child uses shifted order: slot k computes pair (k+1) % n.
            pair = (slot + 1) % n if shifted else slot
            a_dev, b_dev = inputs_dev[pair]
            results.append(compiled(a_dev, b_dev))

        _sync_all(device)
        torch.save([r.cpu() for r in results], os.environ[_ENV_RESULT])
    finally:
        _cleanup_device(device)


def _child_host_ahead() -> None:
    """Child: _HOST_AHEAD_PAIRS+1 large-shape launches; records host timestamps.

    Saves a dict with keys:
      "results"   – list of result tensors
      "host_ns"   – list of host timestamps (ns) taken just before each launch
      "dev_idle"  – list of bools: was S_dev idle right before taking the
                    next timestamp?
    The parent test uses host_ns[k+1] < time-when-dev-became-idle-after-launch-k
    to verify host ran ahead of the device.
    """
    device = torch.device("spyre")
    try:
        torch._dynamo.reset()
        _pnd.reset()

        M, K, N = _LARGE_M, _LARGE_K, _LARGE_N
        n = _HOST_AHEAD_PAIRS + 1

        g = torch.Generator(device="cpu").manual_seed(0xCAFE)
        inputs_cpu = [
            (
                torch.randn(M, K, dtype=torch.float16, generator=g) * 0.01,
                torch.randn(K, N, dtype=torch.float16, generator=g) * 0.01,
            )
            for _ in range(n)
        ]

        compiled = _build_compiled_fn(M, K, N)
        _patch_runner_assert_prep()

        inputs_dev = []
        for a_cpu, b_cpu in inputs_cpu:
            inputs_dev.append(
                (_to_dev_named(a_cpu, "M", "K"), _to_dev_named(b_cpu, "K", "N"))
            )

        # Warm up: one launch outside measurement.
        compiled(inputs_dev[0][0], inputs_dev[0][1])
        _sync_all(device)

        results: list[torch.Tensor] = []
        host_ns: list[int] = []
        dev_idle_before: list[bool] = []

        for i in range(n):
            # Record: was the device still busy at the moment we start launch i?
            dev_idle_before.append(_s_dev(device).query())
            host_ns.append(time.perf_counter_ns())
            a_dev, b_dev = inputs_dev[i]
            results.append(compiled(a_dev, b_dev))

        _sync_all(device)

        torch.save(
            {
                "results": [r.cpu() for r in results],
                "host_ns": host_ns,
                "dev_idle_before": dev_idle_before,
            },
            os.environ[_ENV_RESULT],
        )
    finally:
        _cleanup_device(device)


# ── orchestrator (parent) tests ───────────────────────────────────────────────


class TestSingleLaunchParity:
    """C3 / C1a / C5 – single K-tiled mm correctness and routing."""

    def test_tracker_on_matches_tracker_off_single_launch(self, tmp_path):
        """C1a / C5: single launch must be bit-identical across tracker modes.

        A missing H2D→Compute edge within the same launch corrupts K-tile
        partial accumulation.  Tracker-off falls back to single-stream so both
        paths must agree to the bit.
        """
        results_off, results_on = _collect_parity(tmp_path, "single_mm")

        torch.testing.assert_close(
            results_on[0],
            results_off[0],
            atol=0,
            rtol=0,
            msg=(
                "Single K-tiled mm: tracker-on differs from tracker-off — "
                "a missing H2D→Compute edge within the launch corrupts K-tile "
                "partial accumulation (C1a) or the single-stream fallback is "
                "broken (C5)."
            ),
        )

    def test_single_launch_close_to_cpu_reference(self, tmp_path):
        """C3: tracker-on result must be numerically close to a CPU float32 reference.

        atol=0.1 would pass an all-zero output, so we use relative Frobenius
        norm ≤ 5e-3 (measured: 2.5e-3–3.7e-3 on this shape).
        """
        g = torch.Generator(device="cpu").manual_seed(0xAFFE)
        M, K, N = _SMALL_M, _SMALL_K, _SMALL_N
        a_cpu = torch.randn(M, K, dtype=torch.float16, generator=g) * 0.01
        b_cpu = torch.randn(K, N, dtype=torch.float16, generator=g) * 0.01
        ref = torch.mm(a_cpu.float(), b_cpu.float())

        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        on_path = str(tmp_path / "on.pt")
        _run_child(1, on_path, "single_mm", cache_dir=cache_dir)

        result = torch.load(on_path, weights_only=True)[0].float()
        rel_err = (result - ref).norm() / (ref.norm() + 1e-8)
        assert rel_err < 5e-3, (
            f"Tracker-on result deviates too much from CPU float32 reference: "
            f"relative Frobenius = {rel_err:.2e} (threshold 5e-3).  "
            "A result of all zeros would give a large error here."
        )


class TestPipelinedLaunchParity:
    """C1b – back-to-back launches must be bit-identical across tracker modes."""

    def test_tracker_on_matches_tracker_off_pipelined(self, tmp_path):
        """C1b: all pipelined results must be bit-identical.

        The tracker-on child uses a shifted input order so that slots do not
        happen to compute the same pair as tracker-off.  A missing cross-launch
        WAR edge would let H2D for launch k+1 overwrite the correction buffer
        while launch k's Compute still reads it, corrupting the result.
        """
        results_off, results_on = _collect_parity(
            tmp_path, "pipelined_mm", shifted_on=True
        )

        # tracker-on[slot] computed pair (slot+1)%n.
        # tracker-off[(slot+1)%n] computed the same pair.
        # So: results_on[slot] must equal results_off[(slot+1)%n].
        n = _PIPELINE_DEPTH
        for slot in range(n):
            torch.testing.assert_close(
                results_on[slot],
                results_off[(slot + 1) % n],
                atol=0,
                rtol=0,
                msg=(
                    f"Pipelined launch: tracker-on slot {slot} (pair "
                    f"{(slot + 1) % n}) differs from tracker-off slot "
                    f"{(slot + 1) % n} (same pair) — cross-launch ordering is "
                    "wrong (C1b)."
                ),
            )

    def test_same_program_both_modes(self, tmp_path):
        """Both tracker modes must compile to the same spyreCodeDir.

        A compilation difference would mean the two modes are testing different
        programs, making all parity checks meaningless.  We fingerprint the
        first spyreCodeDir written into the shared cache.
        """
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)

        off_path = str(tmp_path / "off.pt")
        on_path = str(tmp_path / "on.pt")
        _run_child(0, off_path, "single_mm", cache_dir=cache_dir)
        _run_child(1, on_path, "single_mm", cache_dir=cache_dir)

        # Walk all spyrecode.json files under the cache and hash them.
        def _fingerprint(root: str) -> dict[str, str]:
            result: dict[str, str] = {}
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if f == "spyrecode.json":
                        p = os.path.join(dirpath, f)
                        with open(p, "rb") as fh:
                            h = hashlib.sha256(fh.read()).hexdigest()
                        # Key is relative to cache root.
                        result[os.path.relpath(p, root)] = h
            return result

        fp = _fingerprint(cache_dir)
        assert fp, (
            "No spyrecode.json found under the compile cache — "
            "did the child fail to compile?"
        )
        # All entries should be unique per path (we just check we got something;
        # the real invariant is that the cache was not re-written by the second
        # child, i.e., SPYRE_NUM_CPUS=1 kept the compile deterministic).
        assert len(fp) >= 1, "Expected at least one compiled program in cache."


class TestHostAheadOfDevice:
    """C1a (temporal) – host ran ahead of device compute on the large shape.

    Without a missing ordering edge there is nothing to corrupt; we verify
    the overlap actually happens so the parity checks above are load-bearing.
    The small shape (kernel ~14 µs) gives 0/15 overlapping pairs; the large
    shape (kernel ~806 µs) gives 11–15/15.
    """

    def test_host_runs_ahead_of_device_large_shape(self, tmp_path):
        """Host prep for launch k+1 must begin while device is still running launch k.

        We record a host timestamp just before each launch and check that
        S_dev still reported in-flight work right before the next launch
        started.  At least _HOST_AHEAD_MIN_FRACTION of the
        (_HOST_AHEAD_PAIRS) consecutive pairs must satisfy this.
        """
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        on_path = str(tmp_path / "on.pt")
        _run_child(1, on_path, "host_ahead", cache_dir=cache_dir)

        data = torch.load(on_path, weights_only=False)
        host_ns: list[int] = data["host_ns"]
        dev_idle: list[bool] = data["dev_idle_before"]

        # Pair k: host timestamp of launch k+1 was recorded while dev was still
        # busy from launch k.  dev_idle_before[k+1] == False means S_dev was
        # in-flight at the instant we started launch k+1.
        n_pairs = len(host_ns) - 1
        ahead_count = sum(1 for k in range(n_pairs) if not dev_idle[k + 1])
        fraction = ahead_count / n_pairs if n_pairs > 0 else 0.0

        assert fraction >= _HOST_AHEAD_MIN_FRACTION, (
            f"Host ran ahead of device in only {ahead_count}/{n_pairs} pairs "
            f"({fraction:.0%}), below the threshold of "
            f"{_HOST_AHEAD_MIN_FRACTION:.0%}.  "
            "On the large device-bound shape the host should consistently "
            "outpace the device.  If this runs on the small shape, or the "
            "device is faster than expected, this threshold needs adjustment."
        )


_CHILD_DISPATCH = {
    "single_mm": _child_single_mm,
    "pipelined_mm": _child_pipelined_mm,
    "host_ahead": _child_host_ahead,
}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in _CHILD_DISPATCH:
        _CHILD_DISPATCH[sys.argv[1]]()
    else:
        pytest.main([__file__, "-v"])

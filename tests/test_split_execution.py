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

"""Split-execution parity and routing tests (issue #2889).

Every test runs in a child process; the parent never opens the device.
"""

import hashlib
import multiprocessing
import pathlib
import traceback

import pytest
import torch


@pytest.fixture(scope="session")
def shared_compile_cache(tmp_path_factory) -> pathlib.Path:
    """One compile cache shared by every child."""
    return tmp_path_factory.mktemp("spyre_compile_cache", numbered=False)


# Device-bound shape: kernel time >> host compute time.
_MM_M, _MM_K, _MM_N = 1024, 8192, 1024
_MM_DTYPE = torch.float16


def _child_s_prep_routing(tracker_on: bool) -> dict:
    """Child: report the tracker flag and the JobPlan step types and roles."""
    import json
    import os
    import tempfile

    import torch

    from torch_spyre import _C

    # Opens the device, which latches the tracker flag.
    torch.zeros(1, device="spyre")

    tracker_enabled = _C.get_hazard_tracker_enabled()

    # Mock plan with the shape the mm compiler emits.
    with tempfile.TemporaryDirectory() as tmpdir:
        spyrecode_dir = f"{tmpdir}/spyreCodeDir"
        os.makedirs(spyrecode_dir)
        spyrecode_json = {
            "JobPreparationPlan": [
                {"command": "Allocate", "properties": {"size": "1024"}},
                {
                    "command": "InitTransfer",
                    "properties": {
                        "init_bin_file": "init_binary.bin",
                        "dev_ptr": "120259084288",
                        "size": "1024",
                    },
                },
            ],
            "JobExecPlan": [
                {
                    "command": "ComputeOnHost",
                    "properties": {
                        "ohandle": "correction",
                        "size": "256",
                        "ishape": ["0"],
                        "ihandle": "",
                        "hcm": {"vdci": {}, "senConstants": []},
                    },
                },
                {
                    "command": "DataTransfer",
                    "properties": {
                        "dirn": "false",
                        "host_handle": "correction",
                        "dev_ptr": "120259084288",
                        "size": "256",
                    },
                },
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                },
            ],
        }
        with open(f"{spyrecode_dir}/spyrecode.json", "w") as f:
            json.dump(spyrecode_json, f)
        with open(f"{spyrecode_dir}/init_binary.bin", "wb") as f:
            f.write(b"\x00" * 1024)

        job_plan = _C.prepare_kernel(spyrecode_dir)

    num_steps = job_plan.num_steps()
    step_types = [job_plan.get_step_type(i) for i in range(num_steps)]
    step_roles = [job_plan.get_step_stream_role(i) for i in range(num_steps)]

    return {
        "tracker_on": tracker_on,
        "tracker_enabled": tracker_enabled,
        "step_types": step_types,
        "step_roles": step_roles,
    }


def _child_single_launch_correctness(tracker_on: bool) -> dict:
    """Child: run mm once; return the output and CPU reference."""
    import torch
    import torch._dynamo

    import torch_spyre  # noqa: F401  # registers the backend

    torch._dynamo.reset()
    torch.manual_seed(0)
    a_cpu = torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE)
    b_cpu = torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE)
    ref = torch.mm(a_cpu.float(), b_cpu.float())

    a = a_cpu.to("spyre")
    b = b_cpu.to("spyre")
    compiled = torch.compile(torch.mm, backend="inductor")
    out = compiled(a, b).cpu()

    torch.accelerator.synchronize()
    return {
        "out": out.float().numpy(),
        "ref": ref.float().numpy(),
        "tracker_on": tracker_on,
        "shape": list(out.shape),
    }


def _child_back_to_back(
    tracker_on: bool,
    num_launches: int,
    shift: int,
) -> dict:
    """Child: run mm num_launches times in order (k + shift) % n.

    Readback is deferred until all launches are enqueued so they overlap.
    """
    import torch
    import torch._dynamo

    import torch_spyre  # noqa: F401  # registers the backend

    torch._dynamo.reset()
    torch.manual_seed(1)
    device = torch.device("spyre")

    pairs_cpu: list[tuple[torch.Tensor, torch.Tensor]] = [
        (
            torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE),
            torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE),
        )
        for _ in range(num_launches)
    ]
    pairs_spyre = [(a.to(device), b.to(device)) for a, b in pairs_cpu]

    compiled = torch.compile(torch.mm, backend="inductor")
    compiled(pairs_spyre[0][0], pairs_spyre[0][1])

    # No device-to-host reads in the loop: they would serialize the launches.
    order = [(k + shift) % num_launches for k in range(num_launches)]
    outs_spyre: dict[int, torch.Tensor] = {}
    for idx in order:
        a_s, b_s = pairs_spyre[idx]
        outs_spyre[idx] = compiled(a_s, b_s)

    torch.accelerator.synchronize(device)

    outputs_by_pair: dict[int, object] = {}
    refs_by_pair: dict[int, object] = {}
    for idx in order:
        a_c, b_c = pairs_cpu[idx]
        outputs_by_pair[idx] = outs_spyre[idx].cpu().float().numpy()
        refs_by_pair[idx] = torch.mm(a_c.float(), b_c.float()).numpy()

    outputs = [outputs_by_pair[(k + shift) % num_launches] for k in range(num_launches)]
    refs = [refs_by_pair[(k + shift) % num_launches] for k in range(num_launches)]

    return {
        "tracker_on": tracker_on,
        "outputs": outputs,
        "refs": refs,
        "outputs_by_pair": outputs_by_pair,
    }


def _child_wrapper(
    env_overrides: dict,
    target,
    args: tuple,
    conn,
) -> None:
    """Apply env overrides, run target, send back (ok, result)."""
    import os

    try:
        # The runtime reads env vars once at startup: set them first.
        for k, v in env_overrides.items():
            os.environ[k] = v
        result = target(*args)
        conn.send((True, result))
    except Exception as e:  # noqa: BLE001
        conn.send((False, f"{e}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def _run_in_child(
    target,
    args: tuple,
    env_overrides: dict,
    cache_dir: pathlib.Path,
    timeout: float = 120.0,
) -> dict:
    """Run target in a spawned child with env overrides; return its result."""
    all_overrides = {
        "TORCHINDUCTOR_CACHE_DIR": str(cache_dir),
        "SPYRE_NUM_CPUS": "1",
    }
    all_overrides.update(env_overrides)

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_child_wrapper, args=(all_overrides, target, args, child_conn)
    )
    p.start()
    child_conn.close()

    success: bool | None = None
    payload: dict | None = None
    if parent_conn.poll(timeout):
        try:
            success, payload = parent_conn.recv()
        except EOFError:
            pass

    p.join(timeout=5.0)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Child process timed out after {timeout}s")

    parent_conn.close()

    if p.exitcode != 0 and success is None:
        raise RuntimeError(f"Child process exited with code {p.exitcode}")

    if success is False:
        raise RuntimeError(f"Child process failed:\n{payload}")
    if success is None or payload is None:
        raise RuntimeError(
            f"Child process exited without returning a result (exitcode={p.exitcode})"
        )

    return payload


def _compiled_programs(cache_dir: pathlib.Path) -> set[str]:
    """Hashes of every compiled spyrecode.json under cache_dir."""
    return {
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in cache_dir.rglob("spyreCodeDir/spyrecode.json")
    }


def _tracker_on_env() -> dict:
    return {"SPYRE_HAZARD_TRACKER": "1"}


def _tracker_off_env() -> dict:
    return {"SPYRE_HAZARD_TRACKER": "0"}


class TestSPrepRouting:
    """The tracker flag is read and the plan has the expected step roles."""

    def test_tracker_on_enables_split_routing(self, shared_compile_cache):
        """Tracker on: flag is True; HostCompute is Prep and Compute is Dev."""
        result = _run_in_child(
            _child_s_prep_routing,
            args=(True,),
            env_overrides=_tracker_on_env(),
            cache_dir=shared_compile_cache,
        )
        assert result["tracker_enabled"] is True, (
            "SPYRE_HAZARD_TRACKER=1 was set but get_hazard_tracker_enabled() "
            "returned False — the env var was not read at runtime startup."
        )
        assert "HostCompute" in result["step_types"], (
            f"Expected a HostCompute step in the mm JobPlan but got: "
            f"{result['step_types']}"
        )
        hc_idx = result["step_types"].index("HostCompute")
        assert result["step_roles"][hc_idx] == "Prep", (
            f"HostCompute step must have role Prep for S_prep routing, "
            f"got: {result['step_roles'][hc_idx]}"
        )
        compute_idx = result["step_types"].index("Compute")
        assert result["step_roles"][compute_idx] == "Dev", (
            f"Compute step must have role Dev (always on S_dev), "
            f"got: {result['step_roles'][compute_idx]}"
        )

    def test_tracker_off_disables_split_routing(self, shared_compile_cache):
        """Tracker off: flag is False; the plan still has HostCompute."""
        result = _run_in_child(
            _child_s_prep_routing,
            args=(False,),
            env_overrides=_tracker_off_env(),
            cache_dir=shared_compile_cache,
        )
        assert result["tracker_enabled"] is False, (
            "SPYRE_HAZARD_TRACKER=0 was set but get_hazard_tracker_enabled() "
            "returned True — the env var was not read correctly."
        )
        assert "HostCompute" in result["step_types"], (
            "Even with tracker off the mm plan must still contain HostCompute "
            f"(plan shape is tracker-agnostic). Got: {result['step_types']}"
        )


class TestSingleLaunchCorrectness:
    """One mm launch is correct in both tracker modes."""

    @pytest.mark.parametrize(
        "tracker_on", [True, False], ids=["tracker_on", "tracker_off"]
    )
    def test_single_mm_matches_cpu(self, tracker_on, shared_compile_cache):
        """mm output matches the CPU reference."""
        env = _tracker_on_env() if tracker_on else _tracker_off_env()
        result = _run_in_child(
            _child_single_launch_correctness,
            args=(tracker_on,),
            env_overrides=env,
            cache_dir=shared_compile_cache,
        )
        out = torch.tensor(result["out"])
        ref = torch.tensor(result["ref"])
        rel_frob = (torch.linalg.norm(out - ref) / torch.linalg.norm(ref)).item()
        assert rel_frob < 0.01, (
            f"Relative Frobenius error {rel_frob:.4e} exceeds 1% "
            f"(tracker_on={tracker_on})"
        )


_NUM_BACK_TO_BACK = 4


class TestBackToBackParity:
    """Back-to-back launches are correct and use the same kernel.

    The tracker-on child launches pairs in a shifted order.
    """

    def test_back_to_back_tracker_on_matches_cpu(self, shared_compile_cache):
        """Tracker on: every output matches the CPU reference."""
        result = _run_in_child(
            _child_back_to_back,
            args=(True, _NUM_BACK_TO_BACK, 1),
            env_overrides=_tracker_on_env(),
            cache_dir=shared_compile_cache,
        )
        for k, (out_list, ref_list) in enumerate(
            zip(result["outputs"], result["refs"])
        ):
            out_k = torch.tensor(out_list)
            ref_k = torch.tensor(ref_list)
            rel_frob = (
                torch.linalg.norm(out_k - ref_k) / torch.linalg.norm(ref_k)
            ).item()
            assert rel_frob < 0.01, (
                f"tracker_on back-to-back launch {k}: relative Frobenius error "
                f"{rel_frob:.4e} exceeds 1%"
            )

    def test_back_to_back_tracker_off_matches_cpu(self, shared_compile_cache):
        """Tracker off: every output matches the CPU reference."""
        result = _run_in_child(
            _child_back_to_back,
            args=(False, _NUM_BACK_TO_BACK, 0),
            env_overrides=_tracker_off_env(),
            cache_dir=shared_compile_cache,
        )
        for k, (out_list, ref_list) in enumerate(
            zip(result["outputs"], result["refs"])
        ):
            out_k = torch.tensor(out_list)
            ref_k = torch.tensor(ref_list)
            rel_frob = (
                torch.linalg.norm(out_k - ref_k) / torch.linalg.norm(ref_k)
            ).item()
            assert rel_frob < 0.01, (
                f"tracker_off back-to-back launch {k}: relative Frobenius error "
                f"{rel_frob:.4e} exceeds 1%"
            )

    def test_tracker_on_off_outputs_are_identical(self, shared_compile_cache):
        """Tracker on and off give identical outputs from the same program."""
        result_on = _run_in_child(
            _child_back_to_back,
            args=(True, _NUM_BACK_TO_BACK, 1),
            env_overrides=_tracker_on_env(),
            cache_dir=shared_compile_cache,
        )
        programs_on = _compiled_programs(shared_compile_cache)
        result_off = _run_in_child(
            _child_back_to_back,
            args=(False, _NUM_BACK_TO_BACK, 0),
            env_overrides=_tracker_off_env(),
            cache_dir=shared_compile_cache,
        )
        programs_off = _compiled_programs(shared_compile_cache)

        assert programs_on, (
            f"No compiled spyrecode.json found under {shared_compile_cache}"
        )
        assert programs_off == programs_on, (
            "tracker-off child compiled a different program than tracker-on: "
            f"{len(programs_off - programs_on)} new spyrecode.json variant(s)"
        )

        on_by_pair: dict = result_on["outputs_by_pair"]
        off_by_pair: dict = result_off["outputs_by_pair"]

        assert set(on_by_pair.keys()) == set(off_by_pair.keys()), (
            f"Pair index mismatch between children: "
            f"on={sorted(on_by_pair.keys())} off={sorted(off_by_pair.keys())}"
        )

        for pair_idx in sorted(on_by_pair.keys()):
            out_on = torch.tensor(on_by_pair[pair_idx])
            out_off = torch.tensor(off_by_pair[pair_idx])
            if not torch.equal(out_on, out_off):
                torch.testing.assert_close(
                    out_on,
                    out_off,
                    atol=1e-5,
                    rtol=0.0,
                    msg=(
                        f"pair {pair_idx}: tracker-on and tracker-off outputs differ — "
                        "a missing WAR/RAW edge may have corrupted an in-flight kernel"
                    ),
                )

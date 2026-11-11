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

"""Collective metadata (coll_*) on AIUPTI kernel and memcpy trace events.

The non-collective half of this contract (no coll_* args outside a collective)
is covered single-process in tests/profiler/test_spyre_profiler.py.
"""

import json
import os

import pytest
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile
from torch.testing._internal.common_utils import (
    TemporaryFileName,
    TestCase,
    run_tests,
)

# Skip all tests if RANK is not defined, or WORLD_SIZE is not set or less than 2
if "RANK" not in os.environ:
    pytest.skip(
        "RANK environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

if "WORLD_SIZE" not in os.environ:
    pytest.skip(
        "WORLD_SIZE environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

try:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size < 2:
        pytest.skip(
            f"WORLD_SIZE is {world_size}, need at least 2 for distributed tests",
            allow_module_level=True,
        )
except ValueError:
    pytest.skip(
        "WORLD_SIZE environment variable is not a valid integer, skipping distributed tests",
        allow_module_level=True,
    )

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
ACTIVITY_CATEGORIES = {"kernel", "gpu_memcpy"}
COLL_ARGS = ("coll_group", "coll_algo", "coll_bytes")


@pytest.mark.requires_spyre_profiler
class TestProfilerCollMetadata(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the distributed environment once for all tests."""
        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )

        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        """Clean up the distributed environment after all tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _profile_allreduce_trace(self, numel=128):
        input_device = torch.arange(numel, dtype=torch.float16).to(DEVICE)
        # Warm up outside profile() so one-time setup does not dominate the trace.
        dist.all_reduce(input_device, op=dist.ReduceOp.SUM)
        torch.spyre.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as prof:
            dist.all_reduce(input_device, op=dist.ReduceOp.SUM)
            torch.spyre.synchronize()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                return json.load(f)

    def _coll_events(self, trace):
        """Trace events carrying at least one coll_* arg; skips if there are none."""
        events = [
            e
            for e in trace["traceEvents"]
            if any(k in e.get("args", {}) for k in COLL_ARGS)
        ]
        if not events:
            self.skipTest(
                f"Rank {self.comm_rank}: no trace event carries coll_* args "
                "(libaiupti/flex build without coll metadata)"
            )
        return events

    def test_allreduce_activities_carry_coll_args(self):
        """coll_* args of an all_reduce are well-formed, in any combination.

        Each of coll_group/coll_algo/coll_bytes may be present on its own, since
        the producer can fill some fields and not others. Whatever is present
        must have the right type: names are non-empty strings, and coll_bytes is
        a non-negative number, or "" when the size is unknown.
        """
        trace = self._profile_allreduce_trace()
        for e in self._coll_events(trace):
            args = e["args"]
            name = e.get("name")
            for key in ("coll_group", "coll_algo"):
                if key in args:
                    self.assertIsInstance(args[key], str, f"{key} on {name!r}")
                    self.assertTrue(args[key], f"{key} emitted empty on {name!r}")
            if "coll_bytes" in args:
                value = args["coll_bytes"]
                if isinstance(value, str):
                    self.assertEqual(value, "", f"coll_bytes on {name!r}")
                else:
                    self.assertIsInstance(value, int, f"coll_bytes on {name!r}")
                    self.assertGreaterEqual(value, 0, f"coll_bytes on {name!r}")

    def test_coll_algo_names_are_registered(self):
        """Every coll_algo resolves to a name; none falls back to algo_0x<id>.

        The record carries an algorithm ID that spyre-comms registers with flex at
        init. The fallback is written only for an ID nobody registered, so seeing
        it means the registration or a new algorithm's table entry is missing.
        """
        trace = self._profile_allreduce_trace()
        for e in self._coll_events(trace):
            algo = e["args"].get("coll_algo")
            if algo is not None:
                self.assertFalse(
                    algo.startswith("algo_0x"),
                    f"unregistered collective algorithm ID {algo} on {e.get('name')!r}",
                )

    def test_forced_long_algorithm_name_is_not_truncated(self):
        """AllReduce_Offload_GatherSumBcast (32 characters) keeps its last letter.

        Names used to be cut to 31 characters, which dropped the final "t". This
        forces that algorithm. If the library picks another one, there is nothing
        to check and the test skips; if the name appears at all it must be whole.
        """
        full = "AllReduce_Offload_GatherSumBcast"
        previous = os.environ.get("COLL_ALLREDUCE_ALGO")
        os.environ["COLL_ALLREDUCE_ALGO"] = "Offload_GatherSumBcast"
        try:
            # A size no other test uses, so a cached algorithm choice is not reused.
            trace = self._profile_allreduce_trace(numel=384)
        finally:
            if previous is None:
                os.environ.pop("COLL_ALLREDUCE_ALGO", None)
            else:
                os.environ["COLL_ALLREDUCE_ALGO"] = previous

        algos = {
            e["args"]["coll_algo"]
            for e in self._coll_events(trace)
            if "coll_algo" in e["args"]
        }
        cut = full[:31]
        seen = {a for a in algos if a.startswith(cut)}
        if not seen:
            self.skipTest(
                f"Rank {self.comm_rank}: {full} was not selected (saw {sorted(algos)})"
            )
        self.assertEqual(seen, {full})

    def test_coll_args_only_on_device_activities(self):
        """coll_* args appear only on kernel/memcpy events, never on CPU ops."""
        trace = self._profile_allreduce_trace()
        for e in self._coll_events(trace):
            self.assertIn(
                e.get("cat"),
                ACTIVITY_CATEGORIES,
                f"coll_* args on non-device event {e.get('name')!r}",
            )


if __name__ == "__main__":
    run_tests()

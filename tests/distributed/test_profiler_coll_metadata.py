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

    def _profile_allreduce_trace(self):
        input_device = torch.arange(128, dtype=torch.float16).to(DEVICE)
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

    def test_allreduce_activities_carry_coll_args(self):
        """Kernel/memcpy events of an all_reduce carry coll_group/algo/bytes.

        On failure the message lists every distinct coll_* arg set seen, which
        shows how often coll_bytes is "" (unknown) versus a number.
        """
        trace = self._profile_allreduce_trace()
        coll_events = [
            e
            for e in trace["traceEvents"]
            if e.get("ph") == "X"
            and e.get("cat") in ACTIVITY_CATEGORIES
            and "coll_group" in e.get("args", {})
        ]
        self.assertTrue(
            coll_events,
            f"Rank {self.comm_rank}: no kernel/memcpy event carries coll_group",
        )

        for e in coll_events:
            args = e["args"]
            self.assertIsInstance(args["coll_group"], str)
            self.assertTrue(args["coll_group"], "coll_group emitted empty")
            # Every collective record carries coll_bytes. It is a number when
            # the label carried a size (coll_algo set), including a real 0,
            # and "" when flex saw a bare collective name with no size.
            self.assertIn("coll_bytes", args, f"event {e.get('name')!r}")
            if "coll_algo" in args:
                self.assertIsInstance(args["coll_algo"], str)
                self.assertTrue(args["coll_algo"], "coll_algo emitted empty")
                self.assertIsInstance(
                    args["coll_bytes"],
                    int,
                    "args.coll_bytes must be a JSON number when the size is known",
                )
                self.assertGreaterEqual(args["coll_bytes"], 0)
            else:
                self.assertEqual(
                    args["coll_bytes"],
                    "",
                    "args.coll_bytes must be '' when the size is unknown",
                )

        # all_reduce goes through Collective::Convert(), which labels its CBs
        # "[Allreduce,<algo>,<bytes>]", so at least one record has a known size.
        self.assertTrue(
            any(
                e["args"]["coll_group"] == "Allreduce"
                and "coll_algo" in e["args"]
                and isinstance(e["args"]["coll_bytes"], int)
                for e in coll_events
            ),
            f"Rank {self.comm_rank}: no event has coll_group='Allreduce' with "
            f"coll_algo and a numeric coll_bytes; saw "
            f"{sorted({json.dumps(e['args'], sort_keys=True) for e in coll_events})}",
        )

    def test_coll_args_only_with_coll_group(self):
        """coll_algo/coll_bytes never appear on an event without coll_group."""
        trace = self._profile_allreduce_trace()
        for e in trace["traceEvents"]:
            if e.get("cat") not in ACTIVITY_CATEGORIES:
                continue
            args = e.get("args", {})
            if "coll_group" in args:
                continue
            self.assertNotIn("coll_algo", args, f"event {e.get('name')!r}")
            self.assertNotIn("coll_bytes", args, f"event {e.get('name')!r}")


if __name__ == "__main__":
    run_tests()

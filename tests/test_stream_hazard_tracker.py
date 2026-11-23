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

# Owner(s): ["module: stream"]

"""Flex host retention of launches under SPYRE_HAZARD_TRACKER (#6569).

The tracker is latched when the device runtime starts, so each case runs in a
child process with it on. The child needs the card, so the cases skip when
this process already holds it (as in a combined `pytest tests/` run), and
this file uses plain test functions: `TestCase.setUp` seeds the RNG, which
opens the card.
"""

import json
import os
import subprocess
import sys

import pytest
import torch
import torch_spyre  # noqa: F401

# Retained host bytes per launch above which a stream counts as leaking. Flex
# keeps ~860 B per correction-carrying launch with the tracker on today.
_MAX_RETAINED_BYTES_PER_LAUNCH = 64

# Prints the host RSS growth per launch of a kernel whose launches carry
# program corrections (HostCompute and a correction H2D on S_prep under the
# tracker), after a warm-up. "device" synchronizes the device every 1024
# launches; "current" only copies the result to the host, which synchronizes
# the current stream but never S_prep.
_CHILD = r"""
import ctypes, gc, json, os, sys
import torch
import torch_spyre  # noqa: F401

sync, launches = sys.argv[1], int(sys.argv[2])
libc = ctypes.CDLL("libc.so.6")
fn = torch.compile(lambda a, b: torch.softmax(a * b + a, dim=-1))
x = torch.randn(64, 128, dtype=torch.float16).to("spyre")
y = torch.randn(64, 128, dtype=torch.float16).to("spyre")


def run(n):
    for i in range(n):
        out = fn(x, y)
        if i % 1024 == 1023 or i == n - 1:
            if sync == "device":
                torch.spyre.synchronize()
            else:
                out.cpu()


def rss():
    gc.collect()
    libc.malloc_trim(0)
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


run(launches // 4)
before = rss()
run(launches)
print(json.dumps({"bytes_per_launch": (rss() - before) / launches}))
"""


@pytest.fixture
def child_device():
    if os.environ.get("FLEX_DEVICE") != "PF":
        pytest.skip("correction uploads run on S_prep only on a PF device")
    if torch.spyre.is_initialized():
        pytest.skip("the child needs the card this process already holds")


def _retained_bytes_per_launch(sync, recycle_ops):
    env = {
        **os.environ,
        "SPYRE_HAZARD_TRACKER": "1",
        "SPYRE_HAZARD_COMPLETED_READS": "1",
        "TORCH_SPYRE_STREAM_RECYCLE_OPS": str(recycle_ops),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, sync, "20000"],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"child failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])["bytes_per_launch"]


def test_recycling_bounds_launch_state_when_only_current_stream_syncs(
    child_device,
):
    """S_prep is recycled although nothing synchronizes it.

    Only the result copy synchronizes, as on a serving rank whose step ends
    in a D2H of its outputs, or that never waits at all.
    """
    assert _retained_bytes_per_launch("current", 256) < (_MAX_RETAINED_BYTES_PER_LAUNCH)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flex keeps per-launch host state on a hazard-tracking stream "
    "until the stream is destroyed (torch-spyre#6569). An XPASS means flex "
    "frees it: drop TORCH_SPYRE_STREAM_RECYCLE_OPS and this xfail.",
)
def test_flex_frees_hazard_tracked_launch_state_without_recycling(child_device):
    assert _retained_bytes_per_launch("device", 0) < _MAX_RETAINED_BYTES_PER_LAUNCH

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

"""A closed-form multi-rank decode trace (synthetic signals) in the shape trace_bundle.py reads.

4 ranks x 64 steps x (40 layers x 6 span patterns): step 0 is the cold first token, step 1 the cold
first decode step, step 30 a stall that starts on rank 1 and is waited for by the others; rank 2 is the
slowest card otherwise. total = compute + wait + host with wait = 50 - offset, so the awaited rank
(least wait) is the one with the largest offset."""

import math

OFFSET = (0.0, 2.0, 8.0, 1.0)
STALL_STEP, STALL_RANK, STALL = 30, 1, 100.0
LANES = (
    ("host", "launch"),
    ("host", "sync"),
    ("device", "matmul"),
    ("device", "norm"),
    ("collective", "all_reduce"),
    ("collective", "wait_peer"),
)
COLD_WAIT_NS = (1_128_700, 3_300, 668_600, 668_800)


def jitter(step):
    return 0.5 * math.sin(step * 1.7)


def step_values(rank, step):
    off = OFFSET[rank]
    if step == 0:
        return {
            "sig_total": 400.0 + off,
            "sig_wait": 150.0 - off,
            "sig_compute": 240.0 + off,
            "sig_busy": 245.0 + off,
            "sig_aux": 1.0,
        }
    compute = 100.0 + off + jitter(step)
    wait = 50.0 - off
    if step == STALL_STEP:
        if rank == STALL_RANK:
            compute += STALL
        else:
            wait += STALL
    return {
        "sig_total": compute + wait + 10.0 + off,
        "sig_wait": wait,
        "sig_compute": compute,
        "sig_busy": compute + 5.0,
        "sig_aux": 1.0,
    }


def spans(rank, steps, layers):
    out = []
    for s in range(steps):
        base = s * 10_000_000
        for layer in range(layers):
            for i, (lane, name) in enumerate(LANES):
                dur = {"launch": 15_000, "sync": 5_000, "matmul": 100_000 + 1000 * rank, "norm": 20_000,
                       "all_reduce": 20_000 + 500 * rank, "wait_peer": 10_000 + 100 * layer}[name]  # fmt: skip
                if name == "wait_peer" and s == 0 and layer == 0:
                    dur = COLD_WAIT_NS[rank]
                out.append({"step": s, "seq": layer * 6 + i, "parent_seq": layer * 6 if i in (2, 3) else -1,
                            "depth": 1 if i in (2, 3) else 0, "layer": layer, "lane": lane, "name": name,
                            "start_ns": base + layer * 200_000 + i * 30_000, "dur_ns": dur})  # fmt: skip
    return out


def trace(ranks=4, steps=64, layers=40, with_spans=True, group_id="synth-group-1"):
    return {
        "component": "synth-trace",
        "backend": "synth-backend",
        "case": {"name": f"synth_decode_tp{ranks}", "props": {"model": "synth-model", "world_size": ranks}},
        "roles": {"total": "sig_total", "wait": "sig_wait", "compute": "sig_compute", "busy": "sig_busy"},
        "rank_rules": {"sig_total": "max", "sig_compute": "mean"},
        "group_id": group_id,
        "reporter_version": "layout-2",
        "reporter_version_source": "reported",
        "trace_uri": "https://assets.example.invalid/synth/trace.json",
        "ranks": [
            {"rank": r, "prompt": "", "repeat": 0,
             "steps": [{"kind": "ttft" if s == 0 else "itl", "cold": 1 if s < 2 else 0, "values": step_values(r, s)} for s in range(steps)],
             "spans": spans(r, steps, layers) if with_spans else []}
            for r in range(ranks)
        ],
    }  # fmt: skip

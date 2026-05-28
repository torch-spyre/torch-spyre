# Copyright 2025 The Torch-Spyre Authors.
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

"""Regression tests for spyre.overwrite with runtime-varying offsets.

Guards against the paged-KV-cache scatter regression: a Python loop of
overwrite() calls with varying offsets used to silently reuse the first
call's compiled binary because dynamo's default specialize_int=False
left int-list args un-guarded. See customops.py for the fix.
"""

import unittest
import warnings

import torch

warnings.filterwarnings(
    "ignore",
    message=r"torch\.ops\.spyre\.overwrite is deprecated.*",
    category=FutureWarning,
)


DEVICE = "spyre"
DTYPE = torch.float16


def _make_cache(shape, sentinel=-1.0):
    return torch.full(shape, sentinel, dtype=DTYPE).to(DEVICE)


def _make_chunk(shape, value):
    return torch.full(shape, float(value), dtype=DTYPE).to(DEVICE)


class TestOverwriteSingleCall(unittest.TestCase):
    def test_single_call_writes_single_slot(self):
        cache = _make_cache((8, 4, 64))
        token = _make_chunk((1, 4, 64), 7)
        torch.ops.spyre.overwrite(token, cache, [0], [3])
        out = cache.to("cpu")
        torch.testing.assert_close(out[3, 0, 0], torch.tensor(7.0, dtype=DTYPE))
        for i in [0, 1, 2, 4, 5, 6, 7]:
            torch.testing.assert_close(out[i, 0, 0], torch.tensor(-1.0, dtype=DTYPE))

    def test_single_call_writes_multi_slot_chunk(self):
        cache = _make_cache((16, 4, 64))
        chunk = torch.stack(
            [torch.full((4, 64), float(i + 1), dtype=DTYPE) for i in range(5)]
        ).to(DEVICE)
        torch.ops.spyre.overwrite(chunk, cache, [0], [2])
        out = cache.to("cpu")
        for i in range(5):
            torch.testing.assert_close(
                out[2 + i, 0, 0], torch.tensor(float(i + 1), dtype=DTYPE)
            )


class TestOverwriteRuntimeVaryingOffsets(unittest.TestCase):
    """Regression coverage for the paged-KV-cache scatter pattern.

    On the unfixed tree every call after the first silently wrote to
    the first call's offsets, leaving the requested slots at sentinel.
    """

    def test_loop_int_offsets(self):
        cache = _make_cache((64, 4, 32))
        slots = [3, 17, 18, 42, 63]
        for i, slot in enumerate(slots):
            tok = _make_chunk((1, 4, 32), i + 1)
            torch.ops.spyre.overwrite(tok, cache, [0], [slot])
        out = cache.to("cpu")
        for i, slot in enumerate(slots):
            torch.testing.assert_close(
                out[slot, 0, 0],
                torch.tensor(float(i + 1), dtype=DTYPE),
                msg=f"slot {slot}: expected {i + 1}, got {out[slot, 0, 0].item()}",
            )

    def test_revisit_offset_correctness(self):
        """Revisiting a previously-seen offset still produces correct
        output (regardless of whether the underlying compiled binary
        is reused or recompiled — that's an implementation detail).
        """
        cache = _make_cache((32, 4, 64))
        sequence = [3, 17, 3, 25, 17]
        last_value_at_slot = {}
        for i, slot in enumerate(sequence):
            value = float(i + 100)
            tok = _make_chunk((1, 4, 64), value)
            torch.ops.spyre.overwrite(tok, cache, [0], [slot])
            last_value_at_slot[slot] = value
        out = cache.to("cpu")
        for slot, expected in last_value_at_slot.items():
            torch.testing.assert_close(
                out[slot, 0, 0], torch.tensor(expected, dtype=DTYPE)
            )

    def test_loop_does_not_disturb_unrelated_slots(self):
        cache = _make_cache((16, 4, 64))
        written = [2, 5, 11]
        for i, slot in enumerate(written):
            tok = _make_chunk((1, 4, 64), i + 1)
            torch.ops.spyre.overwrite(tok, cache, [0], [slot])
        out = cache.to("cpu")
        for slot in range(16):
            if slot in written:
                idx = written.index(slot)
                torch.testing.assert_close(
                    out[slot, 0, 0], torch.tensor(float(idx + 1), dtype=DTYPE)
                )
            else:
                torch.testing.assert_close(
                    out[slot, 0, 0], torch.tensor(-1.0, dtype=DTYPE)
                )


class TestOverwriteNonZeroDim(unittest.TestCase):
    """Coverage for dim != 0 — the fix should not be axis-specific."""

    def test_loop_dim1(self):
        cache = _make_cache((4, 8, 64))
        slots = [1, 5, 7, 2]
        for i, slot in enumerate(slots):
            tok = _make_chunk((4, 1, 64), i + 1)
            torch.ops.spyre.overwrite(tok, cache, [1], [slot])
        out = cache.to("cpu")
        for i, slot in enumerate(slots):
            torch.testing.assert_close(
                out[0, slot, 0], torch.tensor(float(i + 1), dtype=DTYPE)
            )

    def test_loop_multi_dim(self):
        cache = _make_cache((8, 8, 64))
        # Write a 1x1x64 chunk at varying (dim0, dim1) coordinates.
        coords = [(1, 2), (3, 5), (7, 0), (4, 6)]
        for i, (d0, d1) in enumerate(coords):
            tok = _make_chunk((1, 1, 64), i + 1)
            torch.ops.spyre.overwrite(tok, cache, [0, 1], [d0, d1])
        out = cache.to("cpu")
        for i, (d0, d1) in enumerate(coords):
            torch.testing.assert_close(
                out[d0, d1, 0], torch.tensor(float(i + 1), dtype=DTYPE)
            )


class TestOverwriteIssue1765(unittest.TestCase):
    """Reproducer from torch-spyre#1765: sequential overwrites at
    progressively increasing offsets accumulated divergence from the
    CPU reference (max_diff growing 4.4 → 5.3 across iterations 0–9).
    This is the multi-call manifestation of the dynamo specialize_int
    bug — each call writes to the first call's offset instead of its
    own, so the CPU-vs-Spyre diff at distinct positions compounds."""

    def test_issue_1765_sequential_overwrites_4d_dim2(self):
        torch.manual_seed(42)

        cache = torch.randn(1, 8, 64, 128, dtype=DTYPE)
        cache_sp = cache.to(DEVICE)
        ref = cache.clone()

        for i in range(10):
            new_val = torch.randn(1, 8, 1, 128, dtype=DTYPE)
            ref[:, :, i : i + 1, :] = new_val
            torch.ops.spyre.overwrite(
                input=new_val.to(DEVICE),
                output=cache_sp,
                dims=[2],
                offsets=[i],
            )

        out = cache_sp.to("cpu")
        max_diff = (ref - out).abs().max().item()
        # fp16 round-trip noise is ~2e-3; the issue reports max_diff
        # growing past 5.0 across iterations on the unfixed tree.
        self.assertLess(max_diff, 0.01, f"max_diff={max_diff}")


class TestOverwriteCpuFallback(unittest.TestCase):
    """Smoke check for the CPU kernel registration. The fix does not
    touch CPU behavior, but co-locating asserts the API contract is
    identical across devices so future refactors can't drift them.
    """

    def test_cpu_loop_writes_each_slot(self):
        cache = torch.full((16, 4, 32), -1.0, dtype=DTYPE)
        slots = [1, 7, 14]
        for i, slot in enumerate(slots):
            tok = torch.full((1, 4, 32), float(i + 1), dtype=DTYPE)
            torch.ops.spyre.overwrite(tok, cache, [0], [slot])
        for i, slot in enumerate(slots):
            torch.testing.assert_close(
                cache[slot, 0, 0], torch.tensor(float(i + 1), dtype=DTYPE)
            )


class TestOverwriteFunctionalReinplace(unittest.TestCase):
    """Exercise the spyre.overwrite_f → spyre.overwrite path that the
    inductor reinplace pass takes when overwrite_f is invoked inside
    a torch.compile'd function. Confirms the fix carries through
    reinplacing.
    """

    def test_overwrite_f_in_compiled_loop(self):
        slots = [3, 11, 7, 19]

        def scatter(cache, tokens, slots_tensor):
            # tokens: (N, ...); slots_tensor: (N,) ints — but custom op
            # takes int list, so we unroll. Keep this simple: one
            # overwrite_f per slot, returning new cache each time.
            for i in range(len(slots)):
                cache = torch.ops.spyre.overwrite_f(
                    tokens[i : i + 1], cache, [0], [slots[i]]
                )
            return cache

        compiled_scatter = torch.compile(scatter)

        cache = _make_cache((32, 4, 64))
        tokens = torch.stack(
            [torch.full((4, 64), float(i + 1), dtype=DTYPE) for i in range(len(slots))]
        ).to(DEVICE)
        slots_t = torch.tensor(slots, dtype=torch.int64).to(DEVICE)

        result = compiled_scatter(cache, tokens, slots_t)
        out = result.to("cpu")
        for i, slot in enumerate(slots):
            torch.testing.assert_close(
                out[slot, 0, 0], torch.tensor(float(i + 1), dtype=DTYPE)
            )

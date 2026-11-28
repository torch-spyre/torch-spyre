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

"""The correction upload of a symbolic-count kernel follows the trip count.

A kernel compiled over a declared dimension carries one program-correction
slot per candidate iteration, ``max / granularity`` of them, and the host
fills the whole table on every launch. The device reads slot k at iteration k
only, so the launch uploads the address table plus the first n slot copies and
leaves the rest of the region as it is (``JobPlanStepHostCompute`` in
``csrc/job_plan.cpp``). What the scatter reads past a copy's own flits goes to
dummy addresses by construction, so the stale bytes beyond slot n never reach
the program.

WHAT THIS ASSERTS. A sequence of launches with a fresh declared tensor per
step (new buffers at new addresses, as a serving plugin allocates its page
tables per step) and trip counts going up and down, the maximum included, gives
exactly the values of the same body compiled as a static kernel at that size,
and the CPU's values within fp16 tolerance, for a one-op body, a two-op body, a
matmul body and a loop with a static kernel beside it. More sequences cover a
fresh kernel whose first launch is below the maximum, so most of its correction
region has never been written, and an attention whose query and key loops are
nested over one declared size. The counters confirm that the loop corrections
below the maximum were cut and that a static correction stays whole, as does a
loop correction above 1 MiB. A host-only test feeds canned metadata to the
gate's input.

The body's ops share one correction step per loop (a pointwise kernel and an
extern matmul inside one loop give one step), and a kernel beside the loop is
a step of its own that never reads the dimension slot.
"""

import json
import unittest

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._C import _correction_reads_inputs, _correction_upload_stats
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME

COLS = 64
TILE = 64
MAX_TILES = 32
MAX_ROWS = MAX_TILES * TILE
SPEC = {"min": TILE, "max": MAX_ROWS, "granularity": TILE}
# Trip counts going up and down: the maximum twice, so a launch below it
# follows one that filled every slot, and the minimum in between.
TILES = [32, 4, 1, 9, 2, 32, 3, 1, 16, 5]
BELOW_MAX = sum(1 for c in TILES if c < MAX_TILES)
# A fresh kernel whose first launches run below the maximum, so the copies past
# them have never been written when the program first reads its prefix.
TILES_FIRST_BELOW_MAX = [1, 4, 2, MAX_TILES, 3]
ATOL = 1e-2
RTOL = 1e-2


def _checks(x):
    rows = x.shape[0]
    torch._check(rows >= TILE)
    torch._check(rows <= MAX_ROWS)
    torch._check(rows % TILE == 0)


def tiled_gelu(x):
    _checks(x)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.nn.functional.gelu(tile)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


def tiled_matmul(x, w):
    _checks(x)

    def body(_carry, tiles):
        tile, w_whole = tiles
        return None, torch.matmul(tile, w_whole)

    _carry, out = for_each_tile(body, (x, w), dims=(0, None), tile_size=TILE, out_dim=0)
    return out


def tiled_gelu_matmul(x, w):
    """Two ops in the body: a pointwise kernel and an extern matmul."""
    _checks(x)

    def body(_carry, tiles):
        tile, w_whole = tiles
        return None, torch.matmul(torch.nn.functional.gelu(tile), w_whole)

    _carry, out = for_each_tile(body, (x, w), dims=(0, None), tile_size=TILE, out_dim=0)
    return out


def tiled_gelu_with_static(x, w):
    """The loop plus a static kernel beside it, whose correction stays whole."""
    return tiled_gelu(x), torch.nn.functional.gelu(w) * 2.0


def tiled_tanh(x):
    """A body no other test compiles, so its kernel and correction region are
    new when the first-launch test starts."""
    _checks(x)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.tanh(tile)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


# Above 1 MiB a loop correction uploads whole (kMaxPrefixCorrectionBytes in
# csrc/job_plan.h): a partial upload of a larger region measured slower than
# the whole one. The one-op body carries 640 bytes per copy, so 2048 tiles put
# its correction above the bound.
BIG_TILES = 2048
BIG_SPEC = {"min": TILE, "max": BIG_TILES * TILE, "granularity": TILE}


def tiled_tanh_big(x):
    rows = x.shape[0]
    torch._check(rows >= TILE)
    torch._check(rows <= BIG_TILES * TILE)
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.tanh(tile)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


# The nested case: an online-softmax attention whose query loop and key loop
# both walk one declared size. Its loop correction is laid out per query
# iteration, each holding every key iteration.
NESTED_MAX_TILES = 8
NESTED_SPEC = {"min": TILE, "max": NESTED_MAX_TILES * TILE, "granularity": TILE}
NESTED_TILES = [NESTED_MAX_TILES, 2, 1, 5, NESTED_MAX_TILES, 3]
SCALE = 1.0 / 8.0


def _online_step(carry, q_t, k_t, v_t):
    m, s, acc = carry
    scores = torch.matmul(q_t, k_t.transpose(0, 1)) * SCALE
    s_max = torch.amax(scores, dim=-1, keepdim=True)
    rescale = torch.exp(-torch.relu(s_max - m))
    m_new = torch.maximum(m, s_max)
    p = torch.exp(scores - m_new)
    return (
        m_new,
        s * rescale + p.sum(dim=-1, keepdim=True),
        acc * rescale + torch.matmul(p, v_t),
    )


def _nested_attention(q, k, v):
    def query_body(_carry, tiles):
        q_t, k_whole, v_whole = tiles

        def key_body(carry, key_tiles):
            k_t, v_t, q_tile = key_tiles
            return _online_step(carry, q_tile, k_t, v_t), None

        init = (
            torch.full((TILE, 1), float("-inf"), dtype=q_t.dtype, device=q_t.device),
            torch.zeros((TILE, 1), dtype=q_t.dtype, device=q_t.device),
            torch.zeros((TILE, COLS), dtype=q_t.dtype, device=q_t.device),
        )
        (m, s, acc), _ = for_each_tile(
            key_body,
            (k_whole, v_whole, q_t),
            dims=(0, 0, None),
            tile_size=TILE,
            init=init,
        )
        return None, acc / s

    return for_each_tile(
        query_body, (q, k, v), dims=(0, None, None), tile_size=TILE, out_dim=0
    )[1]


def tiled_nested_attention(q, k, v):
    rows = q.shape[0]
    torch._check(rows >= TILE)
    torch._check(rows <= NESTED_MAX_TILES * TILE)
    torch._check(rows % TILE == 0)
    torch._check(k.shape[0] == rows)
    torch._check(v.shape[0] == rows)
    return _nested_attention(q, k, v)


def ref_attention(q, k, v):
    scores = torch.matmul(q.float(), k.float().transpose(0, 1)) * SCALE
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def static_gelu(x):
    return torch.nn.functional.gelu(x)


def static_matmul(x, w):
    return torch.matmul(x, w)


def static_gelu_matmul(x, w):
    return torch.matmul(torch.nn.functional.gelu(x), w)


def ref_gelu(host, rows, _w):
    return torch.nn.functional.gelu(host[:rows].float())


def ref_matmul(host, rows, w):
    return torch.matmul(host[:rows].float(), w.float())


def ref_gelu_matmul(host, rows, w):
    return torch.matmul(torch.nn.functional.gelu(host[:rows].float()), w.float())


# An fp16 matmul over 64 terms sits up to about 0.12 from the fp32 reference on
# outputs near 40 (measured), so the CPU comparison is a sanity bound and the
# exact check is against the static device kernel.
MATMUL_ATOL = 0.25


def _fresh_declared(rows, seed):
    """A new host tensor, declared at the maximum and resized to ``rows``."""
    generator = torch.Generator().manual_seed(seed)
    host = torch.randn(MAX_ROWS, COLS, dtype=torch.float16, generator=generator)
    on_device = host.to(DEVICE_NAME, dynamic={0: SPEC})
    on_device.resize_((rows, COLS))
    return host, on_device


def _delta(before, after):
    return {key: after[key] - before[key] for key in before}


class TestCorrectionUploadFollowsTheCount(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        # The static reference kernel recompiles once per distinct size.
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, name):
                setattr(torch._dynamo.config, name, 64)
        self.w = torch.randn(
            COLS, COLS, dtype=torch.float16, generator=torch.Generator().manual_seed(7)
        )

    def _launch(
        self, fn, static_fn, ref, with_w, static_ref=None, atol=ATOL, tiles=TILES
    ):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        static = torch.compile(static_fn, backend="inductor", dynamic=False)
        w_dev = self.w.to(DEVICE_NAME) if with_w else None
        below_max = sum(1 for c in tiles if c < MAX_TILES)
        before = _correction_upload_stats()
        for step, count in enumerate(tiles):
            rows = count * TILE
            host, on_device = _fresh_declared(rows, seed=1000 + step)
            out = compiled(on_device, w_dev) if with_w else compiled(on_device)
            if static_ref is not None:
                out, static_out = out
                torch.testing.assert_close(
                    static_out.cpu().float(), static_ref(self.w), atol=ATOL, rtol=RTOL
                )
            plain = host[:rows].to(DEVICE_NAME)
            expected = static(plain, w_dev) if with_w else static(plain)
            if static_ref is not None:
                expected = expected[0]
            # Exactly what the same body gives as a static kernel at this size:
            # the cut upload changes nothing the device reads.
            self.assertTrue(
                torch.equal(out.cpu(), expected.cpu()),
                f"step {step} at {count} tiles differs from the static kernel",
            )
            torch.testing.assert_close(
                out.cpu().float(),
                ref(host, rows, self.w),
                atol=atol,
                rtol=RTOL,
                msg=f"step {step} at {count} tiles",
            )
        torch.spyre.synchronize()
        delta = _delta(before, _correction_upload_stats())
        # Every launch below the maximum cuts each loop correction step once,
        # and a launch at the maximum cuts nothing.
        self.assertGreaterEqual(delta["launches"], len(tiles))
        self.assertGreaterEqual(delta["prefixed"], below_max)
        self.assertEqual(delta["prefixed"] % below_max, 0)
        self.assertLess(delta["bytes_sent"], delta["bytes_full"])
        return delta

    def test_a_fresh_kernel_launched_first_below_the_maximum(self):
        # Review item: the other sequences start at the maximum, so a launch
        # below it always follows one that wrote every copy. Here the first
        # launches leave most of the region as the allocator gave it.
        self._launch(
            tiled_tanh,
            torch.tanh,
            lambda host, rows, _w: torch.tanh(host[:rows].float()),
            with_w=False,
            tiles=TILES_FIRST_BELOW_MAX,
        )

    def test_nested_loops_over_one_declared_size(self):
        # Query and key loops share the one dimension argument, so the gate
        # cuts the upload to the query iterations that run, each with every key
        # iteration it holds.
        compiled = torch.compile(
            tiled_nested_attention, backend="inductor", fullgraph=True
        )
        # The same function compiled for each size. Without fullgraph a static
        # compile of these nested loops fails with DataDependentOutputException.
        static = torch.compile(
            tiled_nested_attention, backend="inductor", fullgraph=True, dynamic=False
        )
        before = _correction_upload_stats()
        for step, count in enumerate(NESTED_TILES):
            rows = count * TILE
            generator = torch.Generator().manual_seed(2000 + step)
            hosts = [
                torch.randn(
                    NESTED_MAX_TILES * TILE,
                    COLS,
                    dtype=torch.float16,
                    generator=generator,
                )
                for _ in range(3)
            ]
            declared = [h.to(DEVICE_NAME, dynamic={0: NESTED_SPEC}) for h in hosts]
            for t in declared:
                t.resize_((rows, COLS))
            out = compiled(*declared)
            expected = static(*[h[:rows].to(DEVICE_NAME) for h in hosts])
            self.assertTrue(
                torch.equal(out.cpu(), expected.cpu()),
                f"step {step} at {count} tiles differs from the static kernel",
            )
            torch.testing.assert_close(
                out.cpu().float(),
                ref_attention(*[h[:rows] for h in hosts]),
                atol=2e-2,
                rtol=RTOL,
                msg=f"step {step} at {count} tiles",
            )
        torch.spyre.synchronize()
        delta = _delta(before, _correction_upload_stats())
        nested_below_max = sum(1 for c in NESTED_TILES if c < NESTED_MAX_TILES)
        self.assertGreaterEqual(delta["prefixed"], nested_below_max)
        self.assertLess(delta["bytes_sent"], delta["bytes_full"])

    def test_a_correction_above_the_bound_uploads_whole(self):
        with spyre_config.patch({"max_buckets": BIG_TILES}):
            compiled = torch.compile(tiled_tanh_big, backend="inductor", fullgraph=True)
            before = _correction_upload_stats()
            for step, count in enumerate((2, 5)):
                rows = count * TILE
                host = torch.randn(
                    BIG_TILES * TILE,
                    COLS,
                    dtype=torch.float16,
                    generator=torch.Generator().manual_seed(3000 + step),
                )
                on_device = host.to(DEVICE_NAME, dynamic={0: BIG_SPEC})
                on_device.resize_((rows, COLS))
                out = compiled(on_device)
                torch.testing.assert_close(
                    out.cpu().float(),
                    torch.tanh(host[:rows].float()),
                    atol=ATOL,
                    rtol=RTOL,
                    msg=f"step {step} at {count} tiles",
                )
            torch.spyre.synchronize()
        delta = _delta(before, _correction_upload_stats())
        # The correction really is above the bound, and nothing was cut.
        self.assertGreater(delta["bytes_full"], delta["launches"] * (1 << 20))
        self.assertEqual(delta["prefixed"], 0)
        self.assertEqual(delta["bytes_sent"], delta["bytes_full"])

    def test_one_op_body(self):
        delta = self._launch(tiled_gelu, static_gelu, ref_gelu, with_w=False)
        self.assertEqual(delta["prefixed"], BELOW_MAX * self._loop_steps(delta))

    def test_two_op_body_shares_one_cut_loop_step(self):
        delta = self._launch(
            tiled_gelu_matmul,
            static_gelu_matmul,
            ref_gelu_matmul,
            with_w=True,
            atol=MATMUL_ATOL,
        )
        # A pointwise kernel and an extern matmul inside one loop share one
        # correction step, so the launches below the maximum cut as many steps
        # as the one-op body's do.
        one_op = self._launch(tiled_gelu, static_gelu, ref_gelu, with_w=False)
        self.assertEqual(self._loop_steps(delta), self._loop_steps(one_op))

    def test_matmul_body(self):
        self._launch(
            tiled_matmul, static_matmul, ref_matmul, with_w=True, atol=MATMUL_ATOL
        )

    def test_a_static_kernel_beside_the_loop_stays_whole(self):
        delta = self._launch(
            tiled_gelu_with_static,
            lambda x, w: (static_gelu(x), torch.nn.functional.gelu(w) * 2.0),
            ref_gelu,
            with_w=True,
            static_ref=lambda w: torch.nn.functional.gelu(w.float()) * 2.0,
        )
        loop_only = self._launch(tiled_gelu, static_gelu, ref_gelu, with_w=False)
        # The static kernel adds correction steps to every launch and cuts none
        # of them, so the uncut steps per launch go up and the cut ones do not.
        self.assertGreater(
            self._whole_steps_per_launch(delta), self._whole_steps_per_launch(loop_only)
        )
        self.assertEqual(delta["prefixed"], loop_only["prefixed"])

    @staticmethod
    def _loop_steps(delta):
        return delta["prefixed"] // BELOW_MAX

    @staticmethod
    def _whole_steps_per_launch(delta):
        return (delta["launches"] - delta["prefixed"]) / len(TILES)


class TestTheGateReadsTheMetadata(unittest.TestCase):
    """Host only: which input slots a correction's definitions read."""

    def test_the_loop_step_reads_the_dimension_slot(self):
        # The shape of the loop correction's metadata on a declared kernel:
        # two address slots and the dimension slot, read once by the trip-count
        # definition.
        hcm = {
            "vdci": {
                "inputSym_": [-3, -4, -78],
                "variableDefs_": [
                    ["V-143", -9, "448", "128", "1", "V-3"],
                    ["V-142", -9, "448", "128", "1", "V-4"],
                    ["V-138", -19, "V-78", "256"],
                ],
            }
        }
        self.assertEqual(_correction_reads_inputs(json.dumps(hcm)), [True, True, True])

    def test_a_step_that_reads_addresses_only(self):
        hcm = {
            "vdci": {
                "inputSym_": [-3, -4, -78],
                "variableDefs_": [["V-120", -9, "64", "128", "1", "V-3"]],
            }
        }
        self.assertEqual(
            _correction_reads_inputs(json.dumps(hcm)), [True, False, False]
        )

    def test_metadata_without_the_fields_gives_nothing(self):
        for hcm in ({}, {"vdci": {}}, {"vdci": {"inputSym_": [-3]}}):
            self.assertEqual(_correction_reads_inputs(json.dumps(hcm)), [])

    def test_a_string_symbol_is_matched_as_written(self):
        hcm = {"vdci": {"inputSym_": ["V-5"], "variableDefs_": [["V-9", -5, "V-5"]]}}
        self.assertEqual(_correction_reads_inputs(json.dumps(hcm)), [True])


if __name__ == "__main__":
    unittest.main()

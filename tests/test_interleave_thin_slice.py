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

# Owner(s): ["module: cpp"]

"""
Copy-only thin slice: an interleaved tensor round-tripping on 1p0-PF.

It drives an **interleaved** (multi-chunk) device allocation through the
**eager** copy path — ``x.to("spyre")`` (H2D) then ``.cpu()`` (D2H) — under
a synthetic multi-domain topology, and requires the round-tripped values to
match a ``Bind{0}`` baseline exactly.

Deliberately **no compute op and no torch.compile**: the compiled JobPlan
builder still asserts single-chunk addresses, and interleaved compute is not
expressible in the 1p0 8-segment model at all. Copy-only is what the emulation
proves on this silicon.

## What "exact" means here, and why dtype matters

The copy path is byte-lossless, but it is not *value*-lossless for every dtype,
because H2D/D2H also convert between the host and device encodings of the dtype
(``generate_dci`` → ``deeptools::ConvertData``):

* ``float32`` is ``IEEE_FP32`` on **both** sides (``types_mapping.h``), so the
  conversion is a pure layout shuffle and the round trip is **bit-exact**. The
  exactness cases below are all fp32 for this reason.
* ``float16`` is ``IEEE_FP16`` on the host but ``SEN169_FP16`` on the device — a
  different 16-bit encoding, with one fewer mantissa bit. A round trip therefore
  rounds, and ``torch.equal`` fails for arbitrary values. That is a property of
  the dtype conversion, **not** of interleaving: it happens identically on the
  single-chunk ``Bind{0}`` path.

So fp16 coverage asserts the two things that are actually meaningful for it —
digest parity with the ``Bind{0}`` baseline, and agreement with the host values
to within one encoding step — rather than bit-exactness. A scatter/gather bug
would break both.

## Running this test

The synthetic topology comes from flex env knobs that are read once when the
device runtime starts, so they must be set **on the pytest command line**:

```bash
FLEX_NUM_MEMORY_DOMAINS=4 FLEX_MAX_REGIONS=8 \
    pytest tests/test_interleave_thin_slice.py -v
```

``FLEX_MAX_REGIONS`` is not optional: ``max_regions`` must be divisible by the
domain count, and the 1p0 default of 7 is not. Without these the test **skips**,
which is what happens in a default ``pytest tests/`` sweep.

Interleaved placement itself is requested in-process via a temporary gate
(``TORCH_SPYRE_EMULATE_INTERLEAVE`` / ``_spyre_debug_set_emulate_interleave``).

Both placements are exercised in **one** process, on purpose: only one process
may hold the Spyre device (see ``torch_spyre/__init__.py`` — "Spyre can't be
used by more than one process"), so a subprocess baseline is not possible. The
upside is a tighter control — the baseline differs from the interleaved run only
in placement, not in topology.
"""

import contextlib
import hashlib
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

SEED = 20260815

# kMaxBufSize in flex's PF DMA path: each chunk larger than this must itself be
# split into multiple transfer buffers.
K_MAX_BUF_SIZE = 4 * 1024 * 1024

# One SEN169_FP16 mantissa step, relative. Round-to-nearest keeps the error at or
# below half of this; the bound is deliberately one full step so the assertion
# tests "same values, re-encoded" and not the rounding mode.
SEN169_FP16_STEP = 2.0**-9

# (case name, shape, dtype, bit_exact) — sizes chosen to stress the flex
# multi-chunk DMA path:
#  * "aligned"    — device size divides evenly across the domains.
#  * "one_stick" / "uneven_sticks" — the per-domain share is NOT a multiple of
#    the 128 B device alignment, so flex rounds each chunk up and total_size()
#    exceeds the tensor's storage size.
#  * "large"      — 24 MiB, i.e. 6 MiB per chunk at 4 domains, so every chunk
#    must itself be split into multiple <= kMaxBufSize transfers.
#
# `bit_exact` is False exactly where the host and device encodings of the dtype
# differ (fp16: IEEE_FP16 vs SEN169_FP16), which makes the *conversion* lossy
# regardless of placement — see the module docstring. fp32 is IEEE_FP32 on both
# sides, so those cases must round-trip bit-for-bit.
#
# Element counts differ between the fp32 and fp16 cases so that the byte sizes —
# which is what the DMA path actually partitions — hit the same interesting
# boundaries in both.
CASES = [
    ("one_stick", [32], torch.float32, True),
    ("aligned", [64, 64], torch.float32, True),
    ("uneven_sticks", [5, 32], torch.float32, True),
    ("large", [3072, 2048], torch.float32, True),
    ("half_aligned", [64, 64], torch.float16, False),
    ("half_uneven_sticks", [5, 64], torch.float16, False),
]

FILL_SHAPE = [64, 64]
FILL_DTYPE = torch.float16
# Exactly representable in both fp16 encodings, and non-zero so a chunk that was
# never filled cannot pass by happening to contain zeros. `torch.full` on spyre
# is `torch.empty` + the device-side FillDMA (see torch_spyre/ops/eager.py).
FILL_VALUE = 2.5


@contextlib.contextmanager
def _interleaved_placement(enabled: bool):
    """Force interleaved (or Bind{0}) placement for allocations in this block."""
    previous = torch.spyre._spyre_debug_set_emulate_interleave(enabled)
    try:
        yield
    finally:
        torch.spyre._spyre_debug_set_emulate_interleave(previous)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _max_relative_error(back: torch.Tensor, host: torch.Tensor) -> float:
    """Largest relative deviation, in units of the host magnitude.

    Exact zeros round-trip to exact zeros, so clamping the denominator only
    guards the division; it cannot mask a real difference.
    """
    a = back.to(torch.float32)
    b = host.to(torch.float32)
    return float(((a - b).abs() / b.abs().clamp_min(2.0**-14)).max().item())


def _round_trip(shape, dtype, interleaved: bool) -> dict:
    """Allocate + H2D + D2H one tensor and report what happened.

    Only the allocation and upload run under the placement gate; the download
    happens with the gate restored, so the round trip is driven by the tensor's
    own address rather than by the flag still being set.
    """
    torch.manual_seed(SEED)
    host = torch.randn(shape, dtype=torch.float32).to(dtype)

    with _interleaved_placement(interleaved):
        dev = host.to("spyre")

    chunks = torch.spyre._spyre_debug_composite_chunks(dev)
    back = dev.cpu()
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "host_bytes": host.untyped_storage().nbytes(),
        "chunks": chunks,
        "chunk_bytes_total": sum(c["size"] for c in chunks),
        "exact": bool(torch.equal(back, host)),
        "max_rel_err": _max_relative_error(back, host),
        "digest": _digest(back),
    }


def _fill_round_trip(interleaved: bool) -> dict:
    """Fill a device tensor (no host source) and read it back."""
    with _interleaved_placement(interleaved):
        dev = torch.full(FILL_SHAPE, FILL_VALUE, dtype=FILL_DTYPE, device="spyre")

    chunks = torch.spyre._spyre_debug_composite_chunks(dev)
    back = dev.cpu()
    return {
        "chunks": chunks,
        "all_filled": bool(torch.all(back == FILL_VALUE).item()),
        "digest": _digest(back),
    }


class TestInterleaveThinSlice(TestCase):
    """Copy-only interleaved round trip under an emulated multi-domain topology."""

    num_domains = 0
    interleaved: dict = {}
    baseline: dict = {}
    interleaved_fill: dict = {}
    baseline_fill: dict = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.spyre.is_initialized():
            torch.spyre._lazy_init()

        cls.num_domains = torch.spyre._spyre_debug_num_memory_domains()
        if cls.num_domains < 2:
            raise unittest.SkipTest(
                "needs an emulated multi-domain topology; flex reports "
                f"{cls.num_domains} memory domain(s). Re-run as: "
                "FLEX_NUM_MEMORY_DOMAINS=4 FLEX_MAX_REGIONS=8 pytest "
                "tests/test_interleave_thin_slice.py"
            )

        # The device work runs once and every test asserts on the result, so the
        # suite does not re-upload 24 MiB per assertion.
        cls.interleaved = {
            name: _round_trip(shape, dtype, interleaved=True)
            for name, shape, dtype, _ in CASES
        }
        cls.baseline = {
            name: _round_trip(shape, dtype, interleaved=False)
            for name, shape, dtype, _ in CASES
        }
        cls.interleaved_fill = _fill_round_trip(interleaved=True)
        cls.baseline_fill = _fill_round_trip(interleaved=False)

    def test_allocation_is_interleaved_across_all_domains(self):
        """Each tensor must be N chunks, one per domain, all device-addressable.

        Without this the numeric tests below would pass vacuously on a
        single-chunk allocation if the interleave gate ever stopped firing.
        """
        for name, case in self.interleaved.items():
            with self.subTest(case=name):
                chunks = case["chunks"]
                self.assertEqual(len(chunks), self.num_domains)
                # One chunk per domain, each in its own region.
                self.assertEqual(
                    sorted(c["domain_id"] for c in chunks),
                    list(range(self.num_domains)),
                )
                self.assertEqual(
                    len({c["region_id"] for c in chunks}), self.num_domains
                )
                # A region carrying UNMAPPED_SEGMENT_ID has no device address
                # and must never be dispatched.
                for chunk in chunks:
                    self.assertTrue(chunk["segment_mapped"], msg=str(chunk))
                # Chunks are equal-sized and cover the whole tensor.
                self.assertEqual(len({c["size"] for c in chunks}), 1)
                self.assertGreaterEqual(case["chunk_bytes_total"], case["host_bytes"])

    def test_baseline_allocation_is_single_chunk(self):
        """With the gate off, allocation must stay exactly Bind{0}."""
        for name, case in self.baseline.items():
            with self.subTest(case=name):
                self.assertEqual(len(case["chunks"]), 1)
                self.assertEqual(case["chunks"][0]["domain_id"], 0)

    def test_round_trip_is_exact(self):
        """H2D → D2H must be bit-identical where the dtype encoding is shared.

        fp32 is IEEE_FP32 on host and device, so the copy path converts layout
        only and nothing may change. Cases whose dtype is re-encoded on the
        device are covered by the reduced-precision test below instead.
        """
        exact_cases = [name for name, _, _, bit_exact in CASES if bit_exact]
        self.assertTrue(exact_cases, msg="no bit-exact case configured")
        for name in exact_cases:
            case = self.interleaved[name]
            with self.subTest(case=name):
                self.assertTrue(
                    case["exact"],
                    msg=(
                        f"{name} ({case['dtype']}): round trip differs; "
                        f"max relative error {case['max_rel_err']:.3e}"
                    ),
                )

    def test_reduced_precision_round_trip_is_within_one_encoding_step(self):
        """Re-encoded dtypes must still come back as the same values.

        fp16 is IEEE_FP16 on the host and SEN169_FP16 on the device — one
        mantissa bit narrower — so a round trip rounds and cannot be bit-exact.
        What must hold is that every element is still within one encoding step of
        what went in: a mis-ordered scatter/gather would produce unrelated values
        and miss this bound by orders of magnitude.
        """
        lossy_cases = [name for name, _, _, bit_exact in CASES if not bit_exact]
        self.assertTrue(lossy_cases, msg="no reduced-precision case configured")
        for name in lossy_cases:
            case = self.interleaved[name]
            with self.subTest(case=name):
                self.assertLessEqual(
                    case["max_rel_err"],
                    SEN169_FP16_STEP,
                    msg=f"{name} ({case['dtype']}): values changed, not just rounded",
                )

    def test_interleaved_matches_bind_baseline(self):
        """Interleaved round-trip values must equal the Bind{0} baseline.

        Stronger than the round-trip check on its own: it rules out a
        scatter/gather bug that is self-consistent (same wrong permutation both
        ways) but still disagrees with the trusted single-chunk path.
        """
        self.assertEqual(set(self.interleaved), set(self.baseline))
        for name, case in self.interleaved.items():
            with self.subTest(case=name):
                self.assertEqual(case["digest"], self.baseline[name]["digest"])

    def test_per_chunk_transfer_exceeds_buffer_size(self):
        """The "large" case must really put > 4 MiB in each chunk.

        This is the only case that exercises size-slice loop nested inside
        the per-chunk loop; assert the premise so the coverage cannot silently
        rot if the shape is edited.
        """
        for chunk in self.interleaved["large"]["chunks"]:
            self.assertGreater(chunk["size"], K_MAX_BUF_SIZE, msg=str(chunk))

    def test_multi_chunk_fill_then_read_back(self):
        """A filled interleaved tensor must carry the pattern in every chunk."""
        self.assertEqual(len(self.interleaved_fill["chunks"]), self.num_domains)
        self.assertTrue(self.interleaved_fill["all_filled"])
        self.assertEqual(self.interleaved_fill["digest"], self.baseline_fill["digest"])

    def test_layout_report(self):
        """Not an assertion — print the observed layout for the gap ledger."""
        for name, case in self.interleaved.items():
            print(
                f"\n{name} {case['dtype']} shape={case['shape']}"
                f" host_bytes={case['host_bytes']}"
                f" chunk_bytes_total={case['chunk_bytes_total']}"
                f" exact={case['exact']}"
                f" max_rel_err={case['max_rel_err']:.3e}"
            )
            for chunk in case["chunks"]:
                print(
                    f"  domain={chunk['domain_id']} region={chunk['region_id']}"
                    f" offset={chunk['offset']} size={chunk['size']}"
                    f" segment={chunk['segment_id']}"
                )


if __name__ == "__main__":
    run_tests()

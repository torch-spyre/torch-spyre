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

"""Reshapes that cut the stick dim off a stick boundary (issues #1353, #5214, #5320).

Layout propagation serves such a read with a dense layout that moves the stick to
a dim the reshape leaves alone (``dense_relocation_serves_read``), and only falls
back to the sparse layout, which costs 64 times the memory at fp16, when no dim
can take the stick. Every case runs on the device: none goes through the host.

Dense or sparse is told apart by the peak extra device memory of the compiled
call relative to its input: a dense relocation pads one dim to whole sticks
(1-2x here), the sparse layout stores one element per stick (64x).
"""

import unittest

import pytest
import torch
import torch._inductor.config as inductor_config
from torch._inductor.exc import InductorError

from utils_inductor import expects_raise

F = torch.nn.functional

# A dense relocation measured at most 2x the input on these shapes; the sparse
# layout measured 64x and more. 8x separates the two with room on either side.
DENSE_MAX_MEMORY_RATIO = 8.0
SPARSE_MIN_MEMORY_RATIO = 32.0


def _silu_reshape(out_shape):
    return lambda x: F.silu(x * 2).reshape(out_shape) + 1


def _compile_and_measure(fn, *shapes):
    """Compile ``fn`` on Spyre, check it against an fp32 CPU reference, and return
    the peak extra device memory of the compiled call relative to its inputs."""
    torch.manual_seed(0)
    xs = [torch.randn(*s, dtype=torch.float16) for s in shapes]
    ref = fn(*[x.float() for x in xs])
    dev = [x.to("spyre") for x in xs]
    mem = torch.accelerator.memory
    torch.spyre.synchronize()
    mem.reset_peak_memory_stats()
    before = mem.memory_allocated()
    torch.compiler.reset()
    with inductor_config.patch(force_disable_caches=True):
        out = torch.compile(fn, dynamic=False)(*dev)
    torch.spyre.synchronize()
    ratio = (mem.max_memory_allocated() - before) / sum(x.numel() * 2 for x in xs)
    outs = out if isinstance(out, tuple) else (out,)
    refs = ref if isinstance(ref, tuple) else (ref,)
    for o, r in zip(outs, refs):
        # fp16 SiLU/add on randn inputs; errors stay well under 5e-2.
        torch.testing.assert_close(o.cpu().float(), r.float(), atol=5e-2, rtol=1e-2)
    return ratio


@pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
class TestStickRelocation(unittest.TestCase):
    def _check_dense(self, cases):
        for name, (shapes, fn) in cases.items():
            with self.subTest(name):
                ratio = _compile_and_measure(fn, *shapes)
                self.assertLess(ratio, DENSE_MAX_MEMORY_RATIO, "fell back to sparse")

    def test_split_and_merge_unaligned_stick(self):
        # W=160 rows are 2.5 sticks. The merge used to compile and silently read
        # every row after the first from the wrong address.
        cases = {}
        for w in (160, 40, 20):
            cases[f"split w={w}"] = ([(1, 32, w * w)], _silu_reshape((1, 32, w, w)))
            cases[f"merge w={w}"] = ([(1, 32, w, w)], _silu_reshape((1, 32, w * w)))
        self._check_dense(cases)

    def test_stick_moved_to_a_wide_dim(self):
        # The stick moves to a dim wider than one stick, with nothing in front of
        # it. This gave wrong values when the padded former stick dim was not
        # outermost.
        cases = {
            "64 alone": ([(64, 3, 7)], lambda x: x.reshape(64, 21) * 2),
            "128 alone": ([(128, 3, 7)], lambda x: x.reshape(128, 21) * 2),
            "128 merge": ([(128, 40, 40)], lambda x: x.reshape(128, 1600) * 2),
            "128 split": ([(128, 1600)], lambda x: x.reshape(128, 40, 40) * 2),
            "128 merge, batch 1": (
                [(1, 128, 40, 40)],
                lambda x: x.reshape(1, 128, 1600) * 2,
            ),
            "192 merge": ([(192, 20, 20)], lambda x: x.reshape(192, 400) * 2),
        }
        self._check_dense(cases)

    def test_stick_moved_to_a_partial_stick_dim(self):
        # 100 and 4 are padded to whole sticks, so these are not as compact as
        # the cases above, but they stay off the sparse layout.
        cases = {
            "100 alone": ([(100, 3, 7)], lambda x: x.reshape(100, 21) * 2),
            "100 with an outer dim": (
                [(3, 100, 3, 7)],
                lambda x: x.reshape(3, 100, 21) * 2,
            ),
            "sub-stick 4x64": ([(4, 64)], lambda x: x.reshape(4, 8, 8) * 2),
        }
        for name, (shapes, fn) in cases.items():
            with self.subTest(name):
                ratio = _compile_and_measure(fn, *shapes)
                self.assertLess(ratio, SPARSE_MIN_MEMORY_RATIO, "fell back to sparse")

    def test_two_outer_dims_with_padded_stick(self):
        # Two or more dims in front of a padded former stick dim (400 is 6.25
        # sticks). The first case is the yolo26 PSA attention output view.
        cases = {
            "yolo26 psa output": ((1, 4, 64, 400), (1, 256, 20, 20)),
            "split only": ((1, 4, 64, 400), (1, 4, 64, 20, 20)),
            "no size-1 dim": ((2, 4, 64, 400), (2, 4, 64, 20, 20)),
            "merge back": ((2, 4, 64, 20, 20), (2, 4, 64, 400)),
            "narrow row": ((1, 4, 64, 160), (1, 4, 64, 2, 80)),
        }
        self._check_dense(
            {name: ([s], _silu_reshape(out)) for name, (s, out) in cases.items()}
        )

    def test_yolo26_merges(self):
        # The yolo26 merges at batch 4 exceed the per-core span limit with the
        # sparse layout (320 and 400 MiB restickifies).
        cases = {
            "psa qkv, batch 1": ((1, 512, 20, 20), (1, 4, 128, 400)),
            "psa qkv, batch 4": ((4, 512, 20, 20), (4, 4, 128, 400)),
            "detect view, batch 4": ((4, 80, 80, 80), (4, 80, 6400)),
        }
        self._check_dense(
            {name: ([s], _silu_reshape(out)) for name, (s, out) in cases.items()}
        )

    def test_aligned_reshape(self):
        self._check_dense(
            {"aligned": ([(1, 32, 128 * 128)], _silu_reshape((1, 32, 128, 128)))}
        )

    def test_no_free_dim_uses_the_sparse_layout(self):
        # No dim the reshape leaves alone can take the stick, so the read goes
        # to the sparse layout, still on the device.
        cases = {
            "one stick to 8x8": ([(64,)], lambda x: x.reshape(8, 8) * 2),
            "1x2x4x8x7": ([(1, 2, 4, 8, 7)], lambda x: x.reshape(1, 8, 56) * 2),
            "six dims": ([(1, 2, 4, 8, 4, 7)], lambda x: x.reshape(1, 8, 224) * 2),
            # TODO: the outer dims are merged as well (2 x 16 -> 32), so no single
            # dim is left alone. The merged dim could take the stick, but the
            # restickify into it iterates over the two host dims and writes the
            # stick coordinate 16*d0 + d1, which codegen rejects. When that is
            # supported this case should become dense.
            "outer dims merged too": (
                [(2, 16, 25600)],
                _silu_reshape((32, 160, 160)),
            ),
        }
        for name, (shapes, fn) in cases.items():
            with self.subTest(name):
                ratio = _compile_and_measure(fn, *shapes)
                self.assertGreater(ratio, SPARSE_MIN_MEMORY_RATIO)

    def test_escaping_view_with_in_graph_consumer(self):
        # The view escapes and also feeds another op. Nothing is rewritten in the
        # graph, so the view stays a view of ``y``.
        def fn(x):
            y = F.silu(x * 2)
            r = y.reshape(1, 32, 25600)
            return y, r, r * 3

        _compile_and_measure(fn, (1, 32, 160, 160))

    def test_escaping_alias_is_kept(self):
        # A reshape the caller sees next to its base must stay an alias of it.
        cases = {
            "base and view": lambda y: (y, y.view(4, 8, 8)),
            "sibling view and view": lambda y: (y.t(), y.view(4, 8, 8)),
            "base and view of view": lambda y: (y, y.view(4, 8, 8).transpose(1, 2)),
        }
        for name, tail in cases.items():
            with self.subTest(name):
                torch.compiler.reset()
                with inductor_config.patch(force_disable_caches=True):
                    first, second = torch.compile(
                        lambda x, tail=tail: tail(x * 2), dynamic=False
                    )(torch.ones(4, 64, dtype=torch.float16).to("spyre"))
                first.fill_(7)
                self.assertEqual(second.cpu().flatten()[:4].tolist(), [7.0] * 4)

    @expects_raise
    def test_stick_size_change_is_rejected(self):
        # TODO(#3999): float16 -> float32 changes the stick size, and the backend
        # reorders elements across that conversion, so these are refused instead
        # of returning wrong values. Make this a positive test when #3999 lands.
        cases = {
            "aligned target": ((2, 8, 40), lambda t: t.reshape(2, 320).float()),
            "split then float": (
                (1, 64, 25600),
                lambda t: t.reshape(1, 64, 160, 160).float(),
            ),
        }
        for name, (shape, fn) in cases.items():
            with self.subTest(name):
                torch.compiler.reset()
                with (
                    inductor_config.patch(force_disable_caches=True),
                    pytest.raises(InductorError, match="no supported output layout"),
                ):
                    torch.compile(fn, dynamic=False)(
                        torch.randn(*shape, dtype=torch.float16).to("spyre")
                    )

    @expects_raise
    def test_cat_of_sparse_reads_along_the_stick_dim_is_rejected(self):
        # TODO(#6163): the reshape merges every dim, so no dim is left to take the
        # stick and the read takes the sparse layout; a cat along the stick dim of
        # sparse operands cannot be expressed. Make this a positive test when it
        # can.
        a = torch.randn(1, 2, 4, 8, 7, dtype=torch.float16).to("spyre")
        torch.compiler.reset()
        with (
            inductor_config.patch(force_disable_caches=True),
            pytest.raises(InductorError, match="Unexpected stick expression"),
        ):
            torch.compile(
                lambda a, b: torch.cat([a.reshape(1, 8, 56), b.reshape(1, 8, 56)], -1)
                * 2,
                dynamic=False,
            )(a, a.clone())

    @expects_raise
    def test_cat_of_reshaped_reads_is_rejected(self):
        # TODO(#6163): each cat operand is restickified straight into its slice
        # of the cat buffer, where its old-stick dim cannot be padded in place.
        # Restickify into a fresh buffer and copy it into the slice, then make this
        # a positive test. The yolo26 Detect head (three views, then a cat along
        # the stick dim) stops at the same point.
        cases = {
            "cat of two reshapes": (
                [(1, 2, 4, 8, 7)] * 2,
                lambda a, b: torch.cat([a.reshape(1, 8, 56), b.reshape(1, 8, 56)], 1)
                * 2,
            ),
            "yolo26 detect head": (
                [(1, 80, 80, 80), (1, 80, 40, 40), (1, 80, 20, 20)],
                lambda a, b, c: torch.cat([t.reshape(1, 80, -1) for t in (a, b, c)], -1)
                * 2,
            ),
        }
        for name, (shapes, fn) in cases.items():
            with self.subTest(name):
                torch.compiler.reset()
                with (
                    inductor_config.patch(force_disable_caches=True),
                    pytest.raises(
                        InductorError, match="written into a slice of another buffer"
                    ),
                ):
                    torch.compile(fn, dynamic=False)(
                        *[
                            torch.randn(*s, dtype=torch.float16).to("spyre")
                            for s in shapes
                        ]
                    )

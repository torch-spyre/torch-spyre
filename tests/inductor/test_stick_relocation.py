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


def _reduce_after(view, kind):
    """``view`` of the input, times two, reduced along its last dim by ``kind``."""
    reduce = {
        "sum": lambda t: t.sum(-1),
        "amax": lambda t: t.amax(-1),
        "argmax": lambda t: t.argmax(-1),
    }[kind]
    return lambda a: reduce(view(a) * 2)


def _compile_and_measure(fn, *shapes, dtype=torch.float16):
    """Compile ``fn`` on Spyre, check it against an fp32 CPU reference, and return
    the peak extra device memory of the compiled call relative to its inputs."""
    torch.manual_seed(0)
    if dtype.is_floating_point:
        xs = [torch.randn(*s, dtype=dtype) for s in shapes]
    else:
        xs = [torch.randint(-100, 100, s, dtype=dtype) for s in shapes]
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
    ratio = (mem.max_memory_allocated() - before) / sum(
        x.numel() * x.element_size() for x in xs
    )
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

    def test_conv_output_reshape_read_by_another_op(self):
        # #5227: conv2d_via_bmm_decomp ends with (N, C, H*W) -> (N, C, H, W), so a
        # conv whose output width is not a multiple of 64 followed by an
        # activation failed to compile; YOLOv5n hits it at widths 160 and 80.
        # Here the consumer reads the reshaped view, unlike the cases above,
        # where the pointwise runs before the reshape and the view is of its
        # output. The issue reports two different errors: "Unsupported coordinate
        # expression 5*c1/2" for a row wider than a stick (160, 96, 80), and "no
        # supported output layout found" for one narrower than a stick (40).
        cases = {}
        for w in (160, 96, 80, 40):
            cases[f"consumer reads the reshape w={w}"] = (
                [(1, 32, w * w)],
                lambda x, w=w: F.silu(x.reshape(1, 32, w, w)),
            )
            cases[f"pointwise before the reshape w={w}"] = (
                [(1, 32, w * w)],
                lambda x, w=w: (x * 2).reshape(1, 32, w, w),
            )
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
            # TODO(#6558): the outer dims are merged as well (2 x 16 -> 32), so
            # no single dim is left alone and both candidates decline at the
            # coordinate stage. The merged pair could carry the stick between
            # them, which needs a restickify able to write the stick coordinate
            # 16*d0 + d1. When that is supported this case should become dense.
            "outer dims merged too": (
                [(2, 16, 25600)],
                _silu_reshape((32, 160, 160)),
            ),
        }
        for name, (shapes, fn) in cases.items():
            with self.subTest(name):
                ratio = _compile_and_measure(fn, *shapes)
                self.assertGreater(ratio, SPARSE_MIN_MEMORY_RATIO)

    def test_reduction_kinds_over_a_reshaped_read(self):
        # _reduction_layout_from_stl derives its output candidates from the
        # relocated layout, and it takes the same path for every reduction kind,
        # so sum alone does not cover it. amax and argmax are checked here on a
        # read with no dim to take the stick (the flatten of [8, 65], which stays
        # sparse) and on one that takes the dense relocation.
        reads = {
            "flatten [8,65] (sparse)": ((8, 65), lambda a: a.reshape(-1)),
            "merge (1,32,20,20) (dense)": (
                (1, 32, 20, 20),
                lambda a: a.reshape(1, 32, 400),
            ),
        }
        for read_name, (shape, view) in reads.items():
            torch.manual_seed(0)
            # Positive values: summing 400 signed fp16 terms cancels, and a row
            # whose total lands near zero has no meaningful relative error.
            x = torch.randn(*shape, dtype=torch.float16).abs() + 0.05
            for kind in ("sum", "amax", "argmax"):
                with self.subTest(f"{read_name} {kind}"):
                    whole = _reduce_after(view, kind)
                    torch.compiler.reset()
                    with inductor_config.patch(force_disable_caches=True):
                        out = torch.compile(whole, dynamic=False)(x.to("spyre")).cpu()
                    torch.spyre.synchronize()
                    if kind == "argmax":
                        # The index is checked by the value it points at, against
                        # the same data the device saw: the device holds fp16 in
                        # DLFloat16, one mantissa bit narrower, so a host-side
                        # reference can rank two values the device sees as equal.
                        # An index into a tie is a correct answer; one into a
                        # smaller value is not.
                        seen = view(x.to("spyre").cpu() * 2).float()
                        rows = seen.reshape(-1, seen.shape[-1])
                        picked = rows.gather(-1, out.reshape(-1, 1).long())
                        torch.testing.assert_close(
                            picked.reshape(-1), rows.max(-1).values
                        )
                    else:
                        ref = whole(x.float())
                        torch.testing.assert_close(
                            out.float(), ref.float(), atol=5e-2, rtol=1e-2
                        )

    def test_prod_does_not_relocate_onto_the_reduced_dim(self):
        # prod cannot reduce along the stick dim (backend #4409), and the widest
        # dim the reshape leaves alone is the reduced one, so the relocation must
        # skip it. The first three shapes have nothing else that can carry the
        # stick, so the read falls through to the sparse layout; the last two have
        # a second real dim and the stick moves there instead. Which dim it lands
        # on is pinned host-side by TestDenseRelocationServesRead in
        # tests/tensor/test_coordinates.py -- all this can see is the right value.
        # main compiles all of them; an earlier version of this branch failed the
        # first three with UnalignedStickSplit.
        cases = {
            "(1,32,20,20) reduce 32": ((1, 32, 20, 20), (1, 32, 400)),
            "(1,64,20,20) reduce 64": ((1, 64, 20, 20), (1, 64, 400)),
            "(2,8,20,20) reduce 8": ((2, 8, 20, 20), (2, 8, 400)),
            "(8,32,20,20) reduce 32, dim 0 free": ((8, 32, 20, 20), (8, 32, 400)),
            "(4,100,3,7) reduce 100, dim 0 free": ((4, 100, 3, 7), (4, 100, 21)),
        }
        for name, (shape, view) in cases.items():
            with self.subTest(name):
                torch.manual_seed(0)
                # Values near 1: a product of 32 signed fp16 terms overflows, and
                # a reference of inf cannot tell a right answer from a wrong one.
                x = torch.rand(*shape, dtype=torch.float16) * 0.2 + 0.9
                fn = lambda t, v=view: t.reshape(v).prod(1)  # noqa: E731
                ref = fn(x.float())
                torch.compiler.reset()
                with inductor_config.patch(force_disable_caches=True):
                    out = torch.compile(fn, dynamic=False)(x.to("spyre"))
                torch.spyre.synchronize()
                torch.testing.assert_close(
                    out.cpu().float(), ref.float(), atol=5e-2, rtol=5e-2
                )

    def test_a_read_at_a_constant_offset_keeps_its_values(self):
        # A sliced read is left to the sparse layout. The relocated coordinates do
        # carry the slice offset on the stick (``d0 + 1``), but the restickify
        # copies the whole buffer and the consumer then reads that copy at
        # ``Mod(c0, 64)``, dropping the offset, so every element lands one stick
        # position off. (5, 448)[1:] came back 1767 of 1792 elements wrong with no
        # error raised; main gets it right through the sparse layout.
        cases = {
            "(5,448)[1:] -> (4,28,16)": ((5, 448), lambda t: t[1:].reshape(4, 28, 16)),
            "(5,448)[1:] -> (4,112,4)": ((5, 448), lambda t: t[1:].reshape(4, 112, 4)),
            # Without the slice the same reshape still takes the relocation.
            "(4,448) -> (4,28,16)": ((4, 448), lambda t: t.reshape(4, 28, 16)),
            # A slice whose reshape does not split the stick never needed it.
            "(5,448)[1:] -> (4,7,64)": ((5, 448), lambda t: t[1:].reshape(4, 7, 64)),
        }
        for name, (shape, view) in cases.items():
            with self.subTest(name):
                torch.manual_seed(0)
                x = torch.rand(*shape, dtype=torch.float16) + 0.5
                fn = lambda t, v=view: v(t) + 1  # noqa: E731
                # The reference in fp32: at fp16 the comparison threshold would sit
                # on the ULP of these values and a third of them would "fail" on
                # rounding alone.
                ref = fn(x.float())
                torch.compiler.reset()
                with inductor_config.patch(force_disable_caches=True):
                    out = torch.compile(fn, dynamic=False)(x.to("spyre"))
                torch.spyre.synchronize()
                torch.testing.assert_close(out.cpu().float(), ref, atol=5e-3, rtol=5e-3)

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

    def test_stick_size_change_after_reshape(self):
        # #6557: a conversion that changes the elements per stick reads a reshape
        # that cuts the stick dim. convert_element_type's layout rule returned no
        # candidate before the relocation or the sparse fallback was consulted,
        # so the cast failed with "no supported output layout" where the same
        # reshape without the cast compiled. It now takes the dense relocation,
        # as the reshape alone does. The first case is the issue's reproducer.
        cases = {
            "split then float": (
                [(1, 64, 25600)],
                lambda x: x.reshape(1, 64, 160, 160).float(),
            ),
            # The conversion's output read by another op, which reads the
            # relocated layout: pointwise, and reductions across and along the
            # split dims.
            "split then float then multiply": (
                [(1, 64, 25600)],
                lambda x: x.reshape(1, 64, 160, 160).float() * 2,
            ),
            "split then float then exp": (
                [(1, 64, 25600)],
                lambda x: x.reshape(1, 64, 160, 160).float().exp(),
            ),
            "split then float then sum of the last dim": (
                [(1, 64, 25600)],
                lambda x: x.reshape(1, 64, 160, 160).float().sum(-1),
            ),
            "split then float then sum of the other split dim": (
                [(1, 64, 25600)],
                lambda x: x.reshape(1, 64, 160, 160).float().sum(2),
            ),
        }
        self._check_dense(cases)

    def test_stick_size_change_after_reshape_for_other_dtypes(self):
        # #6557 for the other conversions the rule handles: fp32 down to fp16
        # (32 to 64 per stick), int32 and int64 to fp16, and bf16, which the
        # device holds as fp16 (bool, held the same way, is
        # test_bool_host_reshaped_to_fp32_cpu in test_inductor_ops.py). Each
        # failed with "no supported output layout". Which layout served the read
        # is pinned host-side by TestTypecastOfUnservableRead in
        # tests/tensor/test_coordinates.py; here only the values are checked.
        f32, i32, bf16 = torch.float32, torch.int32, torch.bfloat16
        cases = {
            "fp32 (2,8,40) to (2,320)": (
                f32,
                (2, 8, 40),
                lambda x: x.reshape(2, 320).half(),
            ),
            "fp32 (1,32,20,20) to (1,32,400)": (
                f32,
                (1, 32, 20, 20),
                lambda x: x.reshape(1, 32, 400).half(),
            ),
            "fp32 (1,64,25000) to (1,64,250,100)": (
                f32,
                (1, 64, 25000),
                lambda x: x.reshape(1, 64, 250, 100).half(),
            ),
            "fp32 (1,2,4,8,7) to (1,8,56), no free dim": (
                f32,
                (1, 2, 4, 8, 7),
                lambda x: x.reshape(1, 8, 56).half(),
            ),
            "int32 (2,8,40) to (2,320)": (
                i32,
                (2, 8, 40),
                lambda x: x.reshape(2, 320).half(),
            ),
            # int64 is held as int32 on the device.
            "int64 (2,8,40) to (2,320)": (
                torch.int64,
                (2, 8, 40),
                lambda x: x.reshape(2, 320).half(),
            ),
            "bf16 (1,64,25600) to (1,64,160,160)": (
                bf16,
                (1, 64, 25600),
                lambda x: x.reshape(1, 64, 160, 160).float(),
            ),
        }
        for name, (dtype, shape, fn) in cases.items():
            with self.subTest(name):
                _compile_and_measure(fn, shape, dtype=dtype)

    @expects_raise
    def test_stick_size_change_after_reshape_onto_a_sub_stick_dim(self):
        # #6557 where the only dim the reshape leaves alone has at most 32
        # elements (the issue's other reproducer is the first case). The
        # conversion now compiles; it raised "no supported output layout". Its
        # fp32 result then fits in a single fp32 stick, which the DL16_TO_FP32
        # read-back rejects, as it does for an unreshaped 4x32 .float() in
        # test_inductor_ops.py. Which layout served the read is pinned host-side
        # by TestTypecastOfUnservableRead in tests/tensor/test_coordinates.py.
        cases = {
            "(2,8,40) to (2,320)": ((2, 8, 40), lambda x: x.reshape(2, 320).float()),
            "(1,32,20,20) to (1,32,400)": (
                (1, 32, 20, 20),
                lambda x: x.reshape(1, 32, 400).float(),
            ),
        }
        for name, (shape, fn) in cases.items():
            with self.subTest(name):
                x = torch.randn(*shape, dtype=torch.float16)
                torch.compiler.reset()
                with inductor_config.patch(force_disable_caches=True):
                    out = torch.compile(fn, dynamic=False)(x.to("spyre"))
                torch.spyre.synchronize()
                self.assertEqual(out.dtype, torch.float32)
                self.assertEqual(out.shape, fn(x).shape)
                with pytest.raises(RuntimeError, match="requires whole stick pairs"):
                    out.cpu()

    def test_stick_size_change_after_reshape_with_no_free_dim(self):
        # #6557 for a reshape that leaves no dim alone: no dim can take the
        # stick, so the conversion reads the sparse layout, as the reshape alone
        # does in test_no_free_dim_uses_the_sparse_layout.
        cases = {
            "1x2x4x8x7": ([(1, 2, 4, 8, 7)], lambda x: x.reshape(1, 8, 56).float()),
            "flatten [8,65]": ([(8, 65)], lambda x: x.reshape(-1).float()),
        }
        for name, (shapes, fn) in cases.items():
            with self.subTest(name):
                ratio = _compile_and_measure(fn, *shapes)
                self.assertGreater(ratio, SPARSE_MIN_MEMORY_RATIO)

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
    def test_yolo26_detect_head_cat_is_rejected(self):
        # TODO(#6163): each cat operand is restickified straight into its slice of
        # the cat buffer, where its old-stick dim cannot be padded in place.
        # Restickify into a fresh buffer and copy it into the slice, then make this
        # a positive test. Case 1 of #6163 on two operands is
        # test_cat_of_reshapes in test_inductor_ops.py; this is the yolo26 Detect
        # head, where three views of different sizes meet a cat on the stick dim.
        cases = {
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


def _check_values_on_host(fn, shape, dtype):
    """Compare a compiled ``fn`` with the fp32 CPU reference, holding no device
    tensor once the comparison runs: a failing test's kept traceback then keeps
    no device memory alive into the peak-memory checks of later tests."""
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype)
    ref = fn(x.float())
    torch.compiler.reset()
    with inductor_config.patch(force_disable_caches=True):
        got = torch.compile(fn, dynamic=False)(x.to("spyre")).cpu()
    torch.compiler.reset()
    torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=1e-2)


@pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
class TestFp32StickSplitValues(unittest.TestCase):
    """#6657: an fp32 value whose stick dim a reshape splits (25600 into 160 x 160:
    5 fp32 sticks, 2.5 fp16 sticks) reads back wrong once a cast or pointwise op
    consumes the reshape; the reshape alone is correct. Found while validating
    #6557, whose fix does not reach these reads. One test per spelling, as each
    may take a different kernel and a fix may cover only some of them. Kept out
    of TestStickRelocation so they run after its peak-memory checks.
    """

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="fp32 input split by a reshape, then .half(): wrong values, #6657",
    )
    def test_fp32_stick_split_then_half(self):
        _check_values_on_host(
            lambda x: x.reshape(1, 64, 160, 160).half(), (1, 64, 25600), torch.float32
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="in-graph .float() split by a reshape, then .half(): wrong values, "
        "#6657",
    )
    def test_fp32_stick_split_then_half_of_an_in_graph_float(self):
        _check_values_on_host(
            lambda x: x.float().reshape(1, 64, 160, 160).half(),
            (1, 64, 25600),
            torch.float16,
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="in-graph .float() split by a reshape, then * 2: wrong values, #6657",
    )
    def test_fp32_stick_split_then_multiply(self):
        _check_values_on_host(
            lambda x: x.float().reshape(1, 64, 160, 160) * 2,
            (1, 64, 25600),
            torch.float16,
        )

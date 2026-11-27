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

"""Tests for torch_spyre._C.get_reserved_dims.

A read-only accessor onto a tensor's dynamic-shape reservation (set by
tensor.to("spyre", dynamic={dim: {min, max, granularity}})). Unlike most
accessors in this codebase, it must never raise: it is meant to be called
on arbitrary graph inputs, including tensors that were never reserved and
tensors that aren't even on this device.

Also covers which eager ops keep the reservation (detach, and views that leave
offset, sizes and strides alone) and which drop it, and that a view that keeps
it behaves correctly.
"""

from unittest import mock

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401
import torch_spyre._inductor as spyre_inductor
from torch_spyre._C import get_reserved_dims
from torch_spyre._inductor.wsr import for_each_tile

DEVICE = torch.device("spyre")

DECLARATION = {0: {"min": 70, "max": 630, "granularity": 70}}
# A for_each_tile tile that divides every size the declaration allows.
TILE = 70


def reinterpret_tensor(*args):
    # torch_spyre/_C.pyi does not declare reinterpret_tensor.
    return torch_spyre._C.reinterpret_tensor(*args)


def declared_tensor():
    """A 560-row tensor with dim 0 declared, and its host copy."""
    x = torch.rand(560, 1024, dtype=torch.float16)
    return x, x.to(DEVICE, dynamic=DECLARATION)


class TestGetReservedDims(TestCase):
    def test_reserved_tensor_returns_declared_dict(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertEqual(
            get_reserved_dims(x_dev),
            {0: {"min": 70, "max": 630, "granularity": 70}},
        )

    def test_unreserved_spyre_tensor_returns_none(self) -> None:
        plain = torch.rand(4, 4, dtype=torch.float16).to(DEVICE)
        self.assertIsNone(get_reserved_dims(plain))

    def test_cpu_tensor_returns_none_not_raises(self) -> None:
        cpu = torch.rand(4, 4, dtype=torch.float16)
        self.assertIsNone(get_reserved_dims(cpu))

    def test_detached_copy_inherits_the_map(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertEqual(get_reserved_dims(x_dev.detach()), get_reserved_dims(x_dev))

    def test_still_reserved_after_in_range_resize(self) -> None:
        # resize_ legitimately changes the shape; it must keep the contract.
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        x_dev.resize_((140, 1024))
        self.assertEqual(
            get_reserved_dims(x_dev),
            {0: {"min": 70, "max": 630, "granularity": 70}},
        )

    def test_transposed_view_is_not_reserved(self) -> None:
        # The map is keyed by absolute dim, so it must not follow an
        # axis-swapping view.
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertIsNone(get_reserved_dims(x_dev.t()))
        self.assertIsNone(get_reserved_dims(x_dev.t().detach()))

    def test_in_place_transpose_drops_the_map(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        d = x_dev.detach()
        d.t_()
        self.assertIsNone(get_reserved_dims(d))
        # The original is a different tensor object and is untouched.
        self.assertIsNotNone(get_reserved_dims(x_dev))

    def test_in_place_rank_change_drops_the_map(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        d = x_dev.detach()
        d.unsqueeze_(0)
        self.assertIsNone(get_reserved_dims(d))

    def test_same_geometry_views_inherit_the_map(self) -> None:
        # A view keeps the map while offset, sizes and strides still match the
        # allocation, as detach() does (has_intact_reservation()).
        _, x_dev = declared_tensor()
        same = {
            "view": x_dev.view(x_dev.shape),
            "alias": torch.ops.aten.alias(x_dev),
            "as_strided": x_dev.as_strided(x_dev.shape, x_dev.stride()),
            "view, then detach": x_dev.view(x_dev.shape).detach(),
            # The flat view carries the map but is not intact; viewing back to
            # the original geometry makes it intact again.
            "round trip": x_dev.view(-1).view(x_dev.shape),
            # What Inductor codegen emits to reuse a buffer.
            "reinterpret_tensor": reinterpret_tensor(
                x_dev, list(x_dev.shape), list(x_dev.stride()), 0
            ),
        }
        for name, v in same.items():
            with self.subTest(view=name):
                self.assertEqual(get_reserved_dims(v), DECLARATION)

    def test_changed_geometry_views_are_not_reserved(self) -> None:
        _, x_dev = declared_tensor()
        changed = {
            "flatten": x_dev.view(-1),
            "reshape": x_dev.view(280, 2048),
            "slice": x_dev[1:],
            "as_strided": x_dev.as_strided((280, 1024), (1024, 1)),
            "reinterpret_tensor": reinterpret_tensor(
                x_dev, [559, 1024], list(x_dev.stride()), 1024
            ),
        }
        for name, v in changed.items():
            with self.subTest(view=name):
                self.assertIsNone(get_reserved_dims(v))

    def test_resize_on_changed_geometry_view_raises(self) -> None:
        # Views copy the map, so a changed-geometry view carries it but fails
        # has_intact_reservation(). resize_ then refuses, as after an in-place
        # reshape. Before views copied the map, these took the unreserved path
        # and the base read back wrong data.
        _, x_dev = declared_tensor()
        resizes = {
            "slice": (x_dev[1:], (139, 1024)),
            "reshape": (x_dev.view(280, 2048), (140, 2048)),
            "flatten": (x_dev.view(-1), (70 * 1024,)),
        }
        for name, (v, new_size) in resizes.items():
            with self.subTest(view=name):
                with self.assertRaisesRegex(
                    RuntimeError, "no longer match the allocation"
                ):
                    v.resize_(new_size)

    def test_same_geometry_views_share_the_base_data(self) -> None:
        # The host-to-device copy rounds fp16, so compare with what the device
        # reads back, not with the host tensor. The same bytes must read back
        # identically through the view.
        _, x_dev = declared_tensor()
        on_device = x_dev.cpu()
        for v in (
            x_dev.view(x_dev.shape),
            torch.ops.aten.alias(x_dev),
            x_dev.as_strided(x_dev.shape, x_dev.stride()),
        ):
            self.assertEqual(v.cpu(), on_device, atol=0, rtol=0)
        x_dev.view(x_dev.shape).zero_()
        self.assertTrue(x_dev.cpu().eq(0).all())

    def test_resize_on_view_matches_cpu(self) -> None:
        # With the map, a view takes the reserved resize_ path. Without it, the
        # unreserved path recomputes the layout from the concrete size and the
        # view reads back wrong data.
        # Compared with what the device reads back, since the host-to-device
        # copy rounds fp16.
        _, x_dev = declared_tensor()
        on_device = x_dev.cpu()
        v = x_dev.view(x_dev.shape)
        v.resize_((140, 1024))
        self.assertEqual(tuple(v.shape), (140, 1024))
        self.assertEqual(v.cpu(), on_device[:140], atol=0, rtol=0)
        # The base is untouched.
        self.assertEqual(tuple(x_dev.shape), (560, 1024))
        self.assertEqual(x_dev.cpu(), on_device, atol=0, rtol=0)

    def test_grow_on_view_matches_cpu(self) -> None:
        # Growing is what the reserved path exists for: the buffer is already
        # sized for the declared max, so nothing is reallocated. Only the rows
        # that existed before the grow have defined values.
        x = torch.rand(140, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic=DECLARATION)
        on_device = x_dev.cpu()
        v = x_dev.view(x_dev.shape)
        v.resize_((630, 1024))
        self.assertEqual(tuple(v.shape), (630, 1024))
        self.assertEqual(v.cpu()[:140], on_device, atol=0, rtol=0)
        self.assertEqual(tuple(x_dev.shape), (140, 1024))
        # The data check alone also passes through the unreserved path, so
        # check the view kept its reservation through the grow.
        self.assertEqual(get_reserved_dims(v), DECLARATION)


# One compiled function reused over these sizes: the kernel sees the whole range.
SIZES = (70, 140, 560, 630)

# Eager ops on a declared tensor. `view(t.shape)` and `aten.alias` work on both
# a concrete and a symbolic tensor.
OPS = {
    "identity": lambda t: t,
    "view": lambda t: t.view(t.shape),
    "detach": lambda t: t.detach(),
    "alias": lambda t: torch.ops.aten.alias(t),
}


def _double_per_tile(t):
    """t + t as a for_each_tile loop over dim 0.

    A pointwise op over a declared dim is refused outside a for_each_tile: not
    every size in the range is a multiple of the compiler's tile.
    """

    def body(_, ops):
        (tile,) = ops
        return None, tile + tile

    _, out = for_each_tile(body, (t,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


def _compile_fresh(fn, **kwargs):
    # The functions compiled here share code objects (_double_per_tile and the
    # lambdas in OPS). Reset Dynamo so they stay under its recompile limit,
    # which fullgraph=True turns into an error.
    torch._dynamo.reset()
    return torch.compile(fn, **kwargs)


class TestViewResultIntoCompiledKernel(TestCase):
    """x.to(dynamic=...) -> eager op -> compiled kernel, against eager x + x.

    An eager result is a new tensor and does not inherit the mark_dynamic that
    .to(dynamic=...) set, so the first compile of it would be static and see no
    symbolic dim. The test marks dim 0 dynamic itself so that every compile has
    one. The compile then reads get_reserved_dims(result) to decide whether that
    dim is declared. With no map it is an ordinary dynamic dim, sized from a
    hint; with one, the kernel is built for the declared maximum. Both give
    right values at the sizes tried, so the test also checks what the compile
    concluded.
    """

    def test_compile_sees_the_declared_range_and_matches_eager(self) -> None:
        real_declared_dims_of = spyre_inductor._declared_dims_of
        for name, op in OPS.items():
            with self.subTest(op=name):
                dev_fn = _compile_fresh(_double_per_tile, fullgraph=True)
                # What the compile wrapper concluded, per device compile.
                seen: list[dict[object, tuple[int, ...]]] = []

                def recording(*args, _seen=seen, **kwargs):
                    result = real_declared_dims_of(*args, **kwargs)
                    _seen.append({k: tuple(v) for k, v in result.items()})
                    return result

                for n in SIZES:
                    x = torch.rand(n, 1024, dtype=torch.float16)
                    v = op(x.to(DEVICE, dynamic=DECLARATION))
                    torch._dynamo.mark_dynamic(v, 0)
                    with mock.patch.object(
                        spyre_inductor, "_declared_dims_of", recording
                    ):
                        out_dev = dev_fn(v)

                    self.assertEqual(
                        out_dev.cpu(),
                        x + x,
                        atol=1e-3,
                        rtol=0,
                        msg=f"{name}@{n}: device differs from CPU",
                    )

                self.assertTrue(seen, "the device kernel never compiled")
                for declared in seen:
                    self.assertEqual(
                        sorted(declared.values()),
                        [(70, 630, 70)],
                        f"{name}: compile did not see the declared range",
                    )


if __name__ == "__main__":
    run_tests()

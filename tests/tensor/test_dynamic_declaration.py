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

"""Tests for validating a dynamic= declaration at .to() time.

All of these are checked before anything is allocated: min/max/granularity
must be positive ints, min <= max, max must be a multiple of granularity
(the declared ceiling -- distinct from the separate check on today's real
size), min must clear a configured floor, and max / granularity must not
exceed the configured bucket cap. Also covers negative-dim canonicalization
and unknown-key rejection in the same validation pass.
"""

import warnings
from types import SimpleNamespace

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401
from torch_spyre._C import (
    get_reserved_dims,
    get_spyre_tensor_layout,
    spyre_empty_reserved,
    SpyreTensorLayout,
)
from torch_spyre._inductor import config as inductor_config
from torch_spyre._monkey_patch import _check_layout_for_reservation, _DynamicSpec

DEVICE = torch.device("spyre")


class TestDynamicDeclarationValidation(TestCase):
    def setUp(self) -> None:
        # Real size at dim 0 is 560, so every valid param set below must
        # keep 560 within [min, max] and 560 % granularity == 0.
        self.x = torch.rand(560, 1024, dtype=torch.float16)

    def _assert_rejected(self, spec, match) -> None:
        with self.assertRaisesRegex(ValueError, match):
            self.x.to(DEVICE, dynamic={0: spec})

    def test_non_int_min_rejected(self) -> None:
        self._assert_rejected(
            {"min": 56.0, "max": 616, "granularity": 8}, "positive int"
        )

    def test_zero_granularity_rejected(self) -> None:
        self._assert_rejected({"min": 56, "max": 616, "granularity": 0}, "positive int")

    def test_negative_max_rejected(self) -> None:
        self._assert_rejected(
            {"min": 56, "max": -616, "granularity": 8}, "positive int"
        )

    def test_min_exceeds_max_rejected(self) -> None:
        self._assert_rejected({"min": 700, "max": 616, "granularity": 8}, "exceeds max")

    def test_max_not_multiple_of_granularity_rejected(self) -> None:
        # The declared ceiling itself must divide evenly -- a different
        # rule from the real-size check below.
        self._assert_rejected(
            {"min": 56, "max": 620, "granularity": 8},
            "not a multiple of granularity",
        )

    def test_min_below_floor_rejected(self) -> None:
        self._assert_rejected(
            {"min": 4, "max": 616, "granularity": 4},
            "minimum supported reservation floor",
        )

    def test_too_many_buckets_rejected(self) -> None:
        over_cap = (inductor_config.max_buckets + 1) * 8
        self._assert_rejected(
            {"min": 8, "max": over_cap, "granularity": 8},
            "exceeding the cap",
        )

    def test_real_size_outside_declared_range_rejected(self) -> None:
        # Not one of the declaration-only rules above: the *current*
        # tensor's size must itself be admissible under the declared range.
        self._assert_rejected(
            {"min": 64, "max": 512, "granularity": 64},  # 560 > 512
            "is not within",
        )

    def test_real_size_off_granularity_rejected(self) -> None:
        # 600 / 100 = 6 buckets, well under the cap, so this clears every
        # declaration rule and only fails on 560 % 100 != 0.
        self._assert_rejected(
            {"min": 100, "max": 600, "granularity": 100},  # 560 % 100 != 0
            "not a multiple of granularity",
        )

    def test_negative_dim_canonicalized(self) -> None:
        # dim=-2 on a 2-D tensor means dim 0. (dim=-1 would resolve to the
        # innermost/stick dim, which is a separate, deliberate restriction
        # -- not what this test is covering.)
        x_dev = self.x.to(
            DEVICE, dynamic={-2: {"min": 70, "max": 630, "granularity": 70}}
        )
        self.assertEqual(
            get_reserved_dims(x_dev),
            {0: {"min": 70, "max": 630, "granularity": 70}},
        )

    def test_unknown_key_rejected(self) -> None:
        # A typo'd key ("granulrity") must not silently fall back to
        # granularity's default of 1 -- that would make the reservation
        # claim no step restriction while the caller believes it declared
        # one, and every later resize_ to an off-step size would be
        # silently accepted instead of refused.
        self._assert_rejected({"min": 70, "max": 630, "granulrity": 70}, "unknown key")

    def test_non_int_dim_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an int"):
            self.x.to(DEVICE, dynamic={0.5: {"min": 70, "max": 630}})

    def test_bool_value_rejected(self) -> None:
        # bool subclasses int, so True would otherwise pass as 1.
        for spec in (
            {"min": 56, "max": 616, "granularity": True},
            {"min": True, "max": 616, "granularity": 8},
            {"min": 56, "max": True, "granularity": 8},
        ):
            self._assert_rejected(spec, "positive int")

    def test_dtype_kwarg_honoured(self) -> None:
        # dynamic= must not swallow the other .to() arguments: the buffer
        # has to come out at the requested dtype, not the source's.
        x32 = torch.rand(560, 1024, dtype=torch.float32)
        spec = {"min": 70, "max": 630, "granularity": 70}
        x_dev = x32.to(DEVICE, dtype=torch.float16, dynamic={0: spec})
        self.assertEqual(x_dev.dtype, torch.float16)
        self.assertEqual(x_dev.shape, x32.shape)
        self.assertEqual(get_reserved_dims(x_dev), {0: spec})

    def test_dtype_positional_honoured(self) -> None:
        x32 = torch.rand(560, 1024, dtype=torch.float32)
        x_dev = x32.to(
            DEVICE,
            torch.float16,
            dynamic={0: {"min": 70, "max": 630, "granularity": 70}},
        )
        self.assertEqual(x_dev.dtype, torch.float16)

    def test_dtype_only_without_device_rejected(self) -> None:
        # No device given: not a CPU -> spyre transfer, so it must raise
        # rather than quietly do something else.
        with self.assertRaisesRegex(ValueError, "only supported for CPU"):
            self.x.to(
                dtype=torch.float32,
                dynamic={0: {"min": 70, "max": 630, "granularity": 70}},
            )

    def test_unsupported_memory_format_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "memory_format"):
            self.x.to(
                DEVICE,
                memory_format=torch.channels_last,
                dynamic={0: {"min": 70, "max": 630, "granularity": 70}},
            )

    def test_huge_max_overflow_rejected(self) -> None:
        # Calls the native entry point directly: the Python wrapper keeps
        # max within ~32x the real size, but spyre_empty_reserved is public
        # and must not let a wrapped byte count undersize the reservation.
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            spyre_empty_reserved((560, 1024), (1024, 1), torch.float16, 0, 70, 2**62, 2)

    def test_device_layout_not_at_max_rejected(self) -> None:
        # The explicit layout describes the buffer at the declared max. A
        # layout for today's 560 rows would be silently too small for 630.
        stl = SpyreTensorLayout((560, 1024), torch.float16)
        with self.assertRaisesRegex(ValueError, "extent max=630, got 560"):
            self.x.to(
                DEVICE,
                device_layout=stl,
                dynamic={0: {"min": 70, "max": 630, "granularity": 70}},
            )

    def test_device_layout_reserved_dim_on_stick_rejected(self) -> None:
        # dim_order puts host dim 0 in the stick: the reserved rows would be
        # interleaved across sticks and could not be resized in place.
        stl = SpyreTensorLayout((630, 1024), (1024, 1), torch.float16, [1, 0])
        with self.assertRaisesRegex(ValueError, "exactly one device dimension"):
            self.x.to(
                DEVICE,
                device_layout=stl,
                dynamic={0: {"min": 70, "max": 630, "granularity": 70}},
            )

    def test_device_layout_at_max_accepted(self) -> None:
        # The derived layout at max, given explicitly: accepted, the tensor
        # keeps its real shape and carries the declaration.
        stl = SpyreTensorLayout((630, 1024), torch.float16)
        y = self.x.to(
            DEVICE,
            device_layout=stl,
            dynamic={0: {"min": 70, "max": 630, "granularity": 70}},
        )
        self.assertEqual(tuple(y.shape), (560, 1024))
        self.assertEqual(
            get_reserved_dims(y), {0: {"min": 70, "max": 630, "granularity": 70}}
        )
        self.assertEqual(
            list(get_spyre_tensor_layout(y).device_size), list(stl.device_size)
        )

    def test_layout_check_is_a_pure_function(self) -> None:
        # The rule, without a device: the reserved host dim must sit whole in
        # exactly one device dim, not the stick one, with extent max.
        spec = _DynamicSpec(dim=0, min=8, max=16, granularity=1)
        size, stride = (12, 32, 1), (32, 1, 1)
        ok = SimpleNamespace(device_size=[16, 1, 32, 32], stride_map=[32, 0, 1, 0])
        _check_layout_for_reservation(ok, spec, size, stride)
        too_small = SimpleNamespace(
            device_size=[12, 1, 32, 32], stride_map=[32, 0, 1, 0]
        )
        with self.assertRaisesRegex(ValueError, "extent max=16, got 12"):
            _check_layout_for_reservation(too_small, spec, size, stride)
        folded = SimpleNamespace(device_size=[1, 32, 32], stride_map=[0, 1, 0])
        with self.assertRaisesRegex(ValueError, "found 0"):
            _check_layout_for_reservation(folded, spec, size, stride)
        on_stick = SimpleNamespace(device_size=[32, 1, 16], stride_map=[1, 0, 32])
        with self.assertRaisesRegex(ValueError, "stick-elements"):
            _check_layout_for_reservation(on_stick, spec, size, stride)

    def test_reserved_transfer_does_not_warn(self) -> None:
        # The transfer used to warn that compiled kernels were built for the
        # first compile's size (#5937). The compiler now builds a declared
        # dimension's geometry at the declared maximum and binds the size at
        # launch, so there is nothing to warn about.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertEqual(
            [str(w.message) for w in caught if "compiled" in str(w.message)], []
        )

    def test_non_dict_spec_rejected(self) -> None:
        # The per-dim declaration must be a dict, not a bare number/list/None.
        for spec in (5, None, [70, 630], "max"):
            with self.assertRaisesRegex(ValueError, "expected a dict"):
                self.x.to(DEVICE, dynamic={0: spec})

    def test_bool_dim_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an int"):
            self.x.to(DEVICE, dynamic={True: {"min": 70, "max": 630}})


if __name__ == "__main__":
    run_tests()

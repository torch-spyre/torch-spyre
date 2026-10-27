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

import functools
import os
import warnings
from contextlib import contextmanager

import psutil
import pytest
import regex as re
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    set_warn_always_context,
    subtest,
)

# 0-dim scalar roundtrip: shared dtype × factory lambdas (used by parametrized fill test).
_SCALAR_ROUNDTRIP_DTYPE_CASES = [
    (torch.int8, lambda fn: fn((), dtype=torch.float32).to(dtype=torch.int8)),
    (torch.bool, lambda fn: fn((), dtype=torch.bool)),
    (torch.int64, lambda fn: fn((), dtype=torch.int64)),
    (
        torch.float8_e4m3fn,
        lambda fn: fn((), dtype=torch.float32).to(dtype=torch.float8_e4m3fn),
    ),
    (torch.float16, lambda fn: fn((), dtype=torch.float16)),
    (torch.float32, lambda fn: fn((), dtype=torch.float32)),
]

# The add is correct, but int8 goes through a CPU fallback for the conversion to
# float32, which warns, and the helper asserts no warning (#5364).
_SCALAR_ADD_XFAIL_TO_DTYPE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the int8 to float32 conversion falls back to the CPU and warns, "
    "and the test asserts no warning (#5364)",
)


# A float16-format bool's float32 or integer result is staggered on device.
_EAGER_BOOL_READBACK_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="a staggered result is read back without restoring its order (#4393)",
)


def _expect_raise(match):
    """Decorator: the test passes only if it raises with a message matching ``match``."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with pytest.raises(Exception, match=match):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


# TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1474 (DataFormats.SEN143_FP8)
# A missing feature, not a design limit: scalar eager add is rejected for SEN143_FP8
# today; when #1474 lands this fails and the case has to go back to a normal one.
_SCALAR_ADD_RAISES_FP8 = _expect_raise(r"add on DataFormats\.SEN143_FP8")
# TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/925
_SCALAR_ADD_SKIP_INT = pytest.mark.skip(
    reason="Spyre backend does not support int32/int16 dtype - causes segfault/crash in data format converter"
)

# TODO:ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1588
_SCALAR_ADD_SKIP_UINT8 = pytest.mark.skip(
    reason="uint8 host transfers unsupported: host format SENUINT32 is 4 bytes, PyTorch stores 1 (#1588)"
)

_SCALAR_ADD_FALLBACK_FULL_WARN = r"torch\.ops\.spyre\.full is falling back to cpu"


@instantiate_parametrized_tests
class TestSpyre(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "spyre")

    def test_autograd_init(self):
        # Make sure autograd is initialized
        torch.ones(2, requires_grad=True, device="spyre").sum().backward()

        pid = os.getpid()
        task_path = f"/proc/{pid}/task"
        all_threads = psutil.Process(pid).threads()

        all_thread_names = set()

        for t in all_threads:
            with open(f"{task_path}/{t.id}/comm") as file:
                thread_name = file.read().strip()
            all_thread_names.add(thread_name)

        for i in range(torch.spyre.device_count()):
            self.assertIn(f"pt_autograd_{i}", all_thread_names)

    def test_empty_factory(self):
        a = torch.empty(50, device="spyre", dtype=torch.float16)
        self.assertEqual(a.device.type, "spyre")

        a.fill_(3.5)

        a_cpu = a.cpu()
        self.assertTrue(a_cpu.eq(3.5).all())

    @parametrize(
        "factory",
        [
            subtest(
                lambda: torch.empty((64, 64), dtype=torch.float16, pin_memory=True),
                name="empty",
            ),
            subtest(
                lambda: torch.empty_strided(
                    (64, 64),
                    (1, 64),
                    dtype=torch.float16,
                    pin_memory=True,
                ),
                name="empty_strided",
            ),
        ],
    )
    def test_cpu_factory_pin_memory(self, factory):
        cpu = factory()

        self.assertEqual(cpu.device.type, "cpu")
        self.assertTrue(cpu.is_pinned())

        expected = (torch.arange(cpu.numel(), dtype=torch.int64) % 31).to(cpu.dtype)
        cpu.copy_(expected.reshape(cpu.shape))
        actual = cpu.to("spyre").cpu()
        torch.testing.assert_close(actual, cpu, rtol=0, atol=0)

    def test_cpu_pin_memory_method_and_interior_view(self):
        values = (torch.arange(4128, dtype=torch.int64) % 31).to(torch.float16)
        base = values.pin_memory()
        view = base[17:4113]

        self.assertTrue(base.is_pinned())
        self.assertTrue(view.is_pinned())
        self.assertEqual(
            view.data_ptr(),
            base.data_ptr() + 17 * base.element_size(),
        )

    def test_explicit_pinned_d2h(self):
        expected = (torch.arange(4096, dtype=torch.int64) % 31).to(torch.float16)
        source = expected.to("spyre")
        destination = torch.empty_like(expected, pin_memory=True)

        destination.copy_(source, non_blocking=True)
        torch.accelerator.synchronize()

        self.assertTrue(destination.is_pinned())
        torch.testing.assert_close(destination, expected, rtol=0, atol=0)

    def test_pageable_d2h_remains_supported(self):
        expected = (torch.arange(4096, dtype=torch.int64) % 31).to(torch.float16)
        source = expected.to("spyre")
        destination = torch.empty_like(expected, pin_memory=False)

        destination.copy_(source, non_blocking=True)
        torch.accelerator.synchronize()

        self.assertFalse(destination.is_pinned())
        torch.testing.assert_close(destination, expected, rtol=0, atol=0)

    # The DCI-skip optimization must reject anything that is not a dense
    # same-format copy: for each layout below, dci_is_identity_copy has to
    # keep the conversion descriptor, and a round trip through the device
    # must reproduce the view's values exactly. (If the predicate wrongly
    # accepted the layout, the raw DMA would move storage bytes a plain
    # memcpy would — including bytes before a storage offset or stride
    # gaps — and the comparison would fail.)
    #
    # These run in same-format dtypes (int8, fp32): fp16/bf16 host and
    # device formats differ, so their DCIs are rejected at the dataformat
    # check before the offset/stride/broadcast logic is ever reached and
    # would not exercise the layout rejection at all.

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_pinned_h2d_storage_offset_view_keeps_conversion(self, dtype):
        base = (torch.arange(8192, dtype=torch.int64) % 31).to(dtype)
        pinned = base.pin_memory()
        view = pinned[37:8101]  # non-zero storage offset + partial slice

        dev = view.to("spyre")
        torch.testing.assert_close(dev.cpu(), view, rtol=0, atol=0)

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_pinned_h2d_strided_view_keeps_conversion(self, dtype):
        base = (torch.arange(4096, dtype=torch.int64) % 31).to(dtype)
        pinned = base.pin_memory()
        view = pinned.view(64, 64)[:, ::2]  # stride gaps between rows

        dev = view.to("spyre")
        torch.testing.assert_close(dev.cpu(), view, rtol=0, atol=0)

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_pinned_h2d_broadcast_strides_keeps_conversion(self, dtype):
        row = (torch.arange(64, dtype=torch.int64) % 31).to(dtype)
        pinned = row.pin_memory()
        view = pinned.unsqueeze(0).expand(32, 64)  # broadcast (zero) stride

        dev = view.to("spyre")
        torch.testing.assert_close(dev.cpu(), view, rtol=0, atol=0)

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_pinned_h2d_into_device_slice_keeps_conversion(self, dtype):
        # The device side is the strided side here: a copy into a device slice
        # must keep its DCI, since a raw DMA would write storage-base-relative
        # bytes instead of the slice's. The offset is stick-aligned (128
        # elements is a whole stick for both int8 and fp32) so the test does
        # not depend on the pre-existing mid-stick int8 slice bug, which is
        # flex-version dependent.
        pinned = (torch.arange(4096, dtype=torch.int64) % 31).to(dtype).pin_memory()
        dev = torch.zeros(8192, dtype=dtype, device="spyre")

        dev[128:4224].copy_(pinned)
        torch.testing.assert_close(dev[128:4224].cpu(), pinned, rtol=0, atol=0)
        torch.testing.assert_close(
            dev[:128].cpu(), torch.zeros(128, dtype=dtype), rtol=0, atol=0
        )

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_pinned_d2h_from_device_slice_keeps_conversion(self, dtype):
        # Mirror of the above: reading a device slice out into a pinned host
        # tensor must also keep the conversion for the device-side offset.
        # Stick-aligned offset, same rationale as above.
        dev = (torch.arange(8192, dtype=torch.int64) % 31).to(dtype).to("spyre")
        y = torch.empty(4096, dtype=dtype, pin_memory=True)

        y.copy_(dev[128:4224])
        torch.testing.assert_close(y, dev[128:4224].cpu(), rtol=0, atol=0)

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_slice_copy_into_same_size_host_buffer(self, dtype):
        # Regression test: the identity-copy skip must compare the copied
        # region, not the host storage size. y is exactly as large as the
        # device allocation, but only y[:4096] is the copy destination -- a
        # whole-allocation raw DMA here would overwrite y[4096:] with device
        # data (silent corruption).
        src = ((torch.arange(8192, dtype=torch.int64) % 31) + 1).to(dtype)
        dev = src.to("spyre")

        for pin in (False, True):
            y = torch.full((8192,), -1, dtype=dtype, pin_memory=pin)
            y[:4096].copy_(dev[:4096])
            torch.testing.assert_close(y[:4096], src[:4096], rtol=0, atol=0)
            torch.testing.assert_close(
                y[4096:], torch.full((4096,), -1, dtype=dtype), rtol=0, atol=0
            )

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_pinned_round_trip_same_format(self, dtype):
        # The flagship fast-path combination: a same-format pinned tensor is
        # submitted as a raw pinned DMA (DCI stripped) in both directions.
        # fp16 keeps the conversion path and never exercises this.
        x = (torch.arange(8192, dtype=torch.int64) % 31).to(dtype)
        pinned = x.pin_memory()

        dev = pinned.to("spyre")
        torch.testing.assert_close(dev.cpu(), x, rtol=0, atol=0)

        y = torch.empty(8192, dtype=dtype, pin_memory=True)
        y.copy_(dev, non_blocking=True)
        torch.accelerator.synchronize()
        torch.testing.assert_close(y, x, rtol=0, atol=0)

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_d2h_dense_leading_slice(self, dtype):
        # d[:4096].cpu() is dense with offset 0 but covers only part of the
        # device allocation: the identity-copy skip must not fire, or the raw
        # DMA (sized from the whole allocation) would overrun the 4096-element
        # host buffer returned by .cpu().
        src = (torch.arange(8192, dtype=torch.int64) % 31).to(dtype)
        dev = src.to("spyre")

        out = dev[:4096].cpu()
        torch.testing.assert_close(out, src[:4096], rtol=0, atol=0)

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_non_blocking_pageable_h2d_source_freed(self, dtype):
        # With the DCI stripped there is no caller-thread staging copy, so
        # flex reads the pageable source pages from the scheduler thread
        # after the copy call returns. Dropping the only reference right
        # after a non-blocking copy must not free those pages under the
        # in-flight DMA.
        n = 8192
        dev = torch.empty(n, dtype=dtype, device="spyre")
        src = ((torch.arange(n, dtype=torch.int64) % 31) + 1).to(dtype)
        expected = src.clone()

        dev.copy_(src, non_blocking=True)
        del src
        torch.accelerator.synchronize()
        torch.testing.assert_close(dev.cpu(), expected, rtol=0, atol=0)

    @parametrize(
        "dtype",
        [
            subtest(torch.int8, name="int8"),
            subtest(torch.float32, name="fp32"),
        ],
    )
    def test_non_blocking_pinned_d2h_freed_and_reused(self, dtype):
        # A pinned destination freed to the staging cache right after a
        # non-blocking D2H must not be handed back out while the DMA is
        # still writing it (the in-flight buffer is retained by the DMA's
        # completion callback). Whether the next allocation recycles the
        # freed buffer is timing-dependent, so this checks the observable
        # contract: the copy completes without error and later pinned
        # traffic through the cache is intact.
        n = 8192
        src = ((torch.arange(n, dtype=torch.int64) % 31) + 1).to(dtype)
        dev = src.to("spyre")

        y = torch.empty(n, dtype=dtype, pin_memory=True)
        y.copy_(dev, non_blocking=True)
        del y

        z = torch.empty(n, dtype=dtype, pin_memory=True)
        z.fill_(-1)
        torch.accelerator.synchronize()

        expected = ((torch.arange(n, dtype=torch.int64) % 31) + 1).to(dtype)
        torch.testing.assert_close(
            z.copy_(dev, non_blocking=False), expected, rtol=0, atol=0
        )

    def test_pinned_h2d_partial_stick_layout_keeps_conversion(self):
        # 64x32 int8 maps to a partial-stick device layout: the DCI carries
        # real stickification and must survive the identity-copy skip.
        expected = (torch.arange(64 * 32, dtype=torch.int64) % 31).to(torch.int8)
        pinned = expected.reshape(64, 32).pin_memory()

        dev = pinned.to("spyre")
        torch.testing.assert_close(dev.cpu(), pinned, rtol=0, atol=0)

    def test_zero_size_pinned_allocation(self):
        cpu = torch.empty(0, dtype=torch.float16, pin_memory=True)
        self.assertEqual(cpu.numel(), 0)
        self.assertEqual(cpu.device.type, "cpu")

    def test_empty_host_cache_preserves_active_allocation(self):
        active = torch.empty(4096, dtype=torch.float16, pin_memory=True)
        active.fill_(7)

        torch.accelerator.empty_host_cache()

        self.assertTrue(active.is_pinned())
        self.assertTrue(active.eq(7).all())

        del active
        torch.accelerator.empty_host_cache()
        replacement = torch.empty(4096, dtype=torch.float16, pin_memory=True)
        self.assertTrue(replacement.is_pinned())

    def test_empty_factory_in_device_context(self):
        # The error only repros if at least one allocation
        # has already happened
        _ = torch.empty(64, dtype=torch.float16, device="spyre")

        with torch.device("spyre"):
            a = torch.empty(50, dtype=torch.float16)
        self.assertEqual(a.device.type, "spyre")

    def test_empty_size_arguments(self):
        # Positional size argument
        a = torch.empty((2, 3), device="spyre", dtype=torch.float16)
        self.assertEqual(tuple(a.shape), (2, 3))

        # Keyword size argument
        a = torch.empty(size=(2, 3), device="spyre", dtype=torch.float16)
        self.assertEqual(tuple(a.shape), (2, 3))

        # Both positonal and keyword size arguments
        with pytest.raises(
            TypeError,
            match="received an invalid combination of arguments",
        ):
            torch.empty((2, 3), size=(2, 3), device="spyre", dtype=torch.float16)

    def test_ones_factory(self):
        a = torch.ones(50, device="spyre", dtype=torch.float16)
        self.assertEqual(a.device.type, "spyre")
        a_cpu = a.cpu()
        self.assertTrue(a_cpu.eq(1.0).all())

    def test_str(self):
        a = torch.tensor([1, 2], dtype=torch.float16).to("spyre")
        a_repr = str(a)

        def normalize_device(s):
            return re.sub(r"(device='spyre):\d+'", r"\1:0'", s)

        a_repr = normalize_device(a_repr)

        # Check the the print includes all elements and Spyre device
        expected_a_repr = "tensor([1., 2.], dtype=torch.float16, device='spyre:0')"
        self.assertEqual(expected_a_repr, a_repr)

    def test_repr(self):
        a = torch.tensor([1.234242424234, 2], dtype=torch.float16).to("spyre")
        try:
            a_repr = f"{a}"
        except RuntimeError as err:
            self.fail(f"Printing tensor failed with runtime error {err}")

        def normalize_device(s):
            return re.sub(r"(device='spyre):\d+'", r"\1:0'", s)

        a_repr = normalize_device(a_repr)

        # Check the the print includes all elements and Spyre device
        expected_a_repr = (
            "tensor([1.2344, 2.0000], dtype=torch.float16, device='spyre:0')"
        )
        self.assertEqual(expected_a_repr, a_repr)

    def test_printing(self):
        t = torch.ones((2, 3), device="spyre", dtype=torch.float16)

        # Try printing
        try:
            print(t)
            print("Tensor printing works!")
        except NotImplementedError as e:
            self.fail(f"Spyre backend should support tensor printing: {e}")

    @contextmanager
    def _device_transfer_warnings(
        self,
        expect_downcast_warning=False,
        ignored_warning_message=None,
    ):
        """CPU↔spyre transfer tests: PyTorch-style warn_always + recorded warnings.

        Uses ``set_warn_always_context(True)`` (same as ``TestCase.assertWarnsOnceRegex``)
        so TORCH_WARN_ONCE behaves consistently, then applies Spyre-specific int64
        downcast expectations (and optional ignored fallback warnings).
        """
        with set_warn_always_context(True):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                if ignored_warning_message is not None:
                    warnings.filterwarnings(
                        "ignore",
                        message=ignored_warning_message,
                    )
                yield

        msg = "does not support int64"
        recorded = [str(w.message) for w in rec]
        if expect_downcast_warning:
            self.assertTrue(
                any(msg in s for s in recorded),
                f"Expected warning containing '{msg}', got: {recorded}",
            )
        else:
            self.assertEqual(len(recorded), 0, f"Expected no warnings, got: {recorded}")

    def _assert_roundtrip_close(self, x_original, x_roundtrip, dtype):
        if dtype == torch.float8_e4m3fn:
            torch.testing.assert_close(
                x_original.float(),
                x_roundtrip.float(),
            )
        else:
            torch.testing.assert_close(
                x_original,
                x_roundtrip,
                rtol=2e-3,
                atol=1e-5,
                check_dtype=False,
            )

    def _assert_zero_dim_roundtrip(self, x_original, x_roundtrip, dtype):
        self.assertEqual(x_original.ndim, 0)
        self.assertEqual(x_original.numel(), 1)
        self.assertEqual(x_roundtrip.ndim, 0)
        self.assertEqual(x_roundtrip.numel(), 1)
        self._assert_roundtrip_close(x_original, x_roundtrip, dtype)

    def _roundtrip_to_spyre_and_back(self, x, expect_warning=False):
        self.assertEqual(x.device.type, "cpu", "initial device is not cpu")

        with self._device_transfer_warnings(expect_downcast_warning=expect_warning):
            x_spyre = x.to("spyre")

        self.assertEqual(x_spyre.device.type, "spyre", "to device is not spyre")
        self.assertEqual(x_spyre.dtype, x.dtype)

        return x_spyre.to("cpu")

    def _run_scalar_add_case(self, dtype, scalar_factory, ignored_warning_message=None):
        dtypes_with_downcast_warning = {torch.int64}
        a = scalar_factory()
        self.assertEqual(a.ndim, 0)
        self.assertEqual(a.numel(), 1)

        with self._device_transfer_warnings(
            expect_downcast_warning=dtype in dtypes_with_downcast_warning,
            ignored_warning_message=ignored_warning_message,
        ):
            b = a.to(device="spyre").add(2.0).to(device="cpu")

        self.assertEqual(b.ndim, 0)
        self.assertEqual(b.numel(), 1)

        expected = a + 2
        if dtype == torch.float8_e4m3fn:
            torch.testing.assert_close(b.float(), expected.float())
        else:
            torch.testing.assert_close(
                b,
                expected,
                rtol=2e-3,
                atol=1e-5,
                check_dtype=False,
            )

    @parametrize(
        "factory_name",
        [
            subtest("zeros", name="zeros"),
            subtest("ones", name="ones"),
        ],
    )
    def test_cross_device_copy_scalar_fill(self, factory_name):
        tensor_factory = getattr(torch, factory_name)
        for dtype, scalar_factory in _SCALAR_ROUNDTRIP_DTYPE_CASES:
            x = scalar_factory(tensor_factory)
            x_cpu = self._roundtrip_to_spyre_and_back(
                x, expect_warning=dtype in {torch.int64}
            )
            self._assert_zero_dim_roundtrip(x, x_cpu, dtype)

    # Scalar 0-dim: spyre -> add(2.) -> cpu. int8: the conversion to float32 warns (#5364).
    # float8: SEN143_FP8 unsupported. float16/float32: pass; ignore known full() CPU fallback warn.
    @parametrize(
        "dtype, scalar_factory, ignored_warning_message",
        [
            subtest(
                (torch.int8, lambda: torch.tensor(10, dtype=torch.int8), None),
                name="int8",
                decorators=[_SCALAR_ADD_XFAIL_TO_DTYPE],
            ),
            subtest(
                (torch.bool, lambda: torch.tensor(True, dtype=torch.bool), None),
                name="bool",
            ),
            subtest(
                (torch.int64, lambda: torch.tensor(10, dtype=torch.int64), None),
                name="int64",
            ),
            subtest(
                (torch.uint8, lambda: torch.tensor(10, dtype=torch.uint8), None),
                name="uint8",
                decorators=[_SCALAR_ADD_SKIP_UINT8],
            ),
            subtest(
                (torch.int16, lambda: torch.tensor(10, dtype=torch.int16), None),
                name="int16",
                decorators=[_SCALAR_ADD_SKIP_INT],
            ),
            subtest(
                (torch.int32, lambda: torch.tensor(10, dtype=torch.int32), None),
                name="int32",
                decorators=[_SCALAR_ADD_SKIP_INT],
            ),
            subtest(
                (
                    torch.float8_e4m3fn,
                    lambda: torch.tensor(10.0, dtype=torch.float32).to(
                        dtype=torch.float8_e4m3fn
                    ),
                    None,
                ),
                name="float8_e4m3fn",
                decorators=[_SCALAR_ADD_RAISES_FP8],
            ),
            subtest(
                (
                    torch.float16,
                    lambda: torch.tensor(10, dtype=torch.float16),
                    _SCALAR_ADD_FALLBACK_FULL_WARN,
                ),
                name="float16",
            ),
            subtest(
                (
                    torch.float32,
                    lambda: torch.tensor(10, dtype=torch.float32),
                    _SCALAR_ADD_FALLBACK_FULL_WARN,
                ),
                name="float32",
            ),
        ],
    )
    def test_cross_device_copy_scalar_add(
        self, dtype, scalar_factory, ignored_warning_message
    ):
        self._run_scalar_add_case(
            dtype,
            scalar_factory,
            ignored_warning_message=ignored_warning_message,
        )

    def test_cross_device_copy(self):
        a = torch.rand(10, dtype=torch.float16)
        b = a.to(device="spyre").add(2.0).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_device_copy_dtypes(self):
        dtype_configs = {
            torch.int8: lambda: (torch.rand(64, 64) * 100).to(dtype=torch.int8),
            torch.bool: lambda: torch.randint(0, 2, (64, 64), dtype=torch.bool),
            torch.int64: lambda: torch.randint(
                -32768, 32767, (64, 64), dtype=torch.int64
            ),
            torch.float8_e4m3fn: lambda: torch.rand(64, 64).to(
                dtype=torch.float8_e4m3fn
            ),
            torch.float16: lambda: torch.rand(64, 64, dtype=torch.float16),
            torch.float32: lambda: torch.rand(64, 64, dtype=torch.float32),
        }

        dtypes_with_downcast_warning = {torch.int64}

        for dtype, tensor_factory in dtype_configs.items():
            x = tensor_factory()
            x_cpu = self._roundtrip_to_spyre_and_back(
                x, expect_warning=dtype in dtypes_with_downcast_warning
            )
            self._assert_roundtrip_close(x, x_cpu, dtype)

    @parametrize(
        "dtype",
        [
            subtest(
                dtype,
                name=str(dtype).removeprefix("torch."),
                decorators=[_EAGER_BOOL_READBACK_XFAIL],
            )
            for dtype in (torch.float32, torch.int32, torch.int64)
        ],
    )
    def test_eager_bool_to_dtype_readback(self, dtype):
        # An eager bool converts on device, and the result is read back directly.
        mask = ((torch.arange(4 * 128) % 3) == 0).reshape(4, 128)
        out = mask.to("spyre").to(dtype).cpu()
        torch.testing.assert_close(out, mask.to(dtype), atol=0, rtol=0)

    def test_eager_bool_to_float32_consumed(self):
        # Converting the float32 back to float16 on device restores the order.
        mask = ((torch.arange(4 * 128) % 3) == 0).reshape(4, 128)
        out = (mask.to("spyre").to(torch.float32) * 2).to(torch.float16).cpu()
        expected = (mask.to(torch.float32) * 2).to(torch.float16)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_cross_device_inplace_error_msg(self):
        # Mirrors upstream test_cross_device_inplace_error_msg: the CPU tensor is the write target, so it must raise despite the scalar exemption below.
        a = torch.tensor(2.0)
        b = torch.tensor(2.0, device="spyre")
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            a += b

    def test_cross_device_scalar_op_allowed(self):
        # A single 0-dim CPU tensor as a non-write operand may mix with a spyre tensor (TensorIterator's allow_cpu_scalars_ exemption).
        spyre_t = torch.tensor(2.0, device="spyre")
        cpu_scalar = torch.tensor(3.0)

        result = spyre_t + cpu_scalar
        self.assertEqual(result.device.type, "spyre")
        self.assertEqual(result.to("cpu"), torch.tensor(5.0))

        spyre_t += cpu_scalar
        self.assertEqual(spyre_t.to("cpu"), torch.tensor(5.0))

    def test_cross_device_nonscalar_op_rejected(self):
        # The CPU-scalar exemption only covers 0-dim tensors; a non-scalar CPU tensor must still raise.
        spyre_t = torch.tensor([1.0, 2.0], device="spyre")
        cpu_t = torch.tensor([1.0, 2.0])
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            spyre_t + cpu_t

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="data-dependent output not supported (no aten::masked_select)",
        strict=True,
    )
    def test_data_dependent_output(self):
        cpu_a = torch.randn(10)
        a = cpu_a.to(device="spyre")
        mask = a.gt(0)
        out = torch.masked_select(a, mask)

        self.assertEqual(out, cpu_a.masked_select(cpu_a.gt(0)))

    # simple test to make sure allocation size is different between spyre and cpu
    # this will be built out more once we have an op running in spyre
    # currently this never finishes because of an issue with closing the
    # program -- that will be solved in separate PR
    # (this was tested in isolation)
    def test_allocation_size(self):
        x = torch.tensor([1, 2], dtype=torch.float16, device="spyre")
        y = torch.tensor([1, 2], dtype=torch.float16)
        x_storage_nbytes = x.untyped_storage().nbytes()
        self.assertEqual(x_storage_nbytes, 128)
        self.assertNotEqual(
            x_storage_nbytes,
            y.untyped_storage().nbytes(),
            "spyre vs cpu allocation size should differ",
        )

    def test_spyre_round_trip(self):
        dtypes = [torch.float16]
        for dtype in dtypes:
            x = torch.tensor([1, 2], dtype=dtype)
            self.assertEqual(x.device.type, "cpu", "initial device is not cpu")
            x_spyre = x.to("spyre")
            self.assertEqual(x_spyre.device.type, "spyre", "to device is not spyre")
            x_cpu = x_spyre.to("cpu")
            torch.testing.assert_close(
                x,
                x_cpu,
                msg=f"round trip copy produces incorrect results for dtype={dtype}",
            )

    def test_default_on_import(self):
        import torch_spyre  # noqa: F401

        self.assertTrue(torch.spyre.get_downcast_warning())

    def test_set_get_roundtrip(self):
        import torch_spyre  # noqa: F401

        torch.spyre.set_downcast_warning(False)
        self.assertFalse(torch.spyre.get_downcast_warning())
        torch.spyre.set_downcast_warning(True)
        self.assertTrue(torch.spyre.get_downcast_warning())

    def test_warning_emitted_when_enabled(self):
        import torch_spyre  # noqa: F401

        t = torch.randint(-32768, 32767, (64, 64), dtype=torch.int64)
        torch.spyre.set_downcast_warning(True)
        with set_warn_always_context(True):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                t.to(device="spyre")
        self.assertTrue(
            any("does not support int64" in str(w.message) for w in rec),
            msg="expected int64 downcast warning when enabled",
        )

    def test_warning_suppressed_when_disabled(self):
        import torch_spyre  # noqa: F401

        torch.spyre.set_downcast_warning(False)
        t = torch.randint(-32768, 32767, (64, 64), dtype=torch.int64)
        with set_warn_always_context(True):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                t.to(device="spyre")
        self.assertEqual(len(rec), 0)

    def test_allocation_and_copy_dtypes(self):
        # allocation and device to host cases
        for dtype in [
            torch.float16,
            torch.float32,
            torch.bool,
            torch.int8,
            torch.bfloat16,
        ]:
            x = torch.empty(64, dtype=dtype, device="spyre")
            x.cpu()

        for dtype in [torch.float64]:
            with self.assertRaises(RuntimeError):
                x = torch.empty(64, dtype=dtype, device="spyre")
                x.cpu()

        # allocation and host to device cases
        for dtype in [
            torch.float16,
            torch.float32,
            torch.bool,
            torch.int8,
            torch.bfloat16,
        ]:
            x = torch.empty(64, dtype=dtype)
            x.to("spyre")

        for dtype in [torch.float64]:
            with self.assertRaises(RuntimeError):
                x = torch.empty(64, dtype=dtype)
                x.to("spyre")

    def test_detach(self):
        # exercises the shallow copy code path
        for dtype in [torch.float16, torch.float32, torch.bool, torch.int8]:
            x = torch.empty(64, dtype=dtype, device="spyre")
            x.detach().cpu()

    def test_hooks_on_import(self):
        dev = torch._C._get_accelerator()
        self.assertEqual(str(dev), "spyre")

    def test_memory_allocated(self):
        torch.spyre.memory.reset_peak_memory_stats()
        torch.spyre.memory.reset_accumulated_memory_stats()

        prev_allocated = torch.spyre.memory.memory_allocated()
        prev_max_allocated = torch.spyre.memory.max_memory_allocated()

        self.assertEqual(
            prev_allocated, prev_max_allocated
        )  # Due to reset_peak_memory_stats
        x = torch.rand((64, 64), dtype=torch.float16)
        mem_size = x.numel() * x.element_size()  # 8192 bytes
        self.assertEqual(x.device.type, "cpu")
        self.assertEqual(torch.spyre.memory.memory_allocated(), prev_allocated)

        x = x.to("spyre")
        self.assertEqual(x.device.type, "spyre")
        self.assertEqual(
            torch.spyre.memory.memory_allocated(), prev_allocated + mem_size
        )

        del x
        self.assertEqual(torch.spyre.memory.memory_allocated(), prev_allocated)

        # Test max
        self.assertEqual(
            torch.spyre.memory.max_memory_allocated(), prev_max_allocated + mem_size
        )

    def test_spyre_device_count_and_set_device(self):
        count = torch.spyre.device_count()

        assert isinstance(count, int)
        assert count > 0

        orig = torch.spyre.current_device()

        try:
            for i in range(min(2, count)):
                torch.spyre.set_device(i)
                assert torch.spyre.current_device() == i

            with pytest.raises(Exception):
                torch.spyre.set_device(count)

            with pytest.raises(Exception):
                torch.spyre.set_device(-1)
        finally:
            torch.spyre.set_device(orig)

    def test_instantiate_device_type_tests_mro(self):
        """Verify that instantiate_device_type_tests works with TestCase
        base class and only_for=("privateuse1",).

        Previously, inheriting from PrivateUse1TestBase caused an MRO
        conflict when instantiate_device_type_tests tried to create a
        dynamic subclass that also inherits PrivateUse1TestBase.
        Using plain TestCase + only_for avoids the conflict.
        """
        from torch.testing._internal.common_device_type import (
            instantiate_device_type_tests,
        )

        class _TestMROCheck(TestCase):
            def test_device_is_spyre(self):
                pass

        ns = {"_TestMROCheck": _TestMROCheck}
        # This must not raise TypeError about MRO
        instantiate_device_type_tests(_TestMROCheck, ns, only_for=("privateuse1",))

        # instantiate_device_type_tests creates a class named either
        # _TestMROCheckPRIVATEUSE1 (PT <=2.10) or _TestMROCheckSPYRE (PT 2.11+)
        expected_names = ("_TestMROCheckPRIVATEUSE1", "_TestMROCheckSPYRE")
        found = [n for n in expected_names if n in ns]
        assert found, f"Expected one of {expected_names} in namespace, got {list(ns)}"

        # The generated class should be instantiable (valid MRO)
        cls = ns[found[0]]
        assert issubclass(cls, TestCase)

    def test_device_to_device(self):
        """Test simple device-to-device copy using tensor.copy_() method."""
        src = torch.randn(3, dtype=torch.float16, device="spyre")
        dst = torch.empty(3, dtype=torch.float16, device="spyre")

        dst.copy_(src)

        # Verify the copy worked
        assert torch.allclose(src.cpu(), dst.cpu())
        assert src.data_ptr() != dst.data_ptr()

    def test_device_to_device_with_view(self):
        """Test more complex device-to-device copy using tensor.copy_() method."""
        a = torch.randn(512, 512).to("spyre")
        b = torch.zeros((512, 512), device="spyre")
        c = b.view((64, 8, 512))
        b.copy_(a)
        assert torch.allclose(a.cpu(), b.cpu())
        assert torch.allclose(a.cpu().view(64, 8, 512), c.cpu())

    def test_d2h_copy_of_expanded_view(self):
        """D2H copy of a tensor with stride-0 (broadcast) dims must produce
        the broadcast values, not the underlying storage's contents."""
        src = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16, device="spyre")
        expected = torch.tensor([[1, 2, 3, 4]] * 3, dtype=torch.float16)

        expanded = src.unsqueeze(0).expand(3, 4)
        self.assertFalse(expanded.is_contiguous())
        self.assertEqual(expanded.stride(), (0, 1))
        self.assertEqual(expanded.cpu(), expected)

        # broadcast_to is the alternate API for the same view shape.
        self.assertEqual(src.broadcast_to(3, 4).cpu(), expected)

        # Multiple stride-0 dims (column-broadcast inside row-broadcast).
        col = torch.tensor([10.0, 20.0], dtype=torch.float16, device="spyre")
        wide = col.unsqueeze(1).expand(2, 5)
        self.assertEqual(
            wide.cpu(),
            torch.tensor([[10] * 5, [20] * 5], dtype=torch.float16),
        )

        # Slice then expand: storage_offset != 0 exercises the asymmetry
        # between the on-device alloc view (must read from offset 0) and
        # the CPU-side view (must read from self.storage_offset()).
        base = torch.tensor(
            [float(i) for i in range(20)], dtype=torch.float16, device="spyre"
        )
        sliced_expanded = base[5:9].unsqueeze(0).expand(3, 4)
        self.assertEqual(sliced_expanded.storage_offset(), 5)
        self.assertEqual(
            sliced_expanded.cpu(),
            torch.tensor([[5, 6, 7, 8]] * 3, dtype=torch.float16),
        )

        data = [
            [float(i) for i in range(0, 20)],
            [float(i) for i in range(20, 40)],
            [float(i) for i in range(40, 60)],
            [float(i) for i in range(60, 80)],
        ]

        # Slice then expand to original sizes: destination sizes == source sizes
        # exercises the edge case where the device tensor sizes and dma_sizes
        # have the same number of elements yet the sizes represent a expanded
        # slice and the dma_sizes represent the original data.
        base = torch.tensor(data, dtype=torch.float16, device="spyre")
        sliced_expanded = base[1:2].expand(4, 20)
        self.assertEqual(sliced_expanded.storage_offset(), 20)
        self.assertEqual(
            sliced_expanded.cpu(),
            torch.tensor([data[1]] * 4, dtype=torch.float16),
        )

    def test_d2h_copy_of_strided_slice(self):
        """D2H of a strided slice (e.g. t[::2]) must produce the slice's
        logical values, not over-DMA the parent's full allocation."""
        t = torch.tensor(
            [float(i) for i in range(10)],
            dtype=torch.float16,
            device="spyre",
        )
        sliced = t[::2]
        self.assertEqual(sliced.size(), (5,))
        self.assertEqual(sliced.stride(), (2,))
        self.assertEqual(
            sliced.cpu(),
            torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0], dtype=torch.float16),
        )

    def test_d2h_copy_of_transposed_view(self):
        """Transpose / column-major D2H must NOT be intercepted by the
        realize-on-CPU path; the existing DMA path handles those layouts
        correctly. Guards against re-broadening the trigger to
        !is_contiguous(), which infinite-loops when dma_strides is
        column-major (as produced by some Inductor-codegen outputs)."""
        m = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            dtype=torch.float16,
            device="spyre",
        )
        mt = m.t()
        self.assertFalse(mt.is_contiguous())
        self.assertEqual(mt.stride(), (1, 4))
        self.assertEqual(
            mt.cpu(),
            torch.tensor(
                [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]],
                dtype=torch.float16,
            ),
        )

    def test_scalar_tensor(self):
        """Test to ensure we have scalar tensor on Spyre"""
        scalar = torch.tensor(3.14, dtype=torch.float16, device="spyre")
        assert scalar.dim() == 0

    def test_uint8_transfer_rejected_not_card_fault(self):
        """uint8 host transfers raise a clean error in both directions."""
        # D2H: the direction that faulted on an unfixed build.
        with self.assertRaisesRegex(RuntimeError, "does not support dtype"):
            torch.empty(272, dtype=torch.uint8, device="spyre").cpu()
        # H2D: refused by this gate too, so a wrong host format cannot reach
        # the card in either direction.
        with self.assertRaisesRegex(RuntimeError, "does not support dtype"):
            torch.ones(272, dtype=torch.uint8).to("spyre")

    def test_uint8_allocation_still_allowed(self):
        """Allocating a uint8 tensor on device must keep working.

        Only the host transfer is refused.  torch.distributed allocates uint8
        tensors internally (distributed_c10d.py, used by barrier() and the
        object collectives), so gating is_supported_dtype() on this would
        break the multi-Spyre barrier tests.
        """
        t = torch.empty(272, dtype=torch.uint8, device="spyre")
        self.assertEqual(t.device.type, "spyre")
        self.assertEqual(t.dtype, torch.uint8)

    def test_supported_dtypes_still_transferable(self):
        """The width gate must not reject dtypes whose host width matches.

        int16 is excluded because isDCIConversionSupported has no
        {SENINT16, SENINT16} pair, so its H2D fails on the conversion check
        rather than on this gate.
        """
        for dtype in (
            torch.int8,
            torch.int32,
            torch.int64,
            torch.bool,
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            with self.subTest(dtype=dtype):
                cpu = torch.ones(128, dtype=dtype)
                dev = cpu.to("spyre")
                self.assertEqual(dev.device.type, "spyre")
                self.assertEqual(dev.cpu().dtype, dtype)

    def test_d2h_h2d_dtype_conversion(self):
        """Test H2D (CPU -> Spyre) and D2H (Spyre -> CPU) dtype conversions.

        Tests roundtrip for each supported dtype conversion: create tensor on
        CPU with src_dtype, transfer to Spyre with dst_dtype (H2D), then back
        to CPU (D2H). Also tests that unsupported dtype conversions raise errors.
        """
        from torch_spyre._inductor.dtype_ops import DtypeOpTable

        float_tols = {
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1.6e-2),
            torch.float32: (1e-5, 1e-5),
        }

        def _make_cpu_tensor(dtype):
            if dtype.is_floating_point:
                # FP8 dtypes don't support torch.randn on CPU, create in FP32 first
                if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    return torch.randn(2, 2, dtype=torch.float32).to(dtype=dtype)
                return torch.randn(2, 2, dtype=dtype)
            if dtype == torch.bool:
                return torch.randint(0, 2, (2, 2), dtype=torch.int32).to(torch.bool)
            return torch.randint(0, 10, (2, 2), dtype=dtype)

        def _assert_values_equal(actual, expected, src_dtype, dst_dtype, ctx):
            if actual.is_floating_point():
                src_atol, src_rtol = float_tols.get(src_dtype, (1e-5, 1e-5))
                dst_atol, dst_rtol = float_tols.get(dst_dtype, (1e-5, 1e-5))
                atol = max(src_atol, dst_atol)
                rtol = max(src_rtol, dst_rtol)
                self.assertTrue(
                    torch.allclose(actual, expected, atol=atol, rtol=rtol),
                    msg=f"{ctx}: actual={actual}, expected={expected}",
                )
            else:
                self.assertEqual(actual, expected, msg=ctx)

        # D2H conversions not yet supported on the D2H path.
        skip_d2h_conversions = {
            (torch.float32, torch.float16),
        }

        # DCI doesn't support either direction for these conversions. int64 is
        # IEEE_INT32 on device, so it shares int32's gap.
        skip_eager_conversions = {
            (torch.float32, torch.int32),
            (torch.int32, torch.float32),
            (torch.float32, torch.int64),
            (torch.int64, torch.float32),
        }

        # Test supported conversions
        for src_dtype, dst_dtype in DtypeOpTable.get_table().keys():
            # Skip all FP8 conversions in eager mode - backend doesn't support them:
            # - H2D with conversion TO FP8: "does not support type conversion yet during copy"
            # - H2D with conversion FROM FP8: "Unsupported data format types"
            # - D2H with conversion FROM FP8: does not match with expected output due to "anywhere valid" issue
            # - D2H with conversion TO FP8: does not match with expected output due to "anywhere valid" issue
            # FP8 operations work correctly in torch.compile mode via custom ops
            if src_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or dst_dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
                continue

            if (src_dtype, dst_dtype) in skip_eager_conversions:
                continue

            ctx = f"H2D {src_dtype}->{dst_dtype}"
            h2d_src = _make_cpu_tensor(src_dtype)
            h2d_expected = h2d_src.to(dtype=dst_dtype)  # ground truth on CPU
            h2d_spyre = h2d_src.to("spyre", dtype=dst_dtype)
            self.assertEqual(h2d_spyre.device.type, "spyre", msg=ctx)
            self.assertEqual(h2d_spyre.dtype, dst_dtype, msg=ctx)
            _assert_values_equal(
                h2d_spyre.to("cpu"), h2d_expected, src_dtype, dst_dtype, ctx
            )

            if (dst_dtype, src_dtype) in skip_d2h_conversions:
                continue

            ctx = f"D2H {dst_dtype}->{src_dtype}"
            d2h_src = _make_cpu_tensor(dst_dtype)
            d2h_expected = d2h_src.to(dtype=src_dtype)  # ground truth on CPU
            d2h_spyre = d2h_src.to("spyre")
            d2h_cpu = d2h_spyre.to("cpu", dtype=src_dtype)
            self.assertEqual(d2h_cpu.device.type, "cpu", msg=ctx)
            self.assertEqual(d2h_cpu.dtype, src_dtype, msg=ctx)
            _assert_values_equal(d2h_cpu, d2h_expected, dst_dtype, src_dtype, ctx)

        # Test unsupported conversions
        unsupported_pairs = [
            (torch.int8, torch.float16),
            (torch.float32, torch.int32),
            (torch.float16, torch.int64),
            (torch.int32, torch.float32),
        ]

        for src_dtype, dst_dtype in unsupported_pairs:
            # Create tensor on CPU with src_dtype
            if src_dtype.is_floating_point:
                tensor = torch.randn(2, 2, dtype=src_dtype)
            else:
                tensor = torch.randint(0, 10, (2, 2), dtype=src_dtype)

            # H2D: Expect error when moving to Spyre with unsupported dst_dtype
            with self.assertRaises(RuntimeError):
                tensor.to("spyre", dtype=dst_dtype)

    def test_d2d_dtype_conversion_overloads(self):
        """Dtype-only ``Tensor.to`` overloads use the compiled D2D conversion.

        In particular, ``x.to(torch.float32)`` must recognize its first
        positional argument as a dtype rather than raw-copying bf16 bits into
        an fp32 allocation. D2D conversions produce a staggered element
        arrangement, so convert back to the source dtype before comparing on
        CPU.
        """
        from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout

        pairs = (
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.float16),
            (torch.float32, torch.bfloat16),
        )

        for src_dtype, dst_dtype in pairs:
            base_cpu = torch.linspace(-4, 4, 512, dtype=src_dtype).reshape(4, 128)
            source_cpu = base_cpu[1:3]
            source = base_cpu.to("spyre")[1:3]
            self.assertEqual(source.storage_offset(), 128)
            expected = source_cpu.to(dst_dtype).to(src_dtype)
            other = torch.empty_like(source, dtype=dst_dtype)
            expected_ea = (
                ElementArrangement.DL16_TO_FP32
                if dst_dtype == torch.float32
                else ElementArrangement.FP32_TO_DL16
            )

            conversions = (
                ("positional dtype", lambda: source.to(dst_dtype)),
                ("keyword dtype", lambda: source.to(dtype=dst_dtype)),
                ("device and dtype", lambda: source.to("spyre", dtype=dst_dtype)),
                ("other tensor", lambda: source.to(other)),
            )
            for overload, convert in conversions:
                with self.subTest(
                    src_dtype=src_dtype,
                    dst_dtype=dst_dtype,
                    overload=overload,
                ):
                    actual = convert()
                    self.assertEqual(actual.device.type, "spyre")
                    self.assertEqual(actual.dtype, dst_dtype)
                    self.assertEqual(
                        get_spyre_tensor_layout(actual).element_arrangement,
                        expected_ea,
                    )
                    restored = actual.to(src_dtype)
                    self.assertEqual(
                        get_spyre_tensor_layout(restored).element_arrangement,
                        ElementArrangement.STANDARD,
                    )
                    atol, rtol = (
                        (1e-2, 1.6e-2)
                        if torch.bfloat16 in (src_dtype, dst_dtype)
                        else (1e-3, 1e-3)
                    )
                    torch.testing.assert_close(
                        restored.cpu(), expected, atol=atol, rtol=rtol
                    )

    def test_unsupported_d2d_dtype_conversion_falls_back(self):
        """Unsupported native D2D pairs retain the CPU fallback behavior."""
        from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout
        from torch_spyre.ops.fallbacks import FallbackWarning

        # int8 has a device format but no conversion op.
        source_cpu = (torch.arange(128) % 100).to(torch.int8).reshape(2, 64)
        source = source_cpu.to("spyre")

        with self.assertWarnsRegex(
            FallbackWarning,
            r"conversion from torch\.int8 to torch\.float16 is falling back to cpu",
        ):
            actual = source.to(torch.float16)

        self.assertEqual(actual.dtype, torch.float16)
        self.assertEqual(
            get_spyre_tensor_layout(actual).element_arrangement,
            ElementArrangement.STANDARD,
        )
        torch.testing.assert_close(actual.cpu(), source_cpu.to(torch.float16))

    @parametrize(
        "sizes",
        [
            [4, 8, 16, 60],
            [8, 10, 30, 64],
            [9, 15, 35, 100],
        ],
    )
    def test_h2d_copy_to_sliced_destination(self, sizes):
        """H2D to a slice (e.g. t[:2]) must write to the slice's logical
        value."""
        x = torch.rand(
            sizes,
            dtype=torch.float16,
            device="spyre",
        )

        slice_sizes = [2, sizes[1], sizes[2], sizes[3]]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[:2] = y
        self.assertEqual(x.cpu()[:2], y, atol=1e-3, rtol=1e-3)

        slice_sizes = [1, sizes[1], sizes[2], sizes[3]]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[2:3] = y
        self.assertEqual(x.cpu()[2:3], y, atol=1e-3, rtol=1e-3)

        slice_sizes = [sizes[1], sizes[2], sizes[3]]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3] = y
        self.assertEqual(x.cpu()[3], y, atol=1e-3, rtol=1e-3)

        slice_sizes = [2, sizes[2], sizes[3]]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3, :2] = y
        self.assertEqual(x.cpu()[3, :2], y, atol=1e-3, rtol=1e-3)

        slice_sizes = [1, sizes[2], sizes[3]]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3, 2:3] = y
        self.assertEqual(x.cpu()[3, 2:3], y, atol=1e-3, rtol=1e-3)

        slice_sizes = [sizes[2], sizes[3]]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3, 3] = y
        self.assertEqual(x.cpu()[3, 3], y, atol=1e-3, rtol=1e-3)

        slice_sizes = [2, sizes[3]]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3, 3, :2] = y
        self.assertEqual(x.cpu()[3, 3, :2], y, atol=1e-3, rtol=1e-3)

        slice_sizes = [1, sizes[3]]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3, 3, 2:3] = y
        self.assertEqual(x.cpu()[3, 3, 2:3], y, atol=1e-3, rtol=1e-3)

        slice_sizes = [sizes[3]]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3, 3, 3] = y
        self.assertEqual(x.cpu()[3, 3, 3], y, atol=1e-3, rtol=1e-3)

    @parametrize(
        "sizes",
        [
            [4, 8, 16, 60],
            [8, 10, 30, 64],
            [9, 15, 35, 100],
        ],
    )
    @pytest.mark.xfail(
        strict=False,  # Allows XPASS for now, will be changed to True to enforce later
        raises=AssertionError,
        reason="Sub-stick slices are not yet supported for DMA",
    )
    def test_h2d_copy_to_substick_sliced_destination(self, sizes):
        """H2D sub-stick tensor slice"""
        x = torch.rand(
            sizes,
            dtype=torch.float16,
            device="spyre",
        )

        slice_sizes = [2]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3, 3, 3, :2] = y
        self.assertEqual(x.cpu()[3, 3, 3, :2], y, atol=1e-3, rtol=1e-3)

        slice_sizes = [1]
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3, 3, 3, 2:3] = y
        self.assertEqual(x.cpu()[3, 3, 3, 2:3], y, atol=1e-3, rtol=1e-3)

        slice_sizes = []
        y = torch.rand(
            slice_sizes,
            dtype=torch.float16,
            device="cpu",
        )
        x[3, 3, 3, 3] = y
        self.assertEqual(x.cpu()[3, 3, 3, 3], y, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()

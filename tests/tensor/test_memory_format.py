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

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.spyre import SpyreTensorLayout


class TestSpyreEmptyMemoryFormat(TestCase):
    """Memory-format guard tests for spyre_empty (aten::empty.memory_format dispatch)."""

    def test_channels_last_raises(self) -> None:
        """channels_last must raise — regression guard for issue #2175."""
        with self.assertRaisesRegex(
            RuntimeError,
            "Spyre backend only supports contiguous memory format",
        ):
            torch.empty(
                (1, 3, 4, 4),
                device="spyre",
                dtype=torch.float16,
                memory_format=torch.channels_last,
            )

    def test_contiguous_format_accepted(self) -> None:
        """Default contiguous allocation must not raise."""
        t = torch.empty((2, 64), device="spyre", dtype=torch.float16)
        self.assertEqual(t.device.type, "spyre")
        self.assertTrue(t.is_contiguous())


class TestEmptyWithLayoutMemoryFormat(TestCase):
    """Memory-format guard tests for empty_with_layout (torch.empty + device_layout path)."""

    def _make_layout(self, size: list) -> SpyreTensorLayout:
        return SpyreTensorLayout(size, torch.float16)

    def test_channels_last_raises(self) -> None:
        """channels_last must raise when device_layout is supplied."""
        stl = self._make_layout([1, 3, 4, 4])
        with self.assertRaisesRegex(
            RuntimeError,
            "Spyre backend only supports contiguous memory format",
        ):
            torch.empty(
                (1, 3, 4, 4),
                device_layout=stl,
                dtype=torch.float16,
                memory_format=torch.channels_last,
            )

    def test_preserve_format_accepted(self) -> None:
        """preserve_format is explicitly allowed alongside contiguous with device_layout."""
        stl = self._make_layout([2, 64])
        t = torch.empty(
            (2, 64),
            device_layout=stl,
            dtype=torch.float16,
            memory_format=torch.preserve_format,
        )
        self.assertEqual(t.device.type, "spyre")
        self.assertTrue(t.is_contiguous())


if __name__ == "__main__":
    run_tests()

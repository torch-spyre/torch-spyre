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

# Owner(s): ["module: stream"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSpyreStream(TestCase):
    """Test cases for Spyre stream functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("spyre")

    def test_legacy_stream_creation_with_device(self):
        """Test creating stream using legacy torch.spyre.Stream API with device."""
        stream = torch.spyre.Stream(self.device)
        self.assertIsNotNone(stream)
        self.assertIsInstance(stream, torch.spyre.Stream)

    def test_legacy_stream_creation_with_priority(self):
        """Test creating stream using legacy API with priority only."""
        stream = torch.spyre.Stream(priority=-1)
        self.assertIsNotNone(stream)
        self.assertIsInstance(stream, torch.spyre.Stream)

    def test_legacy_stream_creation_with_device_and_priority(self):
        """Test creating stream using legacy API with both device and priority."""
        stream = torch.spyre.Stream(self.device, priority=-1)
        self.assertIsNotNone(stream)
        self.assertIsInstance(stream, torch.spyre.Stream)

    def test_legacy_current_stream(self):
        """Test querying current stream using legacy API."""
        current_stream = torch.spyre.current_stream()
        self.assertIsNotNone(current_stream)
        self.assertIsInstance(current_stream, torch.spyre.Stream)

    def test_modern_stream_creation(self):
        """Test creating stream using modern torch.Stream API."""
        stream = torch.Stream(self.device)
        self.assertIsNotNone(stream)

    def test_modern_stream_creation_with_priority(self):
        """Test creating stream using modern API with high priority."""
        stream = torch.Stream(self.device, priority=-1)
        self.assertIsNotNone(stream)

    def test_modern_stream_context_manager(self):
        """Test using stream as context manager."""
        with torch.Stream(self.device):
            # Create tensor within stream context
            a = torch.randn(1, 32, device="spyre")
            self.assertEqual(a.device.type, "spyre")
            self.assertEqual(a.shape, (1, 32))

    def test_modern_current_stream(self):
        """Test querying current stream using modern accelerator API."""
        current_stream = torch.accelerator.current_stream()
        self.assertIsNotNone(current_stream)

    def test_stream_synchronize(self):
        """Test stream synchronization."""
        stream = torch.Stream(self.device)
        # Perform operation on stream
        with stream:
            _ = torch.randn(10, 10, device="spyre")
        # Synchronize the stream
        stream.synchronize()
        self.assertTrue(stream.query())

    def test_device_synchronize(self):
        """Test device-level synchronization."""
        stream1 = torch.Stream(self.device)
        stream2 = torch.Stream(self.device)

        with stream1:
            a = torch.randn(10, 10, device="spyre")  # noqa: F841

        with stream2:
            b = torch.randn(10, 10, device="spyre")  # noqa: F841

        # Synchronize all streams on the device
        torch.accelerator.synchronize(self.device)
        self.assertTrue(stream1.query())
        self.assertTrue(stream2.query())

    def test_global_synchronize(self):
        """Test global synchronization across all devices."""
        stream = torch.Stream(self.device)
        with stream:
            a = torch.randn(10, 10, device="spyre")  # noqa: F841

        # Synchronize all streams on all devices
        torch.accelerator.synchronize()
        self.assertTrue(stream.query())

    def test_multiple_streams(self):
        """Test creating and using multiple streams."""
        stream1 = torch.Stream(self.device)
        stream2 = torch.Stream(self.device, priority=-1)
        stream3 = torch.Stream(self.device, priority=0)

        self.assertIsNotNone(stream1)
        self.assertIsNotNone(stream2)
        self.assertIsNotNone(stream3)

    def test_stream_operations_isolation(self):
        """Test that operations on different streams are isolated."""
        stream1 = torch.Stream(self.device)
        stream2 = torch.Stream(self.device)

        with stream1:
            a = torch.randn(5, 5, device="spyre")
            result1 = a + a

        with stream2:
            b = torch.randn(5, 5, device="spyre")
            result2 = b + b

        # Synchronize both streams
        stream1.synchronize()
        stream2.synchronize()
        self.assertTrue(stream1.query())
        self.assertTrue(stream2.query())

        self.assertEqual(result1.shape, (5, 5))
        self.assertEqual(result2.shape, (5, 5))

    def test_nested_stream_contexts(self):
        """Test nested stream context managers."""
        stream1 = torch.Stream(self.device)
        stream2 = torch.Stream(self.device)

        with stream1:
            a = torch.randn(3, 3, device="spyre")
            with stream2:
                b = torch.randn(3, 3, device="spyre")
                c = a + b

        stream1.synchronize()
        stream2.synchronize()
        self.assertTrue(stream1.query())
        self.assertTrue(stream2.query())
        self.assertEqual(c.shape, (3, 3))

    def test_stream_priority_levels(self):
        """Test creating streams with different priority levels."""
        high_priority = torch.Stream(self.device, priority=-1)
        normal_priority = torch.Stream(self.device, priority=0)
        low_priority = torch.Stream(self.device, priority=1)

        self.assertIsNotNone(high_priority)
        self.assertIsNotNone(normal_priority)
        self.assertIsNotNone(low_priority)

    def test_stream_with_tensor_operations(self):
        """Test various tensor operations within stream context."""
        stream = torch.Stream(self.device)

        with stream:
            # Create tensors
            a = torch.randn(10, 64, dtype=torch.float16, device="spyre")
            b = torch.randn(10, 64, dtype=torch.float16, device="spyre")

            # Perform operations
            c = a + b
            d = torch.add(a, b)

            self.assertEqual(c.shape, (10, 64))
            self.assertEqual(d.shape, (10, 64))

        stream.synchronize()
        self.assertTrue(stream.query())

    def test_default_stream(self):
        """Test that default stream exists and is accessible."""
        default_stream = torch.spyre.current_stream()
        self.assertIsNotNone(default_stream)

        # Operations without explicit stream should use default stream
        a = torch.randn(5, 5, device="spyre")
        self.assertEqual(a.device.type, "spyre")

    def test_stream_query_after_context(self):
        """Test querying stream after exiting context."""
        stream = torch.Stream(self.device)

        with stream:
            current = torch.accelerator.current_stream()
            # Current stream should be our stream within context
            self.assertIsNotNone(current)

        # After exiting context, should return to default stream
        current_after = torch.accelerator.current_stream()
        self.assertIsNotNone(current_after)

    def test_stream_reuse(self):
        """Test reusing the same stream multiple times."""
        stream = torch.Stream(self.device)

        # First use
        with stream:
            a = torch.randn(5, 5, device="spyre")
        stream.synchronize()
        self.assertTrue(stream.query())

        # Second use
        with stream:
            b = torch.randn(5, 5, device="spyre")
        stream.synchronize()
        self.assertTrue(stream.query())

        self.assertEqual(a.shape, (5, 5))
        self.assertEqual(b.shape, (5, 5))


class TestStreamCompatibility(TestCase):
    """Test compatibility between legacy and modern stream APIs."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("spyre")

    def test_legacy_and_modern_interop(self):
        """Test that legacy and modern APIs can work together."""
        # Create stream using legacy API
        legacy_stream = torch.spyre.Stream(self.device)

        # Create stream using modern API
        modern_stream = torch.Stream(self.device)

        # Both should be valid
        self.assertIsNotNone(legacy_stream)
        self.assertIsNotNone(modern_stream)

    def test_current_stream_consistency(self):
        """Test that current_stream returns consistent results."""
        # Query using legacy API
        legacy_current = torch.spyre.current_stream()

        # Query using modern API
        modern_current = torch.accelerator.current_stream()

        # Both should return valid streams
        self.assertIsNotNone(legacy_current)
        self.assertIsNotNone(modern_current)


if __name__ == "__main__":
    run_tests()

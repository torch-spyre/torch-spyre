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

"""Tests for SpyreGenerator Python bindings and Random Number Generation."""

import pytest
import torch
import torch_spyre


class TestSpyreGenerator:
    def test_generator_creation(self):
        """Test creating a Spyre generator."""
        gen = torch.Generator(device="spyre")
        assert gen.device.type == "spyre"
        assert gen is not None

    def test_default_generator_is_cached(self):
        """Test that repeated calls to _get_default_generator return the same object."""

        default_gen1 = torch_spyre._C._get_default_generator()
        default_gen2 = torch_spyre._C._get_default_generator()

        # Verify each generator is on the expected device before comparing identity
        assert default_gen1.device == torch.device("spyre:0")
        assert default_gen2.device == torch.device("spyre:0")
        assert default_gen1 is default_gen2

    @pytest.mark.skipif(
        torch.spyre.device_count() < 2,
        reason="Requires at least 2 Spyre devices",
    )
    def test_default_generators_are_distinct_across_devices(self):
        """Test that default generators for different Spyre devices are distinct objects."""

        gen0 = torch_spyre._C._get_default_generator(0)
        gen1 = torch_spyre._C._get_default_generator(1)

        # Core invariant: each device has its own generator instance
        assert gen0 is not gen1

        # Each generator must be bound to the correct device
        assert gen0.device == torch.device("spyre:0")
        assert gen1.device == torch.device("spyre:1")

    def test_manual_seed(self):
        """Test manual_seed sets the seed correctly and produces reproducible states."""
        gen = torch.Generator(device="spyre")
        seed = 42
        gen.manual_seed(seed)

        # Verify the seed was set correctly
        assert gen.initial_seed() == seed

        # Get state after setting seed
        state1 = gen.get_state()

        # Reset to same seed
        gen.manual_seed(seed)
        state2 = gen.get_state()

        # States should be identical when reset to same seed
        torch.testing.assert_close(state1, state2)

    def test_get_state(self):
        """Test get_state returns a valid state tensor."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(42)

        state = gen.get_state()

        # State should be a CPU byte tensor
        assert state.device.type == "cpu"
        assert state.dtype == torch.uint8
        assert state.numel() > 0

    def test_set_state(self):
        """Test set_state restores generator state."""
        gen1 = torch.Generator(device="spyre")
        gen1.manual_seed(456)

        # Save the state
        state = gen1.get_state()

        # Create new generator with different seed
        gen2 = torch.Generator(device="spyre")
        gen2.manual_seed(999)

        # Restore state from gen1
        gen2.set_state(state)

        # States should now match
        state1 = gen1.get_state()
        state2 = gen2.get_state()
        torch.testing.assert_close(state1, state2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

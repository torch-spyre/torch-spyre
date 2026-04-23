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

# Owner(s): ["module: cpp"]

import os
import subprocess
import sys
from typing import Optional

import pytest


def get_device_count_in_subprocess(env_vars: Optional[dict] = None) -> int:
    """
    Run device_count() in an isolated subprocess with specific env vars.

    This bypasses the std::call_once caching in getVisibleDevices()
    by running each test in a fresh process.
    """
    code = """
import torch # noqa: F401
print(torch.spyre.device_count())
"""
    env = os.environ.copy()
    if env_vars is not None:
        env.update(env_vars)
        # Remove env vars that might interfere if not specified
        for key in ["SPYRE_DEVICES", "AIU_WORLD_SIZE"]:
            if key not in env_vars:
                env.pop(key, None)

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    return int(result.stdout.strip())


class TestDeviceEnumEnvVars:
    """Test environment variable handling for device enumeration."""

    def test_no_env_vars(self):
        """Test No env vars → returns total devices."""
        count = get_device_count_in_subprocess({})
        assert count > 0

    def test_aiu_world_size_smaller_equal_total(self):
        """Test AIU_WORLD_SIZE → returns AIU_WORLD_SIZE."""
        count = get_device_count_in_subprocess({"AIU_WORLD_SIZE": "1"})
        assert count == 1

    def test_aiu_world_size_larger_total(self):
        """Test AIU_WORLD_SIZE larger than total devices → returns total devices."""
        total = get_device_count_in_subprocess({})
        aiu_world_size = total + 1
        count = get_device_count_in_subprocess({"AIU_WORLD_SIZE": str(aiu_world_size)})
        assert count == total

    def test_spyre_visible_devices_single(self):
        """Test SPYRE_DEVICES with single index."""
        count = get_device_count_in_subprocess({"SPYRE_DEVICES": "0"})
        assert count == 1

    @pytest.mark.parametrize(
        "devices_str,expected_count,min_devices,description",
        [
            ("0,1", 2, 2, "basic multiple devices"),
            ("0,1,1", 2, 2, "duplicates should be deduplicated"),
            ("1,0", 2, 2, "out-of-order indices"),
            ("0, {total_plus_one}", 1, 2, "out-of-range index should be filtered out"),
        ],
    )
    def test_spyre_visible_devices_multiple(
        self, devices_str, expected_count, min_devices, description
    ):
        """Test SPYRE_DEVICES with various scenarios."""
        # Check if we have enough devices for this test
        total = get_device_count_in_subprocess({})
        if total < min_devices:
            pytest.skip(f"Need at least {min_devices} devices for: {description}")

        resolved_devices_str = devices_str.format(total_plus_one=total + 1)
        count = get_device_count_in_subprocess({"SPYRE_DEVICES": resolved_devices_str})
        assert count == expected_count, (
            f"SPYRE_DEVICES='{resolved_devices_str}' ({description}) "
            f"should return {expected_count}, got {count}"
        )

    @pytest.mark.parametrize(
        "aiu_world_size,spyre_devices,expected_count,min_devices,description",
        [
            ("2", "0", 1, 2, "AIU_WORLD_SIZE=2, SPYRE_DEVICES=0"),
            ("2", "0,1", 2, 2, "AIU_WORLD_SIZE=2, SPYRE_DEVICES=0,1"),
        ],
    )
    def test_aiu_world_size_with_spyre_devices(
        self, aiu_world_size, spyre_devices, expected_count, min_devices, description
    ):
        """Test AIU_WORLD_SIZE with SPYRE_DEVICES combinations."""
        total = get_device_count_in_subprocess({})
        if total < min_devices:
            pytest.skip(f"Need at least {min_devices} devices for: {description}")

        count = get_device_count_in_subprocess(
            {"AIU_WORLD_SIZE": aiu_world_size, "SPYRE_DEVICES": spyre_devices}
        )
        assert count == expected_count, (
            f"{description} should return {expected_count}, got {count}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

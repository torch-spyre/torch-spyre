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
from pathlib import Path
from typing import Optional

import pytest


# Spyre PCI vendor/device IDs (must match spyre_device_enum.cpp)
SPYRE_VENDOR_ID = 0x1014  # IBM
SPYRE_DEVICE_ID = 0x06A7  # Spyre Accelerator


def discover_spyre_pci_bus_ids() -> list[str]:
    """
    Discover Spyre PCI bus IDs by scanning /sys/bus/pci/devices/.

    Returns list of PCI bus IDs (e.g., ['0000:29:00.0', '0000:3c:00.0'])
    """
    sysfs_path = Path("/sys/bus/pci/devices")
    if not sysfs_path.exists():
        return []

    bus_ids = []
    for device_dir in sysfs_path.iterdir():
        try:
            vendor_file = device_dir / "vendor"
            device_file = device_dir / "device"

            if vendor_file.exists() and device_file.exists():
                vendor = int(vendor_file.read_text().strip(), 16)
                device = int(device_file.read_text().strip(), 16)

                if vendor == SPYRE_VENDOR_ID and device == SPYRE_DEVICE_ID:
                    bus_ids.append(device_dir.name)
        except (OSError, ValueError):
            continue

    return bus_ids


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
        for key in ["SPYRE_VISIBLE_DEVICES", "PCIDEVICE_IBM_COM_AIU_PF"]:
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

    def test_default_pci_scan(self):
        """Test default PCI bus scan - verifies hardware is detected."""
        count = get_device_count_in_subprocess({})
        assert count > 0, "No Spyre devices found via PCI scan"

    def test_spyre_visible_devices_single(self):
        """Test SPYRE_VISIBLE_DEVICES with single index."""
        count = get_device_count_in_subprocess({"SPYRE_VISIBLE_DEVICES": "0"})
        assert count == 1

    def test_spyre_visible_devices_multiple(self):
        """Test SPYRE_VISIBLE_DEVICES with multiple indices."""
        # Check total count first
        total = get_device_count_in_subprocess({})
        if total < 2:
            pytest.skip("Need at least 2 devices")

        count = get_device_count_in_subprocess({"SPYRE_VISIBLE_DEVICES": "0,1"})
        assert count == 2

    def test_k8s_pci_device_env(self):
        """Test PCIDEVICE_IBM_COM_AIU_PF (K8s device plugin)."""
        bus_ids = discover_spyre_pci_bus_ids()
        if not bus_ids:
            pytest.skip("No Spyre devices found via sysfs scan")

        # Test with single PCI bus ID
        count = get_device_count_in_subprocess({"PCIDEVICE_IBM_COM_AIU_PF": bus_ids[0]})
        assert count == 1

    def test_env_var_priority(self):
        """SPYRE_VISIBLE_DEVICES takes priority over PCIDEVICE_IBM_COM_AIU_PF."""
        count = get_device_count_in_subprocess(
            {
                "SPYRE_VISIBLE_DEVICES": "0",
                "PCIDEVICE_IBM_COM_AIU_PF": "invalid_bus_id",
            }
        )
        assert count == 1, "SPYRE_VISIBLE_DEVICES should take priority"

    def test_k8s_invalid_pci_bus_id_filtered(self):
        """Invalid PCI bus IDs are filtered out, count equals valid IDs only."""
        bus_ids = discover_spyre_pci_bus_ids()
        if not bus_ids:
            pytest.skip("No Spyre devices found via sysfs scan")

        # Use first valid PCI bus ID + append an invalid one
        valid_id = bus_ids[0]
        env_val = f"{valid_id},0000:ff:ff.ff"
        count = get_device_count_in_subprocess({"PCIDEVICE_IBM_COM_AIU_PF": env_val})

        # Count should be 1 (invalid ID filtered out)
        assert count == 1, f"Expected 1 (invalid ID filtered), got {count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spyre {

// PCI vendor/device IDs for IBM Spyre Accelerator
constexpr uint16_t kSpyreVendorId = 0x1014;  // IBM
constexpr uint16_t kSpyreDeviceId = 0x06a7;  // Spyre Accelerator

// PCI bus ID, e.g. "0000:29:00.0"
struct SpyreDeviceInfo {
  std::string pci_bus_id;
  int index;  // logical index (0-based)
};

// Returns the list of Spyre devices visible to this process.
//
// Priority:
//   1. SPYRE_VISIBLE_DEVICES env var — comma-separated PCI bus IDs or
//      0-based indices (e.g. "0,1,2" or "0000:29:00.0,0000:2a:00.0")
//   2. PCIDEVICE_IBM_COM_AIU_PF env var — set by K8s device plugin,
//      comma-separated PCI bus IDs
//   3. Full PCI bus scan via /sys/bus/pci/devices/
//
// The result is cached after the first call.
const std::vector<SpyreDeviceInfo>& getVisibleDevices();

// Convenience: returns the number of visible Spyre devices.
int getVisibleDeviceCount();

// If SPYRE_DEVICES is not already set, and PCIDEVICE_IBM_COM_AIU_PF is
// available, synthesize SPYRE_DEVICES from the K8s-assigned PCI bus IDs
// so that flex picks the correct physical cards.
//
// Must be called before flex::initializeRuntime().
void ensureSpyreDevicesEnv();

}  // namespace spyre

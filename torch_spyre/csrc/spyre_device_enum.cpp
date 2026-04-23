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

#include "spyre_device_enum.h"

#include <flex/runtime_stream/runtime_entry.hpp>

namespace spyre {

int getVisibleDeviceCount() {
  // Get the number of available Spyre devices from the flex runtime.
  //
  // This function queries flex::getNumDevices() which determines device count
  // by:
  //
  // 1. For FLEX_DEVICE=PF or VF:
  //    - Reads total devices in the system without opening any device
  //    - Adjusts based on AIU_WORLD_SIZE if set (must be <= total devices)
  //    - If SPYRE_DEVICES is set, parses comma-separated device indices
  //      and returns the count of valid, unique devices specified
  //
  // 2. For FLEX_DEVICE=MOCK:
  //    - Returns AIU_WORLD_SIZE environment variable value
  //    - Throws error if AIU_WORLD_SIZE < 1
  //
  // Environment Variables:
  // - AIU_WORLD_SIZE: Number of devices to use (set by the login script and
  //   matches total devices in most use cases)
  // - SPYRE_DEVICES: Comma-separated list of device indices to use (e.g.,
  //   "0,2,3")
  // - FLEX_DEVICE: Device type ("PF", "VF", or "MOCK")
  //
  // @return Number of visible Spyre devices for this process
  return static_cast<int>(flex::getNumDevices());
}

}  // namespace spyre

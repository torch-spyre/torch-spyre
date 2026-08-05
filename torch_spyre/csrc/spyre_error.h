/*
 * Copyright 2025 The Torch-Spyre Authors.
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

namespace spyre {

class SpyreStream;

// Per-stream error state. Values 2-5 (HardwareFault, Timeout, MemoryFault,
// CoreUnavailable) are reserved pending flex::RuntimeStream typed error codes.
enum class SpyreStreamError : int {
  Success = 0,   // Stream is healthy (not shut down; deferred errors surface
                 // only on synchronize())
  Shutdown = 1,  // Stream has been shut down (unrecoverable)
};

// Device-level aggregate over all streams in the global runtime.
enum class SpyreDeviceState : int {
  Ok = 0,              // All streams healthy; device is usable
  NotInitialized = 1,  // Runtime not yet created — treat as Ok, proceed
  StreamError = 2,     // One or more streams are in an unrecoverable state
};

/**
 * @brief Returns a string literal for a SpyreStreamError value.
 * @returns A non-null string such as "Success" or "Shutdown". Never throws.
 */
const char* SpyreStreamGetErrorString(SpyreStreamError error) noexcept;

/**
 * @brief Query the error state of a single stream.
 * @returns SpyreStreamError::Shutdown if the stream has been shut down,
 *          SpyreStreamError::Success otherwise.
 * @throws if the runtime has not been started or the stream pool is not
 *         initialized for this stream's device.
 */
SpyreStreamError SpyreStreamGetError(const SpyreStream& stream);

/**
 * @brief Return the device-level aggregate state.
 * @returns SpyreDeviceState::NotInitialized if the runtime has not been
 *          created yet (callers should treat this as proceed / no error),
 *          SpyreDeviceState::StreamError if any stream is unrecoverable,
 *          SpyreDeviceState::Ok otherwise.
 */
SpyreDeviceState SpyreGetDeviceState();

}  // namespace spyre

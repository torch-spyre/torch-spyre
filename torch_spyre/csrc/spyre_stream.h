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

#include <ATen/ATen.h>
#include <c10/core/Stream.h>

#include <flex/stream_handle.hpp>

#include "module.h"

namespace spyre {
class SpyreStream {
 private:
  c10::Stream stream_;

 public:
  explicit SpyreStream(c10::Stream stream);
  SpyreStream();

  // Stream properties
  c10::StreamId id() const;
  c10::Device device() const;
  int priority() const;

  // Synchronization
  bool query() const;        // Check if work completed
  void synchronize() const;  // Block until work done

  void copy_async(const at::Tensor& src, const at::Tensor& dst) const;

  // Conversions
  c10::Stream unwrap() const;

 private:
  flex::StreamHandle get_flex_handle() const;
  void copy_async_impl(void* cpu_ptr,
                       flex::DeviceMemoryAllocationPtr& device_allocation,
                       int device_id, const DataConversionInfo& dci,
                       bool host2device) const;
};

/**
 * Get a stream from the global stream pool.
 * Streams are allocated round-robin from a pool maintained by torch-spyre.
 *
 * @param device Device to allocate stream on (default: current device)
 * @param priority Stream priority: 0 (normal) or -1 (high)
 * @return A SpyreStream from the pool
 */
SpyreStream getStreamFromPool(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1),
    int priority = 0);

/**
 * Get the default stream for a device.
 * The default stream is stream ID 0 and is always available.
 *
 * @param device Device to get default stream for
 * @return The default SpyreStream (stream ID 0)
 */
SpyreStream getDefaultStream(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1));

/**
 * Get the current stream for a device (thread-local).
 * Each thread maintains its own current stream per device.
 *
 * @param device Device to get current stream for
 * @return The current SpyreStream for this thread
 */
SpyreStream getCurrentStream(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1));

/**
 * Set the current stream for a device (thread-local).
 *
 * @param stream Stream to set as current
 * @return The previous current stream
 */
SpyreStream setCurrentStream(SpyreStream stream);

void synchronizeDevice(c10::optional<c10::Device> device);

}  // namespace spyre

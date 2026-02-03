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

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spyre {

namespace {

// C++ hooks implementation
struct SpyreHooksArgs : public at::PrivateUse1HooksArgs {};

struct SpyreHooksInterface : public at::PrivateUse1HooksInterface {
  SpyreHooksInterface() = default;
  explicit SpyreHooksInterface(SpyreHooksArgs) {}
  ~SpyreHooksInterface() override = default;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    return true;
  }
  bool isAvailable() const override {
    return true;
  }
};

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, SpyreHooksInterface,
                       SpyreHooksArgs);
C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, SpyreHooksInterface,
                    SpyreHooksArgs);
// Using Create function to get PrivateUse1HooksInterface point from
// PrivateUse1HooksRegistry class.
C10_REGISTER_TYPED_CLASS(PrivateUse1HooksRegistry, "SpyreHooks",
                         SpyreHooksInterface);

PYBIND11_MODULE(_hooks, m) {
  static auto* hooks = new SpyreHooksInterface();
  at::RegisterPrivateUse1HooksInterface(hooks);
  m.doc() =
      "Spyre bootstrap: registers PrivateUse1 hooks only (no heavy init).";
}

c10::Device current_device = c10::Device(c10::DeviceType::PrivateUse1, 0);

// Device guard registration
struct SpyreGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  SpyreGuardImpl() = default;
  explicit SpyreGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == static_type);
  }

  /**
   * Return the type of device managed by this guard implementation.
   */
  c10::DeviceType type() const override {
    return static_type;
  }

  /**
   * Set the current device to Device, and return the previous c10::Device.
   */
  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    auto old_device_index = current_device.index();
    return c10::Device(static_type, old_device_index);
  }

  /**
   * Get the current device.
   */
  c10::Device getDevice() const override {
    py::gil_scoped_acquire acquire;
    return c10::Device(static_type, 0);
  }

  /**
   * Set the current device to c10::Device.
   */
  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    current_device = c10::Device(static_type, d.index());
  }

  /**
   * Set the current device to c10::Device, without checking for errors
   * (so, e.g., this can be called from a destructor).
   */
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    current_device = c10::Device(static_type, d.index());
  }

  /**
   * Get the current stream for a given device.
   */
  c10::Stream getStream(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    // FIXME: This is just assuming 0 - ControlBlockStream
    return c10::Stream(c10::Stream::UNSAFE, current_device, 0);
  }

  /**
   * Get the default stream for a given device.
   */
  c10::Stream getDefaultStream(c10::Device d) const override {
    py::gil_scoped_acquire acquire;
    // FIXME: This is just assuming 0 - ControlBlockStream
    return c10::Stream(c10::Stream::UNSAFE, current_device, 0);
  }

  /**
   * Get a stream from the global pool for a given device.
   */
  c10::Stream getStreamFromGlobalPool(
      c10::Device d, bool isHighPriority = false) const override {
    py::gil_scoped_acquire acquire;
    // FIXME: This is just assuming 0 - ControlBlockStream
    return c10::Stream(c10::Stream::UNSAFE, current_device, 0);
  }

  /**
   * Return a new stream for a given device and priority. The stream will be
   * copied and shared around, device backend should be able to correctly handle
   * the lifetime of the stream.
   */
  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    py::gil_scoped_acquire acquire;
    return c10::Stream(c10::Stream::UNSAFE, current_device, 0);
  }

  /**
   * Set a stream to be the thread local current stream for its device.
   * Return the previous stream for that device. You are NOT required
   * to set the current device to match the device of this stream.
   */
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    py::gil_scoped_acquire acquire;
    return c10::Stream(c10::Stream::UNSAFE, current_device, 0);
  }

  /**
   * Destroys the given event.
   */
  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    py::gil_scoped_acquire acquire;
    // not implemented on spyre - do nothing
  }

  /**
   * Increments the event's version and enqueues a job with this version
   * in the stream's work queue. When the stream process that job
   * it notifies all streams waiting on / blocked by that version of the
   * event to continue and marks that version as recorded.
   * */
  void record(void** event, const c10::Stream& stream,
              const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override {
    py::gil_scoped_acquire acquire;
    // not implemented on spyre - do nothing
  }

  /**
   * Does nothing if the event has not been scheduled to be recorded.
   * If the event was previously enqueued to be recorded, a command
   * to wait for the version of the event that exists at the time of this call
   * is inserted in the stream's work queue.
   * When the stream reaches this command it will stop processing
   * additional commands until that version of the event is marked as recorded.
   */
  void block(void* event, const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    // not implemented on spyre - do nothing
  }

  /**
   * Returns true if (and only if)
   *  (1) the event has never been scheduled to be recorded
   *  (2) the current version is marked as recorded.
   * Returns false otherwise.
   */
  bool queryEvent(void* event) const override {
    py::gil_scoped_acquire acquire;
    // not implemented on spyre - do nothing
    return false;
  }

  /**
   * Get the number of devices.  WARNING: This is REQUIRED to not raise
   * an exception.  If there is some sort of problem, e.g., driver error,
   * you should report that there are zero available devices.
   */
  c10::DeviceIndex deviceCount() const noexcept override {
    py::gil_scoped_acquire acquire;
    return c10::DeviceIndex(1);
  }
  /**
   * Return true if all the work previously enqueued on the stream for
   * asynchronous execution has completed running on the device.
   */
  bool queryStream(const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    // not implemented on spyre - do nothing
    return true;
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * enqueued on the stream has completed running on the device.
   */
  virtual void synchronizeStream(const c10::Stream& stream) const {
    py::gil_scoped_acquire acquire;
    // not implemented on spyre - do nothing
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * recorded on the event has completed running on the device.
   */
  void synchronizeEvent(void* event) const override {
    py::gil_scoped_acquire acquire;
    // not implemented on spyre - do nothing
  }

  /**
   * Ensure the caching allocator (if any) is aware that the given DataPtr is
   * being used on the given stream, and that it should thus avoid recycling the
   * DataPtr until all work on that stream is done.
   */
  void recordDataPtrOnStream(const c10::DataPtr& data_ptr,
                             const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    // not implemented on spyre - do nothing
  }

  /**
   * Fetch the elapsed time between two recorded events.
   */
  double elapsedTime(void* event1, void* event2,
                     const c10::DeviceIndex device_index) const override {
    py::gil_scoped_acquire acquire;
    return 0.0;
  }
};

// Register our device guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, SpyreGuardImpl);

}  // namespace

}  // namespace spyre

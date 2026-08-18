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

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>

#include <algorithm>
#include <flex/flex.hpp>
#include <memory>
#include <mutex>
#include <spyre_comms.hpp>
#include <spyre_comms_tensor.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../logging.h"
#include "../spyre_allocator.h"
#include "../spyre_stream.h"
#include "../spyre_tensor_impl.h"

namespace spyre {

// Identifies which collective produced the pending work
enum class CollectiveKind { Broadcast, AllReduce };

// Structure to hold pending async work
struct PendingWork {
  std::shared_ptr<spyre_comms::WorkSchedule> work;
  CollectiveKind kind;
  // Keep tensors alive while communication is in-flight to avoid UAF
  std::vector<at::Tensor> hold_tensors;
};

// Global map to track pending async operations
// Key: SharedOwnerCtx* (stable per-allocation identity, never reused), Value:
// PendingWork
static std::unordered_map<spyre::SharedOwnerCtx*, PendingWork>
    pending_work_map_;
static std::mutex work_map_mutex_;

// Helper to convert PyTorch ScalarType to spyre_comms TensorDataTypeEnum
spyre_comms::TensorDataTypeEnum torch_dtype_to_spyre_comms(
    c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return spyre_comms::TensorDataTypeEnum::float32;
    case c10::ScalarType::Double:
      return spyre_comms::TensorDataTypeEnum::float64;
    case c10::ScalarType::Half:
      return spyre_comms::TensorDataTypeEnum::float16;
    case c10::ScalarType::BFloat16:
      return spyre_comms::TensorDataTypeEnum::bfloat16;
    case c10::ScalarType::Int:
      return spyre_comms::TensorDataTypeEnum::int32;
    case c10::ScalarType::Long:
      return spyre_comms::TensorDataTypeEnum::int64;
    case c10::ScalarType::Short:
      return spyre_comms::TensorDataTypeEnum::int16;
    case c10::ScalarType::Char:
      return spyre_comms::TensorDataTypeEnum::int8;
    case c10::ScalarType::Byte:
      return spyre_comms::TensorDataTypeEnum::uint8;
    case c10::ScalarType::Bool:
      return spyre_comms::TensorDataTypeEnum::boolean;
    default:
      TORCH_CHECK(false, "Unsupported dtype for spyre_comms: ", dtype);
  }
}

// Helper to get CompositeAddress pointer from a Spyre tensor
// NOTE: The returned pointer is valid only as long as the tensor's storage
// context remains valid. Caller must keep the tensor alive.
const flex::CompositeAddress* get_composite_address(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "Tensor must be on Spyre device for distributed operations");

  TORCH_CHECK(tensor.is_contiguous(),
              "Tensor must be contiguous for distributed operations");

  auto* spyre_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  TORCH_CHECK(spyre_impl != nullptr, "SpyreTensorImpl is null");

  auto& storage = spyre_impl->storage();
  auto* data_ptr = storage.data_ptr().get();
  TORCH_CHECK(data_ptr != nullptr, "Storage data pointer is null");

  auto* ctx = static_cast<SharedOwnerCtx*>(storage.data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null");

  // Return a pointer to the CompositeAddress inside the context
  return &ctx->composite_addr;
}

// Async broadcast implementation - returns immediately
at::Tensor spyre_broadcast_async_impl(const at::Tensor& input, int64_t src_rank,
                                      const std::string& group_name) {
  DEBUGINFO("spyre::broadcast_async called with src_rank=", src_rank,
            ", group=", group_name);

  // Get world context
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                    spyre::getDefaultStreamRuntimeHandle());
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  }

  // Validate src_rank is in bounds
  TORCH_CHECK(
      src_rank >= 0 && src_rank < static_cast<int64_t>(context->getSize()),
      "src_rank out of range: ", src_rank, " (world size is ",
      context->getSize(), ")");

  // Create output tensor
  at::Tensor output = at::empty_like(input);
  TORCH_CHECK(output.nbytes() > 0,
              "Tensor must have non-zero size for broadcast");

  // Get SharedOwnerCtx for map key (stable per-allocation identity)
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      output.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for output tensor");

  // Get the device layout — pass the total buffer element count as a flat 1D
  // shape to avoid spyre_comms interpreting stick dimensions specially.
  SpyreTensorLayout stl = get_spyre_tensor_layout(output);
  uint64_t bcast_nbytes = get_device_size_in_bytes(stl);
  int64_t bcast_total_elems =
      static_cast<int64_t>(bcast_nbytes / input.element_size());

  spyre_comms::TensorDataTypeEnum dtype =
      torch_dtype_to_spyre_comms(input.scalar_type());

  spyre_comms::TensorShape shape({bcast_total_elems});
  spyre_comms::TensorInfo tensor_info(dtype, shape);

  // Copy input to output if we're the source rank
  int current_rank = context->getRank();
  if (current_rank == src_rank) {
    output.copy_(input);
  }

  // Create spyre_comms Tensor with device address
  spyre_comms::Tensor buffer_tensor(tensor_info);
  buffer_tensor.SetSpyreDeviceAddressBorrowed(&ctx->composite_addr);

  // Start broadcast (non-blocking)
  auto work_schedule = context->broadcast(
      buffer_tensor, static_cast<spyre_comms::process_id_t>(src_rank));
  TORCH_CHECK(work_schedule != nullptr,
              "Broadcast operation failed to create work schedule");

  work_schedule->start();  // Start but DON'T wait

  // Store WorkSchedule in map; hold_tensors keeps the allocation alive
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "broadcast_async called twice on the same allocation without "
                "intervening wait_work");
    pending_work_map_.emplace(ctx, PendingWork{std::move(work_schedule),
                                               CollectiveKind::Broadcast,
                                               {output}});
    DEBUGINFO("Stored PendingWork at ctx=", ctx,
              ", pending_work_map size=", pending_work_map_.size());
  }

  return output;  // Return immediately without waiting
}

// Helper to convert reduce_op string to SpyreReductionOpType
spyre_comms::SpyreReductionOpType parse_reduce_op(
    const std::string& reduce_op) {
  if (reduce_op == "sum") {
    return spyre_comms::SpyreReductionOpType::SUM;
  }
  TORCH_CHECK(false, "Unsupported reduce_op for spyre allreduce: ", reduce_op,
              ". Only 'sum' is currently supported.");
}

// All_reduce implementation — operates in-place on the input buffer.
// Non-blocking: starts the reduction and returns immediately; the caller
// must use wait_work to block until the operation completes.
at::Tensor spyre_allreduce_async_impl(const at::Tensor& input,
                                      const std::string& reduce_op,
                                      const std::string& group_name) {
  DEBUGINFO("spyre::all_reduce_async called with reduce_op=", reduce_op,
            ", group=", group_name);

  // Get world context
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                    spyre::getDefaultStreamRuntimeHandle());
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  }

  auto op_type = parse_reduce_op(reduce_op);

  TORCH_CHECK(input.is_privateuseone(),
              "Tensor must be on Spyre device for all_reduce");
  TORCH_CHECK(input.is_contiguous(),
              "Tensor must be contiguous for all_reduce");
  TORCH_CHECK(input.nbytes() > 0,
              "Tensor must have non-zero size for all_reduce");

  // Get SharedOwnerCtx for the input tensor
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      input.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for input tensor");

  SpyreTensorLayout stl = get_spyre_tensor_layout(input);
  uint64_t ar_nbytes = get_device_size_in_bytes(stl);
  int64_t ar_total_elems =
      static_cast<int64_t>(ar_nbytes / input.element_size());

  spyre_comms::TensorDataTypeEnum dtype =
      torch_dtype_to_spyre_comms(input.scalar_type());
  spyre_comms::TensorShape shape({ar_total_elems});
  spyre_comms::TensorInfo tensor_info(dtype, shape);

  // Create tensor with host pointer + device address.
  spyre_comms::Tensor inout_tensor(tensor_info,
                                   input.storage().data_ptr().get());
  inout_tensor.SetSpyreDeviceAddressBorrowed(&ctx->composite_addr);

  auto work_schedule = context->allreduce(inout_tensor, op_type);
  TORCH_CHECK(work_schedule != nullptr,
              "All_reduce operation failed to create work schedule");

  work_schedule->start();

  // Store WorkSchedule in map for later wait_work call
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "all_reduce_async called twice on the same "
                "allocation without intervening wait_work");
    pending_work_map_.emplace(
        ctx, PendingWork{
                 std::move(work_schedule), CollectiveKind::AllReduce, {input}});
    DEBUGINFO("Stored PendingWork for all_reduce at ctx=", ctx,
              ", pending_work_map size=", pending_work_map_.size());
  }

  return input;  // Return the same tensor (allreduce operates in-place)
}

// Wait for async operation to complete
at::Tensor spyre_wait_work_impl(const at::Tensor& tensor) {
  DEBUGINFO("spyre::wait_work called");

  // Get SharedOwnerCtx for map lookup
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      tensor.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr,
              "SharedOwnerCtx is null — is this tensor from broadcast_async?");

  // Extract WorkSchedule under lock, erase map entry, release lock, then wait
  std::shared_ptr<spyre_comms::WorkSchedule> work_to_wait;
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    auto it = pending_work_map_.find(ctx);
    TORCH_CHECK(it != pending_work_map_.end(),
                "No pending async work found for tensor. "
                "wait_work must be called on a tensor returned from "
                "broadcast_async or all_reduce_async.");

    work_to_wait = std::move(it->second.work);
    pending_work_map_.erase(it);
    DEBUGINFO("Extracted and erased PendingWork, map size=",
              pending_work_map_.size());
  }

  // Lock released — concurrent wait_work and broadcast_async can now proceed
  work_to_wait->wait();
  DEBUGINFO("WorkSchedule wait completed");

  // Return the tensor with completed collective data (broadcast or allreduce)
  return tensor;
}

}  // namespace spyre

// Define the spyre namespace and operations
TORCH_LIBRARY(spyre, m) {
  m.def(
      "broadcast_async(Tensor input, int src_rank, str group_name) -> Tensor");
  m.def(
      "all_reduce_async(Tensor(a!) input, str reduce_op=\"sum\", "
      "str group_name=\"default\") -> Tensor(a)");
  // wait_work mutates the tensor in-place (fills in the received data)

  m.def("wait_work(Tensor(a!) tensor) -> Tensor(a)");
}

// Register the implementations with PyTorch's dispatcher
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("broadcast_async", &spyre::spyre_broadcast_async_impl);
  m.impl("all_reduce_async", &spyre::spyre_allreduce_async_impl);
  m.impl("wait_work", &spyre::spyre_wait_work_impl);
}

/*
 * Copyright 2025-2026 The Torch-Spyre Authors.
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
#include "../spyre_composite_address.h"
#include "../spyre_stream.h"

namespace spyre {

enum class CollectiveKind { Broadcast, AllGather, AllReduce };

// Structure to hold pending async work
struct PendingWork {
  CollectiveKind kind;
  std::shared_ptr<spyre_comms::WorkSchedule> work;
  std::vector<at::Tensor> rank_outputs;
  int64_t chunk_size = 0;
  std::vector<at::Tensor> hold_tensors;
};

// Global map to track pending async operations.
// Key: SharedOwnerCtx* (stable per-allocation identity). PendingWork holds
// tensor references (hold_tensors) that prevent the key from being freed
// while communication is in flight.
static std::unordered_map<spyre::SharedOwnerCtx*, PendingWork>
    pending_work_map_;
static std::mutex work_map_mutex_;

// Compile-time plan cache.
enum class PlanKind { Broadcast, AllReduce, AllGather };

struct CachedPlan {
  PlanKind kind;
  spyre_comms::TensorDataTypeEnum dtype;
  int64_t rank_param;  // src_rank (broadcast) or dst_rank (reduce); unused for
                       // allreduce
  spyre_comms::SpyreReductionOpType reduce_op;  // only for allreduce/reduce

  // tensor_info MUST outlive wsi — spyre_comms stores a non-owning reference
  // to it inside the WorkScheduleInfo's sentinel envelope.
  std::unique_ptr<spyre_comms::TensorInfo> tensor_info;
  std::unique_ptr<spyre_comms::WorkScheduleInfo> wsi;
  int64_t num_elems = 0;
  int64_t group_size = 0;  // allgather only
};
static std::vector<CachedPlan> wsi_cache_;
static std::mutex wsi_cache_mutex_;

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

// Ensure spyre_comms is initialized and return the world context.
std::shared_ptr<spyre_comms::Context> ensure_context() {
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                    spyre::getDefaultStreamRuntimeHandle());
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  }
  return context;
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

// ============================================================================
// Compile-time plan ops — store collective parameters at graph load + create
// WSI
// ============================================================================

// Search the cache for an existing plan that matches the given parameters.
// Must be called with wsi_cache_mutex_ held. Returns -1 if not found.
int64_t cache_lookup(PlanKind kind, spyre_comms::TensorDataTypeEnum dtype,
                     int64_t num_elems, int64_t rank_param,
                     spyre_comms::SpyreReductionOpType reduce_op,
                     int64_t group_size) {
  for (size_t i = 0; i < wsi_cache_.size(); i++) {
    auto& entry = wsi_cache_[i];
    if (entry.kind == kind && entry.dtype == dtype &&
        entry.num_elems == num_elems && entry.rank_param == rank_param &&
        entry.reduce_op == reduce_op && entry.group_size == group_size) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

// Ensure the WSI is created for a cached plan entry.
// Must be called with wsi_cache_mutex_ held.
void ensure_wsi(CachedPlan& plan, int64_t num_elems,
                std::shared_ptr<spyre_comms::Context>& context) {
  if (plan.wsi != nullptr) return;

  spyre_comms::TensorShape shape({num_elems});
  plan.tensor_info =
      std::make_unique<spyre_comms::TensorInfo>(plan.dtype, shape);
  plan.num_elems = num_elems;

  switch (plan.kind) {
    case PlanKind::Broadcast:
      plan.wsi = context->broadcast(
          *plan.tensor_info,
          static_cast<spyre_comms::process_id_t>(plan.rank_param));
      break;
    case PlanKind::AllReduce:
      plan.wsi = context->allreduce(*plan.tensor_info, plan.reduce_op);
      break;
    case PlanKind::AllGather: {
      std::vector<spyre_comms::TensorInfo> output_infos(
          static_cast<size_t>(plan.group_size), *plan.tensor_info);
      plan.wsi = context->allgather(output_infos, *plan.tensor_info);
      break;
    }
  }
  TORCH_CHECK(plan.wsi != nullptr, "Failed to create WSI");
}

int64_t spyre_broadcast_plan_impl(int64_t num_elems, int64_t dtype_code,
                                  int64_t src_rank,
                                  const std::string& group_name) {
  DEBUGINFO("spyre::broadcast_plan called with num_elems=", num_elems,
            ", dtype=", dtype_code, ", src_rank=", src_rank);

  auto context = ensure_context();

  TORCH_CHECK(
      src_rank >= 0 && src_rank < static_cast<int64_t>(context->getSize()),
      "src_rank out of range: ", src_rank, " (world size is ",
      context->getSize(), ")");

  auto dtype =
      torch_dtype_to_spyre_comms(static_cast<c10::ScalarType>(dtype_code));

  std::lock_guard<std::mutex> lock(wsi_cache_mutex_);
  int64_t handle = cache_lookup(PlanKind::Broadcast, dtype, num_elems, src_rank,
                                spyre_comms::SpyreReductionOpType::SUM, 0);
  if (handle >= 0) {
    DEBUGINFO("broadcast_plan: cache hit at handle=", handle);
    return handle;
  }

  handle = static_cast<int64_t>(wsi_cache_.size());
  wsi_cache_.push_back(CachedPlan{PlanKind::Broadcast, dtype, src_rank,
                                  spyre_comms::SpyreReductionOpType::SUM,
                                  nullptr, nullptr, 0, 0});
  auto& plan = wsi_cache_.back();
  ensure_wsi(plan, num_elems, context);

  DEBUGINFO("broadcast_plan: created WSI at handle=", handle);
  return handle;
}

int64_t spyre_allreduce_plan_impl(int64_t num_elems, int64_t dtype_code,
                                  const std::string& reduce_op,
                                  const std::string& group_name) {
  DEBUGINFO("spyre::allreduce_plan called with num_elems=", num_elems,
            ", dtype=", dtype_code, ", reduce_op=", reduce_op);

  auto context = ensure_context();
  auto op_type = parse_reduce_op(reduce_op);
  auto dtype =
      torch_dtype_to_spyre_comms(static_cast<c10::ScalarType>(dtype_code));

  std::lock_guard<std::mutex> lock(wsi_cache_mutex_);
  int64_t handle =
      cache_lookup(PlanKind::AllReduce, dtype, num_elems, 0, op_type, 0);
  if (handle >= 0) {
    DEBUGINFO("allreduce_plan: cache hit at handle=", handle);
    return handle;
  }

  handle = static_cast<int64_t>(wsi_cache_.size());
  wsi_cache_.push_back(CachedPlan{PlanKind::AllReduce, dtype, 0, op_type,
                                  nullptr, nullptr, 0, 0});
  auto& plan = wsi_cache_.back();
  ensure_wsi(plan, num_elems, context);

  DEBUGINFO("allreduce_plan: created WSI at handle=", handle);
  return handle;
}

int64_t spyre_allgather_plan_impl(int64_t num_elems, int64_t dtype_code,
                                  int64_t group_size,
                                  const std::string& group_name) {
  DEBUGINFO("spyre::allgather_plan called with num_elems=", num_elems,
            ", dtype=", dtype_code, ", group_size=", group_size);

  auto context = ensure_context();

  TORCH_CHECK(
      group_size > 0 && group_size == static_cast<int64_t>(context->getSize()),
      "group_size must equal world size: got ", group_size, " (world size is ",
      context->getSize(), ")");

  auto dtype =
      torch_dtype_to_spyre_comms(static_cast<c10::ScalarType>(dtype_code));

  std::lock_guard<std::mutex> lock(wsi_cache_mutex_);
  int64_t handle =
      cache_lookup(PlanKind::AllGather, dtype, num_elems, 0,
                   spyre_comms::SpyreReductionOpType::SUM, group_size);
  if (handle >= 0) {
    DEBUGINFO("allgather_plan: cache hit at handle=", handle);
    return handle;
  }

  handle = static_cast<int64_t>(wsi_cache_.size());
  wsi_cache_.push_back(CachedPlan{PlanKind::AllGather, dtype, 0,
                                  spyre_comms::SpyreReductionOpType::SUM,
                                  nullptr, nullptr, 0, group_size});
  auto& plan = wsi_cache_.back();
  ensure_wsi(plan, num_elems, context);

  DEBUGINFO("allgather_plan: created WSI at handle=", handle);
  return handle;
}

// ============================================================================
// Runtime run ops
// ============================================================================

at::Tensor spyre_broadcast_run_impl(const at::Tensor& input,
                                    int64_t plan_handle, int64_t src_rank) {
  DEBUGINFO("spyre::broadcast_run called with plan_handle=", plan_handle,
            ", src_rank=", src_rank);

  auto context = ensure_context();

  std::lock_guard<std::mutex> cache_lock(wsi_cache_mutex_);
  TORCH_CHECK(
      plan_handle >= 0 && plan_handle < static_cast<int64_t>(wsi_cache_.size()),
      "broadcast_run: invalid plan_handle=", plan_handle);
  auto& plan = wsi_cache_[static_cast<size_t>(plan_handle)];

  // Create output tensor
  at::Tensor output = at::empty_like(input);
  TORCH_CHECK(output.nbytes() > 0,
              "Tensor must have non-zero size for broadcast");

  // Copy input to output if we're the source rank
  int current_rank = context->getRank();
  if (current_rank == src_rank) {
    output.copy_(input);
  }

  // Get SharedOwnerCtx for map key
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      output.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for output tensor");

  // Build spyre_comms::Tensor using the plan's TensorInfo (must stay alive)
  spyre_comms::Tensor buffer_tensor(*plan.tensor_info);
  buffer_tensor.SetSpyreDeviceAddressBorrowed(get_composite_address(output));

  auto work_schedule = context->broadcast_applyTensor(*plan.wsi, buffer_tensor);
  TORCH_CHECK(work_schedule != nullptr,
              "broadcast_applyTensor operation failed to create WorkSchedule");

  work_schedule->start();

  // Store pending work
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "broadcast_run called twice on the same allocation without "
                "intervening wait_work");
    pending_work_map_.emplace(ctx, PendingWork{CollectiveKind::Broadcast,
                                               std::move(work_schedule),
                                               {output}});
  }

  return output;
}

at::Tensor spyre_allreduce_run_impl(const at::Tensor& input,
                                    int64_t plan_handle) {
  DEBUGINFO("spyre::allreduce_run called with plan_handle=", plan_handle);

  auto context = ensure_context();

  std::lock_guard<std::mutex> cache_lock(wsi_cache_mutex_);
  TORCH_CHECK(
      plan_handle >= 0 && plan_handle < static_cast<int64_t>(wsi_cache_.size()),
      "allreduce_run: invalid plan_handle=", plan_handle);
  auto& plan = wsi_cache_[static_cast<size_t>(plan_handle)];

  TORCH_CHECK(input.is_privateuseone(),
              "Tensor must be on Spyre device for all_reduce");
  TORCH_CHECK(input.is_contiguous(),
              "Tensor must be contiguous for all_reduce");
  TORCH_CHECK(input.nbytes() > 0,
              "Tensor must have non-zero size for all_reduce");

  // Get SharedOwnerCtx
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      input.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for input tensor");

  // Build spyre_comms::Tensor using the plan's TensorInfo (must stay alive)
  spyre_comms::Tensor inout_tensor(*plan.tensor_info,
                                   input.storage().data_ptr().get());
  inout_tensor.SetSpyreDeviceAddressBorrowed(get_composite_address(input));

  auto work_schedule = context->allreduce_applyTensor(*plan.wsi, inout_tensor);
  TORCH_CHECK(work_schedule != nullptr,
              "allreduce_applyTensor operation failed to create WorkSchedule");

  work_schedule->start();

  // Store pending work
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "allreduce_run called twice on the same allocation without "
                "intervening wait_work");
    pending_work_map_.emplace(
        ctx, PendingWork{
                 CollectiveKind::AllReduce, std::move(work_schedule), {input}});
  }

  return input;
}

at::Tensor spyre_allgather_run_impl(const at::Tensor& input,
                                    int64_t plan_handle, int64_t group_size) {
  DEBUGINFO("spyre::allgather_run called with plan_handle=", plan_handle,
            ", group_size=", group_size);

  auto context = ensure_context();

  std::lock_guard<std::mutex> cache_lock(wsi_cache_mutex_);
  TORCH_CHECK(
      plan_handle >= 0 && plan_handle < static_cast<int64_t>(wsi_cache_.size()),
      "allgather_run: invalid plan_handle=", plan_handle);
  auto& plan = wsi_cache_[static_cast<size_t>(plan_handle)];

  TORCH_CHECK(input.is_privateuseone(),
              "Tensor must be on Spyre device for allgather");
  TORCH_CHECK(input.is_contiguous(), "Tensor must be contiguous for allgather");
  TORCH_CHECK(input.nbytes() > 0,
              "Tensor must have non-zero size for allgather");

  spyre_comms::Tensor input_tensor(*plan.tensor_info,
                                   input.storage().data_ptr().get());
  input_tensor.SetSpyreDeviceAddressBorrowed(get_composite_address(input));

  // Allocate per-rank output tensors (same shape/layout as input)
  std::vector<at::Tensor> rank_outputs;
  rank_outputs.reserve(group_size);
  for (int64_t i = 0; i < group_size; i++) {
    rank_outputs.push_back(at::empty_like(input));
  }

  // Build spyre_comms::Tensor vector for per-rank outputs
  std::vector<spyre_comms::Tensor> output_tensors;
  output_tensors.reserve(group_size);
  for (int64_t i = 0; i < group_size; i++) {
    spyre_comms::Tensor out_tensor(*plan.tensor_info,
                                   rank_outputs[i].storage().data_ptr().get());
    out_tensor.SetSpyreDeviceAddressBorrowed(
        get_composite_address(rank_outputs[i]));
    output_tensors.push_back(std::move(out_tensor));
  }

  auto work_schedule =
      context->allgather_applyTensors(*plan.wsi, output_tensors, input_tensor);
  TORCH_CHECK(work_schedule != nullptr,
              "allgather_applyTensors operation failed to create WorkSchedule");

  work_schedule->start();

  // Allocate final concatenated output
  auto output_sizes = input.sizes().vec();
  output_sizes[0] *= group_size;
  at::Tensor output = at::empty(output_sizes, input.options());

  auto* output_ctx = static_cast<spyre::SharedOwnerCtx*>(
      output.storage().data_ptr().get_context());
  TORCH_CHECK(output_ctx != nullptr,
              "SharedOwnerCtx is null for output tensor");

  // Store pending work with rank_outputs for assembly in wait_work
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(output_ctx) == pending_work_map_.end(),
                "allgather_run called twice on the same allocation without "
                "intervening wait_work");
    pending_work_map_.emplace(output_ctx, PendingWork{CollectiveKind::AllGather,
                                                      std::move(work_schedule),
                                                      std::move(rank_outputs),
                                                      input.size(0),
                                                      {output}});
  }

  return output;
}

// Wait for async operation to complete
at::Tensor spyre_wait_work_impl(const at::Tensor& tensor) {
  DEBUGINFO("spyre::wait_work called");

  // Get SharedOwnerCtx for map lookup
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      tensor.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for tensor");

  PendingWork pending;
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    auto it = pending_work_map_.find(ctx);
    TORCH_CHECK(it != pending_work_map_.end(),
                "No pending async work found for tensor. "
                "wait_work must be called on a tensor returned from "
                "broadcast_run, allgather_run, or allreduce_run.");

    pending = std::move(it->second);
    pending_work_map_.erase(it);
    DEBUGINFO("Extracted and erased PendingWork, map size=",
              pending_work_map_.size());
  }

  // Lock released — concurrent wait_work and run ops can now proceed
  if (pending.work) {
    pending.work->wait();
    DEBUGINFO("WorkSchedule wait completed");
  }

  if (pending.kind == CollectiveKind::AllGather) {
    // _c10d_functional.all_gather_into_tensor concatenates along dim 0 by
    // contract (see torch/distributed/_functional_collectives.py). Verify
    // the output was sized accordingly.
    int64_t world = static_cast<int64_t>(pending.rank_outputs.size());
    TORCH_CHECK(tensor.size(0) == world * pending.chunk_size,
                "wait_work: output dim 0 (", tensor.size(0),
                ") != world_size * chunk_size (", world, " * ",
                pending.chunk_size,
                "). all_gather_into_tensor must concatenate along dim 0.");

    for (size_t i = 0; i < pending.rank_outputs.size(); i++) {
      tensor
          .narrow(0, static_cast<int64_t>(i) * pending.chunk_size,
                  pending.chunk_size)
          .copy_(pending.rank_outputs[i]);
    }
    DEBUGINFO("Assembled allgather output from ", pending.rank_outputs.size(),
              " rank buffers");
  }
  // For Broadcast/AllReduce/Reduce the output data is already in tensor —
  // the collective operates in-place so no further data manipulation is needed.

  // Return the tensor with completed collective data (broadcast or allreduce)
  return tensor;
}

}  // namespace spyre

// Define the spyre namespace and operations
TORCH_LIBRARY(spyre, m) {
  m.def(
      "broadcast_async(Tensor input, int src_rank, str group_name) -> Tensor");
  m.def(
      "all_gather_async(Tensor input, SymInt group_size=1, "
      "str group_name=\"default\") -> Tensor");
  m.def(
      "all_reduce_async(Tensor(a!) input, str reduce_op=\"sum\", "
      "str group_name=\"default\") -> Tensor(a)");
  m.def("wait_work(Tensor(a!) tensor) -> Tensor(a)");

  // Compile-time plan ops — scalar-only, registered with impl directly
  // so they dispatch via CompositeImplicitAutograd (no tensor to key off).
  m.def(
      "broadcast_plan(int num_elems, int dtype, int src_rank, "
      "str group_name) -> int",
      &spyre::spyre_broadcast_plan_impl);
  m.def(
      "allreduce_plan(int num_elems, int dtype, str reduce_op, "
      "str group_name) -> int",
      &spyre::spyre_allreduce_plan_impl);
  m.def(
      "allgather_plan(int num_elems, int dtype, int group_size, "
      "str group_name) -> int",
      &spyre::spyre_allgather_plan_impl);

  // Runtime run ops — bind cached WSI to a tensor and execute
  m.def(
      "broadcast_run(Tensor input, int plan_handle, int src_rank) "
      "-> Tensor");
  m.def("allreduce_run(Tensor(a!) input, int plan_handle) -> Tensor(a)");
  m.def(
      "allgather_run(Tensor input, int plan_handle, int group_size) "
      "-> Tensor");
}

// Register the implementations with PyTorch's dispatcher
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("wait_work", &spyre::spyre_wait_work_impl);

  m.impl("broadcast_run", &spyre::spyre_broadcast_run_impl);
  m.impl("allreduce_run", &spyre::spyre_allreduce_run_impl);
  m.impl("allgather_run", &spyre::spyre_allgather_run_impl);
}

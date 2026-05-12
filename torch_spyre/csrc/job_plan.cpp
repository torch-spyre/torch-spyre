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

#include "job_plan.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "flex/allocator/alloc_address.hpp"
#include "flex/runtime_stream/runtime_operation.hpp"
#include "spyre_allocator.h"

namespace spyre {

std::unique_ptr<flex::RuntimeOperation> JobPlanStepH2D::construct(
    LaunchContext&) const {
  auto op = std::make_unique<flex::RuntimeOperationH2D>(host_address_,
                                                        &device_address_);
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepD2H::construct(
    LaunchContext&) const {
  auto op = std::make_unique<flex::RuntimeOperationD2H>(&device_address_,
                                                        host_address_);
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepCompute::construct(
    LaunchContext&) const {
  auto op = std::make_unique<flex::RuntimeOperationCompute>(&binary_address_);
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

// constexpr int64_t SegmentSize = 16LL * 1024 * 1024 * 1024;

// convert CompositeAddress to address that host compute function expects
int64_t convert_address(flex::CompositeAddress& composite_address) {
  size_t num_chunks = composite_address.chunks().size();
  TORCH_CHECK(num_chunks == 1, "Interleaved not supported yet");

  // TODO(jni): update once resolved on flex support
  // const auto& addr = composite_address.chunks().at(0).addr;
  // int64_t address = addr.segment_id * SegmentSize + addr.offset;

  TORCH_CHECK(false,
              "convert_address not yet implemented - waiting for flex support");
  return 0;
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepHostCompute::construct(
    LaunchContext& ctx) const {
  std::vector<int64_t> addresses(ctx.inputs_outputs.size());
  int addr_idx = 0;
  for (auto& tensor : ctx.inputs_outputs) {
    int64_t addr = convert_address(
        (static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
             ->composite_addr));
    addresses[addr_idx++] = addr;
  }
  auto callback = [this, addresses](void*) {
    function_(metadata_.get(), output_buffer_, &addresses);
  };

  auto op = std::make_unique<flex::RuntimeOperationHostCallback>(
      pipeline_barrier_, std::move(callback), nullptr);

  return op;
}

// TODO(jni): to be added once flex PR merged
// std::unique_ptr<flex::RuntimeOperation>
// JobPlanStepComputeSpecialize::construct(
//     LaunchContext& ctx) const {
//   std::vector<flex::CompositeAddress*> inp;
//   for (auto& tensor : ctx.inputs_outputs) {
//     flex::CompositeAddress* address = &(
//         static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
//             ->composite_addr);
//     inp.push_back((address));
//   }
//   auto op =
//   std::make_unique<flex::RuntimeOperationComputeSpecializeResident>(
//       &binary_address_, inp);
//   return op;
// }

}  // namespace spyre

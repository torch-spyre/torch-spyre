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

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "spyre_allocator.h"
#include "util/processSpyreCodeArtifacts.h"

namespace spyre {

void JobPlanStepH2D::construct(
    LaunchContext&, flex::RuntimeStream* flex_stream) const {
  flex_stream->launchOperationH2D(host_address_, &device_address_);
}

void JobPlanStepH2D::write(std::ostream& os) const {
  os << "  H2D (Host-to-Device)\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Device address: " << device_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepD2H::construct(
    LaunchContext&, flex::RuntimeStream* flex_stream) const {
  flex_stream->launchOperationD2H(&device_address_, host_address_);
}

void JobPlanStepD2H::write(std::ostream& os) const {
  os << "  D2H (Device-to-Host)\n";
  os << "    Device address: " << device_address_ << "\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepCompute::construct(
    LaunchContext& ctx, flex::RuntimeStream* flex_stream) const {
  if (bind_io_addresses_) {
    std::vector<const flex::CompositeAddress*> inp;
    for (auto& tensor : ctx.inputs_outputs) {
      flex::CompositeAddress* address =
          &(static_cast<SharedOwnerCtx*>(
                tensor.storage().data_ptr().get_context())
                ->composite_addr);
      inp.push_back(address);
    }

    flex_stream->launchOperationCompute(&binary_address_, inp, "", bootstrap_addr_, {}, pipeline_barrier_);
  }
  flex_stream->launchOperationCompute(&binary_address_, {}, "", flex::PROG_OFFSET_BASE, {}, pipeline_barrier_);
}

void JobPlanStepCompute::write(std::ostream& os) const {
  os << "  Device Compute\n";
  os << "    Binary address: " << binary_address_ << "\n";
  os << "    Bind I/O addresses: " << (bind_io_addresses_ ? "yes" : "no")
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

// convert CompositeAddress to address that host compute function expects
int64_t convert_address(flex::CompositeAddress& composite_address) {
  size_t num_chunks = composite_address.chunks().size();
  TORCH_CHECK(num_chunks == 1, "Interleaved not supported yet");

  // TODO(jni): update once resolved on flex support
  // const auto& addr = composite_address.chunks().at(0).addr;
  // int64_t address = addr.segment_id * flex::SEGMENT_SIZE + addr.offset;

  TORCH_CHECK(false,
              "convert_address not yet implemented - waiting for flex support");
  return 0;
}

void JobPlanStepHostCompute::construct(
    LaunchContext& ctx, flex::RuntimeStream* flex_stream) const {
  // Case 1: input_buffer_ is provided
  if (input_buffer_ != nullptr) {
    auto callback = [this](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_,
                                             input_buffer_);
    };
    flex_stream->launchOperationHostCallback(pipeline_barrier_, std::move(callback), nullptr);
    return;
  }

  // Case 2: fake symbols (ishape_ is {0})
  // Further discussion is required on "ishape". For now, it's vector<int64_t>,
  // and if it's {0}, it's for fake symbols
  if (ishape_.size() == 1 && ishape_[0] == 0) {
    auto callback = [this](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, nullptr);
    };
    flex_stream->launchOperationHostCallback(pipeline_barrier_, std::move(callback), nullptr);
    return;
  }

  // Case 3: extract addresses from context tensors
  std::vector<int64_t> addresses(ctx.inputs_outputs.size());
  int addr_idx = 0;
  for (auto& tensor : ctx.inputs_outputs) {
    int64_t addr = convert_address(
        (static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
             ->composite_addr));
    addresses[addr_idx++] = addr;
  }

  auto callback = [this, addresses](void*) {
    deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, &addresses);
  };

  flex_stream->launchOperationHostCallback(pipeline_barrier_, std::move(callback), nullptr);
}

void JobPlanStepHostCompute::write(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Output buffer: " << output_buffer_ << "\n";
  os << "    HCM metadata: " << (hcm_ ? "present" : "null") << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::ostream& operator<<(std::ostream& os, const JobPlan& plan) {
  os << "============ JobPlan =============\n";
  os << "Total steps: " << plan.steps.size() << "\n";

  // Job allocation
  if (!plan.job_allocation.chunks().empty()) {
    os << "Job allocation: " << plan.job_allocation << "\n";
  } else {
    os << "Job allocation: <none>\n";
  }

  // Expected input shapes
  if (!plan.expected_input_shapes.empty()) {
    os << "Expected input shapes (" << plan.expected_input_shapes.size()
       << " tensors):\n";
    for (size_t i = 0; i < plan.expected_input_shapes.size(); ++i) {
      os << "  Input " << i << ": [";
      for (size_t j = 0; j < plan.expected_input_shapes[i].size(); ++j) {
        if (j > 0) os << ", ";
        os << plan.expected_input_shapes[i][j];
      }
      os << "]\n";
    }
  }

  // Pinned buffers
  os << "Pinned buffers: " << plan.pinned_buffers.size() << "\n";
  for (size_t i = 0; i < plan.pinned_buffers.size(); ++i) {
    const auto& buf = plan.pinned_buffers[i];
    os << "  Buffer " << i << ": ptr=" << buf.data() << ", size=" << buf.size()
       << " bytes\n";
  }

  // Detailed step information
  os << "\nDetailed Steps:\n";
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    os << "Step " << i << ": ";
    os << *plan.steps[i];
  }

  os << "==================================\n";
  return os;
}

}  // namespace spyre

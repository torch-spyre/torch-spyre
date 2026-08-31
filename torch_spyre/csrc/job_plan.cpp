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
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "spyre_allocator.h"
#include "spyre_stream.h"
#include "spyrecode-host-functions/processSpyreCodeArtifacts.h"

namespace spyre {

void JobPlanStepH2D::construct(LaunchContext&,
                               const SpyreStream& stream) const {
  auto* params =
      flex::createDmaParams(host_address_, device_address_.total_size(),
                            /*to_device=*/true, &device_address_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchH2D(params);
  flex::destroyDmaParams(params);
}

void JobPlanStepH2D::write(std::ostream& os) const {
  os << "  H2D (Host-to-Device)\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Device CompositeAddress: " << device_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepD2H::construct(LaunchContext& ctx,
                               const SpyreStream& stream) const {
  if (std::holds_alternative<flex::CompositeAddress>(device_address_)) {
    const auto& device_address =
        std::get<flex::CompositeAddress>(device_address_);
    auto* params =
        flex::createDmaParams(host_address_, device_address.total_size(),
                              /*to_device=*/false, &device_address);
    params->pipeline_barrier = pipeline_barrier_;
    stream.launchD2H(params);
    flex::destroyDmaParams(params);
  } else {
    const uint64_t dmva = std::get<Dmva>(device_address_).value;
    auto segment_id = flex::dmvaToSegmentId(dmva);
    TORCH_CHECK(segment_id < ctx.inputs_outputs.size(),
                "D2H tensor-segment lookup out of range: segment ", segment_id,
                " but only ", ctx.inputs_outputs.size(),
                " launch args were provided");
    const auto& tensor = ctx.inputs_outputs.at(segment_id);
    const auto& tensor_address =
        static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
            ->composite_addr;
    TORCH_CHECK(tensor_address.chunks().size() == 1,
                "Tensor address must have 1 chunk");
    const auto& base_chunk = tensor_address.chunks()[0];
    uint64_t segment_offset = dmva - (segment_id << flex::SEGMENT_SIZE_BITS);
    TORCH_CHECK(segment_offset + size_ <= tensor_address.total_size(),
                "D2H transfer out of bounds: offset ", segment_offset,
                " + size ", size_, " exceeds tensor allocation size ",
                tensor_address.total_size());
    flex::LogicalAddress offset_addr(base_chunk.addr.region_id,
                                     base_chunk.addr.offset + segment_offset);
    flex::Chunk offset_chunk(offset_addr, size_, base_chunk.domain_id);

    // Create shared_ptr to manage lifetime - will be kept alive by callback
    auto device_address =
        std::make_shared<flex::CompositeAddress>(offset_chunk);

    auto* params =
        flex::createDmaParams(host_address_, device_address->total_size(),
                              /*to_device=*/false, device_address.get());
    params->pipeline_barrier = pipeline_barrier_;
    params->callback = [device_address](void*) {};
    stream.launchD2H(params);
    flex::destroyDmaParams(params);
  }
}

void JobPlanStepD2H::write(std::ostream& os) const {
  os << "  D2H (Device-to-Host)\n";
  if (std::holds_alternative<flex::CompositeAddress>(device_address_)) {
    os << "    Device CompositeAddress: "
       << std::get<flex::CompositeAddress>(device_address_) << "\n";
  } else {
    os << "    Device dmva: " << std::get<Dmva>(device_address_).value << "\n";
  }
  os << "    Host address: " << host_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepCompute::construct(LaunchContext& ctx,
                                   const SpyreStream& stream) const {
  std::vector<const flex::CompositeAddress*> tensor_allocs;
  if (bind_io_addresses_) {
    for (auto& tensor : ctx.inputs_outputs) {
      flex::CompositeAddress* address =
          &(static_cast<SharedOwnerCtx*>(
                tensor.storage().data_ptr().get_context())
                ->composite_addr);
      tensor_allocs.push_back(address);
    }
  }
  auto* params = flex::createComputeParams(
      &program_address_, std::move(tensor_allocs), name_, bootstrap_offset_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchCompute(params);
  flex::destroyComputeParams(params);
}

void JobPlanStepCompute::write(std::ostream& os) const {
  os << "  Device Compute\n";
  os << "    Name: " << (name_.empty() ? "(unnamed)" : name_) << "\n";
  os << "    Program CompositeAddress: " << program_address_ << "\n";
  os << "    Bind I/O addresses: " << (bind_io_addresses_ ? "yes" : "no")
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::vector<int64_t> JobPlanStepHostCompute::resolveSymbolicArgs(
    const std::vector<at::Tensor>& tensors,
    const std::vector<SymbolicArg>& symbolic_args) {
  auto& allocator = SpyreAllocator::instance();
  std::vector<int64_t> resolved(symbolic_args.size());
  for (size_t i = 0; i < symbolic_args.size(); ++i) {
    const SymbolicArg& arg = symbolic_args[i];
    TORCH_CHECK(arg.tensor_id >= 0 &&
                    static_cast<size_t>(arg.tensor_id) < tensors.size(),
                "SymbolicArg[", i, "].tensor_id=", arg.tensor_id,
                " out of range [0, ", tensors.size(), ")");
    switch (arg.kind) {
      case SymbolicArgKind::kAddress:
        resolved[i] = static_cast<int64_t>(allocator.compositeAddressToDmva(
            static_cast<SharedOwnerCtx*>(
                tensors[arg.tensor_id].storage().data_ptr().get_context())
                ->composite_addr));
        break;
      case SymbolicArgKind::kDimension:
        TORCH_CHECK(false,
                    "SymbolicArgKind::kDimension is not yet implemented");
        break;
      default:
        TORCH_CHECK(false, "Unknown SymbolicArgKind value: ",
                    static_cast<int32_t>(arg.kind));
    }
  }
  return resolved;
}

void JobPlanStepHostCompute::construct(LaunchContext& ctx,
                                       const SpyreStream& stream) const {
  // Helper lambda to build HostCallbackParams and launch on the stream.
  // flex::RuntimeStream::launchOperationHostCallback() invokes the callback
  // synchronously in the calling thread, so exceptions propagate directly
  // through launchHostCallback() to the caller
  auto launch_host_callback = [this, &stream](auto&& callback) {
    auto* params = flex::createHostCallbackParams(
        std::forward<decltype(callback)>(callback), nullptr, pipeline_barrier_);
    // Use a scope-exit guard so params is freed even if launchHostCallback
    // throws (which it does when the synchronous host callback raises).
    struct Guard {
      flex::HostCallbackParams* p;
      ~Guard() {
        flex::destroyHostCallbackParams(p);
      }
    } guard{params};
    stream.launchHostCallback(params);
  };

  // Case 1: input_buffer_ is provided
  if (input_buffer_ != nullptr) {
    launch_host_callback([this](void*) {
      // Use regular path - input_buffer_ is already properly formatted
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_,
                                             input_buffer_);
    });
    return;
  }

  // Case 2: fake symbols (ishape_ is {0})
  // Further discussion is required on "ishape". For now, it's vector<int64_t>,
  // and it's {0}, it's for fake symbols
  if (ishape_.size() == 1 && ishape_[0] == 0) {
    launch_host_callback([this](void*) {
      // Fake symbols don't need fast path - use regular path
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, nullptr);
    });
    return;
  }

  // Typed symbolic payload present — resolve each slot by kind.
  if (!ctx.symbolic_args.empty()) {
    std::vector<int64_t> resolved_addresses =
        resolveSymbolicArgs(ctx.inputs_outputs, ctx.symbolic_args);

    // Wrong symbolic_args count is an OOB read inside deeptools
    // (DT_CHECK_MSG_OPT is compiled out by default).
    TORCH_CHECK(resolved_addresses.size() == hcm_->vdci.inputSym_.size(),
                "symbolic_args count (", resolved_addresses.size(),
                ") does not match compiled symbol count (",
                hcm_->vdci.inputSym_.size(), ") for this host-compute step");

    launch_host_callback([this, resolved_addresses](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_,
                                             &resolved_addresses);
    });
    return;
  }

  // Case 3b: no payload — legacy path: treat every context tensor as an
  // address source in iteration order.  Back-compat for callers that pass no
  // symbolic_args (empty payload).
  std::vector<int64_t> addresses(ctx.inputs_outputs.size());
  int addr_idx = 0;
  auto& allocator = SpyreAllocator::instance();
  for (auto& tensor : ctx.inputs_outputs) {
    int64_t addr = static_cast<int64_t>(allocator.compositeAddressToDmva(
        (static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
             ->composite_addr)));
    addresses[addr_idx++] = addr;
  }

  launch_host_callback([this, addresses](void*) {
    // Use fast path with all tensor addresses
    // Returns true if fast path was actually used, false if fell back
    bool used_fast_path = deeptools::processComputeOnHostCommandFast(
        fast_plan_, *hcm_, output_buffer_, addresses.data(), addresses.size());
  });
}

void JobPlanStepHostCompute::write(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Output buffer: " << output_buffer_ << "\n";
  os << "    HCM metadata: " << (hcm_ ? "present" : "null") << "\n";
  os << "    Fast path: "
     << (fast_plan_.valid
             ? "enabled"
             : (fast_plan_.output_size == UINT32_MAX ? "disabled" : "building"))
     << "\n";
  if (fast_plan_.valid) {
    os << "    Fast plan: " << fast_plan_.patches.size() << " patches, "
       << fast_plan_.num_input_symbols << " input symbols, "
       << fast_plan_.output_size << " bytes output\n";
  }
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::ostream& operator<<(std::ostream& os, const JobPlan& plan) {
  os << "============ JobPlan =============\n";
  os << "Total steps: " << plan.steps.size() << "\n";

  // Job allocation
  size_t addr_idx = 0;
  for (const auto& addr : plan.job_allocation) {
    if (addr_idx == 0) {
      os << "Job allocation: " << addr << "\n";
    } else {
      os << "Program " << addr_idx - 1 << ": " << addr << "\n";
    }
    ++addr_idx;
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

StepKind classifyStep(const JobPlanStep& step) {
  if (dynamic_cast<const JobPlanStepHostCompute*>(&step)) {
    return StepKind::HostCompute;
  }
  if (dynamic_cast<const JobPlanStepH2D*>(&step)) {
    return StepKind::H2D;
  }
  if (dynamic_cast<const JobPlanStepD2H*>(&step)) {
    return StepKind::D2H;
  }
  if (dynamic_cast<const JobPlanStepCompute*>(&step)) {
    return StepKind::Compute;
  }
  return StepKind::Unknown;
}

const char* stepKindName(StepKind kind) {
  switch (kind) {
    case StepKind::HostCompute:
      return "HostCompute";
    case StepKind::H2D:
      return "H2D";
    case StepKind::D2H:
      return "D2H";
    case StepKind::Compute:
      return "Compute";
    case StepKind::Unknown:
    default:
      return "Unknown";
  }
}

StepKind stepKindFromName(const std::string& name) {
  if (name == "HostCompute") return StepKind::HostCompute;
  if (name == "H2D") return StepKind::H2D;
  if (name == "D2H") return StepKind::D2H;
  if (name == "Compute") return StepKind::Compute;
  if (name == "Unknown") return StepKind::Unknown;
  TORCH_CHECK(false, "Unknown StepKind name: ", name);
}

StreamRole streamRoleFromName(const std::string& name) {
  if (name == "Prep") return StreamRole::Prep;
  if (name == "Dev") return StreamRole::Dev;
  TORCH_CHECK(false, "Unknown StreamRole name: ", name, " (expected Prep/Dev)");
}

std::string checkJobPlanStepOrdering(const std::vector<StepKind>& kinds,
                                     const std::vector<StreamRole>& roles) {
  if (kinds.size() != roles.size()) {
    return "kinds/roles length mismatch";
  }

  // Gate: only validate plans built as HostCompute-led (the two-stream
  // correction triple). A plan without a HostCompute is legacy single-stream
  // and stays valid (backward-compat with the pre-overlap path: pure
  // ComputeOnDevice, standalone D2H, tensor .to() moves).
  bool has_host_compute = false;
  for (StepKind k : kinds) {
    if (k == StepKind::HostCompute) {
      has_host_compute = true;
    }
  }
  if (!has_host_compute) {
    return "";
  }

  // Project into the two per-stream subsequences, preserving plan order.
  std::vector<StepKind> prep;
  std::vector<StepKind> dev;
  for (size_t i = 0; i < kinds.size(); ++i) {
    if (roles[i] == StreamRole::Prep) {
      prep.push_back(kinds[i]);
    } else {
      dev.push_back(kinds[i]);
    }
  }

  auto name_at = [](const std::vector<StepKind>& seq, size_t i) {
    return std::string(i < seq.size() ? stepKindName(seq[i]) : "<end>");
  };

  // The contract is ordering-only, not an exact triple: prepare can emit longer
  // plans (e.g. HostCompute -> H2D -> Compute -> D2H), which project to
  // S_prep = [HostCompute, H2D] and S_dev = [Compute, D2H]. What must hold is
  // the leading-producer guarantee: prep produces (HostCompute -> H2D) before
  // dev consumes (Compute). On the HAZARD path torch-spyre emits no cross-
  // stream event steps; flex derives the RAW/WAR edges from these subsequences.

  // S_prep must BEGIN with HostCompute -> H2D and carry only {HostCompute, H2D}
  // (the persistent host-compute stream; see StreamRole in job_plan.h).
  {
    if (prep.size() < 2 || prep[0] != StepKind::HostCompute ||
        prep[1] != StepKind::H2D) {
      return "S_prep ordering violation: prep stream must begin with "
             "HostCompute -> H2D, got " +
             name_at(prep, 0) + " -> " + name_at(prep, 1);
    }
    for (size_t i = 2; i < prep.size(); ++i) {
      if (prep[i] != StepKind::HostCompute && prep[i] != StepKind::H2D) {
        return "S_prep ordering violation: " + name_at(prep, i) +
               " is not permitted on the prep stream (prep carries only "
               "HostCompute / H2D)";
      }
    }
  }

  // S_dev must BEGIN with Compute (leading-producer guarantee) and carry only
  // {Compute, D2H} (the device stream; see StreamRole in job_plan.h). No
  // HostCompute/H2D -- host-produce steps belong on S_prep.
  {
    if (dev.empty() || dev[0] != StepKind::Compute) {
      return "S_dev ordering violation: device stream must begin with Compute, "
             "got " +
             name_at(dev, 0);
    }
    for (size_t i = 1; i < dev.size(); ++i) {
      if (dev[i] != StepKind::Compute && dev[i] != StepKind::D2H) {
        return "S_dev ordering violation: " + name_at(dev, i) +
               " is not permitted on the device stream (dev carries only "
               "Compute / D2H)";
      }
    }
  }

  return "";
}

}  // namespace spyre

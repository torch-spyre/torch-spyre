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
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, nullptr);
    });
    return;
  }

  // Case 3: extract addresses from context tensors
  std::vector<int64_t> addresses(ctx.inputs_outputs.size());
  int addr_idx = 0;
  auto& allocator = SpyreAllocator::instance();
  for (auto& tensor : ctx.inputs_outputs) {
    int64_t addr = allocator.compositeAddressToDmva(
        (static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
             ->composite_addr));
    addresses[addr_idx++] = addr;
  }

  launch_host_callback([this, addresses](void*) {
    deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, &addresses);
  });
}

void JobPlanStepHostCompute::write(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Output buffer: " << output_buffer_ << "\n";
  os << "    HCM metadata: " << (hcm_ ? "present" : "null") << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

namespace {

// Scope-exit guard mirroring JobPlanStepHostCompute's Guard: destroy the flex
// EventSignalParams even if launchEventSignal throws.
struct SignalParamsGuard {
  flex::EventSignalParams* p;
  ~SignalParamsGuard() {
    flex::destroyEventSignalParams(p);
  }
};

struct WaitParamsGuard {
  flex::EventWaitParams* p;
  ~WaitParamsGuard() {
    flex::destroyEventWaitParams(p);
  }
};

}  // namespace

void JobPlanStepEventSignalForward::construct(LaunchContext& ctx,
                                              const SpyreStream& stream) const {
  TORCH_CHECK(slot_ < ctx.events.size(), "EventSignalForward slot ", slot_,
              " out of range (events.size()=", ctx.events.size(), ")");
  // Allocate a FRESH event for this launch and publish it so the paired wait in
  // the same launch reads the same object. Single-shot; never reused.
  auto event = flex::createEvent();
  ctx.events[slot_] = event;
  auto* params = flex::createEventSignalParams(event);
  SignalParamsGuard guard{params};
  stream.launchEventSignal(params);
}

void JobPlanStepEventSignalForward::write(std::ostream& os) const {
  os << "  Event Signal (forward, edge-3)\n";
  os << "    Slot: " << slot_ << "\n";
  os << "    Stream role: " << (role_ == StreamRole::Prep ? "Prep" : "Dev")
     << "\n";
}

void JobPlanStepEventWaitForward::construct(LaunchContext& ctx,
                                            const SpyreStream& stream) const {
  TORCH_CHECK(slot_ < ctx.events.size(), "EventWaitForward slot ", slot_,
              " out of range (events.size()=", ctx.events.size(), ")");
  auto event = ctx.events[slot_];
  TORCH_CHECK(event != nullptr, "EventWaitForward slot ", slot_,
              " has no event; the paired forward signal must run first");
  auto* params = flex::createEventWaitParams(event);
  WaitParamsGuard guard{params};
  stream.launchEventWait(params);
}

void JobPlanStepEventWaitForward::write(std::ostream& os) const {
  os << "  Event Wait (forward, edge-3)\n";
  os << "    Slot: " << slot_ << "\n";
  os << "    Stream role: " << (role_ == StreamRole::Prep ? "Prep" : "Dev")
     << "\n";
}

void JobPlanStepEventSignalBack::construct(LaunchContext& /*ctx*/,
                                           const SpyreStream& stream) const {
  // Allocate a FRESH event, enqueue the signal on S_dev, then publish it into
  // the runtime-scoped rolling slot so the NEXT launch's WaitBack (keyed by the
  // same region_id) can read it. Publishing AFTER launchEventSignal keeps the
  // slot pointing at an event that is already enqueued to be signaled.
  auto event = flex::createEvent();
  auto* params = flex::createEventSignalParams(event);
  SignalParamsGuard guard{params};
  stream.launchEventSignal(params);
  SpyreStream::setEdge4Slot(region_id_, event);
}

void JobPlanStepEventSignalBack::write(std::ostream& os) const {
  os << "  Event Signal (back, edge-4)\n";
  os << "    Region id: " << region_id_ << "\n";
  os << "    Stream role: " << (role_ == StreamRole::Prep ? "Prep" : "Dev")
     << "\n";
}

void JobPlanStepEventWaitBack::construct(LaunchContext& /*ctx*/,
                                         const SpyreStream& stream) const {
  // Read the rolling slot for this region. On the first launch the slot is
  // empty → NO-OP (nothing to wait on). On later launches it holds the prior
  // launch's SigBack event, so the next H2D waits for the prior DC to stop
  // reading the shared correction region (edge-4 WAR).
  auto event = SpyreStream::getEdge4Slot(region_id_);
  if (event == nullptr) {
    return;  // first launch: no prior producer to wait on
  }
  auto* params = flex::createEventWaitParams(event);
  WaitParamsGuard guard{params};
  stream.launchEventWait(params);
}

void JobPlanStepEventWaitBack::write(std::ostream& os) const {
  os << "  Event Wait (back, edge-4)\n";
  os << "    Region id: " << region_id_ << "\n";
  os << "    Stream role: " << (role_ == StreamRole::Prep ? "Prep" : "Dev")
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
  if (dynamic_cast<const JobPlanStepEventSignalForward*>(&step)) {
    return StepKind::SignalForward;
  }
  if (dynamic_cast<const JobPlanStepEventWaitForward*>(&step)) {
    return StepKind::WaitForward;
  }
  if (dynamic_cast<const JobPlanStepEventSignalBack*>(&step)) {
    return StepKind::SignalBack;
  }
  if (dynamic_cast<const JobPlanStepEventWaitBack*>(&step)) {
    return StepKind::WaitBack;
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
    case StepKind::SignalForward:
      return "SignalForward";
    case StepKind::WaitForward:
      return "WaitForward";
    case StepKind::SignalBack:
      return "SignalBack";
    case StepKind::WaitBack:
      return "WaitBack";
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
  if (name == "SignalForward") return StepKind::SignalForward;
  if (name == "WaitForward") return StepKind::WaitForward;
  if (name == "SignalBack") return StepKind::SignalBack;
  if (name == "WaitBack") return StepKind::WaitBack;
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

  // Gate: only validate plans built as HostCompute-led and/or two-stream (any
  // event step). A plan with neither is legacy single-stream and stays valid
  // (backward-compat with the pre-overlap path: pure ComputeOnDevice,
  // standalone D2H, tensor .to() moves).
  bool has_host_compute = false;
  bool has_event = false;
  for (StepKind k : kinds) {
    if (k == StepKind::HostCompute) {
      has_host_compute = true;
    }
    if (k == StepKind::SignalForward || k == StepKind::WaitForward ||
        k == StepKind::SignalBack || k == StepKind::WaitBack) {
      has_event = true;
    }
  }
  if (!has_host_compute && !has_event) {
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

  // S_prep: HostCompute -> [WaitBack]? -> H2D -> [SignalForward]?
  {
    size_t i = 0;
    if (i >= prep.size() || prep[i] != StepKind::HostCompute) {
      return "S_prep ordering violation: prep stream must begin with "
             "HostCompute, got " +
             name_at(prep, i);
    }
    ++i;
    if (i < prep.size() && prep[i] == StepKind::WaitBack) {
      ++i;  // optional edge-4 WaitBack, correctly placed after HostCompute
    }
    if (i >= prep.size() || prep[i] != StepKind::H2D) {
      return "S_prep ordering violation: expected H2D after HostCompute (with "
             "an optional WaitBack between them per the Placement Invariant), "
             "got " +
             name_at(prep, i);
    }
    ++i;
    if (i < prep.size() && prep[i] == StepKind::SignalForward) {
      ++i;  // optional forward signal
    }
    if (i != prep.size()) {
      return "S_prep ordering violation: unexpected step " + name_at(prep, i) +
             " (prep allows only HostCompute -> [WaitBack]? -> H2D -> "
             "[SignalForward]?; a misplaced WaitBack or any Compute on the "
             "prep "
             "stream lands here)";
    }
  }

  // S_dev: [WaitForward]? -> Compute -> [SignalBack]?
  {
    size_t i = 0;
    if (i < dev.size() && dev[i] == StepKind::WaitForward) {
      ++i;  // optional forward wait
    }
    if (i >= dev.size() || dev[i] != StepKind::Compute) {
      return "S_dev ordering violation: expected Compute (optionally preceded "
             "by WaitForward), got " +
             name_at(dev, i) +
             " (no HostCompute/H2D permitted on the device stream)";
    }
    ++i;
    if (i < dev.size() && dev[i] == StepKind::SignalBack) {
      ++i;  // optional back signal
    }
    if (i != dev.size()) {
      return "S_dev ordering violation: unexpected step " + name_at(dev, i) +
             " (dev allows only [WaitForward]? -> Compute -> [SignalBack]?)";
    }
  }

  // Forward events are intra-plan: signals must match waits.
  size_t n_sig_fwd = 0;
  size_t n_wait_fwd = 0;
  for (StepKind k : kinds) {
    if (k == StepKind::SignalForward) ++n_sig_fwd;
    if (k == StepKind::WaitForward) ++n_wait_fwd;
  }
  if (n_sig_fwd != n_wait_fwd) {
    return "forward event pairing violation: " + std::to_string(n_sig_fwd) +
           " SignalForward vs " + std::to_string(n_wait_fwd) +
           " WaitForward (intra-plan forward events must be matched)";
  }
  // #6 FORWARD-GUARD: a two-stream plan (any event step present) MUST carry a
  // forward Ef pair to serialize the cross-stream H2D->Compute RAW (edge-3).
  // has_event with zero forward signals means only back events were wired --
  // the device Compute would then race the H2D that feeds it. (The count-match
  // above already caught the asymmetric case; this catches zero-forward.)
  if (has_event && n_sig_fwd == 0) {
    return "forward event pairing violation: a two-stream plan (event steps "
           "present) must carry a matched forward Ef pair to serialize the "
           "cross-stream H2D->Compute RAW; found none";
  }

  // Back events (WaitBack/SignalBack) are CROSS-LAUNCH: intentionally NOT
  // count-matched here. An unmatched WaitBack no-ops on an empty rolling slot
  // (first launch); an unmatched SignalBack simply publishes for a future
  // launch. Their placement is already enforced by the per-stream walks above.

  return "";
}

}  // namespace spyre

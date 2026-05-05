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

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <flex/allocator/alloc_address.hpp>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

// Forward declarations for flex types
namespace flex {
class RuntimeOperation;
}

namespace spyre {

/**
 * @brief Base class for host compute operation metadata
 *
 * This polymorphic base class allows different host compute operations
 * to define their own metadata structures while maintaining type safety
 * and avoiding JSON parsing overhead.
 */
struct HostComputeMetadata {
  virtual ~HostComputeMetadata() = default;
};

/**
 * @brief Function type for host-side computation operations
 *
 * This callable type represents host-side operations such as program
 * correction, collectives, and other host computations that need to be
 * executed as part of a job plan.
 *
 * @param metadata Reference to operation-specific metadata (contains buffer
 * sizes)
 * @param input_buffer Pointer to input buffer containing source data
 * @param output_buffer Pointer to output buffer for results
 */
using HostComputeFunction =
    std::function<void(HostComputeMetadata* metadata, void* output_buffer,
                       const void* input_buffer)>;

/**
 * @brief Context passed to JobPlanStep::construct() at launch time
 *
 * Carries runtime data available at LaunchKernel time that was not available
 * during PrepareKernel.
 */
struct LaunchContext {
  /**
   * @brief at::Tensor list of inputs and outputs
   *
   */
  const std::vector<at::Tensor>& inputs_outputs;
};

/**
 * @brief Polymorphic base class for JobPlan steps
 *
 * Each concrete subclass holds metadata resolved during PrepareKernel and
 * implements construct() to produce a RuntimeOperation at LaunchKernel time.
 * This factory method pattern eliminates special-case branching in
 * SpyreStream::Launch.
 *
 * All RuntimeOperation objects are transient: constructed inside construct(),
 * ownership transferred to RuntimeStream via launchOperation(), and destroyed
 * when the stream completes the operation. No RuntimeOperation is cached in
 * the JobPlan.
 */
class JobPlanStep {
 public:
  virtual ~JobPlanStep() = default;

  /**
   * @brief Construct a RuntimeOperation for this step
   *
   * Called by SpyreStream during LaunchKernel. Produces a fully-populated
   * RuntimeOperation using metadata stored during PrepareKernel and runtime
   * data from the LaunchContext.
   *
   * @param ctx Launch context containing composite addresses
   * @return Unique pointer to the constructed RuntimeOperation
   */
  virtual std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const = 0;

  /**
   * @brief Enable or disable pipeline barrier for this step
   *
   * Pipeline barriers control operation ordering within a stream. When enabled,
   * the operation waits for all prior operations to complete before starting.
   *
   * @param enable True to enable pipeline barrier, false to disable
   */
  void setPipelineBarrier(bool enable) {
    pipeline_barrier_ = enable;
  }

  /**
   * @brief Get the pipeline barrier setting for this step
   *
   * @return True if pipeline barrier is enabled, false otherwise
   */
  bool getPipelineBarrier() const {
    return pipeline_barrier_;
  }

 protected:
  bool pipeline_barrier_ = false;
};

/**
 * @brief Host-to-device transfer step
 *
 * All fields resolved during PrepareKernel. construct() produces a
 * RuntimeOperationH2D.
 *
 * When used for correction tensor DMA, the host_address points into a pinned
 * host buffer allocated during PrepareKernel and shared with the
 * JobPlanStepHostCompute that writes into it. The buffer is allocated once and
 * reused across launches — FIFO ordering within a stream guarantees the
 * HostCompute callback writes the buffer before the H2D reads it.
 */
class JobPlanStepH2D final : public JobPlanStep {
 public:
  /**
   * @brief Construct H2D step with raw host pointer
   *
   * @param host_address Host memory address (lifetime managed by JobPlan)
   * @param device_address Device memory address
   */
  JobPlanStepH2D(void* host_address, flex::CompositeAddress device_address)
      : host_address_(host_address),
        device_address_(std::move(device_address)) {}

  std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const override;

 private:
  void* host_address_;  // Non-owning pointer (JobPlan owns the buffer)
  flex::CompositeAddress device_address_;
};

/**
 * @brief Device-to-host transfer step
 *
 * All fields resolved during PrepareKernel. construct() produces a
 * RuntimeOperationD2H.
 */
class JobPlanStepD2H final : public JobPlanStep {
 public:
  /**
   * @brief Construct D2H step
   *
   * @param device_address Device memory address
   * @param host_address Host memory address (caller manages lifetime)
   */
  JobPlanStepD2H(flex::CompositeAddress device_address, void* host_address)
      : device_address_(std::move(device_address)),
        host_address_(host_address) {}

  std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const override;

 private:
  flex::CompositeAddress device_address_;
  void* host_address_;
};

/**
 * @brief Device compute launch step
 *
 * Binary address resolved during PrepareKernel. construct() produces a
 * RuntimeOperationCompute.
 */
class JobPlanStepCompute final : public JobPlanStep {
 public:
  /**
   * @brief Construct compute step
   *
   * @param binary_address Address of the program binary on device
   */
  explicit JobPlanStepCompute(flex::CompositeAddress binary_address)
      : binary_address_(std::move(binary_address)) {}

  std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const override;

 private:
  flex::CompositeAddress binary_address_;
};

/**
 * @brief Host-side computation step (e.g., program correction)
 *
 * The host function, compiler metadata, and a shared output buffer are stored
 * directly as members during PrepareKernel. The host function (e.g., the
 * program correction routine) is a predefined runtime function — SpyreCode's
 * ComputeOnHost command identifies which function to invoke, and torch-spyre
 * maps it to the corresponding built-in HostComputeFunction during SpyreCode
 * translation.
 *
 * The output buffer is a pointer to pinned host memory, shared
 * with the subsequent JobPlanStepH2D that transfers it to device. construct()
 * builds a closure capturing the function, metadata, composite addresses, and
 * the buffer, and produces a RuntimeOperationHostCallback.
 *
 * The shared buffer is allocated once during PrepareKernel and reused across
 * launches. For tiled execution, the same buffer is reused across iterations —
 * FIFO ordering guarantees each iteration's H2D consumes the buffer before the
 * next iteration's HostCompute overwrites it.
 */
class JobPlanStepHostCompute final : public JobPlanStep {
 public:
  /**
   * @brief Construct host compute step
   *
   * @param function Predefined runtime host compute function (e.g., program
   *                 correction), selected during SpyreCode translation
   * @param metadata Compiler-provided metadata (e.g., hcm.json / vdci.json
   *                 describing how symbolic values must be interpreted)
   * @param output_buffer Pinned host buffer (lifetime managed by JobPlan)
   */
  JobPlanStepHostCompute(HostComputeFunction function,
                         std::unique_ptr<HostComputeMetadata> metadata,
                         void* output_buffer)
      : function_(std::move(function)),
        metadata_(std::move(metadata)),
        output_buffer_(output_buffer) {}

  std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const override;

 private:
  HostComputeFunction function_;
  std::unique_ptr<HostComputeMetadata> metadata_;
  void* output_buffer_;  // Non-owning pointer (JobPlan owns the buffer)
};

/**
 * @brief Specializes a resident kernel using composite addresses
 *
 * Binary address resolved during PrepareKernel; tensor input/output addresses
 * extracted from ctx.composite_addresses during construct(). Produces a
 * RuntimeOperationComputeSpecializeResident.
 */
class JobPlanStepComputeSpecialize final : public JobPlanStep {
 public:
  /**
   * @brief Construct compute specialize step
   *
   * @param binary_address Address of the resident program binary on device
   */
  explicit JobPlanStepComputeSpecialize(flex::CompositeAddress binary_address)
      : binary_address_(std::move(binary_address)) {}

  std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const override;

 private:
  flex::CompositeAddress binary_address_;
};

/**
 * @brief A torch-spyre internal container for executing a unit of work
 *
 * A JobPlan bundles everything needed to execute a unit of work on a stream.
 * It is produced by translating a SpyreCode's Job Execution Plan after the Job
 * Preparation Plan has been executed. flex never sees a JobPlan — SpyreStream
 * extracts the operations and submits them to RuntimeStream.launchOperation()
 * as a vector<RuntimeOperation>.
 *
 * A JobPlan is self-contained: if a compute requires program correction, the
 * correction callback, the correction tensor DMA, and the device compute are
 * all separate steps in the same JobPlan. For pure data movement (e.g., tensor
 * .to(device) or binary loading), a JobPlan with only DMA steps is used.
 *
 * Producers:
 * - Backend compiler (deeptools) via torch-spyre: Deeptools produces a
 *   SpyreCode JSON per SDSC. torch-spyre translates the SpyreCode into a
 *   JobPlan — executing the Job Preparation Plan (allocations, binary loading)
 *   and translating the Job Execution Plan into JobPlanStep entries with
 *   resolved CompositeAddress values. A single torch.compile call may produce
 *   multiple SDSCs, resulting in multiple JobPlans.
 * - Communications libraries: Create JobPlans for inter-device data transfers,
 *   collective operations, or other multi-step communication patterns.
 * - torch-spyre: Assembles JobPlans for tensor .to(device) moves (single
 *   RuntimeOperationH2D step), tensor .to("cpu") readbacks (single
 *   RuntimeOperationD2H step), or any other sequence of operations it needs to
 *   containerize.
 */
struct JobPlan {
  /**
   * @brief Ordered sequence of steps
   *
   * During LaunchKernel, SpyreStream calls construct(ctx) on each step in
   * order, collecting the resulting RuntimeOperations, then submits them to
   * RuntimeStream.
   */
  std::vector<std::unique_ptr<JobPlanStep>> steps;

  /**
   * @brief Compiled tile dimensions from SpyreCode
   *
   * One entry per kernel input tensor. Used by SpyreStream for tiling
   * detection. Empty for pure DMA JobPlans (e.g., tensor .to(device)).
   */
  std::vector<std::vector<int64_t>> expected_input_shapes;

  /**
   * @brief Owning CompositeAddress of the program binary, and conditionally
   * program correction data and spillover tensor data
   *
   * The JobPlan owns this address and is responsible for its lifetime. When the
   * JobPlan is destroyed, the memory is freed.
   *
   * Set during PrepareKernel when it's loaded to device memory. Empty for pure
   * DMA JobPlans (e.g., tensor .to(device)) that don't involve compute
   * operations.
   */
  flex::CompositeAddress job_allocation;

  /**
   * @brief Pinned host buffers owned by this JobPlan
   *
   * Stores pinned memory buffers (e.g., for correction tensors) that must
   * remain alive for the lifetime of the JobPlan. Steps reference these
   * buffers via raw pointers. Buffers are automatically freed when JobPlan
   * is destroyed.
   *
   * Allocated using torch::empty() with .pinned_memory(true) during
   * PrepareKernel. The pinned memory ensures efficient DMA transfers and
   * prevents OS from swapping pages to disk.
   */
  std::vector<torch::Tensor> pinned_buffers;
};

}  // namespace spyre

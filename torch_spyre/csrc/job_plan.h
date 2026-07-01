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

#include <sys/mman.h>
#include <torch/types.h>

#include <cstdint>
#include <flex/flex.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "util/spyrecode.h"

namespace spyre {

// Forward declaration: JobPlanStep::construct() submits through SpyreStream
// rather than the raw flex::RuntimeStream handle.
class SpyreStream;

/**
 * @brief RAII wrapper for page-aligned and pinned host memory
 *
 * Allocates CPU memory aligned to page boundaries. Attempts to pin memory, but
 * gracefully falls back to unpinned memory if mlock fails.
 *
 * Memory is automatically freed and unpinned when the object is destroyed.
 */
class HostBuffer {
 public:
  /**
   * @brief Default constructor - creates empty buffer
   */
  HostBuffer() = default;

  /**
   * @brief Allocate aligned and optionally pinned host memory
   * @param size Size in bytes
   * @param alignment Alignment in bytes (default: system page size)
   */
  explicit HostBuffer(size_t size, size_t alignment = 0)
      : size_(size), pinned_(false) {
    // Use system page size if alignment not specified
    if (alignment == 0) {
      alignment_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    } else {
      alignment_ = alignment;
    }

    // 1. Allocate aligned memory
    int ret = posix_memalign(&ptr_, alignment_, size_);
    if (ret != 0 || ptr_ == nullptr) {
      throw std::bad_alloc();
    }

    // 2. Try to pin memory
    ret = mlock(ptr_, size_);
    if (ret == 0) {
      pinned_ = true;
    } else {
      // mlock failed - log warning but continue with unpinned memory
      // Common reasons: insufficient ulimit -l, not enough RAM
      TORCH_WARN_ONCE(
          "mlock failed: ", std::strerror(errno), ". ",
          "Using unpinned memory (still aligned). ",
          "For best performance, run 'ulimit -l unlimited' before starting.");
    }
  }

  ~HostBuffer() {
    if (ptr_) {
      if (pinned_) {
        munlock(ptr_, size_);
      }
      std::free(ptr_);
    }
  }

  // Disable copy (move-only)
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;

  // Enable move
  HostBuffer(HostBuffer&& other) noexcept
      : ptr_(other.ptr_),
        size_(other.size_),
        alignment_(other.alignment_),
        pinned_(other.pinned_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.alignment_ = 0;
    other.pinned_ = false;
  }

  HostBuffer& operator=(HostBuffer&& other) noexcept {
    if (this != &other) {
      // Clean up current resources
      if (ptr_) {
        if (pinned_) {
          munlock(ptr_, size_);
        }
        std::free(ptr_);
      }

      // Move from other
      ptr_ = other.ptr_;
      size_ = other.size_;
      alignment_ = other.alignment_;
      pinned_ = other.pinned_;

      // Reset other
      other.ptr_ = nullptr;
      other.size_ = 0;
      other.alignment_ = 0;
      other.pinned_ = false;
    }
    return *this;
  }

  /**
   * @brief Get pointer to the allocated memory
   * @return Pointer to aligned (and possibly pinned) memory
   */
  void* data() const {
    return ptr_;
  }

  /**
   * @brief Get size of the allocation
   * @return Size in bytes
   */
  size_t size() const {
    return size_;
  }

  /**
   * @brief Get alignment of the allocation
   * @return Alignment in bytes
   */
  size_t alignment() const {
    return alignment_;
  }

  /**
   * @brief Check if memory is pinned
   * @return True if mlock succeeded, false otherwise
   */
  bool is_pinned() const {
    return pinned_;
  }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
  size_t alignment_ = 0;
  bool pinned_ = false;
};

// Note: host compute metadata is defined in deeptools as Hcm, and host compute
// function is defined as deeptools::processComputeOnHostCommand

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
 * All RuntimeOperation objects are transient: constructed inside flex when
 * construct() calls the matching SpyreStream::launchXXX(), and destroyed when
 * the stream completes the operation. No RuntimeOperation is cached in the
 * JobPlan.
 */
class JobPlanStep {
 public:
  virtual ~JobPlanStep() = default;

  /**
   * @brief Build this step's flex operation params and launch them on the
   * stream
   *
   * Called by SpyreStream during LaunchKernel. Constructs the appropriate
   * flex operation params from metadata stored during PrepareKernel and
   * runtime data from the LaunchContext, then submits them via the matching
   * SpyreStream::launchXXX(). flex owns the RuntimeOperation lifecycle.
   *
   * @param ctx Launch context containing composite addresses
   * @param stream SpyreStream to launch the operation on
   */
  virtual void construct(LaunchContext& ctx,
                         const SpyreStream& stream) const = 0;

  /**
   * @brief Write step information to output stream
   *
   * Pure virtual method for derived classes to implement their specific
   * output format. Called by operator<<.
   *
   * @param os Output stream to write to
   */
  virtual void write(std::ostream& os) const = 0;

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
 * @brief Stream output operator for JobPlanStep
 *
 * @param os Output stream to write to
 * @param step JobPlanStep to output
 * @return Reference to the output stream
 */
inline std::ostream& operator<<(std::ostream& os, const JobPlanStep& step) {
  step.write(os);
  return os;
}

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

  void construct(LaunchContext& ctx, const SpyreStream& stream) const override;

  void write(std::ostream& os) const override;

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

  void construct(LaunchContext& ctx, const SpyreStream& stream) const override;

  void write(std::ostream& os) const override;

 private:
  flex::CompositeAddress device_address_;
  void* host_address_;
};

/**
 * @brief Device compute launch step
 *
 * All fields resolved during PrepareKernel. construct() produces a
 * RuntimeOperationCompute.
 */
class JobPlanStepCompute final : public JobPlanStep {
 public:
  /**
   * @brief Construct compute step
   *
   * @param program_address The program's FULL device allocation. flex bounds
   * the segment-7 translation to its total_size() (the real Allocate
   * footprint), never SEGMENT_SIZE.
   * @param bind_io_addresses Whether to bind the compute operation with inputs
   * and outputs addresses
   * @param bootstrap_offset Offset within the program allocation where
   * execution begins (0 = base; the program-correction region size when
   * correction precedes the binary)
   * @param name Human-readable kernel name forwarded to flex as
   * ComputeParams::kernel_name; surfaces in profiler events
   * (PendingRequest::node_name, aiupti activity name, FLEX JSON CBName).
   * Empty string ("") preserves the old behavior (no name).
   */
  explicit JobPlanStepCompute(flex::CompositeAddress program_address,
                              bool bind_io_addresses,
                              uint64_t bootstrap_offset = 0,
                              std::string name = "")
      : program_address_(std::move(program_address)),
        bind_io_addresses_(bind_io_addresses),
        bootstrap_offset_(bootstrap_offset),
        name_(std::move(name)) {}

  void construct(LaunchContext& ctx, const SpyreStream& stream) const override;

  void write(std::ostream& os) const override;

 private:
  flex::CompositeAddress program_address_;
  bool bind_io_addresses_;
  uint64_t bootstrap_offset_;
  std::string name_;
};

/**
 * @brief Host-side computation step (e.g., program correction)
 *
 * Stores compiler metadata (Hcm) and a shared output buffer during
 * PrepareKernel. The host computation uses
 * deeptools::processComputeOnHostCommand which takes Hcm metadata and performs
 * program correction or other host-side operations.
 *
 * The output buffer is a pointer to pinned host memory, shared
 * with the subsequent JobPlanStepH2D that transfers it to device. construct()
 * builds a closure capturing the metadata, composite addresses, and
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
   * @param hcm Compiler-provided metadata from deeptools (contains vdci and
   *            senConstants describing how symbolic values must be interpreted)
   * @param output_buffer Pinned host buffer (lifetime managed by JobPlan)
   * @param input_buffer Pinned host buffer (lifetime managed by JobPlan)
   * @param ishape used for constructing input buffer
   */
  JobPlanStepHostCompute(std::unique_ptr<Hcm> hcm, void* output_buffer,
                         const void* input_buffer, std::vector<int64_t> ishape)
      : hcm_(std::move(hcm)),
        output_buffer_(output_buffer),
        input_buffer_(input_buffer),
        ishape_(ishape) {}

  void construct(LaunchContext& ctx, const SpyreStream& stream) const override;

  void write(std::ostream& os) const override;

 private:
  std::unique_ptr<Hcm> hcm_;
  void* output_buffer_;       // Non-owning pointer (JobPlan owns the buffer)
  const void* input_buffer_;  // Non-owning pointer (JobPlan owns the buffer)
  std::vector<int64_t> ishape_;
};

/**
 * @brief A torch-spyre internal container for executing a unit of work
 *
 * A JobPlan bundles everything needed to execute a unit of work on a stream.
 * It is produced by translating a SpyreCode's Job Execution Plan after the Job
 * Preparation Plan has been executed. flex never sees a JobPlan — SpyreStream
 * translates each step into flex operation params and submits them via its
 * typed launchXXX() methods.
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
   * @brief vector of CompositeAddress with the first being the owning
   * CompositeAddress of the program binary, and conditionally program
   * correction data and spillover tensor data, and the rest being the
   * non-owning CompositeAddress of each program.
   *
   * The JobPlan owns this address and is responsible for its lifetime. When the
   * JobPlan is destroyed, the memory is freed.
   *
   * Set during PrepareKernel when it's loaded to device memory. Empty for pure
   * DMA JobPlans (e.g., tensor .to(device)) that don't involve compute
   * operations.
   */
  std::vector<flex::CompositeAddress> job_allocation;

  /**
   * @brief Compiled tile dimensions from SpyreCode
   *
   * One entry per kernel input tensor. Used by SpyreStream for tiling
   * detection. Empty for pure DMA JobPlans (e.g., tensor .to(device)).
   */
  std::vector<std::vector<int64_t>> expected_input_shapes;

  /**
   * @brief Pinned host buffers owned by this JobPlan
   *
   * Stores pinned memory buffers (e.g., for correction tensors) that must
   * remain alive for the lifetime of the JobPlan. Steps reference these
   * buffers via raw pointers. Buffers are automatically freed when JobPlan
   * is destroyed.
   *
   */
  // TODO(jni): not safe for multi streams. Make it per-stream. See #2520.
  std::vector<HostBuffer> pinned_buffers;

  /**
   * @brief Compiled programs
   *
   * One entry per program.
   */
  std::vector<std::string> inits;
};

/**
 * @brief Stream output operator for JobPlan
 *
 * Outputs a human-readable summary of the JobPlan including step types,
 * addresses, and metadata. Controlled by TORCH_SPYRE_DEBUG environment
 * variable.
 *
 * @param os Output stream to write to
 * @param plan JobPlan to output
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const JobPlan& plan);

}  // namespace spyre

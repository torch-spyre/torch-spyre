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

#include <filesystem>  // NOLINT
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "flex/flex.hpp"
#include "job_plan.h"
#include "spyre_stream.h"

namespace spyre {

/**
 * @brief Builder class for constructing JobPlan from SpyreCode
 *
 * This class encapsulates the logic for loading SpyreCode artifacts,
 * executing the job preparation plan, and translating the execution
 * plan into a JobPlan.
 */
class JobPlanBuilder {
 public:
  /**
   * @brief Construct a JobPlanBuilder
   *
   * @param spyrecode_dir Path to the SpyreCode directory
   * @param stream Optional stream to use for init transfers. If nullptr, uses
   * the current stream from getCurrentStream()
   */
  JobPlanBuilder(const std::string& spyrecode_dir, const SpyreStream* stream);

  /**
   * @brief Build the JobPlan
   *
   * Executes the preparation pipeline:
   * 1. Execute job preparation plan (allocate + init transfers)
   * 2. Translate job execution plan to JobPlan
   *
   * @return Prepared JobPlan
   */
  std::unique_ptr<JobPlan> build();

 private:
  /**
   * @brief Severity level for validation messages
   */
  enum class Severity {
    ERROR,    // Critical issue that prevents execution
    WARNING,  // Non-critical issue that may affect behavior
    INFO      // Informational message
  };

  /**
   * @brief A single validation message with severity
   */
  struct ValidationMessage {
    Severity severity;
    std::string message;
  };

  /**
   * @brief Result of JobPlan validation
   *
   * Contains the list of validation messages found during JobPlan validation.
   * An empty message list indicates successful validation.
   */
  struct ValidationResult {
    /**
     * @brief List of validation messages
     *
     * Each message describes a validation finding with its severity level.
     * Empty vector indicates the JobPlan passed all validation checks.
     */
    std::vector<ValidationMessage> messages;

    /**
     * @brief Check if validation was successful
     *
     * @return true if no error-level messages were found, false otherwise
     */
    bool isValid() const {
      for (const auto& msg : messages) {
        if (msg.severity == Severity::ERROR) {
          return false;
        }
      }
      return true;
    }
  };

  /**
   * @brief Validate the JobPlan structure and configuration
   *
   * Runs all validation checks (P2-13 through P2-16) and collects all messages.
   * This method is called during build() after JobPlan construction to
   * ensure the plan is well-formed before returning it.
   *
   * Validation checks include:
   * - P2-13: expected_input_shapes validation (blocked - not yet implemented)
   * - P2-14: JobPlan step ordering validation (blocked - not yet implemented)
   * - P2-15: host compute metadata validation (blocked - not yet implemented)
   * - P2-16: Additional structural validation (blocked - not yet implemented)
   *
   * @param job_plan The JobPlan to validate
   * @return ValidationResult containing list of validation messages with
   * severity. Empty message list indicates successful validation.
   *
   * @note This is currently a skeleton implementation that auto-validates
   *       (returns empty message list). Full validation logic will be added
   *       once the blocked dependencies are resolved.
   */
  ValidationResult validate(const JobPlan& job_plan) const;

  /// Path to the SpyreCode directory containing kernel artifacts
  const std::filesystem::path spyrecode_dir_;
  /// Parsed SpyreCode JSON containing preparation and execution plans
  nlohmann::json spyrecode_json_;
  /// Stream used for initialization transfers during preparation
  const SpyreStream stream_;
  /// Device memory allocation for the job (set during preparation and moved to
  /// JobPlan in translation)
  std::vector<flex::CompositeAddress> job_allocation_;
  /// Whether to bind inputs and outputs addresses for compute
  bool bind_io_addresses_;

  std::unordered_map<std::string, HostBuffer> pinned_buffer_map_;

  std::vector<std::string> inits_;

  /// Execute the job preparation plan (allocate + init transfers)
  void executeJobPreparationPlan();
  /// Execute an Allocate command from the preparation plan
  void executeAllocate(const nlohmann::json& cmd);
  /// Execute an InitTransfer command from the preparation plan
  void executeInitTransfer(const nlohmann::json& cmd);

  /// Translate the job execution plan to a JobPlan
  std::unique_ptr<JobPlan> translateJobExecPlan();
  /// Translate a single command from the execution plan to a JobPlanStep.
  /// step_idx is the command's position within JobExecPlan; threaded through
  /// so compute steps can disambiguate their profiler kernel name.
  std::unique_ptr<JobPlanStep> translateCommand(const nlohmann::json& cmd,
                                                size_t step_idx);
  /// Translate a ComputeOnDevice command to a JobPlanStepCompute
  std::unique_ptr<JobPlanStep> translateComputeOnDevice(
      const nlohmann::json& cmd, size_t step_idx);
  /// Translate a ComputeOnHost command to a JobPlanStepHostCompute
  std::unique_ptr<JobPlanStep> translateComputeOnHost(
      const nlohmann::json& cmd);
  /// Translate a DataTransfer command to a JobPlanStepH2D or JobPlanStepD2H
  std::unique_ptr<JobPlanStep> translateDataTransfer(const nlohmann::json& cmd);
};

/**
 * @brief Prepare a kernel from a SpyreCode directory
 *
 * Factory function that creates the JobPlan.
 *
 * @param spyrecode_dir Path to the SpyreCode directory
 * @param stream Optional stream to use for init transfers. If nullptr, uses the
 * current stream from getCurrentStream()
 * @return Prepared JobPlan
 */
std::unique_ptr<JobPlan> prepareKernel(const std::string& spyrecode_dir,
                                       const SpyreStream* stream = nullptr);

}  // namespace spyre

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

#include "flex/flex.hpp"
#include "spyre_stream.h"

namespace spyre {

// Forward declarations
class JobPlan;
class JobPlanStep;

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
  /// Path to the SpyreCode directory containing kernel artifacts
  const std::filesystem::path spyrecode_dir_;
  /// Parsed SpyreCode JSON containing preparation and execution plans
  nlohmann::json spyrecode_json_;
  /// Stream used for initialization transfers during preparation
  const SpyreStream stream_;
  /// Device memory allocation for the job (set during preparation and moved to
  /// JobPlan in translation)
  std::optional<flex::CompositeAddress> job_allocation_;
  /// Whether to bind inputs and outputs addresses for compute
  bool bind_io_addresses_;

  /// Execute the job preparation plan (allocate + init transfers)
  void executeJobPreparationPlan();
  /// Execute an Allocate command from the preparation plan
  void executeAllocate(const nlohmann::json& cmd);
  /// Execute an InitTransfer command from the preparation plan
  void executeInitTransfer(const nlohmann::json& cmd);

  /// Translate the job execution plan to a JobPlan
  std::unique_ptr<JobPlan> translateJobExecPlan();
  /// Translate a single command from the execution plan to a JobPlanStep
  std::unique_ptr<JobPlanStep> translateCommand(const nlohmann::json& cmd);
  /// Translate a ComputeOnDevice command to a JobPlanStepCompute
  std::unique_ptr<JobPlanStep> translateComputeOnDevice(
      const nlohmann::json& cmd);
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

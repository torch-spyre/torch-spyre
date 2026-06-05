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

#include "prepare_kernel.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "job_plan.h"
#include "logging.h"
#include "spyre_allocator.h"
#include "util/spyrecode.h"

namespace spyre {

/**
 * @brief Enum for SpyreCode command types
 */
enum class SpyreCodeCommandType {
  ComputeOnDevice,
  ComputeOnHost,
  DataTransfer,
  Allocate,
  InitTransfer,
  Unknown
};

/**
 * @brief Enum for DataTransfer direction
 */
enum class TransferDirection {
  HostToDevice,  // H2D
  DeviceToHost,  // D2H
  Unknown
};

/**
 * @brief Parse command type string to enum
 * @param command_str The command string from JSON
 * @return Corresponding SpyreCodeCommandType enum value
 */
static SpyreCodeCommandType parse_command_type(const std::string& command_str) {
  static const std::unordered_map<std::string, SpyreCodeCommandType> mapping = {
      {"ComputeOnDevice", SpyreCodeCommandType::ComputeOnDevice},
      {"ComputeOnHost", SpyreCodeCommandType::ComputeOnHost},
      {"DataTransfer", SpyreCodeCommandType::DataTransfer},
      {"Allocate", SpyreCodeCommandType::Allocate},
      {"InitTransfer", SpyreCodeCommandType::InitTransfer}};

  auto it = mapping.find(command_str);
  return it != mapping.end() ? it->second : SpyreCodeCommandType::Unknown;
}

/**
 * @brief Parse transfer direction string to enum
 * @param dirn_str The direction string from JSON ("false" = H2D, "true" = D2H)
 * @return Corresponding TransferDirection enum value
 */
static TransferDirection parse_transfer_direction(const std::string& dirn_str) {
  if (dirn_str == "false") {
    return TransferDirection::HostToDevice;
  } else if (dirn_str == "true") {
    return TransferDirection::DeviceToHost;
  }
  return TransferDirection::Unknown;
}

// Program segment boundaries for validation
static const uint64_t prog_offset_base = flex::PROG_OFFSET_BASE;
static const uint64_t prog_offset_limit =
    flex::PROG_OFFSET_BASE + flex::SEGMENT_SIZE;

/**
 * @brief Helper to compute CompositeAddress with offset from device_addr for
 * program
 */
static flex::CompositeAddress compute_offset_address(
    const flex::CompositeAddress& job_allocation, uint64_t dev_ptr,
    size_t size = 0) {
  // Create CompositeAddress using program_address with offset
  TORCH_CHECK(job_allocation.chunks().size() == 1,
              "job_allocation must have 1 chunk");

  // Validate device pointer is within program segment bounds
  TORCH_CHECK(dev_ptr >= prog_offset_base && dev_ptr < prog_offset_limit,
              "Device pointer 0x", std::hex, dev_ptr,
              " is out of program segment bounds [0x", prog_offset_base, ", 0x",
              prog_offset_limit, ")");

  // Calculate offset
  uint64_t offset = dev_ptr - prog_offset_base;
  if (size == 0) {
    size = job_allocation.total_size() - offset;
  }

  // Get the first chunk and add offset to its address
  const auto& base_chunk = job_allocation.chunks()[0];
  flex::LogicalAddress offset_addr(base_chunk.addr.region_id,
                                   base_chunk.addr.offset + offset);
  flex::Chunk offset_chunk(offset_addr, size, base_chunk.domain_id);
  return flex::CompositeAddress(offset_chunk);
}

/**
 * @brief Helper to check if a file exists
 */
static bool file_exists(const std::filesystem::path& path) {
  return std::filesystem::exists(path) &&
         std::filesystem::is_regular_file(path);
}

/**
 * @brief Helper to read entire file into string
 */
static std::string read_file_to_string(const std::filesystem::path& path) {
  TORCH_CHECK(file_exists(path),
              "Path does not exist, or not a regular file: ", path.string());
  std::ifstream file(path, std::ios::binary);
  TORCH_CHECK(file, "Failed to open file: ", path.string());

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

JobPlanBuilder::JobPlanBuilder(const std::string& spyrecode_dir,
                               const SpyreStream* stream)
    : spyrecode_dir_(spyrecode_dir),
      stream_(stream ? *stream : getCurrentStream()) {
  // Validate directory exists
  TORCH_CHECK(std::filesystem::exists(spyrecode_dir_),
              "SpyreCode directory does not exist: ", spyrecode_dir_.string());

  TORCH_CHECK(std::filesystem::is_directory(spyrecode_dir_),
              "Path is not a directory: ", spyrecode_dir_.string());

  // Load and parse spyrecode.json
  auto spyrecode_json_path = spyrecode_dir_ / "spyrecode.json";
  TORCH_CHECK(file_exists(spyrecode_json_path),
              "Required file spyrecode.json not found in directory: ",
              spyrecode_dir_.string());

  std::string json_str = read_file_to_string(spyrecode_json_path);

  try {
    spyrecode_json_ = nlohmann::json::parse(json_str);
  }
  catch (const std::exception& e) {
    TORCH_CHECK(false, "Failed to parse spyrecode.json: ", e.what());
  }

  // Validate required fields
  TORCH_CHECK(spyrecode_json_.contains("JobPreparationPlan"),
              "SpyreCode JSON missing 'JobPreparationPlan' array");

  TORCH_CHECK(spyrecode_json_.contains("JobExecPlan"),
              "SpyreCode JSON missing 'JobExecPlan' array");
}

void JobPlanBuilder::executeAllocate(const nlohmann::json& cmd) {
  TORCH_CHECK(cmd.contains("command") && cmd["command"].is_string(),
              "Allocate command missing 'command' field");

  std::string allocate_type_str = cmd["command"].get<std::string>();
  SpyreCodeCommandType allocate_type = parse_command_type(allocate_type_str);
  TORCH_CHECK(allocate_type == SpyreCodeCommandType::Allocate,
              "Expected 'Allocate' command, got: " + allocate_type_str);

  const auto& allocate_props =
      cmd.contains("properties") ? cmd["properties"] : nlohmann::json();

  TORCH_CHECK(allocate_props.contains("size"),
              "Allocate command missing 'size' property");

  std::string size_str = allocate_props["size"].get<std::string>();
  size_t size = std::stoull(size_str);

  auto& allocator = SpyreAllocator::instance();
  flex::AllocationDirective directive(flex::PlacementPolicy::Bind, {0},
                                      std::nullopt, flex::MemoryType::Program);
  c10::DataPtr allocated_ptr = allocator.allocate(size, directive);

  job_allocation_ =
      std::move(static_cast<SharedOwnerCtx*>(allocated_ptr.get_context())
                    ->composite_addr);
}

void JobPlanBuilder::executeInitTransfer(const nlohmann::json& cmd) {
  TORCH_CHECK(cmd.contains("command") && cmd["command"].is_string(),
              "InitTransfer command missing 'command' field");

  std::string init_type_str = cmd["command"].get<std::string>();
  SpyreCodeCommandType init_type = parse_command_type(init_type_str);
  TORCH_CHECK(init_type == SpyreCodeCommandType::InitTransfer,
              "Expected 'InitTransfer' command, got: " + init_type_str);

  const auto& init_props =
      cmd.contains("properties") ? cmd["properties"] : nlohmann::json();

  TORCH_CHECK(init_props.contains("init_bin_file"),
              "InitTransfer command missing 'init_bin_file' property");

  std::string binary_file = init_props["init_bin_file"].get<std::string>();
  std::filesystem::path binary_path = spyrecode_dir_ / binary_file;

  std::string binary_data = read_file_to_string(binary_path);

  TORCH_CHECK(init_props.contains("dev_ptr"),
              "InitTransfer command missing 'dev_ptr' property");

  std::string dev_ptr_str = init_props["dev_ptr"].get<std::string>();
  uint64_t dev_ptr = std::stoull(dev_ptr_str);

  TORCH_CHECK(init_props.contains("size"),
              "InitTransfer command missing 'size' property");

  std::string init_size_str = init_props["size"].get<std::string>();
  size_t init_size = std::stoull(init_size_str);

  auto device_addr =
      compute_offset_address(job_allocation_.value(), dev_ptr, init_size);

  stream_.copyProgramAsync(
      const_cast<void*>(static_cast<const void*>(binary_data.data())),
      &device_addr);
}

void JobPlanBuilder::executeJobPreparationPlan() {
  auto job_prep_plan = spyrecode_json_["JobPreparationPlan"];
  TORCH_CHECK(job_prep_plan.is_array() && job_prep_plan.size() >= 2,
              "JobPreparationPlan must be an array with at least 2 commands (1 "
              "Allocate and 1+ InitTransfer)");

  // Execute Allocate command (first item)
  executeAllocate(job_prep_plan[0]);

  // Execute InitTransfer commands (remaining items)
  for (size_t i = 1; i < job_prep_plan.size(); ++i) {
    executeInitTransfer(job_prep_plan[i]);
  }
  stream_.synchronize();
}

std::unique_ptr<JobPlanStep> JobPlanBuilder::translateComputeOnDevice(
    const nlohmann::json& cmd) {
  TORCH_CHECK(cmd.contains("job_bin_ptr"),
              "ComputeOnDevice command missing 'job_bin_ptr' property");

  std::string job_bin_ptr_str = cmd["job_bin_ptr"].get<std::string>();
  uint64_t job_bin_ptr = std::stoull(job_bin_ptr_str);

  auto job_bin_addr =
      compute_offset_address(job_allocation_.value(), job_bin_ptr);
  // Create RuntimeOperationCompute with the allocated program address
  return std::make_unique<JobPlanStepCompute>(std::move(job_bin_addr),
                                              bind_io_addresses_, job_bin_ptr);
}

std::unique_ptr<JobPlanStep> JobPlanBuilder::translateComputeOnHost(
    const nlohmann::json& cmd) {
  // Parse ohandle
  TORCH_CHECK(cmd.contains("ohandle"),
              "ComputeOnHost command missing 'ohandle' property");
  std::string ohandle = cmd["ohandle"].get<std::string>();

  // Allocate pinned buffer
  auto it = pinned_buffer_map_.find(ohandle);
  TORCH_CHECK(it == pinned_buffer_map_.end(), "ohandle '", ohandle,
              "' already exists in pinned buffer map");
  TORCH_CHECK(cmd.contains("size"),
              "ComputeOnHost command missing 'size' property");
  std::string size_str = cmd["size"].get<std::string>();
  size_t buffer_size = std::stoull(size_str);
  pinned_buffer_map_[ohandle] = HostBuffer(buffer_size);

  // Parse ishape
  // TODO(jni): further discussion is required on "ishape". See #2522. For now,
  // it's vector<int64_t>, and it's {0}, it's for fake symbols
  TORCH_CHECK(cmd.contains("ishape"),
              "ComputeOnHost command missing 'ishape' property");
  const nlohmann::json& ishape_json = cmd["ishape"];
  TORCH_CHECK(ishape_json.is_array(),
              "ComputeOnHost 'ishape' must be an array");
  std::vector<int64_t> ishape;
  for (const auto& dim : ishape_json) {
    TORCH_CHECK(dim.is_string(),
                "ComputeOnHost 'ishape' elements must be strings");
    std::string dim_str = dim.get<std::string>();
    ishape.push_back(std::stoll(dim_str));
  }

  // Parse ihandle
  void* inp_ptr = nullptr;
  TORCH_CHECK(cmd.contains("ihandle"),
              "ComputeOnHost command missing 'ihandle' property");
  std::string ihandle = cmd["ihandle"].get<std::string>();
  if (!ihandle.empty()) {
    // Get input buffer from pinned_buffer_map_
    it = pinned_buffer_map_.find(ihandle);
    TORCH_CHECK(it != pinned_buffer_map_.end(), "ihandle '", ihandle,
                "' not found in pinned buffer map");
    inp_ptr = it->second.data();
  }

  // Parse hcm JSON
  TORCH_CHECK(cmd.contains("hcm"),
              "ComputeOnHost command missing 'hcm' property");
  const nlohmann::json& hcm_json = cmd["hcm"];

  // Create Hcm object and import from JSON string
  auto hcm_data = std::make_unique<Hcm>();
  std::string hcm_json_str = hcm_json.dump();
  bool import_success = hcm_data->importJsonStr(hcm_json_str);
  TORCH_CHECK(import_success, "Failed to import Hcm from JSON");

  // Create and return JobPlanStepHostCompute
  return std::make_unique<JobPlanStepHostCompute>(
      std::move(hcm_data), pinned_buffer_map_[ohandle].data(), inp_ptr, ishape);
}

std::unique_ptr<JobPlanStep> JobPlanBuilder::translateDataTransfer(
    const nlohmann::json& cmd) {
  // Extract direction: 0 = H2D, 1 = D2H
  TORCH_CHECK(cmd.contains("dirn"),
              "DataTransfer command missing 'dirn' property");

  std::string dirn_str = cmd["dirn"].get<std::string>();
  TransferDirection direction = parse_transfer_direction(dirn_str);

  switch (direction) {
    case TransferDirection::HostToDevice: {
      // Host-to-Device transfer
      // Extract host and device addresses
      TORCH_CHECK(cmd.contains("dev_ptr"),
                  "DataTransfer H2D missing 'dev_ptr' property");

      TORCH_CHECK(cmd.contains("size"),
                  "DataTransfer H2D missing 'size' property");

      std::string dev_ptr_str = cmd["dev_ptr"].get<std::string>();
      std::string size_str = cmd["size"].get<std::string>();

      TORCH_CHECK(cmd.contains("host_handle"),
                  "DataTransfer H2D missing 'host_handle' property");
      std::string host_handle_str = cmd["host_handle"].get<std::string>();

      // Get host buffer from pinned_buffer_map_
      auto it = pinned_buffer_map_.find(host_handle_str);
      TORCH_CHECK(it != pinned_buffer_map_.end(), "Host handle '",
                  host_handle_str, "' not found in pinned buffer map");
      void* host_addr = it->second.data();
      uint64_t device_ptr = std::stoull(dev_ptr_str);
      size_t transfer_size = std::stoull(size_str);

      // Compute CompositeAddress with offset
      flex::CompositeAddress comp_addr = compute_offset_address(
          job_allocation_.value(), device_ptr, transfer_size);

      return std::make_unique<JobPlanStepH2D>(host_addr, std::move(comp_addr));
    }

    case TransferDirection::DeviceToHost: {
      // Device-to-Host transfer
      // Extract host and device addresses
      TORCH_CHECK(cmd.contains("dev_ptr"),
                  "DataTransfer D2H missing 'dev_ptr' property");

      TORCH_CHECK(cmd.contains("size"),
                  "DataTransfer D2H missing 'size' property");

      std::string dev_ptr_str = cmd["dev_ptr"].get<std::string>();
      std::string size_str = cmd["size"].get<std::string>();

      TORCH_CHECK(cmd.contains("host_handle"),
                  "DataTransfer D2H missing 'host_handle' property");
      std::string host_handle_str = cmd["host_handle"].get<std::string>();

      uint64_t device_ptr = std::stoull(dev_ptr_str);
      size_t transfer_size = std::stoull(size_str);

      // Allocate pinned buffer
      auto it = pinned_buffer_map_.find(host_handle_str);
      TORCH_CHECK(it == pinned_buffer_map_.end(), "Host handle '",
                  host_handle_str, "' already exists in pinned buffer map");

      pinned_buffer_map_[host_handle_str] = HostBuffer(transfer_size);
      void* host_addr = pinned_buffer_map_[host_handle_str].data();

      // Compute CompositeAddress with offset from device_addr
      flex::CompositeAddress comp_addr = compute_offset_address(
          job_allocation_.value(), device_ptr, transfer_size);

      return std::make_unique<JobPlanStepD2H>(std::move(comp_addr), host_addr);
    }

    case TransferDirection::Unknown:
    default:
      TORCH_CHECK(false, "Invalid DataTransfer direction: ", dirn_str);
  }

  // Unreachable, but needed to suppress compiler warning
  return nullptr;
}

std::unique_ptr<JobPlanStep> JobPlanBuilder::translateCommand(
    const nlohmann::json& cmd) {
  TORCH_CHECK(cmd.contains("command") && cmd["command"].is_string(),
              "SpyreCode command missing 'command' field");

  std::string command_type_str = cmd["command"].get<std::string>();
  SpyreCodeCommandType command_type = parse_command_type(command_type_str);
  const auto& properties =
      cmd.contains("properties") ? cmd["properties"] : nlohmann::json();

  switch (command_type) {
    case SpyreCodeCommandType::ComputeOnDevice:
      return translateComputeOnDevice(properties);

    case SpyreCodeCommandType::ComputeOnHost:
      return translateComputeOnHost(properties);

    case SpyreCodeCommandType::DataTransfer:
      return translateDataTransfer(properties);

    case SpyreCodeCommandType::Unknown:
    default:
      TORCH_CHECK(false, "Unknown SpyreCode command type: ", command_type_str);
  }

  // Unreachable, but needed to suppress compiler warning
  return nullptr;
}

std::unique_ptr<JobPlan> JobPlanBuilder::translateJobExecPlan() {
  auto job_exec_plan = spyrecode_json_["JobExecPlan"];
  TORCH_CHECK(job_exec_plan.is_array(), "JobExecPlan must be an array");

  // TODO(jni): further discussions is required on the condition to specialize
  // addresses
  bind_io_addresses_ = true;

  // Parse each command in the JobExecPlan and create JobPlanSteps
  std::vector<std::unique_ptr<JobPlanStep>> steps;
  for (const auto& command : job_exec_plan) {
    try {
      steps.push_back(translateCommand(command));
    }
    catch (const std::exception& e) {
      TORCH_CHECK(false, "Failed to parse SpyreCode command: ", e.what());
    }
  }

  // TODO(jni): expected_input_shapes to be added once provided in SpyreCode
  // Create pinned_buffers vector from pinned_buffer_map_
  // Move tensors from map to avoid unnecessary reference count increments
  std::vector<HostBuffer> pinned_buffers;
  pinned_buffers.reserve(pinned_buffer_map_.size());
  for (auto& [ohandle, tensor] : pinned_buffer_map_) {
    pinned_buffers.push_back(std::move(tensor));
  }

  // Create and return the JobPlan
  // Use brace initialization to construct JobPlan with moved members
  return std::make_unique<JobPlan>(JobPlan{
      std::move(steps),                    // steps
      std::move(job_allocation_.value()),  // job_allocation
      {},                                  // expected_input_shapes
      std::move(pinned_buffers)            // pinned_buffers
  });
}

JobPlanBuilder::ValidationResult JobPlanBuilder::validate(
    const JobPlan& job_plan) const {
  JobPlanBuilder::ValidationResult result;

  // P2-13: expected_input_shapes validation
  // TODO(johngontaryk): Implement once expected_input_shapes validation logic
  // is defined
  // - Verify expected_input_shapes is non-empty for compute JobPlans
  // - Verify shape dimensions are positive
  // - Verify shape count matches number of input tensors

  // P2-14: JobPlan step ordering validation
  // TODO(johngontaryk): Implement once step ordering rules are defined
  // - Verify HostCompute steps precede their corresponding H2D steps
  // - Verify H2D steps precede Compute steps that depend on them
  // - Verify no circular dependencies in step ordering

  // P2-15: Host compute metadata validation
  // TODO(johngontaryk): Implement once host compute metadata structure is
  // finalized
  // - Verify metadata is non-null for HostCompute steps
  // - Verify output buffer sizes match metadata specifications
  // - Verify function pointers are valid

  // P2-16: Additional structural validation
  // TODO(johngontaryk): Implement additional validation checks as needed
  // - Verify job_allocation is valid for compute JobPlans
  // - Verify pinned_buffers are properly allocated
  // - Verify CompositeAddress validity in steps

  // Skeleton implementation: auto-validate (return empty message list)
  // Full validation logic will be added once blocked dependencies are resolved
  return result;
}

std::unique_ptr<JobPlan> JobPlanBuilder::build() {
  // Execute job preparation plan (allocate + init transfers)
  executeJobPreparationPlan();

  // Translate job execution plan to JobPlan
  auto job_plan = translateJobExecPlan();

  // Validate the JobPlan before returning
  auto validation_result = validate(*job_plan);
  if (!validation_result.isValid()) {
    std::string error_msg = "JobPlan validation failed:\n";
    for (const auto& msg : validation_result.messages) {
      if (msg.severity == Severity::ERROR) {
        error_msg += "  ERROR: " + msg.message + "\n";
      } else if (msg.severity == Severity::WARNING) {
        TORCH_WARN("JobPlan validation warning: ", msg.message);
        error_msg += "  WARNING: " + msg.message + "\n";
      }
    }
    TORCH_CHECK(false, error_msg);
  }

  return job_plan;
}

std::unique_ptr<JobPlan> prepareKernel(const std::string& spyrecode_dir,
                                       const SpyreStream* stream) {
  JobPlanBuilder builder(spyrecode_dir, stream);
  auto jobplan = builder.build();

  // Dump JobPlan if debug logging is enabled
  DEBUGINFO("JobPlan:\n", *jobplan);

  return jobplan;
}

}  // namespace spyre

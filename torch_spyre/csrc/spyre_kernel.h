/*
 * Copyright 2025 The Torch-Spyre Authors.
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

#include <ATen/core/Tensor.h>
#include <c10/core/Allocator.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

namespace spyre {

class SpyreStream;
struct KernelArtifacts {
  std::vector<uint8_t> init_bin;  // Program binary from init.txt
  c10::DataPtr device_alloc;
  uint64_t program_size;  // (bytes)

  std::string bundle_mlir_path;  // Path to bundle.mlir
};

std::ostream& operator<<(std::ostream& os, const KernelArtifacts& k);

/**
 * @brief Read hex-encoded init.txt file (production-tested approach)
 *
 * Based on deeptools/dip/dip.cpp:runReverseDip() but ~10x faster
 * Each line: 256 hex chars = 128 bytes
 */
std::vector<uint8_t> readHexEncodedFile(const std::string& filepath);

std::string get_init_path(const std::string& code_dir);

std::string get_pagi_path(const std::string& code_dir);

KernelArtifacts& getOrLoadArtifacts(const std::string& code_dir,
                                    const SpyreStream& stream);
void launchKernel(const std::string& code_dir,
                  const std::vector<at::Tensor>& args);

void clearArtifactCache();

}  // namespace spyre

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

#include "spyre_kernel.h"

#include <c10/util/Exception.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "logging.h"
#include "spyre_allocator.h"
#include "spyre_stream.h"

namespace fs = std::filesystem;

namespace spyre {
std::ostream& operator<<(std::ostream& os, const KernelArtifacts& k) {
  os << "KernelArtifacts {\n";
  os << "  init_bin.size       = " << k.init_bin.size() << " bytes\n";
  os << "  program_size        = " << k.program_size << " bytes\n";
  os << "  bundle_mlir_path    = \"" << k.bundle_mlir_path << "\"\n";
  os << "  sdsc_json_path      = \"" << k.sdsc_json_path << "\"\n";
  os << "}";
  return os;
}

PagiJsonConfig readPagiJson(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open JSON file");
  }

  json j;
  file >> j;

  PagiJsonConfig cfg;
  cfg.dsName = j.at("dsName_").get<std::string>();
  cfg.isMarker = j.at("isMarker_").get<bool>();
  cfg.defaultAddr = std::stoull(j.at("defaultAddr_").get<std::string>());

  cfg.addrMap = j.at("addrMap_").get<std::vector<uint64_t>>();
  cfg.addrMapTag = j.at("addrMapTag_").get<std::vector<std::string>>();
  cfg.inputSym = j.at("inputSym_").get<std::vector<std::string>>();
  cfg.addrIdxSym = j.at("addrIdxSym_").get<std::vector<std::string>>();

  cfg.variableDefs =
      j.at("variableDefs_").get<std::unordered_map<std::string, std::string>>();
  return cfg;
}

std::vector<uint8_t> readHexEncodedFile(const std::string& filepath) {
  std::ifstream inpFile(filepath);
  if (!inpFile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filepath);
  }

  std::vector<uint8_t> binary_data;
  std::string line;

  while (std::getline(inpFile, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    for (size_t pos = 0; pos + 8 <= line.length(); pos += 8) {
      std::string hex_chunk = line.substr(pos, 8);

      std::istringstream strm(hex_chunk);
      uint32_t value;
      strm >> std::hex >> value;

      if (strm.fail()) {
        throw std::runtime_error("Invalid hex at position " +
                                 std::to_string(pos) + " in " + filepath);
      }

      // Convert uint32_t to bytes (big-endian as in init.txt)
      binary_data.push_back((value >> 24) & 0xFF);
      binary_data.push_back((value >> 16) & 0xFF);
      binary_data.push_back((value >> 8) & 0xFF);
      binary_data.push_back(value & 0xFF);
    }
  }

  if (binary_data.empty()) {
    throw std::runtime_error("No data decoded from file: " + filepath);
  }

  return binary_data;
}

std::string get_init_path(const std::string& g2_path) {
  fs::path p(g2_path);
  fs::path dir = p.parent_path();
  std::string kernel_name = dir.filename().string();

  std::string program_dir =
      "loadprogram_to_device/" + kernel_name + "-SenProgSend";

  return (dir / program_dir).string();
}

std::string get_pagi_path(const std::string& g2_path) {
  fs::path p(g2_path);
  fs::path dir = p.parent_path();
  std::string kernel_name = dir.filename().string();

  std::string program_dir = "execute/" + kernel_name;

  return (dir / program_dir).string();
}

// Cache: g2_path -> artifacts (loaded once)
std::unordered_map<std::string, KernelArtifacts> g_artifact_cache;

KernelArtifacts& getOrLoadArtifacts(const std::string& g2_path,
                                    const SpyreStream& stream) {
  // Check cache first
  auto it = g_artifact_cache.find(g2_path);
  if (it != g_artifact_cache.end()) {
    return it->second;
  }

  KernelArtifacts arts;

  // Detect Format B: check for bundle.mlir
  std::string bundle_path = g2_path + "/bundle.mlir";
  // Store bundle.mlir path for future JIT compilation
  arts.bundle_mlir_path = bundle_path;
  TORCH_CHECK(std::filesystem::exists(bundle_path),
              "Bundle not found: ", bundle_path);

  // Read init.bin (hex-encoded program binary)
  std::string init_path = get_init_path(g2_path) + "/init.txt";
  arts.init_bin = readHexEncodedFile(init_path);  // Helper to decode hex

  size_t program_size = arts.init_bin.size();
  auto& allocator = SpyreAllocator::instance();
  arts.device_alloc = std::move(allocator.allocate(program_size));
  auto* ctx = static_cast<SharedOwnerCtx*>(arts.device_alloc.get_context());
  stream.copyProgramAsync(arts.init_bin.data(), &ctx->composite_addr);
  stream.synchronize();

  arts.sdsc_json_path = g2_path + "/sdsc_0.json";
  TORCH_CHECK(std::filesystem::exists(arts.sdsc_json_path),
              "SuperDSC not found: ", arts.sdsc_json_path);

  // Cache and return
  g_artifact_cache[g2_path] = std::move(arts);
  std::cout << arts << std::endl;
  return g_artifact_cache[g2_path];
}

}  // namespace spyre

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
#include <memory>
#include <mutex>
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

  auto c2u = [](int8_t c) -> int {
    return (c >= 'A') ? (c >= 'a') ? (c - 'a' + 10) : (c - 'A' + 10)
                      : (c - '0');
  };

  std::vector<uint8_t> binary_data;
  std::string line;

  while (std::getline(inpFile, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }
    if (line.length() != 256) {
      throw std::runtime_error(
          "In readHexEncodedFile, line is not 256 chars in " + filepath);
    }

    // Read hex in reverse order (right-to-left), matching senlib's
    // parse_flit(). The hardware expects this byte ordering — reading
    // left-to-right (big-endian) produces an invalid program image causing QGI
    // errors (prep_zero_flit_cnt).
    for (auto rit = line.rbegin(); rit != line.rend();) {
      uint8_t byte = static_cast<uint8_t>(c2u(*rit++) + (c2u(*rit++) << 4));
      binary_data.push_back(byte);
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
std::mutex g_artifact_cache_mtx;  // protects g_artifact_cache and g_key_mtxs
std::unordered_map<std::string, std::unique_ptr<std::mutex>> g_key_mtxs;

KernelArtifacts& getOrLoadArtifacts(const std::string& g2_path,
                                    const SpyreStream& stream) {
  // Fast path: check cache under lock
  {
    std::lock_guard<std::mutex> lock(g_artifact_cache_mtx);
    auto it = g_artifact_cache.find(g2_path);
    if (it != g_artifact_cache.end()) {
      return it->second;
    }
    // Ensure per-key mutex exists (also under lock)
    auto& key_mtx = g_key_mtxs[g2_path];
    if (!key_mtx) {
      key_mtx = std::make_unique<std::mutex>();
    }
  }

  // Per-key lock: only one thread loads a given key
  std::mutex* key_mtx = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_artifact_cache_mtx);
    key_mtx = g_key_mtxs[g2_path].get();
  }
  std::lock_guard<std::mutex> key_lock(*key_mtx);

  // Double-check after acquiring per-key lock
  {
    std::lock_guard<std::mutex> lock(g_artifact_cache_mtx);
    auto it = g_artifact_cache.find(g2_path);
    if (it != g_artifact_cache.end()) {
      return it->second;
    }
  }

  // Slow path: load artifacts — only one thread per key reaches here
  KernelArtifacts arts;

  // Detect Format B: check for bundle.mlir
  fs::path g2_dir = fs::path(g2_path).parent_path();
  std::string bundle_path = (g2_dir / "bundle.mlir").string();
  // Store bundle.mlir path for future JIT compilation
  arts.bundle_mlir_path = bundle_path;
  TORCH_CHECK(std::filesystem::exists(bundle_path),
              "Bundle not found: ", bundle_path);

  // Read init.bin (hex-encoded program binary)
  std::string init_path = get_init_path(g2_path) + "/init.txt";
  arts.init_bin = readHexEncodedFile(init_path);  // Helper to decode hex

  arts.program_size = arts.init_bin.size();
  auto& allocator = SpyreAllocator::instance();
  arts.device_alloc = std::move(allocator.allocate(arts.program_size));
  auto* ctx = static_cast<SharedOwnerCtx*>(arts.device_alloc.get_context());
  stream.copyProgramAsync(arts.init_bin.data(), &ctx->composite_addr);
  stream.synchronize();

  arts.sdsc_json_path = (g2_dir / "sdsc_0.json").string();
  TORCH_CHECK(std::filesystem::exists(arts.sdsc_json_path),
              "SuperDSC not found: ", arts.sdsc_json_path);

  // Cache and return
  std::lock_guard<std::mutex> lock(g_artifact_cache_mtx);
  auto [it, inserted] = g_artifact_cache.emplace(g2_path, std::move(arts));
  if (inserted) {
    DEBUGINFO(it->second);
  }
  return it->second;
}

}  // namespace spyre
